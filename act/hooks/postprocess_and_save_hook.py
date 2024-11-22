# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import logging
import os
import threading
import typing as t
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch

from act.hooks.custom_exceptions import TargetModuleReached
from act.hooks.responses_hook import ResponsesHook

from .pooling_ops import get_pooling_op


class PostprocessAndSaveHook(ResponsesHook):
    """
    A PyTorch module to post-process and save outputs of other modules.

    This hook takes as input the output of another module in a model, applies some
    operations on it (like pooling), optionally saves it somewhere, and if required,
    returns the processed output back for further use. Since it is a pytorch
    module, it supports loading and saving state dicts

    Parameters:
        module_name : str
            Name of the parent module whose outputs are being hooked to.

        pooling_op_names : list[str]
            List of pooling operation names to be applied on the input data, e.g., ['max', 'mean'].

        output_path : Path
            The location where the processed outputs (if any) should be saved or not (None).

        save_fields : list[str]
            Fields of interest in the inputs and/or outputs to be stored for later use, e.g., ['features', 'labels'].

        return_outputs : bool, optional
            If True, returns the processed output back from this hooked module for further use. Default is False.

        threaded : bool, optional
            If True, the hook will be launched in a different thread.

        raise_exception : bool, optional
            If True, raises an exception if there are issues saving outputs to disk. Default is False.

    """

    def __init__(
        self,
        module_name: str,
        pooling_op_names: t.List[str],
        output_path: t.Optional[Path],
        save_fields: t.List[str],
        return_outputs: bool = False,
        raise_exception: bool = False,
        threaded: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.module_name = module_name
        # Storing as modules to allow stateful ops
        # these are applied independently, not one after the other
        self.pooling_ops = torch.nn.ModuleList(
            [get_pooling_op(name, dim="auto") for name in pooling_op_names]
        )
        self.output_path = output_path
        self.save_fields = save_fields
        self.return_outputs = return_outputs
        self.raise_exception = raise_exception
        self.batch_idx = None
        self.attention_mask = None
        self.threaded = threaded
        self.thread_handle = None

    def __str__(self):
        txt = (
            f"PostprocessAndSaveHook(module_name={self.module_name}, pooling_ops={self.pooling_ops}, "
            f"output_path={self.output_path}, raise_exception={self.raise_exception})\n"
        )
        txt += super().__str__()
        return txt

    def update(
        self,
        batch_idx: int,
        batch: dict,
    ) -> None:
        """
        Updates the state of this hook with new input data.

        This includes setting the current batch index and updating the inputs,
        which are then processed by the pooling operations defined in __init__().

        Parameters:
            batch_idx : int
                The index of the current mini-batch in a full epoch or dataset.

            batch : dict
                A dictionary containing the input data for this hooked module, e.g., features and labels.

        Returns:
            None
        """
        assert "id" in batch
        self.batch_idx = batch_idx
        self.batch = batch
        self.outputs = defaultdict(list)

    def save(self, module_name: str, output: t.List[dict]) -> None:
        """
        Applies pooling operations on input data and saves them to disk or optionally returns them.

        The processed outputs are saved in torch pickle format at the specified location, with each file named after a specific
        combination of module_name and pooling operation name. These files can later be loaded back into memory for further use.

        Parameters:
            module_name : str
                Name of the parent module whose outputs are being hooked to.

            output : list[dict]
                The processed output from the parent module after applying pooling operations.

        Returns:
            None
        """
        attention_mask = self.batch.get("attention_mask", None)

        if "unet" in module_name:
            output = output.to(torch.float32)  # got some infs
            if len(self.batch["id"]) < output.shape[0]:
                output = output.chunk(2)[1]

        for pooling_op in self.pooling_ops:
            pooled_output = pooling_op(
                output.detach().clone(), attention_mask=attention_mask
            )

            for sample_index in range(len(pooled_output)):
                datum = {}
                sample_id = self.batch["id"][sample_index]
                sample_outputs = pooled_output[sample_index].cpu()
                for field in self.save_fields:
                    datum[field] = self.batch[field][sample_index]
                datum.update(
                    {
                        "responses": sample_outputs.cpu(),
                    }
                )
                subset = self.batch["subset"][sample_index]
                if self.output_path is not None:
                    output_path = (
                        self.output_path
                        / subset
                        / module_name
                        / pooling_op.name
                        / f"{sample_id}.pt"
                    )
                    os.makedirs(output_path.parent, exist_ok=True)
                    torch.save(datum, output_path)
                if self.return_outputs:
                    self.outputs[module_name].append(datum)

    def __call__(self, module, input, output) -> None:
        """
        Called when this hooked module's output changes.

        This method applies pooling operations to the inputs and saves them either on disk or optionally returns them.
        If `raise_exception` is set to True in the initialization, an exception will be raised if the parent module's name
        matches that specified during initialization.

        Parameters:
            module : torch.nn.Module
                The module whose output has changed.

            input : tuple or torch.Tensor or dict of them
                Input to this module.

            output : torch.Tensor
                Output from this module.

        Returns:
            None
        """
        assert (
            self.batch_idx is not None
        ), "update() must be called before executing the hook"

        def _hook(module_name: str, output: t.Any):
            if isinstance(output, torch.Tensor):
                if ":" not in module_name:
                    module_name = f"{module_name}:0"
                self.save(module_name, output.detach())
            elif isinstance(output, (list, tuple)):
                if ":" in module_name:
                    name, idx = module_name.split(":")
                    _hook(module_name, output[int(idx)])
                else:
                    for idx in range(len(output)):
                        _hook(f"{module_name}:{idx}", output[idx])

            else:
                logging.warn(f"Found {type(output)} in {self.module_name}")

        if self.threaded:
            self.thread_handle = threading.Thread(
                target=_hook, args=(self.module_name, output)
            )
            self.thread_handle.start()
        else:
            _hook(self.module_name, output)

        if self.raise_exception:
            raise TargetModuleReached(self.module_name)
