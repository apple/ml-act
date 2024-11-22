# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import abc
import typing as t
from abc import ABC, abstractmethod
from pathlib import Path

import torch


class InterventionHook(torch.nn.Module):
    """
    Abstract base class for a hook that intervenes during the forward pass of a PyTorch module.

    This class allows you to specify which part of the output tensor (or all outputs) to modify and at what point in the computation this modification should occur.

    Args:
        module_name (str): The name of the module or layer where the intervention is needed. If a specific output index is required, it can be specified after a colon (e.g., "module_name:output_index").
        intervention_position (str): Specifies when to intervene in the forward pass. 'all' means at every step, 'last' means only on the last element of the output tensor sequence.
        dtype (torch.dtype): The desired data type for the intervention. Default is torch.float32.
    """

    def __init__(
        self,
        module_name: str,
        device: str,
        intervention_position: t.Literal["all", "last"],
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.module_name = module_name
        self.device = device
        if ":" in module_name:
            self.select_output = int(module_name.split(":")[1])
        else:
            self.select_output = None
        self.intervention_position = intervention_position
        self.dtype = dtype

    def register_named_buffers(self, **kwargs) -> None:
        for k, v in kwargs.items():
            self.register_buffer(k, v.to(self.dtype))

    def save_state_dict(self, state_path: Path) -> None:
        torch.save(self.state_dict(), state_path)

    def from_state_path(self, state_path: Path) -> None:
        """
        Loads intervention state from a state path pointing to a torch-saved state_dict.

        :param state_path: The state path to load.
        """
        self.load_state_dict(torch.load(state_path))

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError("Method fit() must be implemented.")

    def _post_load(self) -> None:
        """
        This method should be called after loading the states of the hook.
        So calls must be placed at the end of .fit() and at the end of .load_state_dict().

        Re-implement as needed, but do not forget to call super()._post_load() in the implementation.
        """
        # Check all buffers are duly initialized.
        for buffer_name, buffer in self.named_buffers():
            assert buffer.numel() > 0, f"Buffer {buffer_name} has not been initialized."

    def update(self, *args, **kwargs):
        """
        Updates the state or arguments of this hook with new input data at runtime.

        This method can be overridden by subclasses to provide custom updating logic. By default, it does nothing and returns None.

        Parameters:
            *args : variable-length argument list
                Variable length argument list that will be used as is for the update operation.

            **kwargs : keyworded arguments
                Keyworded arguments that can also be used to update state or arguments of this hook.

        Returns:
            None
        """
        return None

    def __call__(self, module, input_, output):
        """
        PyTorch call method overridden to implement the intervention logic.

        Args:
            module (torch.nn.Module): The module for which the forward pass is being evaluated.
            input_ (tuple): Input tensors to the module.
            output (tuple or torch.Tensor): Output of the module's forward function. If `select_output` is specified, it will be a tuple containing this single element; otherwise, it's expected to be a tuple of outputs.

        Returns:
            The modified output after intervention. If `select_output` is specified, returns a modified version of the corresponding output in the tuple. Otherwise, returns the entire sequence of modified outputs.
        """
        if isinstance(output, tuple) and self.select_output is not None:
            _output = output[self.select_output]
        else:
            _output = output
        original_ndim = _output.ndim

        if original_ndim == 2:
            _output = _output[:, None, :]

        if self.intervention_position == "last":
            if len(_output.shape) == 3:
                __output = _output[:, -1, None, ...]
        else:
            __output = _output

        dtype = __output.dtype
        device = __output.device
        __output = __output.to(dtype=self.dtype, device=self.device)
        __output = self.forward(module, input_, __output)
        __output = __output.to(dtype=dtype, device=device)

        if self.intervention_position == "last":
            if len(_output.shape) == 3:
                _output[:, -1, ...] = __output[:, 0, ...]
        else:
            _output = __output

        if original_ndim == 2:
            _output = _output[:, 0, :]

        if isinstance(output, tuple) and self.select_output is not None:
            output = list(output)
            output[self.select_output] = _output
            output = tuple(output)
        else:
            output = _output

        return output

    @abc.abstractmethod
    def forward(self, module, input_, output):
        """
        Abstract method to be implemented by subclasses. This method defines the logic for how the intervention should modify the output.

        Args:
            module (torch.nn.Module): The module for which the forward pass is being evaluated.
            input_ (tuple): Input tensors to the module.
            output (torch.Tensor): Output tensor of the module's forward function, modified according to the intervention logic.

        Returns:
            A modified version of the output tensor after applying the intervention logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")
