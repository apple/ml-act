# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import logging
import typing as t

import torch

from act.hooks.custom_exceptions import TargetModuleReached
from act.hooks.responses_hook import ResponsesHook


class ReturnOutputsHook(ResponsesHook):
    """
    A PyTorch module that captures outputs of a specified submodule during forward pass.
    If `raise_exception` flag is set to True, it raises TargetModuleReached exception when the target module reached.

    Attributes:
        module_name (str): The name of the module for which you want to capture outputs.
        raise_exception (bool): Flag indicating whether to raise an exception or not. Default is False.
        outputs (dict): Dictionary storing names and corresponding tensors of submodules during forward pass.
    """

    def __init__(self, module_name: str, raise_exception: bool = False):
        super().__init__()
        self.module_name = module_name
        self.raise_exception = raise_exception
        self.outputs = {}

    def __call__(self, module, input, output):
        def _hook(module_name: str, output: t.Any):
            if isinstance(output, torch.Tensor):
                self.outputs[module_name] = output.detach()
            elif isinstance(output, (list, tuple)):
                for idx in range(len(output)):
                    _hook(f"{module_name}:{idx}", output[idx])
            else:
                logging.warn(f"Found {type(output)} in {self.module_name}")

        _hook(self.module_name, output)

        if self.raise_exception:
            raise TargetModuleReached(self.module_name)
