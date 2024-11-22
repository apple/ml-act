# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import torch

from act.hooks.intervention_hook import InterventionHook


class IdentityHook(InterventionHook):
    """
    A "do nothing" intervention.
    """

    def __init__(
        self,
        module_name: str,
        device: str = None,
        intervention_position: str = "original",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(
            module_name=module_name,
            device=device,
            intervention_position=intervention_position,
            dtype=dtype,
        )

    def __str__(self):
        txt = f"Identity(" f"module_name={self.module_name}" f")"
        return txt

    def fit(self, *args, **kwargs):
        pass

    def forward(self, module, input_, output):
        return self(module, input, output)

    def __call__(self, module, input, output):
        return output
