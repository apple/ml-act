# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import logging
import multiprocessing
import typing as t

import torch

from act.hooks.intervention_hook import InterventionHook
from act.utils.auroc import compute_auroc

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AURAHook(InterventionHook):
    """Applies AuRoc Adaptation (AurA) (https://arxiv.org/abs/2407.12824) to a given module's output.

    AurA is an intervention technique that modifies the output of a neural network
    module based on the AUROC of its outputs at classifying a concept. This helps mitigating
    the concept by dampening each output proportionally to its AuROC.

    Attributes:
        module_name (str): The name of the module to apply AurA intervention to.
        device (str, optional): The device on which the model and hook reside. Defaults to None.
        intervention_position (str, optional): Specifies where in the module's forward pass to apply
            the intervention. Options are 'all', 'pre', or 'post'. Defaults to 'all'.
        dtype (torch.dtype, optional): The data type for tensors used by the hook. Defaults to torch.float32.
        strength (float, optional): Controls the intensity of AurA intervention. A value of 1.0 applies
            AurA fully, while 0.0 disables it. Defaults to 1.0.

    """

    def __init__(
        self,
        module_name: str,
        device: str = None,
        intervention_position: str = "all",
        dtype: torch.dtype = torch.float32,
        strength: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            module_name=module_name,
            device=device,
            intervention_position=intervention_position,
            dtype=dtype,
        )
        self.strength = float(strength)
        self.register_buffer("auroc", torch.empty(0))

    def __str__(self):
        txt = (
            f"AurA("
            f"module_name={self.module_name}, "
            f"strength={self.strength}"
            f")"
        )
        return txt

    def _post_load(self) -> None:
        super()._post_load()
        # Pre-compute dampening once.
        self.alpha = torch.ones_like(self.auroc, dtype=self.dtype)
        mask = self.auroc > 0.5
        self.alpha[mask] = 1 - 2 * (self.auroc[mask] - 0.5)

    def load_state_dict(
        self,
        state_dict: t.Mapping[str, t.Any],
        strict: bool = True,
        assign: bool = False,
    ):
        self.auroc = state_dict["auroc"].to(self.device).to(self.dtype)
        self._post_load()

    def fit(
        self,
        responses: torch.Tensor,
        labels=torch.Tensor,
        pool: multiprocessing.Pool = None,
        **kwargs,
    ) -> None:
        logger.info(f"Computing AUROC on {responses.shape} responses ...")
        auroc = compute_auroc(
            responses=responses.detach().cpu().numpy(),
            labels=labels.detach().cpu().numpy(),
            chunk_size=10,
            pool=None,
        )
        self.auroc = torch.tensor(auroc, dtype=self.dtype, device=self.device)
        self._post_load()

    def forward(self, module, input, output) -> t.Any:
        if output.ndim == 4:
            alpha = self.alpha.view(1, -1, 1, 1)
        elif output.ndim == 3:
            alpha = self.alpha.view(1, 1, -1)
        else:
            raise NotImplementedError()

        # Apply AurA dampening
        output_aura = output * alpha

        # Adding strength to AurA
        output = (1 - self.strength) * output + self.strength * output_aura
        return output
