# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from .aura_hook import AURAHook
from .identity import IdentityHook
from .postprocess_and_save_hook import PostprocessAndSaveHook
from .responses_hook import ResponsesHook
from .return_outputs_hook import ReturnOutputsHook
from .transport import GaussianOTHook, LinearOTHook, OnlyMeanHook

HOOK_REGISTRY = {
    "postprocess_and_save": PostprocessAndSaveHook,
    "return_outputs": ReturnOutputsHook,
    "aura": AURAHook,
    "mean_ot": OnlyMeanHook,
    "gaussian_ot": GaussianOTHook,
    "linear_ot": LinearOTHook,
    "identity": IdentityHook,
    "none": IdentityHook,
}


def get_hook(name: str, *args, **kwargs) -> ResponsesHook:
    hook_cls = HOOK_REGISTRY[name]
    hook = hook_cls(*args, **kwargs)
    return hook
