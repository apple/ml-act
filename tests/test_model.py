# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import torch
from diffusers import StableDiffusionPipeline

from act.models import get_model
from act.models.model_with_hooks import get_model_with_hooks, is_module_name_in_regex


# Define a dummy hook
class DummyHook:
    def __init__(self, module_name=""):
        self.module_name = module_name

    def update(self):
        ...

    def __call__(self, module, input, output):
        self.outputs = {"": "dummy"}


def test_init():
    model = torch.nn.Linear(5, 2)
    mwh = get_model_with_hooks(model, model_task="text-generation")
    assert mwh.get_module() == model
    assert len(mwh._forward_hook_handles) == 0


def test_register_hooks():
    model = torch.nn.Linear(5, 2)
    mwh = get_model_with_hooks(model, model_task="text-generation")

    hook = DummyHook()
    mwh.register_hooks([hook])

    # Test if the hook is registered correctly and outputs are as expected
    assert len(mwh._forward_hook_handles) == 1
    x = torch.randn((2, 5))
    _ = model(x)
    mwh.update_hooks()
    assert "dummy" in mwh.get_hook_outputs()[""]


def test_remove_hooks():
    model = torch.nn.Linear(5, 2)
    mwh = get_model_with_hooks(model, model_task="text-generation")

    hook = DummyHook()
    mwh.register_hooks([hook])

    # Remove the hook and check if it's empty
    mwh.remove_hooks()
    assert len(mwh._forward_hook_handles) == 0
    assert len(mwh.get_hook_outputs()) == 0


def test_find_module_names():
    model, tokenizer = get_model(
        "sshleifer/tiny-gpt2",
        rand_weights=True,
        cache_dir="/tmp/cache",
        device="cpu",
        dtype="float32",
        model_task="text-generation",
    )
    mwh = get_model_with_hooks(model, model_task="text-generation")

    # Test if the method can find modules correctly
    module_names = mwh.find_module_names(mwh.get_module(), [".*0.mlp.*"])
    print(mwh.get_module())
    assert len(module_names) == 5

    # Also checking with an additional "." to see if the glob-style match works (omits layer ending with "mlp")
    module_names = mwh.find_module_names(mwh.get_module(), [".*0.mlp.+"])
    assert len(module_names) == 4


def test_get_target_module_names():
    model, tokenizer = get_model(
        "sshleifer/tiny-gpt2",
        rand_weights=True,
        cache_dir="/tmp/cache",
        device="cpu",
        dtype="float32",
        model_task="text-generation",
    )
    mwh = get_model_with_hooks(model, model_task="text-generation")

    hook = DummyHook("transformer.h.0.mlp.c_fc")
    mwh.register_hooks([hook])

    # Test if the method can get target modules correctly
    target_module_names = mwh.get_target_module_names()
    assert "transformer.h.0.mlp.c_fc" in target_module_names


def test_stable_diffusion():
    pipe = StableDiffusionPipeline.from_pretrained(
        "hf-internal-testing/tiny-stable-diffusion-pipe"
    )
    mwh = get_model_with_hooks(pipe, model_task="text-to-image-generation")
    assert len(mwh.find_module_names(mwh.module, ["vae.*"])) > 0
    assert len(mwh.find_module_names(mwh.module, ["unet.*"])) > 0
    assert len(mwh.find_module_names(mwh.module, ["text_encoder.*"])) > 0


def test_is_module_name_in_regex():
    # Test case where module name matches one or more regex expressions in the list
    assert len(is_module_name_in_regex("foo.bar", [".*", ""])) > 0
    assert len(is_module_name_in_regex("foo.bar", ["f.*", "b.*.r"])) > 0
    assert (
        len(is_module_name_in_regex("hello.world", ["h.*.d", "he?lo.*", "wo?l?.r"])) > 0
    )

    # Test regex with `:`
    assert is_module_name_in_regex("foo.bar:0", [".*bar.*"]) == ["foo.bar:0"]
    assert is_module_name_in_regex("foo.bar:1", [".*bar.*"]) == ["foo.bar:1"]
    assert is_module_name_in_regex("foo.bar:0", [".*bar"]) == ["foo.bar:0"]
    assert is_module_name_in_regex("foo.bar:1", [".*bar"]) == ["foo.bar:1"]
    assert is_module_name_in_regex("foo.bar:0", [".*bar:0"]) == ["foo.bar:0"]
    assert is_module_name_in_regex("foo.bar:1", [".*bar:0"]) == []
    assert is_module_name_in_regex("foo.bar:1", [".*bar:0", ".*bar:1"]) == ["foo.bar:1"]
    assert is_module_name_in_regex("foo.bar:1", [".*bar:0", ".*bar"]) == ["foo.bar:1"]
    assert is_module_name_in_regex("foo.bar", [".*bar:0"]) == ["foo.bar:0"]
    assert is_module_name_in_regex("foo.bar", [".*bar:1"]) == ["foo.bar:1"]
    assert is_module_name_in_regex("foo.bar", [".*bar"]) == ["foo.bar"]
    assert is_module_name_in_regex("foo.bar", [".*bar.*"]) == ["foo.bar"]

    # Test case where module name does not match any regex expressions in the list
    # assert is_module_name_in_regex("foo.bar", ["f.*o"]) is None
    assert len(is_module_name_in_regex("foo.bar", ["f.*o", "b?.r"])) == 0
    assert (
        len(is_module_name_in_regex("hello.world", ["h.*.x", "he?l?.z", "wo?l?.y"]))
        == 0
    )

    # Test case where the list of regex expressions is empty
    assert len(is_module_name_in_regex("foo.bar", [])) == 0
