# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import tempfile

import pytest
import torch
from hydra import compose, initialize

from act.scripts import pipeline


@pytest.mark.parametrize(
    "config_name,intervention_name,tasks",
    [
        ("text_generation", "none", ["rtp"]),
        # ("text_generation", "none", ["text-generation"]),
        # ("text_generation", "none", ["text-generation", "model_perplexity"]),
        # pytest.param(
        #     "text_generation",
        #     "linear_ot",
        #     ["text-generation", "zero_shot"],
        #     marks=pytest.mark.slow,
        # ),
        # pytest.param("text_generation", "linear_ot", ["mmlu"], marks=pytest.mark.slow),
        # ("text_generation", "linear_ot", ["rtp"]),
        # ("text_generation", "linear_ot", ["text-generation"]),
        # ("text_generation", "linear_ot", ["text-generation", "model_perplexity"]),
        # pytest.param(
        #     "text_generation",
        #     "linear_ot",
        #     ["text-generation", "zero_shot"],
        #     marks=pytest.mark.slow,
        # ),
        # pytest.param("text_generation", "linear_ot", ["mmlu"], marks=pytest.mark.slow),
    ],
)
def test_pipeline_main(config_name, intervention_name, tasks):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Assuming that the main function doesn't have any side effects and returns None when successful
    with tempfile.TemporaryDirectory(dir="/tmp/") as tempfolder:
        with initialize(version_base=None, config_path="../act/configs"):
            cfg = compose(
                config_name=config_name,
                overrides=[
                    "fast=true",
                    f"device={device}",
                    "model.model_path=sshleifer/tiny-gpt2",
                    "model.module_names=['transformer.h.0.mlp.c_proj:0', 'transformer.h.1.mlp.c_proj:0']",
                    f"evaluation={tasks}",
                    "responses.tag=toxicity-responses",
                    f"intervention_params.name={intervention_name}",
                    "intervention_params.hook_params.quantiles_src=q_all",
                    "intervention_params.incremental=atonce",
                    "data_dir=tests/data",
                    "cache_dir=tests/data",
                    f"interventions.cache_dir={tempfolder}",
                    "compute_responses=false",
                    f"results_dir={tempfolder}",
                    "wandb.mode=disabled",
                ],
            )
            pipeline.main(cfg)


@pytest.mark.slow
def test_pipeline_diffusion():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Assuming that the main function doesn't have any side effects and returns None when successful
    with tempfile.TemporaryDirectory(dir="/tmp/") as tempfolder:
        with initialize(version_base=None, config_path="../act/configs"):
            cfg = compose(
                config_name="text_to_image_generation",
                overrides=[
                    "fast=true",
                    "responses.batch_size=2",
                    "responses.max_batches=1",
                    "interventions.batch_size=2",
                    "interventions.max_batches=null",
                    "text_to_image_generation.batch_size=1",
                    "text_to_image_generation.max_batches=1",
                    "text_to_image_generation.strength=[0.0, 1.0]",
                    f"device={device}",
                    # "model.model_path='hf-internal-testing/tiny-stable-diffusion-pipe'",
                    "model.module_names=['unet.down_blocks.0.resnets.0.norm1']",
                    f"evaluation=['text-to-image-generation', 'clip_score']",
                    "responses.tag=diffusion-responses",
                    f"intervention_params.name=linear_ot",
                    "intervention_params.hook_params.quantiles_src=q_all",
                    "intervention_params.incremental=incr",
                    "data_dir=tests/data",
                    "cache_dir=tests/data",
                    f"responses.cache_dir={tempfolder}",
                    f"interventions.cache_dir={tempfolder}",
                    "compute_responses=true",
                    f"results_dir={tempfolder}",
                    "wandb.mode=disabled",
                    "text_to_image_generation.num_inference_steps=1",
                ],
            )
            pipeline.main(cfg)
