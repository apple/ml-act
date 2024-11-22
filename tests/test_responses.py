# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import logging
import tempfile
import typing as t
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize

from act.scripts.learn_intervention import ResponsesManager

logger = logging.getLogger("TEST(compute responses)")
logger.setLevel(logging.DEBUG)


@pytest.mark.parametrize(
    "max_batches,batch_size",
    [
        (2, 4),
        (1, 4),
        (4, 1),
        (3, 2),
    ],
)
def test_compute_responses(max_batches, batch_size):
    # Assuming that the main function doesn't have any side effects and returns None when successful
    with tempfile.TemporaryDirectory(dir="/tmp/") as tempfolder:
        with initialize(version_base=None, config_path="../act/configs"):
            # config is relative to a module
            cfg = compose(
                config_name="text_generation",
                overrides=[
                    "device=cpu",
                    f"responses.batch_size={batch_size}",
                    f"responses.max_batches={max_batches}",
                    "model.model_path=sshleifer/tiny-gpt2",
                    "model.module_names=['transformer.h.0.mlp.c_proj:0', 'transformer.h.1.mlp.c_proj:0']",
                    "responses.tag=toxicity-responses",
                    "data_dir=tests/data",
                    f"cache_dir={tempfolder}",
                    "compute_responses=true",
                    "wandb.mode=disabled",
                ],
            )
        rm = ResponsesManager(cfg.responses)
        responses_path = rm.compute_responses()
        assert responses_path.exists()

        def match_subdirs(root: Path, expected: t.Set) -> bool:
            subdirs = list(root.glob("[!.]*"))  # skip hidden files
            return expected == {elem.name for elem in subdirs}

        assert match_subdirs(responses_path, {"toxic", "non-toxic"})
        assert match_subdirs(
            responses_path / "toxic",
            {"transformer.h.0.mlp.c_proj:0", "transformer.h.1.mlp.c_proj:0"},
        )
        assert match_subdirs(
            responses_path / "non-toxic",
            {"transformer.h.0.mlp.c_proj:0", "transformer.h.1.mlp.c_proj:0"},
        )

        pooling_op = cfg.responses.intervention_params.pooling_op
        batches = list(
            (
                responses_path / "toxic" / f"transformer.h.1.mlp.c_proj:0" / pooling_op
            ).glob("*.pt")
        )
        assert len(batches) == max_batches * batch_size / 2
        batch_data = torch.load(batches[0])
        assert "id" in batch_data
        assert batch_data["responses"].numel() == 2  # tiny-gpt2 has 2 neurons
