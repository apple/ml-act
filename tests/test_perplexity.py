# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import logging
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize

# Importing learn_intervention to resolve dtype in hydra
from act.evaluations import evaluate_perplexity

logger = logging.getLogger("TEST(compute responses)")
logger.setLevel(logging.DEBUG)


def input_csv():
    data = {
        "prompt": [
            "My name is Alice and I play",
            "Once upon a time",
            "I like music",
            "Arm tree yellow orthogonal",
        ],
        "sentence": [
            " football at school.",
            " there was a Hobbit.",
            " and dancing.",
            " great pull car.",
        ],
    }
    df = pd.DataFrame(data=data)
    return df


@pytest.mark.parametrize(
    "with_prompt",
    [
        "with_prompt",
        "without_prompt",
    ],
)
def test_evaluate_perplexity(with_prompt):
    # Assuming that the main function doesn't have any side effects and returns None when successful
    with tempfile.TemporaryDirectory(dir="/tmp/") as tempfolder:
        tmpfile = Path(tempfolder) / "ppl.csv"
        df = input_csv()
        print(input_csv)
        df.to_csv(tmpfile)

        column_sentences = (
            ["sentence", "prompt"]
            if with_prompt == "with_prompt"
            else [
                "sentence",
            ]
        )

        with initialize(version_base=None, config_path="../act/configs"):
            # config is relative to a module
            cfg = compose(
                config_name="text_generation",
                overrides=[
                    "fast=true",
                    "device=cpu",
                    f"results_dir={tempfolder}",
                    "model_perplexity.perplexity_model_path=EleutherAI/pythia-70m",
                    f"model_perplexity.data_path={tmpfile}",
                    f"model_perplexity.column_sentences={column_sentences}",
                    "wandb.mode=disabled",
                ],
            )
        evaluate_perplexity.evaluate(cfg.model_perplexity)

        df = pd.read_csv(
            Path(tempfolder) / "evaluate_perplexity" / "model_perplexity.csv"
        )
        assert f"ppl_pythia-70m" in df.columns
        assert df[f"ppl_pythia-70m"].argmax() == 3
