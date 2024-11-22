# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from pathlib import Path

import numpy as np
import pytest

from act.datasets.responses_io import ResponsesLoader


@pytest.fixture(scope="module")
def responses_loader():
    loader = ResponsesLoader(
        Path("./tests/data/toxicity-responses/tiny-gpt2/jigsaw"),
        from_folders=[
            Path("*/transformer*/mean"),
        ],
    )
    return loader


def test_get_attribute_values(responses_loader):
    attribute_name = "module_names"
    values = responses_loader.get_attribute_values(attribute_name)

    # Check if the returned set is not empty
    assert len(values) > 0, f"No values found for attribute {attribute_name}"


def test_load_data_subset(responses_loader):
    filter = {"pooling_op": ["mean"]}
    data = responses_loader.load_data_subset(filter)

    # Check if the returned dictionary is not empty
    assert len(data) > 0, "No data loaded"

    filter = {"module_names": ["transformer.h.0.mlp.c_proj:0"]}
    data = responses_loader.load_data_subset(filter)
    assert set(data["subset"]) == set(["toxic", "non-toxic"])
    assert set(data["module_names"]) == set(["transformer.h.0.mlp.c_proj:0"])

    filter = {"module_names": ["transformer.h.0.mlp.c_proj:0"], "subset": ["non-toxic"]}
    data = responses_loader.load_data_subset(filter)
    assert set(data["subset"]) == set(["non-toxic"])
    assert set(data["module_names"]) == set(["transformer.h.0.mlp.c_proj:0"])

    fail = False
    try:
        responses_loader.load_data_subset({"APPLE": ["PIE"]})
    except:
        fail = True
    assert fail, "load_data_subset should fail with unknown keys."


def test_responses_loader(responses_loader):
    data_subset = {
        "responses": np.arange(10).reshape((10, 1)),
        "subset": np.asarray(["A"] * 5 + ["B"] * 4 + ["C"]),
    }
    labeled_data = ResponsesLoader.label_src_dst_subsets(
        data_subset,
        src_subset=["A", "B"],
        dst_subset=["B"],
        key="subset",
        balanced=False,
        seed=0,
    )
    labels = labeled_data["label"]
    src_data = {k: v[labels == 1] for k, v in labeled_data.items()}
    dst_data = {k: v[labels == 0] for k, v in labeled_data.items()}
    assert "D" not in set(src_data["subset"]) and "D" not in set(dst_data["subset"])
    assert len([s for s in src_data["subset"] if s == "A"]) == 5
    assert len([s for s in dst_data["subset"] if s == "A"]) == 0
    assert len([s for s in src_data["subset"] if s == "B"]) == 2
    assert len([s for s in dst_data["subset"] if s == "B"]) == 2
    labeled_data = ResponsesLoader.label_src_dst_subsets(
        data_subset,
        src_subset=["A", "B"],
        dst_subset=["B"],
        key="subset",
        balanced=True,
        seed=0,
    )
    labels = labeled_data["label"]
    src_data = {k: v[labels == 1] for k, v in labeled_data.items()}
    dst_data = {k: v[labels == 0] for k, v in labeled_data.items()}
    assert len([s for s in src_data["subset"] if s == "A"]) == 1
    assert len([s for s in dst_data["subset"] if s == "A"]) == 0
    assert len([s for s in src_data["subset"] if s == "B"]) == 1
    assert len([s for s in dst_data["subset"] if s == "B"]) == 2
