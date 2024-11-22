# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import typing as t
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torch
import transformers

from act.utils import utils

from .collators import DictCollatorWithPadding


class OneSecDataset(torch.utils.data.Dataset):
    LABEL_MAP = OrderedDict([("negative", 0), ("positive", 1)])
    LABEL_NAMES = ["negative", "positive"]

    def __init__(
        self,
        path: Path,
        split: str,
        subsets: t.List[str],
        tokenizer: transformers.PreTrainedTokenizer,
    ) -> torch.utils.data.Dataset:
        assert split == "train", "Only train split is supported for now"
        self.path = path
        self.split = split
        self.tokenizer = tokenizer
        all_concepts = pd.read_csv(self.path / "concept_list.csv")
        all_concepts = list(sorted(all_concepts["concept"].values))
        self.concepts = set([s.replace("non-", "") for s in subsets])
        assert self.concepts.issubset(set(all_concepts))
        self.target_subsets = set(subsets)
        self.data = []
        self.subsets = []
        self.ids = []
        for concept in self.concepts:
            self.data_path = self.path / "sense" / f"{concept}.json"
            data = utils.load_json(self.data_path)
            if f"non-{concept}" in self.target_subsets:
                self.data += data["sentences"]["negative"]
                self.subsets.extend(
                    [f"non-{concept}"] * len(data["sentences"]["negative"])
                )
                self.ids.extend(list(range(len(data["sentences"]["negative"]))))
            if concept in self.target_subsets:
                self.data += data["sentences"]["positive"]
                self.subsets.extend([concept] * len(data["sentences"]["positive"]))
                self.ids.extend(list(range(len(data["sentences"]["positive"]))))

    def __getitem__(self, item) -> dict:
        datum = {
            "text": self.data[item],
            "subset": self.subsets[item],
            "id": f"{self.ids[item]:04d}",
        }
        tokens = self.tokenizer(datum["text"], truncation=True, padding=False)
        datum.update(tokens)
        return datum

    def __len__(self) -> int:
        return len(self.data)


def get_onesec_dataset(
    path: Path,
    split: str,
    subsets: t.List[str],
    tokenizer: transformers.PreTrainedTokenizer,
    **kwargs,
) -> OneSecDataset:
    return OneSecDataset(
        Path(path) / "OneSecConcepts-100_1.5.0",
        split="train",
        subsets=subsets,
        tokenizer=tokenizer,
    ), DictCollatorWithPadding(tokenizer=tokenizer)
