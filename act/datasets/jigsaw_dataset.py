# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import typing as t
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torch
import transformers

from act.datasets.collators import DictCollatorWithPadding


class JigsawDataset(torch.utils.data.Dataset):
    """
    Implements a loader for the Jigsaw toxicity dataset.
    To get the files download from the following URL into `path`:
    https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
    """

    SUPER_TOXIC_FLAG = "all"
    SUPER_NONTOXIC_FLAG = "non-toxic"
    JIGSAW_CATEGORIES = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]

    def __init__(
        self,
        path: Path,
        split: str,
        subsets: t.Set[str],
        tokenizer: transformers.PreTrainedTokenizer,
    ) -> torch.utils.data.Dataset:
        self.split = split
        self.path = path
        self.target_subsets = set(subsets)
        # Some magic, if we select "all" it appends all the jigsaw categories.
        self.using_all = False
        if "all" in self.target_subsets:
            self.target_subsets.remove("all")
            self.target_subsets.update(set(self.JIGSAW_CATEGORIES))
            self.using_all = True

        assert self.target_subsets.issubset(set(self.JIGSAW_CATEGORIES + ["non-toxic"]))
        self.tokenizer = tokenizer

        if self.split == "train":
            train_data = pd.read_csv(path / "train.csv", index_col="id")
            self.data = self._preprocess(train_data)
        elif self.split == "test":
            test_data = pd.read_csv(path / "test.csv", index_col="id")
            test_labels = pd.read_csv(path / "test_labels.csv", index_col="id")
            test_dataset = pd.concat(
                [test_data, test_labels], axis=1, ignore_index=False
            )
            # test dataset comes with unannotated data (label=-1)
            test_dataset = test_dataset.loc[
                (test_dataset[test_dataset.columns[1:]] > -1).all(axis=1)
            ]
            self.data = self._preprocess(test_dataset)
        _ = self.data[0]  # small test

        def get_label(elem):
            """
            Returns the (binary) label string of a given Jigsaw datapoint.

            If at least one of the categories in self.target_subsets is satisfied, return the "toxic" class.
            If no category is satisfied and we require non-toxic sentences, return the "non-toxic" class.
            If some category is satisfied NOT in self.target_subsets, returns None (meaning we should skip this datapoint).
            """
            is_other = 0
            for categ in self.JIGSAW_CATEGORIES:
                if elem[categ] == 1 and categ in self.target_subsets:
                    return self.SUPER_TOXIC_FLAG if self.using_all else categ
                if elem[categ] == 1 and categ not in self.target_subsets:
                    is_other = True
            return (
                self.SUPER_NONTOXIC_FLAG
                if (not is_other and self.SUPER_NONTOXIC_FLAG in self.target_subsets)
                else None
            )

        self.subsets = [get_label(d) for d in self.data]
        self.data = [d for s, d in zip(self.subsets, self.data) if s is not None]
        self.subsets = [s for s in self.subsets if s is not None]

    def _preprocess(self, df: pd.DataFrame):
        return df.reset_index().to_dict("records")

    def __getitem__(self, item) -> t.Dict:
        datum: t.Dict = self.data[item]
        tokens = self.tokenizer(datum["comment_text"], truncation=True, padding=False)
        datum.update(tokens)
        datum["subset"] = self.subsets[item]
        return datum

    def __len__(self) -> int:
        return len(self.data)


def get_jigsaw_dataset(
    path: Path,
    split: str,
    subsets: t.Set[str],
    tokenizer: transformers.PreTrainedTokenizer,
    **kwargs
) -> torch.utils.data.Dataset:
    return JigsawDataset(
        Path(path) / "jigsaw", split, subsets=subsets, tokenizer=tokenizer
    ), DictCollatorWithPadding(tokenizer)
