# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import json
import typing as t
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader, Dataset, default_collate

from act.datasets.collators import DictCollatorWithPadding


class JsonSubsetsDataset(Dataset):
    """
    A dataset class for loading and handling text data from a JSON file,
    where the data is organized into subsets. Each subset corresponds to
    a key in the JSON file, and the associated value is a list of sentences
    or text entries.

    Attributes:
        data (list): A list containing all the sentences from the specified subsets.
        subsets (list): A list indicating the subset each sentence belongs to.

    Methods:
        get_all_subsets():
            Returns a set of all unique subsets present in the dataset.
        __len__():
            Returns the total number of text entries in the dataset.
        __getitem__(idx):
            Retrieves a dictionary containing the ID, subset, and text entry
            for the specified index.

    Args:
        json_path (str): Path to the JSON file containing the dataset.
        subsets (list, optional): A list of keys corresponding to the subsets
            to be included in the dataset. If not specified or set to None,
            all subsets will be included. Defaults to "*" (all subsets).
    """

    def __init__(
        self,
        json_path: Path,
        subsets: t.List[str],
        tokenizer: transformers.PreTrainedTokenizer,
        **kwargs,
    ):
        super().__init__()
        if subsets is None:
            subsets = "*"
        if not isinstance(subsets, (list, tuple, set)):
            subsets = [subsets]
        with open(json_path, "r") as f:
            data: t.Dict = json.load(f)

        self.tokenizer = tokenizer
        self.data = []
        self.subsets = []
        self.idx_in_subset = []
        for key, sentences in data.items():
            if subsets == "*" or key in subsets:
                subset = key
            else:
                continue
            self.data += sentences
            self.subsets += [subset] * len(sentences)
            self.idx_in_subset += np.arange(len(sentences)).tolist()

    def get_all_subsets(self):
        return set(self.subsets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = {
            "id": idx,
            "subset": self.subsets[idx],
            "text": self.data[idx],
            "idx_in_subset": self.idx_in_subset[idx],
        }
        tokens = self.tokenizer(datum["text"], truncation=True, padding=False)
        datum.update(tokens)
        return datum


def get_json_subsets_dataset(
    *args,
    # file_path=Path("data/giraffe_eagle_situations.json"),
    subsets=None,
    tokenizer=None,
    **kwargs,
) -> torch.utils.data.Dataset:
    assert tokenizer is not None, "Must pass a tokenizer"
    return (
        JsonSubsetsDataset(subsets=subsets, tokenizer=tokenizer, **kwargs),
        DictCollatorWithPadding(tokenizer),
    )
