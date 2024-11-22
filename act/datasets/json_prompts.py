# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import json
import typing as t
from fnmatch import fnmatch
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, default_collate


class JsonPromptsDataset(Dataset):
    """
    A PyTorch Dataset that loads data from a JSON file.

    This dataset loads a list of prompts from a JSON file, and allows for filtering the subsets to include based on wildcard patterns.
    Each item in the dataset is a dictionary containing 'id', 'prompt' and 'subset' fields. The 'id' is an automatically generated index,
    'prompt' is the prompt string loaded from the JSON file, and 'subset' is the subset name from which this prompt belongs to.

    Args:
        file_path (str): Path to the JSON file containing the prompts data.
        subsets (Optional[Union[List[str], str]]): A list of wildcard patterns or a single pattern to include only certain subsets of the data.
            If None, all subsets are included. Default is None.

    Attributes:
        data (List[Dict]): List of dictionaries where each dictionary represents an item in the dataset with keys 'id', 'prompt' and 'subset'.
        subsets (List[str]): List of subset names that were loaded from the JSON file.

    Methods:
        get_all_subsets(): Returns a set of all unique subsets.
        __len__(): Returns the number of items in the dataset.
        __getitem__(idx): Retrieves an item by its index, returns it as a dictionary with keys 'id', 'prompt' and 'subset'.
    """

    def __init__(self, file_path, subsets=t.List[str]):
        super().__init__()
        if subsets is None:
            subsets = "*"
        if not isinstance(subsets, (list, tuple, set)):
            subsets = [subsets]
        with open(file_path, "r") as f:
            data = json.load(f)

        self.data = []
        self.subsets = []
        for subset in data:
            if not any(map(lambda pattern: fnmatch(subset, pattern), subsets)):
                continue
            for i, string in enumerate(data[subset]):
                self.data.append(
                    {
                        "id": len(self.data),
                        "original_prompt": string,
                        "prompt": string,
                        "subset": subset,
                    }
                )
                self.subsets.append(subset)

    def get_all_subsets(self):
        return set(self.subsets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_example_text2img_prompts_dataset(
    root: Path,
    *args,
    subsets: t.List = None,
    json_path: Path = None,
    **kwargs,
) -> torch.utils.data.Dataset:
    return (
        JsonPromptsDataset(json_path, subsets=subsets),
        default_collate,
    )
