# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import json
import typing as t
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, default_collate


class DummyText2ImagePrompts(Dataset):
    def __init__(self, file_path):
        super().__init__()

        with open(file_path, "r") as f:
            data = json.load(f)

        self.data = []
        self.subsets = []
        for subset in data:
            for i, string in enumerate(data[subset]):
                self.data.append(
                    {"id": len(self.data), "prompt": string, "subset": subset}
                )
                self.subsets.append(subset)

    def get_all_subsets(self):
        return set(self.subsets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_dummy_text2image_prompts(
    *args,
    **kwargs,
) -> torch.utils.data.Dataset:
    return (
        DummyText2ImagePrompts(Path("data/dummy_text2img_prompts.json")),
        default_collate,
    )
