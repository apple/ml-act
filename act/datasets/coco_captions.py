# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple

import torch
from torch.utils.data import default_collate


class CocoCaptionsDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for loading COCO captions dataset.

    Args:
        captions_path (Path): The path to the directory containing the COCO captions JSON file.
        split (Literal["train", "val"]): The data split to load, either 'train' or 'val'. Defaults to 'val'.

    Attributes:
        captions_path (Path): The path to the COCO captions JSON file.
        captions (Dict[str, Any]): The loaded captions from the JSON file.

    Methods:
        __getitem__(self, index: int) -> Dict[str, Any]: Returns a dictionary containing the prompt and id for a given index.
        __len__(self) -> int: Returns the number of items in the dataset.
    """

    def __init__(
        self, captions_path: Path, split: Literal["train", "val"] = "val"
    ) -> None:
        """
        Initializes the CocoCaptionsDataset with the specified captions path and data split.

        Args:
            captions_path (Path): The path to the directory containing the COCO captions JSON file.
            split (Literal["train", "val"]): The data split to load, either 'train' or 'val'. Defaults to 'val'.

        Returns:
            None: No return value.
        """
        self.captions_path = Path(captions_path) / f"captions_{split}2017.json"
        with self.captions_path.open("r") as f:
            self.captions = json.load(f)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        datum = {
            "prompt": self.captions["annotations"][index]["caption"],
            "id": self.captions["annotations"][index]["id"],
            "subset": "coco",
        }
        return datum

    def __len__(self) -> int:
        return len(self.captions["annotations"])


class CocoStylesDataset(CocoCaptionsDataset):
    """
    A subclass of `CocoCaptionsDataset` for adding style prompts to image captions.

    This dataset extends the functionality of `CocoCaptionsDataset` by incorporating style information into the caption prompts.
    It allows you to specify subsets of images (e.g., "realistic", "painterly") and adds corresponding style tags to the prompts.

    Args:
        captions_path (Path): The path to the directory containing the COCO captions JSON file.
        split (Literal["train", "val"]): The data split to load, either 'train' or 'val'. Defaults to 'val'.
        subsets (List[str]): A list of subset names for which style prompts should be added (e.g., ["realistic", "painterly"]).
        num_tags (int): The total number of tags to include in the prompt. Defaults to 10.

    Attributes:
        captions_path (Path): The path to the COCO captions JSON file.
        captions (Dict[str, Any]): The loaded captions from the JSON file.
        subset_prompts (Dict[str, str]): A dictionary mapping subset names to style prompt strings (e.g., {"realistic": "photographic, detailed, realistic"}).
        subsets (List[str]): List of subsets to which style prompts will be added.
        num_tags (int): Total number of tags in the prompt.
        num_important_tags (int): Number of important tags (half of `num_tags`).
        num_other_tags (int): Number of other tags (remaining after allocating for important tags).

    Methods:
        __getitem__(self, index: int) -> Dict[str, Any]: Returns a dictionary containing the image caption and ID with added style prompts for a given index.
                                                   If the image belongs to a specified subset, style tags are appended to the original prompt.

    """

    def __init__(
        self,
        captions_path: Path,
        split: Literal["train", "val"] = "val",
        subsets: List[str] = [],
        num_tags: int = 10,
    ):
        """
        Initializes the CocoStylesDataset with the specified captions path, data split, subsets, and number of tags.

        Args:
            captions_path (Path): The path to the directory containing the COCO captions JSON file.
            split (Literal["train", "val"]): The data split to load, either 'train' or 'val'. Defaults to 'val'.
            subsets (List[str]): A list of subset names for which style prompts should be added. Defaults to an empty list.
            num_tags (int): The total number of tags to include in the prompt. Defaults to 10.

        Returns:
            An instance of CocoStylesDataset
        """
        super().__init__(captions_path, split)
        assert isinstance(subsets, (list, tuple, set))
        with Path("data/style_prompts.json").open("r") as infile:
            self.subset_prompts = json.load(infile)
        self.subset_prompts = {
            k: v for k, v in self.subset_prompts.items() if k in subsets
        }
        self.subsets = subsets * math.ceil(len(self) / len(subsets))
        self.subsets = self.subsets[: len(self)]
        self.num_tags = num_tags
        self.num_important_tags = self.num_tags // 2
        self.num_other_tags = self.num_tags - self.num_important_tags

    def __getitem__(self, index: int) -> Dict[str, Any]:
        datum = super().__getitem__(index)
        subset = self.subsets[index]
        datum["subset"] = subset
        datum["original_prompt"] = datum["prompt"][:]
        if subset == "none":
            return datum
        else:
            tags = self.subset_prompts[subset].split(", ")
            important_tags = tags[: self.num_important_tags]
            other_tags = tags[self.num_important_tags :]
            choice_tags = torch.randperm(len(other_tags))[
                : min(self.num_other_tags, len(other_tags))
            ]
            post_prompt = important_tags + [other_tags[i] for i in choice_tags]
            post_prompt = ", ".join(post_prompt)
            datum["prompt"] = " ".join([datum["prompt"], post_prompt])
            return datum


class CocoConceptDataset(CocoCaptionsDataset):
    """
    A subclass of `CocoCaptionsDataset` that enriches image captions with concept prompts, guiding models towards generating descriptions with specific themes or styles.

    Args:
        captions_path (Path): The path to the directory containing the COCO captions JSON file.
        split (Literal["train", "val"]): The data split to load ('train' or 'val'). Defaults to 'val'.
        subsets (List[str]): A list of subset names for which concept prompts should be added (e.g., ["animals", "nature"]).

    Attributes:
        captions_path (Path): Path to the COCO captions JSON file.
        captions (Dict[str, Any]): Loaded captions from the JSON file.
        subset_prompts (Dict[str, List[str]]): Dictionary mapping subset names to lists of concept prompts loaded from "./data/diffusion_concept_prompts.json".
        subsets (List[str]): List of subsets to which concept prompts will be added.

    Methods:
        __getitem__(self, index: int) -> Dict[str, Any]: Returns a dictionary containing the image ID, caption, and original prompt, with an added concept prompt for the specified subset at the given index.

    Example:

    To add concept prompts related to "animals" and "nature" subsets:
        dataset = CocoConceptDataset(captions_path="./data/coco", split="train", subsets=["animals", "nature"])

    This dataset will append relevant concept prompts from "./data/diffusion_concept_prompts.json" to captions belonging to the "animals" and "nature" subsets during retrieval using __getitem__().
    """

    def __init__(
        self,
        captions_path: Path,
        split: Literal["train", "val"] = "val",
        subsets: List[str] = [],
    ) -> None:
        super().__init__(captions_path, split)
        assert isinstance(subsets, (list, tuple, set))
        with Path("./data/diffusion_concept_prompts.json").open("r") as infile:
            self.subset_prompts = json.load(infile)
        self.subset_prompts = {
            k: v for k, v in self.subset_prompts.items() if k in subsets
        }
        self.subsets = subsets * math.ceil(len(self) / len(subsets))
        self.subsets = self.subsets[: len(self)]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        datum = super().__getitem__(index)
        subset = self.subsets[index]
        datum["subset"] = subset
        datum["original_prompt"] = datum["prompt"][:]
        if subset == "none":
            return datum
        else:
            datum["prompt"] += (
                " "
                + self.subset_prompts[subset][index % len(self.subset_prompts[subset])]
            )
            return datum


def get_coco_captions_dataset(
    data_dir: Path, split: Literal["train", "val"], **kwargs
) -> Tuple[CocoCaptionsDataset, Callable]:
    """
    Creates a CocoCaptionsDataset instance for the specified data directory and split.

    Args:
        data_dir (Path): The root directory of the COCO dataset.
        split (Literal["train", "val"]): The data split to load, either 'train' or 'val'.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[CocoCaptionsDataset, Callable]: A tuple containing the CocoCaptionsDataset instance and the default collate function.
    """
    return (
        CocoCaptionsDataset(Path(data_dir) / "coco_captions_2017", split),
        default_collate,
    )


def get_coco_styles_dataset(
    data_dir: Path,
    split: Literal["train", "val"],
    subsets: List[str],
    num_tags: int = 10,
    **kwargs,
) -> Tuple[CocoStylesDataset, Callable]:
    """
    Creates a CocoStylesDataset instance for the specified data directory, split, and subsets.

    Args:
        data_dir (Path): The root directory of the COCO dataset.
        split (Literal["train", "val"]): The data split to load, either 'train' or 'val'.
        subsets (List[str]): A list of subset names for which style prompts should be added.
        num_tags (int): The total number of tags to include in the prompt. Defaults to 10.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[CocoStylesDataset, Callable]: A tuple containing the CocoStylesDataset instance and the default collate function.
    """
    return (
        CocoStylesDataset(
            Path(data_dir) / "coco_captions_2017",
            split=split,
            subsets=subsets,
            num_tags=num_tags,
        ),
        default_collate,
    )


def get_coco_concepts_dataset(
    data_dir: Path,
    split: Literal["train", "val"],
    subsets: List[str],
    **kwargs,
) -> Tuple[CocoConceptDataset, Callable]:
    """
    Creates a CocoConceptDataset instance for the specified data directory, split, and subsets.

    Args:
        data_dir (Path): The root directory of the COCO dataset.
        split (Literal["train", "val"]): The data split to load, either 'train' or 'val'.
        subsets (List[str]): A list of subset names for which concept prompts should be added.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[CocoConceptDataset, Callable]: A tuple containing the CocoConceptDataset instance and the default collate function.
    """
    return (
        CocoConceptDataset(
            Path(data_dir) / "coco_captions_2017",
            split=split,
            subsets=subsets,
        ),
        default_collate,
    )
