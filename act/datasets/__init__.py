# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import typing as t
from pathlib import Path

import torch
import transformers

from .coco_captions import (
    get_coco_captions_dataset,
    get_coco_concepts_dataset,
    get_coco_styles_dataset,
)
from .jigsaw_dataset import get_jigsaw_dataset
from .json_prompts import get_example_text2img_prompts_dataset
from .json_subsets_dataset import get_json_subsets_dataset
from .onesec_dataset import get_onesec_dataset
from .samplers import StratifiedSampler

DATASET_LOADERS_REGISTRY = {
    "jigsaw": get_jigsaw_dataset,
    "OneSecConcepts-100_1.5.0": get_onesec_dataset,
    "example-text2img-prompts": get_example_text2img_prompts_dataset,
    "diffusion-prompts": get_example_text2img_prompts_dataset,
    "coco-captions-2017": get_coco_captions_dataset,
    "coco-captions-styles": get_coco_styles_dataset,
    "coco-captions-concepts": get_coco_concepts_dataset,
    "json-subsets": get_json_subsets_dataset,
}


def get_dataset(
    name: str,
    datasets_folder: Path,
    split: str,
    subsets: t.Set[str],
    tokenizer: t.Optional[transformers.PreTrainedTokenizer] = None,
    **kwargs,
) -> t.Tuple[torch.utils.data.Dataset, t.Callable]:
    """Loads and returns a dataset split given its name. It also returns a collator function for the dataloader

    Args:
        name (str): dataset name
        datasets_folder (Path): path where dataset is located
        split (bool): train, val, test
        tokenizer (t.Optional[transformers.PreTrainedTokenizer], optional): a huggingface tokenizer in case it is a text dataset. Defaults to None.

    Returns:
        t.Tuple[torch.utils.data.Dataset, t.Callable]: pytorch Dataset instance and collator function
    """
    assert (
        name in DATASET_LOADERS_REGISTRY
    ), f"{name} not in DATASET_LOADERS_REGISTRY ({DATASET_LOADERS_REGISTRY.keys()})"
    data_loader = DATASET_LOADERS_REGISTRY[name]
    return data_loader(
        datasets_folder,
        split=split,
        subsets=subsets,
        tokenizer=tokenizer,
        **kwargs,
    )


def get_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    collate_fn: int,
    drop_last: bool,
    shuffle: bool,
    balanced: bool = False,
    seed: int = 0,
    **kwargs: dict,
) -> torch.utils.data.DataLoader:
    if balanced:
        sampler = StratifiedSampler(dataset.subsets, seed=seed)
        shuffle = False
    else:
        sampler = None
        shuffle = shuffle
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        shuffle=shuffle,
        sampler=sampler,
        **kwargs,
    )


if __name__ == "__main__":
    pass
