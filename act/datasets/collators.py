# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import typing as t
from collections import defaultdict

import torch
import transformers


class DictCollatorWithPadding(transformers.DataCollatorWithPadding):
    """Helper class that pads text sequences of multiple lengths
    to a contiguous batch by applying huggingface's
    DataCollatorWithPadding. For the rest of data types
    it converts them from list of dicts to dict of lists.
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        return_tensors: str = "pt",
        pad_to_multiple_of: int = 2,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            return_tensors=return_tensors,
            pad_to_multiple_of=pad_to_multiple_of,
            **kwargs,
        )
        self.tensor_set = set(["input_ids", "attention_mask"])

    def __call__(self, batch: t.List) -> t.Tuple[t.Dict[str, torch.Tensor], t.Dict]:
        """Function to be applied on list of samples from dataloader to form a batch

        Args:
            batch (t.List): list of samples

        Returns:
            t.Tuple[t.Dict[str, torch.Tensor], t.Dict]: Tuple with tokens and additional metadata like labels
        """
        ret = defaultdict(list)
        tensors = []
        meta_set = set(batch[0].keys()) - self.tensor_set
        for sample in batch:
            tensors.append({k: sample[k] for k in self.tensor_set})
            for k in meta_set:
                ret[k].append(sample[k])
        tensors = transformers.DataCollatorWithPadding.__call__(
            self, tensors
        )  # huggingface does the padding
        ret.update(tensors)
        return ret
