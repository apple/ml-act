# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import typing as t

import torch


class StratifiedSampler(torch.utils.data.Sampler):
    """Stratified Sampling

    Provides equal representation of target classes in each batch
    """

    def __init__(self, labels: t.List[str], seed: int):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        label_map = {l: i for i, l in enumerate(set(labels))}
        self.class_vector = torch.tensor([label_map[l] for l in labels])
        self.idx = torch.arange(len(self.class_vector))
        self.idx_per_label = {}
        uniques, counts = torch.unique(self.class_vector, return_counts=True)
        for label in uniques:
            self.idx_per_label[label] = self.idx[self.class_vector == label]
        self.min_count = torch.min(counts)
        self.seed = seed
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        """Puts RNG to the correct epoch

        Args:
            epoch (int): epoch to put the generator in
        """
        self.epoch = epoch
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)
        for _ in range(epoch):
            iter(self)

    def __iter__(self):
        indices = []
        tail = []
        for label, idx in self.idx_per_label.items():
            indices.append(
                idx[torch.randperm(len(idx), generator=self.rng)[: self.min_count]]
            )
        indices = torch.stack(indices, 1).ravel()
        return iter(indices)

    def __len__(self):
        return self.min_count * len(self.idx_per_label)
