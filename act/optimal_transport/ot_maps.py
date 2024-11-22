# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import typing as t

import torch


def solve_ot_1d(
    p: torch.Tensor, q: torch.Tensor
) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """
    OT 1D for same number of points amounts to sorting.
    """
    assert len(p) == len(q), (
        f"Very simple 1D OT matching for now. "
        f"Please use the same number of samples for p, q."
    )
    p_sort, _ = torch.sort(p, 0)
    q_sort, _ = torch.sort(q, 0)
    return p_sort, q_sort
