# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import functools
import logging
import time
import typing as t

import numpy as np
import torch
from torch import nn


class LinearProj(nn.Module):
    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(1, dim))
        self.b1 = nn.Parameter(torch.zeros((1, dim)))

    def forward(self, x: torch.Tensor, reverse: bool = False):
        assert x.shape[-1] == self.w1.shape[-1]
        if not reverse:
            return x * self.w1 + self.b1
        else:
            return (x - self.b1) / (self.w1 + 1e-10)

    def optimize(self, x: np.ndarray, y: np.ndarray) -> t.Tuple[np.ndarray, t.Dict]:
        x, y = x.astype(np.float64), y.astype(np.float64)

        m_x = np.mean(x, axis=0, keepdims=True)
        m_y = np.mean(y, axis=0, keepdims=True)

        # Add small noise to prevent divisions by 0
        x += 1e-8 * np.random.randn(*x.shape)

        x_bar = x - m_x
        y_bar = y - m_y
        beta = np.sum((x_bar * y_bar), axis=0, keepdims=True) / np.sum(
            (x_bar**2), axis=0, keepdims=True
        )
        alpha = m_y - beta * m_x
        params = np.concatenate([beta, alpha], 0)
        beta = torch.tensor(beta, dtype=self.w1.dtype, device=self.w1.device)
        alpha = torch.tensor(alpha, dtype=self.w1.dtype, device=self.w1.device)
        self.load_state_dict(
            {
                "w1": beta,
                "b1": alpha,
            }
        )
        return params, {}
