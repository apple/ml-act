# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import typing as t

import torch


def compute_quantiles(z: torch.Tensor) -> t.Dict[str, t.List[torch.Tensor]]:
    # TODO: Only keep the quantile of interest to reduce footprint! Using many now for research purposes only.
    qs = {
        "q_0_100": [
            torch.quantile(z, q=0.00, dim=0),
            torch.quantile(z, q=1.0, dim=0),
        ],
        "q_0.5_99.5": [
            torch.quantile(z, q=0.005, dim=0),
            torch.quantile(z, q=0.995, dim=0),
        ],
        "q_1_99": [
            torch.quantile(z, q=0.01, dim=0),
            torch.quantile(z, q=0.99, dim=0),
        ],
        "q_2_98": [
            torch.quantile(z, q=0.02, dim=0),
            torch.quantile(z, q=0.98, dim=0),
        ],
        "q_5_95": [
            torch.quantile(z, q=0.05, dim=0),
            torch.quantile(z, q=0.95, dim=0),
        ],
        "q_10_90": [
            torch.quantile(z, q=0.10, dim=0),
            torch.quantile(z, q=0.90, dim=0),
        ],
        "q_20_80": [
            torch.quantile(z, q=0.20, dim=0),
            torch.quantile(z, q=0.80, dim=0),
        ],
        "q_30_70": [
            torch.quantile(z, q=0.30, dim=0),
            torch.quantile(z, q=0.70, dim=0),
        ],
        "q_40_60": [
            torch.quantile(z, q=0.40, dim=0),
            torch.quantile(z, q=0.60, dim=0),
        ],
    }
    return qs
