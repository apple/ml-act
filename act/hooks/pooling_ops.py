# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import typing as t
from functools import partial

import torch


def nanmax(tensor, dim=None, keepdim=False):
    """
    This function takes a tensor and along with a dimension 'dim', and a boolean flag 'keepdim'.
    It returns another tensor where for each 'dim' the values are replaced with maximum value in that dimension.
    If 'tensor' has any NaNs, it will return infinity instead of NaN.

    Parameters:
        tensor (Tensor): Input tensor from which to compute max
        dim (int or None): Dimension along which the maximum is computed
        keepdim (bool): Determines whether the output tensors have 'dim' retained or not

    Returns:
        Tensor: The resultant tensor after applying nanmax
    """
    min_value = torch.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value).amax(dim=dim, keepdim=keepdim)
    return output


def nanmin(tensor, dim=None, keepdim=False):
    """
    Compute the minimum of tensor elements along a specified axis.

    Parameters:
        tensor (Tensor): Input Tensor.
        dim (int or tuple of ints, optional): Dimensions to reduce along. Default is None, which will return the minimum over all elements.
        keepdim (bool, optional): If True, retains reduced dimensions with length 1. Default is False.

    Returns:
        Tensor: The minimum value along the specified dimension(s).

    Note:
        This function ignores NaN values and finds the minimum among non-NaN elements.
    """
    max_value = torch.finfo(tensor.dtype).max
    output = tensor.nan_to_num(max_value).amin(dim=dim, keepdim=keepdim)
    return output


class TorchPoolingOP(torch.nn.Module):
    """
    A module that applies a pooling operation on input tensor along given dimension.

    Parameters:
        op_name (str): Name of the pooling function to be used, from BASE_POOLING_FUNCTIONS.
        dim (int): Dimension along which the operation is performed.

    Attributes:
        name (str): The name of the pooling function being applied.
        dim (int): The dimension along which the operation is performed.
        op (function): The actual pooling function to be used.

    """

    TORCH_POOLING_FUNCTIONS = {
        "min": nanmin,
        "max": nanmax,
        "mean": torch.nanmean,
        "median": torch.nanmedian,
        "last": partial(torch.select, index=-1),  # equivalent to array[-1]
        "all": lambda x, *args, **kwargs: x,
    }

    def __init__(self, op_name: str, dim: t.Union[int, str]):
        super().__init__()
        self.name = op_name
        self.dim = dim
        self.op = self.TORCH_POOLING_FUNCTIONS[self.name]

    def forward(
        self,
        tensor: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Applies the pooling operation on input tensor along given dimension.

        Parameters:
            tensor (torch.Tensor): The input tensor to which the operation is applied.

        Returns:
            torch.Tensor: Result of applying the pooling function on the input tensor,
                          along specified dimension.

        """
        tensor_to_op = tensor
        if self.name == "all":
            return tensor
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            # assert (
            #     attention_mask[:, 0].all() == True
            # ), "Attention mask contains 0s at the end while assuming right padding."
            # nans will be ignored (used w/ attention mask)
            tensor_to_op[~attention_mask] = torch.nan
        if self.dim == "auto":
            if len(tensor.shape) == 2:  # Single token ops are directly returned
                return tensor
            elif len(tensor.shape) == 3:
                dim = 1
            elif len(tensor.shape) == 4:
                dim = (2, 3)
            else:
                raise RuntimeError(
                    f"Tensor shape {tensor.shape} not supported in pooling op auto mode."
                )
        else:
            dim = 1
        ret = self.op(tensor_to_op, dim=dim)
        assert not torch.any(ret != ret), "NaNs or inf in output of pooling op."
        return ret


POOLING_FUNCTIONS_REGISTRY = {
    "min": TorchPoolingOP,
    "max": TorchPoolingOP,
    "mean": TorchPoolingOP,
    "median": TorchPoolingOP,
    "std": TorchPoolingOP,
    "last": TorchPoolingOP,  # equivalent to array[-1]
    "all": TorchPoolingOP,
}


def get_pooling_op(pooling_op_name: str, dim: t.Union[int, str]):
    """
    Returns a pooling operation based on the provided name and dimension.

    Parameters:
        pooling_op_name (str): The name of the pooling operation to be returned.
        dim (int): The dimension along which the pooling will be performed.

    Returns:
        A callable object representing the desired pooling function.

    Raises:
        KeyError: If an invalid `pooling_op_name` is provided.

    Note:
        This function relies on a global registry of available pooling functions (POOLING_FUNCTIONS_REGISTRY).
        The specifics of this dictionary are not included in the docstring for brevity, but should be consulted if you want to use this function.
    """

    return POOLING_FUNCTIONS_REGISTRY[pooling_op_name](op_name=pooling_op_name, dim=dim)
