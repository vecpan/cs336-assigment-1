"""Softmax along a caller-chosen dimension."""

import torch
from torch import nn


class Softmax(nn.Module):
    """Turn a tensor into a probability distribution along one dimension.

    Along ``dim`` each output slice sums to one and is proportional to
    ``exp(x)``. The per-slice maximum is subtracted before the exponential so
    that large logits cannot overflow.

    Args:
        dim: Dimension along which the entries are normalized. Negative values
            index from the last dimension, as in most PyTorch reductions.

    Attributes:
        dim: Dimension along which ``forward`` normalizes its input.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the softmax of ``x`` along ``self.dim``.

        Args:
            x: Floating point tensor of any shape.

        Returns:
            Tensor with the same shape and dtype as ``x`` whose slices sum to one
            along ``self.dim``.

        Raises:
            AssertionError: If ``x`` is not floating point, is 0-dimensional, or
                ``self.dim`` is out of range for the shape of ``x``.
        """
        assert x.is_floating_point(), f"softmax expects a floating point tensor, got {x.dtype}"
        assert x.ndim > 0, "softmax expects a tensor with at least one dimension"
        # The axis is stored unresolved so a single module works for inputs of any
        # rank; normalize it here instead of trusting the caller's sign convention.
        normalized_dim = self.dim if self.dim >= 0 else self.dim + x.ndim
        assert 0 <= normalized_dim < x.ndim, f"dim {self.dim} is out of range for shape {tuple(x.shape)}"
        # Subtract the per-slice maximum before exponentiating: exp overflows above
        # roughly 88 in float32, while the shift cancels out of the final ratio.
        max_values = torch.max(x, dim=normalized_dim, keepdim=True).values
        exp_values = torch.exp(x - max_values)
        # `keepdim` on both reductions lets the sum broadcast against `exp_values`
        # without an explicit reshape that would have to know the input rank.
        return exp_values / torch.sum(exp_values, dim=normalized_dim, keepdim=True)
