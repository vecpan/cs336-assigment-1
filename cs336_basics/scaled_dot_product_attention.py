"""Scaled dot-product attention (SDPA)."""

import math

import torch
from einops import einsum
from torch import nn

from .softmax import Softmax


class ScaledDotProductAttention(nn.Module):
    """Attend to values by comparing every query against every key.

    Given a query tensor of shape ``(..., queries, d_k)``, a key tensor of shape
    ``(..., keys, d_k)`` and a value tensor of shape ``(..., keys, d_v)``, the layer
    computes

    .. code-block:: text

        softmax(query @ key.T / sqrt(d_k), dim=-1) @ value

    and returns a tensor of shape ``(..., queries, d_v)``. Dividing by ``sqrt(d_k)``
    keeps the dot products from growing with the head width, which would otherwise
    saturate the softmax and shrink its gradient.

    The layer has no parameters and reads ``d_k`` from the inputs, so a single
    instance handles any head width and any number of leading batch/head axes.

    Attributes:
        softmax: Module normalizing the attention scores along the key axis.
    """

    def __init__(self) -> None:
        super().__init__()
        # `Softmax` is stateless, so it is built once here instead of once per call.
        self.softmax = Softmax(dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the attention output of `query` against `key` and `value`.

        Args:
            query: Tensor of shape (..., queries, d_k).
            key: Tensor of shape (..., keys, d_k); its leading axes must broadcast
                against those of `query`.
            value: Tensor of shape (..., keys, d_v); its sequence length must match
                the one of `key`.
            mask: Optional boolean tensor broadcastable to (..., queries, keys) in
                which ``True`` marks a position the query may attend to. Masked
                positions are set to ``-inf`` before the softmax, giving them weight
                exactly zero.

        Returns:
            Tensor of shape (..., queries, d_v).

        Raises:
            AssertionError: If an input is not at least 2-dimensional, if the query
                and key widths differ, if the key and value sequence lengths differ,
                or if `mask` is not boolean.
        """
        # Shape-only guards, matching the other modules: they never force a device
        # synchronization. Checking the widths here names the offending input instead
        # of leaving the mismatch to surface inside `einsum`.
        assert query.ndim >= 2, f"query must be at least 2-dimensional, got shape {tuple(query.shape)}"
        assert key.ndim >= 2, f"key must be at least 2-dimensional, got shape {tuple(key.shape)}"
        assert value.ndim >= 2, f"value must be at least 2-dimensional, got shape {tuple(value.shape)}"
        d_k = query.shape[-1]
        assert key.shape[-1] == d_k, f"query width is {d_k}, but key width is {key.shape[-1]}"
        assert value.shape[-2] == key.shape[-2], (
            f"key sequence length is {key.shape[-2]}, but value sequence length is {value.shape[-2]}"
        )

        # Scale by 1/sqrt(d_k) before the softmax so the scores stay in a range where
        # the exponential does not saturate for large head widths.
        scores = einsum(query, key, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
        if mask is not None:
            assert mask.dtype == torch.bool, f"mask must be boolean, got {mask.dtype}"
            # -inf gives the masked entries zero weight after the softmax. A query row
            # whose mask is entirely False is then all -inf and produces NaN, which
            # surfaces the invalid mask instead of silently attending to every key.
            scores = scores.masked_fill(~mask, float("-inf"))

        attention_weights = self.softmax(scores)
        return einsum(attention_weights, value, "... queries keys, ... keys d_v -> ... queries d_v")
