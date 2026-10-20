import math

import torch
from einops import einsum
from torch import nn

# Samples falling outside `TRUNCATION_STD_MULTIPLIER` standard deviations are redrawn,
# which keeps the tails of the Gaussian from producing outlier activations.
TRUNCATION_STD_MULTIPLIER = 3.0


class Linear(nn.Module):
    """Bias-free linear transformation: y = x @ W.T.

    Args:
        in_features: Size of each input sample (d_in).
        out_features: Size of each output sample (d_out).
        device: Device on which the weight is created.
        dtype: Floating point dtype of the weight.

    Attributes:
        in_features: Width of the last dimension the layer accepts.
        out_features: Width of the last dimension the layer produces.
        weight: Parameter of shape (out_features, in_features).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Stored as (out_features, in_features) so that the "weight" entry of the state
        # dict matches the reference PyTorch layout used by the tests.
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.init_weight(self.weight)

    @staticmethod
    def init_weight(weight: torch.Tensor) -> torch.Tensor:
        """Fill `weight` in place with the truncated-normal Linear initialization.

        The fan-in and fan-out sizes are read from the last two dimensions, so the
        same helper can initialize a stacked tensor such as the (3, d_model, d_model)
        QKV projection used by multi-head self-attention.

        Args:
            weight: Tensor of shape (..., out_features, in_features).

        Returns:
            The same tensor, initialized in place.

        Raises:
            AssertionError: If `weight` has fewer than 2 dimensions.
        """
        assert weight.ndim >= 2, f"expected at least 2 dims, got shape {tuple(weight.shape)}"
        out_features, in_features = weight.shape[-2:]
        # Xavier-style scale averaged over fan-in and fan-out, as specified by the
        # assignment for the truncated normal initialization.
        std = math.sqrt(2.0 / (in_features + out_features))
        return nn.init.trunc_normal_(
            weight,
            mean=0.0,
            std=std,
            a=-TRUNCATION_STD_MULTIPLIER * std,
            b=TRUNCATION_STD_MULTIPLIER * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the layer to the trailing dimension of `x`.

        Args:
            x: Tensor of shape (..., in_features).

        Returns:
            Tensor of shape (..., out_features), broadcasting over all leading dims.

        Raises:
            AssertionError: If `x` is 0-dimensional or its last dim is not `in_features`.
        """
        # Both checks are shape-only, so they never force a device synchronization.
        # `assert` costs nothing under `python -O`, which is how a tuned training run
        # would drop them.
        assert x.ndim >= 1, f"expected at least 1 dim, got shape {tuple(x.shape)}"
        assert x.shape[-1] == self.in_features, (
            f"last dim of input is {x.shape[-1]}, but in_features is {self.in_features}"
        )
        return einsum(x, self.weight, "... dim_in, dim_out dim_in -> ... dim_out")
