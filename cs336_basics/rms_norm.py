import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization without the mean subtraction.

    For each input vector ``a`` of width ``d_model`` the layer computes
    ``g * a / sqrt(mean(a**2) + eps)``, where ``g`` is a learned per-channel gain.

    Args:
        d_model: Size of the normalized (last) dimension.
        eps: Value added to the mean square before the square root; keeps the
            reciprocal well defined when the input is all zeros.
        device: Device on which the gain is created.
        dtype: Floating point dtype of the gain.

    Attributes:
        d_model: Width of the last dimension the layer accepts.
        eps: Numerical stability constant used in the denominator.
        weight: Gain parameter ``g`` of shape (d_model,).
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        # Configuration guards: a non-positive width makes an empty gain, and a
        # non-positive eps can turn the reciprocal RMS into a division by zero.
        assert d_model > 0, f"d_model must be positive, got {d_model}"
        assert eps > 0, f"eps must be positive, got {eps}"
        self.d_model = d_model
        self.eps = eps
        # The gain starts at ones, so a freshly built layer is a pure
        # normalization and does not rescale the activations.
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the trailing dimension of `x` and scale it channel-wise.

        Args:
            x: Tensor of shape (..., d_model).

        Returns:
            Tensor with the same shape and dtype as `x`.

        Raises:
            AssertionError: If `x` is 0-dimensional or its last dim is not `d_model`.
        """
        # Both checks are shape-only, so they never force a device synchronization.
        # The width check also catches the common mistake of normalizing over the
        # wrong axis: without it a mismatched width would broadcast silently and the
        # mean would end up taken over the batch instead of the features. Asserts are
        # stripped under `python -O`, so they cost nothing in a tuned training run.
        assert x.ndim >= 1, f"expected at least 1 dim, got shape {tuple(x.shape)}"
        assert x.shape[-1] == self.d_model, f"last dim of input is {x.shape[-1]}, but d_model is {self.d_model}"
        # Accumulate in float32 even when the input is low precision: the mean of the
        # squares loses too much range in bf16/fp16, whereas the caller still expects
        # the output in the dtype it passed in.
        input_dtype = x.dtype
        x_float32 = x.to(torch.float32)
        # Mean of the squares over the feature axis; keepdim lets the inverse RMS below
        # broadcast against the (..., d_model) input.
        mean_square = x_float32.pow(2).mean(dim=-1, keepdim=True)
        # rsqrt is a single fused op, so this needs no separate division by the root.
        inverse_rms = torch.rsqrt(mean_square + self.eps)
        normalized = x_float32 * inverse_rms
        # Apply the learned per-channel gain, then hand back the caller's dtype.
        return (normalized * self.weight).to(input_dtype)
