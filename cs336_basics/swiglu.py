import torch
from torch import nn

from .linear import Linear


class SwiGLU(nn.Module):
    """Gated feed-forward network of a transformer block.

    The layer computes ``w2(silu(w1(x)) * w3(x))``: ``w1`` projects up to ``d_ff`` and
    forms the gate, ``w3`` projects up to ``d_ff`` and carries the value being gated,
    and ``w2`` projects the elementwise product back down to ``d_model``.

    Args:
        d_model: Width of the input and output (last) dimension.
        d_ff: Width of the internal hidden dimension.
        device: Device on which the weights are created.
        dtype: Floating point dtype of the weights.

    Attributes:
        d_model: Width of the input and output dimension.
        w1: Gate projection, mapping d_model -> d_ff.
        w2: Down projection, mapping d_ff -> d_model.
        w3: Up projection, mapping d_model -> d_ff.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        # Configuration guards: a zero or negative width would create an empty
        # projection and only fail later, deep inside the matrix multiply.
        assert d_model > 0, f"d_model must be positive, got {d_model}"
        assert d_ff > 0, f"d_ff must be positive, got {d_ff}"
        self.d_model = d_model
        # The submodule names follow the reference checkpoint layout
        # (`ffn.w1.weight`, `ffn.w2.weight`, `ffn.w3.weight`), so a state dict taken
        # from it can be loaded without renaming any keys.
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated feed-forward transformation to the trailing dimension of `x`.

        Args:
            x: Tensor of shape (..., d_model).

        Returns:
            Tensor of shape (..., d_model).

        Raises:
            AssertionError: If `x` is 0-dimensional or its last dim is not `d_model`.
        """
        # Same shape-only guards as `Linear`/`RMSNorm`: they never force a device
        # synchronization and are stripped under `python -O`. `w1`/`w3` would surface a
        # width mismatch on their own, but checking here names the offending input
        # instead of failing inside a projection.
        assert x.ndim >= 1, f"expected at least 1 dim, got shape {tuple(x.shape)}"
        assert x.shape[-1] == self.d_model, f"last dim of input is {x.shape[-1]}, but d_model is {self.d_model}"
        gate = self.w1(x)
        # SiLU(a) = a * sigmoid(a), applied to the gate projection only.
        gate_activated = gate * torch.sigmoid(gate)
        # The up projection is a separate linear map of the same input; the two
        # branches are combined elementwise, which is what makes this a GLU.
        value = self.w3(x)
        return self.w2(gate_activated * value)
