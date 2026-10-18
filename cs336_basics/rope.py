"""Rotary position embedding (RoPE) for query and key tensors."""

import torch
from einops import einsum, rearrange
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    """Rotate adjacent channel pairs of a tensor by an angle that grows with the position.

    Channel pair ``i`` (``i = 0 .. d_k/2 - 1``) of the vector at position ``p`` is
    rotated by the angle ``p * theta ** (-2i / d_k)``. Because the angle only depends
    on the position, the dot product of two rotated vectors depends on the distance
    between the positions rather than on the positions themselves.

    The angles only depend on ``(p, i)``, so ``cos`` and ``sin`` are precomputed once
    for every position up to ``max_seq_len`` and then gathered by position, instead of
    being recomputed from scratch on every call.

    Args:
        theta: Base of the geometric progression of rotation frequencies (10000 in
            the reference implementation); larger values make the angles grow slower.
        d_k: Width of the rotated (last) dimension. Must be even, since the rotation
            acts on adjacent pairs of channels.
        max_seq_len: Number of positions to precompute. The tables have
            ``max_seq_len`` rows, so every position passed to ``forward`` must lie in
            ``[0, max_seq_len)``.
        device: Device on which the tables are created.

    Attributes:
        d_k: Width of the last dimension the layer accepts.
        max_seq_len: Number of cached positions, i.e. the valid position range.
        cos_table: Non-persistent buffer of shape (max_seq_len, d_k/2) holding
            ``cos(p * theta ** (-2i / d_k))``.
        sin_table: Non-persistent buffer of shape (max_seq_len, d_k/2) holding
            ``sin(p * theta ** (-2i / d_k))``.
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        # Shape-only guards on the configuration: an odd d_k cannot be split into
        # pairs, and the remaining two values feed a power and a table length.
        assert d_k > 0 and d_k % 2 == 0, f"d_k must be a positive even number, got {d_k}"
        assert theta > 0, f"theta must be positive, got {theta}"
        assert max_seq_len > 0, f"max_seq_len must be positive, got {max_seq_len}"
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # `pair_indices` is 1-based so the first pair gets angle 0 (no rotation) and
        # the exponent sweeps [0, (d_k - 2) / d_k] across the pair axis.
        pair_indices = torch.arange(1, d_k // 2 + 1, device=device, dtype=torch.float32)
        exponent = (2.0 * pair_indices - 2.0) / d_k
        inv_freq = 1.0 / torch.pow(theta, exponent)
        # Outer product of positions and inverse frequencies: row `p`, column `i` is
        # the angle of channel pair `i` at position `p`, so a plain row lookup by
        # position is all `forward` needs.
        position_ids = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = einsum(position_ids, inv_freq, "seq_len, d_pair -> seq_len d_pair")
        # Registered as non-persistent buffers rather than parameters: they are
        # constants, so they must not show up in `state_dict` (a `strict=True` load of
        # a reference checkpoint would then fail), but they still have to follow the
        # module across `.to(device)` / `.to(dtype)` calls.
        self.register_buffer("cos_table", torch.cos(angles), persistent=False)
        self.register_buffer("sin_table", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Apply the rotation to every position of `x`.

        Args:
            x: Tensor of shape (..., sequence_length, d_k), e.g. the queries or keys
                of one attention layer after the head split.
            token_positions: Integer tensor of shape (..., sequence_length) with the
                position of each token in `x`; leading dimensions must broadcast
                against the leading dimensions of `x`.

        Returns:
            Tensor with the same shape and dtype as `x`.

        Raises:
            AssertionError: If `x` is not at least 2-dimensional, its last dim is not
                `d_k`, `token_positions` is not integral or outside
                ``[0, max_seq_len)``, or the two sequence lengths disagree.
        """
        # Shape and dtype only, so none of these force a device synchronization.
        # Checking the last dim here, rather than letting the gather fail, names the
        # offending input and catches the wrong-tensor bug (e.g. passing the full
        # (..., d_model) projection instead of one head's slice).
        assert x.ndim >= 2, f"expected at least 2 dims, got shape {tuple(x.shape)}"
        assert x.shape[-1] == self.d_k, f"last dim of input is {x.shape[-1]}, but d_k is {self.d_k}"
        assert token_positions.dtype in (torch.int32, torch.int64), (
            f"token positions must be integers, got {token_positions.dtype}"
        )
        assert token_positions.shape[-1] == x.shape[-2], (
            f"got {token_positions.shape[-1]} positions for a sequence of length {x.shape[-2]}"
        )
        # The tables are only `max_seq_len` rows long, so an out-of-range position is
        # now a hard error instead of being computed on the fly. The check is a
        # reduction and therefore syncs with the accelerator once per call; drop it
        # (or run with `-O`) if it ever shows up in a profile.
        if token_positions.numel() > 0:
            min_position, max_position = int(token_positions.min()), int(token_positions.max())
            assert 0 <= min_position and max_position < self.max_seq_len, (
                f"token position outside [0, {self.max_seq_len}): min={min_position}, max={max_position}"
            )

        # Gather the rows of the tables for the positions in this call, then match the
        # activation dtype. The tables are kept in float32 because the angles span a
        # large range, so low-precision cos/sin would lose accuracy; the caller still
        # expects the input dtype back.
        cos = self.cos_table[token_positions].to(x.dtype)
        sin = self.sin_table[token_positions].to(x.dtype)

        # Split the d_k channels into d_k/2 adjacent pairs and rotate each pair in its
        # own plane:
        #   y[2i]   =  x[2i] * cos_i - x[2i+1] * sin_i
        #   y[2i+1] =  x[2i] * sin_i + x[2i+1] * cos_i
        # which is the multiplication by [[cos, -sin], [sin, cos]] used by the
        # on-the-fly variant, only written elementwise because cos/sin are now tensors
        # indexed by position rather than 2x2 matrices.
        x_pairs = rearrange(x, "... (d_pair two) -> ... d_pair two", two=2)
        x_even = x_pairs[..., 0]
        x_odd = x_pairs[..., 1]
        rotated_pairs = torch.stack([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1)
        # Interleave the results back into the original channel order.
        return rearrange(rotated_pairs, "... d_pair two -> ... (d_pair two)")
