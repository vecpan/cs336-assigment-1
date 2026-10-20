"""Causal multi-head self-attention."""

import torch
from einops import einsum, rearrange
from torch import nn

from .linear import Linear
from .rope import RotaryPositionalEmbedding
from .scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with an optional RoPE module.

    The input is projected once into queries, keys and values for all heads, so the
    three projections share a single matrix multiply. Each head then applies scaled
    dot-product attention with a causal mask, the heads are concatenated, and the
    result is projected back to ``d_model``:

    ``MultiHeadSelfAttention(x) = W_O MultiHead(W_Q x, W_K x, W_V x)``

    Args:
        d_model: Width of the input and output (last) dimension.
        num_heads: Number of attention heads. ``d_model`` must be divisible by it.
        device: Device on which the weights are created.
        dtype: Floating point dtype of the weights.
        rope: Optional rotary position embedding applied to each head's queries and
            keys. When omitted, the attention is position independent.

    Attributes:
        d_model: Width of the input and output dimension.
        num_heads: Number of attention heads.
        d_head: Width of a single head (``d_model // num_heads``).
        qkv_proj_weight: Fused query/key/value projection of shape
            (3, d_model, d_model), ordered as query, key, value.
        output_proj_weight: Output projection ``W_O`` of shape (d_model, d_model).
        rope: The rotary position embedding module, or ``None``.
        attention: Shared scaled dot-product attention layer.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        # Configuration guards: the head split below would silently mix channels
        # between heads if d_model were not divisible by num_heads.
        assert d_model > 0, f"d_model must be positive, got {d_model}"
        assert num_heads > 0, f"num_heads must be positive, got {num_heads}"
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        if rope is not None:
            assert rope.d_k == self.d_head, f"rope.d_k ({rope.d_k}) must equal d_model // num_heads ({self.d_head})"
        # Each (d_model, d_model) block, including the three QKV blocks, gets the same
        # truncated-normal distribution a stand-alone Linear layer would use.
        self.qkv_proj_weight = nn.Parameter(torch.empty(3, d_model, d_model, device=device, dtype=dtype))
        self.output_proj_weight = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        Linear.init_weight(self.qkv_proj_weight)
        Linear.init_weight(self.output_proj_weight)

        self.rope = rope
        # Scaled dot-product attention is stateless, so one instance can serve every
        # call; constructing it in forward would allocate a module per invocation.
        self.attention = ScaledDotProductAttention()

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """Apply causal multi-head self-attention to the last dimension of `x`.

        Args:
            x: Tensor of shape (..., sequence_length, d_model).
            token_positions: Integer tensor of shape (..., sequence_length) holding
                the position of each token. It is required when a RoPE module was
                supplied and is ignored otherwise.

        Returns:
            Tensor of shape (..., sequence_length, d_model).

        Raises:
            AssertionError: If `x` is not at least 2-dimensional, its last dimension
                is not `d_model`, or a RoPE module was supplied but
                `token_positions` is None.
        """
        assert x.ndim >= 2, f"expected at least 2 dims, got shape {tuple(x.shape)}"
        assert x.shape[-1] == self.d_model, f"last dim of input is {x.shape[-1]}, but d_model is {self.d_model}"
        sequence_length = x.shape[-2]

        # Project all heads at once: (..., seq, d_model) x (3, d_model, d_model)
        # -> (..., 3, seq, d_model). Splitting the trailing axis into (heads, d_head)
        # then yields (..., 3, heads, seq, d_head).
        qkv = einsum(x, self.qkv_proj_weight, "... seq d_in, p d_out d_in -> ... p seq d_out")
        qkv = rearrange(qkv, "... p seq (h d_head) -> ... p h seq d_head", h=self.num_heads)
        query, key, value = qkv.unbind(dim=-4)

        # True marks a position the query may attend to: token i sees tokens <= i.
        causal_mask = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=x.device))

        if self.rope is not None:
            # RoPE needs an explicit position per token; silently falling back to
            # 0..seq-1 would hide a caller that forgot to shift positions for a
            # chunk that does not start at the beginning of the sequence.
            assert token_positions is not None, "token_positions must be provided when a RoPE module is set"
            query = self.rope.forward(query, token_positions)
            key = self.rope.forward(key, token_positions)

        attended = self.attention(query, key, value, causal_mask)
        # (..., heads, seq, d_head) -> (..., seq, heads * d_head = d_model).
        attended = rearrange(attended, "... h seq d_head -> ... seq (h d_head)")
        return einsum(attended, self.output_proj_weight, "... seq d_in, d_out d_in -> ... seq d_out")
