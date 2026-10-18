import torch
from torch import nn

# Truncation bound for the (mean 0, std 1) embedding initialization.
TRUNCATION_STD_MULTIPLIER = 3.0


class Embedding(nn.Module):
    """Lookup table mapping token ids to dense vectors.

    Args:
        num_embeddings: Size of the vocabulary, i.e. the number of rows.
        embedding_dim: Size of each embedding vector (d_model).
        device: Device on which the table is created.
        dtype: Floating point dtype of the table.

    Attributes:
        num_embeddings: Number of rows of `weight`, i.e. the valid id range.
        embedding_dim: Number of columns of `weight`.
        weight: Parameter of shape (num_embeddings, embedding_dim).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        initial_weight = nn.init.trunc_normal_(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
            mean=0.0,
            std=1.0,
            a=-TRUNCATION_STD_MULTIPLIER,
            b=TRUNCATION_STD_MULTIPLIER,
        )
        self.weight = nn.Parameter(initial_weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Gather the embedding vector of every id in `token_ids`.

        Args:
            token_ids: Integer tensor of shape (...,), values in [0, num_embeddings).

        Returns:
            Tensor of shape (..., embedding_dim).

        Raises:
            AssertionError: If `token_ids` is not integral, or holds an id outside
                [0, num_embeddings).
        """
        assert token_ids.dtype in (torch.int32, torch.int64), f"token ids must be integers, got {token_ids.dtype}"
        # Check the range explicitly: an out-of-range id is not reliably diagnosed on
        # accelerator backends, it can just read garbage. Unlike the dtype check this
        # one is a reduction, so it syncs with the GPU; drop it (or run with -O) if it
        # ever shows up in a profile.
        if token_ids.numel() > 0:
            min_id, max_id = int(token_ids.min()), int(token_ids.max())
            assert 0 <= min_id and max_id < self.num_embeddings, (
                f"token id outside [0, {self.num_embeddings}): min={min_id}, max={max_id}"
            )
        return self.weight[token_ids]
