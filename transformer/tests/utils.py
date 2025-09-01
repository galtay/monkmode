from pydantic import BaseModel, Field
import torch


class AttentionMockData(BaseModel):
    """Create mock data for testing attention functions.

    d_attn should be evenly divisible by n_head
    """

    l_x: int = Field(2, description="Primary sequence length (queries).")
    l_z: int = Field(3, description="Context sequence length (keys/values).")
    d_x: int = Field(4, description="Dimension of primary embeddings.")
    d_z: int = Field(5, description="Dimension of context embeddings.")
    d_attn: int = Field(
        6, description="Dimension of attention for each head (queries/keys)."
    )
    d_out: int = Field(7, description="Dimension of output vectors (values).")
    batch_size: int = Field(1, description="Batch size for queries, keys, and values.")
    n_head: int = Field(1, description="Number of attention heads.")
    seed: int = Field(3937, description="Random seed for reproducibility.")

    def _get_generator(self):
        return torch.Generator().manual_seed(self.seed)

    def _get_q_shape(self):
        return (self.batch_size, self.n_head, self.l_x, self.d_attn)

    def _get_k_shape(self):
        return (self.batch_size, self.n_head, self.l_z, self.d_attn)

    def _get_v_shape(self):
        return (self.batch_size, self.n_head, self.l_z, self.d_out)

    def get_rand_qkv(
        self,
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """Returns random queries, keys, and values."""
        generator = self._get_generator()
        queries = torch.randn(
            *self._get_q_shape(), device=device, dtype=dtype, generator=generator
        )
        keys = torch.randn(
            *self._get_k_shape(), device=device, dtype=dtype, generator=generator
        )
        values = torch.randn(
            *self._get_v_shape(), device=device, dtype=dtype, generator=generator
        )
        return queries, keys, values

    def get_repeated_qk(
        self,
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """Returns constant queries/keys for uniform attention testing."""
        generator = self._get_generator()

        # Create one random q and one random k vector
        query = torch.randn(
            self.d_attn, device=device, dtype=dtype, generator=generator
        )
        key = torch.randn(self.d_attn, device=device, dtype=dtype, generator=generator)

        # Repeat to shape: (batch_size, n_head, seq_len, d_attn)
        queries = query.view(1, 1, 1, -1).repeat(
            self.batch_size, self.n_head, self.l_x, 1
        )
        keys = key.view(1, 1, 1, -1).repeat(self.batch_size, self.n_head, self.l_z, 1)
        # use random values
        values = torch.randn(
            *self._get_v_shape(),
            device=device,
            dtype=dtype,
            generator=generator,
        )

        return queries, keys, values
