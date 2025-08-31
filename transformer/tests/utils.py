from pydantic import BaseModel, Field
import torch


class AttentionMockData(BaseModel):
    """Configuration for single query attention.

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

    def get_rand_qkv(self):
        shape_q = (self.batch_size, self.n_head, self.l_x, self.d_attn)
        shape_k = (self.batch_size, self.n_head, self.l_z, self.d_attn)
        shape_v = (self.batch_size, self.n_head, self.l_z, self.d_out)

        queries = torch.randn(*shape_q)
        keys = torch.randn(*shape_k)
        values = torch.randn(*shape_v)
        return queries, keys, values

    def get_const_qk(self):
        """Returns constant queries/keys for uniform attention testing."""
        q_vec = torch.randn(self.d_attn).view(1, 1, 1, -1)
        k_vec = torch.randn(self.d_attn).view(1, 1, 1, -1)

        # Repeat to shape: (batch_size, n_head, seq_len, d_attn)
        queries = q_vec.repeat(self.batch_size, self.n_head, self.l_x, 1)
        keys = k_vec.repeat(self.batch_size, self.n_head, self.l_z, 1)
        values = torch.randn(self.batch_size, self.n_head, self.l_z, self.d_out)

        return queries, keys, values
