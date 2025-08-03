from pydantic import BaseModel, Field
import torch


class AttentionTestConfig(BaseModel):
    """Configuration for single query attention."""

    l_x: int = Field(2, description="Length of primary sequence (number of queries).")
    l_z: int = Field(
        3, description="Length of context sequence (number of keys and values)."
    )
    d_x: int = Field(4, description="Dimension of primary embeddings.")
    d_z: int = Field(5, description="Dimension of context embeddings.")
    d_attn: int = Field(6, description="Dimension of attention vectors.")
    d_out: int = Field(7, description="Dimension of output vectors.")

    def get_rand_qkv(self):
        queries = torch.randn(self.l_x, self.d_attn)
        keys = torch.randn(self.l_z, self.d_attn)
        values = torch.randn(self.l_z, self.d_out)
        return queries, keys, values

    def get_const_qk(self):
        query_vec = torch.randn(self.d_attn)
        queries = query_vec.repeat(self.l_x, 1)

        key_vec = torch.randn(self.d_attn)
        keys = key_vec.repeat(self.l_z, 1)

        values = torch.randn(self.l_z, self.d_out)
        return queries, keys, values
