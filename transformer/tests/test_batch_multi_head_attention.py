import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

from transformer.layers import batch_multi_head_attention
from transformer.exceptions import FullyMaskedQueryError
from tests.utils import AttentionMockData


def test_attn_sums_to_one():
    amd = AttentionMockData(batch_size=3, n_head=2)
    queries, keys, values = amd.get_rand_qkv()

    result = batch_multi_head_attention(queries, keys, values)
    row_sums = result["attn"].sum(dim=-1)
    assert row_sums.shape == (amd.batch_size, amd.n_head, amd.l_x)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_masked_position_has_zero_attention():
    amd = AttentionMockData(n_head=2)
    queries, keys, values = amd.get_rand_qkv()

    mask = torch.zeros((amd.batch_size, amd.n_head, amd.l_x, amd.l_z))
    mask[0, 0, 0, 2] = float("-inf")
    result = batch_multi_head_attention(queries, keys, values, mask=mask)
    assert torch.allclose(result["attn"][0, 0, 0, 2], torch.tensor(0.0), atol=1e-6)


def test_uniform_scores_give_uniform_attention():
    amd = AttentionMockData(n_head=2)
    queries, keys, values = amd.get_const_qk()

    result = batch_multi_head_attention(queries, keys, values)
    expected = torch.full((amd.batch_size, amd.n_head, amd.l_x, amd.l_z), 1.0 / amd.l_z)
    assert result["attn"].shape == expected.shape
    assert torch.allclose(result["attn"], expected, atol=1e-5)


def test_fully_masked_query_raises_error():
    amd = AttentionMockData(n_head=2)
    queries, keys, values = amd.get_rand_qkv()

    mask = torch.zeros((amd.batch_size, amd.n_head, amd.l_x, amd.l_z))
    mask[0, 0, 1, :] = float("-inf")
    with pytest.raises(FullyMaskedQueryError):
        batch_multi_head_attention(queries, keys, values, mask=mask)


def test_against_pytorch_sdpa():
    amd = AttentionMockData(n_head=2)
    queries, keys, values = amd.get_rand_qkv()

    result = batch_multi_head_attention(queries, keys, values)

    expected = scaled_dot_product_attention(
        queries, keys, values, dropout_p=0.0, is_causal=False
    )

    assert result["output"].shape == expected.shape
    assert torch.allclose(result["output"], expected, atol=1e-5)


def test_against_pytorch_spda_causal_mask():
    amd = AttentionMockData(n_head=2)
    queries, keys, values = amd.get_rand_qkv()

    bool_mask = torch.ones(amd.l_x, amd.l_z).tril().to(dtype=torch.bool)
    bool_mask = bool_mask.expand(amd.batch_size, amd.n_head, -1, -1)
    additive_mask = torch.zeros_like(bool_mask, dtype=torch.float)
    additive_mask.masked_fill_(~bool_mask, float("-inf"))

    result = batch_multi_head_attention(queries, keys, values, mask=additive_mask)

    expected = scaled_dot_product_attention(
        queries, keys, values, attn_mask=additive_mask, dropout_p=0.0, is_causal=False
    )
    assert expected.shape == result["output"].shape
    assert torch.allclose(result["output"], expected, atol=1e-5)

    expected = scaled_dot_product_attention(
        queries, keys, values, dropout_p=0.0, is_causal=True
    )
    assert expected.shape == result["output"].shape
    assert torch.allclose(result["output"], expected, atol=1e-5)
