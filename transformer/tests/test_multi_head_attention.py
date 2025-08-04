import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

from transformer.layers import multi_head_attention
from transformer.exceptions import FullyMaskedQueryError
from tests.utils import AttentionMockData


def test_attn_sums_to_one():
    atc = AttentionMockData(n_head=2)
    queries, keys, values = atc.get_rand_qkv()
    ib = 0  # Batch
    queries = queries[ib, :, :, :]
    keys = keys[ib, :, :, :]
    values = values[ib, :, :, :]

    result = multi_head_attention(queries, keys, values)
    row_sums = result["attn"].sum(dim=-1)
    assert row_sums.shape == (atc.n_head, atc.l_x)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_masked_position_has_zero_attention():
    atc = AttentionMockData(n_head=2)
    queries, keys, values = atc.get_rand_qkv()
    ib = 0  # Batch
    queries = queries[ib, :, :, :]
    keys = keys[ib, :, :, :]
    values = values[ib, :, :, :]

    mask = torch.zeros((atc.n_head, atc.l_x, atc.l_z))
    mask[0, 0, 2] = float("-inf")
    result = multi_head_attention(queries, keys, values, mask=mask)
    assert torch.allclose(result["attn"][0, 0, 2], torch.tensor(0.0), atol=1e-6)


def test_uniform_scores_give_uniform_attention():
    atc = AttentionMockData(n_head=2)
    queries, keys, values = atc.get_const_qk()
    ib = 0  # Batch
    queries = queries[ib, :, :, :]
    keys = keys[ib, :, :, :]
    values = values[ib, :, :, :]

    result = multi_head_attention(queries, keys, values)
    expected = torch.full((atc.n_head, atc.l_x, atc.l_z), 1.0 / atc.l_z)
    assert result["attn"].shape == expected.shape
    assert torch.allclose(result["attn"], expected, atol=1e-5)


def test_fully_masked_query_raises_error():
    atc = AttentionMockData(n_head=2)
    queries, keys, values = atc.get_rand_qkv()
    ib = 0  # Batch
    queries = queries[ib, :, :, :]
    keys = keys[ib, :, :, :]
    values = values[ib, :, :, :]

    mask = torch.zeros((atc.n_head, atc.l_x, atc.l_z))
    mask[0, 1, :] = float("-inf")
    with pytest.raises(FullyMaskedQueryError):
        multi_head_attention(queries, keys, values, mask=mask)


def test_against_pytorch_scaled_dot_product_attention():
    atc = AttentionMockData(n_head=2)
    queries, keys, values = atc.get_rand_qkv()
    ib = 0  # Batch
    q_flat = queries[ib, :, :, :]
    k_flat = keys[ib, :, :, :]
    v_flat = values[ib, :, :, :]

    # Call your implementation
    result = multi_head_attention(q_flat, k_flat, v_flat)

    # Call PyTorch reference
    expected = scaled_dot_product_attention(
        queries, keys, values, dropout_p=0.0, is_causal=False
    ).squeeze(0)  # remove batch dim

    assert result["output"].shape == expected.shape
    assert torch.allclose(result["output"], expected, atol=1e-5)
