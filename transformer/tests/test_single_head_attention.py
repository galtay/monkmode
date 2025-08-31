import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

from transformer.layers import single_head_attention
from transformer.exceptions import FullyMaskedQueryError
from tests.utils import AttentionMockData


def test_attn_sums_to_one():
    amd = AttentionMockData()
    queries, keys, values = amd.get_rand_qkv()
    ib, ih = 0, 0  # Batch, Head
    queries = queries[ib, ih, :, :]
    keys = keys[ib, ih, :, :]
    values = values[ib, ih, :, :]

    result = single_head_attention(queries, keys, values)
    row_sums = result["attn"].sum(dim=1)
    expected = torch.ones(amd.l_x)
    assert row_sums.shape == expected.shape
    assert torch.allclose(row_sums, torch.ones(amd.l_x), atol=1e-5)


def test_masked_position_has_zero_attention():
    amd = AttentionMockData()
    queries, keys, values = amd.get_rand_qkv()
    ib, ih = 0, 0  # Batch, Head
    queries = queries[ib, ih, :, :]
    keys = keys[ib, ih, :, :]
    values = values[ib, ih, :, :]

    mask = torch.zeros(amd.l_x, amd.l_z)
    mask[0, 2] = float("-inf")  # Mask out position 2 for query 0

    result = single_head_attention(queries, keys, values, mask)
    assert torch.allclose(result["attn"][0, 2], torch.tensor(0.0), atol=1e-6)


def test_uniform_scores_give_uniform_attention():
    amd = AttentionMockData()
    queries, keys, values = amd.get_const_qk()
    ib, ih = 0, 0  # Batch, Head
    queries = queries[ib, ih, :, :]
    keys = keys[ib, ih, :, :]
    values = values[ib, ih, :, :]

    result = single_head_attention(queries, keys, values)
    expected = torch.full((amd.l_x, amd.l_z), 1.0 / amd.l_z)
    assert result["attn"].shape == expected.shape
    assert torch.allclose(result["attn"], expected, atol=1e-5)


def test_fully_masked_query_raises_error():
    amd = AttentionMockData()
    queries, keys, values = amd.get_rand_qkv()
    ib, ih = 0, 0  # Batch, Head
    queries = queries[ib, ih, :, :]
    keys = keys[ib, ih, :, :]
    values = values[ib, ih, :, :]

    mask = torch.zeros(amd.l_x, amd.l_z)
    mask[1, :] = float("-inf")  # Query 1 fully masked

    with pytest.raises(FullyMaskedQueryError):
        single_head_attention(queries, keys, values, mask=mask)


def test_against_pytorch_scaled_dot_product_attention():
    amd = AttentionMockData()
    queries, keys, values = amd.get_rand_qkv()
    ib, ih = 0, 0  # Batch, Head
    q_flat = queries[ib, ih, :, :]
    k_flat = keys[ib, ih, :, :]
    v_flat = values[ib, ih, :, :]

    # Call your implementation
    result = single_head_attention(q_flat, k_flat, v_flat)

    # Call PyTorch reference
    expected = (
        scaled_dot_product_attention(
            queries, keys, values, dropout_p=0.0, is_causal=False
        )
        .squeeze(0)
        .squeeze(0)
    )  # remove batch and head dims

    assert result["output"].shape == expected.shape
    assert torch.allclose(result["output"], expected, atol=1e-5)
