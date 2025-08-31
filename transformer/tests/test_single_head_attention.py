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
    row_sums = result["attn"].sum(dim=-1)
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
    # mask out attention from query iq_mask to key ik_mask
    iq_mask = 0
    ik_mask = 2
    mask[iq_mask, ik_mask] = float("-inf")

    result = single_head_attention(queries, keys, values, mask)
    assert torch.allclose(
        result["attn"][iq_mask, ik_mask], torch.tensor(0.0), atol=1e-6
    )


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


def test_against_pytorch_sdpa():
    amd = AttentionMockData()
    queries, keys, values = amd.get_rand_qkv()
    ib, ih = 0, 0  # Batch, Head
    q_flat = queries[ib, ih, :, :]
    k_flat = keys[ib, ih, :, :]
    v_flat = values[ib, ih, :, :]

    result = single_head_attention(q_flat, k_flat, v_flat)

    expected = (
        scaled_dot_product_attention(
            queries, keys, values, dropout_p=0.0, is_causal=False
        )
        .squeeze(0)
        .squeeze(0)
    )  # remove batch and head dims

    assert result["output"].shape == expected.shape
    assert torch.allclose(result["output"], expected, atol=1e-5)


def test_against_pytorch_sdpa_with_mask():
    amd = AttentionMockData()
    queries, keys, values = amd.get_rand_qkv()
    ib, ih = 0, 0  # Batch, Head
    q_flat = queries[ib, ih, :, :]
    k_flat = keys[ib, ih, :, :]
    v_flat = values[ib, ih, :, :]

    mask = torch.zeros(amd.l_x, amd.l_z)
    # create padding mask
    mask[1, 1:] = float("-inf")  # Mask out key positions > 1 for query 1

    result = single_head_attention(q_flat, k_flat, v_flat, mask=mask)

    # PyTorch expects mask with True for positions that will participate in attention
    pt_mask = mask != float("-inf")

    expected = (
        scaled_dot_product_attention(
            queries, keys, values, attn_mask=pt_mask, dropout_p=0.0, is_causal=False
        )
        .squeeze(0)
        .squeeze(0)
    )  # remove batch and head dims

    assert result["output"].shape == expected.shape
    assert torch.allclose(result["output"], expected, atol=1e-5)
