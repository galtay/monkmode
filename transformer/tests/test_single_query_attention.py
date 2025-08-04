import pytest
import torch

from transformer.layers import single_query_attention
from transformer.exceptions import FullyMaskedQueryError
from tests.utils import AttentionMockData


def test_attn_sums_to_one():
    atc = AttentionMockData()
    queries, keys, values = atc.get_rand_qkv()
    ib, ih, iq = 0, 0, 0  # Batch, Head, Query index
    query = queries[ib, ih, iq, :]
    keys = keys[ib, ih, :, :]
    values = values[ib, ih, :, :]

    result = single_query_attention(query, keys, values)
    attn_sum = result["attn"].sum()
    assert attn_sum.shape == torch.Size([])
    assert torch.isclose(attn_sum, torch.tensor(1.0), atol=1e-5)


def test_masked_position_has_zero_attention():
    atc = AttentionMockData()
    queries, keys, values = atc.get_rand_qkv()
    ib, ih, iq = 0, 0, 0  # Batch, Head, Query index
    query = queries[ib, ih, iq, :]
    keys = keys[ib, ih, :, :]
    values = values[ib, ih, :, :]

    mask = torch.zeros(atc.l_z)
    mask[2] = float("-inf")  # mask out position 2

    result = single_query_attention(query, keys, values, mask)

    # Check that attention at position 2 is ~0
    assert torch.isclose(result["attn"][2], torch.tensor(0.0), atol=1e-6)


def test_uniform_scores_gives_uniform_attention():
    atc = AttentionMockData()
    queries, keys, values = atc.get_const_qk()
    ib, ih, iq = 0, 0, 0  # Batch, Head, Query index
    query = queries[ib, ih, iq, :]
    keys = keys[ib, ih, :, :]
    values = values[ib, ih, :, :]

    result = single_query_attention(query, keys, values)

    expected = torch.full((atc.l_z,), 1.0 / atc.l_z)
    assert result["attn"].shape == expected.shape
    assert torch.allclose(result["attn"], expected, atol=1e-5)


def test_all_positions_masked_raises_error():
    atc = AttentionMockData()
    queries, keys, values = atc.get_rand_qkv()
    ib, ih, iq = 0, 0, 0  # Batch, Head, Query index
    query = queries[ib, ih, iq, :]
    keys = keys[ib, ih, :, :]
    values = values[ib, ih, :, :]
    mask = torch.full((atc.l_z,), float("-inf"))

    with pytest.raises(FullyMaskedQueryError):
        single_query_attention(query, keys, values, mask=mask)
