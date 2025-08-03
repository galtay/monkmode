import pytest
import torch

from transformer.layers import single_query_attention
from transformer.exceptions import FullyMaskedQueryError
from tests.utils import AttentionTestConfig


def test_attn_sums_to_one():
    atc = AttentionTestConfig()
    queries, keys, values = atc.get_rand_qkv()
    query = queries[0, :]

    out = single_query_attention(query, keys, values)
    assert torch.isclose(out["attn"].sum(), torch.tensor(1.0), atol=1e-5)


def test_masked_position_has_zero_attention():
    atc = AttentionTestConfig()
    queries, keys, values = atc.get_rand_qkv()
    query = queries[0, :]

    mask = torch.zeros(atc.l_z)
    mask[2] = float("-inf")  # mask out position 2

    out = single_query_attention(query, keys, values, mask)

    # Check that attention at position 2 is ~0
    assert torch.isclose(out["attn"][2], torch.tensor(0.0), atol=1e-6)


def test_uniform_scores_gives_uniform_attention():
    atc = AttentionTestConfig()
    queries, keys, values = atc.get_const_qk()
    query = queries[0, :]

    out = single_query_attention(query, keys, values)

    expected_attn = torch.full((atc.l_z,), 1.0 / atc.l_z)
    assert torch.allclose(out["attn"], expected_attn, atol=1e-5)


def test_all_positions_masked_raises_error():
    atc = AttentionTestConfig()
    queries, keys, values = atc.get_rand_qkv()
    query = queries[0, :]
    mask = torch.full((atc.l_z,), float("-inf"))

    with pytest.raises(FullyMaskedQueryError):
        single_query_attention(query, keys, values, mask=mask)
