import pytest
import torch

from transformer.layers import single_head_attention
from transformer.exceptions import FullyMaskedQueryError
from tests.utils import AttentionTestConfig


def test_attn_sums_to_one():
    atc = AttentionTestConfig()
    queries, keys, values = atc.get_rand_qkv()
    out = single_head_attention(queries, keys, values)
    row_sums = out["attn"].sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(atc.l_x), atol=1e-5)


def test_masked_position_has_zero_attention():
    atc = AttentionTestConfig()
    queries, keys, values = atc.get_rand_qkv()

    mask = torch.zeros(atc.l_x, atc.l_z)
    mask[0, 2] = float("-inf")  # Mask out position 2 for query 0

    out = single_head_attention(queries, keys, values, mask)
    assert torch.allclose(out["attn"][0, 2], torch.tensor(0.0), atol=1e-6)


def test_uniform_scores_give_uniform_attention():
    atc = AttentionTestConfig()
    queries, keys, values = atc.get_const_qk()

    out = single_head_attention(queries, keys, values)
    expected = torch.full((atc.l_x, atc.l_z), 1.0 / atc.l_z)
    assert torch.allclose(out["attn"], expected, atol=1e-5)


def test_fully_masked_query_raises_error():
    atc = AttentionTestConfig()
    queries, keys, values = atc.get_rand_qkv()

    mask = torch.zeros(atc.l_x, atc.l_z)
    mask[1, :] = float("-inf")  # Query 1 fully masked

    with pytest.raises(FullyMaskedQueryError):
        single_head_attention(queries, keys, values, mask=mask)
