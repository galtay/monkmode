import pytest
import torch

from transformer.attn_layers import single_query_attention
from transformer.exceptions import FullyMaskedQueryError
from tests.utils import AttentionMockData, get_devices, get_dtypes, ensure_supported


def get_amd():
    return AttentionMockData(
        l_x=8,
        l_z=12,
        d_x=16,
        d_z=18,
        d_attn=20,
        d_out=22,
        batch_size=1,
        n_head=1,
    )


def get_qkv_slices(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
):
    ib, ih, iq = 0, 0, 0  # Batch, Head, Query
    return (
        queries[ib, ih, iq, :],
        keys[ib, ih, :, :],
        values[ib, ih, :, :],
    )


@pytest.mark.parametrize("device", get_devices())
@pytest.mark.parametrize("dtype", get_dtypes())
def test_attn_sums_to_one(device: str, dtype: torch.dtype):
    ensure_supported(device, dtype)
    amd = get_amd()
    queries, keys, values = amd.get_rand_qkv(device=device, dtype=dtype)
    q_slice, k_slice, v_slice = get_qkv_slices(queries, keys, values)
    result = single_query_attention(q_slice, k_slice, v_slice)
    assert result["attn"].device.type == device
    assert result["attn"].dtype == dtype
    assert result["attn"].shape == (amd.l_z,)
    assert torch.isfinite(result["attn"]).all()

    row_sum = result["attn"].sum().to(torch.float32)
    expected = torch.ones_like(row_sum, device=device, dtype=torch.float32)
    assert row_sum.shape == torch.Size([])
    assert torch.isclose(row_sum, expected, atol=1e-5)


@pytest.mark.parametrize("device", get_devices())
@pytest.mark.parametrize("dtype", get_dtypes())
def test_uniform_scores_gives_uniform_attention(device: str, dtype: torch.dtype):
    ensure_supported(device, dtype)
    amd = get_amd()
    queries, keys, values = amd.get_repeated_qk(device=device, dtype=dtype)
    q_slice, k_slice, v_slice = get_qkv_slices(queries, keys, values)
    result = single_query_attention(q_slice, k_slice, v_slice)
    assert result["attn"].device.type == device
    assert result["attn"].dtype == dtype
    assert result["attn"].shape == (amd.l_z,)
    assert torch.isfinite(result["attn"]).all()

    expected = torch.full((amd.l_z,), 1.0 / amd.l_z, device=device, dtype=dtype)
    assert result["attn"].shape == expected.shape
    assert torch.allclose(result["attn"], expected, atol=1e-5)


@pytest.mark.parametrize("device", get_devices())
@pytest.mark.parametrize("dtype", get_dtypes())
def test_additive_masked_position_has_zero_attention(device: str, dtype: torch.dtype):
    ensure_supported(device, dtype)
    amd = get_amd()
    queries, keys, values = amd.get_rand_qkv(device=device, dtype=dtype)
    q_slice, k_slice, v_slice = get_qkv_slices(queries, keys, values)
    attn_mask = torch.zeros(amd.l_z, device=device, dtype=dtype)
    ik_mask = 2  # prevent attention at position ik_mask
    attn_mask[ik_mask] = float("-inf")

    result = single_query_attention(q_slice, k_slice, v_slice, attn_mask=attn_mask)
    assert result["attn"].device.type == device
    assert result["attn"].dtype == dtype
    assert result["attn"].shape == (amd.l_z,)
    assert torch.isfinite(result["attn"]).all()

    assert torch.isclose(
        result["attn"][ik_mask],
        torch.tensor(0.0, device=device, dtype=dtype),
        atol=1e-6,
    )


@pytest.mark.parametrize("device", get_devices())
@pytest.mark.parametrize("dtype", get_dtypes())
def test_boolean_masked_position_has_zero_attention(device: str, dtype: torch.dtype):
    ensure_supported(device, dtype)
    amd = get_amd()
    queries, keys, values = amd.get_rand_qkv(device=device, dtype=dtype)
    q_slice, k_slice, v_slice = get_qkv_slices(queries, keys, values)

    attn_mask = torch.ones(amd.l_z, device=device, dtype=torch.bool)
    ik_mask = 2  # prevent attention at position ik_mask
    attn_mask[ik_mask] = False

    result = single_query_attention(q_slice, k_slice, v_slice, attn_mask=attn_mask)
    assert torch.isclose(
        result["attn"][ik_mask],
        torch.tensor(0.0, device=device, dtype=dtype),
        atol=1e-6,
    )

@pytest.mark.parametrize("device", get_devices())
@pytest.mark.parametrize("dtype", get_dtypes())
def test_all_positions_additive_masked_raises_error(device: str, dtype: torch.dtype):
    ensure_supported(device, dtype)
    amd = get_amd()
    queries, keys, values = amd.get_rand_qkv(device=device, dtype=dtype)
    q_slice, k_slice, v_slice = get_qkv_slices(queries, keys, values)

    attn_mask = torch.ones(amd.l_z, device=device, dtype=dtype) * float("-inf")
    with pytest.raises(FullyMaskedQueryError):
        single_query_attention(q_slice, k_slice, v_slice, attn_mask=attn_mask)

@pytest.mark.parametrize("device", get_devices())
@pytest.mark.parametrize("dtype", get_dtypes())
def test_all_positions_boolean_masked_raises_error(device: str, dtype: torch.dtype):
    ensure_supported(device, dtype)
    amd = get_amd()
    queries, keys, values = amd.get_rand_qkv(device=device, dtype=dtype)
    q_slice, k_slice, v_slice = get_qkv_slices(queries, keys, values)

    attn_mask = torch.zeros(amd.l_z, device=device, dtype=torch.bool)
    with pytest.raises(FullyMaskedQueryError):
        single_query_attention(q_slice, k_slice, v_slice, attn_mask=attn_mask)



