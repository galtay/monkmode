import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

from transformer.attn_layers import single_head_attention
from transformer.exceptions import FullyMaskedQueryError
from tests.utils import AttentionMockData, get_devices, get_dtypes, ensure_supported


def get_amd():
    return AttentionMockData(
        l_x=8,
        l_z=12,
        d_x=16,
        d_z=18,
        d_attn=20,
        d_out=20,
        batch_size=1,
        n_head=1,
    )


def get_qkv_slices(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
):
    ib, ih = 0, 0  # Batch, Head
    return (
        queries[ib, ih, :, :],
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
    result = single_head_attention(q_slice, k_slice, v_slice)

    assert result["attn"].device.type == device
    assert result["attn"].dtype == dtype
    assert result["attn"].shape == (amd.l_x, amd.l_z)
    assert torch.isfinite(result["attn"]).all()

    row_sums = result["attn"].sum(dim=-1).to(torch.float32)
    expected = torch.ones(amd.l_x, device=device, dtype=torch.float32)
    assert row_sums.shape == expected.shape
    assert torch.allclose(row_sums, expected, atol=1e-5)


@pytest.mark.parametrize("device", get_devices())
@pytest.mark.parametrize("dtype", get_dtypes())
def test_uniform_scores_give_uniform_attention(device: str, dtype: torch.dtype):
    ensure_supported(device, dtype)
    amd = get_amd()
    queries, keys, values = amd.get_repeated_qk(device=device, dtype=dtype)
    q_slice, k_slice, v_slice = get_qkv_slices(queries, keys, values)
    result = single_head_attention(q_slice, k_slice, v_slice)
    assert result["attn"].device.type == device
    assert result["attn"].dtype == dtype
    assert result["attn"].shape == (amd.l_x, amd.l_z)
    assert torch.isfinite(result["attn"]).all()

    expected = torch.full((amd.l_x, amd.l_z), 1.0 / amd.l_z, device=device, dtype=dtype)
    assert result["attn"].shape == expected.shape
    assert torch.allclose(result["attn"], expected, atol=1e-5)


@pytest.mark.parametrize("device", get_devices())
@pytest.mark.parametrize("dtype", get_dtypes())
def test_additive_masked_position_has_zero_attention(device: str, dtype: torch.dtype):
    ensure_supported(device, dtype)
    amd = get_amd()
    queries, keys, values = amd.get_rand_qkv(device=device, dtype=dtype)
    q_slice, k_slice, v_slice = get_qkv_slices(queries, keys, values)
    attn_mask = torch.zeros(amd.l_x, amd.l_z, device=device, dtype=dtype)
    # prevent attention from query iq_mask to key ik_mask
    iq_mask = 0
    ik_mask = 2
    attn_mask[iq_mask, ik_mask] = float("-inf")

    result = single_head_attention(q_slice, k_slice, v_slice, attn_mask=attn_mask)
    assert result["attn"].device.type == device
    assert result["attn"].dtype == dtype
    assert result["attn"].shape == (amd.l_x, amd.l_z)
    assert torch.isfinite(result["attn"]).all()

    assert torch.isclose(
        result["attn"][iq_mask, ik_mask],
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
    attn_mask = torch.ones(amd.l_x, amd.l_z, device=device, dtype=torch.bool)
    # prevent attention from query iq_mask to key ik_mask
    iq_mask = 0
    ik_mask = 2
    attn_mask[iq_mask, ik_mask] = False

    result = single_head_attention(q_slice, k_slice, v_slice, attn_mask=attn_mask)
    assert result["attn"].device.type == device
    assert result["attn"].dtype == dtype
    assert result["attn"].shape == (amd.l_x, amd.l_z)
    assert torch.isfinite(result["attn"]).all()

    assert torch.isclose(
        result["attn"][iq_mask, ik_mask],
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

    attn_mask = torch.zeros(amd.l_x, amd.l_z, device=device, dtype=dtype)
    iq_mask = 2
    attn_mask[iq_mask, :] = float("-inf")
    with pytest.raises(FullyMaskedQueryError):
        single_head_attention(q_slice, k_slice, v_slice, attn_mask=attn_mask)


@pytest.mark.parametrize("device", get_devices())
@pytest.mark.parametrize("dtype", get_dtypes())
def test_all_positions_boolean_masked_raises_error(device: str, dtype: torch.dtype):
    ensure_supported(device, dtype)
    amd = get_amd()
    queries, keys, values = amd.get_rand_qkv(device=device, dtype=dtype)
    q_slice, k_slice, v_slice = get_qkv_slices(queries, keys, values)

    attn_mask = torch.ones(amd.l_x, amd.l_z, device=device, dtype=torch.bool)
    iq_mask = 2
    attn_mask[iq_mask, :] = False
    with pytest.raises(FullyMaskedQueryError):
        single_head_attention(q_slice, k_slice, v_slice, attn_mask=attn_mask)


@pytest.mark.parametrize("device", get_devices())
@pytest.mark.parametrize("dtype", get_dtypes())
def test_against_pytorch_sdpa(device: str, dtype: torch.dtype):
    ensure_supported(device, dtype)
    amd = get_amd()
    queries, keys, values = amd.get_rand_qkv(device=device, dtype=dtype)
    q_slice, k_slice, v_slice = get_qkv_slices(queries, keys, values)
    result = single_head_attention(q_slice, k_slice, v_slice)

    expected = (
        scaled_dot_product_attention(queries, keys, values).squeeze(0).squeeze(0)
    )  # remove batch and head dims

    assert result["output"].shape == expected.shape
    assert torch.allclose(result["output"], expected, atol=1e-5)


@pytest.mark.parametrize("device", get_devices())
@pytest.mark.parametrize("dtype", get_dtypes())
def test_against_pytorch_sdpa_boolean_masked(device: str, dtype: torch.dtype):
    ensure_supported(device, dtype)
    amd = get_amd()
    queries, keys, values = amd.get_rand_qkv(device=device, dtype=dtype)
    q_slice, k_slice, v_slice = get_qkv_slices(queries, keys, values)
    attn_mask = torch.ones(amd.l_x, amd.l_z, device=device, dtype=torch.bool)
    attn_mask[2, 3:] = False
    result = single_head_attention(q_slice, k_slice, v_slice, attn_mask=attn_mask)

    expected = (
        scaled_dot_product_attention(queries, keys, values, attn_mask=attn_mask)
        .squeeze(0)
        .squeeze(0)
    )  # remove batch and head dims

    assert result["output"].shape == expected.shape
    assert torch.allclose(result["output"], expected, atol=1e-5)


@pytest.mark.parametrize("device", get_devices())
@pytest.mark.parametrize("dtype", get_dtypes())
def test_against_pytorch_sdpa_additive_masked(device: str, dtype: torch.dtype):
    ensure_supported(device, dtype)
    amd = get_amd()
    queries, keys, values = amd.get_rand_qkv(device=device, dtype=dtype)
    q_slice, k_slice, v_slice = get_qkv_slices(queries, keys, values)
    attn_mask = torch.zeros(amd.l_x, amd.l_z, device=device, dtype=torch.float32)
    attn_mask[2, 3:] = float("-inf")
    result = single_head_attention(q_slice, k_slice, v_slice, attn_mask=attn_mask)

    expected = (
        scaled_dot_product_attention(queries, keys, values, attn_mask=attn_mask)
        .squeeze(0)
        .squeeze(0)
    )  # remove batch and head dims

    assert result["output"].shape == expected.shape
    assert torch.allclose(result["output"], expected, atol=1e-5)
