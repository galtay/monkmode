import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

from transformer.attn_layers import batch_multi_head_attention
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
        batch_size=3,
        n_head=4,
    )


@pytest.mark.parametrize("device", get_devices())
@pytest.mark.parametrize("dtype", get_dtypes())
def test_attn_sums_to_one(device: str, dtype: torch.dtype):
    ensure_supported(device, dtype)
    amd = get_amd()
    queries, keys, values = amd.get_rand_qkv(device=device, dtype=dtype)
    result = batch_multi_head_attention(queries, keys, values)

    assert result["attn"].device.type == device
    assert result["attn"].dtype == dtype
    assert result["attn"].shape == (amd.batch_size, amd.n_head, amd.l_x, amd.l_z)
    assert torch.isfinite(result["attn"]).all()

    row_sums = result["attn"].sum(dim=-1).to(torch.float32)
    expected = torch.ones(
        amd.batch_size, amd.n_head, amd.l_x, device=device, dtype=torch.float32
    )
    assert row_sums.shape == expected.shape
    assert torch.allclose(row_sums, expected, atol=1e-5)


@pytest.mark.parametrize("device", get_devices())
@pytest.mark.parametrize("dtype", get_dtypes())
def test_uniform_scores_give_uniform_attention(device: str, dtype: torch.dtype):
    ensure_supported(device, dtype)
    amd = get_amd()
    queries, keys, values = amd.get_repeated_qk(device=device, dtype=dtype)
    result = batch_multi_head_attention(queries, keys, values)
    assert result["attn"].device.type == device
    assert result["attn"].dtype == dtype
    assert result["attn"].shape == (amd.batch_size, amd.n_head, amd.l_x, amd.l_z)
    assert torch.isfinite(result["attn"]).all()

    expected = torch.full(
        (amd.batch_size, amd.n_head, amd.l_x, amd.l_z),
        1.0 / amd.l_z,
        device=device,
        dtype=dtype,
    )
    assert result["attn"].shape == expected.shape
    assert torch.allclose(result["attn"], expected, atol=1e-5)


@pytest.mark.parametrize("device", get_devices())
@pytest.mark.parametrize("dtype", get_dtypes())
def test_additive_masked_position_has_zero_attention(device: str, dtype: torch.dtype):
    ensure_supported(device, dtype)
    amd = get_amd()
    queries, keys, values = amd.get_rand_qkv(device=device, dtype=dtype)
    attn_mask = torch.zeros(
        amd.batch_size, amd.n_head, amd.l_x, amd.l_z, device=device, dtype=dtype
    )
    # prevent attention from query iq_mask to key ik_mask
    ib_mask = 1
    ih_mask = 3
    iq_mask = 0
    ik_mask = 2
    attn_mask[ib_mask, ih_mask, iq_mask, ik_mask] = float("-inf")

    result = batch_multi_head_attention(queries, keys, values, attn_mask=attn_mask)
    assert result["attn"].device.type == device
    assert result["attn"].dtype == dtype
    assert result["attn"].shape == (amd.batch_size, amd.n_head, amd.l_x, amd.l_z)
    assert torch.isfinite(result["attn"]).all()

    assert torch.allclose(
        result["attn"][ib_mask, ih_mask, iq_mask, ik_mask],
        torch.tensor(0.0, device=device, dtype=dtype),
        atol=1e-6,
    )


@pytest.mark.parametrize("device", get_devices())
@pytest.mark.parametrize("dtype", get_dtypes())
def test_boolean_masked_position_has_zero_attention(device: str, dtype: torch.dtype):
    ensure_supported(device, dtype)
    amd = get_amd()
    queries, keys, values = amd.get_rand_qkv(device=device, dtype=dtype)
    attn_mask = torch.ones(
        amd.batch_size, amd.n_head, amd.l_x, amd.l_z, device=device, dtype=torch.bool
    )
    # prevent attention from query iq_mask to key ik_mask
    ib_mask = 1
    ih_mask = 3
    iq_mask = 0
    ik_mask = 2
    attn_mask[ib_mask, ih_mask, iq_mask, ik_mask] = False

    result = batch_multi_head_attention(queries, keys, values, attn_mask=attn_mask)
    assert result["attn"].device.type == device
    assert result["attn"].dtype == dtype
    assert result["attn"].shape == (amd.batch_size, amd.n_head, amd.l_x, amd.l_z)
    assert torch.isfinite(result["attn"]).all()

    assert torch.allclose(
        result["attn"][ib_mask, ih_mask, iq_mask, ik_mask],
        torch.tensor(0.0, device=device, dtype=dtype),
        atol=1e-6,
    )


@pytest.mark.parametrize("device", get_devices())
@pytest.mark.parametrize("dtype", get_dtypes())
def test_all_positions_additive_masked_raises_error(device: str, dtype: torch.dtype):
    ensure_supported(device, dtype)
    amd = get_amd()
    queries, keys, values = amd.get_rand_qkv(device=device, dtype=dtype)

    attn_mask = torch.zeros(
        amd.batch_size, amd.n_head, amd.l_x, amd.l_z, device=device, dtype=dtype
    )
    ib_mask = 1
    iq_mask = 2
    attn_mask[ib_mask, :, iq_mask, :] = float("-inf")
    with pytest.raises(FullyMaskedQueryError):
        batch_multi_head_attention(queries, keys, values, attn_mask=attn_mask)


@pytest.mark.parametrize("device", get_devices())
@pytest.mark.parametrize("dtype", get_dtypes())
def test_all_positions_boolean_masked_raises_error(device: str, dtype: torch.dtype):
    ensure_supported(device, dtype)
    amd = get_amd()
    queries, keys, values = amd.get_rand_qkv(device=device, dtype=dtype)

    attn_mask = torch.ones(
        amd.batch_size, amd.n_head, amd.l_x, amd.l_z, device=device, dtype=torch.bool
    )
    ib_mask = 1
    iq_mask = 2
    attn_mask[ib_mask, :, iq_mask, :] = False
    with pytest.raises(FullyMaskedQueryError):
        batch_multi_head_attention(queries, keys, values, attn_mask=attn_mask)


@pytest.mark.parametrize("device", get_devices())
@pytest.mark.parametrize("dtype", get_dtypes())
def test_against_pytorch_sdpa(device: str, dtype: torch.dtype):
    ensure_supported(device, dtype)
    amd = get_amd()
    queries, keys, values = amd.get_rand_qkv(device=device, dtype=dtype)
    result = batch_multi_head_attention(queries, keys, values)
    expected = scaled_dot_product_attention(queries, keys, values)

    assert result["output"].shape == expected.shape
    assert torch.allclose(result["output"], expected, atol=1e-5)


@pytest.mark.parametrize("device", get_devices())
@pytest.mark.parametrize("dtype", get_dtypes())
def test_against_pytorch_sdpa_boolean_masked(device: str, dtype: torch.dtype):
    ensure_supported(device, dtype)
    amd = get_amd()
    queries, keys, values = amd.get_rand_qkv(device=device, dtype=dtype)
    attn_mask = torch.ones(
        amd.batch_size, amd.n_head, amd.l_x, amd.l_z, device=device, dtype=torch.bool
    )
    attn_mask[1, :, 2, 3:] = False
    result = batch_multi_head_attention(queries, keys, values, attn_mask=attn_mask)
    expected = scaled_dot_product_attention(queries, keys, values, attn_mask=attn_mask)

    assert result["output"].shape == expected.shape
    assert torch.allclose(result["output"], expected, atol=1e-5)


@pytest.mark.parametrize("device", get_devices())
@pytest.mark.parametrize("dtype", get_dtypes())
def test_against_pytorch_sdpa_additive_masked(device: str, dtype: torch.dtype):
    ensure_supported(device, dtype)
    amd = get_amd()
    queries, keys, values = amd.get_rand_qkv(device=device, dtype=dtype)
    attn_mask = torch.zeros(
        amd.batch_size, amd.n_head, amd.l_x, amd.l_z, device=device, dtype=torch.float32
    )
    attn_mask[1, :, 2, 3:] = float("-inf")
    result = batch_multi_head_attention(queries, keys, values, attn_mask=attn_mask)
    expected = scaled_dot_product_attention(queries, keys, values, attn_mask=attn_mask)

    assert result["output"].shape == expected.shape
    assert torch.allclose(result["output"], expected, atol=1e-5)


@pytest.mark.parametrize("device", get_devices())
@pytest.mark.parametrize("dtype", get_dtypes())
def test_against_pytorch_spda_causal_boolean_mask(device: str, dtype: torch.dtype):
    ensure_supported(device, dtype)
    amd = get_amd()
    queries, keys, values = amd.get_rand_qkv(device=device, dtype=dtype)

    attn_mask = torch.ones(amd.l_x, amd.l_z, device=device, dtype=torch.bool).tril()
    attn_mask = attn_mask.expand(amd.batch_size, amd.n_head, -1, -1)
    result = batch_multi_head_attention(queries, keys, values, attn_mask=attn_mask)

    expected = scaled_dot_product_attention(
        queries, keys, values, attn_mask=attn_mask, is_causal=False
    )
    assert expected.shape == result["output"].shape
    assert torch.allclose(result["output"], expected, atol=1e-5)

    expected = scaled_dot_product_attention(queries, keys, values, is_causal=True)
    assert expected.shape == result["output"].shape
    assert torch.allclose(result["output"], expected, atol=1e-5)


@pytest.mark.parametrize("device", get_devices())
@pytest.mark.parametrize("dtype", get_dtypes())
def test_against_pytorch_spda_causal_additive_mask(device: str, dtype: torch.dtype):
    ensure_supported(device, dtype)
    amd = get_amd()
    queries, keys, values = amd.get_rand_qkv(device=device, dtype=dtype)

    attn_mask = torch.ones(amd.l_x, amd.l_z, device=device, dtype=torch.bool).tril()
    attn_mask = attn_mask.expand(amd.batch_size, amd.n_head, -1, -1)
    additive_mask = torch.zeros_like(attn_mask, dtype=torch.float32)
    additive_mask.masked_fill_(~attn_mask, float("-inf"))

    result = batch_multi_head_attention(queries, keys, values, attn_mask=attn_mask)

    expected = scaled_dot_product_attention(
        queries, keys, values, attn_mask=attn_mask, is_causal=False
    )
    assert expected.shape == result["output"].shape
    assert torch.allclose(result["output"], expected, atol=1e-5)

    expected = scaled_dot_product_attention(queries, keys, values, is_causal=True)
    assert expected.shape == result["output"].shape
    assert torch.allclose(result["output"], expected, atol=1e-5)
