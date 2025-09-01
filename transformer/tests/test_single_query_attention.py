import pytest
import torch

from transformer.attn_layers import single_query_attention
from transformer.exceptions import FullyMaskedQueryError
from tests.utils import AttentionMockData


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICES.append("mps")

DTYPES = [torch.bfloat16, torch.float16, torch.float32]


def _ensure_supported(device: str, dtype: torch.dtype):
    """
    Try a tiny softmax on (device, dtype). Skip if unsupported.
    This covers kernels and dtype/device support differences across backends.
    """
    try:
        x = torch.zeros(4, dtype=dtype, device=device)
        torch.softmax(x, dim=-1)
    except (RuntimeError, TypeError, ValueError) as e:
        pytest.skip(f"{dtype} not supported on {device}: {e}")


def _get_qkv_slices(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    device: str,
    dtype: torch.dtype,
):
    ib, ih, iq = 0, 0, 0  # Batch, Head, Query
    return (
        queries[ib, ih, iq, :].to(device=device, dtype=dtype),
        keys[ib, ih, :, :].to(device=device, dtype=dtype),
        values[ib, ih, :, :].to(device=device, dtype=dtype),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_attn_sums_to_one(device: str, dtype: torch.dtype):
    _ensure_supported(device, dtype)
    amd = AttentionMockData()
    queries, keys, values = amd.get_rand_qkv()
    q_slice, k_slice, v_slice = _get_qkv_slices(queries, keys, values, device, dtype)
    result = single_query_attention(q_slice, k_slice, v_slice)
    assert result["attn"].device.type == device
    assert result["attn"].dtype == dtype
    assert result["attn"].shape == (amd.l_z,)
    assert torch.isfinite(result["attn"]).all()

    attn_sum = result["attn"].sum().to(torch.float32)
    assert attn_sum.shape == torch.Size([])
    assert torch.isclose(attn_sum, torch.tensor(1.0), atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_uniform_scores_gives_uniform_attention(device: str, dtype: torch.dtype):
    _ensure_supported(device, dtype)
    amd = AttentionMockData()
    queries, keys, values = amd.get_repeated_qk()
    q_slice, k_slice, v_slice = _get_qkv_slices(queries, keys, values, device, dtype)
    result = single_query_attention(q_slice, k_slice, v_slice)
    assert result["attn"].device.type == device
    assert result["attn"].dtype == dtype
    assert result["attn"].shape == (amd.l_z,)
    assert torch.isfinite(result["attn"]).all()

    expected = torch.full((amd.l_z,), 1.0 / amd.l_z).to(device=device, dtype=dtype)
    assert result["attn"].shape == expected.shape
    assert torch.allclose(result["attn"], expected, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_additive_masked_position_has_zero_attention(device: str, dtype: torch.dtype):
    _ensure_supported(device, dtype)
    amd = AttentionMockData()
    queries, keys, values = amd.get_rand_qkv()
    q_slice, k_slice, v_slice = _get_qkv_slices(queries, keys, values, device, dtype)
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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_boolean_masked_position_has_zero_attention(device: str, dtype: torch.dtype):
    amd = AttentionMockData()
    queries, keys, values = amd.get_rand_qkv()
    q_slice, k_slice, v_slice = _get_qkv_slices(queries, keys, values, device, dtype)

    attn_mask = torch.ones(amd.l_z, device=device, dtype=torch.bool)
    ik_mask = 2  # prevent attention at position ik_mask
    attn_mask[ik_mask] = False

    result = single_query_attention(q_slice, k_slice, v_slice, attn_mask=attn_mask)
    assert torch.isclose(
        result["attn"][ik_mask],
        torch.tensor(0.0, device=device, dtype=dtype),
        atol=1e-6,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_all_positions_boolean_masked_raises_error(device: str, dtype: torch.dtype):
    amd = AttentionMockData()
    queries, keys, values = amd.get_rand_qkv()
    q_slice, k_slice, v_slice = _get_qkv_slices(queries, keys, values, device, dtype)

    attn_mask = torch.zeros(amd.l_z, device=device, dtype=torch.bool)
    with pytest.raises(FullyMaskedQueryError):
        single_query_attention(q_slice, k_slice, v_slice, attn_mask=attn_mask)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_all_positions_additive_masked_raises_error(device: str, dtype: torch.dtype):
    amd = AttentionMockData()
    queries, keys, values = amd.get_rand_qkv()
    q_slice, k_slice, v_slice = _get_qkv_slices(queries, keys, values, device, dtype)

    attn_mask = torch.ones(amd.l_z) * float("-inf")
    with pytest.raises(FullyMaskedQueryError):
        single_query_attention(q_slice, k_slice, v_slice, attn_mask=attn_mask)
