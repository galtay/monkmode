import math

import torch

from transformer.exceptions import FullyMaskedQueryError


def single_query_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
):
    """Single query attention.

    Args:
        query: Tensor of shape (d_attn,). A single query.
        keys: Tensor of shape (l_z, d_attn). A sequence of keys.
        values: Tensor of shape (l_z, d_out). A sequence of values.
        attn_mask: Optional tensor of shape (l_z,).
            Boolean or additive attention mask. If boolean, attention is allowed where True.
            If not boolean, attn_mask is added to scores before softmax.
        scale: Optional float.
            Scaling factor for attention. Default value is 1/sqrt(d_attn).

    Returns:
        dict:
            * scores: Tensor of shape (l_z,). Attention scores (pre softmax)
            * attn: Tensor of shape (l_z,). Attention weights (post softmax)
            * output: Tensor of shape (d_out,). Attention-weighted output tensor


    Notation::
        Computes scaled dot-product attention with one query.

        l_z = number of keys (context length)
        m = l_z - 1

        The scores and attn tensors have shape (l_z,) and look like:

            |  k_0  |  k_1  |   ...  |  k_m  |
            ----------------------------------
        q_0 |  a_0  |  a_1  |   ...  |  a_m  |

    """

    assert query.dim() == 1
    d_attn = query.shape[0]

    assert keys.dim() == 2 and keys.shape[1] == d_attn
    l_z = keys.shape[0]

    assert values.dim() == 2 and values.shape[0] == l_z
    d_out = values.shape[1]

    if scale is None:
        scale = 1.0 / math.sqrt(d_attn)
    scale = float(scale)

    scores = query @ keys.T * scale
    assert scores.shape == (l_z,)
    neg_inf = float("-inf")

    if attn_mask is not None:
        assert attn_mask.shape == (l_z,)

        if attn_mask.dtype == torch.bool:
            if (~attn_mask).all():
                raise FullyMaskedQueryError("The query is fully masked.")
            scores = scores.masked_fill(~attn_mask, neg_inf)
        else:
            scores = scores + attn_mask.to(dtype=scores.dtype, device=scores.device)
            if scores.isneginf().all():
                raise FullyMaskedQueryError("The query is fully masked.")

    attn = torch.softmax(scores, dim=-1)
    assert attn.shape == (l_z,)

    output = attn @ values
    assert output.shape == (d_out,)

    return {
        "scores": scores,
        "attn": attn,
        "output": output,
    }


def single_head_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
):
    """Single-head attention.

    Args:
        queries: Tensor of shape (l_x, d_attn). A sequence of queries.
        keys: Tensor of shape (l_z, d_attn). A sequence of keys.
        values: Tensor of shape (l_z, d_out). A sequence of values.
        attn_mask: Optional tensor of shape (l_x, l_z).
            Boolean or additive attention mask. If boolean, attention is allowed where True.
            If not boolean, attn_mask is added to scores before softmax.
        scale: Optional float.
            Scaling factor for attention. Default value is 1/sqrt(d_attn).

    Returns:
        dict:
            * scores: Tensor of shape (l_x, l_z). Attention scores (pre softmax)
            * attn: Tensor of shape (l_x, l_z). Attention weights (post softmax)
            * output: Tensor of shape (l_x, d_out). Attention-weighted output tensor

    Notation:
        l_x = number of queries (primary length)
        l_z = number of keys (context length)
        n = l_x - 1
        m = l_z - 1

        The scores and attn tensors have shape (l_x, l_z) and looks like:

            |  k_0  |  k_1  |   ...  | k_m   |
            ----------------------------------
        q_0 | a_0,0 | a_0,1 |   ...  | a_0,m |
        q_1 | a_1,0 | a_1,1 |   ...  | a_1,m |
        ... | ...   | ...   |   ...  | ...   |
        q_n | a_n,0 | a_n,1 |   ...  | a_n,m |

        where a can be the score or attention element. After softmax, each row sums to 1:
            a_i,0 + a_i,1 + ... + a_i,m = 1

        the tensor is not symmetric and a_i,j represents the attention of query i to key j.

    """

    assert queries.dim() == 2
    l_x, d_attn = queries.shape

    assert keys.dim() == 2 and keys.shape[1] == d_attn
    l_z = keys.shape[0]

    assert values.dim() == 2 and values.shape[0] == l_z
    d_out = values.shape[1]

    if scale is None:
        scale = 1.0 / math.sqrt(d_attn)
    scale = float(scale)

    scores = queries @ keys.T * scale
    assert scores.shape == (l_x, l_z)
    neg_inf = float("-inf")

    if attn_mask is not None:
        assert attn_mask.shape == (l_x, l_z)

        if attn_mask.dtype == torch.bool:
            if (~attn_mask).all(dim=-1).any():
                raise FullyMaskedQueryError("Some queries are fully masked.")
            scores = scores.masked_fill(~attn_mask, neg_inf)
        else:
            scores = scores + attn_mask.to(dtype=scores.dtype, device=scores.device)
            if scores.isneginf().all(dim=-1).any():
                raise FullyMaskedQueryError("Some queries are fully masked.")

    attn = torch.softmax(scores, dim=-1)
    assert attn.shape == (l_x, l_z)

    output = attn @ values
    assert output.shape == (l_x, d_out)

    return {
        "scores": scores,
        "attn": attn,
        "output": output,
    }


def multi_head_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
):
    """Multi-head attention.

    Args:
        queries: Tensor of shape (n_head, l_x, d_attn). A sequence of queries.
        keys: Tensor of shape (n_head, l_z, d_attn). A sequence of keys.
        values: Tensor of shape (n_head, l_z, d_out). A sequence of values.
        attn_mask: Optional tensor broadcastable to shape (n_head, l_x, l_z).
            Boolean or additive attention mask. If boolean, attention is allowed where True.
            If not boolean, attn_mask is added to scores before softmax.
        scale: Optional float.
            Scaling factor for attention. Default value is 1/sqrt(d_attn).

    Returns:
        dict:
            * scores: Tensor of shape (n_head, l_x, l_z). Attention scores (pre softmax)
            * attn: Tensor of shape (n_head, l_x, l_z). Attention weights (post softmax)
            * output: Tensor of shape (n_head, l_x, d_out). Attention-weighted output tensor

    """
    assert queries.dim() == 3
    n_head, l_x, d_attn = queries.shape

    assert keys.dim() == 3 and keys.shape[0] == n_head and keys.shape[2] == d_attn
    l_z = keys.shape[1]

    assert values.dim() == 3 and values.shape[0] == n_head and values.shape[1] == l_z
    d_out = values.shape[2]

    # einsum 2-d matrix multiplication is ik,kj -> ij
    # transposed matrix multiplication is ik,jk -> ij

    if scale is None:
        scale = 1.0 / math.sqrt(d_attn)
    scale = float(scale)

    scores = torch.einsum("nqd, nkd -> nqk", queries, keys) * scale
    assert scores.shape == (n_head, l_x, l_z)
    neg_inf = float("-inf")

    if attn_mask is not None:
        assert attn_mask.shape == (n_head, l_x, l_z)

        if attn_mask.dtype == torch.bool:
            if (~attn_mask).all(dim=-1).any():
                raise FullyMaskedQueryError("Some queries are fully masked.")
            scores = scores.masked_fill(~attn_mask, neg_inf)
        else:
            scores = scores + attn_mask.to(dtype=scores.dtype, device=scores.device)
            if scores.isneginf().all(dim=-1).any():
                raise FullyMaskedQueryError("Some queries are fully masked.")

    attn = torch.softmax(scores, dim=-1)
    assert attn.shape == (n_head, l_x, l_z)

    output = torch.einsum("nqk, nkd -> nqd", attn, values)
    assert output.shape == (n_head, l_x, d_out)

    return {
        "scores": scores,
        "attn": attn,
        "output": output,
    }


def batch_multi_head_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor = None,
):
    """Multi-head attention.

    Args:
        queries: (bsz, n_head, l_x, d_attn)
        keys:    (bsz, n_head, l_z, d_attn)
        values:  (bsz, n_head, l_z, d_out)
        mask:    (bsz, n_head, l_x, l_z), optional additive mask

    Returns:
        dict:
            * scores: (bsz, n_head, l_x, l_z)
            * attn:   (bsz, n_head, l_x, l_z)
            * output: (bsz, n_head, l_x, d_out)

    """
    assert queries.dim() == 4
    bsz, n_head, l_x, d_attn = queries.shape

    assert keys.dim() == 4
    assert keys.shape[0] == bsz
    assert keys.shape[1] == n_head
    l_z = keys.shape[2]
    assert keys.shape[3] == d_attn

    assert values.dim() == 4
    assert values.shape[0] == bsz
    assert values.shape[1] == n_head
    assert values.shape[2] == l_z
    d_out = values.shape[3]

    # einsum 2-d matrix multiplication is ik,kj -> ij
    # transposed matrix multiplication is ik,jk -> ij

    scores = torch.einsum("bnqd,bnkd->bnqk", queries, keys) / math.sqrt(d_attn)
    assert scores.shape == (bsz, n_head, l_x, l_z)

    if mask is not None:
        assert mask.shape == (bsz, n_head, l_x, l_z)
        fully_masked = torch.isneginf(mask).all(dim=-1).any(dim=-1)
        if fully_masked.any():
            raise FullyMaskedQueryError("Some queries are fully masked.")
        scores += mask

    attn = torch.softmax(scores, dim=-1)
    assert attn.shape == (bsz, n_head, l_x, l_z)

    output = torch.einsum("bnqk,bnkd->bnqd", attn, values)
    assert output.shape == (bsz, n_head, l_x, d_out)

    return {
        "scores": scores,
        "attn": attn,
        "output": output,
    }
