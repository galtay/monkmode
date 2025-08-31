import math

import torch

from transformer.exceptions import FullyMaskedQueryError


def single_query_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor = None,
):
    """Single query attention.

    Args:
        query (d_attn): a single query vector.
        keys (l_z, d_attn): a sequence of key vectors.
        values (l_z, d_out): a sequence of value vectors.
        mask (l_z,): optional:
            Additive attention mask. 0 for valid positions, -inf for masked.
            Applied to scores before softmax.

    Returns:
        dict:
            * scores (l_z,): attention scores (pre softmax)
            * attn (l_z,): attention weights (post softmax)
            * output (d_out,): attention-weighted output vector


    Description:
        Computes scaled dot-product attention for a single query vector.

            scores = (query @ keys.T) / sqrt(d_attn)
            if mask is provided:
                scores += mask
            attn = softmax(scores, dim=-1)
            output = attn @ values

        Softmax is applied over the key positions (l_z).
        The output is a weighted sum of the value vectors.

    Notation:
        l_z = number of keys (context length)
        m = l_z - 1

        The scores and attn tensors have shape (l_z,) and looks like:

            |  k_0  |  k_1  |   ...  |  k_m  |
            ----------------------------------
        q_0 |  a_0  |  a_1  |   ...  |  a_m  |

    """

    assert query.dim() == 1
    d_attn = query.shape[0]

    assert keys.dim() == 2
    l_z = keys.shape[0]
    assert keys.shape[1] == d_attn

    assert values.dim() == 2
    assert values.shape[0] == l_z
    d_out = values.shape[1]

    scores = query @ keys.T / math.sqrt(d_attn)
    assert scores.shape == (l_z,)

    if mask is not None:
        assert mask.shape == (l_z,)
        if torch.isneginf(mask).all():
            raise FullyMaskedQueryError("Some queries are fully masked.")
        scores = scores + mask

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
    mask: torch.Tensor = None,
):
    """Single-head attention.

    Args:
        queries (l_x, d_attn): a sequence of query vectors.
        keys (l_z, d_attn): a sequence of key vectors.
        values (l_z, d_out): a sequence of value vectors.
        mask (l_x, l_z): optional:
            Additive attention mask. 0 for valid positions, -inf for masked.
            Applied to scores before softmax.

    Returns:
        dict:
            * scores (l_x, l_z): attention scores (pre softmax)
            * attn (l_x, l_z): attention tensor (post softmax)
            * output (l_x, d_out): attention-weighted outputs

    Description:
        Computes scaled dot-product attention.

            scores = (queries @ keys.T) / sqrt(d_attn)
            if mask is provided:
                scores += mask
            attn = softmax(scores, dim=-1)
            output = attn @ values

        Softmax is applied row-wise over the context dimension (keys).
        Each query attends to all keys in parallel.

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

    assert keys.dim() == 2
    l_z = keys.shape[0]
    assert keys.shape[1] == d_attn

    assert values.dim() == 2
    assert values.shape[0] == l_z
    d_out = values.shape[1]

    scores = queries @ keys.T / math.sqrt(d_attn)
    assert scores.shape == (l_x, l_z)

    if mask is not None:
        assert mask.shape == (l_x, l_z)
        fully_masked = torch.isneginf(mask).all(dim=-1).any(dim=-1)
        if fully_masked.any():
            raise FullyMaskedQueryError("Some queries are fully masked.")
        scores = scores + mask

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
    mask: torch.Tensor = None,
):
    """Multi-head attention.

    Args:
        queries: (n_head, l_x, d_attn)
        keys:    (n_head, l_z, d_attn)
        values:  (n_head, l_z, d_out)
        mask:    (n_head, l_x, l_z), optional additive mask

    Returns:
        dict:
            * scores: (n_head, l_x, l_z)
            * attn:   (n_head, l_x, l_z)
            * output: (n_head, l_x, d_out)

    """
    assert queries.dim() == 3
    n_head, l_x, d_attn = queries.shape

    assert keys.dim() == 3
    assert keys.shape[0] == n_head
    l_z = keys.shape[1]
    assert keys.shape[2] == d_attn

    assert values.dim() == 3
    assert values.shape[0] == n_head
    assert values.shape[1] == l_z
    d_out = values.shape[2]

    # einsum 2-d matrix multiplication is ik,kj -> ij
    # transposed matrix multiplication is ik,jk -> ij

    scores = torch.einsum("nqd,nkd->nqk", queries, keys) / math.sqrt(d_attn)
    assert scores.shape == (n_head, l_x, l_z)

    if mask is not None:
        assert mask.shape == (n_head, l_x, l_z)
        fully_masked = torch.isneginf(mask).all(dim=-1).any(dim=-1)
        if fully_masked.any():
            raise FullyMaskedQueryError("Some queries are fully masked.")
        scores += mask

    attn = torch.softmax(scores, dim=-1)
    assert attn.shape == (n_head, l_x, l_z)

    output = torch.einsum("nqk,nkd->nqd", attn, values)
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
