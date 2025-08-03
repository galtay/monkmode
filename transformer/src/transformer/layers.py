import math
import warnings

import pydantic
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
            attn = softmax(scores, dim=0)
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
            raise FullyMaskedQueryError("Mask fully excludes all positions.")
        scores = scores + mask

    attn = torch.softmax(scores, dim=0)
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
    """Single head attention.

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
            attn = softmax(scores, dim=1)
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
        fully_masked_rows = torch.isneginf(mask).all(dim=1)
        if fully_masked_rows.any():
            raise FullyMaskedQueryError(
                f"{fully_masked_rows.sum().item()} queries are fully masked."
            )
        scores = scores + mask

    attn = torch.softmax(scores, dim=1)
    assert attn.shape == (l_x, l_z)

    output = attn @ values
    assert output.shape == (l_x, d_out)

    return {
        "scores": scores,
        "attn": attn,
        "output": output,
    }
