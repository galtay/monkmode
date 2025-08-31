import torch

def test_matmul():
    x = torch.randn(3, 4)
    y = torch.randn(4, 5)
    z_exp = torch.matmul(x, y)
    z_act = torch.einsum("ij,jk->ik", x, y)
    assert torch.allclose(z_exp, z_act, atol=1e-5)

def test_transposed_matmul():
    x = torch.randn(3, 4)
    y = torch.randn(5, 4)
    z_exp = torch.matmul(x, y.T)
    z_act = torch.einsum("ij,kj->ik", x, y)
    assert torch.allclose(z_exp, z_act, atol=1e-5)

def test_batch_matmul_1():
    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 4, 5)
    z_exp = torch.matmul(x, y)
    z_act = torch.einsum("bij,bjk->bik", x, y)
    assert torch.allclose(z_exp, z_act, atol=1e-5)

def test_batch_matmul_2():
    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 4, 5)
    z_exp = torch.bmm(x, y)
    z_act = torch.einsum("bij,bjk->bik", x, y)
    assert torch.allclose(z_exp, z_act, atol=1e-5)