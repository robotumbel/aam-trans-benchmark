"""test_v4.py: unit checks for the v4 candidates (entmax gradients, shapes,
input gradients). Run before any training: python test_v4.py"""
import torch
import torch.nn.functional as F

from models import build, entmax, n_params

torch.manual_seed(0)

# alpha-entmax: simplex output, sparsemax limit, numerical gradient check.
X = torch.randn(3, 2, 5, 7, dtype=torch.double, requires_grad=True)
a = torch.full((3, 2, 5, 1), 1.5, dtype=torch.double, requires_grad=True)
P = entmax(X, a)
assert torch.allclose(P.sum(-1), torch.ones_like(P.sum(-1)))
print("entmax zeros share at alpha 1.5:", float((P == 0).double().mean()))
ok = torch.autograd.gradcheck(lambda x, al: entmax(x, al), (X, a), eps=1e-6, atol=1e-4)
print("entmax gradcheck (input and alpha):", ok)
p2 = entmax(X.detach(), torch.full_like(a.detach(), 1.999))
z = X.detach()
print("alpha~2 vs sparsemax-like sparsity:", float((p2 == 0).double().mean()))
p1 = entmax(X.detach(), torch.full_like(a.detach(), 1.001))
print("alpha~1 max abs diff to softmax:", float((p1 - F.softmax(z, -1)).abs().max()))

# Every model: forward, aux, input gradient, parameter count.
d, C = 39, 2
x = torch.randn(16, d)
for name in ["transformer", "ft", "aam_noGate", "aam_trans", "aam_v4a",
             "aam_v4b", "aam_v4c", "aam_v4d", "aam_v4a+d", "aam_v4b+c"]:
    m = build(name, d, C)
    m.train()
    lg, rec = m(x, return_aux=True)
    m.eval()
    xi = x.clone().requires_grad_(True)
    g, = torch.autograd.grad(F.cross_entropy(m(xi), torch.zeros(16, dtype=torch.long)), xi)
    print(f"{name:12s} params {n_params(m):7d} logits {tuple(lg.shape)} "
          f"aux {None if rec is None else tuple(rec.shape)} "
          f"grad-norm {g.norm():.3e} finite {bool(torch.isfinite(g).all())}")
