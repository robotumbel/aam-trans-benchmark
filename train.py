"""
train.py: standard training, PGD-AT (Madry et al., 2018) and PGD-TRADES
(Zhang et al., 2019), with the optional AAM-TRANS reconstruction term.

Training runs for a fixed number of epochs and keeps the final weights; the
validation split is only logged, never used for selection, so the test split
is untouched until evaluation.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from attacks import _clip_ball


def _kl_pgd(model, x, eps, steps):
    """TRADES inner maximisation: PGD on KL(p(x) || p(x + delta))."""
    alpha = 2.5 * eps / steps
    with torch.no_grad():
        lp = F.log_softmax(model(x), 1)
    xa = x + 0.001 * torch.randn_like(x)
    for _ in range(steps):
        xa.requires_grad_(True)
        l = F.kl_div(F.log_softmax(model(xa), 1), lp, reduction="sum",
                     log_target=True)
        g, = torch.autograd.grad(l, xa)
        xa = _clip_ball(xa.detach() + alpha * g.sign(), x, eps, None)
    return xa.detach()


def _ce_pgd(model, x, y, eps, steps):
    alpha = 2.5 * eps / steps
    xa = x + torch.empty_like(x).uniform_(-eps, eps)
    for _ in range(steps):
        xa.requires_grad_(True)
        g, = torch.autograd.grad(F.cross_entropy(model(xa), y), xa)
        xa = _clip_ball(xa.detach() + alpha * g.sign(), x, eps, None)
    return xa.detach()


def loader(X, y, bs, shuffle, seed=0):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, generator=g,
                      drop_last=False)


@torch.no_grad()
def predict(model, X, device, bs=2048):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(torch.from_numpy(X[i:i + bs]).to(device)).float().cpu())
    return torch.cat(out)


def _gate_reg(model, reg, g_clean, g_adv):
    """Gate v2 loss terms (PROTOCOL_GATE_V2.md)."""
    kind, w = reg
    if kind == "sparsity":
        return w * torch.stack([g.mean() for g in g_clean]).mean()
    if kind == "consistency":
        return w * torch.stack([F.mse_loss(a, c) for c, a in
                                zip(g_clean, g_adv)]).mean()
    raise ValueError(kind)


def train(model, s, device, method="trades", epochs=20, eps=0.1, steps=7,
          beta=6.0, lam_aux=0.1, bs=512, lr=1e-3, seed=0, log=print,
          gate_reg=None, class_weight=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dl = loader(s.Xtr, s.ytr, bs, True, seed)
    # Optional class-balanced cross-entropy: weights proportional to
    # 1/sqrt(class frequency), normalised to mean 1.
    cw = None
    if class_weight:
        cnt = np.bincount(s.ytr, minlength=int(s.ytr.max()) + 1).astype(float)
        w = 1.0 / np.sqrt(np.maximum(cnt, 1))
        cw = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    hist = []
    for ep in range(epochs):
        t0, tot, nb = time.time(), 0.0, 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            if method == "trades":
                model.eval(); xa = _kl_pgd(model, x, eps, steps); model.train()
                logits, rec = model(x, return_aux=True)
                g_clean = model.live_gates() if gate_reg else None
                logits_adv = model(xa)
                g_adv = model.live_gates() if gate_reg else None
                # log_target avoids 0 * log 0 = NaN when softmax underflows.
                loss = F.cross_entropy(logits, y, weight=cw) + beta * F.kl_div(
                    F.log_softmax(logits_adv, 1), F.log_softmax(logits, 1),
                    reduction="batchmean", log_target=True)
                if gate_reg:
                    loss = loss + _gate_reg(model, gate_reg, g_clean, g_adv)
            elif method == "pgdat":
                model.eval(); xa = _ce_pgd(model, x, y, eps, steps); model.train()
                logits, rec = model(xa, return_aux=True)
                loss = F.cross_entropy(logits, y, weight=cw)
            else:                                       # standard
                model.train()
                logits, rec = model(x, return_aux=True)
                loss = F.cross_entropy(logits, y, weight=cw)
            if rec is not None:
                src = xa if method == "pgdat" else x
                loss = loss + lam_aux * F.mse_loss(rec, src)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {ep+1}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        va = predict(model, s.Xva, device).argmax(1).numpy()
        h = dict(epoch=ep + 1, loss=tot / nb,
                 val_acc=float((va == s.yva).mean()), sec=time.time() - t0)
        hist.append(h)
        log(f"    ep {ep+1:2d} loss {h['loss']:.4f} val_acc {h['val_acc']:.4f} "
            f"({h['sec']:.0f}s)")
    model.eval()
    return hist
