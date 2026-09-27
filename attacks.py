"""
attacks.py: L_inf evasion attacks in standardised feature space.

White-box:  FGSM, PGD (cross-entropy), PGD (CW margin), and a per-sample
            worst case over the two PGD losses.
Black-box:  Square (score-based, query-only) and transfer (PGD run on an
            independently trained surrogate, replayed on the target).
Control:    Gaussian noise of matched magnitude.
Constrained: any of the above with a Constraint projector that keeps each
            feature inside the training range and integer-valued features
            integral in raw units. This is a constraint-respecting
            feature-space threat model, not a problem-space (packet-level)
            one: feature inter-dependencies are not modelled.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class Constraint:
    """Keeps a standardised vector inside the training range of every
    feature and integer-valued features integral in raw units (model space
    is log1p(raw) for the features in log_mask)."""

    def __init__(self, lo, hi, is_int, mean, scale, device, log_mask=None):
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
        self.lo, self.hi = t(lo), t(hi)
        self.mean, self.scale = t(mean), t(scale)
        self.is_int = t(is_int, torch.bool)
        self.log = (t(log_mask, torch.bool) if log_mask is not None
                    else torch.zeros_like(self.is_int))

    def to_raw(self, x):
        m = x.double() * self.scale.double() + self.mean.double()
        return torch.where(self.log, torch.expm1(m), m)

    def from_raw(self, r):
        m = torch.where(self.log, torch.log1p(r.clamp_min(0)), r)
        return ((m - self.mean.double()) / self.scale.double()).float()

    def _bounds(self, x0):
        """Training range, widened per sample to include the clean value
        (a clean test record is valid by definition)."""
        if x0 is None:
            return self.lo, self.hi
        return torch.min(self.lo, x0), torch.max(self.hi, x0)

    def project(self, x, x0=None):
        lo, hi = self._bounds(x0)
        x = torch.max(torch.min(x, hi), lo)
        if self.is_int.any():
            raw = self.to_raw(x)
            raw = torch.where(self.is_int, torch.round(raw), raw)
            x = torch.max(torch.min(self.from_raw(raw), hi), lo)
        return x

    @torch.no_grad()
    def valid(self, x, x0=None, tol=1e-4):
        """Per-sample validity: every feature inside the (per-sample) range
        and integer features integral. The integrality tolerance is relative
        (1e-3 + 1e-5 |raw|) because float32 model space cannot represent
        large integer counts exactly."""
        lo, hi = self._bounds(x0)
        ok = (x >= lo - tol) & (x <= hi + tol)
        raw = self.to_raw(x)
        ok &= ~self.is_int | ((raw - torch.round(raw)).abs()
                              <= 1e-3 + 1e-5 * raw.abs())
        return ok.all(dim=1)


def _margin(logits, y):
    """CW margin: true-class logit minus best other logit (<0 = fooled)."""
    true = logits.gather(1, y[:, None]).squeeze(1)
    other = logits.clone()
    other.scatter_(1, y[:, None], -1e9)
    return true - other.max(1).values


def _clip_ball(xa, x, eps, cons):
    xa = torch.max(torch.min(xa, x + eps), x - eps)
    return cons.project(xa, x) if cons is not None else xa


def fgsm(model, x, y, eps, cons=None):
    x_ = x.clone().requires_grad_(True)
    g, = torch.autograd.grad(F.cross_entropy(model(x_), y), x_)
    return _clip_ball(x + eps * g.sign(), x, eps, cons).detach()


def pgd(model, x, y, eps, steps=20, loss="ce", cons=None, rand=True):
    alpha = 2.5 * eps / steps
    xa = x + (torch.empty_like(x).uniform_(-eps, eps) if rand else 0)
    xa = _clip_ball(xa, x, eps, cons)
    for _ in range(steps):
        xa.requires_grad_(True)
        out = model(xa)
        l = F.cross_entropy(out, y) if loss == "ce" else -_margin(out, y).sum()
        g, = torch.autograd.grad(l, xa)
        xa = _clip_ball(xa.detach() + alpha * g.sign(), x, eps, cons)
    return xa.detach()


def pgd_worst(model, x, y, eps, steps=20, cons=None):
    """Per-sample worst case over PGD-CE and PGD-CW."""
    a = pgd(model, x, y, eps, steps, "ce", cons)
    b = pgd(model, x, y, eps, steps, "cw", cons)
    with torch.no_grad():
        ma, mb = _margin(model(a), y), _margin(model(b), y)
    return torch.where((mb < ma)[:, None], b, a)


@torch.no_grad()
def square(model, x, y, eps, n_queries=500, p_init=0.3, cons=None):
    """Square attack (Andriushchenko et al., 2020) adapted to 1-D feature
    vectors: each query re-draws a random contiguous window of features at
    +-eps and keeps the change if the margin loss drops."""
    B, d = x.shape
    xa = _clip_ball(x + eps * torch.sign(torch.randn_like(x)), x, eps, cons)
    best = _margin(model(xa), y)
    for i in range(n_queries - 1):
        active = best > 0
        if not active.any():
            break
        frac = p_init * (0.5 ** (8 * i // n_queries))       # halving schedule
        w = max(1, int(round(frac * d)))
        start = torch.randint(0, d - w + 1, (B, 1), device=x.device)
        cols = torch.arange(d, device=x.device)[None]
        win = (cols >= start) & (cols < start + w)
        sgn = torch.sign(torch.randn(B, 1, device=x.device))
        cand = torch.where(win, x + sgn * eps, xa)
        cand = _clip_ball(cand, x, eps, cons)
        m = _margin(model(cand), y)
        imp = (m < best) & active
        xa[imp], best[imp] = cand[imp], m[imp]
    return xa


def gaussian(x, sigma, cons=None):
    xa = x + sigma * torch.randn_like(x)
    return cons.project(xa, x) if cons is not None else xa
