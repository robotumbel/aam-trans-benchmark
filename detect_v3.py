"""
detect_v3.py: the v3a token gate as an adversarial-input detector
(PROTOCOL_DETECT_V3.md).

For every dataset and seed: train aam_v3a with TRADES, attack the test
split, and score clean vs perturbed records with four detectors (gate,
residual, msp, mahalanobis). Writes runs/<tag>/detect.csv (one row per
dataset, seed, attack, score) and the trained model.

Usage:
  python detect_v3.py                       # full protocol, seeds 31-40
  python detect_v3.py --smoke --cpu         # code check: TON_IoT, seed 1, 1 epoch
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve

import attacks as A
from data import load_split
from models import build
from run import batched, seed_all
from train import predict, train

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["CICIoT2023", "CICIoMT2024", "TON_IoT"]
EPS = 0.10
SCORES = ["gate", "residual", "msp", "mahalanobis"]


def gate_closure(m, x):
    """Differentiable 1 - mean_j gamma_j(x) of the v3a token gate."""
    r = (x - m.loo_predict(x)).abs()
    g = torch.sigmoid(m.tg_a - F.softplus(m.tg_b) * r)
    return 1.0 - g.mean(1)


class Maha:
    def __init__(self, Xtr):
        self.mu = Xtr.mean(0)
        cov = np.cov(Xtr, rowvar=False) + 1e-3 * np.eye(Xtr.shape[1])
        self.P = np.linalg.inv(cov)

    def __call__(self, X):
        D = X - self.mu
        return np.sqrt(np.einsum("ij,jk,ik->i", D, self.P, D))


@torch.no_grad()
def all_scores(m, X, dev, maha, bs=2048):
    out = {k: [] for k in SCORES}
    pred = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).to(dev)
        p = F.softmax(m(x).float(), 1)
        pred.append(p.argmax(1).cpu())
        out["gate"].append(gate_closure(m, x).cpu())
        out["residual"].append((x - m.loo_predict(x)).abs().mean(1).cpu())
        out["msp"].append((1.0 - p.max(1).values).cpu())
    res = {k: torch.cat(v).numpy() for k, v in out.items() if v}
    res["mahalanobis"] = maha(X)
    return res, torch.cat(pred).numpy()


def adaptive_pgd(m, x, y, eps, lam, steps=20):
    """Detector-aware PGD: maximise CE - lam * gate closure."""
    alpha = 2.5 * eps / steps
    xa = (x + torch.empty_like(x).uniform_(-eps, eps)).detach()
    for _ in range(steps):
        xa.requires_grad_(True)
        # Both terms summed per record, so lam sets their per-record balance.
        l = (F.cross_entropy(m(xa), y, reduction="sum")
             - lam * gate_closure(m, xa).sum())
        g, = torch.autograd.grad(l, xa)
        xa = A._clip_ball(xa.detach() + alpha * g.sign(), x, eps, None)
    return xa.detach()


def tpr_at(y, s, fpr_max=0.05):
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= fpr_max].max())


def main(a):
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    out_dir = os.path.join(HERE, "runs", a.tag + ("_smoke" if a.smoke else ""))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "detect.csv")
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    fh = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new:
        w.writerow(["dataset", "seed", "attack", "score", "auroc",
                    "auroc_success", "tpr5", "n", "n_success", "mcc_attacked",
                    "clean_mcc"])
        fh.flush()
    dss, seeds, epochs, sur_ep = ((["TON_IoT"], [1], 1, 1) if a.smoke else
                                  (a.datasets, a.seeds, 15, 10))
    log = lambda s: print(s, flush=True)
    for ds in dss:
        for seed in seeds:
            done = os.path.join(out_dir, f"{ds}_aam_v3a_s{seed}.pt")
            if os.path.exists(done):
                continue
            t0 = time.time()
            s = load_split(ds, seed, "binary")
            d, C = s.Xtr.shape[1], len(s.class_names)
            cons = A.Constraint(s.lo, s.hi, s.is_int, s.mean, s.scale, dev,
                                s.log_mask)
            log(f"\n=== {ds} seed={seed} | train {len(s.Xtr)} test {len(s.Xte)}")

            # Transfer surrogate: standard vanilla Transformer.
            seed_all(10_000 + seed)
            sm = build("transformer", d, C).to(dev)
            train(sm, s, dev, "standard", epochs=sur_ep, seed=10_000 + seed,
                  log=lambda *_: None)
            X_tf = batched(lambda x, t: A.pgd(sm, x, t, EPS, 20, "ce"),
                           s.Xte, s.yte, dev)
            del sm

            seed_all(seed)
            m = build("aam_v3a", d, C).to(dev)
            train(m, s, dev, "trades", epochs=epochs, eps=EPS, steps=7,
                  beta=6.0, seed=seed, log=log)
            m.eval()
            maha = Maha(s.Xtr)
            sc0, p0 = all_scores(m, s.Xte, dev, maha)
            clean_mcc = matthews_corrcoef(s.yte, p0)
            log(f"  clean mcc {clean_mcc:.4f}")

            sets = {
                "pgd": batched(lambda x, t: A.pgd_worst(m, x, t, EPS, 20),
                               s.Xte, s.yte, dev),
                "pgd_constrained": batched(
                    lambda x, t: A.pgd_worst(m, x, t, EPS, 20, cons=cons),
                    s.Xte, s.yte, dev),
                "transfer_tf": X_tf,
                "gaussian": batched(lambda x, t: A.gaussian(x, EPS),
                                    s.Xte, s.yte, dev),
                "adaptive_l1": batched(lambda x, t: adaptive_pgd(m, x, t, EPS, 1.0),
                                       s.Xte, s.yte, dev),
                "adaptive_l10": batched(lambda x, t: adaptive_pgd(m, x, t, EPS, 10.0),
                                        s.Xte, s.yte, dev),
            }
            n = len(s.Xte)
            lab = np.r_[np.zeros(n), np.ones(n)]
            ok0 = p0 == s.yte
            for att, Xa in sets.items():
                sc1, p1 = all_scores(m, Xa, dev, maha)
                succ = ok0 & (p1 != s.yte)
                mcc_att = matthews_corrcoef(s.yte, p1)
                for k in SCORES:
                    sc = np.r_[sc0[k], sc1[k]]
                    auc = roc_auc_score(lab, sc)
                    if succ.sum() >= 10:
                        ls = np.r_[np.zeros(ok0.sum()), np.ones(succ.sum())]
                        auc_s = roc_auc_score(ls, np.r_[sc0[k][ok0], sc1[k][succ]])
                    else:
                        auc_s = float("nan")
                    w.writerow([ds, seed, att, k, auc, auc_s, tpr_at(lab, sc),
                                n, int(succ.sum()), mcc_att, clean_mcc])
                fh.flush()
                g = [float(x) for x in (
                    roc_auc_score(lab, np.r_[sc0[k], sc1[k]]) for k in SCORES)]
                log(f"  {att:16s} mcc {mcc_att:.3f} | AUROC gate {g[0]:.3f} "
                    f"resid {g[1]:.3f} msp {g[2]:.3f} maha {g[3]:.3f}")
            torch.save(m.state_dict(), done)
            log(f"  done {time.time() - t0:.0f}s")
            del m
    fh.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--tag", default="detect_v3")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(31, 41)))
    main(ap.parse_args())
