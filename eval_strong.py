"""eval_strong.py: stronger attacks on saved checkpoints (PROTOCOL_V5.md, part C).

PGD-100 with five random restarts (CE and CW margin, per-record worst case)
and Square with 5,000 queries, on the round-four checkpoints. Evaluation only.
Output: runs/strong/results.csv

Usage: python eval_strong.py [--smoke]
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef

import attacks as A
from data import load_split
from models import build
from train import predict

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 0.10


def worst_pgd(m, x, y, steps, restarts):
    """Per-record worst case (lowest CW margin) over restarts and two losses."""
    best, bm = x.clone(), torch.full((len(x),), float("inf"), device=x.device)
    for _ in range(restarts):
        for loss in ("ce", "cw"):
            xa = A.pgd(m, x, y, EPS, steps, loss)
            with torch.no_grad():
                mg = A._margin(m(xa), y)
            upd = mg < bm
            best[upd], bm[upd] = xa[upd], mg[upd]
    return best


def run(fn, X, y, dev, bs=1024):
    out = []
    for i in range(0, len(X), bs):
        out.append(fn(torch.from_numpy(X[i:i + bs]).to(dev),
                      torch.from_numpy(y[i:i + bs]).to(dev)).cpu())
    return torch.cat(out).numpy()


def main(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(HERE, "runs", "strong_smoke" if a.smoke else "strong")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results.csv")
    done = set()
    if os.path.exists(path) and os.path.getsize(path):
        done = {(r.dataset, r.backbone, r.seed) for r in pd.read_csv(path).itertuples()}
    fh = open(path, "a", newline="")
    w = csv.writer(fh)
    if not done:
        w.writerow(["dataset", "backbone", "seed", "n", "clean", "pgd20", "pgd100r5",
                    "n_sq", "clean_sq", "pgd20_sq", "square5000"])
        fh.flush()
    steps, restarts, queries, n_max, n_sq = ((10, 1, 100, 500, 200) if a.smoke
                                             else (100, 5, 5000, 5000, 1000))
    mcc = lambda m, X, y: matthews_corrcoef(y, predict(m, X, dev).argmax(1).numpy())
    for ds in a.datasets:
        for seed in a.seeds:
            s = load_split(ds, seed, "binary")
            rng = np.random.default_rng(1000 + seed)
            idx = np.sort(rng.choice(len(s.Xte), min(n_max, len(s.Xte)), replace=False))
            X, y = s.Xte[idx], s.yte[idx]
            for bb in a.backbones:
                if (ds, bb, seed) in done:
                    continue
                ck = os.path.join(HERE, "runs", a.tag, f"{ds}_binary_trades_{bb}_s{seed}.pt")
                if not os.path.exists(ck):
                    print("missing", ck, flush=True)
                    continue
                m = build(bb, X.shape[1], len(s.class_names)).to(dev)
                m.load_state_dict(torch.load(ck, map_location=dev))
                m.eval()
                p20 = run(lambda x, t: A.pgd_worst(m, x, t, EPS, 20), X, y, dev)
                p100 = run(lambda x, t: worst_pgd(m, x, t, steps, restarts), X, y, dev)
                Xs, ys = X[:n_sq], y[:n_sq]
                sq = run(lambda x, t: A.square(m, x, t, EPS, queries), Xs, ys, dev)
                row = [ds, bb, seed, len(X), mcc(m, X, y), mcc(m, p20, y), mcc(m, p100, y),
                       len(Xs), mcc(m, Xs, ys), mcc(m, p20[:n_sq], ys), mcc(m, sq, ys)]
                w.writerow(row); fh.flush()
                print(f"{ds} s{seed} {bb:12s} clean {row[4]:.4f} pgd20 {row[5]:.4f} "
                      f"pgd100r5 {row[6]:.4f} | sq-subset clean {row[8]:.4f} "
                      f"pgd20 {row[9]:.4f} square5000 {row[10]:.4f}", flush=True)
                del m
    fh.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="v4_confirm")
    ap.add_argument("--datasets", nargs="+", default=["CICIoT2023", "TON_IoT"])
    ap.add_argument("--backbones", nargs="+", default=["transformer", "ft", "aam_trans"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(41, 51)))
    main(ap.parse_args())
