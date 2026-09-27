"""
eval_realistic.py: attacker-controllable-feature threat model, evaluated on
the saved checkpoints of the main grid (no retraining).

The adversary shapes only its own traffic. Per dataset, every feature is
  * immutable  (protocol, service, flags, victim-side responses), or
  * free       (timing, rate, TTL, header length, packet counts), or
  * increase-only (sizes that can only grow by padding: Min, Max, AVG,
    Tot sum, Tot size, bytes and payload lengths; durations that can only
    grow by delaying).
On top of this, the range-and-integrality projection of attacks.Constraint
applies. Because log1p and standardisation are monotone, "increase-only"
in raw units is "increase-only" in model space.

The attack is the worst-of-two PGD-20 used elsewhere, at eps in {0.1, 0.2,
0.5} standard deviations (larger budgets are relevant because only a subset
of features may move). Output: runs/realistic/results.csv

Usage: python eval_realistic.py [--backbones ...] [--methods trades pgdat]
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef

import attacks as A
from data import load_split
from models import build
from run import batched
from train import predict

HERE = os.path.dirname(os.path.abspath(__file__))

# Attacker-controllable features: name -> "free" | "up" (increase-only).
# Anything not listed is immutable.
CONTROL = {
    "CICIoT2023": {
        "Header_Length": "free", "Time_To_Live": "free", "Rate": "free",
        "IAT": "free", "Number": "free", "Std": "free", "Variance": "free",
        "Tot sum": "up", "Min": "up", "Max": "up", "AVG": "up", "Tot size": "up",
    },
    "TON_IoT": {
        "duration": "up", "src_bytes": "up", "src_pkts": "up",
        "src_ip_bytes": "up", "http_request_body_len": "up",
    },
}
CONTROL["CICIoMT2024"] = CONTROL["CICIoT2023"]      # same 39-feature schema
EPS = (0.1, 0.2, 0.5)


class RealisticConstraint(A.Constraint):
    def __init__(self, s, dev, ds):
        super().__init__(s.lo, s.hi, s.is_int, s.mean, s.scale, dev, s.log_mask)
        spec = CONTROL[ds]
        t = lambda v: torch.tensor(v, dtype=torch.bool, device=dev)
        self.free = t([spec.get(f) == "free" for f in s.feature_names])
        self.up = t([spec.get(f) == "up" for f in s.feature_names])
        self.n_mutable = int((self.free | self.up).sum())

    def project(self, x, x0=None):
        if x0 is not None:
            x = torch.where(self.up, torch.max(x, x0), x)
            x = torch.where(self.free | self.up, x, x0)
        return super().project(x, x0)


def main(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(HERE, "runs", "realistic")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results.csv")
    done = set()
    if os.path.exists(path):
        import pandas as pd
        done = {(r.dataset, r.method, r.backbone, r.seed)
                for r in pd.read_csv(path).itertuples()}
    fh = open(path, "a", newline="")
    w = csv.writer(fh)
    if not done:
        w.writerow(["dataset", "method", "backbone", "seed", "n_mutable", "n_features",
                    "eps", "mcc_clean", "mcc", "valid_rate", "mean_abs_change"])
    for ds in ("CICIoT2023", "CICIoMT2024", "TON_IoT"):
        for seed in a.seeds:
            s = load_split(ds, seed, "binary")
            cons = RealisticConstraint(s, dev, ds)
            X0 = torch.from_numpy(s.Xte).to(dev)
            for method in a.methods:
                for bb in a.backbones:
                    if (ds, method, bb, seed) in done:
                        continue
                    ck = os.path.join(HERE, "runs", "main",
                                      f"{ds}_binary_{method}_{bb}_s{seed}.pt")
                    if not os.path.exists(ck):
                        continue
                    m = build(bb, s.Xtr.shape[1], len(s.class_names)).to(dev)
                    m.load_state_dict(torch.load(ck, map_location=dev))
                    m.eval()
                    clean = matthews_corrcoef(s.yte, predict(m, s.Xte, dev).argmax(1).numpy())
                    for e in EPS:
                        Xa = batched(lambda x, t: A.pgd_worst(m, x, t, e, 20, cons=cons),
                                     s.Xte, s.yte, dev)
                        mcc = matthews_corrcoef(s.yte, predict(m, Xa, dev).argmax(1).numpy())
                        xa = torch.from_numpy(Xa).to(dev)
                        vr = float(cons.valid(xa, X0).float().mean())
                        chg = float((xa - X0).abs().mean())
                        w.writerow([ds, method, bb, seed, cons.n_mutable,
                                    len(s.feature_names), e, clean, mcc, vr, chg])
                        fh.flush()
                    print(f"{ds} {method} {bb} s{seed}: clean {clean:.3f} -> "
                          f"eps0.5 {mcc:.3f} (mutable {cons.n_mutable}/{len(s.feature_names)})",
                          flush=True)
                    del m
                    torch.cuda.empty_cache()
    fh.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+",
                    default=["transformer", "aam_trans", "mlp", "lstm"])
    ap.add_argument("--methods", nargs="+", default=["trades"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    main(ap.parse_args())
