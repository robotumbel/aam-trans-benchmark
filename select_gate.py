"""
select_gate.py: gate v2 selection on VALIDATION data only
(PROTOCOL_GATE_V2.md). The test split of every seed is never read.

Writes runs/gate_select/val_scores.csv and prints the selected candidate.

Usage:
  python select_gate.py            # full protocol
  python select_gate.py --smoke    # 1 epoch, TON_IoT seed 1, CPU-friendly
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
from models import GATE_REG, build
from run import batched, seed_all
from train import predict, train

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = ["aam_trans", "aam_g1", "aam_g2", "aam_g3"]
DATASETS = ["CICIoT2023", "TON_IoT"]
SEEDS = [1, 2, 3]
EPS = 0.10


def main(a):
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    global CANDIDATES
    if a.candidates:
        CANDIDATES = a.candidates
    out_dir = os.path.join(HERE, "runs", a.tag + ("_smoke" if a.smoke else ""))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "val_scores.csv")
    done = set()
    if os.path.exists(path):
        done = {(r.dataset, r.backbone, r.seed) for r in pd.read_csv(path).itertuples()}
    fh = open(path, "a", newline="")
    w = csv.writer(fh)
    if not done:
        w.writerow(["dataset", "backbone", "seed", "val_clean_mcc", "val_pgd_mcc"])
    dss, seeds, epochs = ((["TON_IoT"], [1], 1) if a.smoke
                          else (DATASETS, SEEDS, 15))
    for ds in dss:
        for seed in seeds:
            s = load_split(ds, seed, "binary")
            # Validation-only: the split's test arrays are dropped here.
            s.Xte = s.yte = None
            d, C = s.Xtr.shape[1], len(s.class_names)
            for bb in CANDIDATES:
                if (ds, bb, seed) in done:
                    continue
                seed_all(seed)
                m = build(bb, d, C).to(dev)
                train(m, s, dev, "trades", epochs=epochs, eps=EPS, steps=7,
                      beta=6.0, seed=seed, log=lambda *_: None,
                      gate_reg=GATE_REG.get(bb))
                m.eval()
                clean = matthews_corrcoef(s.yva, predict(m, s.Xva, dev).argmax(1).numpy())
                Xp = batched(lambda x, t: A.pgd_worst(m, x, t, EPS, 20),
                             s.Xva, s.yva, dev)
                rob = matthews_corrcoef(s.yva, predict(m, Xp, dev).argmax(1).numpy())
                w.writerow([ds, bb, seed, clean, rob]); fh.flush()
                print(f"{ds} s{seed} {bb:10s} val clean {clean:.4f} val PGD {rob:.4f}",
                      flush=True)
                del m
    fh.close()
    df = pd.read_csv(path)
    score = df.groupby("backbone").val_pgd_mcc.mean().sort_values(ascending=False)
    print("\nMean validation PGD MCC:\n" + score.to_string())
    cand = score.drop("aam_trans", errors="ignore")
    best = cand.index[0]
    beats = cand.iloc[0] > score.get("aam_trans", -np.inf)
    print(f"\nSELECTED: {best} ({'beats' if beats else 'does NOT beat'} aam_trans "
          f"on validation)")
    with open(os.path.join(out_dir, "SELECTED.txt"), "w") as f:
        f.write(f"{best}\nbeats_v1={beats}\n{score.to_string()}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--candidates", nargs="+", default=None,
                    help="backbones to compare (default: the gate v2 set)")
    ap.add_argument("--tag", default="gate_select", help="output folder under runs/")
    main(ap.parse_args())
