"""
analyze_detect.py: tests of PROTOCOL_DETECT_V3.md on runs/detect_v3/detect.csv.

H_det1: AUROC(gate) > 0.5; H_det2: AUROC(gate) > AUROC(msp); for pgd,
pgd_constrained and transfer_tf, per dataset. One-sided exact sign-flip over
seeds, Holm over the 18 tests. Secondary comparisons are printed unadjusted.
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PRIMARY = ["pgd", "pgd_constrained", "transfer_tf"]


def signflip(x):
    """One-sided exact sign-flip p-value for mean(x) > 0."""
    x = np.asarray(x, float)
    obs = x.mean()
    c = sum(np.mean(x * np.array(s)) >= obs - 1e-12
            for s in itertools.product([1, -1], repeat=len(x)))
    return c / 2 ** len(x)


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def main(tag="detect_v3"):
    df = pd.read_csv(os.path.join(HERE, "runs", tag, "detect.csv"))
    piv = df.pivot_table(index=["dataset", "attack", "seed"], columns="score",
                         values="auroc")
    rows = []
    for (ds, att), g in piv.groupby(level=[0, 1]):
        n = len(g)
        for name, x in (("gate>0.5", g["gate"] - 0.5),
                        ("gate>msp", g["gate"] - g["msp"]),
                        ("gate>residual", g["gate"] - g["residual"]),
                        ("gate>mahalanobis", g["gate"] - g["mahalanobis"])):
            rows.append(dict(dataset=ds, attack=att, test=name, n=n,
                             gate=g["gate"].mean(), diff=x.mean(),
                             wins=int((x > 0).sum()), p=signflip(x.values)))
    r = pd.DataFrame(rows)
    fam = r.test.isin(["gate>0.5", "gate>msp"]) & r.attack.isin(PRIMARY)
    r["p_holm"] = np.nan
    r.loc[fam, "p_holm"] = holm(r.loc[fam, "p"].values)
    r["family"] = np.where(fam, "primary", "secondary")
    pd.set_option("display.width", 200)
    print(r.sort_values(["family", "test", "dataset", "attack"])
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    r.to_csv(os.path.join(HERE, "runs", tag, "tests.csv"), index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
