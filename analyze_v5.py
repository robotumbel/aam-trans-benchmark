"""analyze_v5.py: analyses of PROTOCOL_V5.md.

Outcomes per model and seed: clean MCC, PGD MCC (eps 0.1) and the
robustness-specific loss L = clean - PGD. Two-sided exact sign-flip tests,
Holm as stated in the protocol. Prints tables and writes runs/v5_tests.csv.
"""
import os

import numpy as np
import pandas as pd

from analyze_detect import holm, signflip

H = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.join(H, "runs", *p)
DS = ["CICIoT2023", "CICIoMT2024", "TON_IoT"]
OUT = ["clean", "pgd", "L"]


def two(x):
    return min(1.0, 2 * min(signflip(x), signflip(-x)))


def outcomes(df, ds, method="trades", mode="binary"):
    """seed x backbone frames for clean, pgd and L."""
    g = df[(df.dataset == ds) & (df.method == method) & (df["mode"] == mode)]
    c = g[g.attack == "clean"].pivot_table(index="seed", columns="backbone", values="mcc")
    p = g[(g.attack == "pgd") & np.isclose(g.eps, 0.1)].pivot_table(
        index="seed", columns="backbone", values="mcc")
    return {"clean": c, "pgd": p, "L": c - p}


def cmp_rows(o, a, b, **meta):
    rows = []
    for k in OUT:
        if a not in o[k] or b not in o[k]:
            continue
        x = (o[k][a] - o[k][b]).dropna().values
        if len(x) == 0:
            continue
        rows.append(dict(meta, a=a, b=b, outcome=k, n=len(x), mean_a=o[k][a].mean(),
                         mean_b=o[k][b].mean(), diff=x.mean(),
                         wins=int((x > 0).sum()), p=two(x)))
    return rows


def main():
    rows = []
    # Robustness-specific loss in the existing rounds (AAM-TRANS vs vanilla).
    for tag, meth in [("main", "trades"), ("main", "pgdat"), ("gate_confirm", "trades"),
                      ("gate3_confirm", "trades"), ("v4_confirm", "trades")]:
        df = pd.read_csv(R(tag, "results.csv"))
        for ds in DS:
            o = outcomes(df, ds, meth)
            for a, b in [("aam_trans", "transformer"), ("ft", "transformer"), ("aam_trans", "ft")]:
                rows += cmp_rows(o, a, b, part="rounds", tag=tag, method=meth, dataset=ds)
    # B: factorial (factorial tag + transformer/ft from main, same seeds).
    fpath = R("factorial", "results.csv")
    if os.path.exists(fpath):
        df = pd.concat([pd.read_csv(fpath), pd.read_csv(R("main", "results.csv"))])
        cells = ["grp_none", "grp_learn", "ft_sin", "ft_learn", "ft_nobias",
                 "shared_none", "shared_learn"]
        for ds in ["CICIoT2023", "TON_IoT"]:
            o = outcomes(df, ds)
            for ref in ["transformer", "ft"]:
                for c in cells + ["aam_trans", "aam_noGate", "aam_noPE", "aam_noTok"]:
                    rows += cmp_rows(o, c, ref, part="factorial", tag="factorial",
                                     method="trades", dataset=ds)
    # A3: multiclass.
    mc = pd.read_csv(R("multiclass", "results.csv"))
    for ds in DS:
        o = outcomes(mc, ds, mode="multiclass")
        for a, b in [("ft", "transformer"), ("aam_trans", "ft"), ("aam_trans", "transformer")]:
            rows += cmp_rows(o, a, b, part="multiclass", tag="multiclass", method="trades",
                             dataset=ds)
    r = pd.DataFrame(rows)
    # Holm: over datasets for rounds/multiclass; over cells for the factorial.
    r["p_holm"] = np.nan
    for _, g in r[r.part != "factorial"].groupby(["part", "tag", "method", "a", "b", "outcome"]):
        r.loc[g.index, "p_holm"] = holm(g.p.values)
    for _, g in r[r.part == "factorial"].groupby(["dataset", "b", "outcome"]):
        r.loc[g.index, "p_holm"] = holm(g.p.values)
    r.to_csv(R("v5_tests.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_rows", 1000)
    f = lambda v: f"{v:.4f}"
    for part in r.part.unique():
        print(f"\n===== {part} =====")
        print(r[r.part == part].drop(columns=["part"]).to_string(index=False, float_format=f))
    # C: stronger attacks.
    spath = R("strong", "results.csv")
    if os.path.exists(spath):
        s = pd.read_csv(spath)
        print("\n===== strong attacks (mean MCC) =====")
        print(s.groupby(["dataset", "backbone"])[["clean", "pgd20", "pgd100r5", "clean_sq",
                                                   "pgd20_sq", "square5000"]].mean().round(4))
        out = []
        for ds in s.dataset.unique():
            for col in ["pgd20", "pgd100r5", "square5000"]:
                p = s[s.dataset == ds].pivot_table(index="seed", columns="backbone", values=col)
                for a, b in [("ft", "transformer"), ("aam_trans", "transformer"), ("aam_trans", "ft")]:
                    x = (p[a] - p[b]).dropna().values
                    out.append(dict(dataset=ds, attack=col, cmp=f"{a}-{b}", diff=x.mean(),
                                    wins=int((x > 0).sum()), n=len(x), p=two(x)))
        print(pd.DataFrame(out).to_string(index=False, float_format=f))


if __name__ == "__main__":
    main()
