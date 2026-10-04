"""analyze_v4.py: tests of PROTOCOL_V4.md on runs/v4_confirm/results.csv.

Primary H_v4: selected - ft > 0 for five attacks at eps 0.1, per dataset;
one-sided exact sign-flip, Holm over 15. Secondary comparisons two-sided,
Holm over the three datasets within each comparison and attack.
"""
import os

import numpy as np
import pandas as pd

from analyze_detect import holm, signflip

HERE = os.path.dirname(os.path.abspath(__file__))
DS = ["CICIoT2023", "CICIoMT2024", "TON_IoT"]
PRIMARY = ["pgd", "pgd_constrained", "square", "transfer_tf", "transfer_mlp"]


def two_sided(x):
    return min(1.0, 2 * min(signflip(x), signflip(-x)))


def main():
    sel = open(os.path.join(HERE, "runs", "v4_select", "SELECTED_V4.txt")).readline().strip()
    d = pd.read_csv(os.path.join(HERE, "runs", "v4_confirm", "results.csv"))
    print("runs per dataset/backbone:\n",
          d[d.attack == "clean"].groupby(["dataset", "backbone"]).size().unstack())

    def diff(ds, att, eps, a, b):
        g = d[(d.dataset == ds) & (d.attack == att) & np.isclose(d.eps, eps)]
        p = g.pivot_table(index="seed", columns="backbone", values="mcc")
        return (p[a] - p[b]).dropna().values

    rows = []
    for att in PRIMARY:
        for ds in DS:
            x = diff(ds, att, 0.1, sel, "ft")
            rows.append(dict(cmp=f"{sel}-ft", attack=att, dataset=ds, n=len(x),
                             diff=x.mean(), wins=int((x > 0).sum()), p=signflip(x)))
    prim = pd.DataFrame(rows)
    prim["p_holm"] = holm(prim.p.values)
    print(f"\nPRIMARY H_v4 ({sel} - ft, one-sided, Holm over 15)")
    print(prim.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    rows = []
    for a, b in [(sel, "aam_noGate"), ("aam_noGate", "ft"), ("aam_trans", "ft"),
                 ("ft", "transformer"), ("aam_trans", "transformer")]:
        for att, eps in [("clean", 0.0)] + [(k, 0.1) for k in PRIMARY] + \
                [("pgd", 0.05), ("pgd", 0.2), ("transfer_tf", 0.2)]:
            xs = [diff(ds, att, eps, a, b) for ds in DS]
            hp = holm([two_sided(x) for x in xs])
            for ds, x, h in zip(DS, xs, hp):
                rows.append(dict(cmp=f"{a}-{b}", attack=att, eps=eps, dataset=ds,
                                 diff=x.mean(), wins=int((x > 0).sum()), n=len(x),
                                 p_holm=h))
    sec = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_rows", 500)
    print("\nSECONDARY (two-sided, Holm over datasets)")
    w = sec.assign(cell=[f"{r['diff']:+.4f} ({r.wins}/{r.n}){'*' if r.p_holm < 0.05 else ''}"
                         for _, r in sec.iterrows()])
    print(w.pivot_table(index=["cmp", "attack", "eps"], columns="dataset", values="cell",
                        aggfunc="first", sort=False)[DS].to_string())
    out = os.path.join(HERE, "runs", "v4_confirm")
    prim.to_csv(os.path.join(out, "tests_primary.csv"), index=False)
    sec.to_csv(os.path.join(out, "tests_secondary.csv"), index=False)
    m = d[np.isclose(d.eps, 0.1) | (d.attack == "clean")].pivot_table(
        index=["dataset", "backbone"], columns="attack", values="mcc")
    print("\nMEAN MCC\n", m[["clean"] + PRIMARY].round(4).to_string())


if __name__ == "__main__":
    main()
