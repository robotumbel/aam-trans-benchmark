"""
make_tables_v3.py: tables and numbers for the v3 gate study
(PROTOCOL_GATE_V3.md) and the gate-as-detector study (PROTOCOL_DETECT_V3.md).

Reads runs/gate3_select/val_scores.csv, runs/gate3_confirm/results.csv and
runs/detect_v3/detect.csv; writes runs/main/tables/tab_gate3.tex,
tab_detect.tex and runs/main/tables/v3_numbers.txt.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from analyze_detect import holm, signflip

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "main", "tables")
DS = ["CICIoT2023", "CICIoMT2024", "TON_IoT"]
PRIMARY = ["pgd", "pgd_constrained", "square", "transfer_tf"]
ANAME = {"pgd": "PGD", "pgd_constrained": "Constrained PGD", "square": "Square",
         "transfer_tf": "Transfer (Transf.)", "clean": "Clean",
         "gaussian": "Gaussian (control)", "adaptive_l1": "Adaptive, $\\lambda=1$",
         "adaptive_l10": "Adaptive, $\\lambda=10$"}
DNAME = {"CICIoT2023": "CICIoT2023", "CICIoMT2024": "CICIoMT2024",
         "TON_IoT": "TON\\_IoT"}


def two_sided(x):
    x = np.asarray(x, float)
    return min(1.0, 2 * min(signflip(x), signflip(-x)))


def paired(d, att, eps, a, b):
    g = d[(d.attack == att) & (np.isclose(d.eps, eps))]
    p = g.pivot_table(index="seed", columns="backbone", values="mcc")
    return (p[a] - p[b]).dropna().values


def cell(x, star=False):
    v = x.mean()
    s = "0.0000" if abs(v) < 5e-5 else f"{v:+.4f}".replace("-", "$-$")
    return f"{s}{'$^{*}$' if star else ''} ({int((x > 0).sum())}/{len(x)})"


def gate3(lines):
    sel = pd.read_csv(os.path.join(HERE, "runs", "gate3_select", "val_scores.csv"))
    sc = sel.groupby("backbone").val_pgd_mcc.mean().sort_values(ascending=False)
    lines.append("validation PGD MCC (selection):\n" + sc.round(4).to_string())
    d = pd.read_csv(os.path.join(HERE, "runs", "gate3_confirm", "results.csv"))
    d = d[d["mode"] == "binary"]
    # Primary family H_v3: one-sided v3b > noGate, Holm over 12.
    keys, ps, xs = [], [], {}
    for att in PRIMARY:
        for ds in DS:
            eps = 0.1
            x = paired(d[d.dataset == ds], att, eps, "aam_v3b", "aam_noGate")
            keys.append((att, ds)); ps.append(signflip(x)); xs[(att, ds)] = x
    ph = dict(zip(keys, holm(ps)))
    lines.append("\nH_v3 (v3b - noGate), one-sided, Holm over 12:")
    for k in keys:
        x = xs[k]
        lines.append(f"  {k[0]:16s} {k[1]:12s} {x.mean():+.4f} "
                     f"{(x > 0).sum()}/10 p={signflip(x):.4f} holm={ph[k]:.4f}")
    rows = []
    for att in PRIMARY:
        rows.append(f"Masking (v3) $-$ no gate & {ANAME[att]} & " + " & ".join(
            cell(xs[(att, ds)], ph[(att, ds)] < 0.05) for ds in DS) + r" \\")
    rows.append(r"\midrule")
    # Secondary comparisons, two-sided, Holm over the three datasets.
    for a, b, lab, att in [
            ("aam_v3b", "aam_noGate", "Masking (v3) $-$ no gate", "clean"),
            ("aam_v3b", "aam_trans", "Masking (v3) $-$ gate v1", "pgd"),
            ("aam_trans", "aam_noGate", "Gate v1 $-$ no gate", "pgd"),
            ("aam_trans", "transformer", "AAM-TRANS (v1) $-$ vanilla", "pgd"),
            ("aam_trans", "transformer", "AAM-TRANS (v1) $-$ vanilla", "pgd_constrained")]:
        xx = [paired(d[d.dataset == ds], att, 0.0 if att == "clean" else 0.1, a, b)
              for ds in DS]
        hp = holm([two_sided(x) for x in xx])
        rows.append(f"{lab} & {ANAME[att]} & " + " & ".join(
            cell(x, h < 0.05) for x, h in zip(xx, hp)) + r" \\")
        lines.append(f"{lab:28s} {att:16s} " + "  ".join(
            f"{x.mean():+.4f} ({(x > 0).sum()}/10, holm {h:.4f})"
            for x, h in zip(xx, hp)))
    # eps sweep, v3b - noGate under PGD
    for e in (0.05, 0.2):
        xx = [paired(d[d.dataset == ds], "pgd", e, "aam_v3b", "aam_noGate") for ds in DS]
        lines.append(f"v3b-noGate PGD eps {e}: " + "  ".join(
            f"{x.mean():+.4f} ({(x > 0).sum()}/10, p2 {two_sided(x):.3f})" for x in xx))
        xx = [paired(d[d.dataset == ds], "pgd", e, "aam_trans", "transformer") for ds in DS]
        lines.append(f"v1-vanilla PGD eps {e}: " + "  ".join(
            f"{x.mean():+.4f} ({(x > 0).sum()}/10)" for x in xx))
    with open(os.path.join(OUT, "tab_gate3.tex"), "w") as f:
        f.write("\n".join(rows) + "\n")


def detect(lines):
    df = pd.read_csv(os.path.join(HERE, "runs", "detect_v3", "detect.csv"))
    t = pd.read_csv(os.path.join(HERE, "runs", "detect_v3", "tests.csv"))
    sig = {(r.dataset, r.attack, r.test): r.p_holm < 0.05
           for r in t.itertuples() if r.family == "primary"}
    m = df.pivot_table(index=["dataset", "attack"], columns="score", values="auroc")
    tp = df[df.score == "gate"].pivot_table(index=["dataset", "attack"], values="tpr5")
    rows = []
    order = ["pgd", "pgd_constrained", "transfer_tf", "gaussian", "adaptive_l10"]
    for ds in DS:
        for i, att in enumerate(order):
            r = m.loc[(ds, att)]
            mark = ""
            if att in ("pgd", "pgd_constrained", "transfer_tf"):
                a = sig.get((ds, att, "gate>0.5"), False)
                b = sig.get((ds, att, "gate>msp"), False)
                mark = "$^{" + ",".join(s for s, ok in (("a", a), ("b", b)) if ok) + "}$" \
                    if (a or b) else ""
            first = (r"\multirow{5}{*}{" + DNAME[ds] + "}") if i == 0 else ""
            rows.append(f"{first} & {ANAME[att]} & {r['gate']:.3f}{mark} & "
                        f"{r['residual']:.3f} & {r['msp']:.3f} & {r['mahalanobis']:.3f} & "
                        f"{tp.loc[(ds, att), 'tpr5']:.3f} \\\\")
        if ds != DS[-1]:
            rows.append(r"\midrule")
    with open(os.path.join(OUT, "tab_detect.tex"), "w") as f:
        f.write("\n".join(rows) + "\n")
    lines.append("\nDetection: primary tests\n" + t[t.family == "primary"][
        ["dataset", "attack", "test", "gate", "diff", "wins", "p_holm"]].to_string(index=False))
    sec = t[(t.family == "secondary") & t.attack.isin(["pgd", "pgd_constrained", "transfer_tf"])]
    lines.append("\nDetection: secondary\n" + sec[
        ["dataset", "attack", "test", "diff", "wins", "p"]].to_string(index=False))
    succ = df[df.score == "gate"].pivot_table(index=["dataset", "attack"],
                                              values=["auroc_success", "mcc_attacked", "clean_mcc"])
    lines.append("\n" + succ.round(3).to_string())


if __name__ == "__main__":
    lines = []
    gate3(lines)
    detect(lines)
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "v3_numbers.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)
