"""make_tables.py: LaTeX tables not produced by analyze.py.

Writes into runs/main/tables/:
  tab_ablation.tex     single-switch ablations, per dataset, PGD eps 0.1
  tab_gate.tex         gate v2 confirmation (runs/gate_confirm)
  tab_budget.tex       AAM minus vanilla across budgets (TRADES, PGD-AT, eps-0.2 training)
  tab_cost.tex         inference cost (runs/deploy.csv)
  tab_perclass.tex     lowest-recall CICIoT2023 classes, plain vs class-weighted CE
  tab_realistic.tex    attacker-controllable-feature attack (runs/realistic)
Run after analyze.py, analyze_gate.py and bench_deploy.py.
"""
import json
import glob
import os

import numpy as np
import pandas as pd

from analyze import compare, fmt_p, DATASETS

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "runs", "main", "tables")
DS_SHORT = {"CICIoT2023": "CICIoT", "CICIoMT2024": "CICIoMT", "TON_IoT": "TON\\_IoT"}
LBL = {"aam_noGate": "w/o gate", "aam_noTok": "w/o feature tokeniser",
       "aam_noPE": "w/o learned PE", "aam_noAux": "w/o reconstruction",
       "vanilla_gate": "Vanilla + gate (vs.\\ vanilla)"}


def sgn(x):
    """Signed 3-decimal number without a spurious '-0.000'."""
    return f"{0.0 if abs(x) < 5e-4 else x:+.3f}"


def cell(r):
    star = "$^{*}$" if r.p_holm < 0.05 else ""
    return f"{sgn(r.mean_diff)}{star} ({r.wins}/{r.wins + r.losses})"


def holm_within(st):
    from analyze import holm
    st = st.copy()
    st["p_holm"] = holm(st.p.values)
    return st


def write(name, lines):
    with open(os.path.join(TAB, name), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def ablation(df):
    rows = []
    for b in ["aam_noGate", "aam_noTok", "aam_noPE", "aam_noAux", "vanilla_gate"]:
        ref = "transformer" if b == "vanilla_gate" else "aam_trans"
        st = compare(df, ref, b, ["clean", "pgd"])
        st["b"] = b
        rows.append(st[st.scope != "pooled"])
    st = pd.concat(rows)
    m = st.attack == "pgd"
    st.loc[m, "p_holm"] = holm_within(st[m]).p_holm
    st.loc[~m, "p_holm"] = holm_within(st[~m]).p_holm
    lines = []
    for b in ["aam_noGate", "aam_noTok", "aam_noPE", "aam_noAux", "vanilla_gate"]:
        cells = []
        for atk in ["clean", "pgd"]:
            for ds in DATASETS:
                r = st[(st.b == b) & (st.attack == atk) & (st.scope == ds)].iloc[0]
                cells.append(cell(r))
        lines.append(f"{LBL[b]} & " + " & ".join(cells) + " \\\\")
    write("tab_ablation.tex", lines)


def gate():
    st = pd.read_csv(os.path.join(HERE, "runs", "gate_confirm", "tables",
                                  "stats_gate_confirm.csv"))
    fams = [("primary: v2 - noGate", "Gate v2 $-$ no gate"),
            ("secondary: v2 - v1", "Gate v2 $-$ gate v1"),
            ("replication: aam_trans - aam_noGate", "Gate v1 $-$ no gate"),
            ("replication: aam_trans - transformer", "AAM-TRANS (v1) $-$ vanilla")]
    lines = []
    for fam, lab in fams:
        cells = []
        for ds in DATASETS:
            r = st[(st.family == fam) & (st.attack == "pgd") & (st.scope == ds)].iloc[0]
            cells.append(cell(r))
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    write("tab_gate.tex", lines)


def budget(df, df02):
    lines = []
    for meth, lab, d in [("trades", "TRADES, train $\\epsilon{=}0.1$", df),
                         ("pgdat", "PGD-AT, train $\\epsilon{=}0.1$", df),
                         ("trades", "TRADES, train $\\epsilon{=}0.2$", df02)]:
        if d is None:
            continue
        cells = []
        for ds in DATASETS:
            for e in (0.05, 0.10, 0.20):
                st = compare(d, "aam_trans", "transformer", ["pgd"], eps=e, method=meth)
                q = st[st.scope == ds] if not st.empty else st
                cells.append("--" if q.empty else
                             f"{sgn(q.iloc[0].mean_diff)} ({q.iloc[0].wins})")
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    write("tab_budget.tex", lines)


def cost():
    d = pd.read_csv(os.path.join(HERE, "runs", "deploy.csv")).set_index("backbone")
    names = [("transformer", "Vanilla Transformer"), ("aam_trans", "AAM-TRANS"),
             ("aam_noGate", "\\quad w/o gate"), ("aam_noTok", "\\quad w/o feature tok."),
             ("mlp", "MLP"), ("lstm", "LSTM")]
    lines = []
    for k, lab in names:
        r = d.loc[k]
        lines.append(f"{lab} & {r.params/1e3:.0f}k & {r.macs/1e6:.1f} & "
                     f"{r.cpu1_b1_med_us/1e3:.2f} & {r.cpu1_b1_p95_us/1e3:.2f} & "
                     f"{r.cpu1_b1_int8_med_us/1e3:.2f} & {r.cpu1_b256_samples_per_s/1e3:.1f} & "
                     f"{r.size_fp32_mb:.2f} \\\\")
    write("tab_cost.tex", lines)


def perclass():
    def load(tag, b):
        return [json.load(open(f)) for f in
                glob.glob(os.path.join(HERE, "runs", tag,
                                       f"CICIoT2023_multiclass_trades_{b}_s*.json"))]
    plain, cw = load("multiclass", "aam_trans"), load("multiclass_cw", "aam_trans")
    names = plain[0]["class_names"]
    diag = pd.read_csv(os.path.join(HERE, "runs", "diag_deadclass.csv"))
    diag = diag.pivot_table(index="cls", columns="check", values="value")
    rec = lambda R, i: np.mean([r["results"]["clean@0"]["per_class"]["recall"][i] for r in R])
    order = sorted(range(len(names)), key=lambda i: rec(plain, i))[:9]
    lines = []
    for i in order:
        c = names[i]
        sup = np.mean([r["results"]["clean@0"]["per_class"]["support"][i] for r in plain])
        dz = diag.loc[c] if c in diag.index else None
        extra = (f"{dz.knn5_benign:.2f} & {dz.rf_balanced:.2f}" if dz is not None
                 else "-- & --")
        lines.append(f"{c.replace('_', '\\_')} & {sup:.0f} & {rec(plain, i):.3f} & "
                     f"{rec(cw, i):.3f} & {extra} \\\\")
    bi = names.index("BENIGN")
    lines.append("\\midrule")
    lines.append(f"BENIGN & {np.mean([r['results']['clean@0']['per_class']['support'][bi] for r in plain]):.0f}"
                 f" & {rec(plain, bi):.3f} & {rec(cw, bi):.3f} & -- & -- \\\\")
    write("tab_perclass.tex", lines)


def realistic():
    """Attacker-controllable-feature attack (runs/realistic): mean MCC per
    backbone and AAM-TRANS minus vanilla with seeds won; Holm within method."""
    from analyze import signflip_p, holm
    path = os.path.join(HERE, "runs", "realistic", "results.csv")
    if not os.path.exists(path):
        return
    d = pd.read_csv(path)
    d = d[d.backbone.isin(["transformer", "aam_trans", "mlp"])]
    stats = []
    for meth in ("trades", "pgdat"):
        for ds in DATASETS:
            for e in (0.1, 0.2, 0.5):
                q = d[(d.method == meth) & (d.dataset == ds) & (d.eps == e)]
                q = q.pivot_table(index="seed", columns="backbone", values="mcc")
                x = (q.aam_trans - q.transformer).dropna().values
                stats.append(dict(method=meth, ds=ds, eps=e, diff=x.mean(),
                                  wins=int((x > 0).sum()), n=len(x), p=signflip_p(x)))
    st = pd.DataFrame(stats)
    for meth in ("trades", "pgdat"):
        m = st.method == meth
        st.loc[m, "p_holm"] = holm(st.loc[m, "p"].values)
    st.to_csv(os.path.join(HERE, "runs", "realistic", "stats.csv"), index=False)
    mean = d.groupby(["method", "dataset", "backbone", "eps"]).mcc.mean()
    lines = []
    for meth, mlab in (("trades", "TRADES"), ("pgdat", "PGD-AT")):
        for ds in DATASETS:
            cells = []
            for e in (0.1, 0.2, 0.5):
                r = st[(st.method == meth) & (st.ds == ds) & (st.eps == e)].iloc[0]
                star = "$^{*}$" if r.p_holm < 0.05 else ""
                v = mean.get((meth, ds, "transformer", e), np.nan)
                a = mean.get((meth, ds, "aam_trans", e), np.nan)
                cells.append(f"{v:.3f} & {a:.3f} & {sgn(r['diff'])}{star} ({r.wins})")
            lines.append(f"{mlab} & {ds.replace('_', chr(92) + '_')} & " + " & ".join(cells) + " \\\\")
    write("tab_realistic.tex", lines)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(HERE, "runs", "main", "results.csv"))
    p02 = os.path.join(HERE, "runs", "train_eps02", "results.csv")
    df02 = pd.read_csv(p02) if os.path.exists(p02) else None
    ablation(df); gate(); budget(df, df02); cost(); perclass(); realistic()
    for f in sorted(glob.glob(os.path.join(TAB, "tab_*.tex"))):
        print("wrote", os.path.basename(f))
