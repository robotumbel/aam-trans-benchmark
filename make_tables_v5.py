"""make_tables_v5.py: tables of the v5 study (PROTOCOL_V5.md).

Run analyze_v5.py first (writes runs/v5_tests.csv). Writes to
runs/main/tables: tab_main_trades.tex (with the FT-style control),
tab_factorial.tex, tab_loss.tex, tab_strong.tex.
"""
import os

import numpy as np
import pandas as pd

H = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.join(H, "runs", *p)
OUT = R("main", "tables")
DS = ["CICIoT2023", "CICIoMT2024", "TON_IoT"]
T = pd.read_csv(R("v5_tests.csv"))


def write(name, lines):
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"--- {name}\n" + "\n".join(lines))


def sg(v, nd=3):
    s = f"{v:+.{nd}f}"
    if float(s) == 0:
        return f"{0:.{nd}f}"
    return s.replace("-", "$-$")


def main_table():
    d = pd.read_csv(R("main", "results.csv"))
    d = d[(d["mode"] == "binary") & (d.method == "trades")]
    names = [("transformer", "Vanilla Transformer"), ("ft", "FT-style control"),
             ("aam_trans", "AAM-TRANS"), ("mlp", "MLP"), ("lstm", "LSTM")]
    atk = ["clean", "pgd", "pgd_constrained", "square", "transfer_tf", "transfer_mlp"]
    lines = []
    for ds in DS:
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{ds.replace('_', chr(92) + '_')}}}}} \\\\")
        for k, lab in names:
            cells = []
            for a in atk:
                q = d[(d.dataset == ds) & (d.backbone == k) & (d.attack == a)
                      & np.isclose(d.eps, 0 if a == "clean" else 0.1)].mcc
                cells.append(f"{q.mean():.3f}\\,$\\pm$\\,{q.std():.3f}")
            lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        lines.append("\\addlinespace")
    write("tab_main_trades.tex", lines[:-1])


def factorial():
    rows = [("transformer", "Group (vanilla)", "9", "sinusoidal code"),
            ("grp_none", "Group", "9", "none"),
            ("grp_learn", "Group", "9", "learned code"),
            ("shared_none", "Shared", "40", "none"),
            ("shared_learn", "Shared", "40", "learned code"),
            ("ft_sin", "Per-feature", "40", "$w_j$, $b_j$ + sinusoidal code"),
            ("ft_nobias", "Per-feature", "40", "$w_j$"),
            ("ft", "Per-feature (FT-style)", "40", "$w_j$, $b_j$"),
            ("ft_learn", "Per-feature", "40", "$w_j$, $b_j$ + learned code")]
    f = T[(T.part == "factorial") & (T.b == "transformer")]
    m = T[(T.part == "rounds") & (T.tag == "main") & (T.method == "trades")
          & (T.a == "ft") & (T.b == "transformer")]
    ref = T[(T.part == "rounds") & (T.tag == "main") & (T.method == "trades")
            & (T.a == "aam_trans") & (T.b == "transformer")]
    lines = []
    for k, tok, n, ident in rows:
        cells = []
        for ds in ["CICIoT2023", "TON_IoT"]:
            for o in ["clean", "pgd", "L"]:
                if k == "transformer":
                    v = ref[(ref.dataset == ds) & (ref.outcome == o)].mean_b.iloc[0]
                    cells.append(f"{v:.3f}")
                    continue
                src = m if k == "ft" else f[f.a == k]
                r = src[(src.dataset == ds) & (src.outcome == o)].iloc[0]
                cells.append(f"{r.mean_a:.3f}{'$^{*}$' if r.p_holm < 0.05 else ''}")
        lines.append(f"{tok} & {n} & {ident} & " + " & ".join(cells) + " \\\\")
        if k in ("grp_learn", "shared_learn"):
            lines.append("\\addlinespace")
    write("tab_factorial.tex", lines)


def loss():
    rounds = [("main", "trades", "TRADES, seeds 1--10"), ("main", "pgdat", "PGD-AT, seeds 1--10"),
              ("gate_confirm", "trades", "TRADES, seeds 11--20"),
              ("gate3_confirm", "trades", "TRADES, seeds 21--30"),
              ("v4_confirm", "trades", "TRADES, seeds 41--50")]
    lines = []
    for a, lab in [("aam_trans", "AAM-TRANS"), ("ft", "FT-style control")]:
        for tag, meth, rl in rounds:
            q = T[(T.part == "rounds") & (T.tag == tag) & (T.method == meth) & (T.a == a)
                  & (T.b == "transformer") & (T.outcome == "L")]
            if q.empty:
                continue
            cells = []
            for ds in DS:
                r = q[q.dataset == ds]
                if r.empty:
                    cells += ["--", "--", "--"]
                    continue
                r = r.iloc[0]
                star = "$^{*}$" if r.p_holm < 0.05 else ""
                cells += [f"{r.mean_b:.3f}", f"{r.mean_a:.3f}",
                          f"{sg(r['diff'])}{star} ({int(r.n - r.wins)}/{int(r.n)})"]
            lines.append(f"{lab} & {rl} & " + " & ".join(cells) + " \\\\")
        lines.append("\\addlinespace")
    write("tab_loss.tex", lines[:-1])


def strong():
    p = R("strong", "results.csv")
    if not os.path.exists(p):
        return
    s = pd.read_csv(p)
    names = [("transformer", "Vanilla Transformer"), ("ft", "FT-style control"),
             ("aam_trans", "AAM-TRANS")]
    lines = []
    for ds in ["CICIoT2023", "TON_IoT"]:
        g = s[s.dataset == ds]
        for i, (k, lab) in enumerate(names):
            r = g[g.backbone == k][["clean", "pgd20", "pgd100r5", "clean_sq", "pgd20_sq",
                                    "square5000"]].mean()
            first = ds.replace("_", "\\_") if i == 0 else ""
            lines.append(f"{first} & {lab} & " + " & ".join(f"{v:.3f}" for v in r) +
                         f" & {len(g[g.backbone == k])} \\\\")
        lines.append("\\addlinespace")
    write("tab_strong.tex", lines[:-1])


if __name__ == "__main__":
    main_table()
    factorial()
    loss()
    strong()
