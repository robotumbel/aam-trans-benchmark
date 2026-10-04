"""make_tables_v4.py: Table for the v4 study (PROTOCOL_V4.md).

Reads runs/v4_confirm/tests_primary.csv and tests_secondary.csv (written by
analyze_v4.py) and writes runs/main/tables/tab_v4.tex.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "main", "tables")
DS = ["CICIoT2023", "CICIoMT2024", "TON_IoT"]
ANAME = {"clean": "Clean", "pgd": "PGD", "pgd_constrained": "Constrained PGD",
         "square": "Square", "transfer_tf": "Transfer (Transf.)",
         "transfer_mlp": "Transfer (MLP)"}
CNAME = {"aam_v4d-ft": "GELU (v4) $-$ FT", "aam_noGate-ft": "AAM w/o gate $-$ FT",
         "aam_trans-ft": "AAM-TRANS $-$ FT", "ft-transformer": "FT $-$ vanilla",
         "aam_trans-transformer": "AAM-TRANS $-$ vanilla"}


def cell(diff, wins, n, star):
    s = "0.0000" if abs(diff) < 5e-5 else f"{diff:+.4f}".replace("-", "$-$")
    return f"{s}{'$^{*}$' if star else ''} ({wins}/{n})"


def main():
    d = os.path.join(HERE, "runs", "v4_confirm")
    prim = pd.read_csv(os.path.join(d, "tests_primary.csv"))
    sec = pd.read_csv(os.path.join(d, "tests_secondary.csv"))
    rows = []
    for att in ["pgd", "pgd_constrained", "square", "transfer_tf", "transfer_mlp"]:
        g = prim[prim.attack == att].set_index("dataset")
        rows.append(f"{CNAME['aam_v4d-ft']} & {ANAME[att]} & " + " & ".join(
            cell(g.loc[ds, "diff"], g.loc[ds, "wins"], g.loc[ds, "n"],
                 g.loc[ds, "p_holm"] < 0.05) for ds in DS) + r" \\")
    for cmp in ["aam_noGate-ft", "aam_trans-ft", "ft-transformer", "aam_trans-transformer"]:
        rows.append(r"\midrule")
        for att in ["clean", "pgd", "transfer_tf", "transfer_mlp"]:
            eps = 0.0 if att == "clean" else 0.1
            g = sec[(sec.cmp == cmp) & (sec.attack == att) & ((sec.eps - eps).abs() < 1e-9)
                    ].set_index("dataset")
            rows.append(f"{CNAME[cmp]} & {ANAME[att]} & " + " & ".join(
                cell(g.loc[ds, "diff"], g.loc[ds, "wins"], g.loc[ds, "n"],
                     g.loc[ds, "p_holm"] < 0.05) for ds in DS) + r" \\")
    with open(os.path.join(OUT, "tab_v4.tex"), "w") as f:
        f.write("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
