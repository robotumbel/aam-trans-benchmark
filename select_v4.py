"""select_v4.py: selection rules of PROTOCOL_V4.md on runs/v4_select/val_scores.csv.

  python select_v4.py combo   # prints the combination to train, or "none"
  python select_v4.py final   # prints the selected model, writes SELECTED_V4.txt
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "runs", "v4_select")
CANDS = ["aam_v4a", "aam_v4b", "aam_v4c", "aam_v4d"]


def scores():
    df = pd.read_csv(os.path.join(DIR, "val_scores.csv"))
    n = df.groupby("backbone").size()
    return df.groupby("backbone").val_pgd_mcc.mean(), n


def main(mode):
    sc, n = scores()
    if mode == "combo":
        top = sc.reindex(CANDS).dropna().sort_values(ascending=False)
        a, b = top.index[:2]
        both = top.iloc[0] > sc["ft"] and top.iloc[1] > sc["ft"]
        print(f"{a}+{b[-1]}" if both else "none")
        return
    pool = [m for m in sc.index if m in CANDS or "+" in m]
    pool = [m for m in pool if n[m] == 6]
    best = sc[pool].sort_values(ascending=False)
    sel = best.index[0]
    with open(os.path.join(DIR, "SELECTED_V4.txt"), "w") as f:
        f.write(f"{sel}\nbeats_ft={best.iloc[0] > sc['ft']}\n"
                f"{sc.sort_values(ascending=False).to_string()}\n")
    print(sel)


if __name__ == "__main__":
    main(sys.argv[1])
