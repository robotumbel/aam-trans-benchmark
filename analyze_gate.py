"""analyze_gate.py: confirmatory analysis of PROTOCOL_GATE_V2.md
(runs/gate_confirm, seeds 11-20, test split)."""
import os
import numpy as np
import pandas as pd
from analyze import compare, holm, DATASETS

ATT = ["pgd", "pgd_constrained", "square", "transfer_tf"]
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "runs", "gate_confirm", "results.csv"))
sel = open(os.path.join(HERE, "runs", "gate_select", "SELECTED.txt")).readline().strip()
out = os.path.join(HERE, "runs", "gate_confirm", "tables")
os.makedirs(out, exist_ok=True)

parts = []
for ref, fam in [("aam_noGate", "primary: v2 - noGate"),
                 ("aam_trans", "secondary: v2 - v1"),
                 ("transformer", "secondary: v2 - vanilla")]:
    st = compare(df, sel, ref, ["clean"] + ATT)
    st = st[st.scope != "pooled"].copy()
    st["family"] = fam
    m = st.attack.isin(ATT)
    st["p_holm"] = np.nan
    st.loc[m, "p_holm"] = holm(st.loc[m, "p"])
    parts.append(st)
# Replication of the main-grid result on the new seeds: v1 vs vanilla, v1 vs noGate.
for a, b in [("aam_trans", "transformer"), ("aam_trans", "aam_noGate")]:
    st = compare(df, a, b, ["clean"] + ATT)
    st = st[st.scope != "pooled"].copy()
    st["family"] = f"replication: {a} - {b}"
    m = st.attack.isin(ATT)
    st["p_holm"] = np.nan
    st.loc[m, "p_holm"] = holm(st.loc[m, "p"])
    parts.append(st)
res = pd.concat(parts)
res.to_csv(os.path.join(out, "stats_gate_confirm.csv"), index=False)
pd.set_option("display.width", 200)
for fam, g in res.groupby("family", sort=False):
    print(f"\n== {fam}")
    print(g[["scope", "attack", "mean_diff", "ci_lo", "ci_hi", "wins", "losses",
             "p", "p_holm"]].round(4).to_string(index=False))
