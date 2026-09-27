"""
diag_deadclass.py: are the never-predicted CICIoT2023 classes separable
from benign traffic at all? Model-free and model-agnostic checks on the
same leakage-safe split (seed 1..3), no adversarial training:
  * random forest, unweighted and class-balanced: per-class recall
  * 5-nearest-neighbour label purity in standardised feature space
Output: runs/diag_deadclass.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.neighbors import NearestNeighbors

from data import load_split

DEAD = ["BACKDOOR_MALWARE", "BROWSERHIJACKING", "COMMANDINJECTION",
        "RECON-PINGSWEEP", "SQLINJECTION", "UPLOADING_ATTACK", "XSS",
        "RECON-OSSCAN", "VULNERABILITYSCAN"]
rows = []
for seed in (1, 2, 3):
    s = load_split("CICIoT2023", seed, "multiclass")
    names = s.class_names
    ben = names.index("BENIGN")
    for cw in (None, "balanced"):
        rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed,
                                    class_weight=cw).fit(s.Xtr, s.ytr)
        rec = recall_score(s.yte, rf.predict(s.Xte), labels=range(len(names)),
                           average=None, zero_division=0)
        for c in DEAD:
            rows.append(dict(seed=seed, check=f"rf_{cw or 'plain'}", cls=c,
                             value=rec[names.index(c)]))
    nn = NearestNeighbors(n_neighbors=6).fit(s.Xtr)
    for c in DEAD:
        i = names.index(c)
        q = s.Xte[s.yte == i]
        _, idx = nn.kneighbors(q)
        lab = s.ytr[idx[:, :5]]
        rows.append(dict(seed=seed, check="knn5_same_class", cls=c, value=(lab == i).mean()))
        rows.append(dict(seed=seed, check="knn5_benign", cls=c, value=(lab == ben).mean()))
df = pd.DataFrame(rows)
df.to_csv("runs/diag_deadclass.csv", index=False)
print(df.pivot_table(index="cls", columns="check", values="value").round(2).to_string())
