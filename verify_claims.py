"""verify_claims.py: recompute every number quoted in the manuscript text from
the raw per-seed results and report MATCH / MISMATCH for each claim.
Usage: python verify_claims.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from analyze import compare, holm

H = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.join(H, "runs", *p)
main = pd.read_csv(R("main", "results.csv"))
conf = pd.read_csv(R("gate_confirm", "results.csv"))
e02 = pd.read_csv(R("train_eps02", "results.csv"))
real = pd.read_csv(R("realistic", "results.csv"))
dep = pd.read_csv(R("deploy.csv")).set_index("backbone")
DS = ["CICIoT2023", "CICIoMT2024", "TON_IoT"]
bad = []


def chk(label, claimed, value, tol=0.0015):
    ok = abs(float(claimed) - float(value)) <= tol
    print(f"{'MATCH   ' if ok else 'MISMATCH'} {label}: text {claimed}  data {value:.4f}")
    if not ok:
        bad.append(label)


def mean(df, ds, bb, atk, eps, meth="trades"):
    q = df[(df.dataset == ds) & (df.backbone == bb) & (df.attack == atk)
           & np.isclose(df.eps, eps) & (df.method == meth)]
    return q.mcc.mean()


def diff(df, ds, atk, eps, meth="trades", a="aam_trans", b="transformer"):
    st = compare(df, a, b, [atk], eps=eps, method=meth)
    return st[st.scope == ds].iloc[0]


# --- 6.1 main comparison
for ds, a, v, m in [("CICIoT2023", .793, .768, .775), ("TON_IoT", .972, .947, .959)]:
    chk(f"clean AAM {ds}", a, mean(main, ds, "aam_trans", "clean", 0))
    chk(f"clean vanilla {ds}", v, mean(main, ds, "transformer", "clean", 0))
    chk(f"clean MLP {ds}", m, mean(main, ds, "mlp", "clean", 0))
cm = [mean(main, "CICIoMT2024", b, "clean", 0) for b in ["transformer", "aam_trans", "mlp", "lstm"]]
chk("CICIoMT clean min", .982, min(cm)); chk("CICIoMT clean max", .990, max(cm))
chk("PGD AAM CICIoT", .752, mean(main, "CICIoT2023", "aam_trans", "pgd", .1))
chk("PGD vanilla CICIoT", .710, mean(main, "CICIoT2023", "transformer", "pgd", .1))
chk("AAM loss clean->PGD", .041, .793 - .752, 1e-9)
chk("AAM loss (data)", .041, mean(main, "CICIoT2023", "aam_trans", "clean", 0) - mean(main, "CICIoT2023", "aam_trans", "pgd", .1))
chk("vanilla loss (data)", .058, mean(main, "CICIoT2023", "transformer", "clean", 0) - mean(main, "CICIoT2023", "transformer", "pgd", .1))
pooled = compare(main, "aam_trans", "transformer", ["pgd", "pgd_constrained", "square", "transfer_tf"])
pooled = pooled[pooled.scope == "pooled"]
chk("pooled primary min", .020, pooled.mean_diff.min()); chk("pooled primary max", .023, pooled.mean_diff.max())
print("   pooled Holm p:", holm(pooled.p.values).round(6).tolist())
r = diff(main, "CICIoT2023", "pgd", .1); chk("CICIoT PGD diff", .042, r.mean_diff); chk("CI lo", .039, r.ci_lo); chk("CI hi", .046, r.ci_hi)
print("   CICIoT wins", r.wins, "/", r.wins + r.losses)
r = diff(main, "TON_IoT", "pgd", .1); chk("TON PGD diff", .026, r.mean_diff); chk("TON CI lo", .025, r.ci_lo); chk("TON CI hi", .028, r.ci_hi)
print("   TON wins", r.wins, "/", r.wins + r.losses)
chk("CICIoMT PGD diff", .002, diff(main, "CICIoMT2024", "pgd", .1).mean_diff, .0006)
chk("Square AAM CICIoT", .752, mean(main, "CICIoT2023", "aam_trans", "square", .1))
chk("Square vanilla CICIoT", .710, mean(main, "CICIoT2023", "transformer", "square", .1))
chk("MLP PGD CICIoT", .711, mean(main, "CICIoT2023", "mlp", "pgd", .1))
# --- 6.2 replication
for ds, v in [("CICIoT2023", .046), ("TON_IoT", .025), ("CICIoMT2024", .002)]:
    r = diff(conf, ds, "pgd", .1); chk(f"replication {ds}", v, r.mean_diff, .0006); print("   wins", r.wins, "/", r.wins + r.losses)
# --- 6.3 budgets
chk("AAM PGD eps.2 CICIoT", .534, mean(main, "CICIoT2023", "aam_trans", "pgd", .2))
chk("vanilla PGD eps.2 CICIoT", .636, mean(main, "CICIoT2023", "transformer", "pgd", .2))
r = diff(main, "CICIoT2023", "pgd", .2); chk("diff eps.2", -.101, r.mean_diff); print("   wins", r.wins)
chk("PGD-AT diff eps.2", -.119, diff(main, "CICIoT2023", "pgd", .2, "pgdat").mean_diff)
chk("transfer TRADES eps.2", -.14, diff(main, "CICIoT2023", "transfer_tf", .2).mean_diff, .006)
chk("transfer PGD-AT eps.2", -.18, diff(main, "CICIoT2023", "transfer_tf", .2, "pgdat").mean_diff, .006)
chk("gaussian eps.2", -.03, diff(main, "CICIoT2023", "gaussian", .2).mean_diff, .006)
r = diff(main, "TON_IoT", "pgd", .2); chk("TON eps.2", -.008, r.mean_diff); print("   wins", r.wins)
chk("noTok eps.2", .638, mean(main, "CICIoT2023", "aam_noTok", "pgd", .2))
chk("noPE eps.2", .647, mean(main, "CICIoT2023", "aam_noPE", "pgd", .2))
chk("noGate eps.2", .528, mean(main, "CICIoT2023", "aam_noGate", "pgd", .2))
r = diff(e02, "CICIoT2023", "pgd", .2); chk("train.2 CICIoT eps.2", .024, r.mean_diff); print("   wins", r.wins)
r = diff(e02, "TON_IoT", "pgd", .2); chk("train.2 TON eps.2", .009, r.mean_diff); print("   wins", r.wins)
chk("train.2 AAM PGD eps.2", .689, mean(e02, "CICIoT2023", "aam_trans", "pgd", .2))
chk("train.2 AAM clean", .760, mean(e02, "CICIoT2023", "aam_trans", "clean", 0))
chk("train.2 diff eps.1", .015, diff(e02, "CICIoT2023", "pgd", .1).mean_diff)
# --- 6.4 ablation
for b, lo, hi in [("aam_noTok", .037, .042), ("aam_noPE", .037, .042)]:
    v = diff(main, "CICIoT2023", "pgd", .1, a="aam_trans", b=b).mean_diff
    print(f"{'MATCH   ' if lo - .0015 <= v <= hi + .0015 else 'MISMATCH'} ablation {b} CICIoT in [{lo},{hi}]: {v:.4f}")
for b in ["aam_noTok", "aam_noPE"]:
    v = diff(main, "TON_IoT", "pgd", .1, a="aam_trans", b=b).mean_diff
    print(f"{'MATCH   ' if .023 - .0015 <= v <= .027 + .0015 else 'MISMATCH'} ablation {b} TON in [0.023,0.027]: {v:.4f}")
chk("noTok PGD", .709, mean(main, "CICIoT2023", "aam_noTok", "pgd", .1))
chk("noPE PGD", .715, mean(main, "CICIoT2023", "aam_noPE", "pgd", .1))
mx = max(abs(diff(main, ds, "pgd", .1, a="aam_trans", b=b).mean_diff) for ds in DS for b in ["aam_noGate", "aam_noAux"])
print(f"{'MATCH   ' if mx <= .0015 else 'MISMATCH'} gate/recon PGD |diff| <= 0.001: max {mx:.4f}")
# --- 6.6 PGD-AT
for ds, v in [("CICIoT2023", .032), ("TON_IoT", .018), ("CICIoMT2024", .001)]:
    r = diff(main, ds, "pgd", .1, "pgdat"); chk(f"PGD-AT {ds}", v, r.mean_diff, .0006); print("   wins", r.wins)
# --- 6.7 constrained
chk("constrained AAM CICIoT", .756, mean(main, "CICIoT2023", "aam_trans", "pgd_constrained", .1))
gap = max(abs(mean(main, ds, b, "pgd_constrained", .1) - mean(main, ds, b, "pgd", .1))
          for ds in DS for b in ["transformer", "aam_trans"])
print(f"{'MATCH   ' if gap <= .0055 else 'MISMATCH'} constrained within 0.005 of PGD (attention): max {gap:.4f}")
# --- 6.8 realistic
q = real[(real.method == "trades") & (real.dataset == "CICIoT2023") & (real.backbone == "transformer") & (real.eps == .1)]
chk("realistic vanilla CICIoT eps.1", .751, q.mcc.mean())
t = real[(real.dataset == "TON_IoT") & real.backbone.isin(["transformer", "aam_trans"])]
drop = (t.mcc_clean - t.mcc).groupby([t.method, t.backbone, t.eps]).mean()
chk("TON max drop TRADES", .004, drop.loc["trades"].max(), .0006)
chk("TON max drop PGD-AT", .021, drop.loc["pgdat"].max(), .0006)
st = pd.read_csv(R("realistic", "stats.csv"))
print("   realistic significant cells:", st.groupby("method").apply(lambda g: int((g.p_holm < .05).sum())).to_dict())
# --- 6.10 cost
chk("MACs AAM", 14.2, dep.loc["aam_trans"].macs / 1e6, .05); chk("MACs vanilla", 3.0, dep.loc["transformer"].macs / 1e6, .05)
chk("latency AAM ms", 1.09, dep.loc["aam_trans"].cpu1_b1_med_us / 1e3, .006)
chk("latency vanilla ms", .62, dep.loc["transformer"].cpu1_b1_med_us / 1e3, .006)
chk("ratio 1.75", 1.75, dep.loc["aam_trans"].cpu1_b1_med_us / dep.loc["transformer"].cpu1_b1_med_us, .02)
chk("throughput ratio 5.2", 5.2, dep.loc["transformer"].cpu1_b256_samples_per_s / dep.loc["aam_trans"].cpu1_b256_samples_per_s, .06)
chk("MLP latency", .07, dep.loc["mlp"].cpu1_b1_med_us / 1e3, .006)
chk("sixteen times", 16, dep.loc["aam_trans"].cpu1_b1_med_us / dep.loc["mlp"].cpu1_b1_med_us, .6)
# --- multiclass
mc = pd.read_csv(R("multiclass", "results.csv"))
for ds, v in [("CICIoT2023", .033), ("CICIoMT2024", .026), ("TON_IoT", .016)]:
    r = diff(mc.assign(mode="binary"), ds, "pgd", .1); chk(f"multiclass {ds}", v, r.mean_diff, .0015); print("   wins", r.wins)
print("\nMISMATCHES:", bad if bad else "none")
