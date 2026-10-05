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


# ======================= v3 and v4 studies ================================
from analyze_detect import holm as holm1, signflip


def flag(label, ok, info=""):
    print(f"{'MATCH   ' if ok else 'MISMATCH'} {label} {info}")
    if not ok:
        bad.append(label)


def pd_diff(df, ds, atk, eps, a, b):
    g = df[(df.dataset == ds) & (df.attack == atk) & np.isclose(df.eps, eps)]
    p = g.pivot_table(index="seed", columns="backbone", values="mcc")
    return (p[a] - p[b]).dropna().values


P4 = ["pgd", "pgd_constrained", "square", "transfer_tf"]
# --- third generation (seeds 21-30)
sel3 = pd.read_csv(R("gate3_select", "val_scores.csv")).groupby("backbone").val_pgd_mcc.mean()
for bb, v in [("aam_v3b", .8567), ("aam_v3c", .8551), ("aam_v3a", .8535), ("aam_trans", .8568)]:
    chk(f"v3 validation {bb}", v, sel3[bb], .00006)
g3 = pd.read_csv(R("gate3_confirm", "results.csv"))
xs = [pd_diff(g3, ds, a, .1, "aam_v3b", "aam_noGate") for a in P4 for ds in DS]
flag("v3 all |diff| <= 0.0014", max(abs(x.mean()) for x in xs) <= .00145,
     f"max {max(abs(x.mean()) for x in xs):.4f}")
ph = holm1([signflip(x) for x in xs])
chk("v3 smallest Holm p", .79, ph.min(), .006)
for ds, v, w in [("CICIoT2023", .043, 10), ("TON_IoT", .022, 10), ("CICIoMT2024", .002, 10)]:
    x = pd_diff(g3, ds, "pgd", .1, "aam_trans", "transformer")
    chk(f"round 3 AAM-vanilla {ds}", v, x.mean(), .0006); flag(f"round 3 wins {ds}", (x > 0).sum() == w)
x = pd_diff(g3, "CICIoT2023", "pgd", .2, "aam_trans", "transformer")
chk("round 3 reversal", -.105, x.mean(), .0006); flag("round 3 reversal 0/10", (x > 0).sum() == 0)

# --- detector (seeds 31-40)
det = pd.read_csv(R("detect_v3", "detect.csv"))
tst = pd.read_csv(R("detect_v3", "tests.csv"))
prim = tst[tst.family == "primary"]
flag("detector 15 of 18 significant", int((prim.p_holm < .05).sum()) == 15)
chk("detector Holm p", .018, prim.p_holm.min(), .0006)
au = det.pivot_table(index=["dataset", "attack"], columns="score", values="auroc")
g = au.loc[[(d, a) for d in DS for a in ("pgd", "transfer_tf")], "gate"]
chk("gate AUROC min (PGD, transfer)", .56, g.min(), .005); chk("gate AUROC max", .67, g.max(), .005)
chk("gate TON constrained", .475, au.loc[("TON_IoT", "pgd_constrained"), "gate"], .0006)
m = au.loc[[(d, a) for d in DS for a in ("pgd", "pgd_constrained", "transfer_tf")
            if (d, a) != ("TON_IoT", "pgd_constrained")], "mahalanobis"]
chk("Mahalanobis min", .65, m.min(), .005); chk("Mahalanobis max", .93, m.max(), .005)
c = [(d, a) for d in DS for a in ("pgd", "pgd_constrained", "transfer_tf")]
flag("Mahalanobis > gate in all other cells",
     all(au.loc[k, "mahalanobis"] > au.loc[k, "gate"] for k in c if k != ("TON_IoT", "pgd_constrained")))
gr = (au.loc[c, "gate"] - au.loc[c, "residual"]).abs().max()
flag("gate vs residual about 0.01 or less", gr <= .0105, f"max {gr:.4f}")
tp = det[(det.score == "gate") & det.attack.isin(["pgd", "pgd_constrained", "transfer_tf"])
         ].groupby(["dataset", "attack"]).tpr5.mean()
chk("TPR@5% min", .05, tp.min(), .005); chk("TPR@5% max", .08, tp.max(), .005)
ad = au.xs("adaptive_l10", level="attack")["gate"]
chk("adaptive AUROC min", .27, ad.min(), .005); chk("adaptive AUROC max", .38, ad.max(), .005)

# --- fourth generation and FT control (seeds 41-50)
sel4 = pd.read_csv(R("v4_select", "val_scores.csv")).groupby("backbone").val_pgd_mcc.mean()
for bb, v in [("aam_v4d", .8572), ("aam_noGate", .8571), ("aam_trans", .8568), ("ft", .8553),
              ("aam_v4a", .8548), ("aam_v4b", .8532), ("aam_v4c", .8529)]:
    chk(f"v4 validation {bb}", v, sel4[bb], .00006)
v4 = pd.read_csv(R("v4_confirm", "results.csv"))
xs = [pd_diff(v4, ds, a, .1, "aam_v4d", "ft") for a in P4 + ["transfer_mlp"] for ds in DS]
chk("v4 primary min diff", -.0003, min(x.mean() for x in xs), .00006)
chk("v4 primary max diff", .0028, max(x.mean() for x in xs), .00006)
ph = holm1([signflip(x) for x in xs])
chk("v4 smallest Holm p", .31, ph.min(), .006); flag("v4 none significant", (ph >= .05).all())
for ds, v in [("CICIoT2023", .038), ("TON_IoT", .023)]:
    chk(f"FT-vanilla PGD {ds}", v, pd_diff(v4, ds, "pgd", .1, "ft", "transformer").mean(), .0006)
for ds, v in [("CICIoT2023", .001), ("TON_IoT", -.002)]:
    chk(f"AAM-FT PGD {ds}", v, pd_diff(v4, ds, "pgd", .1, "aam_trans", "ft").mean(), .0006)
for atk, e in [("transfer_tf", .1), ("transfer_mlp", .1), ("clean", 0)]:
    x = pd_diff(v4, "CICIoT2023", atk, e, "aam_noGate", "ft")
    chk(f"noGate-FT {atk} CICIoT", .004, x.mean(), .0006); flag(f"noGate-FT {atk} 10/10", (x > 0).sum() == 10)
chk("AAM-FT transfer_tf CICIoT", .003, pd_diff(v4, "CICIoT2023", "transfer_tf", .1, "aam_trans", "ft").mean(), .0006)
x = pd_diff(v4, "CICIoT2023", "pgd", .2, "ft", "transformer")
chk("FT reversal", -.113, x.mean(), .0006); flag("FT reversal 0/10", (x > 0).sum() == 0)
for ds, v, w in [("CICIoT2023", .040, 10), ("TON_IoT", .021, 10), ("CICIoMT2024", .003, 9)]:
    x = pd_diff(v4, ds, "pgd", .1, "aam_trans", "transformer")
    chk(f"round 4 AAM-vanilla {ds}", v, x.mean(), .0006); flag(f"round 4 wins {ds}", (x > 0).sum() == w)
x = pd_diff(v4, "CICIoT2023", "pgd", .2, "aam_trans", "transformer")
chk("round 4 reversal", -.108, x.mean(), .0006); flag("round 4 reversal 0/10", (x > 0).sum() == 0)

# ======================= v5: FT on seeds 1-10, factorial, strong attacks ====
T = pd.read_csv(R("v5_tests.csv"))     # written by analyze_v5.py


def t(part, tag, method, ds, a, b, outcome):
    q = T[(T.part == part) & (T.tag == tag) & (T.method == method) & (T.dataset == ds)
          & (T.a == a) & (T.b == b) & (T.outcome == outcome)]
    assert len(q) == 1, (part, tag, method, ds, a, b, outcome, len(q))
    return q.iloc[0]


M = lambda ds, a, b, o, meth="trades": t("rounds", "main", meth, ds, a, b, o)
# main comparison with the FT-style control
chk("ft clean CICIoT", .792, M("CICIoT2023", "ft", "transformer", "clean").mean_a, .0006)
chk("ft clean TON", .972, M("TON_IoT", "ft", "transformer", "clean").mean_a, .0006)
for ds, v in [("CICIoT2023", .042), ("TON_IoT", .026)]:
    r = M(ds, "ft", "transformer", "pgd"); chk(f"ft-vanilla PGD {ds}", v, r["diff"], .0006)
    flag(f"ft-vanilla PGD 10/10 {ds}", r.wins == 10)
d_af = [M(ds, "aam_trans", "ft", "pgd")["diff"] for ds in DS]
chk("AAM-ft PGD min", -.0003, min(d_af), .00006); chk("AAM-ft PGD max", .0006, max(d_af), .00006)
flag("AAM-ft PGD none significant", all(M(ds, "aam_trans", "ft", "pgd").p_holm >= .05 for ds in DS))
# loss under attack
r = M("CICIoT2023", "aam_trans", "transformer", "L")
chk("L vanilla CICIoT", .059, r.mean_b, .0006); chk("L AAM CICIoT", .042, r.mean_a, .0006)
La = [-t("rounds", tg, me, "CICIoT2023", "aam_trans", "transformer", "L")["diff"]
      for tg, me in [("main", "trades"), ("main", "pgdat"), ("gate_confirm", "trades"),
                     ("gate3_confirm", "trades"), ("v4_confirm", "trades")]]
chk("L diff AAM min", .017, min(La), .0006); chk("L diff AAM max", .019, max(La), .0006)
Lf = [-t("rounds", tg, me, "CICIoT2023", "ft", "transformer", "L")["diff"]
      for tg, me in [("main", "trades"), ("main", "pgdat"), ("v4_confirm", "trades")]]
chk("L diff FT min", .018, min(Lf), .0006); chk("L diff FT max", .020, max(Lf), .0006)
oth = T[(T.part == "rounds") & (T.outcome == "L") & (T.dataset != "CICIoT2023")]
flag("L <= 0.007 on the other binary tasks", max(oth.mean_a.max(), oth.mean_b.max()) <= .0075,
     f"max {max(oth.mean_a.max(), oth.mean_b.max()):.4f}")
chk("clean lead CICIoT", .025, M("CICIoT2023", "aam_trans", "transformer", "clean")["diff"], .0006)
chk("clean lead TON", .025, M("TON_IoT", "aam_trans", "transformer", "clean")["diff"], .0006)
# factorial
F = lambda ds, a, o: t("factorial", "factorial", "trades", ds, a, "transformer", o)
for a, c, v in [("grp_learn", "CICIoT2023", .710), ("grp_learn", "TON_IoT", .944),
                ("shared_learn", "CICIoT2023", .712), ("shared_learn", "TON_IoT", .952),
                ("shared_none", "CICIoT2023", .252), ("shared_none", "TON_IoT", .851),
                ("ft_nobias", "CICIoT2023", .747), ("ft_nobias", "TON_IoT", .952),
                ("ft_learn", "CICIoT2023", .752), ("ft_learn", "TON_IoT", .966),
                ("ft_sin", "CICIoT2023", .712), ("ft_sin", "TON_IoT", .943)]:
    chk(f"factorial PGD {a} {c}", v, F(c, a, "pgd").mean_a, .0006)
chk("factorial ft PGD TON", .970, M("TON_IoT", "ft", "transformer", "pgd").mean_a, .0006)
flag("shared_learn PGD not different from vanilla",
     all(F(c, "shared_learn", "pgd").p_holm >= .05 for c in ("CICIoT2023", "TON_IoT")))
flag("grp_none worse on CICIoT", F("CICIoT2023", "grp_none", "pgd")["diff"] < 0)
chk("factorial L group", .059, F("CICIoT2023", "grp_learn", "L").mean_a, .0006)
chk("factorial L ft_nobias", .037, F("CICIoT2023", "ft_nobias", "L").mean_a, .0006)
chk("factorial L ft_learn", .040, F("CICIoT2023", "ft_learn", "L").mean_a, .0006)
chk("factorial L shared_learn", .046, F("CICIoT2023", "shared_learn", "L").mean_a, .0006)
# replication rounds
for tg, a, b in [("gate_confirm", .046, .025), ("gate3_confirm", .043, .022), ("v4_confirm", .040, .021)]:
    chk(f"{tg} CICIoT", a, t("rounds", tg, "trades", "CICIoT2023", "aam_trans", "transformer", "pgd")["diff"], .0006)
    chk(f"{tg} TON", b, t("rounds", tg, "trades", "TON_IoT", "aam_trans", "transformer", "pgd")["diff"], .0006)
# PGD-AT with the control
for ds, v, w in [("CICIoT2023", .034, 10), ("TON_IoT", .013, 8)]:
    r = M(ds, "ft", "transformer", "pgd", "pgdat"); chk(f"PGD-AT ft-vanilla {ds}", v, r["diff"], .0006)
    flag(f"PGD-AT ft wins {ds}", r.wins == w)
chk("PGD-AT AAM-ft CICIoT", -.002, M("CICIoT2023", "aam_trans", "ft", "pgd", "pgdat")["diff"], .0006)
chk("PGD-AT AAM-ft TON", .005, M("TON_IoT", "aam_trans", "ft", "pgd", "pgdat")["diff"], .0006)
for meth, v in [("trades", -.116), ("pgdat", -.148)]:
    x = pd_diff(main[(main["mode"] == "binary") & (main.method == meth)], "CICIoT2023", "pgd", .2,
                "ft", "transformer")
    chk(f"ft reversal {meth}", v, x.mean(), .0006); flag(f"ft reversal 0/10 {meth}", (x > 0).sum() == 0)
# multiclass
X = lambda ds, a, b, o: t("multiclass", "multiclass", "trades", ds, a, b, o)
for ds, v, w in [("CICIoT2023", .036, 10), ("CICIoMT2024", .026, 10), ("TON_IoT", .006, 5)]:
    r = X(ds, "ft", "transformer", "pgd"); chk(f"mc ft-vanilla {ds}", v, r["diff"], .0006)
    flag(f"mc ft wins {ds}", r.wins == w)
cl = [X(ds, a, "transformer", "clean")["diff"] for ds in DS for a in ("ft", "aam_trans")]
chk("mc clean lead min", .017, min(cl), .0006); chk("mc clean lead max", .051, max(cl), .0006)
chk("mc L AAM CICIoT", -.015, X("CICIoT2023", "aam_trans", "transformer", "L")["diff"], .0006)
chk("mc L ft CICIoT", -.019, X("CICIoT2023", "ft", "transformer", "L")["diff"], .0006)
r = X("TON_IoT", "aam_trans", "transformer", "L"); chk("mc L AAM TON", .079, r.mean_a, .0006)
chk("mc L vanilla TON", .045, r.mean_b, .0006)
chk("mc L ft TON", .090, X("TON_IoT", "ft", "transformer", "L").mean_a, .0006)
flag("mc AAM-ft none significant", all(X(ds, "aam_trans", "ft", "pgd").p_holm >= .05 for ds in DS))
# stronger attacks
sg = pd.read_csv(R("strong", "results.csv"))
for ds, vf, va in [("CICIoT2023", .037, .038), ("TON_IoT", .022, .021)]:
    p = sg[sg.dataset == ds].pivot_table(index="seed", columns="backbone", values="pgd100r5")
    chk(f"strong ft-vanilla {ds}", vf, (p.ft - p.transformer).mean(), .0006)
    chk(f"strong AAM-vanilla {ds}", va, (p.aam_trans - p.transformer).mean(), .0006)
gm = sg.groupby(["dataset", "backbone"])[["pgd20", "pgd100r5", "pgd20_sq", "square5000"]].mean()
flag("PGD-100 within 0.001 of PGD-20 (means)", (gm.pgd20 - gm.pgd100r5).abs().max() <= .001)
flag("Square-5000 within 0.001 of PGD-20 (means)", (gm.pgd20_sq - gm.square5000).abs().max() <= .001)
# attacker-controllable features, control
rt = real[real.method == "trades"]
ps = []
for ds in DS:
    for e in (.1, .2, .5):
        p = rt[(rt.dataset == ds) & np.isclose(rt.eps, e)].pivot_table(index="seed", columns="backbone", values="mcc")
        x = (p.ft - p.transformer).values
        ps.append(min(1, 2 * min(signflip(x), signflip(-x))))
        flag(f"realistic ft > vanilla {ds} {e}", x.mean() > 0)
flag("realistic ft: 7 of 9 significant", int((holm1(ps) < .05).sum()) == 7)
# error rates under PGD and latency of the control
mb = main[(main["mode"] == "binary") & (main.method == "trades") & (main.attack == "pgd") & np.isclose(main.eps, .1)]
for ds, v, f_ in [("CICIoT2023", 12.9, 10.7), ("TON_IoT", 2.7, 1.5)]:
    g = mb[mb.dataset == ds].groupby("backbone").acc.mean()
    chk(f"error vanilla {ds} (%)", v, 100 * (1 - g.transformer), .06)
    chk(f"error ft {ds} (%)", f_, 100 * (1 - g.ft), .06)
d5 = pd.read_csv(R("deploy_v5.csv")).set_index("backbone").cpu1_b1_med_us / 1e3
chk("latency ft", .90, d5.ft, .006); chk("latency noGate (v5 run)", .90, d5.aam_noGate, .006)
chk("latency vanilla (v5 run)", .65, d5.transformer, .006); chk("latency AAM (v5 run)", 1.12, d5.aam_trans, .006)
chk("ft / vanilla latency", 1.39, d5.ft / d5.transformer, .006)
print("\nMISMATCHES:", bad if bad else "none")
