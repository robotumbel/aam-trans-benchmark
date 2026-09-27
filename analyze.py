"""
analyze.py: statistics, tables and figures from runs/<tag>/.

Statistical protocol (fixed before the main grid was run):
  * Unit of analysis: the seed. Backbones share the split, the surrogates
    and the transfer examples of a seed, so comparisons are paired.
  * Test: exact sign-flip permutation test on the paired differences
    (two-sided; all 2^n sign patterns when n <= 16, e.g. one dataset x 10
    seeds; 200,000 Monte Carlo patterns for the pooled n = 30).
    It assumes only that the differences are symmetric about zero under H0.
  * Primary family (H1): AAM-TRANS vs the vanilla Transformer, MCC, eps=0.1,
    one test per attack in PRIMARY_ATTACKS, pooling the three datasets
    (dataset x seed = 30 paired differences). Holm correction over the
    family. Per-dataset tests are reported as secondary, Holm-corrected
    within each dataset.
  * Ablations: each single-component ablation vs AAM-TRANS, same design,
    Holm over (ablation x attack).
  * Effect sizes: mean paired difference with a 95 % percentile bootstrap
    CI (10,000 resamples) and d_z = mean / sd of the differences.

Usage:
  python analyze.py --tag main [--tag-mc multiclass] [--fig-dir ../figs_r2]
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PRIMARY_ATTACKS = ["pgd", "square", "transfer_tf", "pgd_constrained"]
ALL_ATTACKS = ["clean", "gaussian", "fgsm", "pgd", "pgd_constrained",
               "square", "transfer_mlp", "transfer_tf"]
DATASETS = ["CICIoT2023", "CICIoMT2024", "TON_IoT"]
ABLATIONS = ["aam_noGate", "aam_noTok", "aam_noPE", "aam_noAux"]
LABEL = {"transformer": "Vanilla Transformer", "aam_trans": "AAM-TRANS",
         "aam_noGate": "w/o gate", "aam_noTok": "w/o feature tokeniser",
         "aam_noPE": "w/o learned PE", "aam_noAux": "w/o reconstruction head",
         "vanilla_gate": "Vanilla + gate", "mlp": "MLP", "lstm": "LSTM"}
ATK_LABEL = {"clean": "Clean", "gaussian": "Gaussian", "fgsm": "FGSM",
             "pgd": "PGD-20 (worst of CE/CW)", "pgd_constrained": "Constrained PGD",
             "square": "Square (500 q)", "transfer_mlp": "Transfer (MLP surr.)",
             "transfer_tf": "Transfer (Transformer surr.)"}
COLORS = {"aam_trans": "#2a78d6", "transformer": "#eb6834", "mlp": "#1baf7a",
          "lstm": "#eda100", "aam_noGate": "#e87ba4", "aam_noTok": "#008300",
          "aam_noPE": "#4a3aa7", "aam_noAux": "#e34948", "vanilla_gate": "#52514e"}


# ── statistics ──────────────────────────────────────────────────────────────
def signflip_p(d, max_exact=16, n_mc=200_000, seed=0):
    """Two-sided sign-flip permutation p-value for mean(d) = 0. Exact over
    all 2^n sign patterns for n <= max_exact, Monte Carlo otherwise (the
    observed pattern is counted, so p >= 1/(n_mc+1))."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0 or np.allclose(d, 0):
        return 1.0
    obs = abs(d.mean()) - 1e-12
    if n <= max_exact:
        signs = np.array(list(itertools.product([1, -1], repeat=n)), np.int8)
        return float(np.mean(np.abs(signs @ d / n) >= obs))
    rng = np.random.default_rng(seed)
    hits, done = 0, 0
    while done < n_mc:
        k = min(50_000, n_mc - done)
        signs = rng.choice(np.array([1, -1], np.int8), size=(k, n))
        hits += int((np.abs(signs @ d / n) >= obs).sum())
        done += k
    return (hits + 1) / (n_mc + 1)


def boot_ci(d, n=10_000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired(df, a, b, attack, eps, metric="mcc", datasets=DATASETS, method="trades"):
    q = df[(df.attack == attack) & (np.isclose(df.eps, eps)) &
           (df.method == method) & (df.dataset.isin(datasets))]
    pa = q[q.backbone == a].set_index(["dataset", "seed"])[metric]
    pb = q[q.backbone == b].set_index(["dataset", "seed"])[metric]
    j = pa.to_frame("a").join(pb.to_frame("b"), how="inner").dropna()
    return (j.a - j.b).values, j


def compare(df, a, b, attacks, eps=0.10, metric="mcc", method="trades"):
    rows = []
    for atk in attacks:
        e = 0 if atk == "clean" else eps
        for scope in ["pooled"] + DATASETS:
            dss = DATASETS if scope == "pooled" else [scope]
            d, _ = paired(df, a, b, atk, e, metric, dss, method)
            if len(d) == 0:
                continue
            lo, hi = boot_ci(d)
            sd = d.std(ddof=1) if len(d) > 1 else np.nan
            rows.append(dict(a=a, b=b, method=method, attack=atk, scope=scope,
                             metric=metric, n=len(d), mean_diff=d.mean(),
                             ci_lo=lo, ci_hi=hi,
                             d_z=d.mean() / sd if sd and sd > 0 else np.nan,
                             wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                             p=signflip_p(d)))
    return pd.DataFrame(rows)


# ── loading ────────────────────────────────────────────────────────────────
def load(tag):
    d = os.path.join(HERE, "runs", tag)
    df = pd.read_csv(os.path.join(d, "results.csv"))
    js = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        with open(f) as fh:
            j = json.load(fh)
        js[(j["dataset"], j["mode"], j["method"], j["backbone"], j["seed"])] = j
    return df, js


def summary(df):
    g = df.groupby(["dataset", "mode", "method", "backbone", "attack", "eps"])
    s = g[["acc", "bal_acc", "f1_macro", "mcc", "auroc"]].agg(["mean", "std"])
    s.columns = [f"{m}_{k}" for m, k in s.columns]
    s["n_seeds"] = g.size()
    return s.reset_index()


# ── LaTeX helpers ──────────────────────────────────────────────────────────
def fmt_p(p):
    return "$<\\!10^{-3}$" if p < 1e-3 else f"{p:.3f}"


def fmt_ms(m, s):
    return f"{m:.3f}\\,$\\pm$\\,{s:.3f}"


def table_main(summ, out, method="trades", eps=0.10,
               backbones=("transformer", "aam_trans", "mlp", "lstm"),
               attacks=("clean", "pgd", "pgd_constrained", "square",
                        "transfer_tf", "transfer_mlp")):
    lines = []
    for ds in DATASETS:
        dsl = ds.replace("_", "\\_")
        lines.append(f"\\multicolumn{{{len(attacks)+1}}}{{l}}{{\\textit{{{dsl}}}}} \\\\")
        q = summ[(summ.dataset == ds) & (summ["mode"] == "binary") &
                 (summ.method == method)]
        for b in backbones:
            cells = []
            for atk in attacks:
                e = 0 if atk == "clean" else eps
                r = q[(q.backbone == b) & (q.attack == atk) & np.isclose(q.eps, e)]
                cells.append(fmt_ms(r.mcc_mean.iloc[0], r.mcc_std.iloc[0])
                             if len(r) else "--")
            lines.append(f"{LABEL[b]} & " + " & ".join(cells) + " \\\\")
        lines.append("\\addlinespace")
    with open(out, "w") as fh:
        fh.write("\n".join(lines[:-1]) + "\n")


def table_stats(st, out):
    lines = []
    for _, r in st.iterrows():
        lines.append(
            f"{ATK_LABEL[r.attack]} & {r.scope} & {r.n} & {r.mean_diff:+.3f} & "
            f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & {r.d_z:.2f} & "
            f"{r.wins}/{r.losses} & {fmt_p(r.p)} & {fmt_p(r.p_holm)} \\\\")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ── figures ────────────────────────────────────────────────────────────────
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8,
        "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.6,
        "grid.linewidth": 0.4, "grid.color": "#dddddd", "lines.linewidth": 1.4,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
    return plt


def fig_forest(st, out, title):
    plt = _mpl()
    st = st[st.scope != "pooled"].copy()
    atks = [a for a in ALL_ATTACKS if a in set(st.attack)]
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(7.0, 0.32 * len(atks) + 0.8),
                             sharey=True)
    for ax, ds in zip(axes, DATASETS):
        q = st[st.scope == ds].set_index("attack").reindex(atks)
        y = np.arange(len(atks))[::-1]
        ax.axvline(0, color="#888888", lw=0.6)
        ax.errorbar(q.mean_diff, y, xerr=[q.mean_diff - q.ci_lo, q.ci_hi - q.mean_diff],
                    fmt="o", ms=4, color="#2a78d6", ecolor="#2a78d6",
                    elinewidth=1.2, capsize=0)
        ax.set_yticks(y); ax.set_yticklabels([ATK_LABEL[a] for a in atks])
        ax.set_title(ds); ax.grid(axis="x")
        ax.set_xlabel(r"$\Delta$ MCC (AAM-TRANS $-$ vanilla)")
    fig.suptitle(title, fontsize=8, y=1.0)
    fig.savefig(out); plt.close(fig)


def fig_eps(summ, out, method="trades",
            backbones=("aam_trans", "transformer", "mlp", "lstm")):
    plt = _mpl()
    atks = ["pgd", "transfer_tf", "gaussian"]
    fig, axes = plt.subplots(len(atks), len(DATASETS), figsize=(7.0, 4.6),
                             sharex=True)
    for i, atk in enumerate(atks):
        for j, ds in enumerate(DATASETS):
            ax = axes[i, j]
            q = summ[(summ.dataset == ds) & (summ["mode"] == "binary") &
                     (summ.method == method)]
            for k, b in enumerate(backbones):
                c = q[(q.backbone == b) & (q.attack == "clean")]
                r = q[(q.backbone == b) & (q.attack == atk)].sort_values("eps")
                if not len(r) or not len(c):
                    continue
                x = np.r_[0, r.eps.values]
                m = np.r_[c.mcc_mean.values[:1], r.mcc_mean.values]
                s = np.r_[c.mcc_std.values[:1], r.mcc_std.values]
                ax.plot(x, m, marker="osD^"[k], ms=3.5, color=COLORS[b],
                        label=LABEL[b])
                ax.fill_between(x, m - s, m + s, color=COLORS[b], alpha=0.12, lw=0)
            ax.grid(axis="y")
            if i == 0: ax.set_title(ds)
            if j == 0: ax.set_ylabel(f"MCC: {ATK_LABEL[atk].split(' (')[0]}")
            if i == len(atks) - 1: ax.set_xlabel(r"budget $\epsilon$ ($\sigma$ for Gaussian)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out); plt.close(fig)


def fig_gates(js, out, method="trades"):
    plt = _mpl()
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(7.0, 3.0))
    for j, ds in enumerate(DATASETS):
        runs = [v for k, v in js.items() if k[0] == ds and k[1] == "binary"
                and k[2] == method and k[3] == "aam_trans" and v.get("gates_clean")]
        if not runs:
            continue
        gc = np.array([r["gates_clean"]["mean"] for r in runs])   # seeds x L x H
        gp = np.array([r["gates_pgd"]["mean"] for r in runs])
        for i, (g, name) in enumerate([(gc.mean(0), "clean"),
                                       (gp.mean(0) - gc.mean(0), "PGD $-$ clean")]):
            ax = axes[i, j]
            if i == 0:
                im = ax.imshow(g, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            else:
                v = max(1e-3, np.abs(g).max())
                im = ax.imshow(g, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
            for (a, b), val in np.ndenumerate(g):
                ax.text(b, a, f"{val:.2f}" if i == 0 else f"{val:+.3f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if (i == 0 and val > 0.6) else "black")
            ax.set_xticks(range(g.shape[1]))
            ax.set_xticklabels([f"h{h+1}" for h in range(g.shape[1])])
            ax.set_yticks(range(g.shape[0]))
            ax.set_yticklabels([f"block {l+1}" for l in range(g.shape[0])])
            ax.set_title(f"{ds}: gate, {name}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_confusion(js, out, backbone="aam_trans", method="trades"):
    """Row-normalised confusion matrices (multiclass, summed over seeds), with
    the mean per-seed test support after each class name. CICIoT2023 (34
    classes) gets the left half of the figure; the other two share the right."""
    plt = _mpl()
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.45, 1], wspace=0.55, hspace=0.45)
    axes = {"CICIoT2023": fig.add_subplot(gs[:, 0]),
            "CICIoMT2024": fig.add_subplot(gs[0, 1]),
            "TON_IoT": fig.add_subplot(gs[1, 1])}
    im = None
    for ds, ax in axes.items():
        runs = [v for k, v in js.items() if k[0] == ds and k[1] == "multiclass"
                and k[2] == method and k[3] == backbone]
        if not runs:
            ax.axis("off"); continue
        cm = sum(np.array(r["results"]["clean@0"]["confusion"]) for r in runs)
        sup = cm.sum(1, keepdims=True)
        im = ax.imshow(cm / np.maximum(sup, 1), cmap="Blues", vmin=0, vmax=1)
        n = len(cm)
        names = runs[0]["class_names"]
        fs = 4.6 if n > 20 else 5.2
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(range(n), fontsize=fs - 0.6)
        ax.set_yticklabels([f"{names[i][:22]} ({int(round(sup[i, 0] / len(runs)))}) {i}"
                            for i in range(n)], fontsize=fs)
        ax.tick_params(length=1.5, pad=1)
        ax.set_title(ds, fontsize=7.5); ax.set_xlabel("predicted class index", fontsize=6)
    if im is not None:
        cax = fig.add_axes([0.93, 0.25, 0.012, 0.5])
        cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=6)
        cb.set_label("per-class recall (row-normalised)", fontsize=6)
    fig.savefig(out); plt.close(fig)


def per_class_table(js, out_csv, backbone="aam_trans", method="trades"):
    rows = []
    for ds in DATASETS:
        runs = [v for k, v in js.items() if k[0] == ds and k[1] == "multiclass"
                and k[2] == method and k[3] == backbone]
        if not runs:
            continue
        names = runs[0]["class_names"]
        for atk in ("clean@0", "pgd@0.1"):
            pc = [r["results"][atk]["per_class"] for r in runs if atk in r["results"]]
            for i, c in enumerate(names):
                rows.append(dict(
                    dataset=ds, attack=atk, cls=c,
                    support=np.mean([p["support"][i] for p in pc]),
                    recall=np.mean([p["recall"][i] for p in pc]),
                    recall_sd=np.std([p["recall"][i] for p in pc], ddof=1),
                    precision=np.mean([p["precision"][i] for p in pc]),
                    f1=np.mean([p["f1"][i] for p in pc]),
                    never_predicted_runs=int(sum(p["n_predicted"][i] == 0 for p in pc)),
                    n_runs=len(pc)))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return pd.DataFrame(rows)


def main(a):
    fig_dir = os.path.abspath(a.fig_dir)
    tab_dir = os.path.join(HERE, "runs", a.tag, "tables")
    os.makedirs(fig_dir, exist_ok=True); os.makedirs(tab_dir, exist_ok=True)
    df, js = load(a.tag)
    summ = summary(df)
    summ.to_csv(os.path.join(tab_dir, "summary.csv"), index=False)

    # Primary family.
    st = compare(df, "aam_trans", "transformer", ALL_ATTACKS)
    prim = (st.scope == "pooled") & st.attack.isin(PRIMARY_ATTACKS)
    st["family"] = np.where(prim, "primary", "secondary")
    st["p_holm"] = np.nan
    st.loc[prim, "p_holm"] = holm(st.loc[prim, "p"])
    for ds in DATASETS:
        m = (st.scope == ds) & st.attack.isin(PRIMARY_ATTACKS)
        if m.any():
            st.loc[m, "p_holm"] = holm(st.loc[m, "p"])
    st.to_csv(os.path.join(tab_dir, "stats_aam_vs_vanilla.csv"), index=False)
    table_stats(st[prim | (st.attack == "clean") & (st.scope == "pooled")],
                os.path.join(tab_dir, "tab_stats_primary.tex"))
    table_stats(st[(st.scope != "pooled") & st.attack.isin(PRIMARY_ATTACKS)],
                os.path.join(tab_dir, "tab_stats_perdataset.tex"))
    print(st[st.scope == "pooled"][["attack", "n", "mean_diff", "ci_lo", "ci_hi",
                                     "d_z", "wins", "losses", "p", "p_holm"]]
          .to_string(index=False))

    # Ablations (only if present).
    abl = [b for b in ABLATIONS + ["vanilla_gate"] if b in set(df.backbone)]
    if abl:
        parts = []
        for b in abl:
            ref = "transformer" if b == "vanilla_gate" else "aam_trans"
            parts.append(compare(df, ref, b, ["clean"] + PRIMARY_ATTACKS))
        sa = pd.concat(parts)
        m = sa.scope == "pooled"
        sa["p_holm"] = np.nan
        sa.loc[m, "p_holm"] = holm(sa.loc[m, "p"])
        sa.to_csv(os.path.join(tab_dir, "stats_ablation.csv"), index=False)
        print("\nAblations (reference minus ablation, pooled):")
        print(sa[m][["a", "b", "attack", "mean_diff", "ci_lo", "ci_hi", "p",
                     "p_holm"]].to_string(index=False))

    # Second AT method (only if present).
    if "pgdat" in set(df.method):
        sp = compare(df, "aam_trans", "transformer", ["clean"] + PRIMARY_ATTACKS,
                     method="pgdat")
        m = sp.scope == "pooled"
        sp["p_holm"] = np.nan
        sp.loc[m, "p_holm"] = holm(sp.loc[m, "p"])
        sp.to_csv(os.path.join(tab_dir, "stats_pgdat.csv"), index=False)

    table_main(summ, os.path.join(tab_dir, "tab_main_trades.tex"))
    if "pgdat" in set(df.method):
        table_main(summ, os.path.join(tab_dir, "tab_main_pgdat.tex"),
                   method="pgdat", backbones=("transformer", "aam_trans"))
    fig_forest(st, os.path.join(fig_dir, "fig_forest.pdf"),
               "Paired MCC difference, 95% bootstrap CI over seeds")
    fig_eps(summ, os.path.join(fig_dir, "fig_eps.pdf"))
    fig_gates(js, os.path.join(fig_dir, "fig_gates.pdf"))

    if a.tag_mc:
        dfm, jsm = load(a.tag_mc)
        summary(dfm).to_csv(os.path.join(tab_dir, "summary_multiclass.csv"),
                            index=False)
        fig_confusion(jsm, os.path.join(fig_dir, "fig_confusion.pdf"))
        pc = per_class_table(jsm, os.path.join(tab_dir, "per_class.csv"))
        print("\nClasses never predicted in >= 1 run (clean):")
        print(pc[(pc.attack == "clean@0") & (pc.never_predicted_runs > 0)]
              .to_string(index=False))
        smc = compare(dfm.assign(mode="binary"), "aam_trans", "transformer",
                      ["clean"] + PRIMARY_ATTACKS)
        smc.to_csv(os.path.join(tab_dir, "stats_multiclass.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--tag-mc", default=None)
    ap.add_argument("--fig-dir", default=os.path.join(HERE, "..", "figs_r2"))
    main(ap.parse_args())
