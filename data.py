"""
data.py: leakage-safe data pipeline for the AAM-TRANS benchmark.

Protocol (fixed for every dataset, every model, every seed):
  1. Read the source CSV, coerce every feature to numeric (non-numeric, NaN
     and +-inf become 0) and drop configured identifier columns.
  2. Deduplicate: keep one row per distinct feature vector, where two
     vectors are the same if every feature agrees to within DEDUP_RES
     standard deviations. Vectors that occur with more than one label are
     ambiguous and are removed entirely.
     After this step no feature vector can appear in both train and test.
  3. Class-capped pool: keep at most CAP rows per attack class and
     CAP_BENIGN benign rows (all rows of smaller classes); classes with fewer than MIN_CLASS distinct rows are
     dropped and logged, since they cannot be split and evaluated.
     The pool is drawn once with POOL_SEED and cached, so
     every seed and every model sees the same pool.
  4. Per seed: stratified train / val / test split (70 / 10 / 20) of the
     pool. Heavy-tailed non-negative counters are log1p-transformed, then a
     StandardScaler is fit on the training partition only. Feature bounds
     for the constrained attack are also taken from the training partition;
     integrality is enforced in raw units.

The seed therefore controls the split and the weight initialisation; the
pool itself is fixed.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, "data_cache")
# Folder holding the merged CSVs: $AAMTRANS_DATA if set, else ./data.
DATA_DIR = os.environ.get("AAMTRANS_DATA", os.path.join(_HERE, "data"))


def _find(name: str) -> str:
    """Locate a merged CSV in DATA_DIR (or the authors' original layout)."""
    legacy = os.path.dirname(os.path.dirname(_HERE))
    for d in (DATA_DIR, legacy, os.path.join(legacy, "AAM_TRANS_Paper3")):
        if os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    return os.path.join(DATA_DIR, name)


DATASETS = {
    "CICIoT2023": dict(
        path=_find("merged_CICIoT_train_shuffled.csv"),
        benign="BENIGN", drop=[]),
    "CICIoMT2024": dict(
        path=_find("merged_train_ciciomt_shuffled.csv"),
        benign="Benign", drop=[]),
    "TON_IoT": dict(
        path=_find("merged_TONIoT_clean.csv"),
        # src_port is an ephemeral, per-connection value that acts as a
        # near-identifier in the testbed capture; dst_port is kept.
        benign="normal", drop=["src_port"]),
    # Reported only as a documented degenerate case (see audit): ~0.4 % of
    # its rows are distinct, attack classes hold 2-21 distinct vectors.
    "CICIoV2024": dict(
        path=_find("merged_train_binary_shuffled.csv"),
        benign="BENIGN", drop=[]),
}

LABEL = "Label"
CAP = 3000          # rows per attack class in the pool
CAP_BENIGN = 30000  # rows of benign traffic (keeps binary tasks non-degenerate)
LOG_THRESHOLD = 100.0  # log1p features that are >= 0 with max above this
DEDUP_RES = 1e-3    # duplicate resolution, in feature standard deviations
MIN_CLASS = 30      # classes with fewer distinct rows cannot be evaluated
POOL_SEED = 0
SPLIT = (0.70, 0.10, 0.20)


def to_model_space(X, log_mask):
    return np.where(log_mask, np.log1p(np.maximum(X, 0)), X)


def dedup_key(Xt, mu, sd):
    return np.round((Xt - mu) / sd / DEDUP_RES).astype(np.int64)


def _pool_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}_pool_cap{CAP}.npz")


def build_pool(name: str, verbose: bool = True) -> dict:
    """Steps 1-3. Cached to data_cache/<name>_pool_cap<CAP>.npz."""
    path = _pool_path(name)
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}

    spec = DATASETS[name]
    df = pd.read_csv(spec["path"])
    df.columns = [c.strip() for c in df.columns]
    y = df[LABEL].astype(str).str.strip().values
    X = df.drop(columns=[LABEL] + spec["drop"])
    feats = list(X.columns)
    X = (X.apply(pd.to_numeric, errors="coerce")
           .replace([np.inf, -np.inf], np.nan).fillna(0.0)
           .values.astype(np.float64))
    n_raw = len(X)
    del df

    # Heavy-tailed non-negative counters (bytes, packets, durations, rates)
    # are modelled as log1p(x); everything downstream -- deduplication,
    # standardisation, attack budget -- works in this transformed space.
    log_mask = (X.min(0) >= 0) & (X.max(0) > LOG_THRESHOLD)
    Xt = to_model_space(X, log_mask)

    # Deduplicate; drop label-ambiguous vectors. Two rows are duplicates when
    # every transformed feature agrees to within DEDUP_RES standard
    # deviations -- far below the attack budget (0.1 sd), so the model cannot
    # usefully tell them apart.
    key_mu, key_sd = Xt.mean(0), np.where(Xt.std(0) > 0, Xt.std(0), 1.0)
    keys = pd.util.hash_pandas_object(
        pd.DataFrame(dedup_key(Xt, key_mu, key_sd)), index=False).values
    del Xt
    kdf = pd.DataFrame({"k": keys, "y": y})
    n_lab = kdf.groupby("k")["y"].nunique()
    ambiguous = set(n_lab.index[n_lab > 1])
    first = ~kdf.duplicated("k", keep="first").values
    keep = first & ~kdf["k"].isin(ambiguous).values
    X, y = X[keep], y[keep]
    n_distinct = int(first.sum())

    # Class-capped pool; classes too small to split are dropped and logged.
    rng = np.random.default_rng(POOL_SEED)
    idx, dropped = [], {}
    for c in np.unique(y):
        ic = np.flatnonzero(y == c)
        if len(ic) < MIN_CLASS:
            dropped[str(c)] = int(len(ic))
            continue
        cap = CAP_BENIGN if c == spec["benign"] else CAP
        if len(ic) > cap:
            ic = rng.choice(ic, cap, replace=False)
        idx.append(ic)
    idx = np.sort(np.concatenate(idx))
    X, y = X[idx], y[idx]

    # Integer-valued features (for the constrained attack's integrality).
    is_int = np.all(np.isclose(X, np.round(X)), axis=0)

    info = dict(dataset=name, n_raw=n_raw, n_distinct=n_distinct,
                n_ambiguous_vectors=len(ambiguous), n_pool=int(len(X)),
                dropped_classes=dropped,
                class_counts={str(c): int((y == c).sum()) for c in np.unique(y)})
    if verbose:
        print(json.dumps(info, indent=1))
    os.makedirs(CACHE_DIR, exist_ok=True)
    info["log_features"] = [f for f, m in zip(feats, log_mask) if m]
    out = dict(X=X, y=y, features=np.array(feats), is_int=is_int,
               log_mask=log_mask, key_mu=key_mu, key_sd=key_sd,
               info=json.dumps(info))
    np.savez_compressed(path, **out)
    return out


@dataclass
class Split:
    Xtr: np.ndarray; ytr: np.ndarray
    Xva: np.ndarray; yva: np.ndarray
    Xte: np.ndarray; yte: np.ndarray
    class_names: list
    feature_names: list
    lo: np.ndarray            # per-feature lower bound, scaled space
    hi: np.ndarray            # per-feature upper bound, scaled space
    mean: np.ndarray          # scaler mean (model space)
    scale: np.ndarray         # scaler std  (model space)
    is_int: np.ndarray        # integer-valued feature mask (raw units)
    log_mask: np.ndarray      # features modelled as log1p(raw)
    fingerprint: str          # hash of test indices, for the audit trail
    key_te: np.ndarray = None # dedup keys of test rows (overlap audit)
    key_tr: np.ndarray = None


def load_split(name: str, seed: int, mode: str = "binary") -> Split:
    """Step 4. mode = 'binary' (benign vs attack) or 'multiclass'."""
    p = build_pool(name, verbose=False)
    X, y_raw = p["X"], p["y"].astype(str)
    if mode == "binary":
        y = (y_raw != DATASETS[name]["benign"]).astype(np.int64)
        class_names = ["Benign", "Attack"]
    else:
        class_names = sorted(np.unique(y_raw).tolist())
        lut = {c: i for i, c in enumerate(class_names)}
        y = np.array([lut[c] for c in y_raw], dtype=np.int64)

    idx = np.arange(len(X))
    # Stratify on the fine label so every class appears in every partition.
    strat = y_raw
    i_tr, i_te = train_test_split(idx, test_size=SPLIT[2], random_state=seed,
                                  stratify=strat)
    va_frac = SPLIT[1] / (SPLIT[0] + SPLIT[1])
    i_tr, i_va = train_test_split(i_tr, test_size=va_frac, random_state=seed,
                                  stratify=strat[i_tr])

    lm = p["log_mask"].astype(bool)
    Xm = to_model_space(X, lm)
    sc = StandardScaler().fit(Xm[i_tr])
    scale = np.where(sc.scale_ > 0, sc.scale_, 1.0)
    f = lambda A: ((A - sc.mean_) / scale).astype(np.float32)
    lo_m, hi_m = Xm[i_tr].min(0), Xm[i_tr].max(0)

    return Split(
        Xtr=f(Xm[i_tr]), ytr=y[i_tr], Xva=f(Xm[i_va]), yva=y[i_va],
        Xte=f(Xm[i_te]), yte=y[i_te],
        class_names=class_names, feature_names=list(p["features"]),
        lo=f(lo_m[None])[0], hi=f(hi_m[None])[0], log_mask=lm,
        mean=sc.mean_.astype(np.float32), scale=scale.astype(np.float32),
        is_int=p["is_int"].astype(bool),
        fingerprint=hashlib.sha1(np.sort(i_te).tobytes()).hexdigest()[:12],
        key_tr=dedup_key(Xm[i_tr], p["key_mu"], p["key_sd"]),
        key_te=dedup_key(Xm[i_te], p["key_mu"], p["key_sd"]),
    )


def overlap_check(s: Split) -> float:
    """Fraction of test rows sharing a deduplication key with a training row
    (0 by construction, up to rounding at key-cell boundaries)."""
    tr = {r.tobytes() for r in s.key_tr}
    return float(np.mean([r.tobytes() in tr for r in s.key_te]))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    a = ap.parse_args()
    for d in a.datasets:
        build_pool(d)
        s = load_split(d, 1)
        print(f"{d}: train {len(s.Xtr)} val {len(s.Xva)} test {len(s.Xte)} "
              f"| test-in-train overlap {100*overlap_check(s):.2f}%")
