"""
run.py: experiment grid for the AAM-TRANS benchmark.

For every (dataset, mode, seed) the script
  1. loads the leakage-safe split (data.py),
  2. trains two surrogates with standard training on the same training
     partition but an independent initialisation (an MLP and a vanilla
     Transformer) and pre-computes transfer examples on the test split,
  3. for each (method, backbone): trains the target, evaluates the full
     attack suite on the test split, and writes one JSON per run plus rows
     in results.csv.
Completed runs (JSON present) are skipped, so the script can be resumed.

Examples:
  python run.py --tag main --datasets CICIoT2023 CICIoMT2024 TON_IoT \
      --backbones transformer aam_trans aam_noGate aam_noTok aam_noPE aam_noAux \
      --methods trades --seeds 1 2 3 4 5 6 7 8 9 10
  python run.py --tag smoke --datasets TON_IoT --seeds 1 --epochs 2 --quick
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             roc_auc_score, precision_recall_fscore_support)

import attacks as A
from data import load_split, overlap_check
from models import GATE_REG, build, n_params
from train import predict, train

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 0.10                       # primary evaluation budget (= training budget)
EPS_SWEEP = (0.05, 0.10, 0.20)
SQUARE_N, SQUARE_Q = 2000, 500


def metrics(logits, y, n_classes, full=False):
    p = torch.softmax(logits, 1).numpy()
    yh = p.argmax(1)
    m = dict(acc=accuracy_score(y, yh), bal_acc=balanced_accuracy_score(y, yh),
             f1_macro=f1_score(y, yh, average="macro", zero_division=0),
             mcc=matthews_corrcoef(y, yh))
    try:
        m["auroc"] = (roc_auc_score(y, p[:, 1]) if n_classes == 2 else
                      roc_auc_score(y, p, multi_class="ovr", average="macro",
                                    labels=list(range(n_classes))))
    except ValueError:
        m["auroc"] = float("nan")
    m = {k: float(v) for k, v in m.items()}
    if full:
        pr, rc, f1, sup = precision_recall_fscore_support(
            y, yh, labels=list(range(n_classes)), zero_division=0)
        m["per_class"] = dict(precision=pr.tolist(), recall=rc.tolist(),
                              f1=f1.tolist(), support=sup.tolist(),
                              n_predicted=np.bincount(yh, minlength=n_classes).tolist())
        m["confusion"] = confusion_matrix(y, yh, labels=list(range(n_classes))).tolist()
    return m


def batched(fn, X, y, device, bs=1024):
    out = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).to(device)
        t = torch.from_numpy(y[i:i + bs]).to(device)
        out.append(fn(x, t).cpu())
    return torch.cat(out).numpy()


def gate_stats(model, X, device, bs=2048):
    if not hasattr(model, "gates"):
        return None
    acc = None
    with torch.no_grad():
        for i in range(0, len(X), bs):
            model(torch.from_numpy(X[i:i + bs]).to(device))
            g = model.gates()
            if not g:
                return None
            g = torch.stack(g, 1).cpu()                 # [B, L, H]
            acc = g if acc is None else torch.cat([acc, g])
    return dict(mean=acc.mean(0).tolist(), std=acc.std(0).tolist())


def seed_all(s):
    torch.manual_seed(s); np.random.seed(s); torch.cuda.manual_seed_all(s)


def main(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out_dir = os.path.join(HERE, "runs", a.tag)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")
    new_csv = not os.path.exists(csv_path)
    fh = open(csv_path, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    cols = ["dataset", "mode", "method", "backbone", "seed", "attack", "eps",
            "acc", "bal_acc", "f1_macro", "mcc", "auroc", "valid_rate"]
    if new_csv:
        w.writerow(cols); fh.flush()
    log = lambda m: print(m, flush=True)

    for ds in a.datasets:
        for mode in a.modes:
            for seed in a.seeds:
                todo = [(m, b) for m in a.methods for b in a.backbones
                        if not os.path.exists(os.path.join(
                            out_dir, f"{ds}_{mode}_{m}_{b}_s{seed}.json"))]
                if not todo:
                    continue
                s = load_split(ds, seed, mode)
                C = len(s.class_names)
                ov = overlap_check(s)
                log(f"\n=== {ds} {mode} seed={seed} | train {len(s.Xtr)} "
                    f"test {len(s.Xte)} | classes {C} | overlap {100*ov:.2f}%")
                cons = A.Constraint(s.lo, s.hi, s.is_int, s.mean, s.scale, dev,
                                     s.log_mask)
                d = s.Xtr.shape[1]
                rng = np.random.default_rng(seed)
                sq_idx = np.sort(rng.choice(len(s.Xte), min(SQUARE_N, len(s.Xte)),
                                            replace=False))

                # Surrogates: standard training, independent initialisation.
                transfer = {}
                for sname, sb in (("mlp", "mlp"), ("tf", "transformer")):
                    seed_all(10_000 + seed)
                    sm = build(sb, d, C).to(dev)
                    log(f"  surrogate {sname}")
                    train(sm, s, dev, "standard", epochs=a.sur_epochs,
                          seed=10_000 + seed, log=lambda *_: None)
                    sacc = float((predict(sm, s.Xte, dev).argmax(1).numpy()
                                  == s.yte).mean())
                    for e in (EPS,) if a.quick else EPS_SWEEP:
                        transfer[(sname, e)] = batched(
                            lambda x, t: A.pgd(sm, x, t, e, 20, "ce"),
                            s.Xte, s.yte, dev)
                    log(f"    surrogate {sname} clean acc {sacc:.3f}")
                    del sm

                for method, bb in todo:
                    tag = f"{ds}_{mode}_{method}_{bb}_s{seed}"
                    log(f"  --- {tag}")
                    seed_all(seed)
                    model = build(bb, d, C).to(dev)
                    t0 = time.time()
                    try:
                        hist = train(model, s, dev, method, epochs=a.epochs,
                                     eps=a.train_eps, steps=a.pgd_steps, beta=a.beta,
                                     seed=seed, log=log,
                                     gate_reg=GATE_REG.get(bb),
                                     class_weight=a.class_weight)
                    except FloatingPointError as err:
                        # No JSON is written, so a resumed grid retries it.
                        log(f"    FAILED {tag}: {err}")
                        with open(os.path.join(out_dir, "failures.log"), "a") as ff:
                            ff.write(f"{tag}\t{err}\n")
                        continue
                    t_train = time.time() - t0
                    model.eval()
                    rows, res = [], {}

                    def ev(name, eps, Xa, idx=None, full=False, valid=None):
                        y = s.yte if idx is None else s.yte[idx]
                        lg = predict(model, Xa, dev)
                        m = metrics(lg, y, C, full)
                        if valid is not None:
                            m["valid_rate"] = valid
                        res[f"{name}@{eps}"] = m
                        rows.append([ds, mode, method, bb, seed, name, eps,
                                     m["acc"], m["bal_acc"], m["f1_macro"],
                                     m["mcc"], m["auroc"],
                                     m.get("valid_rate", "")])

                    X0 = torch.from_numpy(s.Xte).to(dev)

                    def vrate(Xa):
                        return float(cons.valid(torch.from_numpy(Xa).to(dev), X0)
                                     .float().mean().item())

                    ev("clean", 0, s.Xte, full=True)
                    sweep = (EPS,) if a.quick else EPS_SWEEP
                    for e in sweep:
                        ev("gaussian", e, batched(lambda x, t: A.gaussian(x, e),
                                                  s.Xte, s.yte, dev))
                        ev("fgsm", e, batched(lambda x, t: A.fgsm(model, x, t, e),
                                              s.Xte, s.yte, dev))
                        Xp = batched(lambda x, t: A.pgd_worst(model, x, t, e, 20),
                                     s.Xte, s.yte, dev)
                        ev("pgd", e, Xp, full=(e == EPS), valid=vrate(Xp))
                        for sname in ("mlp", "tf"):
                            ev(f"transfer_{sname}", e, transfer[(sname, e)])
                    Xc = batched(lambda x, t: A.pgd_worst(model, x, t, EPS, 20,
                                                          cons=cons),
                                 s.Xte, s.yte, dev)
                    ev("pgd_constrained", EPS, Xc, full=True, valid=vrate(Xc))
                    Xs = batched(lambda x, t: A.square(model, x, t, EPS,
                                                       SQUARE_Q),
                                 s.Xte[sq_idx], s.yte[sq_idx], dev)
                    ev("square", EPS, Xs, idx=sq_idx)
                    ev("square_clean_subset", 0, s.Xte[sq_idx], idx=sq_idx)

                    out = dict(
                        dataset=ds, mode=mode, method=method, backbone=bb,
                        seed=seed, epochs=a.epochs, eps=EPS, train_eps=a.train_eps,
                        beta=a.beta,
                        pgd_steps=a.pgd_steps, n_params=n_params(model),
                        n_params_train=n_params(model, inference=False),
                        train_sec=t_train, history=hist,
                        class_names=s.class_names, split_fingerprint=s.fingerprint,
                        test_train_overlap=ov, results=res,
                        gates_clean=gate_stats(model, s.Xte, dev),
                    )
                    # Gate response to adversarial input (primary budget).
                    Xp_eps = batched(lambda x, t: A.pgd(model, x, t, EPS, 20),
                                     s.Xte, s.yte, dev)
                    out["gates_pgd"] = gate_stats(model, Xp_eps, dev)
                    if a.save_models:
                        torch.save(model.state_dict(),
                                   os.path.join(out_dir, f"{tag}.pt"))
                    with open(os.path.join(out_dir, f"{tag}.json"), "w") as jf:
                        json.dump(out, jf)
                    w.writerows(rows); fh.flush()
                    log(f"    done {time.time()-t0:.0f}s | clean mcc "
                        f"{res['clean@0']['mcc']:.3f} | pgd mcc "
                        f"{res[f'pgd@{EPS}']['mcc']:.3f} | square mcc "
                        f"{res[f'square@{EPS}']['mcc']:.3f}")
                    del model
                    torch.cuda.empty_cache()
    fh.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--datasets", nargs="+",
                    default=["CICIoT2023", "CICIoMT2024", "TON_IoT"])
    ap.add_argument("--modes", nargs="+", default=["binary"])
    ap.add_argument("--methods", nargs="+", default=["trades"])
    ap.add_argument("--backbones", nargs="+", default=["transformer", "aam_trans"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--sur-epochs", type=int, default=10)
    ap.add_argument("--pgd-steps", type=int, default=7)
    ap.add_argument("--beta", type=float, default=6.0)
    ap.add_argument("--train-eps", type=float, default=EPS,
                    help="AT budget (evaluation budgets are unchanged)")
    ap.add_argument("--quick", action="store_true", help="primary eps only")
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument("--class-weight", action="store_true",
                    help="class-balanced CE (1/sqrt frequency)")
    main(ap.parse_args())
