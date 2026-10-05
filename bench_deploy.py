"""
bench_deploy.py: inference cost of every backbone, as a deployment proxy
for resource-constrained IoT gateways.

Reports, per backbone (d = 39 input features, binary head):
  * inference parameters and fp32 / dynamic-int8 model size on disk
  * CPU latency, batch 1, ONE thread (median and p95 over 2,000 calls):
    the single-core budget of an ARM-class gateway is the relevant regime
  * CPU throughput, batch 256, one thread and all threads
  * dynamic int8 quantisation (torch.ao, Linear layers): size and latency
  * GPU throughput, batch 1024 (if CUDA is present)
  * approximate multiply-accumulates per sample (analytic, attention +
    linear layers)

This is a desktop-CPU proxy, not a measurement on gateway hardware; the
paper states it as such. Run on the target device with --cpu-only.

Usage:
  python bench_deploy.py [--cpu-only] [--out runs/deploy.csv]
"""
from __future__ import annotations

import argparse
import io
import os
import platform
import time

import numpy as np
import pandas as pd
import torch

from models import BACKBONES, build, n_params, TabTransformer

D, C = 39, 2
NAMES = ["transformer", "ft", "aam_trans", "aam_noGate", "aam_noTok", "aam_noPE",
         "aam_noAux", "mlp", "lstm"]


def size_mb(m):
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    return buf.tell() / 2 ** 20


def macs(m):
    """Multiply-accumulates per sample, counting nn.Linear and attention
    products via forward hooks on a batch of one."""
    total = [0]
    hooks = []

    def lin(mod, inp, out):
        total[0] += inp[0].numel() // inp[0].shape[-1] * mod.in_features * mod.out_features

    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear):
            hooks.append(mod.register_forward_hook(lin))
    with torch.no_grad():
        m(torch.randn(1, D))
    for h in hooks:
        h.remove()
    if isinstance(m, TabTransformer):
        T = m.tok.n_tokens + 1
        dm = m.head.in_features
        total[0] += len(m.blocks) * 2 * T * T * dm          # QK^T and AV
    if m.__class__.__name__ == "LSTMNet":
        h = m.lstm.hidden_size
        total[0] += D * (4 * h * (1 + h) + 4 * h * (h + h))  # 2 layers
    return total[0]


def lat(m, bs, reps, threads, device="cpu"):
    torch.set_num_threads(threads)
    x = torch.randn(bs, D, device=device)
    with torch.inference_mode():
        for _ in range(50):
            m(x)
        ts = []
        for _ in range(reps):
            if device == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            m(x)
            if device == "cuda":
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
    ts = np.array(ts)
    return np.median(ts), np.percentile(ts, 95)


def main(a):
    n_threads = os.cpu_count()
    rows = []
    for name in NAMES:
        torch.manual_seed(0)
        m = build(name, D, C).eval()
        q = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear},
                                                   dtype=torch.qint8)
        r = dict(backbone=name, params=n_params(m), size_fp32_mb=size_mb(m),
                 size_int8_mb=size_mb(q), macs=macs(m))
        med, p95 = lat(m, 1, 2000, 1)
        r.update(cpu1_b1_med_us=1e6 * med, cpu1_b1_p95_us=1e6 * p95)
        med, _ = lat(q, 1, 2000, 1)
        r.update(cpu1_b1_int8_med_us=1e6 * med)
        med, _ = lat(m, 256, 200, 1)
        r.update(cpu1_b256_samples_per_s=256 / med)
        med, _ = lat(m, 256, 200, n_threads)
        r.update(cpuN_b256_samples_per_s=256 / med)
        if torch.cuda.is_available() and not a.cpu_only:
            g = build(name, D, C).cuda().eval()
            med, _ = lat(g, 1024, 200, 1, "cuda")
            r.update(gpu_b1024_samples_per_s=1024 / med)
        rows.append(r)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()},
              flush=True)
    df = pd.DataFrame(rows)
    df["cpu"] = platform.processor() or platform.machine()
    df["threads_all"] = n_threads
    df["torch"] = torch.__version__
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--out", default=os.path.join("runs", "deploy.csv"))
    main(ap.parse_args())
