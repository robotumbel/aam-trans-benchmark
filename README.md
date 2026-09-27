# AAM-TRANS: a controlled, leakage-audited evaluation under adversarial training

Code, protocol, per-seed results and trained models for the manuscript
*Does the Backbone Matter under Adversarial Training? A Controlled,
Leakage-Audited Evaluation of an Adaptive-Attention Transformer for IoT
Intrusion Detection* (E. A. Winanto, M. Y. Idris, F. T. Ramadhanti).

AAM-TRANS (Adaptive Attention Mechanism Transformer) is a tabular Transformer
with a per-feature tokeniser, a learned positional encoding, a per-head
attention gate and a reconstruction head. It is compared with a vanilla
Transformer that differs from it only in these four components, under
identical TRADES and PGD adversarial training, on deduplicated CICIoT2023,
CICIoMT2024 and TON_IoT data, with ten seeds per configuration.

## Layout

| file | purpose |
|---|---|
| `preprocess_toniot.py` | TON_IoT: drops the binary label and identifier columns, encodes categorical fields |
| `data.py` | leakage-safe pipeline: log1p of heavy-tailed counters, deduplication, class-capped pool, per-seed 70/10/20 split, train-only scaler and bounds |
| `models.py` | AAM-TRANS, the matched vanilla Transformer, single-component ablations, gate v2 candidates, MLP and LSTM baselines |
| `attacks.py` | FGSM, PGD (CE / CW / worst of both), Square, Gaussian, range-and-integrality constraint projector |
| `train.py` | standard training, PGD-AT, PGD-TRADES, optional class-weighted loss |
| `run.py` | experiment grid, resumable; one JSON per run and one CSV row per attack |
| `analyze.py`, `make_tables.py` | sign-flip permutation tests, Holm correction, bootstrap CIs, tables, figures |
| `PROTOCOL_GATE_V2.md`, `select_gate.py`, `analyze_gate.py` | pre-registered gate v2 study: validation-only selection, confirmation on seeds 11 to 20 |
| `diag_deadclass.py` | separability check for rarely predicted CICIoT2023 classes (kNN, random forest) |
| `eval_realistic.py` | attacker-controllable-feature threat model evaluated on the saved checkpoints |
| `bench_deploy.py` | single-thread CPU latency, int8, model size, MACs |
| `run_all.ps1`, `run_gate.ps1`, `run_cw.ps1` | the exact command sequence used for the reported runs |
| `figs/fig_arch.tex` | TikZ source of the architecture figure |

## Results in this repository

`runs/<tag>/results.csv` has one row per (dataset, mode, method, backbone,
seed, attack, budget); `runs/<tag>/<run>.json` holds, for each run, the
training curve, all metrics, per-class precision/recall/support, confusion
matrices and gate statistics. `runs/main/tables/` contains the statistics and
LaTeX tables used in the manuscript.

| tag | content |
|---|---|
| `main` | binary; TRADES: 9 backbones x 3 datasets x seeds 1 to 10; PGD-AT: vanilla and AAM-TRANS |
| `multiclass`, `multiclass_cw` | multiclass, plain and class-weighted loss |
| `gate_select`, `gate_confirm` | gate v2 selection (validation) and confirmation (seeds 11 to 20) |
| `train_eps02` | TRADES trained at eps 0.2 (CICIoT2023, TON_IoT) |
| `realistic` | attacker-controllable-feature attack on the `main` checkpoints |
| `deploy.csv`, `diag_deadclass.csv` | inference cost; dead-class diagnostics |

Trained checkpoints (`*.pt`, about 730 MB) are attached to the GitHub release
`v1.0` as one zip per tag; unzip them into `runs/<tag>/`.

## Data

The datasets are public: CICIoT2023 and CICIoMT2024 from the Canadian
Institute for Cybersecurity (https://www.unb.ca/cic/datasets/), TON_IoT
(network part, `Train_Test_Network.csv`) from UNSW Canberra. Place the merged
CSVs in `data/` (or point `AAMTRANS_DATA` to another folder):

| dataset | file in `data/` | label column |
|---|---|---|
| CICIoT2023 | `merged_CICIoT_train_shuffled.csv` | `Label` |
| CICIoMT2024 | `merged_train_ciciomt_shuffled.csv` | `Label` |
| TON_IoT | `merged_TONIoT_clean.csv`, made by `python preprocess_toniot.py` from `train_test_network.csv` | `Label` |

`python data.py` builds the cached pools and prints the deduplication
statistics and the test-in-train overlap (0.00 % on every dataset).

## Reproduce

```bash
pip install -r requirements.txt
python data.py
# main benchmark: TRADES and PGD-AT, 10 seeds
python run.py --tag main --methods trades --save-models \
  --backbones transformer aam_trans mlp lstm aam_noGate aam_noTok aam_noPE aam_noAux vanilla_gate
python run.py --tag main --methods pgdat --save-models --backbones transformer aam_trans
# multiclass
python run.py --tag multiclass --modes multiclass --quick --save-models --backbones transformer aam_trans
python analyze.py --tag main --tag-mc multiclass
python make_tables.py
python bench_deploy.py
```

The remaining experiments (gate v2, class-weighted loss, eps 0.2 training,
realistic threat model) are run by `run_gate.ps1`, `run_cw.ps1` and the
commands in the header of each script. Runs were made on one NVIDIA RTX 4050
laptop GPU; one AAM-TRANS run on CICIoT2023 takes about ten minutes.

## Threat model

All attacks are L-infinity evasion attacks in the standardised model space
(budget 0.1 standard deviations unless stated). The constrained variant keeps
every feature inside its training range and integer features integral. The
realistic variant additionally lets the attacker change only features of its
own traffic (timing, rate, TTL, header length, packet counts; sizes only
upwards). Neither is a problem-space (packet-level) attack.
