# Adaptive mechanism v3: selection and confirmation protocol

Written 2026-09-28, **before** any v3 model was trained. Nothing below is
changed after the pilot starts; any deviation is reported in the paper.

## Motivation

Two gate designs have no measurable effect on robustness (runs/main,
runs/gate_confirm):
- v1, the record-level head gate: nearly constant (0.51–0.72), so W^O
  absorbs it;
- v2, a per-token value gate with sparsity: selected on validation, null
  on seeds 11–20.

The diagnosis is that neither gate sees a signal of the perturbation, and
v1 acts after the softmax, where a constant scale can be absorbed. The SLR
of the authors (IEEE Access 2026, gap G5 and ORQ5.1) suggests adaptive
attention mechanisms (gating, masking, sparsity) co-trained with
PGD-TRADES.

## Candidates (fixed in advance)

Each candidate replaces the v1 gate of AAM-TRANS and keeps the per-feature
tokeniser, the learned positional encoding and the reconstruction head.

| name | mechanism | why it cannot be absorbed |
|---|---|---|
| `aam_v3a` | **Perturbation-aware token gate.** A leave-one-out linear predictor estimates every feature from the others, x_hat_j = (W x)_j with diag(W) = 0, trained with an MSE term (weight 0.1) on clean inputs. Each feature token is scaled before the encoder by gamma_j = sigmoid(a_j - softplus(b_j) * abs(x_j - x_hat_j)); a_j is initialised to 2 and b_j to 0. | the gate depends on a perturbation signal and changes which features enter attention |
| `aam_v3b` | **Adaptive attention masking.** Per head and key token, log sigmoid(w_h . h_t + b_h) is added to the attention logits before the softmax; w_h is initialised to 0 and b_h to 2. | acts before the softmax, so it changes the attention pattern itself |
| `aam_v3c` | **Adaptive attention sharpness.** Per head and per record, the attention temperature is tau_h(x) = 0.5 + softplus(MLP(mean_t h_t)), initialised to 1; logits are divided by tau_h. | changes how concentrated attention is, input by input |

## Selection (validation data only)

- Data: CICIoT2023 and TON_IoT, seeds 1, 2, 3.
- Training: TRADES, identical to the main grid.
- Score: worst-of-two PGD-20 MCC at eps 0.1 on the **validation** split.
- Models: `aam_v3a`, `aam_v3b`, `aam_v3c` and the reference `aam_trans`.
- Rule: the candidate with the highest mean validation score is selected.

## Confirmation (test data, new seeds)

- Seeds 21–30, never used before.
- Datasets: CICIoT2023, CICIoMT2024, TON_IoT; TRADES; the same attack
  suite and eps sweep (0.05/0.1/0.2) as the main grid.
- Models: `transformer`, `aam_noGate`, `aam_trans` and the selected v3
  candidate.
- Primary hypothesis H_v3: the v3 mechanism improves on no gate, i.e.
  MCC(v3) - MCC(aam_noGate) > 0, for PGD, constrained PGD, Square and
  transfer-TF at eps 0.1. Exact per-dataset sign-flip tests, with Holm
  correction over the 12 tests.
- Secondary: v3 minus v1; v3 minus vanilla; the eps sweep.
- All results are reported whatever their direction, together with v1 and
  v2.
