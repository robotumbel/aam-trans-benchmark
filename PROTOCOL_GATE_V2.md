# Gate v2: selection and confirmation protocol

Written 2026-09-26, **before** any gate-v2 model was trained. Nothing below
is changed after the pilot starts; any deviation is reported in the paper.

## Motivation (from the completed main grid, `runs/main`)

The record-level gate of AAM-TRANS (v1) has no measurable effect:
- AAM minus noGate, pooled PGD ΔMCC = -0.0003 (Holm p = 1.0);
- vanilla + gate minus vanilla = 0.000.

The learned gate values are 0.51–0.72, vary little across records
(SD ≈ 0.08) and shift by less than 0.015 under PGD. The gate is close to a
constant per-head scale, which the output projection W^O absorbs.

## Candidates (fixed in advance)

All candidates replace the v1 gate inside AAM-TRANS and keep every other
component. The gate acts on the **values** of each head, per token, i.e.
per input feature, so a head can stop reading from individual features.

| name | gate | extra loss term |
|---|---|---|
| `aam_g1` | gamma_{h,t} = sigmoid(w_h . h_t + b_h), scales V_{h,t} | none |
| `aam_g2` | as g1 | + 0.01 * mean(gamma) (L1 sparsity) |
| `aam_g3` | as g1 | + 1.0 * mean((gamma(x) - gamma(x*))^2) (gate consistency under the TRADES perturbation) |

The gate bias is initialised to +2 (gamma ≈ 0.88, gate open).

## Selection (validation data only)

- Data: CICIoT2023 and TON_IoT; seeds 1, 2, 3 (the existing splits).
- Training: TRADES, identical to the main grid (15 epochs, eps 0.1, K 7,
  beta 6).
- Score: worst-of-two PGD-20 MCC at eps 0.1 on the **validation** split.
  The test split is not read.
- Models: `aam_g1`, `aam_g2`, `aam_g3` and the reference `aam_trans` (v1).
- Rule: the candidate with the highest mean validation score (over 2
  datasets x 3 seeds) is selected. If no candidate beats `aam_trans` on
  that mean, the best candidate is still carried forward and the paper
  reports that selection did not favour any v2 gate.

## Confirmation (test data, new seeds)

- Seeds 11–20, which were not used anywhere before; datasets CICIoT2023,
  CICIoMT2024 and TON_IoT; TRADES; the same attack suite as the main grid.
- Models: `transformer`, `aam_noGate`, `aam_trans` (v1 gate) and the
  selected v2 candidate.
- Primary hypothesis H_gate: the selected v2 gate improves on no gate,
  i.e. MCC(v2) - MCC(aam_noGate) > 0. Four attacks (PGD, constrained PGD,
  Square, transfer-TF); per-dataset exact sign-flip tests with Holm
  correction over the 3 x 4 = 12 tests.
- Secondary: v2 minus v1; v2 minus vanilla.
- All results are reported whatever their direction, together with the v1
  null result.
