# Gate as an adversarial-input detector: protocol

Written 2026-09-29, **before** any detection score was computed on any
data. The only prior run of this code is a smoke test (TON_IoT, seed 1,
1 epoch, CPU) to check that it executes; its numbers are not used. Nothing
below is changed after the study starts; any deviation is reported.

## Motivation

The v3 study (PROTOCOL_GATE_V3.md) and the v1/v2 studies found that no
adaptive gate improves adversarial robustness or clean accuracy. This
study tests a different role for the gate, formulated before looking at
any detection data: the v3a perturbation-aware token gate computes, for
every feature, how far its value is from a leave-one-out estimate made
from the other features. An L-inf evasion attack moves many features at
once without respecting their joint structure, so the gate should close
on adversarial inputs more than on clean ones. The question is whether
this signal **detects** adversarial inputs, and whether it does so better
than standard baselines.

## Model and data

- Model: `aam_v3a` exactly as in PROTOCOL_GATE_V3.md, trained with
  PGD-TRADES (eps 0.1, K 7, beta 6, 15 epochs), identical to the main grid.
- Datasets: CICIoT2023, CICIoMT2024, TON_IoT; binary task.
- Seeds: 31-40, never used before. Test split only for scoring; the
  detector has no free parameter fitted on test data.

## Detection scores (fixed in advance)

For an input x, computed from the trained model without any extra fitting:

| name | score (higher = more suspicious) | role |
|---|---|---|
| `gate` | 1 - mean_j gamma_j(x), the mean closure of the v3a token gate | **tested** |
| `residual` | mean_j abs(x_j - x_hat_j), the same leave-one-out residual without the learned gate weights | ablation: does the gate add to the raw residual? |
| `msp` | 1 - max softmax probability of the same model (Hendrycks and Gimpel, 2017) | standard baseline |
| `mahalanobis` | Mahalanobis distance of x to the training mean (covariance from the training split, ridge 1e-3) | input-space baseline |

## Inputs to separate

Clean test records (label 0) against their perturbed versions (label 1):

- `pgd`: worst of CE/CW PGD-20 at eps 0.1 against the v3a model;
- `pgd_constrained`: the same with range and integrality constraints;
- `transfer_tf`: PGD-20 (CE) at eps 0.1 on a standard-trained vanilla
  Transformer surrogate, replayed on the test records;
- `gaussian` (control): Gaussian noise with sigma 0.1; a useful detector
  should score this lower than the attacks;
- `adaptive_l1`, `adaptive_l10` (secondary): a detector-aware white-box
  PGD-20 that maximises CE(x) - lambda * gate(x) with lambda 1 and 10
  (Carlini and Wagner, 2017; Tramer et al., 2020).

Metric: AUROC over all test records. Secondary: AUROC on the subset where
the clean record is correctly classified and the attack flips it
(success-only), and TPR at 5% FPR.

## Hypotheses and tests

Per-seed AUROC values (10 seeds per dataset) are paired across scores.

- **H_det1**: AUROC(gate) > 0.5, for pgd, pgd_constrained and transfer_tf,
  per dataset (9 tests).
- **H_det2**: AUROC(gate) > AUROC(msp), same 9 cells.

One-sided exact sign-flip permutation tests over the 10 seeds; Holm
correction over the 18 tests of H_det1 and H_det2 together.

Secondary (reported, not part of the Holm family): gate vs residual, gate
vs mahalanobis, the Gaussian control, success-only AUROC, TPR@5%FPR and
the adaptive attacks (AUROC and the classifier MCC under attack).

All results are reported whatever their direction. If the adaptive attack
removes the detection signal, the paper says so.

## Code check (before the study)

The smoke test found one bug, fixed before any seed 31-40 run: the adaptive
attack mixed a batch-mean cross-entropy with a batch-sum gate term, so the
gate term dominated and lambda 1 and 10 gave identical inputs. Both terms
are now summed per record.
