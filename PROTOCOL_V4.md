# Adaptive attention v4: designs that target robustness, against FT-Transformer

Written 2026-10-03, **before** any v4 model was trained. The only prior run
of this code is a smoke test (1 epoch, TON_IoT, seed 1) to check that it
executes; its numbers are not used. Nothing below is changed after the
selection starts; any deviation is reported.

## Motivation

The per-feature tokeniser that carries the robustness of AAM-TRANS is the
FT-Transformer design (Gorishniy et al., 2021), and in the thesis (Chapter 4,
undefended training) AAM-TRANS and the FT-Transformer are statistically tied.
Seven explicit attention-modulation designs (v1, v2, v3) gave no gain. A
contribution of AAM-TRANS beyond the FT-Transformer therefore needs a design
that improves robustness **over the FT-Transformer under the same
adversarial training**. Four designs were chosen because each has a
mechanistic reason to reduce the effect of an L-inf perturbation, not
merely to re-weight attention.

## Models

All models share the TabTransformer skeleton of models.py (d_model 96,
4 heads, 3 post-norm blocks, FFN 384, dropout 0.1, CLS pooling) and the
TRADES protocol of the main grid (eps 0.1, K 7, beta 6, 15 epochs).

| name | tokens | position | attention / extra | role |
|---|---|---|---|---|
| `transformer` | 8 groups | sinusoidal | softmax | vanilla control (continuity) |
| `ft` | per feature | none (per-feature bias only) | softmax | **FT-Transformer-style control**, matched skeleton |
| `aam_noGate` | per feature | learned | softmax, recon head | AAM-TRANS without gate |
| `aam_trans` | per feature | learned | softmax, record gate, recon head | AAM-TRANS v1 |
| `aam_v4a` | per feature | learned | **adaptively sparse attention**: alpha-entmax with a learned alpha per head (Correia et al., 2019), alpha in (1, 2), initialised at 1.5; recon head | candidate A |
| `aam_v4b` | per feature | learned | **adaptive feature purification**: leave-one-out linear predictor x_hat (diag 0, MSE weight 0.1 on clean inputs); input replaced by gamma * x + (1 - gamma) * x_hat with gamma_j = sigmoid(a_j - softplus(b_j) abs(x_j - x_hat_j)), a_j = 2, b_j = 0 at init; recon head | candidate B |
| `aam_v4c` | per feature | learned | **Lipschitz (L2-distance) attention** with tied query and key projections, scores = -abs(q_i - k_j)^2 / sqrt(d_k) (Kim et al., 2021); recon head | candidate C |
| `aam_v4d` | per feature | learned | **smooth activation**: GELU instead of ReLU in the FFN (Xie et al., 2020); recon head | candidate D |

None of the candidates has the v1 gate.

## Selection (validation data only)

- Data: CICIoT2023 and TON_IoT, seeds 1, 2, 3, binary.
- Models: `ft`, `aam_noGate`, `aam_trans`, `aam_v4a`, `aam_v4b`, `aam_v4c`,
  `aam_v4d`.
- Score: worst-of-two PGD-20 MCC at eps 0.1 on the **validation** split,
  averaged over the six runs.
- Combination: if the two best candidates both exceed `ft` on this score,
  their combination (both mechanisms in one model) is trained on the same
  six runs and enters the comparison.
- Rule: the candidate (or combination) with the highest score is selected,
  whether or not it exceeds `ft`.

## Confirmation (test data, new seeds)

- Seeds 41-50, never used before; CICIoT2023, CICIoMT2024, TON_IoT; binary;
  TRADES; the attack suite and eps sweep (0.05/0.1/0.2) of the main grid.
- Models: `transformer`, `ft`, `aam_noGate`, `aam_trans` and the selected
  candidate.
- **Primary hypothesis H_v4**: MCC(selected) - MCC(ft) > 0 under worst-of-two
  PGD, constrained PGD, Square, transfer from the Transformer surrogate and
  transfer from the MLP surrogate, at eps 0.1, per dataset. One-sided exact
  sign-flip tests over the ten seeds, Holm correction over the 15 tests.
- Secondary (two-sided, Holm within each comparison over datasets):
  selected - aam_noGate (does the mechanism itself help), aam_noGate - ft
  (do the learned position and recon head help over FT), ft - transformer,
  clean MCC of every comparison, and the eps sweep.
- All results are reported whatever their direction.
