# v5: FT-style control on the original seeds, tokeniser x position factorial, stronger attacks

Written 2026-10-04, **before** any of the runs below was started. It follows
an internal review of the v4 manuscript, which raised four points: (i) the
FT-style control exists only for seeds 41-50, TRADES, binary; (ii) the vanilla
control changes two things at once (group tokens and a sinusoidal code), so
the mechanism behind the gain is not identified; (iii) part of the "robust"
gain is a clean-accuracy gain; (iv) the white-box attack is PGD-20 and the
Square attack uses 500 queries. Nothing below is changed after the runs
start; any deviation is reported.

## Outcomes

For every model and seed: clean MCC, MCC under worst-of-two PGD-20 at eps 0.1,
and the **robustness-specific loss** L = MCC(clean) - MCC(PGD). A model is
"more robust" in the strict sense only if its L is smaller; a higher PGD MCC
with an equal L is a clean-accuracy gain that persists under attack. The
paired L differences for the existing rounds were computed from the existing
per-seed results on 2026-10-04 before this protocol was written and are
reported as they are.

## A. FT-style control on seeds 1-10

- `ft`, TRADES, CICIoT2023, CICIoMT2024, TON_IoT, binary, full attack suite
  and eps sweep (tag `main`, paired with the existing runs of the same
  seeds; transfer examples are regenerated from re-trained surrogates of
  the same seed).
- `ft`, PGD-AT, CICIoT2023 and TON_IoT (tag `main`).
- `ft`, TRADES, multiclass, three datasets (tag `multiclass`).
- Attacker-controllable-feature attack on the `ft` TRADES checkpoints
  (tag `realistic`).
- Tests (two-sided exact sign-flip over ten seeds, Holm over datasets):
  ft - transformer and aam_trans - ft, for clean MCC, PGD MCC and L.

## B. Tokeniser x position factorial (plain skeleton: no gate, no recon head)

| | no position code | sinusoidal | learned |
|---|---|---|---|
| group tokens (8 groups, shared map) | `grp_none` | `transformer` (existing) | `grp_learn` |
| per-feature tokens (w_j, b_j) | `ft` (part A) | `ft_sin` | `ft_learn` |

Three further cells isolate the feature identity at equal token count (40
tokens, i.e. the compute of AAM-TRANS):
- `ft_nobias`: per-feature weights w_j, no bias b_j, no position code;
- `shared_none`: one shared embedding (token_j = x_j w + b), no position
  code, so a token carries no information about which feature it is;
- `shared_learn`: the same shared embedding plus a learned position code,
  so the identity comes from the position code only.

TRADES, CICIoT2023 and TON_IoT, seeds 1-10, binary, primary budget only
(tag `factorial`). Questions fixed in advance:
1. Do group tokens without the sinusoidal code (`grp_none`, `grp_learn`)
   reach the level of per-feature tokens? If yes, the gain of the paper is
   due to the sinusoidal code of the vanilla control and not to the tokeniser.
2. Does `ft_sin` lose the gain (replication of the `aam_noPE` ablation on
   the plain skeleton)?
3. Does a model with 40 tokens but no feature identity (`shared_none`) lose
   the gain, and does a learned position code restore it (`shared_learn`)?
   This separates the token count and compute from the feature identity.
4. Does the per-feature bias matter (`ft_nobias` vs `ft`)?

Every cell is compared with `transformer` and with `ft` (two-sided exact
sign-flip, Holm over the seven cells within dataset and outcome), for clean
MCC, PGD MCC and L. All cells are reported whatever the direction.

## C. Stronger attacks on saved checkpoints (evaluation only)

Checkpoints of round four (seeds 41-50), `transformer`, `ft`, `aam_trans`,
CICIoT2023 and TON_IoT, eps 0.1, on a fixed random subset of 5,000 test
records per seed (all records if fewer):
- PGD with 100 steps and five random restarts, cross-entropy and CW margin
  losses, per-record worst case over the ten runs;
- PGD-20 (worst of two) on the same subset, for comparison;
- Square with 5,000 queries on the first 1,000 records of the subset, with
  PGD-20 and clean MCC on the same 1,000 records.
Reported: MCC per attack and the paired differences ft - transformer and
aam_trans - transformer. Question: does the ordering of the models and the
size of the difference hold under the stronger attacks?
