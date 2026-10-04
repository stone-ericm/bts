# W1.4b due read: MDP quality-bin collapse (2026 season)

**Date:** 2026-10-04. **Plan:** W1.4b, row "`mdp_policy_alignment` bin collapse": quantify served p against the boundary distribution by epoch. The plan's caution: a bin histogram is not evidence of useful discrimination.
**Disposition:** **completed (descriptive).** No original pre-registered protocol exists for this read (the register's P-01…P-06 do not cover it). Incident-register records E81 and EX2 route here.
**Exposure:** served probabilities and policy boundaries only; **no outcome was read**, so no register row is needed. The epoch definition (calendar month × the policy regime recorded in the W1.1 ledger) was fixed in code (`4025135`) before the run.

## Inputs
- **Served p:** the W1.1 ledger's `p_stated` for every production selection (152 primary, 122 double-down slots), plus the declined candidate's `p_game_hit` from `decision.json` on the 25 skip days. Inputs are hashed in the output.
- **Boundaries:** the deployed reach-57 policy `data/models/mdp_policy.npz` (sha256 `66d15471…`): 0.7960, 0.8115, 0.8252, 0.8407, giving five quality bins Q0–Q4.
- **Classifier:** the production health check's own `_classify` (`src/bts/health/mdp_policy_alignment.py`).
- **Script:** `scripts/audit/bin_collapse_read.py` at `4025135`. **Output:** box `data/validation/w14b_bin_collapse/4025135.json`.

## What the policy saw

| Slots | Below the lowest boundary (Q0) |
|---|---|
| Primary picks, whole season | **133 / 152 (87.5%)** |
| Double-down legs | **121 / 122** |
| Declined candidates on skip days | **24 / 25** |

By month (primary picks in Q0 / n; all epochs before 9/03 ran the reach-57 policy):
- **March–May:** 62/64. The ledger records no objective field for these dates; the reach-57 policy was the only policy then.
- **June:** 15/23.
- **July:** 15/21.
- **August:** 23/26.
- **September before the tail switch:** 2/2.

The highest quality bin (Q4, p ≥ 0.841) held one primary pick all season.

**The tail period (from 9/03, `emax_season_best`):** 16/16 primaries, 16/16 double-down legs and 9/9 declined candidates sit below 0.796. The tail artifact has one late-season bin by construction (`docs/audit/2026-09-03-emax-tail-policy.md`), so these reach-57 boundaries did not drive those decisions. They are listed only for completeness.

## Reading it
- **The reach-57 policy's quality dimension was nearly inert in 2026.** Almost every decision was taken in Q0, so the same transition row applied regardless of the stated probability, and the action depended mainly on streak, days left and the saver.
- **This repeats, at season scale, an already documented mechanism.** The saved bins were built on actual-PA hindsight profiles: their median was 0.817 against a live median of about 0.765 (CLAUDE.md "PROFILE BASIS"; `docs/audit/2026-06-29-skip-threshold-and-discrimination.md`). That memo also found the 0.796 threshold cosmetic on the estimated-PA backtest.
- **What this read does not show:** it does not show that finer bins would have chosen better, or that the stated probabilities discriminate within Q0. Those are outcome questions. Under D3 they belong to a registered 2027 design (W4 rank 4b, policy-only changes, with independent state and eligibility replay).

## Limits
- Served probabilities are as recorded. Selection effects (which candidate was served) are not modelled.
- The "unknown" objective label means the ledger's decision evidence lacks the field. It is not a third regime.
- Month boundaries are calendar conventions, not documented change points.
