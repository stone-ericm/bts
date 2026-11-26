# DD-leg tripwire — formal read at the first n ≥ 80 snapshot (P-01)

**Date computed:** 2026-09-22 · **Plan item:** W1.4a · **Register:** X-05 → P-01 (`docs/audit/2026-09-22-exposure-register.md`) · **Evidence:** `docs/audit/2026-09-22-dd-tripwire-formal-read.json` (all numbers below), recipe output `/tmp/slot_dataset_2026.csv` on the box (regenerable)

## Protocol (quoted, unchanged)
- 7/13 (`docs/audit/2026-07-13-dd-p-policy-value-sensitivity.md` L133–135): "Tripwire: at the 7/12 doc's own accumulation checkpoint (~40 more legs, season-to-date n≈80-90), recompute the season-to-date leg gap over pick files (the 7/12 measurement); if it holds ≥10pp, rerun this script and [take the re-solve question seriously]."
- 8/09 binding commitment (`docs/audit/2026-08-09-dd-tripwire-recompute.md`): "the formal read happens at the first measurement snapshot with n≥80 season DD legs …; the ≥10pp threshold and the escalation action are unchanged regardless of tonight's numbers; the formal read cannot be canceled or delayed; no further interim reads before it."
- Measurement = the 7/12 recipe = `scripts/audit/build_slot_dataset.py` semantics: every `data/picks/2026-*.json` (date-stem files only); a slot row exists when `p_game_hit` is present; outcome = `slot_results[slot]` when hit/miss/void; legacy single-pick days fall back to the day result for the primary; legacy DD days without `slot_results` are excluded; exact tails = Poisson-binomial over the stated per-leg probabilities.

## Protocol deviation, recorded
No measurement snapshot was actually taken when n crossed 80 (the crossing date was 2026-09-08; the 9/11 scorecard used a different inclusion rule and is not a P-01 measurement). This read is a **reconstruction** run on 2026-09-22 over the same pick files. Under the recipe the result is deterministic given the pick files, and every DD leg dated ≤ 9/08 had its `slot_results` settled by the 9/09 grading cron, so the reconstruction equals what a 9/08 snapshot would have produced unless a pick file for a date ≤ 9/08 was modified after grading (none is known; W1.1's ledger will confirm from file provenance). Both historical anchors reproduce the published numbers exactly, which is the strongest available check on recipe fidelity.

## Result

| Read | DD legs | realized | stated | shortfall | exact tail | span |
|---|---|---|---|---|---|---|
| **P-01 formal read — first snapshot with n ≥ 80 (2026-09-08)** | **56/80** | **0.700** | **0.743** | **−4.3 pp** | P(≤56) = 0.226 | 5/02 → 9/08 |
| anchor: through 7/12 (7/12 doc: 25/42, .595 vs .734) | 25/42 | 0.595 | 0.734 | −13.9 pp | 0.035 | ✓ reproduces |
| anchor: through 8/09 (8/09 doc: 39/57, .684 vs .740) | 39/57 | 0.684 | 0.740 | −5.6 pp | 0.207 | ✓ reproduces |
| legs 8/10 → 9/08 (after the interim look; not pre-registered, descriptive) | 17/23 | 0.739 | 0.749 | −1.0 pp | 0.536 | |
| season-end update through 9/13 (last contest day; separately labelled) | 59/85 | 0.694 | 0.743 | −4.9 pp | 0.182 | |
| all graded incl. private 9/14–9/18 (research only, not contest) | 62/90 | 0.689 | 0.743 | −5.4 pp | 0.146 | |
| primaries through 9/08 (context) | 81/109 | 0.743 | 0.769 | −2.6 pp | 0.291 | |
| primaries through 9/13 (context) | 85/114 | 0.746 | 0.769 | −2.3 pp | 0.309 | |

Monthly DD legs (hits/legs): May 16/28 · June 7/7 · July 10/16 · Aug 18/21 · Sep 11/18. DD slot rows in pick files: 123; graded under the recipe: 90; excluded as unresolved/void/legacy-DD: 33.

## Disposition
**The ≥ 10 pp escalation trigger did NOT fire (4.3 pp at the pre-registered snapshot).** Per the 7/13 rule the escalation action (rerun `dd_p_policy_value_sensitivity.py`, "take the re-solve question seriously") is **not** invoked by this protocol. The register row X-05 moves to *completed (reconstructed snapshot; trigger not fired)*.

What this does and does not say:
- It closes the pre-registered question: the season-to-date DD-leg shortfall at n=80 is ~4 pp, not ≥ 10 pp. The 7/12 anchor's −13.9 pp was a small-n reading that regressed as legs accumulated (legs after the interim look ran −1.0 pp).
- It does **not** establish DD-leg calibration is fine: the shortfall is consistently negative (all cumulative reads −4 to −6 pp; exact tails 0.15–0.23) and the primaries also run ~2–3 pp under stated. That belongs to the W1.2 bridge / W1.3 "calibration without ranking loss" test, not to this rule.
- Stated probabilities here are the served (estimated-PA) `p_game_hit`; realized outcomes are per-slot production grading. Inclusion is "every pick file with a graded slot", which is broader than "delivered picks" — the same as the 7/12 measurement (so the anchors reproduce), but not the contest record. W1.1 keeps those apart.
- The season-end update (through 9/13) is reported separately and does not change the disposition.

## Reproduce
On the box: `.venv/bin/python scripts/audit/build_slot_dataset.py` (writes `/tmp/slot_dataset_2026.csv`), then the reader in the session scratch (`p01_read.py`, first-80 crossing + Poisson-binomial tails); JSON output committed alongside this memo.
