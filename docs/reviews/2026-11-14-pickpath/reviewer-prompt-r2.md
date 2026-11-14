# Adversarial review, round 2 of 2 (final): the reduction is removed

Same rules as round 1 (`.codex-review/pp-c1-review/prompt-r1.md`): read-only in `/Users/eric/projects/bts-pp-c1`; the change is the UNCOMMITTED diff against HEAD plus `tests/test_pickpath_review_pp2.py`. Your round-1 report is at `.codex-review/pp-c1-review/report-r1.md`.

## What changed since round 1
Your B1 was accepted, and the lead concluded the contract was the defect: three rounds in a row found a way for "reduce a double to its playable leg" to lose that leg. The new contract (`.codex-review/pp-c1/kickoff-2.md`; ruling R13 in `docs/superpowers/plans/2026-11-12-2027-season-roadmap.md` on `main`, quoted in the kickoff) removes the reduction:
- The delivery guard refuses a pick only when NONE of its games is playable (a single on a void game; a double whose games are all void).
- A double with at least one playable leg is delivered unchanged, as the scheduler did before the guard existed (`git show 4199360:src/bts/scheduler.py`).
- Accepted and ledgered costs, not open for re-litigation unless they produce a wrong outcome nobody has named: the DM and decision record name the void leg; the entry checker may report it as not entered; the cutoff still follows the earliest leg even when that leg is void.
The author's handover is `.codex-review/pp-c1/handover-2.md`. Treat its claims, and especially its reference-comparison method (executing the old scheduler source into the imported module), as claims to test.

## What to establish
1. **Your round-1 B1 cases** (earlier void cutoff after a crash; the caught pre-send error; the corrected-status variant): for each, is the outcome now the same as the pre-guard reference `4199360`? Where the reference delivered, does the change deliver? Is any case WORSE than the reference? Run your own probes (`/tmp/test_ppc1_review_r1.py` adapted), not only the author's.
2. **The refusal condition** `len(void_games) == len(_committed_pick_game_pks(daily))`: find a pick for which it refuses a playable pick or delivers a pick with no playable game (same-game doubles, a game missing from the status answer, mixed Postponed/Cancelled/Canceled spellings, a status answer keyed differently).
3. **Versus `main`'s HEAD and versus `4199360`:** any input where this change produces a worse production outcome than BOTH. Any input where HEAD's behaviour was right and is now lost for a reason other than the removed reduction.
4. **Grading and records for a delivered double with one void leg:** both scorers, the reconcile replay, `decision.json`, the shadow and live-forward readers, and the `late_delivery` and `postponed_pick` health checks. Anything that now mis-scores, double-counts, or raises.
5. **False greens** in the rewritten tests, and whether the adopted day-level regressions still execute the paths your probes executed.

## Deliverable
Write `/Users/eric/projects/bts-pp-c1/.codex-review/pp-c1-review/report-r2.md` with: `Verdict` (SIGN, SIGN WITH EDITS, or BLOCK), `Round-1 B1 against the reference`, `Blocking findings`, `Notes`, `False greens`, `What I ran`. Under 130 lines. A BLOCK names a concrete input, the wrong production outcome, file and line, the smallest fix, and how you demonstrated it.

When the file is complete, print one line `PPR2-<first 8 hex characters of the sha256 of the report file>` and then `DONE`.
