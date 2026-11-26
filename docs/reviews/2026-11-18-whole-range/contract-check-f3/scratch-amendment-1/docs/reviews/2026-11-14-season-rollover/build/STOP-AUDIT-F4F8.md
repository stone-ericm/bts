# STOP: the audit change set (findings F4 to F8) after three BLOCKs, 2026-11-18

Work policy: three BLOCKs stop the session, never the work. This is the record of obligations and the kickoff for a fresh start. **Nothing from branch `sb-audit-f4f8` is merged.** Worktree `~/projects/bts-work/audit-f4f8` (studio), head `ec136bf0`, base `3d49308d` on `main`; the gate was green at that head (9,221 passed, the 53 known failures).

## What happened
| Round | Commit | Verdict | Findings |
|---|---|---|---|
| Batch review | `153ff8c2` | BLOCK | F4 exception hid a missing-coverage alarm in July; F5 treated last June's outage as an offseason gap; F7 retry could score or publish after an outcome, and could replace verified evidence; F8 dropped the aged-contradiction warning. F6 clean |
| Confirmation | `8cac32ea` | BLOCK | F4 still quiet after a sparse mid-season restore; F5 quiet for a table already critical at the season's close and for an unconfigured season end; F7 verified profiles that had lost a row in conversion |
| Confirmation 2 | `ec136bf0` | BLOCK | F7: a full ten-row capture REWRITTEN to nine rows (both parquets, `row_counts` and `effective_top_n` in the manifest) verifies, where the base rejects it; F5 producer: an empty mid-season refresh writes `ok=true` over a stale export and clears the last-run-failed warning |

Reports and probes: `build/batch-2026-11-18/` (`report.md`, `confirm/`, `confirm2/`). Kickoffs: `kickoff-audit-f4f8.md`, `kickoff-audit-f4f8-fixes.md`. Ruling R39 withdrew the F4 and F5 alarm exceptions.

## Why it kept failing (the lead's reading)
Each fix RELAXED a check (an alarm, a verifier) to remove a false positive, and each round a reviewer found an input where the relaxed check now passed something real. The kickoffs stated the false positive precisely and the "must still catch" side only in general words. Three of the five findings (F4, F5's alarm, F7's retry) were then withdrawn or removed rather than repaired, which is itself the signal: for these, the false alarm is cheaper than the relaxed check.

## State of each finding at `ec136bf0`
- **F6 (research-mode fetch cron sends no DM): found correct by all three reviews.** Commit `ecc977ba`. Ready to land on its own.
- **F8 (shadow statistics scoped to one season, warnings from all seasons) and the P-03 closeout pin: found correct by the last two reviews.** Commits `86a5a58f`, `1446cda8`, `8cac32ea`. Ready to land on their own.
- **F4 (contest-state health check): withdrawn**, byte-identical to the base. The false `CRITICAL` only arises on the automatic path in a season's first two days; the opening days run on dated observations, where the check is quiet. Expected alarm, to be noted in the checklist.
- **F5 alarm (park-drag freshness): withdrawn**, byte-identical to the base. Expected `CRITICAL` on the first nights of a season until the first refresh sees a game. To be noted in the checklist.
- **F5 producer (empty store): NOT ready.** Commit `46e16b67`. Open finding: when both stores are empty and an export or manifest already exists, the refresh must fail (`ProducerError`, `ok=false`) instead of succeeding with zero rows; a clean seed with no export is the only case that may succeed empty.
- **F7 (thin capture): NOT ready.** Commits `5277dd70`, `043dbd02`, `ec136bf0`. Open finding: the verifier takes the lower row count from the same manifest it is checking. The reviewer's direction: keep the configured row floor (`--expected-top-n`) unless an INDEPENDENTLY recorded export-time scored count authorizes the lower one (something the capture wrapper writes at export and the verifier reads from outside the artifact directory, or binds by hash). Also: a legacy manifest with a non-string `run_kind` must fail its checks, not raise (`artifacts.py`, the new membership test).

## Obligations carried forward
1. Land F6, F8 and the P-03 pin as their own small change set (cherry-pick the four commits onto current `main`, gate, daily batch review). They need no further design.
2. Decide F7 afresh, with something material changed: either the independent export-time count described above, designed FIRST as a short contract with the "must still reject" cases written as a table (truncated parquet; rewritten manifest; rows lost in conversion; legacy captures), or leave the capture as on the base and accept the audit's wedge on a nine-batter Opening Night (a permanent capture alarm for that date, evidence lost for one day; ruling R18 already ledgers the thin early lock).
3. Decide the F5 producer afresh with the reviewer's rule (fail when an export already exists), or leave it as on the base and keep checklist row B7's reseed instruction as the mitigation.
4. Add to `docs/ops/2027-season-start.md`: the two expected alarms (R39), in the row for the opening days.
5. The reviewers' probes for F6 and F8 (`build/batch-2026-11-18/review_probes.py`, `confirm/confirmation_probes.py`) are to be adopted as tests when those land.

## Kickoff for the fresh session
Read this file, then `build/batch-2026-11-18/confirm2/report.md`. Do obligation 1 first (mechanical). For obligations 2 and 3, do NOT reopen the branch's F5 and F7 commits: start from `main`, and before any code write the table of inputs that must still be rejected, have that table checked by a fresh reviewer (as the season-boundary contracts were), and only then commission an author. If the table cannot be made complete, choose "leave as on the base" and ledger it.
