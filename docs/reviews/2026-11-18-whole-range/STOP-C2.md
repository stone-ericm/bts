# STOP: whole-range change set C2 (the unknown-outcome alert after a restart, W22)

Stopped 2026-11-19 22:00 ET by the fresh session (`bts-wr-fresh`) after a second-round Codex BLOCK, as its kickoff requires. **Branch `fix/wr-c2` (head `e897c766`, worktree `~/projects/bts-work/wr-fix-c2`) is NOT merged.** Records: `kickoff-fix-c2.md` (version 3, amendment 1), `brief-contract-check-c2.md`, `contract-check-c2/`, `review-brief-fix-c2.md`, `review-brief-fix-c2-r2.md`, `review-fix-c2-codex/`, `review-fix-c2-carbon/`, `review-fix-c2-codex-r2/`, `review-fix-c2-carbon-r2/`.

## Verdicts
| Round | Codex (the gate) | Carbon |
|---|---|---|
| Contract check (draft 2) | BUILD WITH THESE EDITS (new operator text U7; own per-slate receipt; message-id acknowledgement) | — |
| 1, `dfe774df` | BLOCK, one finding (an own receipt with a wrong `schema_version` still deduplicated) | SIGN, notes and an adopt file (four surviving mutants) |
| 2, `e897c766` | BLOCK, two findings (below) | SIGN (158 receipt variants, no false suppression; 12/12 mutants killed) and an adopt file |

## What held (both reviewers, both rounds)
The silent day is gone: a restart between the attempt marker and the send now pages exactly once per slate on both routes, the pick is never re-sent, the lock, decision bytes, `committed_pick_written`, polling and return values match the base; the own receipt is independent of the cutoff alert in both orders, of unrelated health DMs and of other slates, across midnight and New Year; only a non-blank message id acknowledges; every existing caller of the dispatcher is byte-identical to the base; special and damaged files never hang or raise; interrupts propagate after the durable lock; the operator text (U7) tells the operator to inspect and not to enter or change a pick from the alert. Gates on `dfe774df` (10,191 passed) and `e897c766` (10,235 passed): failing set exactly `~/bts-runs/gate-failed.txt`.

## Why it stopped (Codex round 2, `review-fix-c2-codex-r2/report.md`)
Both findings need an own receipt file whose content no code path writes today:
- **B1.** The dedup predicate checks five fields and ignores everything else, and `json.load` keeps the LAST value of a duplicated key. A receipt with the right five fields plus an extra or contradicting field (`"error": "send_dm returned no usable message id"`), or with `"status":"failed"` followed by `"status":"sent"`, suppresses the alert on both routes. Smallest fix: require exactly the keys the writer emits for `sent`, and reject duplicate members in the opt-in reader (`object_pairs_hook`).
- **B2.** A receipt written by ANOTHER opt-in alert for the same slate and source key (the reviewer's setup: the cutoff refusal sent through the opt-in route; today's cutoff caller uses the default route, so no production path writes it) suppresses C2. Smallest fix: store the SHA-256 of the alert body in the receipt and require it to match; the reviewer's scratch copy did this, sent once against a foreign receipt and still deduplicated genuine C2 acknowledgements. A pre-change receipt is then re-acknowledged once (one repeated alert, never a pick).
Codex N1 (not a block): a very long receipt path can fill the whole 200-character diagnostic and lose the file name and error.

## What a fresh session may do (obligations)
1. **Lead's decision.** The two fixes are small and local to `_dispatch_with_acknowledgement` and `_read_receipt` (`src/bts/health/alert.py`); both reviewers' adopt files and the round-2 probes (`probe_receipt_identity.py`, `probe_receipt_correction.py`) are the tests. Either a ruling that lands `fix/wr-c2` with those two fixes and a fresh confirmation review, or leave W22 open (today on `main`: a restart mid-send is silent until the entry watchdog or 08:30).
2. Known and not fixed whatever is chosen (ledgered in the branch's `CLAUDE.md` entry): no end-of-day second chance; a lost acknowledgement or failed receipt write repeats the alert; an undated start after midnight selects the new slate; a pick transport that returns no id leaves a scoreable `delivered` decision while C2 says not to enter from the alert and the entry watchdog says "Fix it now!".
