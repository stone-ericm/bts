# Verdict

**BLOCK** `aa61bb43` against `c2039541`. Reconcile can silently save a wrong streak and saver after dropping an entered pick. It can also suppress a valid pre-cutoff scoring correction.

## Blocking findings

1. **A scheduler restart erases the only record of an entered private pick.** `bts preview` saves April 18 with `chosen_on.required=false`. Private fallback locks it and the operator enters it. The scheduler stops after the lock save and before the scoreable decision. A same-day restart saves `pick_locked=false` and stops before the first check. A restore with a malformed lock that `load_state` quarantines produces the same replay result. `src/bts/scheduler.py:1262–1274`, `:2946–2957`, `:1045–1070` and `src/bts/picks.py:947–961`. The restart control preserved the previous lock and the quarantine control held on its archived state file. Both retained the streak and named April 18. [Reproduction and controls](writers-verified.log)
2. **A non-scoreable decision or a preview stamp can survive an accepted send.** The scheduler sends an April 18 DM after recording an attempt. The transport accepts it, the operator enters it, and a response timeout clears the local attempt. EOD then writes `action=skip, scoreable=false` from the earlier skip candidate with no scoreable decision or lock. With a reused preview under non-live config, the stamp alone suffices even without EOD. A fresh selection in non-live `orchestrate` also writes `required=false`, then a post can land just before interruption ahead of its marker save. `src/bts/scheduler.py:986–1010`, `:1315–1335`, `src/bts/orchestrator.py:307–308`, `:431–465` and `src/bts/picks.py:936–961`. Preserve the attempt on ambiguous responses and write one before orchestrator posts, including catch-up. Both controls blocked with dated reasons. Existing ambiguous records need confirmed non-entry before replay can skip them. [Reproduction and controls](writers-verified.log)
3. **Terminal compatibility stops at the correction reader.** April 19 is a delivered hit with a numeric-string `batter_id` in a format the export/regeneration paths preserve. Nine earlier hits and a corrected final boxscore at 02:00 ET require `(0, True)`. The base grades and saves the miss. `src/bts/picks.py:878–902`, `:1585–1589` and `:1625–1629` reject the terminal pick before grading and the candidate saves the stale `(10, True)` without warning. Both guarded correction reads should use the legacy decoder for gradable terminal picks. A control that retained the regular-file read and restored that decoder applied the real boxscore correction and saved `(0, True)`. [Reproduction and control](writers-verified.log)

In findings 1 and 2, nine earlier hits, an entered April 18 hit and an April 19 hit-to-miss correction give the independent result `(10, False)`. The base applies the correction, reports replay unavailable in its receipt and preserves `(10, True)` byte for byte. The candidate applies the correction and saves `(0, True)` with no warning. The entrant and transport outcomes were synthetic. Producer writes, grading and reconcile ran through repository code. [Full traces](writers-verified.log)

## Notes

- Retaining an existing streak on a block can preserve a stale saver until the entered pick is recovered. The fix controls demonstrate retention and a dated diagnostic.
- An unstamped old preview with no decision still blocks as accepted by Amendment 3. A corrupt retained streak with a blocked replay still has receipt `outcome=completed`, with `replay.state=unavailable` and exit 1.
- No tracked files changed. The real `data/picks` tree was absent, so the deployed comparison was verified over synthetic data.

## 1. Entered picks and interrupted writes

The four round-one paths on both revisions left `(10, True)` byte-identical after the later correction. The candidate named April 18 as `no decision or preview stamp` for the bare lost acknowledgement and interrupted `bts run`, `skip record contradicts a delivery` for the durable DM plus EOD skip, and `scheduler state marks the pick locked` for the durable private lock. Those four tests address those exact states. Five nearby producer sequences still dropped the entered day, as in findings 1 and 2. [Original four](deliveries.log), [additional five](writers-verified.log)

| Writer window | Evidence and replay outcome |
| --- | --- |
| Scheduler DM or public send | Persisted attempt, delivery marker, lock or scoreable decision holds an unresolved past day. The exception handler clears the attempt on an unknown result. An EOD skip or a reused preview can then let it through. |
| Private scheduler | A scoreable decision or intact lock holds. Before the decision, restart initialization or quarantine can remove the hold. No state after quarantine and a present unlocked state after restart both reproduced. |
| `bts run` | Fresh choice uses `required=true`. Reused `required=false` was refused by the real live gate, including when an automatic observation was supplied. Successful posts durably mark delivery. An interrupted post before that save leaves ambiguity. |
| `orchestrate` | Non-live selection creates the same `required=false` stamp as preview. Post acceptance before the marker save reproduces a silent skip. Catch-up uses the same send-before-save order. |
| `bts preview` and grading | A preview save itself records selection, with no commit. Grading a pending result re-saves the stamp. A pre-stamp decoder and writer dropped it and then the file blocked. An interruption between the scorer’s streak update and terminal pick save holds on the applied-date marker. |
| Dashboard and recovery | The dashboard reads the pick, scheduler and decision files. Its POST updates contest saver state. Recovery/regeneration can retain string IDs and can restore past terminal files. |

The producer trace and contract matrix cover durable evidence and the named failure windows. The five entrant-side reproductions used the actual writer calls with synthetic upstream transport, schedule and game results. For older processes the probe used the archived loader and saver immediately before the stamp was introduced. [Writer and grading controls](writers-verified.log), [old writer](grading.log), [round-two source](../../src/bts/scheduler.py)

## 2. Date directory discovery

| Carbon fixture | Base | Candidate | Disposition |
| --- | --- | --- | --- |
| Mode `000` date directory, scoreable decision, missing root pick | Saves `(1, True)` | Saves `(1, True)` | Pre-existing NOTE. The scoreable-decision requirement is unmet. |
| Date symlink to skip, delivered pending root | Retains `(9, False)` | Retains `(9, False)` and warns of a contradiction | Round-one wrong write fixed by the delivery check. The candidate still traverses the symlink. |
| FIFO in place of date directory, missing root pick | Saves `(1, True)` | Saves `(1, True)` | Pre-existing NOTE. The bad date entry is silently omitted. |
| Mode `0300` picks root with 10 hits and saver miss | Saves `(0, True)` | Saves `(0, True)` | Pre-existing NOTE. A failed listing is treated as an empty season. |

Use a single `os.scandir` for both canonical name forms and propagate a listing error. Validate the date parent without following links before reading a decision. A scan-only scratch patch named April 16 for bad date entries and kept `(9, False)`. The unreadable root raised and kept `(10, False)`. The healthy skip still saved `(1, True)`. On the synthetic 162-date tree, 20 warm folds averaged 6.2 ms before and 4.0 ms after. The guard adds at most one date-parent `lstat` per day whose decision is read. [Base and candidate](directories.log), [patch and timings](scan-control.log)

## 3. Rule 1 corpus

| Corpus | Same observation | Base value to unavailable | Base unavailable to value | Unavailable on both with a changed exception |
| --- | ---: | ---: | ---: | ---: |
| Codex 630 | 560 | 14 | 21 | 35 |
| Carbon 435 | 386 | 22 | 14 | 13 |

- No value-to-value difference. The historical sparse miss and all four numeric-string ID shapes fold as on the base. An invalid decision beside a terminal pick is bypassed as Amendment 3 requires.
- All 36 value-to-block differences are enumerated in the ledgers. Twenty-eight have a scoreable decision with no root pick, a required blocker. The remaining eight are seven Codex `today_unknown_result` mixes and Carbon `blocker_today_malformed_minimal`. Those eight fail item 3's literal compatibility gate at `src/bts/picks.py:991–993`. They are parseable, malformed current-day files with no applied marker. The base ignored them and the candidate retains the streak with a dated warning. The earlier contract expressly requires rejecting these shapes. Authorize those eight guard exceptions in the review criterion or reconcile that criterion with the guarded-read requirement. No ordinary writer was identified for those shapes.
- An independent forward fold agreed with all 35 newly available histories. These were genuinely unentered preview/skip leftovers in the fixtures or a foreign-season marker on today. Terminal misses, hits, double hits and voids were folded directly from their recorded outcomes. The original round-one oracle marked some restored legacy terminals invalid. I reused its fixtures and replaced that oracle with the terminal fold in the contract. [Codex counts](codex-r2.log), [Codex differences](corpus-differences.jsonl), [Carbon differences](carbon.log)

## 4. Rule 2 table

All **4,480** cells matched the amendment’s rules for the supplied evidence. There were **932 skips**. For readable picks, an unreadable decision takes precedence over an applied marker. Same-season and tagless markers hold. A foreign marker is ignored. Today otherwise contributes zero. For a past day, a scoreable decision holds, then any of the six signals holds, then a locked or unreadable scheduler state holds. A non-scoreable decision or a valid preview stamp with no decision permits a skip after those checks. Invalid stamps are treated as absent.

The TSV gives each dated outcome and marks existing exact fixtures. The parametrized table covers 21 distinct cells across 23 rows. Fourteen more exact combinations were identified elsewhere in the focused cases. The remaining 4,445 have no exact fixture identified in that audit. These counts measure exact combinations. The entrant-side probes show where the evidence itself is misleading. [Full matrix](matrix.tsv), [counts](matrix.log)

## 5. `bts reconcile` end to end

- Both correction directions applied at 02:00 and 07:59:59 ET. At 08:00, on a late feed response and when the scoring lock was acquired at cutoff, the result stayed settled. Blocked runs with a readable streak returned 0 and said `Streak retained`. With corrupt retained state the CLI returned 1, preserved bytes, and emitted no traceback. A complete fold atomically replaced corrupt state. Concurrent runs produced one correction, two receipts and one final hit. Interrupting the streak replacement left the pick correction durable and retry refolded it. [Cutoffs](corrections.log), [state and retry](failures.log), [concurrency](concurrent.log)
- Six-second subprocess bounds covered old and recent pick, decision and streak paths, decision parents, the picks root, scoring lock, lineup journal and receipt paths. Bad pick files named the date. Bad streak targets exited 1 with a path diagnostic. A terminal pick bypassed its special decision file as expected. Receipt root/date FIFOs emitted publication errors promptly. The unchanged `.scoring.lock` and lineup-journal FIFO paths still timed out. The picks-root FIFO and lock directory still raised filesystem errors. [Special-file results](special-r2.log)

## 6. Tests and mutations

| Revision | Corpus failures | Delivery failures | Reconcile failures | Total |
| --- | ---: | ---: | ---: | ---: |
| `c70effe9` | 322 | 4 | 15 | 341 failed, 344 passed |
| `c2039541` | 70 | 4 | 50 | 124 failed, 561 passed |

Round one fails on strict terminal decoding, overbroad skips and missing evidence vetoes. The base fails on the required skip/block changes, dated diagnostic/retained wording and guarded reads. All four base delivery cases preserve the streak as intended and fail the new diagnostic assertions. The logs list every failed test ID and its first failing assertion or exception. [Round one red](red-c70effe9.log), [base red](red-c2039541.log)

All five scratch mutations were killed. Dropping the lock check failed 3 tests. Treating a missing stamp as a preview failed 24. Letting a skip override a delivery failed 7. Strict terminal validation failed 273. Ignoring unreadable decisions failed 1. The last mutation is caught only by the explicit unreadable-decision preview row. [Lock](mutant-no-lock.log), [stamp](mutant-missing-preview.log), [delivery](mutant-skip-over-delivery.log), [terminal](mutant-strict-terminal.log), [decision](mutant-ignore-invalid-decision.log)

The combined run passed **1,189** tests at each of the native, April 2027 and April 2028 process clocks. The initial sandboxed run passed 1,188 and hit the loopback bind restriction in `test_web_saver.py`. The reruns permitted loopback for that test and blocked external Python networking. A process-local realtime clock offset preserved the monotonic clock. [Native](native.log), [2027](wall-2027.log), [2028](wall-2028.log)

## 7. Deployed comparison

Round one's command compared `c2039541` and `c70effe9`, and reported only a full-horizon comparison. The corrected probe archives `619abda7` and `aa61bb43`, calls their real replay functions across ET dates, and prints the first divergence, each first block and the final results. `619abda7` excludes today's terminal result, so a one-day difference can precede the first genuine block. The synthetic control detected both events and proved all input bytes were unchanged. [Dry-run output](dry.log)

## 8. Other diff observations

The existing test edits match the authorized delivered-pick and retained-wording changes. The CS2 summary mock omits `retained_streak`, so it tests the CLI fallback while other cases test the actual receipt path. The receipt docs still omit `invalid_pick_file` and the added replay fields. The lineup journal and scoring lock FIFO failures predate the diff. No separate change to the shared pick parser or `update_streak` was implicated. [Existing test edits](../../tests/test_streak_season_cs2.py#L391), [receipt docs](../../docs/ops/reconcile-receipt-v1.md)

## For the restored box

Run from the checkout containing the review scripts and both git objects. This scans the whole 2026 season. Set `--today` to the first reconcile's ET date to compare that exact run. Exit 2 means divergence, exit 3 means a block without divergence, and exit 0 means agreement with no block. The worker rejects file writes and bounds each replay. The real tree was unavailable here.

```sh
env TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .codex-review/wr-c3-r2/dry_compare.py data/picks --today 2026-12-31
```

## What I ran

Archived revision probes with pinned ET fixture clocks, `compare_probes.py` in all four modes, `writer_probes.py` on both revisions with fix controls and the pre-stamp resave, `scan_control.py`, `matrix.py`, `e2e.py` for corrections, failures, concurrency and special files, and `dry_test.py`. Ran each pytest job serially through `~/bts-runs/yield_run.sh`, with archived sources for the two red runs and five mutants. Reviewed `git diff c2039541 HEAD` and the relevant writers. The focused suite included the C3, streak, reconcile, scoring-lock, LightGBM, web saver and related streak tests. No full fast suite was run.
