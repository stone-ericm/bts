# Verdict

**BLOCK** on `6ff1e87e` against `88b69a16`. The replay and streak bytes matched the base. The checklist still leaves a path to setting aside a committed pick.

## Blocking findings

- **B2. B8b's specified checks can miss a committed private pick.** On April 18, 2027, run the scheduler with private delivery and research entry intent. Interrupt it after `save_state` persists `pick_locked: true` and before it writes the decision. There is no MLB entry by design. With a prior skip the decision still reads `"scoreable": false`. Without a prior decision the file is missing. Both cases get past the two checks the checklist describes. The checklist asks the operator to confirm the pick was uncommitted without saying how to detect an interrupted private lock or what to do when the decision is missing, unreadable or ambiguous. [Checklist B8b](../../docs/ops/2027-season-start.md#L32) · [Private lock sequence](../../src/bts/scheduler.py#L1262)

  `operator_probe.py` interrupted the real scheduler writer at live-forward capture. Reconcile held the streak and named April 18 in both cases. Performing the documented move made reconcile save 9 after omitting that date. Keeping the pick file and resolving its synthetic result as a hit rebuilt 10. Amend B8b to check for private locks and delivery evidence, and explicitly leave the file in place and report if the operator cannot independently confirm the pick was never committed or entered. A false, missing, malformed or unreadable decision is insufficient confirmation. [Reproduction](operator.log)

## Notes

- A diagnostic for a noncanonical stem displays escapes. Substituting the printed `2027-04-18\n` literally into B8b's `mv` exited 1 and left the actual LF-named blocker in place. The raw stem in the receipt did identify the file. B8b could direct operators to report escaped dates or identify the literal filename before moving anything. [Reproduction](operator.log)
- Round 1's injected failure inside `ReconcileReceipt.replayed` still leaves `not_reached` and prints the existing recalculated summary. The injected exception has no corresponding disk or network input to the hook's dict assignment. For a real receipt publication failure, the CLI still prints the one blocker line alongside the receipt error. [Output probes](round1-carbon-output-readers.log)

## 1. B1 and rendering

- The inherited glob fixes the ASCII year prefix, and the regex permits decimal digits, literal hyphens and at most one final LF. The lexical date cutoff also applies. The new escaping emits printable ASCII for this entire input set and leaves each of the four fixed reasons unchanged, including the apostrophe in `today's` and the exact B8b quotation. [Replay scan](../../src/bts/picks.py#L845) · [Rendering](../../src/bts/cli.py#L2956)
- Ran 22 real filenames. Canonical and impossible ASCII dates printed verbatim. An LF, Arabic-Indic and supplementary-plane decimal digits, and digit-plus-LF variants were accepted when in range, printed in one line, and stayed raw in the receipt. CR, CRLF, repeated or interior LF, other controls, NEL, U+2028/U+2029, a superscript digit, and backslash were ignored by the scan. The non-ASCII month case sorted after the pinned date.
- The filesystem refused invalid UTF-8 with `EILSEQ` outside the sandbox. Injected control characters, Unicode separators, backslash and a surrogate into both fields at the receipt hook. Stderr was exactly one printable ASCII line, and both receipt strings decoded to the original values. All four reason paths also passed through real replay, including unlink after the initial read for the hit-loader refusal. [Rendering evidence](rendering-filesystem.log)

## 2. Base equivalence

- Isolated workers imported full archives of both revisions, verified the touched source files against Git, pinned ET and the UTC streak-write timestamp, and compared complete file bytes. The 630 original histories matched on 462 saves, 133 refusals and 35 exceptions. All 18 writer and symlink sequences matched and retained the streak. Seventeen exercised a real correction write before the refusal. [Corpus](equivalence-corpus.log) · [Writer cases](equivalence-writers.log)
- Round 1's 233 Codex and 133 Carbon hostile cases matched on replay, CLI exit and exception, and streak bytes or absence. The replay interrupt, actual contended lock across ET midnight and New Year, corrupt and missing streak, and bounded FIFO deadline behaved as before. The Carbon output probe also passed 15 pairwise blocker cases and same-file precedence. [Codex](equivalence-hostile.log) · [Carbon](equivalence-carbon-retry.log) · [Output and readers](round1-carbon-output-readers.log)

## 3. Operator step

- With known preview provenance and no entry, one move rebuilt the streak. Two blockers required separate moves and the first rerun reported April 19. A readable true decision protected a private lock. A false decision was insufficient and the missing-decision case had the same result. An unreadable true decision raised `PermissionError` on inspection and the pick was left in place. [Operator evidence](operator.log)
- The traced consumers recorded zero opens of the moved file, including with an invalid UTF-8 archive. Replay, pick listing and state export or verification scan top-level files. Nested decision, scheduler-state, quarantine and deferred-fallback readers use fixed names or narrower patterns. Restore verification searches recursively for four fixed names. An opaque Restic backup can still copy the file. [Reader checks](round1-carbon-output-readers.log)

## 4. Existing tests

- `git diff 88b69a16 HEAD --stat -- tests` lists five additions and 1,616 inserted lines. Name-status reports only `A` for all five. A path-specific diff confirms `tests/test_streak_season_cs2.py` is byte-identical to the base.

## 5. New tests and mutations

- On archived `88b69a16`, 34 failed and 635 passed. The failures were 18 writer or symlink cases, 11 reconcile cases and five adopted review cases. The base lacks the warning, including when receipt publication fails. The missing-streak case stops at the old stdout wording. All 630 corpus cases passed. [Base run](base-red.log)
- All 978 passed in one process on `6ff1e87e`, including the five new files plus `test_streak_replay.py`, `test_streak_season_cs2.py`, `test_reconcile_receipt.py`, `test_lightgbm_optional.py` and `test_web_saver.py`. The first run passed 977 and hit the sandbox's loopback bind denial in the existing web-saver test. The approved loopback run passed all 978. [Focused run](focused.log) · [Loopback run](focused-loopback.log)

Each scratch copy ran all 669 new tests. No mutant survived.

| Mutation | Failed / passed | Decisive assertion |
| --- | ---: | --- |
| [Remove stderr escaping](no-escaping.log) | 2 / 667 | LF filename and injected date or reason must render as one line |
| [Escape receipt fields only on disk](escape-receipt.log) | 2 / 667 | Both tests require raw decoded receipt values |
| [Render with `repr`](repr.log) | 34 / 635 | Canonical dates and fixed reasons must print without added quotes |
| [M5b](M5b.log) | 2 / 667 | Today's terminal hit replays despite the applied marker, and unreadable JSON takes precedence |
| [M5c](M5c.log) | 1 / 668 | Corrupt today plus an applied marker reports the corrupt file |
| [M5d](M5d.log) | 1 / 668 | An unresolved past result takes precedence over a hit-loader failure |
| [M6](M6.log) | 1 / 668 | Corrections with replay `not_reached` preserve the old recalculated summary |

## 6. Rest of the diff

- The receipt reference documents the optional `blocked_date` and `reason`. Saved and not-reached outputs followed the base in the comparisons. The correction write path and replay scan had no other changes. `git diff --check 88b69a16 HEAD` passed. [Receipt reference](../../docs/ops/reconcile-receipt-v1.md#L32)

## What I ran

- Local Git archive and diff checks, `equivalence_probe.py` for the two archived revisions, `rendering_probe.py`, `operator_probe.py`, and the archived Carbon output and reader probes. JSON observations and logs are in this directory.
- `mutation_probe.py` prepared the base and seven scratch sources and verified import origins for pytest. Ran each top-level pytest command serially through `yield_run.sh` after 06:35 ET with the specified interpreter, ET timezone and cache disabled. The corresponding `.log.watch` files record each exact command and exit. Only the focused set was run. The tracked tree remained clean.
