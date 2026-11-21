# W15/W15b final-round review · 0ce9e5ee

## Verdict

**SIGN WITH EDITS**

Correct the operator wording in `CLAUDE.md:251`. The two production blockers from the prior review are fixed on their original inputs.

## Blocking findings

None.

## Notes

- **Documentation edit.** `CLAUDE.md:251` still says health logs “unreadable status paths.” It now logs a validated basename, a date, or a fixed relative folder description. Update that sentence. Scope the OOM guarantee to W15 message bodies. With those same bodies behind a journal hostname of `room-node`, the inherited scan still accepts the line as death evidence. The matcher looks at the journal envelope too. [Matcher replay](text-replay.log)
- The legacy unreadable-cache warning remains the explicit exception. A cache path containing `oom`, a backtick and a newline reached stderr in identical bytes at f02e6b9c, 284dc97a and 0ce9e5ee on each tested encoding. [Stream replay](stream.log)

## 1. Original blocking inputs

- Reused the round-one sequence with the same July 22 clock, corrupt `2026-07-22.oom-kill.json`, dispatched inline shadow, committed pick and delivery receipt. At 284dc97a the first 741-grapheme DM sent, then the restart emitted a false `analytics_artifacts_missing` CRITICAL, failed the 1,009-grapheme send and marked the receipt failed. At HEAD the first 741-grapheme DM sent, the restart produced no CRITICAL or repeat send, and the original receipt stayed sent. Genuine OOM after the W15 detail still triggered and sent an 860-grapheme DM. [Prior replay](original-prev.log) · [HEAD replay](original-head.log)
- The original backtick timestamp and newline in `cache_path` reached the old health log. HEAD renders the time with a `T`, logs only the validated evidence basename and emits one detail line plus the summary. The record bytes and key stayed raw and the captured 610-grapheme DM sent. [HEAD replay](original-head.log)

## 2. Text and both matchers

- Ran the real producer and runner on 28 hostile records covering both reasons and dates, malformed and genuinely shaped names, nine timestamp separators, custom roots and an unreadable FIFO status directory. Captured all three producer diagnostics, both health log branches and the real formatted DM. Forty message lines were printable, carried no raw controls/backticks and supplied no OOM evidence to either matcher with an ordinary journal prefix.
- Replayed each line before and after six genuine journal forms, including kernel out-of-memory, systemd OOM, `code=killed`, `signal=kill` and `status=137`. All 480 pairs and the whole batch in both orders returned the genuine line. Attention output, including the genuine OOM plus memory-growth cases, matched its genuine-only controls. [Matcher replay](text-replay.log)

## 3. Scope, routing and artifact parity

- The round-two source delta follows amendment 4. It changes diagnostic rendering and selects the same two dates while passing the selected date to the label. Neither matcher changed. The UTF-8 descriptor proxy lacking `errors` emitted the expected warning on HEAD after emitting no bytes at 284dc97a. [Stream replay](stream.log)
- With custom absolute picks and health running from a different cwd, the configured picks parent led to the records. The record’s raw `cache_path` led to the cache and its sibling `rejected/` from the producer’s cwd. Two malformed files on the same date gave identical detail lines on HEAD, so their individual names require directory inspection. A FIFO status directory yielded the relative folder warning and no readable record. A producer-only failure needs the writer’s cache configuration and cwd, which the new stderr lines intentionally omit. [Routing replay](identity.log)
- Compared exact bytes for active caches, original evidence copies, valid and malformed records, their filenames and transport bodies. Both revisions gave identical results across same-day outcomes, late D versus D+1, an unreadable cache, artifact obstructions and raw malformed filenames. The matched incident keys included a direct unreadable-directory case with a backtick in its leaf name. [Parity replay](identity.log)

## 4. Changed operator sentences

| Text | Worst state and literal action |
|---|---|
| Opening timestamp in the valid invalid-entry and unreadable-cache texts | A later run can see repairs or a different cache. Use the canonical time to identify the producing run and preserve current inputs before review. |
| “That run recorded an evidence copy named … in the rejected folder beside the cache” | The copy can predate a failed cache rename or live in a different checkout. Use `cache_path` from the record and the producer’s cwd for relative paths. Verify the named copy before recovery. |
| Status unreadable, with a validated record basename | An eligible basename can be a non-regular entry. Look for the basename in the configured status folder and forward the warning. The name alone establishes neither a usable record nor a usable cache. |
| Status unreadable, with a date and unnamed entry | Several unrelated malformed entries can share the same date. Inspect the directory names and records before acting. The log alone cannot identify which item failed. |
| Status unreadable, with the fixed folder description | The folder can itself be a FIFO or inaccessible. Locate it beside the run’s configured picks directory and send the alert to the lead or Eric. Reading a FIFO as a regular file can wait. |
| Dropped-entry stderr | Counts reflect that feature call, before evidence, status or cache publication may fail. Use the samples and subsequent diagnostics. The feature call can use a changed lookup even when disk was untouched. |
| Preservation-failure stderr | The obstruction can be a bad destination or missing original bytes. Keep the active cache and identify its `rejected/` location from the writer. The in-memory lookup can still affect a pick. |
| Status-publication-failure stderr | The named record may never have existed. Check the cache-relative status folder using the writer’s configuration. Original evidence can exist even while the active cache and health record stay as before. |
| Legacy unreadable-cache stderr | “Rebuilding” can mean an in-memory rebuild with no cache replacement. The raw path can split the line or mislead the unrelated journal scanner. Verify disk state before a repair. |
| `CLAUDE.md:251` log-path description and restricted-text sentence | The operator now locates the folder by configuration and unnamed entries by date. The current “unreadable status paths” wording wrongly makes them search the log for a full path. |
| `CLAUDE.md:251` OOM guarantee | Read the claim as a guarantee about W15 bodies. External journal prefixes and the inherited warning can still supply a false match. |
| `CLAUDE.md:251` path lookup and legacy stderr sentences | Start with the run’s picks parent and use the record to resolve the producer’s cache and evidence. An unreadable folder or failed publish can require the writer context. A strict encoding handler can still suppress unencodable legacy text. |

## 5. Tests and mutations

- The requested one-process suite passed **1,076**, with only the opt-in base corpus reporter skipped. Both real-cache copy cases ran. The offline guard blocked external traffic and allowed numeric loopback for the saver fixture. [Focused run](focused.log)
- **37 of 39 scratch-source mutants detected.** Reinstating a path in any of the six changed warning sites, allowing arbitrary status or evidence names, printing the raw timestamp and removing the `getattr` fallback all failed tests. The prior applicable mutants ran again. The six removed `display()` mutations were covered by the raw-text variants where the behavior still exists. [Mutation log](mutations.log)
- Two mutants passed the full selection at 1,076 tests each. Escaping `cache_path` only in the producer’s identity preimage still survives from round one. Escaping the unreadable directory basename in the incident key survived when the helper is called with a custom final component. The real runner always uses `probable_pitcher_cache` as that component. The independent byte comparison checked both raw identities on HEAD. [Producer identity survivor](mutation-results/escape-path-in-producer-identity-only.combined.log) · [Directory key survivor](mutation-results/escape-directory-name-in-incident-key.combined.log) · [Direct comparison](identity.log)

## What I ran

- Archive-verified copies of the original two probes, bounded checks of three source revisions, captured journal and capped transport, operator routing and artifact comparison, both text matchers and the previous HEAD feature/restart fixtures on the new code. The two real cache copies matched the archived base fingerprint from the earlier run, and all six hard-stop points recovered. [Feature and restart replay](carried-forward.log)
- After 07:50 ET with an idle wrapper, ran all `test_wr_w15*.py` files, `test_probable_cache_review.py`, `test_leads_small_fixes.py`, `tests/health/`, `test_lightgbm_optional.py` and `test_web_saver.py` in one process through `yield_run.sh`. Ran the mutations serially through the wrapper after that process exited.
- The first two text-matrix attempts used invalid fixture assumptions about slash-containing basenames and case-distinct names on this APFS volume. The corrected replay used independent valid filenames. All review artifacts are in this directory. The tracked tree is clean.

## What I did not run

- The full fast suite, live systemd or cron, actual picks, R2 or Bluesky, or host power-loss recovery.
