# W15/W15b review · 284dc97a against f02e6b9c

## Verdict

**BLOCK**

## Blocking findings

1. **W15 log text still supplies false OOM evidence and W15b makes this replay miss delivery.** `health/probable_pitcher_cache.py:22` logs an eligible corrupt file named `data/health_state/probable_pitcher_cache/2026-07-22.oom-kill.json`. With a committed pick, inline shadow dispatch and no shadow artifact, HEAD first sends a 741-grapheme WARN digest. On restart in the same ET day, `health/analytics_artifacts_missing.py:165,243` accepts the W15 journal line as scheduler death evidence. HEAD emits a false `analytics_artifacts_missing` CRITICAL. The captured endpoint rejects its 1,009-grapheme digest and the receipt switches to failed. d7c66234 sends the false CRITICAL in 958 graphemes. Genuine fatal controls still fire on both revisions. Exclude W15 diagnostics from the fatal journal evidence scan before matching, and keep paths single-line. Preserve genuine fatal signals and the stable WARN key. [HEAD replay](journal-head.log) · [Prior replay](journal-r2.log)
2. **A status timestamp can still put a literal backtick in logged text.** `probable_pitcher_status.py:78–83` accepts a raw `recorded_at` of `2027-01-01` + backtick + `04:59:00+00:00` for ET date 2026-12-31. Python accepts the separator and `health/probable_pitcher_cache.py:26,50,103` prints the input verbatim in the detailed journal warning. An operator forwarding that text can break Markdown quoting, contrary to R6. Render the parsed time with `isoformat()` and leave the record and key raw. `display()` also leaves literal line breaks in displayed paths. The replay split an evidence location across two journal lines. Escape line breaks and other controls at display time so the diagnostic stays on one line. [Health replay](health-head.log)

## Notes

- P1/P3 and A3 tradeoffs surfaced three real feature losses. An otherwise valid entry with a missing opposite-side pitcher encoded as `NaN` loses usable away history, which goes beyond the three named W15-S2 losses but follows P1. A dangling cache symlink is left in place, as P3 requires, where the base installed a regular file. A status obstruction can leave a transiently available unrelated game uncommitted until its feed disappears. All three changed the synthetic subsequent bullpen value from 0.5 to missing. [Base](features-base.log) · [HEAD](features-head.log)
- Accepted W15-S1 limits include a producer root and picks parent that disagree. The inherited dispatcher and digest limits still apply. Late-D wording takes its dates from the processed slot, and a D observation on D+1 can still be labeled “1st observed day.”

## 1. Round-2 replays and policy

- Under `oom/data` and four alternate OOM spellings, d7c66234 returned WARN plus false `analytics_job_oom` and was rejected at 1,143–1,167 graphemes. HEAD returned one WARN and sent 610 in every case. The backtick basename, custom root and record cache path replays lost their literal backticks on HEAD. The combined valid/bad record DM was 630 and the raw key and files matched. [HEAD](health-head.log) · [d7c66234](health-r2.log)
- All seven kind sets across counts 1–1,000 maxed out at 578 characters. The summary gets only fixed labels and a decimal count. A Python list length fits in 19 digits on this host, bounding the summary at 593. Blockers 1 and 2 concern the detail text and its consumers.
- HEAD’s 48 policy combinations exactly matched genuine-only policy output. At d7c66234, W15 first with `Result=oom-kill` CRITICAL and `memory_growth` changed the `memory_oom_correlation` quote to W15. HEAD quotes the genuine CRITICAL. The table’s “the same” holds for source/severity and misses that text difference. No loss of genuine OOM detection was observed. [Policy comparison](policy-prior-followup.log)

## 2. Whole branch against base

- Real `compute_all_features` at f02e6b9c raised `AttributeError` for game 100’s null member both with and without its feed. HEAD completed both. The present feed gave pitcher/team tuples `(2, True, 61), (23, True, 62)`. The absent feed gave missing bullpen inputs and a dated warning. [Base](features-base.log) · [HEAD](features-head.log)
- Copied the two real caches (5,095 and 20,087 entries) and generated 40 fixtures including Unicode decimal and signed/zero-padded aliases, reversed member orders, nulls and large ints. Base and HEAD matched exactly for dict equality, both key orders, types, bytes, mtime and empty stderr. No record or evidence file appeared. The ordered cross-revision fingerprints matched.
- Six hard exits immediately before/after each rename preserved the expected old bytes, evidence and record at each stage. Fresh processes recovered, and the recovered game survived removing its raw feed. In a separate failure sequence, the game was lost if the status destination was obstructed and the feed left before retry. Same-day changed outcomes produced separate records, and late D selected D’s two observations with D+1 already present. D+1 selected all three and both slot DMs were delivered. [Hard exits](restarts.log) · [D/D+1](health-head.log)

## 3. Scope, routing and stderr

- `d7c66234..284dc97a` changes presentation in the reader, producer warnings, docs and related tests. The summary gives the lookup relative to the run’s picks parent. A FIFO where that directory should be gives a warning and a path in the log. For a custom root the DM alone cannot identify the directory. An absolute custom picks path let health in another cwd read the record, while its evidence pointer resolved only from the producer’s cwd. The producer/consumer root mismatch remains W15-S1. [Routing replay](routing-head.log)
- Confirmed escaping at the two health log sites and all three producer sites, including simultaneous evidence/status failures. Independently hashed a producer record’s raw cache path to reproduce the full identity and the stored filename, and hashed the raw bad basename into the summary key. The actual evidence file was under the raw path. [Raw identity and stderr](followup-head.log)
- Valid Unicode encoded identically at all three revisions for the exempt unreadable-cache line on UTF-8 stderr. Base and HEAD emit `\xe9` on ASCII where d7c66234 emitted `?`. On UTF-8 with a synthetic unencodable surrogate, HEAD’s `backslashreplace` emits `\udcff`, and `strict` suppresses the diagnostic. A UTF-8 descriptor proxy with no `errors` attribute suppresses even ASCII text on HEAD. `getattr(stream, "errors", None)` would honor the fallback for that proxy. [Stream bytes](stderr.log) · [Proxy replay](routing-head.log)

## 4. Changed operator sentences

| Sentence or diagnostic | Worst state and literal action |
|---|---|
| Summary count and “recorded today or yesterday,” plus log and status location | A late D run can include a date two days behind the inbox, or the named directory can itself be unreadable. Look in the health run’s log and use the processed date/picks parent. Blindly following the record pointer can delay discovery, especially on a custom root. |
| Summary “Keep the active cache, raw feeds, evidence copies and other copies intact” | The active cache may already be repaired or an earlier publication may have failed. Keeping each available version preserves evidence and current inputs. |
| Summary “Send this alert promptly …” through pending pick and R2 clauses | A selection or sync can precede receipt. Forward to the lead or Eric for the latest pick and a usable R2 copy, without editing inputs. The detailed log retains the instruction about already-entered pick records and the current/previous manifest. |
| “Probable-pitcher cache status unreadable at …” | A status path can be a FIFO, symlink or bad file. Use the escaped path to identify it. Opening a FIFO normally can wait. In finding 1 the journal classifier itself sends the operator toward a false shadow failure. |
| Status log “Send this alert …” | The status envelope may be the only visible symptom. The lead or Eric receives the path for investigation. Forwarding it does not authorize a restore or pick change. |
| “That run recorded an evidence copy at …” | Evidence can precede a failed active-cache rename, be moved later, or be relative to the producer’s cwd. Decode the displayed path and verify the file before repair. A multiline path can cause the operator to copy a partial location. |
| Producer stderr dropped-entry line | Counts describe the lookup and may precede failed artifact publication. Use the sample to investigate and read subsequent stderr before inferring disk state. An in-memory loss can affect a pick. |
| Producer stderr evidence-failure line | A conflicting file at that location may be the obstruction. Keep it and the old active file. An unrelated usable scan can still affect the current run without persisting. |
| Producer stderr status-failure line | The evidence may already exist, but health may have no readable observation. Keep the active file and evidence and forward the diagnostic. A feed that disappears before retry can leave history missing. |
| Legacy unreadable-cache stderr sentence | Retains the raw spelling, including backticks. “Rebuilding” can mean an empty in-memory lookup with no disk replacement. Verify the state before a repair. |
| `CLAUDE.md:251` summary pointer and OOM claim | The fixed summary itself cannot match the attention regex. The operator can follow the journal pointer into the false received CRITICAL or failed send in finding 1. |
| `CLAUDE.md:251` path escaping and stderr-handling sentence | Decode displayed backslashes/backticks to recover the actual path and leave records as written. A UTF-8 stream set to strict can suppress a warning on an unencodable path. The log still admits the timestamp backtick and path line break in finding 2. |

## 5. Tests and mutations

- The requested one-process selection passed **1,056**, with one opt-in base corpus reporter skipped. The first sandboxed run passed 1,055 and failed only at the saver fixture’s localhost bind. The wrapped rerun enabled numeric loopback and kept external network blocked. Both real-cache copy tests ran. [Completed run](focused-loopback.log) · [First run](focused.log)
- **31 of 33 scratch-source mutants detected.** Restoring the status path, dropping backticks, escaping the stored path or unreadable-basename incident key, and restoring `errors="replace"` were detected. Skipping `display()` was detected for the unreadable status log, evidence log, drop stderr and status-failure stderr.
- Two mutants survived the full selection at 1,056 passed each. Removing `display()` from preservation-failure stderr leaked literal backticks. Escaping the cache path only in the producer’s identity preimage changed the identity while leaving the visible record path raw. The independent obstruction and raw-identity probes cover those gaps on HEAD. The tracked suite also missed the journal feedback and timestamp separator in findings 1 and 2. [Mutation outcomes](mutations.log) · [Direct checks](followup-head.log)
- All applicable earlier-round mutations were detected, including cache/record validation, rename order, evidence reuse, dates, identity, record symlinks, occupied records, raw-root failure, size cap, descriptor ownership, closed stderr and the detailed log. The removed 600-character path fallback has no remaining branch to bypass. The path-reinsertion mutant restores the old unbounded-path failure mode.

## 6. Rest of the diff

- `git diff --check f02e6b9c HEAD` passed. The strictly-earlier reliever join and pick eligibility are outside the diff. The independent prior-journal follow-up replaced an earlier probe assertion that incorrectly expected d7c66234’s false CRITICAL to exceed the cap. The stderr follow-up used fresh obstructed paths after an earlier probe reused a status directory.

## What I ran

- Direct bounded probes on archived f02e6b9c, archived d7c66234 and the pinned HEAD, real feature/health entry points with pinned clocks, hard-exit/restart children, a captured journal response, and real DM code over a captured capped transport. The pytest selection was all `test_wr_w15*.py` files, `test_probable_cache_review.py`, `test_leads_small_fixes.py`, `tests/health/`, `test_lightgbm_optional.py` and `test_web_saver.py`. Both the suite and serial mutations ran through `yield_run.sh` after 07:00 ET. The probes and logs are beside this report. The tracked tree is clean.

## What I did not run

- The full fast suite, live cron/systemd, live picks, actual R2 or Bluesky, or a host power-loss experiment.
