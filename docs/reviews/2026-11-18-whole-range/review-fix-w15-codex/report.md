# W15 round 1 review

## Verdict

**BLOCK** for `a924d748` against `f02e6b9c`. The original crash is fixed. A clean-cache scan can silently lose a usable game, an additional legacy shape loses usable history, and ordinary overlapping records can overrun the DM transport.

## Blocking findings

1. **Directory candidates leak file descriptors and silently change the feature.** [`probable_pitcher_status.py:35`](../../src/bts/probable_pitcher_status.py#L35) passes the result of `os.open` straight to `os.fdopen`. On this Mac, `fdopen` raises `IsADirectoryError` before taking ownership. With a valid cache lacking game 109, 64 `0001.json`–`0064.json` directories ahead of the valid feed `109.json`, and a child with a 48-descriptor soft limit, base committed 109 and computed `0.5` for `opp_bullpen_hr_30g`. HEAD exhausted descriptors, skipped 109, computed `NaN` and emitted no stderr or incident. The same helper handles evidence and health records. Own the descriptor in a `try/finally`, check `fstat` before wrapping, and use `closefd=False` or transfer ownership only after wrapping succeeds. [Probe and log](review_files.log)
2. **The exact-key rule drops usable history beyond the stated accepted losses.** Game 100 had `{ "away": 1, "home": 22, "away_tid": 51 }`, with no feed or alias. Its PA faced the away bullpen, so base used the correct pitcher and team and the game contributed the tenth qualifying prior PA. HEAD drops the whole object and a later game's `opp_bullpen_hr_30g` changes from `0.5` to `NaN`. The warning is accurate. P1 commands the drop at [`probable_pitcher_cache.py:23`](../../src/bts/features/probable_pitcher_cache.py#L23), but the explicit accepted-loss decision lists extra keys and integral floats. Resolve the contract before release. The smallest behavior-preserving remedy is a supervised migration that fills missing fields with `null` on otherwise valid legacy objects, while preserving their originals. Alternatively obtain explicit approval of this added loss. [Feature comparison](review_features.log)
3. **The new W15 aggregation can lose the entire EOD DM, including unrelated warnings in the same digest.** For one persistent `{"100":null}` cache with no feeds, the real producer wrote one observation on D and one on D+1. The real runner, policy, formatter, dispatcher and `send_dm` made a 926-character request on D and an 1,829-character request on D+1. With a captured endpoint enforcing the repo's documented 1,000-grapheme chat cap, D sent and D+1 failed with zero CRITICALs. A same-day invalid/unreadable pair also failed on both processed days at 1,723/1,729 ASCII characters. Sixteen valid records produced bodies over 14,000 characters and failed even local `MessageInput` validation. [`runner.py:327`](../../src/bts/health/runner.py#L327) forwards the newly overlapping W15 alerts to a single request. Bound the wire messages and record coverage only for delivered alerts. Keeping every exact P6/A7 message requires allowing multiple wire messages per digest. [Producer replay](review_retention.log) · [Transport replay](review_health.log) · [Local chat-limit review](../../docs/reviews/2026-11-14-season-rollover/build/cs6b1-check/contract-check.md)

## Notes

- P2 deliberately sorts raw filenames. With three candidate filenames including `09.json` and `9.json`, the copy probe observed a base/HEAD insertion-order and serialized-byte difference. The selected entries and all fixture features agreed. A1 requires exact parity when there are no recognized candidates for uncached games. Candidates discovered later use P2.
- The focused suite's first run failed when my guard blocked the web-saver fixture's loopback socket. The rerun allowed numeric loopback and kept external connections and DNS blocked. The first descriptor-pressure probe also exhausted a descriptor needed for its own pickle output. The separate bounded replay captured the actual feature, stderr and resulting disk state.

## 1. Defect, parity and sequences

- Archived `src` from base and verified `bts.__file__` in each child. Real `compute_all_features` raised `AttributeError` for a `null` entry on base with and without its feed. HEAD gave `0.5` with the feed and `NaN` without it, with the expected incident warning and record. Extra-key and integral-float controls reproduced the documented `0.5` → `NaN` loss. The missing-team case independently demonstrates finding 2.
- Compared copied 5,095- and 20,087-entry caches and a generated cache with aliases, nullable fields and a 121-digit integer. Empty raw roots, already-cached feeds and one new usable feed agreed on the returned entries, both orders, exact types, resulting bytes, stderr, artifacts and each feature value. Mtime stayed at the original nanosecond without a scan. Both rewrites advanced mtime. The raw-alias discovery exception is noted above.
- An unrelated bad member plus newly scanned game 109 committed evidence, record and cache. After deleting 109's feed, a fresh feature process still computed `0.5` and emitted no warning. Six hard child exits covered before and after each rename. Restart recovered successfully with old bytes until the cache commit, retained exact evidence and records after their commits, and coalesced a repeat after publication. A no-copy observation and later copied observation on the same ET date both survived, as did observations across the year boundary. [Feature replay](review_features.log) · [Rename and date replay](review_evidence.log)

## 2. Evidence and record identity

- Independently computed A4 compact, ASCII-escaped SHA-256 for both reasons, including the full original digest, preserved basename, and selected per-game results. Swapping a recovery beyond the first ten IDs changed identity with the same sample and totals. Exact same-day repeats kept the earlier timestamp. A date change retained a separate record even when the identity matched.
- Compared exact bytes after invalid UTF-8, UTF-16, broken key conversion and a non-object top level. A fake same-suffix file and exact-byte symlink supplied no evidence. With multiple regular matches, the first lexically sorted verified copy won. A directory, FIFO, symlink, wrong bytes or conflicting/malformed record at the occupied destination left the active bytes intact. The caller still got the usable lookup. Preservation-only, publication-only and joint failure emitted the incident line plus one line per failure. Publication accepted 65,536 serialized bytes and rejected 65,537. [Evidence replay](review_evidence.log)

## 3. Health validation and DM

- The reader rejected 71 malformed envelope or filename cases, including schema/field sets, reason/date/name agreement, UTC offset and ET date, path type, exact ints/count sums, empty unreadable IDs, sample length/order/uniqueness, identity case/length/prefix, evidence basename and original digest. Valid controls included both reasons, a null digest, signed IDs and a ten-ID sample. Two full digests sharing their twelve-character prefix yielded different incident keys. Matching malformed names used the basename failure key. Missing or out-of-window paths were silent.
- An unconstrained captured transport accepted the D and D+1 requests. A late D run with D+1 present selected D and D−1. Implicit ET on the pinned Jan 1 wall clock selected D+1 and D. Receipts suppressed a repeat on each processed day, and a new same-day identity triggered a digest. The capacity-constrained replay demonstrates finding 3. [Reader and transport replay](review_health.log) · [Producer-backed D and D+1](review_retention.log)

## 4. Operator text

P6/A7 matched character-for-character for both evidence branches of each reason, including singular/plural recovery and the status-unreadable path. Each passed through the real formatter. The first incident line and the two failure lines matched the required forms. Finding 3 concerns delivery of the rendered text. [Rendered output](review_health.log) · [Failure diagnostics](review_evidence.log)

| Text | Worst state on this tree and literal operator action |
|---|---|
| Invalid cache time, count and first IDs | A list of over ten affected games exposes a sample. Use the time and counts to scope review. Repairing only the displayed IDs can leave other affected games. |
| Valid aliases and usable feeds | Several malformed aliases can refer to one recovered game. Count affected entries. A recovery can still change the feature if the feed supplies different IDs. |
| Unusable, unverified and missing feeds | An unlistable season can hide an existing feed. Investigate raw discovery and compare previews. The instruction asks for no deletion or pick edit. |
| Named evidence | The evidence rename landed and the cache rename failed, or a later repair happened. Find and verify the recorded copy before recovery. Treating the named original as the repair could reintroduce bad members. |
| No evidence, active cache unchanged | A FIFO, zero scans or failed preservation can precede a later successful rewrite. Read this as the producing run's state and check current files. Assuming the old bytes still occupy the active path can send recovery to the wrong version. |
| Observation may repeat next day | A D+1 repeat can arrive after repair. Compare its timestamp and current state before touching files. A stale repeat does not call for reversing a repair. |
| Keep cache, feeds, evidence and copies intact | An active path may already point to a rebuilt subset. Keep all versions and send the alert. This preserves recovery options while the scheduler can continue. |
| Send promptly to `bts-lead` or Eric | The recipient can be unavailable and selection can precede the message. Forward to the fallback. Forwarding alone changes no pick or cache. |
| Review latest preview and pending pick | The preview may have used stale input and a selection may be imminent. Ask the lead to compare current state. Any pick change needs a separate decision. |
| Report a sent or entered pick and preserve records | The producing run may postdate entry. Send its status and retain records. The text does not ask to void or replace the pick. |
| Unreadable at recorded time | A later feature run can already have repaired the path. Inspect the timestamp and verify current state before acting. |
| Empty initial lookup, built entries for k games | A rebuild can have usable feeds for just a fraction of games. Treat k as a run outcome. The active write can still fail. |
| Other games may have been missing | Cache-only histories can disappear from that run's lookup and change features even when the file stays put. Verify the relevant preview and recover from backups under supervision. |
| Send immediately to `bts-lead` or Eric | One usable R2 object may remain only in the previous manifest. Escalate immediately. The sync timer can win the race. |
| Preserve and verify R2 copy before next successful sync | A successful sync can move the only good version out of both manifests. Ask the lead to secure a verified copy before rotation. Reading a manifest alone does not secure its data. |
| Unreferenced object can be pruned at seven days | The prior good object can already be seven days old when it loses its reference. The age is object age. Waiting seven more days can lose the only good copy. |
| Status unreadable, escalate named path | A FIFO or corrupt record can be the sole notice of the observation. Forward its path, avoiding manual reads of special files and destructive cleanup. The alert gives no recovery verdict. |
| Stderr invalid entries | The sampled counts can precede preservation, publication or cache replacement failure. Read following diagnostics and keep the originals. The line cannot confirm a disk commit. |
| Stderr unreadable, rebuilding from raw feeds | No usable feeds or a special path can leave disk untouched. The legacy wording refers to the in-memory rebuild. Verify current disk state before any restore. |
| Stderr evidence failure | Full digest verification can fail at an occupied destination. Keep the active bytes and suspect destination. The returned lookup can still affect the run. |
| Stderr status failure | An evidence copy may exist despite an unreadable record destination. Keep both paths and send the stderr observation manually. Health can miss the detailed report. |
| `CLAUDE.md` validity, cache aliases, rescan, rebuild | Apply exact IDs or `null` on all four keys, check which alias/feed was selected, and verify unreadable rebuild coverage. Finding 2 exposes an additional loss that needs resolution. |
| `CLAUDE.md` evidence/order, failure and dated alerts | A failed publish leaves the active cache in place after successful preservation. Keep the evidence and read the dated observation on D and D+1. The claim is about health inclusion. DM delivery can fail as in finding 3. |
| `CLAUDE.md` digest receipts, new keys, retries, banner, empty ID, restart | A failure, blank ID, changed key set or restart around the receipt can miss or repeat a message. Inspect health state and logs and preserve pick records. These are the inherited dispatcher limits. |
| `CLAUDE.md` escalation and staged repair | A pending sync or feature refresh can overwrite recovery or consume suspect data. Preserve copies, route to the lead/Eric, verify feeds, stage repair, stop scheduling and quiesce features before cutover. A verified repair can legitimately change a subsequent pick's inputs. |
| `CLAUDE.md` R2, path, backtest and race limits | A FIFO can block sync, a non-default picks root can miss records, and a same-checkout backtest or concurrent run can replace a repair. The bullet matches W15-S1. |
| `CLAUDE.md` valid entries, game ID and symlink | Changing the raw root does not refresh a cached valid ID. A rewrite can replace a regular-target symlink while leaving its target untouched. Verify the active file before treating a feed refresh or symlink target edit as a repair. |

## 5. Unusual files and inputs

- Bounded children covered FIFO, directory, regular-target symlink, symlink to FIFO, dangling symlink, invalid encoding and >64 KiB padded content at cache and raw-feed paths. Special caches reported and stayed in place. A regular cache symlink was replaced and its target retained. Oversized valid cache/feed content was read because the 64 KiB cap applies to status. Special feeds were unusable. `².json`, signed names and a regular-file raw root were skipped. An inaccessible raw root and unlistable season made unresolved drops unusable.
- FIFO, symlink, binary/oversized non-directory, real directory and inaccessible-directory cases at `rejected/` and status roots produced the expected classifications. Record FIFO, directory, symlink, mode 000, non-UTF-8 and over-limit content warned within child deadlines. Broken stderr suppressed all three diagnostics and returned the usable lookup. Single directory cases completed. Repeated directories exposed finding 1. [Filesystem replay](review_files.log) · [Bounded status replay](review_health.log)

## 6. Tests and mutation results

- Focused wrapper run **273 passed, 1 skipped** in one process. Included all three W15 files with the real-artifact copy cases enabled, probable-cache review, leads small fixes, health runner/attention/dispatcher, optional LightGBM and web saver. Only the opt-in base corpus reporter skipped. [Focused log](focused-final.log)
- **10 killed, zero survivors.** Corpus tests killed bool acceptance and skipped scanner validation. Rename interruption tests killed replace-before-preserve and publish-after-replace. Occupied evidence killed reuse by name. Incident and health tests killed date omission, sampled-only identity hashing, dropping yesterday, per-reason keys and following a record symlink. The date-omission check reached the full W15 fallback. [Mutation log](mutations.log)

## 7. Rest of the diff and `CLAUDE.md`

The bullet at [`CLAUDE.md:251`](../../CLAUDE.md#L251) describes the intended rule, ordering, artifact paths, staging, routing and inherited limits. Findings 1–3 qualify the runtime guarantees. The strictly-earlier bullpen join and pick eligibility code were outside the diff, and `ARCHITECTURE.md` and `FINDINGS.md` were unchanged. `git diff --check f02e6b9c HEAD` passed.

## What I ran

Real archived base and HEAD feature calls, timed reader/writer and hard-exit subprocesses, the real health runner/policy/dispatcher and DM code against captured endpoints, the focused pytest set, and serial fault injection on scratch source copies. Probes and logs are in this directory. No tracked files changed.

## What I did not run

No full fast suite, external network, production health job, live pick, live DM, R2 sync or power-loss experiment. Rewritten mtimes were compared against the original pinned mtime because the two commits ran at different wall-clock instants. DM endpoint limits were simulated locally using the repo's documented chat cap and the installed SDK validator.
