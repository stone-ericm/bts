# W15 round 2 review

## Verdict

**BLOCK** for `d7c66234` against `a924d748`. The original failure fixtures pass. The new summary can still lose the whole DM through a false OOM escalation, and dynamic paths can violate R6.

## Blocking findings

1. **A W15 summary can create its own false CRITICAL and overrun the chat limit.** Produce one `invalid_entries` observation from `{"100":null}` without feeds. Point `picks_dir` at `oom/data/picks`, with `oom/data` an alias of the very same producer data directory. The health check reads the right record. The new summary inserts `oom/data/health_state/probable_pitcher_cache` into the Alert at [`health/probable_pitcher_cache.py:75`](../../../src/bts/health/probable_pitcher_cache.py#L75). The existing text matcher at [`attention.py:162`](../../../src/bts/health/attention.py#L162) treats the directory name as OOM evidence and duplicates the summary in an `analytics_job_oom` CRITICAL. There was no memory alert or other health source in the fixture. The real dispatcher made a 1,143-character request, the captured 1,000-grapheme endpoint rejected it, and the receipt recorded failure with one CRITICAL. Round 1 sent one 926-character WARN on the same input. The smallest fix is to exclude W15 operator text from OOM evidence matching. Keep evidence from actual OOM-reporting sources eligible. [Replay](health.log)
2. **R6 still fails on paths read from disk.** In the default status directory, a date-eligible broken basename containing a literal backtick produced that character in the status warning from [`health/probable_pitcher_cache.py:24`](../../../src/bts/health/probable_pitcher_cache.py#L24). A matching readable custom status root with a backtick in its path sent one in the real DM, and its recorded evidence path logged the same character at line 33. Removing static Markdown delimiters leaves these inputs unchanged. Escape literal backticks reversibly in display fields before logging or formatting the summary, while keeping actual paths and incident identities intact. [Replay](health.log)

## Notes

- **W15-S2 accepted.** Missing `home_tid` on game 100 still moves a later away-facing bullpen feature from `0.5` to missing with an explicit warning. `CLAUDE.md` and the author’s updated corpus table name the tradeoff. [Feature replay](features.log)
- With ASCII stderr, a corrupt `café.json` cache showed `caf\xe9.json` in round 1 and `caf?.json` in round 2. The direct write uses `errors="replace"` at [`features/probable_pitcher_cache.py:40`](../../../src/bts/features/probable_pitcher_cache.py#L40), losing P2’s byte-for-byte diagnostic for this input. Use the stream’s configured error handler to retain the path spelling. [Feature replay](features.log)
- The first local preview probe let Matplotlib create its default temporary cache. The repeat redirected its cache and temp directory under this review. The final routing log is from that repeat.

## 1. Round 1 blocking inputs

- Real `compute_all_features`, 64 numeric directory candidates before `109.json`, `RLIMIT_NOFILE=48`. Base and round 2 both committed 109, returned `opp_bullpen_hr_30g=0.5` and held descriptors at 4 → 4. Round 1 exhausted descriptors, omitted 109 and returned missing without a warning. Package imports were asserted per child. [Feature replay](features.log)
- A real closed-reader stderr pipe and a recoverable cache incident completed features at `0.5` in both revisions. The former child exited 120 and round 2 exited 0. [Feature replay](features.log)
- Real producer, runner, attention policy, DM formatter, dispatcher and `send_dm` with captured HTTP transport. A persistent bad member’s D and D+1 bodies were 559 and 560 ASCII characters. The same-day invalid/unreadable pair sent one 578-character body on each processed day. All passed the captured 1,000-grapheme cap. Sixteen findings and a long path produced one 525-character WARN and one 598-character wire body. Finding 1 uses the same policy and transport with a different path. [Health replay](health.log)

## 2. Aggregation, dedup and log routing

- Independently hashed the sorted identity and unreadable-basename list. A late D run saw D while a D+1 record was already present. D+1 counted both records even though their full identities matched across dates. Same processed day and same inputs suppressed another send. Adding a same-day recovered record or an unreadable basename changed the key and sent again. Details logged on each run even when its DM was suppressed. [Health replay](health.log)
- Ran the actual scheduler `run_day` EOD path with a synthetic finished slate, pinned wall time, real W15 health and fake delivery. With default Python logging, both detailed warnings reached stderr. The shipped service starts `bts schedule` without stream overrides, so stock systemd routing puts them under `journalctl --user -u bts-scheduler`. The local replay proves the process stream. It cannot verify the live user manager’s settings or journal retention. [Routing replay](routing.log) · [Service](../../../scripts/systemd/bts-scheduler.service#L14)
- The shipped Hetzner cron runs preview at 03:00, not `run_all_checks`. Through `preview` with a fixture pipeline running the real feature function, it produced the cache stderr line and dated record, with no full A7 health detail. Cron redirects both streams to `~/logs/cron.log`. A directly invoked health runner under that redirect wrote full details to the file. The inherited scheduler no-games branch bypassed EOD health in the replay. Two consecutive off-days can let a dated record age out before the next EOD check. [Routing replay](routing.log) · [Cron](../../../scripts/cron-setup-hetzner.sh#L99)
- The DM drops timestamps, sampled games, recovery counts, evidence paths, exact R2 generation guidance and the instruction to preserve the records of a sent or entered pick. Readable records and scheduler stderr retain those details. With only unreadable status or a fallback path to a different directory, the journal is the useful route. The short DM and inherited unbounded multi-source digest are both documented. Finding 1 is caused by W15 alone.

## 3. Operator text

Compared the summary and full logged variants character-for-character with R4/2a and P6/A7/R6 for standard paths, including both evidence branches and unreadable status. The formatter adds its existing streak suffix. [Health replay](health.log) · [Routing replay](routing.log)

| Sentence or clause | Worst state and literal action |
|---|---|
| DM count and kinds, “recorded today or yesterday” | Late D after midnight can cover wall-clock day before yesterday, and a FIFO directory yields “1 observation” without any readable observation. Use the processed date and the logs. Someone searching by the inbox wall date can miss the only relevant record. |
| DM “details in” | An unreadable directory can itself be a FIFO. The over-600 fallback can name a directory that does not exist or contains a different cache’s records. Both reproduced. Follow the lead/Eric route and use the health run’s log when the named path is unusable. The fallback follows the amended wording. |
| DM keep active cache, feeds, evidence and copies intact | A successful repair may already have replaced the active bytes. Keeping all versions protects recovery and preserves the current pick inputs. |
| DM route, latest preview and pending pick | Selection or entry can precede the read. Forward promptly to the lead or Eric and compare current state. The DM omits the instruction to preserve already-entered pick records, which is still present in the log. |
| DM unreadable-cache R2 clause | The only usable copy can be in the previous manifest and a sync can run before the DM is seen. Secure and verify that copy before rotation. The log clarifies current/previous and immediate escalation. A status-unreadable-only DM cannot identify an underlying whole-cache incident. |
| DM “1st observed day” suffix | D’s observation can repeat on D+1 with a fresh summary key and the same suffix. It reflects summary-key age. Use the record timestamp before inferring an incident is new. |
| Invalid log timestamp and sampled IDs, then recovery and missing-feed counts | More than ten games or many aliases can contribute. Inspect the whole observation context and raw coverage. Treat the counts as run outcomes and IDs as a sample. |
| Unreadable log opening, empty start, rescan count and potentially missing games | A usable scan for one game can coexist with other missing histories and a failed cache replace. Verify previews and the active cache before any supervised repair. |
| Evidence log path | Preservation can finish before a failed cache replace, or later evidence may be moved. Verify the named copy before using it. Path delimiters are now plain text, and finding 2 shows dynamic backticks remain possible. |
| No-copy log and next-day repeat | No evidence is named and that call left the original in place. A later call can rewrite it. Compare timestamps and current files before restoring. |
| Log preservation and escalation, preview and already-entered pick | Keep all copies and pick records, then route to the lead or Eric. A repaired cache can change future inputs. These instructions authorize review and preservation. |
| Log immediate R2 escalation, manifests and seven-day age | The last good object can already qualify for pruning as soon as its final reference is gone. Verify and secure the object before the next sync. Reading a manifest alone leaves the object exposed. |
| Status-unreadable log and route | A matching FIFO, symlink or bad record is unreadable to the check. Forward its path. Opening a FIFO as a normal file can wait indefinitely. |
| Producer stderr and `CLAUDE.md` updates | Normal diagnostics retain their text and the broken pipe no longer fails the child. Check logs for the chosen path before repair in the ASCII case. The docs describe the accepted drops and short WARN but omit the fallback’s possible misdirection and the OOM interaction. |

## 4. Focused tests and mutations

- **281 passed, 1 skipped** in one wrapped process. Included the three W15 files, their real-cache copy cases, probable-cache and leads-small regressions, health runner, attention and dispatcher, optional LightGBM, and web saver. The opt-in base corpus reporter skipped. [Focused log](focused.log)
- Mutation checks pending the shared wrapper slot.

## What I ran

Independent feature children over archived base, archived round 1 and round 2. Real health and scheduler EOD with synthetic state, pinned clocks and fake transport. CLI preview with a fixture pipeline and cron-style stream capture. Focused pytest with external transport blocked and numeric loopback allowed for the web-saver fixture. `git diff --check a924d748 HEAD`. The tracked tree is clean.

## What I did not run

Full fast suite, real systemd or cron services, live preview or picks, external network, Bluesky or R2. The capacity limit was simulated at the HTTP boundary.
