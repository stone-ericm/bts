| Lead | Verdict | Cost | Trigger likelihood |
|---|---|---|---|
| 1. Scheduler rollback | REAL BUT MINOR | Lost day-state, quarantine WARN, possible repeat skip DM. Delivered pick and existing scoreable decision survive normal restart. | Future incompatible same-day rollback only. No new state fields in this deployment range. |
| 2. Feed void | REAL BUT MINOR | Public-feed recovery can write an unresolved pick and wrong recovered streak or saver. | Manual/legacy recovery or drift check with a public void. Ordinary DM picks are absent from the feed. |
| 3. Probable-pitcher cache | REAL | Stale bullpen history can reach training and live scores. A torn cache can lose historical coverage. | Persistent for an affected ID once cached. First reads of premature feeds or changed probables are conditional. |
| 4. Skip-policy checkpoint | REAL | A published shadow-policy conclusion can change with a different sample at the same checkpoint. | Requires a late older outcome, recovered historical gap, or correction. Ordinary append-only recovery is stable. |
| 5. Shadow model version | NOT A DEFECT | No tag loss through supported writers. Forced misuse excludes the pick from current-version shadow evidence. | Requires bypassing the shadow loader or passing shadow bytes to the production parser and deliberately saving them back. |

## 1. Scheduler rollback

The supplied deploy base `619abda7` is an ancestor of HEAD `df454725`. The local deploy ref is absent. Inspecting `git log 619abda7..HEAD -- src/bts/scheduler.py` and comparing both `SchedulerState` declarations and runtime field sets found **zero new fields**. The morning check uses separate health status files. The range does add a `void_games` key inside entries of the existing `delivery_refusals` field.

The scheduler probe writes a HEAD state, then runs the full `run_day` on HEAD and on the archived deploy source, with and without an injected future field. Both versions quarantine the unknown top-level key. Both build fresh state with `pick_locked=False` even for a compatible file. The compatible path carries skip intent, notice and commit state, refusals and fallback history into the fresh state. Quarantine loses those. A separate skip-notice probe sent again only after quarantine. The quarantine is visible to the `scheduler_state_integrity` WARN check. [Probe](scheduler_probe.py) · [Log](scheduler.log)

With the delivered pick file intact, the first lineup check locks on an existing scoreable decision or on the pick's public `bluesky_posted` flag or DM `notification_sent` plus ID. It skips reselection and pick transport for either channel. The classification path keeps existing scoreable decision bytes and resyncs the commit flag. With no decision file it recovers a scoreable `delivered` record from the pick. The final fallback and missed-pick alert also check delivery evidence. Normal polling can still grade a pending result.

HEAD's morning check, when otherwise eligible, returns before reading contest state on a scoreable decision or a delivered/attempted pick. `619abda7` predates that check. Separately, calling `_deliver_and_lock_pick` directly on a marked pick bypasses the 5a saved-pick guard and transport, but re-saves the pick and rewrites the commit, losing original streak provenance in the probe. That helper branch is distinct from the ordinary restart's classification path. 5a checks the selection state only on a live delivery that still needs to happen. [Scheduler](../../src/bts/scheduler.py) · [Pick markers](../../src/bts/picks.py)

**Smallest fix.** Have tolerant readers pass known top-level keys to the state constructor and still quarantine malformed or incomplete required data. Ship a tolerant version as the rollback target before adding fields, or have the rollback workflow stop the new scheduler and atomically project its latest valid state into the old schema before restarting the old scheduler. A new reader alone cannot protect the current rollback target. Keeping the writer on the old schema while an old reader remains a rollback target also works. The automatic canary rollback currently resets code and restarts services against the same data. [Deploy workflow](../../.github/workflows/deploy.yml)

| Case | Must still happen |
|---|---|
| Valid old state or extra top-level field | Keep recognized state and same-day skip idempotency. |
| Torn JSON or missing required fields | Quarantine and surface the warning. |
| Delivered public/DM pick or unconfirmed attempt | Suppress a second pick send and preserve a scoreable commit's provenance. |
| Undelivered saved pick on a mismatched state | Enforce the 5a refusal. |

## 2. Void from the public feed

A formatter-to-regenerator round trip ignored `Void. Streak: 12`, wrote the day's result as JSON null, and produced streak 0 and saver available even though the synthetic snapshot held streak 12 and a spent saver. A second probe drove `bts state regenerate` through the actual feed parser with a fake Bluesky client. With a preceding recognized hit at 10, the void stayed null and the streak stayed 10. A subsequent ordinary miss with its own pick still paired to that day and consumed the inferred saver. [Probe](void_probe.py) · [Log](void.log)

Regeneration takes the last recognized reply's streak, defaulting to zero if there are feed picks but no recognized result. The void produces no saver transition. Saver inference starts available and consumes it on the first recognized miss with a preceding inferred streak of 10–15. Whenever the feed contributes picks, composition uses the timeline's values even if the snapshot has different values. An unresolved output also blocks a later state export.

`bts state regenerate` is the manual public-feed disaster-recovery/migration tool. `bts state verify` runs the same pipeline into a temp directory and can report false drift. The historic Fly design describes periodic verification. The current Hetzner cron has no such entry. The CLI remains callable in a DM season, including for old public dates, but normal DM picks never appear in the author feed and result replies require a public `bluesky_uri`. DM grading reads the local pick files independently. [Regeneration](../../src/bts/state/regenerate.py) · [CLI](../../src/bts/cli.py) · [Cron](../../scripts/cron-setup-hetzner.sh)

**Smallest fix.** Recognize the formatter's canonical void line as a terminal `void` with its published streak. Use the snapshot as the baseline for replay when reconstructing state across the cutoff.

| Case | Must still happen |
|---|---|
| Canonical void reply | Write `void`, carry the reported streak, preserve inferred saver state. |
| Hit, miss, skip, unrelated reply | Parse the established result formats, keep real miss saver behavior, leave skips and unrelated replies out of result matching. |
| Empty public history after the cutoff | Keep snapshot state. |

## 3. Probable-pitcher cache

| Claim | Finding |
|---|---|
| Game ID is the only key | True. Each numeric filename stem indexes one record for both sides and team IDs. |
| Never invalidated | True per cached ID in normal reads. Changed, replaced or removed feeds do not evict it. Removing the cache or losing its JSON forces a rebuild. |
| Ignores `raw_dir` | The scanner honors it on misses. The fixed cache path and unscoped hits can return entries from another raw root or from a root no longer available. |
| Missing probable sticks | True for a successfully parsed feed whose side ID is absent. That side's null is persisted and the entire ID is skipped on later calls. |
| Non-atomic write | True. Injecting an interruption after truncation left an unreadable file. The next read silently used an empty lookup, then rebuilt from readable feeds in the supplied root. With no usable feeds it returned empty and left the torn file in place. Cache-only entries absent from the scanned root are lost on a successful rebuild. |

The disk probe changed a feed under the same ID, switched roots and interrupted a write. An independent end-to-end feature probe warmed a team for ten days, then compared a stale cache with a fresh one after a missing probable was filled. Both doubleheader games retained the same prior-history feature on D. On D+1 the stale feature was 1.000000 versus 0.954545 with the repaired input. The two game IDs were independent and their qualifying relief PAs shared one team/date bucket. A rescheduled or resumed game with the same ID keeps the cached value even if its probable changes. An obsolete non-null ID can label the actual starter as relief, or all PAs as relief when the listed pitcher never appears. [Probe](probable_probe.py) · [Log](probable.log)

An unknown probable suppresses that side's relief PAs throughout subsequent recomputations. With known team ID and a warm window, HEAD still attaches ordinary strictly prior history on D. The lost PAs change D's contribution to later training and serving rows until that bucket would age out of the last 30 qualifying dates. Where no other relief PAs qualify on D, the whole bucket is missing and warm-up can take longer. The contract check's five 2025 sides lacked a probable in both cache and raw, so cache invalidation alone cannot recover those inputs. The ordinary predictor refresh pulls through yesterday and live slots fetch current schedule/feed separately. A manual or premature historical pull can seed this problem, and the raw downloader also skips an existing file. No change to a specific live pick was established. [Compute](../../src/bts/features/compute.py) · [Predict](../../src/bts/model/predict.py) · [Raw pull](../../src/bts/data/pull.py) · [2025 contract check](../../docs/reviews/2026-11-18-bullpen-feature/contract-check/contract-check.md)

**Smallest fix.** Scope mutable entries to the resolved raw root and refresh on source-file fingerprint changes, including null-to-known and changed non-null values. Atomically replace the published lookup and serialize or merge simultaneous refreshes if there are concurrent writers. Retain the current lookup format for synced readers, using a sidecar for fingerprints or an explicitly selected immutable seed for cache-only use. Refresh the raw source too when it contains a premature snapshot.

| Case | Must still happen |
|---|---|
| Fixed input, synced history without raw, genuinely unknown probable | Reuse stable data, retain trusted historical cache coverage, keep the unknown side excluded from relief classification. |
| Distinct doubleheader IDs, same-ID changes, alternate raw root | Keep separate game classifications and same-day team pooling. Pick up authoritative same-ID updates without importing another root's mutable entries. |
| Interrupted update; training/serving rebuild | Readers see a complete snapshot. Keep the strictly earlier bullpen windows and the empty-lookup schema. |

## 4. Skip-policy shadow checkpoint

The CLI backfills authoritative reach-57 MDP skips through UTC age 10. Hit/miss becomes checkpoint-eligible at age 4. `reconcile_decision` calls the checker on a pending row before testing staleness. Even at age 10, a returned hit/miss is admitted. Resolved hit, miss and void records are terminal individually. The status re-sorts eligible hit/miss rows by date and recomputes the first 30/60/90 each time. The documented terminal decision consequently depends on the prefix staying fixed. [Implementation](../../src/bts/skip_policy_shadow.py) · [Design](../../docs/audit/2026-06-20-skip-policy-shadow.md)

The pinned-clock CLI probe published `above_breakeven` with 29 hits of 30. Five days later a historical decision supplied an age-10 miss that sorted before the last member. The first look became `straddles_breakeven`, with 28 hits of 30, eligible count 30→31 and one substituted date. An independent path used a pre-existing aged pending row left by `--no-reconcile`. The next day's normal update produced the same change without an outage. A five-day append-only outage increased eligible count to 31 while preserving the original first look. Aged authoritative contradictions are flagged and excluded on purpose, which can also remove a fired look. The probe dropped to 29 eligible and `insufficient_n` while leaving the two shadow files on disk. [Probe](checkpoint_probe.py) · [Log](checkpoint.log)

**Smallest fix.** Persist each fired look's ordered date/outcome membership and verdict basis atomically, including inconclusive looks. Replay saved looks and keep late/backfilled or corrected rows visible in the monitoring totals. Keep later looks nested under an explicit late-data rule. If corrections must revise an old look, publish a new revision that identifies the changed membership. Simply checking staleness first would discard recovered outcomes and would leave the contradiction path.

| Case | Must still happen |
|---|---|
| Recent temporary failure, eventual hit/miss/void, cron catch-up | Keep retries and recoverable outcomes for monitoring. Keep voids outside the hit-rate denominator. |
| Fired decisive or inconclusive look, routine update | Preserve published dates, counts and basis. Permit subsequent prespecified looks after an inconclusive one. |
| Superseded decisions and corrections | Prune recent supersessions and surface aged corrections with an auditable history. |

## 5. Shadow model version

`parse_daily_pick` drops the field. `load_pick` and `load_pick_bytes` normally read `<date>.json`. The scheduler and `check-results` use `load_shadow_pick`/`save_shadow_pick` for `<date>.shadow.json`. Backfill reads with the shadow loader and applies result edits to the raw JSON, retaining version and other keys. No supported save-back path passes a shadow file through the production parser. [Pick I/O](../../src/bts/picks.py) · [Shadow evaluation](../../src/bts/shadow_eval.py)

The grading probe ran the CLI and kept the tag and current-version pair. A forced production-parser round trip lost both. The independent manifest/apply probe kept the tag and excluded an untagged legacy file. `shadow-report`, shadow cycle status and backfill-manifest generation use the version to select the current cohort. Missing tags default to v1 for that comparison. This affects shadow evidence and review eligibility. Production selection, delivery and streak do not depend on the tag. [Probe](shadow_version_probe.py) · [Log](shadow_version.log)

**Smallest fix.** No production change is required. For defensive parser parity, read `shadow_model_version=data.get("shadow_model_version")` in the generic parser.

| Case | Must still happen |
|---|---|
| Current shadow pick graded or backfilled | Keep its version and current-cohort eligibility. |
| Untagged legacy pick or older explicit version | Keep the original tag or absence and its exclusion from the current cohort. |

## Other notes

- **NOTE** Older pick readers also lack the newer `chosen_on` and `automatic_reading` fields. A real old-code pick re-save, such as when grading a pending result after rollback, can strip that provenance even though a normal delivered-pick classification leaves the file alone.
- **NOTE** Feed regeneration pairs a recognized reply with the newest unresolved pick without using the reply's parent URI. A late reply can be attributed to the wrong day independently of void parsing.
- **NOTE** `skip_policy_shadow.write_status(now=...)` uses `now` for aged contradictions but does not pass an `as_of` date to the checkpoint builder. Historical replays need to pin both clocks. The probe does.
