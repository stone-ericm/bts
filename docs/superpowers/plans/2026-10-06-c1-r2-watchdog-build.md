# C1 rank 2 build plan: the outcome / entry / restart watchdog (production)

**Implements:** the FROZEN registration `docs/sota_audit/2026-10-04-prereg-c1-watchdog.md` (rulings R1–R6, §2 checks, §3 schedule, §4 gate).
- **Process:** this is **production code**. Each piece below is reviewed until SIGN under the deploy rules: the watchdog itself, and each producer prerequisite separately. The concrete deployment then goes to Eric under D7. It is installed only after his recorded approval, and operational acceptance follows commissioning. Nothing here approves activation (A8) or play.
- **Code:** `src/bts/watchdog/` (new). Tests: `tests/watchdog/`.

**Map of what it observes:** gathered 2026-10-06, read-only, without touching `data/`. File:line references are as of `adec2bd`.

| Area | Where | Notes for the watchdog |
|---|---|---|
| Pick file | `picks.save_pick` (`picks.py:301`); `DailyPick` (`:205`) | delivery fields; `pick_was_delivered` (`:321`); writers: scheduler, check-results, reconcile, `save_nonterminal_result` |
| Decision | `daily_decision.py` (v1–v3); scheduler is the only writer (`scheduler.py:668,726,759`) | `delivery_status` ∈ delivered / locked_unconfirmed / private_locked / not_applicable |
| Streak / saver | `picks.save_streak` (`:511`), `scoring_lock` (`:527`); `saver_state.py` (flock + transitions log) | W-state must **not** use `_replay_season_streak` as its oracle |
| Scheduler state | `SchedulerState` (`scheduler.py:351`); `load_state` renames corrupt files | the watchdog must not call `load_state` (it writes) |
| Delivery | `_deliver_and_lock_pick` (`scheduler.py:926`); mode `_pick_delivery_mode` (`:882`) | DM id / Bluesky uri / `delivered_at`; `delivery_refusals[]` in the state |
| Entry | `check-pick-entered` (`cli.py:1609`); `contest_fetch.pick_entry_status` (`:128`); marker `health_state/pick_entry_check.json` | the marker has **no account / selection / game binding** and no failure record, so it cannot confirm entry (R4) |
| Grading | `check-results` (`cli.py:2087`); `grade_pick_in_feed` (`picks.py:1003`) | 01:00 cron, waits to 06:00 |
| Reconcile | `reconcile` (`cli.py:2382`); `reconcile_results` (`picks.py:1090`) | **always rewrites `streak.json`; prints "No scoring changes" and keeps no receipt** (I-207) |
| Contest | `fetch-contest-streak` (`cli.py:1846`); `contest_ledger.jsonl` (plain append; **rows lack user_id / username**) | account provenance comes from `contest_streak.json` (`user_id`, `username`) |
| Liveness | `heartbeat_watchdog`; `data/.heartbeat`; `scripts/check_heartbeat.py` pure `is_stale` / `assess_churn` (windows 1200/3, 3600/3, 10800/4) | never call its `main()` (it writes and pings) |
| Health system | `health.runner.run_all_checks`; `alert.dispatch_dm_for_health_alerts` | never invoked (R5); existing checks are credited only as explicit dependencies |
| Cron | `scripts/cron-setup-hetzner.sh` static `CRON_LINES` | **hazard:** `check-pick-entered` (`:67`) is installed unconditionally. Any plain `install` re-enables entry nags whatever the intent (the box has been private since 9/14). R4's intent-aware cron fixes this |
| Backups | R2 `sync_to_r2` manifest (`data/sync.py:177`: pa parquets, probable-pitcher lookup, `mdp_policy.npz`, `mdp_tail_policy.npz`); restic ops/archive; `restore_drill` (ops only) | **no model/policy restore code exists**; blend pickles are not in R2. W-restore covers the registered subset only |
| Incident register | `docs/audit/2026-09-29-incident-register.json`, `watchdog.boundary` | delivery 27, none 25, grading 20, restart 18, preservation 13, private_vs_contest 8, entry 7, singleton_slate 1 |

## Phase P: producer prerequisites
Each is separately reviewed until SIGN. These are production changes at **unchanged authenticated request counts and cadence**.

| # | Change | Contract |
|---|---|---|
| P1 | **Entry receipt** from the existing `check-pick-entered` fetch | A versioned, atomic record per attempt, including failures and no-attempt outcomes. It binds: account (`user_id`) and season; ET date; the committed selection's semantic identity (slot role, batter, game, delivery reference); observed rows; the verifier outcome; and the start, response and publication times. The existing-confirmed/no-fetch path references the original receipt without a new observation time. No cookies or tokens are stored |
| P2 | **Reconcile receipt** from `reconcile` | Per target date and slot: attempted / observed / pending / skipped / failed; correction applied or refused; source-payload hash; terminal/void basis; per-slot response time; the 08:00 ET next-day cutoff. An empty corrections list establishes nothing |
| P3 | **Consistent capture** for W-state | A producer-issued immutable snapshot / completion receipt written under the scoring writers' serialization (`scoring_lock`). It records the actual persisted pick, decision and streak bytes plus the replay-input inventory and the as-of interval. Tests pause or kill the producer between saves |
| P4 | **`[scheduler].entry_intent`** loader (`enter` / `research`, no default) **and an intent-aware `cron-setup-hetzner.sh`** | `research` keeps private delivery and does not install the entry cron; `enter` expects dm/public delivery plus the entry cron. Checklist C2 (legacy `shadow_mode`) is a prerequisite. This also closes the hazard in the map above |
| P5 | Checklist **C1 private-mode posting-transport guard** | Lands before the delivery-mode tests (§4.4) |

## Phase W: the watchdog (`bts watchdog <job>`)
| # | Piece | Notes |
|---|---|---|
| W0 | **Package skeleton.** The owned root `data/watchdog/` (symlink / escape refused; all locks, logs, dedup, temp and restore scratch live there). Notification state: a short owned critical section; failed or uncertain sends stay pending; the dedup key binds incident, ET date, selection and state, and survives restarts. DM transport. A clock seam | §4.2 write-confinement tracing (deny writes outside the root, including same-byte and write/undo cases) |
| W1 | **W-deliver, W-postponed, W-decision, W-mode** (every 5 min, every day) | the cutoff equals the earlier selected leg minus `SUBMISSION_CUTOFF_MIN`; DST, midnight and DD-leg tests |
| W2 | **W-entry** | uses P1 receipts only. Absence is asserted only from a successful applicable observation; anything else is unverifiable |
| W3 | **W-state** (every 15 min) | uses P3 capture plus an independent replay from a declared initial state; it never uses `_replay_season_streak` |
| W4 | **W-grade** | 08:15 plus retries at 10:35, 13:35 and the next days within the 30-date horizon. Slots come from the contest ledger (`not_hit` → `miss`, void and DD slots kept). Account provenance comes from the same fetch's `contest_streak.json` |
| W5 | **W-restart** | uses `check_heartbeat`'s pure `is_stale` / `assess_churn` plus `read_nrestarts`. It declares the external monitor as a dependency; multiple-instance and deploy-version dispositions are registered |
| W6 | **W-restore** (Sunday 04:00) | the pinned R2 manifest goes into an empty contained scratch directory under `data/watchdog/`. Checks: manifest membership, hashes, tail/base pairing, then a fixed decision fixture. Concurrent live drift gives unverifiable. One rehearsal goes through the C1 launcher (0.1 CPU-hours) |
| W7 | **W-self** | Eric's dedicated Healthchecks check, pinged only on completed due work (env var name to be fixed at review; the value is never committed) |
| W8 | **W-capture-stop** | the trio design's C-E2: a planted-stop positive alert and an absent-stop control |
| W9 | **Coverage matrix** over all 119 incident records | implemented predicate / fixture, existing-control dependency / fixture, or deferred / unverifiable with a reason. The 25 `none` records are out of scope |

## Gates (§4), in order
1. Failure/recovery fixtures red then green, driving the real CLI and cron selection, with HTTP / DM / R2 / Healthchecks / process observations patched (never the predicate under test).
2. No added writer races: write-confinement tracing and concurrent-writer attribution.
3. Notification, scheduling and self-monitor fixtures.
4. Intent and evidence prerequisites: P1–P5 must land. The gate cannot pass on planted receipts that current producers never emit.
5. A deploy-gating code SIGN on the watchdog and on each producer, then D7, then install via the intent-aware cron. Operational acceptance (Eric's Healthchecks check commissioned, a missed-completion rehearsal, producer contracts verified on the box) precedes any A8 activation.

**Order:** P4 + P5 first (they also remove the cron hazard), then P1, P2, P3 (each with its own review), then W0 → W1 → W5 → W7 → W2 → W4 → W3 → W6 → W8 → W9.
