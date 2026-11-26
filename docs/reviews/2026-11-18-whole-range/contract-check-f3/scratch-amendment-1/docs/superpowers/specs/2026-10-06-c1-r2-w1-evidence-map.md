# W1 evidence map: what W-deliver, W-postponed, W-decision and W-mode can observe (code map, 2026-10-06)

**What this is:** a read-only map of on-disk evidence, from code and docs only (no `data/` read). It was gathered by a delegated code-search agent on 2026-10-06, at about `a1607c8` (line references are approximate after later commits). It is the input to the W1 spec. Registration rows: `docs/sota_audit/2026-10-04-prereg-c1-watchdog.md` lines 41–48.

**Timestamps:**
- pick `run_time` is UTC;
- `delivered_at`, `pick_locked_at`, scheduler-state times and archive `*_at` are ET-aware ISO;
- decision `finalized_at` is UTC with `Z`;
- `Pick.game_time` is the schedule `gameDate`, normally `...Z`; a naive value is read as UTC.

## 1. Pick file `data/picks/<ET date>.json` (`DailyPick`)
**Fields:**
- selection: `pick`, `double_down` (each a full `Pick` with `game_pk` / `game_time`), `runner_up` (best different-game, set even on singles);
- delivery: `bluesky_posted`, `bluesky_uri`, `notification_sent`, `notification_channel` (the scheduler writes only `bluesky_dm`), `notification_id`, `delivery_attempted`, `delivered_at` (ET);
- grading: `result`, `slot_results` (`pick` / `double_down`);
- provenance, including `policy_decision`.

**`pick_was_delivered`** = `bluesky_posted OR (notification_sent AND notification_id)`; `delivery_attempted` is not part of it.

| State | Pick-file evidence | Elsewhere |
|---|---|---|
| Preview or uncommitted candidate | every delivery field at its default | none |
| DM delivered | `delivery_attempted=true`, `delivered_at`, `notification_sent=true`, `notification_channel=bluesky_dm`, `notification_id` | — |
| Public delivered | `delivery_attempted=true`, `delivered_at`, `bluesky_posted=true`, `bluesky_uri` | — |
| Private commit | **identical to a preview** | `decision.json` `private_locked` and scheduler `pick_locked` |
| Caught send failure | `delivery_attempted` reset to false | **the failure is not persisted** |
| Crash mid-send | `delivery_attempted=true` with no delivered flags | becomes `locked_unconfirmed` |

**Preview vs candidate:** they are **indistinguishable in the pick file**. Only outside evidence separates them: `runs_completed` in scheduler state, the slate `written_at`, or `lineup_evolution_<date>.jsonl` (appended on every save; summaries only, with no delivery fields and no writer id).

**Writers:**
- **Scheduler:** candidate save every non-locked cycle, the fallback refresh, delivery saves, result polling, and `_archive_and_remove_pick` (unlinks).
- **`bts preview`:** writes **tomorrow's** file at 03:00, skipping only if it has a result or a post. A locked existing pick is still re-saved with provenance.
- **`check-results`:** results only.
- **`reconcile`.**
- **Legacy `bts run` / `orchestrate`:** these post via `should_post_now` with no cutoff guard and no `delivered_at` / `decision.json`.

## 2. `data/picks/<date>/decision.json`
**Schema versions:**
- v3 is written; v1–v3 are read.
- **v2 adds** `second_candidate`, `state_source`, `state_status`, `allow_double`, `contest_source_date`.
- **v3 adds** `objective`, `best_*`, `effective_best`, `tail_policy_sha256`, `degraded_reason`.

**Value sets:**
- `action` ∈ skip / single / double;
- `source` ∈ mdp / heuristic / unknown;
- `delivery_status` ∈ delivered / private_locked / locked_unconfirmed / not_applicable;
- `scoreable` is true for every commit, false for a skip.

**Invalid or unreadable means absent:** `is_scoreable_commit` then falls back to `pick_was_delivered`, so a corrupt record makes a private commit non-scoreable.

**Writers** (scheduler only; best-effort, returns None on failure):
1. **Commit,** in order: decision, then `committed_pick_written`, then `save_state`.
   - Branches: already-delivered → delivered; crash guard → locked_unconfirmed (plus a CRITICAL DM); private → private_locked; dm / public → delivered.
   - `_trigger_live_forward_capture_on_lock` (a subprocess, 10 s timeout) runs **before** the decision write, so there is a transient window where the pick is delivered but no decision exists.
2. **Classification lock:** writes only for a delivered pick or `delivery_attempted`, and never clobbers a scoreable record. A non-delivered classification lock writes **nothing**.
3. **End-of-day MDP skip:** only if `not committed_pick_written`, `final_skip_candidate` is set, and no scoreable record exists. It runs after the final fallback, E3, doubleheader rechecks and result polling, so it can land after midnight. A later same-day commit can overwrite it.

**No-games day:** `run_day` returns early. There is **no decision.json** and no scheduler state.

**Binding caveats for W-decision:**
- A `single` commit record can carry a non-null `double_down` (the runner-up `double_candidate`). Only the pick file's `double_down` is authoritative.
- A commit after a no-pick fallback selection has `primary=null` and `streak=0`.
- `second_candidate` is always null on commits.

**Derived:** `<date>.policy_shadow.json` (the 23:30 cron; pruned if the decision flips).

## 3. Scheduler state `data/picks/<date>/scheduler_state.json`
**Fields:**
- `games` (`game_pk`, `game_time_et`, `lineup_confirmed`, `is_doubleheader_game2`; from the **startup** schedule fetch, with no status) and `confirmed_game_pks`;
- `runs_completed` (`time` = the **end** of each check, `pick_name`, `pick_p`, …; coalesced runs marked);
- `pick_locked` / `pick_locked_at`: set on delivery **and** on non-delivered classification locks, so **not proof of delivery**;
- `result_status`;
- `next_wakeup` (the **next day's** wake);
- `analytics_jobs`;
- `skip_summary`;
- `skip_notified_at` (dm mode only);
- `final_skip_candidate`: set on a genuine MDP skip or a fallback-confirmed skip, cleared on a genuine selection;
- `committed_pick_written`;
- `delivery_refusals[]`: `at`, `label`, `batter`, `double_down`, `cutoff_et`, `late_min`, `archive`;
- `fallback_refreshes[]`: in-loop fallback only (the final fallback appends nothing). `reused_check_cascade` is effectively always false in production.

**Restart:** state is rebuilt. `pick_locked`, `pick_locked_at`, `runs_completed`, `confirmed_game_pks`, `result_status` and `next_wakeup` are reset. The skip, commit, refusal and fallback fields are carried forward. `games` is re-fetched, which is the only time game times in state refresh.

**`load_state` is a writer:** it renames a corrupt file to `.corrupt-<stamp>`. The watchdog must not call it.

**Planned run times are NOT persisted:**
- Lineup checks are at game − `lineup_check_offset_min` (box 60), clustered.
- The day wake is the earlier of 10:00 ET and the first game − 60.
- The in-loop fallback deadline is the earliest pick game − fallback_min, floored at 5 + cascade budget (12) + reserve (10).
- E3 is at the earliest game − 10.
- The only persisted "next wake" is `data/.heartbeat` `sleeping_until` (the current target only). Run times are reconstructable from `games[].game_time_et` plus the TOML offsets, but are stale if the schedule moved (I-077).

## 4. Cutoffs and game times
- `SUBMISSION_CUTOFF_MIN = 5`; `earliest_pick_game_et` is the minimum over both legs; a naive time is read as UTC.
- **Delivery guard:** refuses at or after the cutoff (equality refuses), re-checked just before each send. It uses the **pick file's `game_time`**; **there is no schedule refresh at delivery**.
- **A refusal:** an archive, a `delivery_refusals` entry and a CRITICAL DM; the next cycle re-picks.
- **`plan_fallback_action` rule order:** should_post true or None → deliver; ungated → deliver; gap → defer only if a contender window fits, else deliver; projected → defer if a window is pending; otherwise legacy.
- **The scheduler never re-fetches today's schedule during the day.** The only independent on-disk refreshed times are in `data/lineup_posting_times/<date>.jsonl` (the */5 collector). It overwrites `game_time_et`, has **no status** and no per-game timestamp (file mtime only), and never removes games.

## 5. Abandonment, deferral, refusal, skip and no-games days
**Archives** (`_archive_and_remove_pick`): `data/picks/<date>/<prefix>_<%Y%m%dT%H%M%S%z>.json`, written by a non-atomic `write_text`, then the live pick file is unlinked.

| Prefix | Reason | State record |
|---|---|---|
| `stale_pick` | `committed_game_past_cutoff` | **none** |
| `deferred_fallback` | the plan reason | a `fallback_refreshes` defer entry |
| `refused_delivery` | `past_submission_cutoff` | `delivery_refusals` |

**Changes that leave no archive:**
- a stale-status regeneration overwrites the file (and a stale file can remain if the new cycle skips);
- a past-cutoff current pick dropped in the fallback refresh is overwritten.

**MDP skip:**
- the live `<date>.json` is left untouched; the evidence is `final_skip_candidate` / `skip_summary` until the end-of-day decision write.
- After it, the decision is `skip` / `mdp` / `not_applicable` / non-scoreable.

**Locks without delivery:** `status_lookup_failed`, or a game started on an undelivered candidate. This sets `pick_locked=True`, skips the final fallback, and writes no decision.

**No-games day:** only a heartbeat (plus an empty lineup-times file; P1 `no_pick_file` receipts if installed). There is **no state, no decision, and no end-of-day health suite** (I-206).

**No 01:00 delivery fallback exists.** The 01:00 job is `check-results` scoring.

**Telling the cases apart:**

| Case | File evidence |
|---|---|
| MDP skip | `final_skip_candidate` plus `committed_pick_written=false`, then the skip decision |
| Abandoned candidate | a `deferred_fallback` / `refused_delivery` / `stale_pick` archive |
| Later chosen game | the final pick's `game_pk` differs from earlier candidates (archives, the evolution log) and `delivered_at` is before **its** cutoff |
| Missed early game | **no marker**: inferred from an earlier candidate on game X whose cutoff passed undelivered |

**E3 silence:** E3 does not fire when no live pick file exists, for example after a deferral archive.

## 6. Postponed, suspended and resume
- **Void states:** postponed and cancelled (`_classify_unposted_game_status`). Warmup (`PW`) is not locked, but is not available for fresh selection.
- **`is_resume_date_game`** applies only in the prediction slot fetch.
- **Status checks:** at selection and at the lineup-path lock (both slots). **Delivery makes no status check**, so the cached-fallback paths can deliver a postponed game (I-203, xfail).
- **Grading:** postponed or cancelled → slot `void`; a suspended game counts pre-suspension PAs only.
- **Persisted status evidence:** only `data/picks/slates/<date>.json` (`bts_slate_v2`). Per row it has `game_time` and `status`; it is last-write-wins and written only when predictions are non-empty, and `status` is absent for SSH tiers.
- **Not persisted:** each leg's status at send time.

## 7. Delivery mode and the entry cron
- **`_pick_delivery_mode`:** precedence `pick_delivery`, then `posting_mode`, with the C2 `shadow_mode` refusal; `private_mode`; aliases. It is resolved once at `run_day` start and **not persisted**.
- **Config:** `/home/bts/.bts-orchestrator.toml`.
- **`check_entry_intent`:** see P4. The entry cron line is installed only for an agreeing `enter` (once P4 is deployed; **the installed crontab may predate P4**).
- **Observation:**
  - the installed crontab, via `crontab -l` as `bts`, filtered on `# BTS-HETZNER`;
  - runtime evidence that it ran: P1 receipts (sealed discovery), the `pick_entry_check.json` marker, and `~/logs/cron.log`.
- **Recommendation-delivery evidence:** the decision's `delivery_status` (`private_locked` vs `delivered`); the pick's `bluesky_dm` / `notification_id` (dm) vs `bluesky_uri` (public); `skip_notified_at` (dm only). Operational DMs are recorded in `health_dm_delivery_status.json` and are not recommendation delivery.

## 8. Existing health checks (end-of-day suite: once per `run_day`; not on no-games days)
- **post_failure:** CRITICAL when a scoreable commit is undelivered, suppressed before 22:00 ET on the same date. A private commit would CRITICAL after 22:00.
- **fallback_defer:** CRITICAL when a deferral archive exists with no delivered final pick once the window has closed.
- **late_delivery:** each refusal is CRITICAL; `delivered_at` at or after the cutoff is CRITICAL; WARN inside the reserve. It falls back to `pick_locked_at`, which is re-stamped on restart.
- **postponed_pick:** undelivered picks only (a live statsapi read).
- **pick_entry:** reads the marker.
- **scheduler_state_integrity.**
- **Writers in the suite** (the watchdog must not run them): restart_spike, memory_growth, slate_auc, attention, and the alert dispatcher (`health_dm_delivery_status.json`).
- **Timing gap:** with no commit, the suite usually runs before 22:00, so post_failure and fallback_defer return nothing and nothing re-runs later.
- **E3** (`_maybe_alert_missed_pick`): once, after the loop. It is suppressed by an in-memory skip, a scoreable decision, a **missing pick file**, or a delivered pick. Its only persisted evidence is `health_dm_delivery_status.json` `sent_sources`, which later dispatches overwrite.

## Gaps (not persisted or not observable)
1. Planned check times, the fallback deadline, the cascade budget and the E3 time. Only the heartbeat's current `sleeping_until` survives.
2. Game times refreshed after the scheduler starts. The only refreshed on-disk source is the lineup collector, which has no status.
3. Each leg's game status at send time.
4. Caught send failures. Preview vs private commit cannot be told apart in the pick file.
5. The final-fallback plan; `stale_pick` with no state record; overwrites that leave no archive.
6. The effective delivery mode, and whether the installed crontab matches the script.
7. Any day disposition on a no-games day.
8. Restart resets of `pick_locked`, `pick_locked_at`, `runs_completed` and `result_status`; `pick_locked` does not imply delivery.
9. The decision's `double_down` on `single` records can be the runner-up; a commit record can have `primary=null`.
