# P3 design: the consistent capture for W-state (DRAFT, for design review before any code)

**Implements:** watchdog build plan row P3 (`docs/superpowers/plans/2026-10-06-c1-r2-watchdog-build.md`). The registration (`docs/sota_audit/2026-10-04-prereg-c1-watchdog.md`) sets the requirement in the W-state row (line 46) and the capture rule (lines 65–66):
- the capture binds the producer transaction or generation, the actual consumed pick, decision and streak bytes, the complete replay-input inventory, and the capture's as-of completion interval;
- it demonstrates that the relevant writers **could not interleave** the capture;
- it captures actual persisted outputs, never a recomputed expected state;
- matching hashes across repeated reads, a sleep, an idle PID or an assumed cron finish is not closure;
- tests pause or kill the actual producer between saves and require rejection of unclosed evidence, alongside a completed no-fault case;
- the watchdog takes only owned locks, and never repairs or blocks a production writer.

**Status:** a draft. It is production code once built, reviewed until SIGN, then D7.

## 1. The writers of W-state's inputs (inventory at `5c87de6`)
W-state replays the season from each date's pick file (`result`, `slot_results`) and decision (`scoreable`, `action`), and compares the result with `streak.json` (streak and saver). Every writer of those files:

| Writer | Call sites | Dates it writes | Holds `scoring_lock`? |
|---|---|---|---|
| Result scoring, cron (`check-results`) | `cli.py` ~2377–2391 | yesterday (and older, chronologically, with `--allow-stale-scoring`) | **yes** |
| Result scoring, daemon (`run_result_polling`) | `scheduler.py` 2430–2493 | today (polls until the cap hour, about 05:00 the next day) | **yes** |
| `save_nonterminal_result` | `scheduler.py` 2338–2346 | today | **yes** |
| `reconcile_results` (pick results plus the streak replay) | `picks.py` 1208–1250 | dates before their cutoff | **yes** |
| `update_streak` / `save_streak` | `picks.py` 576–584 (called from the scorers above) | `streak.json` | inside the callers' lock |
| Selection and preview saves | `orchestrator.py` 403/421/446, `cli.py` 1216/1240/1258/1356 (`run`, `preview`) | today and tomorrow | **no** |
| Delivery, lock and commit saves | `scheduler.py` 968–1102, 1709, 2089 | today | **no** |
| Commit decision (`_write_commit_decision`) | `scheduler.py` 683 | today | **no** |
| End-of-day MDP skip decision (`_write_endofday_skip`) | `scheduler.py` 775, called at 3147 after result polling | **the day just played, possibly after midnight**, because polling runs to the cap hour | **no** |
| Saver transitions | `saver_state.py` (its own flock and transition log) | saver state | its own lock |

**Consequence:** a snapshot taken under `scoring_lock` alone does not exclude the unlocked writers. Restricting the capture to "finalized dates" is not a proof either, because the end-of-day skip writes a past date's decision outside the lock, after midnight.

## 2. Options
- **A. One state lock for every writer of W-state inputs, plus a producer-issued snapshot after each write.**
  - **The lock:** `scoring_lock` is generalized into a reentrant (per process, per thread) `state_lock`. Every writer in §1 takes it around its read-modify-write: the scorers already do; the delivery, commit, end-of-day skip and selection or preview saves are added.
  - **The snapshot:** before releasing the outermost hold, the writer publishes an immutable **generation**: a manifest (generation number, writer, the lock's acquire and release instants, the full input inventory with each file's sha256) whose files are stored content-addressed (`blobs/<sha256>`, written once). Snapshotting is a few hundred small files per season, and unchanged blobs are already stored, so the cost is milliseconds.
  - **For W-state:** it reads only the latest complete generation. It is current only if `state_generation` (bumped inside the lock) still equals its number.
  - **Pros:** a real closure. The capture **is** the writer's own transaction output.
  - **Cons:** it touches the scheduler's hot write paths. Delivery writers may wait sub-second for a scorer. Nested acquisition must be reentrant, because `flock` on a second open file in the same process self-deadlocks.
- **B. A sequence lock (writers bump an odd/even generation around each write; the capture copies between two equal even readings).**
  - **Pros:** writers never wait.
  - **Cons:** it still requires every writer to participate (the same invasiveness as A). The capture is then the watchdog's own copy, not producer output. The registration accepts "a producer-issued snapshot" explicitly, and other mechanisms only if separately justified.
- **C. Scope restriction without writer changes.** Rejected by §1: an unlocked writer can modify past dates.

## 3. Recommendation: Option A
- **`bts.state_lock`:** `state_lock(picks_dir, writer)` takes the same file as today's `scoring_lock` (`.scoring.lock`), so older binaries still serialize with new ones during a deploy transition. A thread-local depth counter makes it reentrant: only the outermost hold takes `flock` and, on exit, publishes the generation. `scoring_lock` becomes an alias, so its existing semantics are unchanged.
- **Snapshot publication:** inside the outermost hold, after the writer's saves.
  - **Location:** `data/health_state/state_generations/` holds `gen-<n>.json` and `blobs/`, published with `bts.receipt_io` (tombstone discovery applies).
  - **The inventory:** every `data/picks/<season date>.json`, every `<date>/decision.json` and `.policy_shadow.json` that W-state consumes, `streak.json`, and the saver state file.
  - **A failure:** the generation is unavailable; the writer's own outcome is unchanged (guarded, as in P1/P2).
- **Wrapping the unlocked writers:** each call site in §1 wraps its existing read-modify-write in `state_lock`, with no other change. Writes that already hold `scoring_lock` gain the snapshot at release.
- **Generation counter:** `state_generation` in the lock directory is incremented inside the lock at the outermost acquire. A W-state reader compares it with the latest manifest's number. If they differ, a writer ran after that snapshot without completing one, which means in progress or crashed: pending, and an overdue alert after a bound.

## 4. Tests (gate 1–2 shaped)
- **Production-shaped completed case:** run each real writer path (scorer, delivery commit, end-of-day skip, preview) in a fixture. Each yields exactly one complete generation whose blobs equal the persisted bytes.
- **Pause or kill between saves:**
  - inject a pause or `SystemExit` in a writer between its pick save and its streak or decision save. The generation counter moves ahead of the last manifest, and the capture is rejected as unclosed;
  - in a separate test, a concurrent writer blocks until the first releases.
- **Reentrancy:** a scorer that calls `save_pick` inside its hold publishes once, with no self-deadlock. A thread or process without the hold blocks.
- **Behaviour unchanged:** the existing scheduler, check-results, reconcile and preview suites pass unchanged. Lock waits are bounded (scoring is sub-second).
- **Failure isolation:** a generation-publication fault leaves the writer's files, outcome and exit code unchanged, and no discoverable generation.

## 5. Open questions for the design review
1. Is wrapping the scheduler's delivery, commit and selection writes in the lock acceptable production risk? The alternative is B, with its caveats.
2. What is the precise inventory W-state needs? Is the saver state file in or out, given the separate saver comparison?
3. How should a deploy transition work, with old binaries holding the old `scoring_lock` but not snapshotting? Proposed: a generation is only "current" if the counter exists and matches, so the first post-deploy write establishes it.
4. Retention and size of `blobs/` (proposed: no pruning in 2027; measure).
