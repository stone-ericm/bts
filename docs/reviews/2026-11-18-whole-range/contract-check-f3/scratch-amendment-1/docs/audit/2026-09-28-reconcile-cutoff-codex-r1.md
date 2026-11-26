## Round

r1 — reviewed HEAD 134eaf0 + working-tree diff, including the new `tests/test_reconcile_cutoff.py` file (absent from the tracked `git diff` output).

Verification: 83 tests passed in `tests/test_reconcile_cutoff.py`, `tests/test_picks.py`, and `tests/test_scoring_lock.py`; another 39 passed in `tests/test_streak_replay.py`, `tests/test_check_results_wait.py`, and `tests/test_shadow_eval.py`. Both runs used `UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest <files> -q`. `git diff --check` passed. The counterexamples below are source-derived, not executed reproductions. No mutations or extra test files were created under the read-only constraint.

## Findings

1. **BLOCKER — the only correction attempt is now too early, and a missed attempt cannot recover.** `src/bts/picks.py:1115-1125`; `scripts/cron-setup-hetzner.sh:53-54`.

   A settled August 20 hit still appears as a hit at the August 21 02:00 reconcile. MLB corrects it to an error at 04:00, inside the allowed correction window. There is no later scheduled reconcile that morning. The August 22 run skips August 20 permanently. Previously that later run could recover this legitimate correction. `check-results` and the daemon do not fill the gap: both skip terminal production results (`src/bts/cli.py:2241-2245`, `src/bts/scheduler.py:2392-2396`), even when their polling deadlines have not expired.

   The same loss of recovery affects slot backfill: the daemon can store an early terminal hit without `slot_results` (`src/bts/scheduler.py:2438-2440`). If a game is still pending at 02:00, reconcile gets `None` and skips it. Once the cutoff passes, this result never gets a final check or slot backfill through these paths. A missed 02:00 job similarly leaves no subsequent opportunity.

   **Suggested fix:** pair the guard with a reviewed finalization procedure: retry/final capture before the cutoff, persist evidence of successful observation, and surface dates whose final check was missed. Recover those dates using trusted pre-cutoff evidence or the contest's authoritative result. A single 02:00 observation cannot establish the result through 08:00; moving it nearer 08:00 alone still leaves an observation gap. Keep the prohibition on using later mutable MLB data for historical repair. Add a two-run regression with a correction between 02:00 and 08:00, plus pending-at-02:00 and missing-slot cases.

2. **BLOCKER — the guard checks entry time, so a request crossing 08:00 can still apply a post-cutoff change.** `src/bts/picks.py:1109`, `src/bts/picks.py:1117-1126`, `src/bts/picks.py:1133-1151`.

   Start a manual reconcile at 07:59:59 ET. The date passes the guard. Its schedule request takes several seconds; the subsequent live-feed request sees a hit changed to an error at 08:00:01. The resolver returns the miss, and Phase 2 writes it without checking the clock again. Every comparison uses the original `now`. The network is explicitly outside the lock, and the resolver makes sequential schedule/feed requests (`src/bts/picks.py:812-825`, `src/bts/picks.py:1035-1043`), so this does not require a long outage. The lock prevents concurrent writes but does not enforce the deadline.

   **Suggested fix:** use an injectable clock and reject observations that finish after the cutoff unless they carry trusted pre-cutoff provenance. Recheck eligibility at the locked mutation boundary to enforce the stated freeze policy. Add a test where time advances across 08:00 during resolution and neither the settled result nor its slot outcomes changes.

3. **SHOULD — a daytime/late-night reconcile still rolls local streak state back before today's already-scored result. Pre-existing defect.** `src/bts/picks.py:1160-1164`, `src/bts/picks.py:599-601`.

   With one hit yesterday and one terminal hit today, `streak.json` correctly holds 2. Run reconcile at 23:00 ET today: the new guard produces no proposals, but the unconditional replay excludes today's file and saves 1. If today's miss consumed the local saver, replay can also restore `saver_available=True`. The new cutoff is therefore not a no-op guarantee for a post-cutoff invocation. This behavior exists in HEAD too; it is not attributed to the new guard. It affects the local model/replay state, distinct from the contest-account authority (`src/bts/contest_state.py:1-6`).

   **Suggested fix:** preserve terminal results already applied today when replaying current state, while continuing to exclude unplayed previews and future dates. Add manual afternoon/23:00 tests asserting both streak and saver state, including a no-proposals run.

4. **SHOULD — the new `now` argument has an undocumented naive-datetime failure.** `src/bts/picks.py:1093`, `src/bts/picks.py:1109-1117`.

   `reconcile_results(path, now=datetime(2026, 8, 21, 2))` first interprets the naive value in the host timezone through `astimezone(et)`, then compares that unchanged naive value with an aware cutoff and raises `TypeError`. This affects callers of the newly exposed argument; the CLI's default aware clock is safe.

   **Suggested fix:** define an aware-only contract and reject naive values explicitly before processing, or document and implement one deliberate interpretation. Normalize once to ET and test both naive rejection and equivalent aware UTC/ET inputs.

5. **SHOULD — the tests do not establish timezone-independent or complete freeze behavior.** `tests/test_reconcile_cutoff.py:40-69`; `tests/test_scoring_lock.py:207-232`.

   All injected cutoff times are August ET values. A mutant using a fixed UTC-04:00 cutoff year-round would pass these cases but reject a winter 07:30 ET correction because its cutoff falls an hour too early. A mutant deriving `today` from `now.date()` also passes because no injected timestamp has a different UTC and ET calendar date. No test advances time across the network phase, asserts that an expired date performs zero resolution calls, or checks post-cutoff streak/saver preservation. The lock test probes `save_pick` only, despite its assertion claiming coverage of pick/streak writes.

   **Suggested fix:** add equivalent UTC inputs, UTC/ET midnight disagreement, winter and both DST transition dates, an advancing clock, and persisted result/slot/streak/saver assertions. Probe `save_streak` under the lock separately. Use fixed calendar fixtures: the re-pinned existing tests still derive dates from `date.today()`, and their hit-replay expectations can fail on January 1 because yesterday belongs to the prior replay season (`tests/test_picks.py:727-728`, `src/bts/picks.py:599`, `src/bts/picks.py:1161`).

**Other requested checks and limits**

- For aware inputs and a run that stays on one side of the boundary, the calendar calculation is correct by inspection: `ZoneInfo("America/New_York")` handles DST, the cutoff itself is unambiguous at 08:00, and `>=` freezes at exactly 08:00. Positive lookbacks admit only yesterday before 08:00; after 08:00 they admit no date. Today's results are never reconsidered because the loop starts at 1. Increasing the lookback cannot restore missed correction coverage.
- Source-level mutation assessment: removing the guard would fail the day+6 and 08:00 tests; moving the cutoff one day or one hour in either direction would fail an existing pre-cutoff/boundary case. A UTC cutoff would fail the August 07:59 ET case. The narrower timezone mutants in finding 5 survive. These are predictions from the assertions, not measured mutation-test results.
- The re-pinned tests still exercise actual hit-to-miss correction, preview/shadow exclusion, partial-void +1 replay, and a locked pick write. Existing saver replay tests pass. Skipping old dates preserves stored hit/miss/void and DD slot outcomes; it also deliberately removes their live-feed slot backfill. A legacy DD hit without slots continues to count as +2 (`src/bts/picks.py:841-851`), so repairing ambiguous old partial-void records needs trusted evidence rather than a late feed fetch.
- I found no ordinary alternate writer that re-grades an already-terminal production result: `check-results` checks before resolution and again under the lock (`src/bts/cli.py:2242`, `src/bts/cli.py:2308`); daemon early/final writes and nonterminal marking also refuse terminal overwrite (`src/bts/scheduler.py:2337`, `src/bts/scheduler.py:2433`, `src/bts/scheduler.py:2479`). Unresolved production is a separate residual gap: manual `check-results` can first-grade yesterday after 08:00 or a two-day-old pick from today's feed (`src/bts/cli.py:2264-2279`); `--allow-stale-scoring` permits still older first grades. This patch does not make those results cutoff-authenticated.
- Ordinary shadow reconciliation skips terminal shadow results and writes only the shadow file (`src/bts/cli.py:2182-2200`). Shadow backfill can re-evaluate historical production for its report, and rewrite historical shadow results from cached/current MLB data, but its normal generated manifest applies only to shadow paths (`src/bts/shadow_eval.py:689-721`, `src/bts/shadow_eval.py:775-790`). Canonicalization derives `actual_hit` from PA data and writes a separate parquet; it retains `pick_file_result` for comparison and does not rewrite production picks (`scripts/canonicalize_realized_picks.py:344-365`, `scripts/canonicalize_realized_picks.py:391-400`, `scripts/canonicalize_realized_picks.py:695`). Those derived outcomes can still differ from contest-frozen results.

## Verdict

BLOCK — the patch prevents the demonstrated day+6 overwrite, but permanently loses valid later-morning corrections and still admits post-cutoff data when resolution crosses 08:00.
