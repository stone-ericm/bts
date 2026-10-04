# C1 rank-1 prerequisites: receipted static capture and the all-player binding decision (no blend)

**Status:** design rev 1, 2026-10-04, for Codex design review (at most 2 rounds, then freeze). The capture change is production code: reviewed until SIGN, and shipped only with Eric's D7 approval.
**Cycle:** C1 (`docs/sota_audit/2026-10-04-c1-cycle-index.md`). Eric's D4 ruling includes only these prerequisites: **no blend is built or fitted in C1.**
**Sources:**
- frozen W2.3 memo `docs/sota_audit/2026-10-04-field-mlb-forecast-benchmark.md` §§2, 6;
- decision memo D4 rank-1 text;
- checklist §D;
- the plan's open item "static-capture fetch log retention".

## In plain words
- **The problem.** MLB's own "% chance to hit" might help us one day, but two things block any test of it.
  - **No record of fetches.** We don't log when we fetched it, so we can't show it was available before our lock.
  - **No game binding.** The sheet that covers every player doesn't say which game a number belongs to.
- **The fix, two parts:**
  - **Receipts.** The existing capture writes a receipt for every fetch, and stops for good on any 403/429 (MLB blocking us).
  - **A binding check.** One outcome-free check on the 2026 captures settles whether the all-player number can be tied to a game.

## Part A. Receipted capture: a production change to `src/bts/leaderboard/static_capture.py`
**Today** (read 10/04):
- the cron runs every 30 minutes, anonymously (no contest cookies) against `mlb-play.mlbstatic.com`;
- it stores a file only when the content's sha256 changes;
- the log has no timestamps, HTTP status or hashes;
- a 403/429 is an ordinary error, retried 30 minutes later.

It is still running: `players` has 3,788 stored versions, the latest on 10/04.

**Rulings:**
- **A1, one capture.** Change the production capture rather than run a second one, which would double requests to MLB.
- **A2, receipts.** Append-only, one per feed per run, in `data/leaderboard/static_snapshots/_receipts/<UTC date>.jsonl`. That path is in the restic archive set; `~/logs` is not.
  - **An intent line before the request**, so a crash mid-request still leaves a record: `{run_stamp, feed, url, started_utc}`.
  - **A completion line after it:**
    - `ended_utc`, `duration_s` (monotonic);
    - `http_status`;
    - `outcome`: `stored` | `unchanged` | `invalid` | `fetch_error` | `rate_limited`;
    - `bytes`, `wire_sha256`, `decoded_sha256` (hashed before validation);
    - `previous_marker_sha256`, and `stored_path` or `"unchanged"`;
    - the `Date`, `Last-Modified`, `ETag`, `Age` and `Cache-Control` headers when present.
  - **How a receipt binds to stored bytes:** the decoded sha256 matches the gunzipped stored file and the `.last_sha256` marker. Stored `.gz` bytes embed a timestamp and cannot bind.
- **A3, the 403/429 stop** (C1 stop rule 4).
  - Abort the remaining feeds in that run and write a persistent `_receipts/STOP_403_429.json` (time, feed, status).
  - Every later run refuses with exit code 3 while it exists.
  - Only Eric's recorded decision removes it.
  - The watchdog (rank 2) alerts on its presence; this is added there as a check.
- **A4, cadence:** unchanged, every 30 minutes, now registered. **Limit:** pre-lock availability can be shown only to within one capture interval.
- **A5:** keep the module stdlib-only, as now.
- **A6, User-Agent:** refresh the browser User-Agent string before the 2027 season (a checklist item). The current string will be over a year old.

**Gate:**
- **Tests first** (`tests/leaderboard/test_static_capture.py` harness, injected fetch). Each must fail before the code exists:
  - one receipt pair for each outcome;
  - a decoded-sha binding to a stored `.gz`;
  - an intent line left behind by a simulated crash;
  - a 403 and a 429 each writing the stop marker and aborting the run;
  - a later run refusing while the marker exists;
  - unchanged fetches still receipted.
- **Then:** deploy-gating Codex review until SIGN, Eric's D7 approval, and a deploy inside a sleep window.
- **After the deploy, on the box:** one receipt pair per feed from the next cron run.

## Part B. Can the all-player `probabilityStarter` be bound to a game? (outcome-free; decided before any 2027 outcome)
**What is known:**
- `players` carries `probabilityStarter` for every player but has no round id.
- `most_selected_players` rows carry a `roundId` and cover today's and tomorrow's rounds.
- Nothing in the repo shows when the all-player value rolls over or how it treats doubleheaders.

**The check:**
- **Data:** the stored 2026 captures, 7/04–9/27.
- **Pairing:** each stored `players` capture is paired with the latest stored `most_selected_players` capture at or before it. Each player present in both is classified as matching:
  - today's round row exactly;
  - tomorrow's round row exactly;
  - both (equal values);
  - or neither.
- **Rollover:** per date, the earliest capture at which the all-player value switches from matching today's round to matching tomorrow's.
- **Doubleheaders:** a player with two games in one round cannot be bound to a game, so those players are counted and excluded, as in W2.3.

**Decision rule, fixed now:**
- **Bindable:** at least **99%** of comparable (player, capture) pairs match exactly one round, *and* that round is a fixed function of capture time on at least **95%** of dates (one rollover rule).
- **Otherwise not bindable:** any later rank-1 rule stays restricted to the most-selected sheet's coverage.

**Exposure:** no outcome is read. A register row (X-33) is still published before the run, because X-23 excluded this field.

**Execution:** the code in `scripts/audit/c1_r1/` is written test-first and runs on the box through the C1 launcher (`c1-r1-binding`), **declared 0.5 CPU-hours**. It reads both `.json` and `.json.gz`.

## Out of scope in C1
No blend, no weights, no outcome read, no new authenticated traffic. Rank 1's later registration (if Eric ever selects the blend) inherits this capture and this binding decision.

## Limits
- **Availability, not freshness:** receipts witness retrieval availability, not the provider's model-generation age.
- **Coarse timing:** 30-minute cadence.
- **Run-start stamps:** file names stay run-start stamps; receipts give per-feed times.
- **The binding check uses 2026 captures:** if MLB changes the sheet's behaviour in 2027, it must be re-run on early-2027 captures before any 2027 outcome is read.
