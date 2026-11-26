# C1 rank-1 prerequisites: receipted static capture and the all-player binding decision (no blend)

**Status:** **FROZEN 2026-10-04** after Codex trio design r1 BLOCK (C-E1–C-E3, X-E1) → r2 **SIGN WITH EDITS** (C-R2-E1–E2), all applied verbatim by script (reviews `docs/audit/2026-10-04-c1-trio-design-codex-r{1,2}.md`). All-player binding: **not established**. The capture change is production code: reviewed until SIGN, shipped only with Eric's D7 approval.
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
  - **A concordance diagnostic.** One outcome-free read of the 2026 captures describes agreement with round-tagged most-selected rows. It does not establish broader-sheet game identity; without independent binding evidence, that prerequisite is reported not established.

## Part A. Receipted capture: a production change to `src/bts/leaderboard/static_capture.py`
**Today** (read 10/04):
- the cron runs every 30 minutes, anonymously (no contest cookies) against `mlb-play.mlbstatic.com`;
- it stores a file only when the content's sha256 changes;
- the log has no timestamps, HTTP status or hashes;
- a 403/429 is an ordinary error, retried 30 minutes later.

It is still running: `players` has 3,788 stored versions, the latest on 10/04.

**Rulings:**
- **A1, one capture.** Change the production capture rather than run a second one, which would double requests to MLB.
- A2. Receipts are append-only intent/completion pairs for each request actually attempted, under data/leaderboard/static_snapshots/_receipts/<UTC intent date>.jsonl. Both lines repeat a unique run_id and attempt_id plus feed/url. Intent contains started_utc and is durably flushed before network starts. Completion contains ended_utc, monotonic duration_s, HTTP status when observed, outcome/error/completeness, wire_bytes/wire_sha256, decoded_bytes/decoded_sha256, and the observed Date/Last-Modified/ETag/Age/Cache-Control headers. Null denotes unavailable metadata; a partial body is never labelled a complete response. Hash wire bytes before decompression and decoded bytes before validation; the stdlib fetch seam exposes response metadata and body bytes instead of returning only decoded JSON.
  - A stored completion names an immutable, collision-free relative stored_path, stored byte length and stored sha256; gunzipping those exact stored bytes must reproduce decoded_sha256. Compression timestamps do not prevent binding to stored bytes. Record previous_marker_sha256 and resulting_marker_sha256 at this completion; historical receipts are not required to equal a later current marker.
  - An unchanged completion names the existing retained stored_path and verifies its decoded hash against the fetched payload; marker equality alone is insufficient. A missing/corrupt retained object is not unchanged success. Pre-instrumentation objects must be verified and explicitly referenced. Failed/invalid responses record available hashes and explicit null storage/marker-result fields, without inventing stored objects. Receipt or storage failure never certifies availability. Reader acceptance checks intent/completion identity and times, content binding and state transitions from the consumed bytes.
- A3. A stdlib single-writer lock covers preflight, request scheduling, receipt writes, dedupe and stop/marker updates. Every invocation checks the persistent stop under that lock before any request and rechecks before each later feed. Native urllib HTTPError codes 403 and 429, and any equivalent observed response status, are classified before generic fetch/body/validation errors. On observing either, durably record the limit and STOP_403_429.json, abort all remaining requests and exit 3. Do not read an error body before recognizing the stop. Every later invocation refuses with exit 3 while stopped; only Eric's recorded reset decision permits capture resumption. Failure to record required intent/receipt/stop state aborts without further requests and leaves the incomplete attempt unavailable pending explicit reconciliation. No fabricated attempted-fetch receipt is written for skipped feeds. The canonical marker is _receipts/STOP_403_429.json under the capture output root. Under the same writer lock, preflight also validates instrumented intent/completion reconciliation and recorded resets before any network request. A prior unreset 403/429 receipt or stop record refuses with exit 3 even if the marker is absent; rebuild the marker when possible without fetching. A missing completion, malformed/unreadable receipt or stop state, or failed required state publication refuses further requests with a nonzero exit pending recorded reconciliation. Missing/unreadable state is never treated as evidence of no stop. If independent retained evidence cannot rule out a limit for an unresolved attempt, capture resumption requires Eric's fresh recorded decision. Deleting a marker is not itself a reset; the recorded reset identifies the stop/attempts it resolves. Known unresolved limit evidence is also a stop input to the required watchdog and C1 cycle gate, independent of marker presence; capture reset and cycle resumption retain their separate owner decisions.
- Add W-capture-stop to rank 2 with a planted-stop positive alert and absent-stop no-alert control. C1 research execution also refuses while this stop is unresolved and the cycle has not been resumed by Eric; an alert alone does not satisfy the cycle-level pause. This integration gets its own production-code review/approval where applicable.
- **A4, cadence:** unchanged, every 30 minutes, now registered. Qualified receipts timestamp actual per-feed retrieval. Between attempts availability is unobserved; failed/crashed attempts can make the gap longer than one interval. Retrieval witnesses do not establish provider generation age or actual use by a pick decision.
- **A5:** keep the module stdlib-only, as now.
- **A6, User-Agent:** refresh the browser User-Agent string before the 2027 season (a checklist item). The current string will be over a year old.

**Gate:**
Red/green tests cover every outcome's correctly paired receipt, native urllib 403 and 429 paths through the transport seam, plain and wire-gzipped responses, wire/decoded/stored digest distinctions, unchanged-object references, missing/corrupt dedupe objects, marker transitions, intent-only crash, receipt/storage failure, abort of remaining feeds, refusal on the next invocation, and serialized competing invocations. All use synthetic bodies/clocks and patched transport. After production-code SIGN, Eric's D7 approval and deploy/canary checks, verify the next cron's attempted-feed receipt pairs and retained-byte binding. No live rate-limit probe is authorized. Add next-invocation tests for a durable 403/429 receipt followed by marker-publication failure or marker removal, an intent-only crash, and malformed/unreadable reconciliation state. In each unresolved case, assert no transport request; after a properly bound recorded reconciliation/reset, assert the permitted behavior with transport patched. Verify that known-limit evidence without the marker still reaches the required watchdog/cycle stop integration. No live probe is authorized.

## Part B. Can the all-player `probabilityStarter` be bound to a game? (outcome-free; decided before any 2027 outcome)
**What is known:**
- `players` carries `probabilityStarter` for every player but has no round id.
- `most_selected_players` rows carry a `roundId` and cover today's and tomorrow's rounds.
- Nothing in the repo shows when the all-player value rolls over or how it treats doubleheaders.

The one outcome-free 2026 read is a concordance diagnostic, not a Bindable gate. Pin an X-33 input manifest covering both plain/gzipped sheets and the required rounds/player/unit/schedule identity sources for 7/04–9/27. No outcome, grade, hit/miss or blend fit is read. The run is still declared 0.5 CPU-hours through the launcher.

Pair each stored players sheet with the latest whole most-selected sheet at or before its run-start stamp, without substituting older rows to obtain a match. Validate typed player/round identities, duplicates/conflicts and probabilities before comparison. Resolve dated rounds and report exact today-only, tomorrow-only, both, neither, missing-today, missing-tomorrow, invalid and unmapped counts. Exact-one discrimination requires both round values to be observed, valid and distinct; missing values do not count as inequality. Report player/date coverage, stale reference-sheet ages and all missing dates. Report the observed intervals containing match changes, not exact rollover times; deduped run-start stamps do not witness every successful fetch or synchronous provider rollover.

The 99% exact-one and 95% date-consistency figures are descriptive screens only. No rollover cutoff or unrestricted time function selected on these same captures passes an identity gate. Agreement on most-selected players does not establish the rule for unlisted players. Without independent, outcome-free evidence binding the broader field to a round for the intended players and times, disposition is not established and later rank-1 work remains most-selected-only. Even a resolved round requires complete, contradiction-free player/squad/unit/schedule evidence for unique-game inference; unknown multiplicity, doubleheaders and ambiguous rows remain unavailable. Label inferred links as inferred, never witnessed. Provider target and eligibility remain independent open prerequisites; receipts establish retrieval only.

Publish X-33 before any new field read because X-23 excluded it. Receipt-only instrumentation has no outcome exposure; the historical concordance diagnostic does. This registration authorizes no new acquisition to resolve a missing identity contract and no blend. Any later proposed positive broader-sheet binding contract must be independently specified and reviewed before outcomes rather than inferred from favorable concordance percentages.

**Execution:** the code in `scripts/audit/c1_r1/` is written test-first and runs on the box through the C1 launcher (`c1-r1-binding`), **declared 0.5 CPU-hours**. It reads both `.json` and `.json.gz`.

## Out of scope in C1
No blend, no weights, no outcome read, no new authenticated traffic. A later rank-1 registration carries this capture contract and the not-established broader-sheet disposition unless it first supplies the independently specified, reviewed, outcome-free identity/target contract required by Part B; this diagnostic selects no blend or binding rule.

## Limits
- **Availability, not freshness:** receipts witness retrieval availability, not the provider's model-generation age.
- **Observation gaps:** planned 30-minute cadence; actual successful retrieval times come from qualified receipts, and failures can leave longer gaps.
- **Run-start stamps:** file names stay run-start stamps; receipts give per-feed times.
- Changes in 2027 sheet behavior invalidate carried identity assumptions. Before broader-sheet use, independently resolve and freeze its outcome-free identity/target contract on admissible sources; no retrospective outcome fit selects that contract.

## Freeze manifest and cross-design rules (trio review X-E1)
Before each authorized fitting/evaluation/diagnostic run, publish the appropriate exposure row and an outcome-free freeze manifest: reviewed registration/code commit and hashes; old serving recipe, model-training/retraining schedule, active blend and aggregation/fallback definitions; configuration/environment including calibration/deterministic/seed flags; fixed calendar/as-of/eligibility rules; and count/identity/reader artifacts as applicable. Future forecast/model/input hashes are recorded with each immutable capture. Pin/hash the exact consumed fitting/outcome/input bytes at the declared freeze, and parse those same bytes. A missing pin, mismatched hash, unsupported schema or unregistered recipe change refuses acceptance; naming a directory is not a pin. Ordinary model retraining under the frozen schedule is allowed and its artifact hashes are recorded. A recipe change does not trigger a post-result refit, window reset or silent pooling; it is reported and the affected study is inconclusive pending a separately approved prospective registration.

X-32 covers 4a's fit and test; X-34 covers rank 3's historical fitting and prospective evaluation; X-33 covers only the outcome-free broader-field/identity diagnostic. The index distinguishes X-33 from receipt-only capture. These rows are published before their respective reads, with no 2026 candidate outcome test. All new artifacts/readers preserve both plain and gzipped static input support and the declared missingness/fallback rules.

Each forecast study has one fixed primary comparison and its declared regression gate. Their shared dates create dependent evidence, not replication; secondary metrics are descriptive and cannot select another map/count specification or a combination. Keep 4a, rank 3 and 4b comparisons separate under D2. No fitted 4a output is supplied to rank 3 and no rank-3 output is supplied to 4a. Independent result acceptance and Eric's D7 approval precede any named production change; applicable policy replay and D1 trade approval remain separate. The approved C1 launcher, cumulative caps, sleep-window/production-safety and calendar stops apply. An unresolved rank-1 403/429 stop pauses C1 until Eric's recorded resumption and keeps capture separately disabled until his recorded reset.
