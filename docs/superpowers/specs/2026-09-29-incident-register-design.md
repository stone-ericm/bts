# 2026 incident register (W1.5): design v2

**Date:** 2026-09-29.

**Plan item:** W1.5 in `docs/superpowers/plans/2026-09-14-season-wrap-plan.md` (approved 9/22): "7/16 singleton-slate gap · 8/11 MLB auth flap · 8/13 silent pass (Warmup) · 8/30 late pick (Kwan) · 9/03 idle (all-skip table) · any private-mode/tail anomalies through 9/27. Each → mechanism, fix status, **failure-path fixture** that reproduces it, residual gap. Feeds W4 rank 2."

**History:**
- v1 (`88a4bc7`) → Codex design r1: **BLOCK**, 4 blockers + 5 should-fixes (`docs/audit/2026-09-29-incident-register-codex-design-r1.md`).
- v2 applies all nine (§14 maps each finding to a section).

**Status:** draft for Codex design r2. The review gets no data access.

## 1. Purpose and success
The register is the evidence base for W4 rank 2, the watchdog and restore checks. It answers five questions:
1. What went wrong in the 2026 deployment?
2. How and when was each failure detected, notified, mitigated and recovered?
3. Does a test reproduce each historical failure?
4. Does a test still guard today's code against it?
5. What remains open?

It measures no model quality, calibration or policy value.

**Success:**
- (a) Every candidate raised by any discovery route (§6) ends with one disposition (§2), with evidence, or with an explicit `unresolved_candidate` reason. The claim is "all candidates within the evidenced coverage", with coverage stated per source (§4). It is never "every live incident": retention does not support that claim.
- (b) Every fixed Tier-A observed incident records two things separately:
  - *historical reproduction*: a fixture fails on the pre-fix code with the incident's observable symptom (§9.2);
  - *current defence*: a single-mechanism mutant at the pinned baseline is killed on the production path (§9.3).
  A surviving mutant gets a named classification (§9.5), never "verified".
- (c) Every plan-named incident gets an executable fixture: reproduction, characterization or invariant. Otherwise it gets an explicit `deferred`/`unavailable` disposition with a reason.
- (d) Unfixed defects whose contract is fixed (§10) get strict expected-failure fixtures that pass the §9.7 validation.
- (e) The memo passes a Codex result review. A published fixture gap is an honest deliverable, not a W4 repair acceptance.

## 2. Scope, dispositions, classes
**Scope:**
- the deployed BTS service: box, cron, systemd units, deploy workflow, DMs, dashboard;
- its research streams: shadow v1/v2, skip-policy shadow, live-forward #16, D8 research capture, leaderboard captures;
- the preservation and restore dependencies they rely on, on any host (e.g. the 6/05 MacBook loss of `pooled_bins_run`).

Window: 2026-03-25 (first round) through the W0.7 freeze, 2026-09-28 08:00 ET.

**Dispositions** (exactly one per candidate):

| Disposition | Meaning |
|---|---|
| `observed_incident` | A live deviation from the intended contract, evidenced by a machine observation or a contemporaneous operator report (§3) |
| `deployed_latent_defect` | A defect in deployed code or config with no evidenced firing; fixed or unfixed. Kept when it matters to watchdog or restore coverage |
| `near_miss_control_held` | A hazard was present but an existing control held (e.g. the 7/28 stale preview on a skip day that the scoreable gate ignored). Listed as a control, not counted as an incident |
| `pre_ship_exclusion` | Introduced and fixed inside an unshipped change, or a test-only defect. Listed, not counted. Tests that exposed a deployed flaw, and tooling that really runs production or restore jobs, are **not** excluded |
| `unresolved_candidate` | Evidence is insufficient to decide. Kept, with the missing evidence named |

**Effect classes.** One episode can carry a primary class plus linked secondary ones.

| Class | Meaning |
|---|---|
| **D** Delivery | Missed, late, wrong, duplicate or refused pick delivery |
| **E** Entry | Entry-check false or missed alarms; partial or absent entry the system should have caught |
| **G** Grading and state | Local grade, streak or saver wrong; local and contest state diverging |
| **A** Alerting | False alarm, storm, wrong advice, missing alert, attempted-but-failed notification |
| **L** Liveness | Idle, restart loop or thrash, crash, OOM, hang, latency |
| **P** Decision contract | The policy layer violates its own contract |
| **R** Research integrity | Research-stream data lost, corrupted or stranded |
| **X** External dependency | MLB statsapi and the contest API, Savant, Open-Meteo, Bluesky, GitHub Actions, R2/restic, Healthchecks: shapes, flaps, lags, in-place re-scoring |
| **S** Preservation and deploy | Data loss, backup or restore failure, failed or rolled-back deploys, config drift, unrecoverable artifacts |
| **U** Display | Dashboard or report showing a false state |

**Tiers.**
- **Tier A:** delivery, entry, contest or state impact; an alert storm; a liveness failure; **loss of recoverability**; or a **material research-integrity failure** (data lost, or a protocol read affected).
- **Tier B:** display and wording issues, research issues without data loss, and near misses.
- **Impact unknown:** `tier_pending`, listed in its own section until evidence settles it.
- "Residual reaches production" (Tier B → A promotion) means the unfixed residual can change delivery, entry, local or contest state, alerting, or recoverability on the deployed service.

## 3. Evidence kinds and strength
Every evidence item carries a `kind` and a `locator`:
- **kind:** `machine_observation` (log line, state-file field, deploy run, contest-ledger line), `contemporaneous_operator_report` (commit body or memo written within 48 h of the event describing what was seen), or `inference` (derived from code or later analysis);
- **locator:** path or commit or run id, plus line, byte offset or JSON path; plus the pointer trail to the underlying artifact.

A memo or memory note that repeats a claim with no locatable artifact behind it is `reported`. It cannot on its own make a candidate `observed_incident`.

## 4. Source inventory (availability before enumeration)
Before any sweep, one inventory table lists for every source:
- host and path, or unit;
- retained interval (first and last record);
- rotation, overwrite or archive semantics (an overwritten status file is a latest-state object, not a history);
- content hash;
- record count;
- **unavailable intervals with reasons**.

Missing evidence stays `unknown`/`unavailable`. It never counts as "no incident". Sources:

| id | Source | Notes |
|---|---|---|
| R1 | Repo documents: `docs/audit/**`, `INCIDENT.md`, `ARCHITECTURE.md`, `docs/optimization-ideas.md`, plans and specs | Read for incident evidence only. A quoted outcome statement cites its existing exposure row (X-05, X-09, …); unrelated analyses are not consumed |
| R2 | All git commits from the first commit (2026-03-29) to the freeze, **including docs/config/artifact-only commits** | Identity + changed paths inventoried before any exclusion |
| R3 | GitHub issues and PRs, all states, paginated to the first | — |
| R4 | GitHub Actions deploy runs, **all outcomes**, paginated | Job and step conclusions: test gate, SSH deploy, canary, auto-rollback step executed or not |
| B1 | W0.7 frozen snapshot `final-20260928/` on the box | Journals (`bts-scheduler`, `bts-live-forward-capture`, `-resolve`, `bts-leaderboard`); `cron.log`; `config/` (TOML + 9/14 snapshot, crontab + snapshot, installed units/timers); `deploy_history.txt`; `data/picks/**`; `data/health_state/**`; `data/validation/**` (live-forward official + D8 research roots, shadow statuses) |
| B2 | Box sources **not** in the snapshot, copied read-only into a new evidence bundle (§8) with a manifest before extraction | `~/logs/heartbeat.log`, `backup.log`, `park_drag.log`, `static_capture.log` (and rotations); user-unit journals not exported by W0.7: `bts-dashboard`, `bts-shadow-prediction`, `bts-lineup-collect`, transient units; restic snapshot list for the `ops`/`archive`/`season2026` sets (ids, times, tags only); systemd unit and drop-in files as installed |
| B3 | W1.1 accepted ledger build `…-compile2-dd430abe/` | Bound by build directory + bundle manifest sha `6ebb0953…` |
| O1 | External, owner-gated, **not in default scope**: Bluesky DM history of the bot account (would give send times before 5/11); healthchecks.io notification history | Each needs Eric's OK. Without it, those intervals are `unavailable` |

**Known coverage limits:**
- Journals start 2026-05-11. For 3/25–5/10, evidence is R1–R4 plus pick files, archives and `cron.log` if retained. The inventory states the exact retained interval of each.
- `decision.json` starts 6/23.
- Health state files are overwritten latest-state objects.

## 5. Exposure contract (register row X-20, predeclared and pushed before any B-source read)
**5.1 Field-limited extraction.** Box and external sources are read only through `scripts/audit/incident_register/` extractors, built test-first (§12).
- Each extractor parses one source kind and writes `events.jsonl` with **only**: `source_id`, `locator`, `ts_utc`, `date_et`, `event_kind` (closed enum), `severity`, `unit_or_job`, `count`, `template_id`, plus whitelisted booleans (e.g. `has_msg_id`).
- `event_kind` comes from a fixed template table built from the code's own message statements: scheduler prints, health source names, CLI outputs, systemd lifecycle lines. Capture groups are discarded except whitelisted non-outcome fields: unit names, NRestarts, minutes-to-cutoff.
- **No message body, exception payload, probability, player result, streak value or calibration number is ever written**, and none goes into any agent prompt.

**5.2 Unknown shapes.** A line that matches a generic failure token (`Traceback`, `Error`, `Exception`, `Killed`, `failed`, `CRITICAL`, `WARN`) but no template becomes `event_kind = unclassified` with its locator only.
- It is reviewed once through a redacting viewer. The viewer masks every digit run, percentage, the tokens `hit|hits|miss|missed|void|HIT|MISS|VOID|No Hit|Pass`, and `\d+-for-\d+`.
- The reviewer then either adds a template (with a test) or records the line as `unclassified_reviewed`. Raw lines are never copied into evidence.

**5.3 JSON state readers.** Field whitelists per file type:

| File type | Whitelisted fields |
|---|---|
| Pick files | Delivery and lock fields (`notification_sent`, `notification_id` presence, `delivered_at`, `delivery_attempted`, `bluesky_posted`), slot presence, game times, `projected_lineup`, run timestamps. **Not** `result`, `slot_results`, `p_game_hit`, streak |
| `decision.json` | `schema_version`, `action`, `source`, `scoreable`, `delivery_status`, `objective`, `degraded_reason`, timestamps. **Not** streak/state values |
| `scheduler_state.json` | Lock, commit, skip-candidate presence, refusal counts, refresh durations, `result_status` presence |
| Archives | Prefix, reason, timestamps |
| Health state | Keys and statuses (`status`, `updated_at`, `sent_sources`, `ok`/error category, per-set timestamps). **Not** metric values |
| Account state | Presence and timestamps only. A needed streak/saver value requires an **amendment** to X-20 first |
| Live-forward / D8 roots | Presence, schema, date, status and acceptance fields, timestamps. **Not** forecasts, labels or result payloads |

**5.4 Ledger reads (B3).**
- (i) Per-selection `date, slot, row_kind, finalization, commit_status, entry_status, delivery_confirmed, delivery_basis` and the anomaly rows (unfinalized 3, unobserved 7, unconfirmed 12, quarantined 3).
- (ii) The `local_vs_contest_disagreement = true` rows, projected to exactly `(date, slot, local_normalized, contest_normalized, comparison_basis)`.
- If the row count ≠ 2: **stop** and amend. A NO_HIT/HOLD pair names a symptom; it does not prove a mechanism.

**5.5 Repo documents.** Restricted to incident evidence; quoted outcome statements cite their exposure rows.

**5.6 Tests.** The extractor tests feed synthetic inputs through every parser:
- calibration WARN/CRITICAL alerts carrying rates;
- result lines (`All picks have hits! Streak: 5`, `Result already scored elsewhere (miss)`);
- multi-line tracebacks, malformed JSON, unknown lines.

A property test asserts that no output field outside the whitelist exists, and that no banned token or out-of-whitelist number appears anywhere in the outputs.

**5.7 X-20 disposition text.** "A limited, registered outcome exposure for operations diagnosis (the two disagreement rows); everything else is operational metadata. Any model or policy idea motivated by a register finding still validates prospectively in 2027 (D3). Ops candidates (W4 rank 2) are validated by failure-path fixtures, not 2026 outcomes."

## 6. Discovery: three routes, then reconciled
**Route H — history** (R1–R3):
- Every commit is inventoried (hash, date, paths, subject/body) and classified as `observed_incident_fix`, `deployed_latent_fix`, `pre_ship`, `ops_config`, `feature`, `docs`, `experiment`, or `unknown`.
- Keywords are only search hints. `observed_incident_fix` needs an explicit observation citation, plus a deployment interval from R4/`deploy_history.txt` showing the defective code was live.
- A subagent does the first pass. I then review:
  - every positive;
  - every mixed-purpose commit;
  - **every negative that touches the deployed services' runtime closure**:
    - `src/bts/**`, except `experiment/`, `validate/`, `evaluate/` and the `simulate/` files not loaded at runtime (runtime ones: `mdp.py`, `tail_policy.py`, `pooled_policy.py`, `quality_bins.py`, `strategies.py`);
    - scripts run by cron or units (`scripts/cron*`, `scripts/*_once.py`, `scripts/check_heartbeat.py`) and `scripts/systemd/**`;
    - `.github/workflows/**`, `pyproject.toml`, `uv.lock`, `data/models/*.npz`.
- A seeded 10 % QC sample covers the remaining non-production-path negatives. It is quality control only, not a completeness proof.

**Route R — runtime invariants** over **every season date in coverage**. Each is computed from §5 metadata only, and each violation is a candidate:

| # | Invariant |
|---|---|
| V1 | Finalization: every date has a decision, a pick file or a skip record (ledger row kinds; `unfinalized`/`unobserved` rows are candidates) |
| V2 | Delivery timeliness: in dm/public mode, the first confirmed delivery of a committed pick falls before the earliest slot's first pitch − 5 min. Sources: journal `Pick DM sent` / `Posted to Bluesky` timestamps from 5/11, `delivered_at` from 8/30, game times from pick files |
| V3 | Delivery uniqueness: at most one pick delivery per date per slot set (a resend after a recorded failure is allowed and labelled) |
| V4 | Entry completeness: a delivered slot without an entry-confirmed contest match, while the other slot of the same delivered double is confirmed (partial entry), or a delivered pick with no entry and a scoreable commit. Source: ledger `entry_status` |
| V5 | Scheduler liveness: unplanned restarts per date (planned = the daily idle → exit → restart), restart bursts, days with scheduled games and no lineup check before first pitch − 5 |
| V6 | EOD health ran on every game date |
| V7 | Cron coverage: each scheduled job ran as expected (`check-results` 01:00, `reconcile` 02:00 [+07:40 later], `fetch-contest-streak` ×4, the 03:00 chain, `park-drag-refresh`, `capture-static`, `check-pick-entered` window, backups, heartbeat), from each log's own run markers. A gap is a candidate unless a recorded config change explains it (e.g. the 9/14 entry-check disable) |
| V8 | Research streams: official live-forward capture per decision day, resolve status, D8 research sidecar + acceptance marker per skip day from 9/19, shadow file + reconciliation status per production day, skip-shadow record per MDP skip |
| V9 | Deploy timeline: every run's outcome, the active SHA per interval (runs + `deploy_history.txt`), failed/rolled-back transitions, and deploys landing inside game-time windows (restart during live polling) |
| V10 | Alert delivery: per-date health DM status and entry-check markers; attempted vs failed vs confirmed sends. An `alerted` marker is not proof of a sent DM |
| V11 | Private/tail period 9/14–9/27: private-mode commits carry `private_locked`, no pick DMs, the nag cron stays silent after its disable, the tail stop holds (the P-05 mechanism audit is reused with its scope stated), D8 captures only on skip days |

**Route X — external reports:**
- R3/R4 and INCIDENT/ARCHITECTURE notes;
- memory notes, as pointers only; each must be traced to an artifact or stays `reported`.

**Reconciliation:** a table of candidate × route.
- Runtime-only candidates are silent incidents and get priority review.
- History-only candidates need runtime corroboration or stay `reported`.
- Every candidate ends in a §2 disposition.

## 7. Record schema (`I-nn`)
**Header fields:**
- `id`, `title`, `disposition`, `classes` (primary + linked), `tier`, `operating_mode`, `authority`
  - `operating_mode`: dm / public / private / tail / research-only;
  - `authority`: what was authoritative for the delivery, entry or grade concerned — e.g. contest vs local.
- `contract` — what should have happened, citing its source.
- `mechanism` — the causal chain as numbered links, with code citations at the defective ref.
- `fix` — for each link: `implemented` (commit), `deployed` (SHA + time), `mitigated` (operator action + time), `verified_recovered` (evidence + time), or `unfixed`.
- `fixtures` — historical reproduction and current defence, each with node ids, patch id and verdict (§9).
- `residual` — including review deferrals.
- `watchdog` — the proposed W4 rank-2 trigger, the boundary it sits on (delivery / entry / restart / singleton-slate / private-vs-contest / grading / preservation), and the required recovery or restore assertion.
- `evidence[]` — per §3.

**`occurrences[]`** — one entry per occurrence, each with:
- onset (time, or an interval when not observed exactly);
- first detectable time;
- first machine detection (which detector, which time);
- alert: attempted / confirmed / failed, with times;
- operator awareness and action;
- mitigation time;
- restored-service verification;
- whether it recurred before or after the fix.

**Latencies** (each with bounds, or `unknown`):
- *detection* = first detection − onset;
- *notification* = confirmed alert − first detection;
- *recovery* = verified recovery − onset.

## 8. Evidence bundle
B2 sources are copied read-only (the same acquisition discipline as the ledger) into `data/hetzner_results/season_2026_incident_evidence/v1/`, which sits inside the archive backup set.
- The bundle carries a manifest (path, sha256, size, source mtime, acquisition time) and records declared-missing sources as `missing`.
- Extraction and the invariants run offline over B1 + the bundle + B3, and write `…/v1-extract/<code-sha>-<run>/` (metadata only).
- The outputs are copied into the repo evidence directory (§11).
- One restic `archive` backup of the bundle, with the snapshot id recorded and a read-back check.

## 9. Fixture protocol
**9.1 Before any mutation, per incident**, write down:
- the contract (§7);
- the historical entry point and config (which function, CLI or cron path, which mode);
- the timing or fault trigger;
- the externally visible symptom;
- the declared allowed mocks: external boundaries only (MLB/contest HTTP, DM transport, clock, filesystem roots).

Then **pin** the baseline SHA, the historical refs (fix `F`, `F^`, deployed interval), the test-file sha256s, the config/inputs and the environment (`uv.lock` sha, Python version). All runs happen in isolated worktrees with their own venv. Every run logs which `bts` package was imported, the collected node ids and count, the exit code and any skips.

**9.2 Historical reproduction.** Worktree at `F` with `src/` from `F^`, so tests, conftest and lock stay at `F`.
- The pinned fixture must fail **in the call phase** at an **observable-contract assertion**: delivery happened or not and when, grade value, persisted state, or alert sent.
- The actual and expected values are recorded.
- Collection errors, `ImportError`/`AttributeError`/`TypeError` from new API, setup failures and skips do not count.
- The same fixture passes on `F`'s `src/`.
- A fixture whose failure lies only in a diagnostic assertion (reason string, call count, helper return) is reported as `diagnostic_only`, not a reproduction.

**9.3 Current defence** (pinned baseline, i.e. the commit where W1.5's tests land):
1. Unmarked green baseline run first.
2. The **smallest semantic mutant** restoring the pre-fix decision at one mechanism point. This applies even when a full reverse patch applies cleanly: every hunk is inspected, and tests or expected values are never touched. The patch is saved with its sha256.
3. A **path witness.** A coverage run (`coverage` restricted to the mutated module) shows that the mutated lines executed, and that the production entry point named in 9.1 was on the path (entry function executed), during the killing node.
4. The identical fixture and oracle run on the mutant (red) and on the restored baseline (green), with identical test content hashes.

**9.4 Multi-cause incidents** (e.g. 7/12, 8/30):
- enumerate the causal chain;
- one mutant per link;
- plus the combined historical replay where one exists.

The August replay (`tests/test_incident_2026_08_30.py`) certifies only what its mocks leave real (`run_single_check`, refresh, polling and DM send are patched, lines 73–98), and it states which assertion killed which mutant.

**9.5 Survivors** are classified as `not_reached`, `masked_by_independent_guard` (the guard is named), `equivalent`, `invalid`, or `coverage_gap`. More guards are **never** disabled to force a failure. Historical reproduction and current defence are reported separately.

**9.6 Symptom kinds.** For each incident, declare whether the historical symptom is a *late send*, *missed send*, *wrong state*, *duplicate alert*, and so on. A present-day guard that **refuses** delivery (e.g. the cutoff refusal) is reported as containment, not as a reproduction of the late send.

**9.7 Strict expected-failure fixtures** (unfixed defects with a fixed contract, §10).
- **Dedicated exception.** Each incident gets its own exception class (not an `AssertionError` subclass), raised only by a final oracle helper once setup, real execution and path-witness checks have passed. The marker is `xfail(strict=True, raises=<that class>)`. Setup, fixture validation and unrelated invariants raise ordinary errors.
- **Validation before marking.** A run with `--runxfail` shows the exact traceback, the phase (`call`), and the actual and required values. After that, a marked run shows XFAIL for those nodes.
- **Controls:**
  - an ordinary passing control per harness path;
  - an in-suite meta-test (`pytester`) proving that a marked test failing with an unrelated `AssertionError`, or failing during setup, is reported **FAILED**, not XFAIL.
- **Parameters.** Each genuinely failing parameter is marked separately; passing parameters stay ordinary tests.
- **Not evidence of reproduction:** SKIP, collection or setup failure, an unexpected XFAIL, or a missing node.

**9.8 Characterization fixtures** (the contract is known, the repair design is open — e.g. the 7/16 singleton slate). These reproduce the observed failure sequence through the real planner path with an advancing clock. The oracle is the observable contract (e.g. "an enterable pick existed and nothing was delivered before cutoff"). If they still fail at the baseline, they carry a dedicated-exception strict xfail (9.7). They must not presuppose the repair (no T−120 vs T−90 choice).

**9.9 Config/ops fixtures.** Config and ops fixes have no production-code mutant. Representative old/new config fixtures check the configured behaviour without touching the box. Example: `check-pick-entered` under `pick_delivery = private` with a `private_locked` commit still nags, which is why 9/14 needed the cron edit. Observed operational verification is recorded separately.

## 10. Contracts fixed before fixtures
**10.1 BTS Pass grading.** Pinned text: `https://www.mlb.com/apps/beat-the-streak/official-rules`, fetched 2026-09-29T03:29:29Z (page sha256 `50e8c1e3…`; extracted text `deaa8622…`; section 6 archived in the evidence directory).
- A **Hit** needs a hit "so long as your Pick had at least one (1) official at-bat or one (1) sacrifice fly".
- A **Pass** is any of: "A. … does not make an official at-bat appearance, does not make a sacrifice fly, or does not play …; B. all of your Pick's at-bat appearances resulted in a base on balls, hit batsman, defensive interference or obstruction, a sacrifice bunt, or a walk …; C. the game … is suspended and, at the time of suspension, the player has not yet recorded a hit".
- The same page also says: "if your Pick is involved in a suspended game, your Pick will be deemed a Hit, Pass, or No Hit based on game activity only up until the time of suspension".
- Double Down table: Hit+Pass = +1; Pass+Pass = preserved; any No Hit = reset unless the saver applies.

| Complete synthetic feed case | HEAD (`grade_pick_in_feed`) | Required | Fixture |
|---|---|---|---|
| Normal final, present, H=0, AB=0, SF=0 (BB/HBP/sac bunt/interference only) | `miss` | `void` | strict xfail |
| Normal final, in boxscore, no PA at all (evidenced DNP) | `miss` | `void` | strict xfail |
| Normal final, H=0, AB=0, SF=1 | `miss` | `miss` | control |
| Absent from both rosters | `None` | `None` (unlocated; pending) | control |
| Suspended, pre-suspension PA all BB/HBP/sac bunt (0 AB/SF) | `miss` | `void` (clauses A/B) | strict xfail |
| Suspended, pre-suspension AB without a hit | `miss` | clause C literal: Pass; but "Hit, Pass, or No Hit" leaves doubt | **`contract_ambiguous`**: characterization only, no marker. The existing `tests/test_check_hit_suspension.py::test_grade_resumed_hit_does_not_count` (asserts `miss`) is flagged as conflicting with clause C's literal text, not changed |
| Resumed-only PA | `void` | `void` | control |
| Pre-suspension hit, later resumed events | `hit` | `hit` | control |

Every case runs through `grade_pick_in_feed` **and** the real slot-resolution + `update_streak` path, mocking only HTTP. Downstream assertions:
- all-Pass preserves streak and saver;
- Hit+Pass adds exactly one;
- Pass is `void`, never merely "not `miss`" (`None` is pending, not a Pass).

Incomplete evidence (missing stats, plays or timestamps) must not be read as zero. The production contract for that case is not fixed today, so it is recorded as a characterization only.

**10.2 Same-day replay rollback.** The invariant, verbatim from Codex r1 #7: *with consistent, reconstructible local history and no admissible outcome or slot changes, reconciliation preserves all already-applied terminal results through the locked replay instant, including today's streak increment and saver consumption; it excludes unplayed previews and future dates.*
- Cases: today's terminal hit (streak 2 must stay 2); today's saver-consuming miss (`(10, False)` must not become `(10, True)`); today's unplayed preview (control: excluded).
- Assert after a repeated 23:00 ET run: persisted streak **and** saver; unchanged pick and slot results; empty corrections; no fetch for an expired date.
- The fixed midnight-lock case stays an ordinary regression test (pre-ship).

**10.3 Characterization candidates** (from code; not asserted as observed). Each must first fail unmarked. If it passes, it becomes a control or a superseded disposition.
- **7/16 singleton slate:** morning plan with a 19:10 start; the game moves to 18:10; the only check fires at first pitch; the day ends classification-locked and undelivered.
- **Postponed cached fallback:** the selected game is evidenced postponed before send; refresh fails; the cached pick is delivered. Oracle: no delivery naming a postponed game. Replacement, wait and alert policy stay open, and this is distinct from the unknown-status case.
- **Scoring crash/restart:** a fault between the streak update and the terminal pick persist, then a restart. Oracle: one result affects streak and saver at most once, including the saver case.

## 11. Outputs
| Path | Contents |
|---|---|
| `docs/audit/2026-09-29-incident-register.md` | The memo: summary; coverage inventory; records by disposition; near misses; exclusions; `tier_pending`; reconciliation table; fixture results; rulings |
| `docs/audit/2026-09-29-incident-register.json` | The records |
| `docs/audit/2026-09-29-incident-register-evidence/` | Inventory, commit table, runtime-invariant outputs (metadata only), deploy timeline, pinned rules text, fixture logs, mutant patches, coverage witnesses |
| `scripts/audit/incident_register/` | Extractors, invariants, redacting viewer |
| `tests/scripts/incident_register/` | Tests for the above |
| `tests/test_incident_register_2026.py` | Expected-failure and characterization fixtures with their controls, plus the `pytester` meta-test |
| Wrap index / register / corrections | Row W1.5, row X-20, C-03 cross-reference |

## 12. Gates and order
1. Codex design r2 (repo only, no data).
2. Implementation plan with full code, replayed from its fences (the ledger discipline).
3. Build test-first: extractors, invariants, viewer, fixtures. Fresh reviewer → Codex code review (no data).
4. X-20 committed and pushed; then **Eric's go-ahead for the box run** (acquire B2 → bundle + restic; extract; invariants). Nothing is deployed; the box is read-only apart from the new bundle and output directories.
5. Fixture work (repo-only) can run from step 3.
6. Memo → Codex result review (evidence access allowed) → wrap index, register, memory.

## 13. Rulings in this design (each with its cost if wrong)
1. **Claim wording.** Coverage is claimed "within evidenced coverage". *Cost if wrong:* none; the stronger claim is unsupportable.
2. **Pass grading scope.** Clause C beyond zero-AB/SF is left `contract_ambiguous`. *Cost if wrong:* one real defect class is under-fixtured until 2027 evidence (a real suspended-game pick) settles it.
3. **Owner-gated sources.** O1 sources stay out of default scope. *Cost if wrong:* March–May delivery times remain `unavailable`.
4. **Negative review.** I review production-path negatives myself rather than running a second classifier. *Cost if wrong:* a misclassification my review misses. Route R is the independent check.

## 14. Codex r1 findings → v2
| r1 finding | Addressed in |
|---|---|
| #1 exposure | §5 (field-limited extraction, redaction, tests, projection, X-20 text) |
| #2 sources | §4, §6 Route R, §8 |
| #3 commit rubric | §6 Route H |
| #4 xfail specificity | §9.7 |
| #5 mechanism mutants | §9.1–9.6 |
| #6 Pass contract | §10.1 |
| #7 replay invariant | §10.2 |
| #8 fixtures for named/design-open incidents | §1(c), §9.8–9.9, §10.3 |
| #9 scope, dispositions, occurrences, evidence | §§2, 3, 7 |
