# 2026 incident register (W1.5): design v1

**Date:** 2026-09-29 · **Plan item:** W1.5 in `docs/superpowers/plans/2026-09-14-season-wrap-plan.md` (approved 9/22): "7/16 singleton-slate gap · 8/11 MLB auth flap · 8/13 silent pass (Warmup) · 8/30 late pick (Kwan) · 9/03 idle (all-skip table) · any private-mode/tail anomalies through 9/27. Each → mechanism, fix status, **failure-path fixture** that reproduces it, residual gap. Feeds W4 rank 2." · **Status:** draft for Codex design review (no data access in the review).

## 1. Purpose and success
The register is the evidence base for W4 rank 2, the watchdog and restore checks. It answers four questions:
1. What went wrong in production in 2026?
2. How was each problem detected, and how late?
3. Is each fix held in place by a test that fails without it?
4. What is still open?

It is an operations record. It measures no model quality, calibration or policy value.

**Success:**
- (a) Every incident found by the sweep in §3 gets a record (§4) or a written exclusion reason.
- (b) Every fixed Tier-A incident has a fixture shown to fail when the fix is removed (§6). Where no such fixture exists, the gap is stated.
- (c) Every unfixed incident states its residual gap. Where its contract is already fixed (§6.3), it also gets a strict-xfail fixture that reproduces it on HEAD.
- (d) The memo passes a Codex result review.

## 2. What counts as an incident
An **incident** is a live deviation from the system's intended contract, observed on the production box or on something it drives: the scheduler, cron, the deploy workflow, DMs, the dashboard, or the research streams. The window is 2026-03-25 (first round) through 2026-09-28 08:00 ET (the W0.7 freeze). The deviation must fall into one of these classes:

| Class | Meaning |
|---|---|
| **D** Delivery | Missed, late, wrong, duplicate or refused pick delivery |
| **E** Entry | Entry-check false or missed alarms; a partial or absent entry that the system should have caught |
| **G** Grading and state | Local grade, streak or saver wrong; local and contest state diverging |
| **A** Alerting | False alarm, alert storm, wrong advice, or a missing alert |
| **L** Liveness | Idle, restart loop or thrash, crash, OOM, hang |
| **P** Decision contract | The policy layer violates its own contract (e.g. the 9/03 all-skip idle) |
| **R** Research streams | Shadow, skip-shadow, live-forward or D8 capture integrity |
| **X** External data | MLB/Savant API shapes, flaps, lags, in-place re-scoring |
| **S** Preservation and deploy | Data loss, backups, failed or rolled-back deploys, config drift |
| **U** Display | Dashboard or report showing a false state |

A **latent defect** is a defect that has not been seen to fire, or whose firing is unknown. It is included only if one of these holds:
- live evidence established it (e.g. C-03 found in data);
- it is unfixed and can reach production decisions, state, delivery or alerting.

Defects that reviews found and fixed before they shipped are **excluded**. So are test-only defects and development tooling. The exclusion list names them with a reason whenever the sweep surfaces them.

**Evidence requirement.** Each incident cites at least one primary artifact:
- a commit body stating a live observation;
- an audit memo;
- a journal or cron.log line;
- a health-state record;
- a pick, decision or scheduler-state field;
- a contest-ledger line;
- a deploy run.

Memory notes are pointers, not evidence.

## 3. Enumeration: every authoritative source, searched in reverse
The preliminary inventory (Appendix B) comes from docs, commit messages and memory. It is a starting list, not the population. The population is whatever the following sweeps surface. Each sweep's output is kept in the evidence directory (§8), whether or not it adds anything.

| # | Source | Sweep | Reads outcomes? |
|---|---|---|---|
| S1 | `docs/audit/*.md`, `INCIDENT.md`, `ARCHITECTURE.md`, `docs/optimization-ideas.md` | Read each; tag incident content | no (already committed) |
| S2 | Git history since 2026-03-01: every commit touching `src/`, `scripts/`, `.github/`, cron or systemd files (subject **and** body) | Classify each commit with the rubric below: incident fix / latent-defect fix / pre-ship review fix / feature / other | no |
| S3 | GitHub issues and PRs (all states) | Tag incident-bearing ones (e.g. #144, #74, #119) | no |
| S4 | Deploy workflow run history (`gh run list`) | Every failed, cancelled or rolled-back run | no |
| S5 | Scheduler journal 2026-05-11 → freeze (frozen snapshot `final-20260928/`) | Signature grep (Appendix A): counts per date per signature. Each date-cluster is mapped to a known incident or opened as a new candidate | no — signatures exclude result lines |
| S6 | `cron.log` (in the snapshot) | Same signature grep | no |
| S7 | `data/health_state/**` (snapshot) | Alert records: source, level, date, `incident_key` only. Metric values are not read | no |
| S8 | W1.1 accepted build (`data/validation/season_2026_ledger/…dd430abe/`) | Dates and statuses of the anomaly rows: `unfinalized_day` (3), `unobserved_day` (7), `commit_status = unconfirmed` (12), quarantines (3). Dates of the `local_vs_contest_disagreement = true` rows (2). The disagreement reason category (e.g. local NO_HIT vs contest HOLD). No outcome columns beyond that flag | **partially** — see §7 |
| S9 | Scheduler state and decision files for incident dates only | Delivery, lock and timestamp fields only | no |
| S10 | systemd unit history in the journal | Start, stop, exit and restart lines per day; unplanned restarts counted | no |

**Commit rubric (S2).** A commit is an **incident fix** when its message or linked doc states a live observation: a date, "observed", "live", "box", "prod", "incident", "Eric caught", a symptom seen in production, or a GH issue about production. It is a **latent-defect fix** when the defect was found by an audit or review of code that was already deployed. It is a **pre-ship review fix** when the defect was introduced and fixed inside the same unshipped change. A subagent classifies the commits. I verify every commit classified as incident or latent, plus a random 10 % of the rest, against the diff.

## 4. Per-incident record (`I-nn`)
Each record carries these fields:
- `id`, `dates` (first and last live occurrence), `class` (§2), `tier` (§5)
- `symptom` — what was seen, quoted from the primary artifact
- `impact` — delivery / entry / contest / state / alerting / research / none. Contest impact is quoted from existing records only (§7); nothing is re-derived.
- `detection` — who or what detected it (monitor, alert, Eric, audit, data review) and the detection latency. This is the key input for the watchdog.
- `mechanism` — root cause with code citations at the fix's parent commit
- `fix` — commit(s); deployed SHA and date (from the deploy runs), or `unfixed`, or `config/ops only`
- `fixture` — test id(s), the §6 verification result, the mutant patch path
- `residual` — what can still happen; review deferrals attached to it
- `watchdog_relevance` — which W4 rank-2 check would have caught it, and at what boundary: delivery / entry / restart / singleton-slate / private-vs-contest / grading
- `sources` — primary artifacts, with paths or ids

Incidents that share a mechanism across dates are one record listing every date. Recurrences of a class after a fix are linked (e.g. the 6/09 no-games restart thrash and the 7/12 eve-of-break loop).

## 5. Depth tiers
- **Tier A** (full treatment: journal timeline where available, mechanism, fix, mutation-verified fixture, residual). Any incident with delivery, entry, contest or state impact, an alert storm, or a liveness failure. **All incidents named in the plan are Tier A.**
- **Tier B** (catalogued: symptom, fix commit, fixture test ids identified but not mutation-verified, residual). Display, research-stream and alert-wording incidents with no delivery or state impact.

A Tier-B incident is promoted to Tier A if its residual still reaches production.

## 6. Fixture verification: prove the test fails without the fix
### 6.1 Fixed incidents (Tier A)
1. Identify the fix commit(s) and the tests aimed at the mechanism.
2. In a scratch worktree at the current main HEAD, build a **mutant** that restores the pre-fix behaviour at the mechanism point:
   - first try the reverse of the fix's `src/` hunks (`git show <fix> -- src/ | git apply -R`);
   - on conflict, hand-write the smallest mutant that restores the pre-fix decision, with its rationale.
3. Run the fixture tests on the mutant. At least one must fail with an assertion about the incident's symptom. An import error or an unrelated failure does not count.
4. Run the same tests on HEAD. They must pass.
5. Record the fix ids, the mutant patch (`…-evidence/mutants/I-nn.patch`), the failing test ids with their assertion lines, and the green run.

If no test fails on the mutant, the record reads `fixture_does_not_reproduce`, a residual gap. W1.5 may then add a fixture test-first: red on the mutant, green on HEAD.

### 6.2 Config or ops fixes
Examples are the 9/14 silence and the 7/04 scrape stop. There is nothing to mutate. The record gives the procedure and the observed verification from the primary artifact, and states the gap: no automated check.

### 6.3 Unfixed incidents and latent defects
A **strict-xfail fixture** is written only when the contract is already fixed, either by an external rule or by an existing documented invariant:
- `@pytest.mark.xfail(strict=True, raises=AssertionError, reason="I-nn …")` in `tests/test_incident_register_2026.py`, following the `tests/test_incident_2026_08_30.py` precedent;
- it must fail on HEAD with the asserted symptom, and a setup error must not count as an expected failure (hence `raises=AssertionError`).

Candidates where the contract is fixed:
- **grading Pass rules.** The official rules say walks-only / did-not-play / suspended-before-a-hit is a Pass. Production's `grade_pick_in_feed` returns `miss`.
- **same-day replay rollback.** A reconcile run that proposes no corrections must leave today's applied streak and saver unchanged (Codex reconcile-cutoff r1 #3).

Where the contract is still a design choice (e.g. the 7/16 singleton-slate gap, and 7/10 F1 cached-fallback delivery of a postponed-game pick), the record gives an exact **reproduction scenario** (inputs, clock, expected-vs-actual) and leaves the executable fixture to the W4 rank-2 build. A strict-xfail test there would fix the design early.

No production code changes in W1.5.

## 7. Exposure (register row X-20, predeclared before any box read)
**Draft row:**
- **What is read:**
  - S5–S7 and S10 lines and records by signature: dates, sources, levels, counts;
  - S8 anomaly-row dates and statuses, plus the `local_vs_contest_disagreement = true` rows (date, slot, the two normalized values, the comparison's reason). This checks whether the grading-Pass defect (local NO_HIT vs contest HOLD) or C-03 produced them;
  - S9 delivery, lock and timestamp fields for incident dates.
- **What is not computed:** no hit rates, calibration values, model or policy comparisons, and no streak or outcome tallies. Incident impacts are quoted from existing records (e.g. the 8/13 legs "would have hit", recorded 8/14).
- **Allowed next look:** none from these reads. Any candidate a register finding motivates is an ops candidate (W4 rank 2), validated by failure-path fixtures and not by 2026 outcomes, so D3 is unaffected.

## 8. Outputs
- `docs/audit/2026-09-29-incident-register.md`, the memo:
  - a summary table;
  - one record per incident;
  - an exclusions list;
  - a sweep coverage table (per source: items read, candidates raised, mapped, new);
  - a rulings list.
- `docs/audit/2026-09-29-incident-register.json`, machine-readable records.
- `docs/audit/2026-09-29-incident-register-evidence/`: sweep outputs (S2 classification table, S4 run list, S5/S6/S10 signature counts, S7 alert index, S8 anomaly list), `mutants/*.patch`, and test run logs.
- `tests/test_incident_register_2026.py`, strict-xfail fixtures for §6.3 only, plus any added fixtures for §6.1 gaps.
- Wrap index row W1.5, exposure register row X-20, and corrections-index cross-references (C-03).

## 9. Gates
1. Codex design review of this document (repo access, **no data directories, no box access**).
2. X-20 committed and pushed before S5–S9 are read.
3. Execution: sweeps (subagents allowed for S2 and per-incident mutation checks; I verify every incident-class claim and re-run a sample of mutants myself).
4. Codex result review of the memo, with evidence access allowed.
5. Wrap index and memory updated.

Nothing is deployed. The box is read-only throughout (the snapshot and the accepted ledger build).

---

## Appendix A — S5/S6 signature list (frozen with X-20)
Case-sensitive substrings drawn from the scheduler's own messages (`src/bts/scheduler.py`) plus generic failure tokens:
- **Delivery and lock:** `MISSED-PICK ALERT` · `DELIVERY REFUSED` · `DELIVERY OUTCOME UNKNOWN` · `Pick DM failed` · `Bluesky post failed` · `Skip DM failed` · `CONTEST STATE ERROR` · `FALLBACK REFRESH` · `FALLBACK DEFERRED` · `past the` (cutoff messages) · `game_started_or_final`
- **Liveness and idling:** `Failed to fetch/compute` · `Idle until` · `already ` (past-wake handoff) · `quarantine` · `Traceback` · `Error` · `ERROR` · `Exception` · `Killed` · `MemoryError`
- **Results and polling:** `Result polling capped` · `unresolved` · `vanished`
- **Shadow model:** `[SHADOW MODEL] Failed` · `Trigger failed` · `Trigger returned`
- **Health:** `CRITICAL` · `WARN`
- **systemd lifecycle:** `Main process exited` · `Failed with result` · `Scheduled restart job` · `Started ` · `Stopped `

Result-bearing lines (`All picks have hits`, `Result already scored`, `Streak:`) are **excluded** by a negative filter. The same list is used for `cron.log`, plus `CORRECTIONS FOUND` (reconcile) and the auth categories (`TransientAuthError`, `RateLimited`).

## Appendix B — preliminary inventory (repo-derived; the sweep decides the population)
Occurred live, with dates and fix commits from commit bodies, audit memos and memory pointers:

| Date(s) | Incident | Fix |
|---|---|---|
| 4/15 | 02:00 reconcile reset the streak to 0 nightly (walk hit today's preview) | `1d61908` |
| 4/29 | bpm cumsum prediction-cycle latency | PR #5 |
| 5/06 | Stale picks on postponed games | `7b701c9`, PR #24/#25 |
| 5/09 | Postponed locked picks graded as void | PR #74 |
| 5/08–5/09 | Shadow result reconciliation | PR #56 |
| 5/10, 5/12 | Dashboard pick-state / scorecard display | PR #80/#93 |
| 5/13 → 5/26 | Live-forward snapshot drift / null provenance | PR #97/#99/#132 |
| 5/15 | Preview NaN pitcher ids | PR #98 |
| 5/22–5/23 | Inline shadow OOM loop | PR #102/#105 |
| 5/23 | Memory-growth false alerts | PR #114 |
| 5/24–5/26 | Postponed-game candidate filtering / lock-status fallback | PR #119/#131 |
| 5/27 | Restart-spike wording; dashboard health responsiveness | PR #136/#137 |
| 6/05 | MacBook loss: `pooled_bins_run` profiles lost, so the shipped policy cannot be re-solved | preservation |
| 6/07 | Nightly false-CRITICAL on contest-state staleness | `58c9adc` |
| 6/09 | No-games days restart-thrash | `736ea8f` |
| 6/10 | fetch-contest-streak daily false "failed" DM | `8cd7207` |
| 6/10 | Reconcile under-counted saver-preserved streaks | `3a6e48b` |
| 6/11–6/12 | Pick-entry check: settled-only endpoint → daily false alarm → cron disabled | `4f13eb3` → `2d68102` |
| 6/17 | Local streak 10 vs contest 8 inflation → contest anchoring | PR #143 |
| 6/21 | GH #144: check-results scored undelivered previews on skip days | PR #145 |
| 6/29–6/30 | Resumed-portion PA counted in scoring | `a364b11` and siblings |
| 7/01–7/02 | Skip-day dashboard visibility | `258aaa4`, `2c62d72` |
| 7/06 | check-pick-entered premature DM on a deferred double-down | `af6329f`, `540b1ab` |
| 7/08 | Partial entry (Harris un-entered) after a single DM | F1 v3 `8bceda1` |
| 7/12 | Eve-of-break restart loop + ~47 duplicate CRITICAL DMs; confounded drift metric | `9551818`, `ec242da`, `230f65c` |
| 7/16 | Singleton-slate gap | **unfixed** (backlog `7b70da7`) |
| 7/10 → 8/09 | Shadow result stranded a month | `4f0257a..64f0ffb`, `41b2bb1` |
| 7/28 | Stale preview on a skip day (gate held) | — |
| 8/11 | Auth/login flap → wrong DM advice; park_drag table stale | `404358d` |
| 8/13 | Silent pass (Warmup) | `1b50b78`, `224ddce` |
| 8/30 | Late pick (Kwan) | `ac0ce8d`, `67338cd`, `c0c0a97`, `3697512`, `2ff2db9` |
| 9/03 | All-skip idle | tail policy `0abf503`, `eb010fd` |
| 9/14 | `pick_delivery = private` does not silence the entry-check cron | config only |
| 9/16 | `stale_pick_snapshot` recaptures in the official live-forward root | pre-existing, unfixed |
| season | C-01 leaderboard parser (active streak in `all_season` rows) | `44df03f` |
| 5/10, 8/20 | C-03 reconcile post-cutoff flips | `ce6676d` (main only; deploys 2027) |

Latent or unfixed:

| Defect | Source |
|---|---|
| Grading Pass rules in `grade_pick_in_feed` | ledger spec §12 |
| Same-day replay rollback | reconcile-cutoff Codex r1 #3 |
| Cached fallback can deliver a postponed-game pick | 7/10 F1 deferral |
| Streak + pick two-file crash atomicity | 7/09 deferral |
| `delivery_unknown` non-scoreable redesign; feed-file validation / atomic downloads; in-transport deadline enforcement | 8/30 F2/F10/F1 |
| Independent day-outcome watchdog; no-games-day early return bypasses EOD health | 8/14 backlog |
| C-03 best-effort residuals | — |
| Unretried Open-Meteo fetch | — |
| `private_locked` / `locked_unconfirmed` with a failed `decision.json` write → no alert | 7/06 known limitation |
