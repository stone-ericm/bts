# Production state audit — 2026-07-01

Full-system audit of BTS (repo + bts-hetzner production), requested by Eric.
Successor to `2026-06-09-fable5-full-audit.md`; covers everything shipped since
(streak saver, skip visibility, real-streak anchoring, decision.json/#145,
suspended-game scoring). All claims below were verified live this session
(box state, journals, GitHub, local test run) — not recalled from memory.

## Verdict

**Healthy. No critical findings.** The week's one CRITICAL alert is identified,
root-caused, and confirmed resolved by the suspended-game fix deployed
2026-06-30 → 07-01. The #145 decision.json model is demonstrably working in
production. A short list of small opens and hygiene items below.

Production streak state at audit time: **real streak 17 (season best), saver
active (unused)** — source_date 2026-06-30, auto-fetched fresh 13:30 ET today.

## Production ops (verified on box)

| Check | State |
|---|---|
| Deployed SHA | box HEAD `83a5cad` == GitHub `deploy` ref; GitHub `main` ahead by one docs-only commit (`680a63a`, +9 lines ARCHITECTURE/CLAUDE) — invariant intact. (Refs read via `git ls-remote` against GitHub; the box's local `origin/main` is stale by design — deploys pull only `deploy`.) |
| Deploys | 4 in the ~39h before audit close (28416280226 @ 6/30 02:31Z; 28466759515; 28468207448; 28520680537 @ 7/01 13:21Z), all success + canary; `.last_deploy_iso` = 2026-07-01T13:26:19Z ✓ |
| Units | `bts-scheduler`, `bts-dashboard` active; dashboard HTTP 200; bound to tailnet IP only (`100.100.43.24:3003`) |
| Box | up 81 days; disk 7%; scheduler RSS 2.95 GB EOD 6/30 (thresholds 4.6/5.1/6.1 GB) |
| Heartbeat | fresh, `state=sleeping` consistent with schedule; **0 stalls ever** in `cascade_stage_durations.jsonl` (H5b dataset accumulating, 392 KB) |
| Cron | full set installed + firing; 3am pipeline ran clean today (pull → build → preview → R2 sync **gated on build success** per `b0420a0` — gate exercised, 12 files synced); contest fetch 4×/day writing streak=17 |
| Timers | `bts-live-forward-{capture,resolve}` + `bts-leaderboard` (2×/day) all running |
| Scheduler today | 09:25 restart = the deploy (see below); 14:07 check → provisional Arraez 82.8% (projected, undelivered); 17:40 + 18:15 checks on confirmed lineups → **18:26 ET flipped to MDP SKIP** ("best Luis Arraez (SF) 81.0% below the pick bar; streak holds at 17") — see "Live event" below |

## The week's one CRITICAL — closed

- **What fired:** Jun 29 22:36 ET, `live_forward_resolution` CRITICAL DM:
  canonical resolution stalled 12d on **2026-06-17** (missing outcomes for
  game 824912 — the suspended/resumed game).
- **This is exactly the incident class the suspended-game scoring work fixed**
  (`793a093`…`83a5cad`, deployed 6/30→7/01).
- **Confirmed resolved:** today 12:01 ET the resolve timer regenerated
  2026-06-17 → "Candidate artifact verification: PASS (0 failures)".
  `pa_2026.parquet` carries `is_resumed_portion` with 824912 split 20 scoreable
  / 58 resumed ✓.
- The surrounding "Failed with result 'exit-code'" unit exits (Jun 29 22:36,
  Jun 30 14:28/14:53, Jul 01 09:25) are **not crashes** — all four are SIGTERM
  (143) from deploy-workflow restarts. (Codex's pass surfaced the 4th deploy
  run, 28416280226 @ 6/30 02:31Z, which my first pass had misread as a manual
  restart.) The Jun 29 22:36 CRITICAL fired one minute after the restart for
  the deploy that carried the first suspended-game commits — the resolver
  artifact was simply still stale until its next timer run. See finding F3.

## #145 decision.json model — working in production

- `decision.json` written **every day 2026-06-23 → 06-30** (8/8). Delivered
  days carry `scoreable=true`; 6/30 shows `action=double, source=mdp,
  delivery_status=delivered` and scored `hit`.
- **Skip-policy shadow accumulating correctly:** 4 genuine MDP skips
  (6/24–6/28, streaks 13–14), 3 resolved → 2 hit / 1 miss / 1 void; band rate
  0.667, Wilson CI [0.21, 0.94] vs breakeven 0.744 → `insufficient_n`
  (needs 30) — exactly as designed. Void handling exercised (Wetherholt 6/25;
  suspended-game voids now surface post-`63f3d1d`).
- 6/30 the MDP **doubled at streak 15** (Arraez 0.828 + Y. Díaz 0.798,
  different games ✓, inside the saver-protected zone); both hit → 15 → 17.
- Tomorrow (7/02) previewed as a **projected skip** (top candidate Xavier
  Edwards 74.6%) and the 3am preview correctly wrote **no** pick file — the
  post-#145 behavior working. Note: this makes 7/02 another *clean* skip
  candidate, NOT the decisive C1/C2 case (which needs a stale
  projected-**pick** preview flipping to a real skip). Hours later, **TODAY
  became exactly that case** — see "Live event" below.

## Live event during audit — TONIGHT is the decisive C1/C2 test

At 18:26 ET, with confirmed lineups in, the MDP flipped today from a
provisional pick to a genuine skip: `SKIP — best Luis Arraez (SF) 81.0% below
the pick bar; streak holds at 17.` (At streak 17 — outside the saver zone —
the policy's bar is higher than the 0.796 Q1 boundary, so 0.810 is declined;
yesterday's double at saver-protected streak 15 and today's skip are the same
policy being consistent, not a contradiction.)

Critically, a **stale provisional pick file already exists** for today
(`data/picks/2026-07-01.json`, Arraez 82.8% from the 14:07 projected-lineup
save; `delivery_attempted=false`, `notification_sent=false`). That is the
projected-pick → real-skip flip the #145 daemon-path work was built for and
which has never yet occurred live. Tonight exercises, in production:

- **C1**: fallback/refresh paths must NOT deliver the cached Arraez file on a
  standing MDP skip (a *genuine* late pick flipping back and delivering is
  also a pass — it just tests the other branch).
- **EOD (~22:05 ET)**: `_write_endofday_skip` should write
  `decision.json {action: skip, source: mdp, scoreable: false}`; the tentative
  skip DM should arrive once; **no false missed-pick / post-failure alert**
  (D6 gating).
- **C2 / #144 (1am cron)**: `check-results` must NOT score the stale Arraez
  file (`decision.scoreable=false` gate). **Streak must stay 17** unless a
  hit is genuinely delivered late.
- Skip-policy shadow gains divergent day #5 (first skipped candidate above
  0.796).

**Operator caveat for tonight (see F1):** the dashboard will NOT show the SKIP
banner — the banner is suppressed whenever a today pick-file exists
(`web.py` `_has_today_pick`), and the stale undelivered file satisfies that.
The dashboard hero will show Arraez as if he were today's pick. The skip DM is
the authoritative signal; do not enter the dashboard pick in the MLB app
(unless a later *delivered-pick* DM arrives).

## Suspended-game scoring — integration verified

- Parquet-based scoring consumers all read through
  `read_pa_for_bts_scoring`/`filter_out_resumed_portion`: `data/build`,
  resolver `experiment/artifacts`, `health/realized_calibration`,
  `health/slate_auc`, `model/calibrate`, `scripts/canonicalize_realized_picks` ✓.
- Feed-based graders are resume-aware: `picks.py` (grade_pick_in_feed),
  scheduler mid-game, `skip_policy_shadow` ✓.
- Remaining raw `pa_*.parquet` reads are non-BTS-scoring contexts by design
  (experiment CLIs — training/eval keeps resumed PA). One gray spot noted
  (F7, negligible).

## Tests / CI / repo

- Fast scoped regression **local: 1514 passed, 2 deselected, 26s** (up from
  1483 — new suspended-game tests).
- CI green on main (4/4 latest), deploy workflow 3/3.
- **No open PRs, no open issues** (#144 closed by #145).
- Security spot-check (public repo): `.env` gitignored ✓; `audit_driver.py`
  relay IP properly parameterized (`BTS_RELAY_HOST_PUBLIC` env, empty default)
  before commit — the standing "do not commit as-is" concern is resolved ✓;
  no suspicious files in commits since 6/10 ✓; dashboard tailnet-bound ✓.

## Memory-vs-reality corrections (hub updated this session)

1. **"QUEUED: contest_state gap==1 daily false WARN" is stale — the fix
   shipped** (`56f9726` "align contest-state staleness to the Phase-1 coverage
   split"): gap==1 → INFO at any time of day, message rewritten to "coverage
   lag". Last real WARN episode was 6/15–6/17 (gap≥2 settlement lag, correct
   behavior). It also never fired daily post-Phase-1 — episodic only.
2. Statcast residual screen ("IN FLIGHT" in the hub pickup line) closed 6/14
   as stage-2 NULL — superseded by the kcontact arc already recorded.

## Findings (ranked)

- **F1 — flip-day dashboard shows the pick the system skipped** (NEW — live
  TONIGHT). The SKIP banner is suppressed whenever a today pick-file exists
  (`web.py`: `_has_today_pick`), and the hero card renders the pick file
  without checking delivery/decision state. On a projected-pick→real-skip flip
  day, the stale provisional file makes the dashboard show a pick while
  production skipped — at streak 17, an operator following the dashboard could
  enter a pick the MDP declined. The 6/18 "file exists ⇒ pick" assumption
  predates #145 and is no longer valid; banner + hero should gate on
  `decision.json` / `pick_was_delivered`. (The skip DM is unaffected and
  remains the authoritative channel.)
- **F2 — ARCHITECTURE drift on contest-state semantics** (Codex catch, the
  substantive doc drift): the health-table row still documents "gap==1 → WARN
  at/after noon ET" (superseded by `56f9726`: gap==1 → INFO any time,
  "coverage lag" wording), and the Contest-account section still says stale
  state "freezes the effective streak at max(model, contest) and disables
  doubles" (superseded by real-streak-anchoring Phase 1). A reader debugging
  from ARCHITECTURE would trust failure semantics that no longer exist. Also:
  the health table lists 16 sources; shipped modules add
  `analytics_artifacts_missing`, `live_forward_resolution`,
  `mdp_policy_alignment`, `postponed_pick`, `fallback_defer`.
- **F3 — deploy restarts log as unit failures.** `bts-scheduler.service` lacks
  `SuccessExitStatus=143` (or a SIGTERM handler), so every deploy stop logs
  "Failed with result 'exit-code'". All four "failures" this week were
  deploys; real crash signal is being diluted. One-line unit fix.
- **F4 — saver `do_POST` socket test still missing** (open since 6/18). The
  dashboard `/saver/transition` wiring has no committed test (pure helpers
  covered; wiring verified ad-hoc only). Small, contained; worth closing.
- **F5 — untracked local artifacts:**
  `docs/audit/2026-06-14-team-record-result.md` (a referenced negative-result
  doc — should be committed for durability),
  `data/validation/realized_picks_canonical_2026-06-15.parquet` (its 05-04
  twin IS tracked), `data/external/team_context/` + `data/validation/team_record/`
  (commit or gitignore — decide).
- **F6 — merged local branches deletable:** `audit-fixes-2026-06-09`,
  `contest-health-coverage-lag`, `contest-streak-automation`,
  `phase2a-decide-action`, `real-streak-anchoring`, `skip-policy-shadow`.
  (Experiment branches `kcontact-screen`/`resolution-audit`/`swing-escalation`/
  `team-record-experiment` retained by design.)
- **F7 — falsification-harness CLI reads raw PA** (`cli.py:781`) including
  resumed portions (~0.06% of 2026 rows) into realized-outcome diagnostics.
  Offline research path, negligible effect — noted for methodological
  completeness only.
- **F8 — `datetime.utcnow()` deprecation** in `leaderboard/scraper.py:202`
  (warns on every test run).
- **F9 — audit-progress full app-auth** — **CLOSED WON'T-FIX 2026-07-02, risk
  accepted by Eric** ("not necessary — there's nothing sensitive at risk").
  Rationale: read-only monitoring endpoint, tailnet-bound + traversal-guarded,
  only meaningful while an audit fleet exists; exploiting it requires an
  already-compromised tailnet device. Do not re-flag in future audits absent
  a change in exposure (e.g. the dashboard ever binding beyond the tailnet).

## Watch items (no action now)

- **`slate_auc` arms ~tomorrow:** currently `n_days=14` (computed 6/25,
  7-day cache); next recompute ≈ 7/02 crosses `min_days=20` → first real AUC.
  If ≥ 0.61 the M3 revisit WARN fires by design (re-run
  `scripts/replay_m3_serving_parity.py`). The `n_rows=0` in the current status
  is a short-circuit artifact (verified in code), not a broken join.
- `mdp_policy_alignment` Q0-collapse WARN quiet since 6/19 (recent picks ≥
  0.796); Gate-B fingerprint bank keeps accumulating toward ~season-end.
- DD overconfidence (+14pp, n=29) unchanged — known, blocked on Gate-B.
- Scheduler memory: RSS 3.23 GB live at ~18:40 ET after three full cascades
  (cgroup `MemoryCurrent` reads 4.8 GB, but that includes page cache — the
  health check keys on RSS; thresholds 4.6/5.1/6.1 GB). Day-boundary restart
  resets it. Within thresholds; pattern unchanged.

## Scope notes

Model/analytics conclusions (kcontact powered-null, resolution-audit ceiling
verdict, skip-threshold keep) were each Codex-reviewed at the time and are not
re-litigated here; this audit verified their *operational encoding* (no swing
features in prod FEATURE_COLS, threshold unchanged, shadow accumulating).
An earlier draft closed with today's pick mid-cascade; at 18:26 ET it flipped
to a genuine MDP skip — see "Live event" above.

## Codex adversarial pass (gpt-5.5, repo + box access)

Six findings; triage:

- **Folded in:** the 4th deploy run 28416280226 (all four unit "failures" were
  deploys — no manual restart); ARCHITECTURE contest-state drift (now F2, the
  substantive doc-drift catch); local-vs-GitHub ref precision on the deploy
  invariant (wording fixed); the skip flip (caught live in parallel with
  Codex's read — expanded into the "Live event" section).
- **Refined, not adopted:** "memory envelope falsified at 4.6 GB" — Codex read
  cgroup `MemoryCurrent` (includes page cache); the health check and this
  audit's numbers are RSS (3.23 GB live). Direction noted, metric corrected.
- **False flag:** "local test run unverifiable" — the 1514-passed run executed
  in this session (transcript evidence); Codex correctly couldn't reproduce it
  read-only. No change.
