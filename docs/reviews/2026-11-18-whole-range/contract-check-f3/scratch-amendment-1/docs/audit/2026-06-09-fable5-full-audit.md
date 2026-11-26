# BTS Full Audit — 2026-06-09 (Fable 5)

Read-only audit of the entire `bts` project. 10 parallel deep-dive auditors (pipeline, live-decision path,
scheduler, web/security, secrets/supply-chain, tests/CI, health, model/simulation/validation math),
findings deduped and the headline items re-verified against source by the orchestrator.

**Verification tags:** ✓ = orchestrator read the code / ran the check directly · ▸ = auditor-reported, high
confidence · ⚠ = subtle (esp. probability math); flagged for human re-check rather than over-claimed.

**Live state at audit time:** deployed SHA `a436827` (main==deploy==origin, clean). Scheduler + dashboard
`active`. Disk 7%, RAM fine, no WARN+ in 48h journal. Port 22 reachable from public IPv4; **port 3003 times
out from public IPv4** (✓ external probe) → dashboard is firewalled at the network layer today.

**Test baseline (Mac, this session):** `pytest -m "not slow"`, model modules excluded → **1668 passed, 0 real
failures**. The 57 "failures" + 5 collection errors are all `OSError: libomp.dylib not loaded` — lightgbm has no
libomp on this loaner. Non-model logic is green; the model/training/experiment surface is unrunnable locally.

---

## Act on these first (highest leverage, all verified)

| # | Finding | File | Why now |
|---|---------|------|---------|
| 1 | realized_calibration runs on the **biased** attribution path in prod | `scheduler.py:1808` | The daily CRITICAL DMs you get are computed on the basis the project itself diagnosed as wrong. One-line fix. |
| 2 | MDP quality-bin collapse is a **probability-basis mismatch**, not just drift | `backtest_blend.py:579` vs `predict.py:671` | The queued "re-solve MDP" will **not** fix it unless re-solved on the *estimated-PA* basis. Saves a wasted 2–3h run. |
| 3 | `platoon_hr` is **always NaN at production inference** | `predict.py:452-455,556` | A baseline FEATURE_COL silently absent live but present in backtest → live underperforms the backtest claim, invisibly. |
| 4 | `.env` + 4 operational data dirs are **not gitignored** | `.gitignore:1-6` | One `git add -A && git push` on the box publishes every secret to the public repo. Trivial fix. |
| 5 | `bts schedule` defaults the day in **UTC** | `cli.py:1641` | An 8 pm–midnight ET restart starts *tomorrow's* run_day and abandons tonight's polling + late-slate delivery. |
| 6 | DH game-2 recheck is a **bare `time.sleep(15min)×10`** | `scheduler.py:1731-1732` | Two un-pinged sleeps = the 1800s systemd watchdog → SIGABRT crash-loop (narrow path, but real). |
| 7 | **No CI runs the tests** | `.github/workflows/deploy.yml` | 35k-line suite runs only when a human remembers; deploy canary checks "service up + HTTP 200", not correctness. |

---

## Model & pipeline correctness

**[P1] ✓ `platoon_hr` always NaN at inference** — `predict.py:442-455,556`
Pitcher handedness is only populated from live `allPlays` (425-430), which is empty at the game-time−45min pick.
The probable-pitcher fallback (452-455) sets `id`/`fullName` but **never `pitcher_hand`**, so
`lookups["platoon"].get((bid, None))` → None for every batter. The hand *is* in the `&hydrate=probablePitcher`
response — the code just doesn't read it. Fix: `opp_pitcher_hand = pp.get("pitchHand",{}).get("code")`.

**[P1] ⚠ MDP quality-bin collapse = actual-PA vs estimated-PA basis mismatch** — `backtest_blend.py:579` vs `predict.py:671`
The shipped policy (`mdp_policy.npz`, boundaries `[0.796,0.811,0.825,0.841]`, embedded `optimal_p57=0.0817`)
was binned on backtest profiles whose `p_game_hit` is computed over **realized** PA rows. Production computes it
with **lineup-slot estimated** PAs, plus pooled multi-seed averaging that removes the single-seed upward bias on
top-1. Both push live top-1 (0.690–0.826) *below* `boundaries[0]` → ~every day maps to bin 0. From the npz, the
bin-0 column degenerates to "double until streak ≈8, then **skip for every streak ≥10**." The lookup
(`mdp.py:159-164`) and the `mdp_policy_alignment` health check are both *correct* — the policy artifact is the
problem. **A re-solve that keeps `actual_pa` profiles will not fix this; re-solve on the `estimated_pa` basis
(`backtest_blend.py:733`).** Bonus: `actual_pa` profiles condition transition probs on realized PA counts (a mild
optimism leak), so the in-sample 8.17% rests partly on that surface — `estimated_pa` is the honest basis.

**[P1] ▸ Inference lookups stale by one played day; bpm collapses to the prior for single-meeting pairs** — `predict.py:187,253-258`
Every lookup is `df.dropna(...).groupby(...)[col].last()`. The row value at a batter's most recent played date was
computed with `shift(1)` (data strictly before that date), so the lookup **excludes the most recent game** for all
rolling features. Worst case is `batter_pitcher_shrunk_hr`: `_cum_hits_prior = cumsum - current_day`
(`compute.py:652`) means a pair's only-meeting date stores exactly the `0.2195` prior, so single-meeting pairs are
indistinguishable from never-met at inference. The 2026-04-30 fix added the lookup but not freshness. (Recommend a
human spot-check before acting — the fix touches the training/serving contract.)

**[P2] ▸ `batter_hard_contact_30g` leakage (shadow-only)** — `compute.py:544-551`
Rolling is over **PA rows** (not date-aggregated) so post-shift the window still holds same-date PAs; `.shift(1)`
is applied after `reset_index(drop=True)` so each batter's first row inherits the **previous batter's** value; and
`window=120` contradicts the `_30g` name. CONTEXT_COLS only → not in production picks, but it contaminates the
context-stack promotion eval. Fix: aggregate to (batter,date) then `shift(1).rolling(30,…)` like every other feature.

**[P2] ▸ Doubleheader blend scores averaged across both games** — `predict.py:701-713`
`blend_game_scores` is keyed by `batter_id` only; a batter in both games of a DH gets one probability averaging two
matchups, overwriting both slots. Fix: key by `(batter_id, game_pk)`.

**[P2] ▸ `_fetch_prior_lineup` hardcodes the 2026 season window** — `predict.py:323-324`
`startDate=2026-03-20&endDate=2026-12-31`. In 2027 the projected-lineup fallback returns last year's lineup. Fix:
derive the year from the prediction date (as `_refresh_season_data` already does at 731).

**[P2] ✓ `scripts/leakage_audit.py` is dead — `ImportError` on first line** — `leakage_audit.py:28`
`from bts.features.compute import _compute_pitcher_archetypes` — that function exists nowhere in `src/` (grep
confirms only this script references it). The CLAUDE.md safety rule "run leakage_audit.py after any feature change"
is non-executable, and even when it ran it covered only a few features (not bpm, bullpen, Statcast, or CONTEXT_COLS
— exactly where the live `platoon_hr` leak above hides). Fix: drop the archetype audit; add the promoted/context features.

**[P3] ▸ misc:** wind LF/RF parse differs train vs serve (shadow only, `compute.py:525` vs `predict.py:597`);
whiff codes omit blocked-swing `W`/foul-tip `T` (consistent, slightly biased); probable-pitcher cache is
CWD-relative with silent-empty fallback → `opp_bullpen_hr_30g` all-NaN off the systemd WorkingDirectory; COVID-DH
≤7-inning filter over/under-matches rain-shortened DH games.

---

## Strategy & contest-decision path

**[P1] ▸ `save_pick`/`save_streak` are bare `write_text`; every loader parses JSON unguarded** — `picks.py:286,441`
A crash/SIGKILL mid-write (deploy restarts are routine) leaves truncated JSON. `load_pick`/`load_streak`/
`load_saver_available` then crash `check-results`, `reconcile` (the self-healing recompute sits *behind* the crash
point at `picks.py:816`), scheduler polling, and `load_decision_streak_state` → no pick. The fetcher already has
`_atomic_write_json` (`cli.py:1392`) — reuse it (tmp + fsync + `os.replace`).

**[P2] ▸ Corrupt auto contest JSON raises *through* precedence, disabling the manual emergency override** — `contest_state.py:133-134`
Both files are parsed eagerly and `_parse_state_file` raises before the precedence check, so a corrupt
`contest_streak.json` blocks delivery even when a valid unexpired `contest_streak.manual.json` exists — the exact
case `set-contest-streak` is meant to rescue. Fail-closed (no bad pick) but costs contest days. Fix: parse each
file in its own try; treat unparseable as absent. (Also: `schema_version` is written but never read by the loader.)

**[P2] ▸ Fresh auto contest state pins MDP `saver=False` forever** — `contest_fetch.py:97` + `contest_state.py:213`
The profile API can't observe the mulligan, so `build_observation` always writes `saver_available=None`, which the
loader maps to `False` — discarding `model_saver_available`. With the fetcher running 4×/day this is the normal
path, so the saver-aware (more aggressive, higher-EV) policy line at streak 10–15 is unreachable. Fix: fall back to
`model_saver` when contest saver is None.

**[P2] ▸ `reconcile` 2 am recompute ignores saver semantics → clobbers a saver-preserved streak to 0** — `picks.py:864`
`elif r == "miss": break` with no saver replay. A miss-at-12-with-saver is correctly held at 12 by the 1 am
`check-results`, then reset to 0 by the 2 am reconcile. `regenerate.py:334-347` already does the replay — mirror it.

**[P2] ✓ Quality-bin floor collapse (the live `mdp_policy_alignment` WARN)** — `mdp.py:159-164`
Confirmed: indices clamp safely (no crash), but everything below `boundaries[0]` lands in bin 0. This is the
*symptom* of the M2 basis mismatch above; nothing wrong in the lookup itself.

**[P3] ▸** update-streak-then-save crash window can double-count a day (`cli.py:1759` / `scheduler.py:1384`); scheduler
lineup cascade opts out of strict detailed-status mode (`scheduler.py:719`, uncertain if deliberate); `bts run`
bypasses the `contest_state_required` gate (`cli.py:1100`); regenerated picks use `game_pk=0` which would crash
`reconcile`'s `check_hit(0,…)` if inside the 8-day window (`regenerate.py:465`).

---

## Scheduler & ops

**[P1] ✓ DH game-2 recheck bare `time.sleep` → watchdog SIGABRT** — `scheduler.py:1731-1732`
`for _ in range(10): time.sleep(dh_recheck_min*60)` with `dh_recheck_min=15` default → two un-pinged sleeps = the
`WatchdogSec=1800` → SIGABRT, restart, re-enter, repeat. Only fires on an unlocked-pick day with a doubleheader, and
the loop is vestigial (only updates `confirmed_sides`). Fix: `_watchdog_ping_sleep(...)` or delete the loop.

**[P1] ✓ UTC date default abandons the night on an evening restart** — `cli.py:1641`
`datetime.now(timezone.utc)` on an ET box: a restart (deploy / OOM / watchdog / crash) 20:00–24:00 ET initializes
`run_day` for ET-tomorrow → tonight's result polling, late-slate delivery, DH game-2, and end-of-day health are
abandoned; only the 1 am cron repairs the streak (never the Bluesky post). An undelivered pick on an all-late slate
is a missed contest day with no alert. Fix: default to ET (as `predict` does at `cli.py:854`).

**[P2] ▸ No-games day → ~30s restart thrash for ~30h** — `scheduler.py:151,1485`
`compute_wakeup_time([])` returns *today* 10 am (already past); after 8 pm ET `run_day` returns at the "No games"
guard with no idle sleep → restart every ~35s through an All-Star break / league off-day. Heartbeat goes stale all
day (external monitor pages) and NRestarts spikes → likely false `restart_spike` CRITICAL next game-day. Fix: idle
until tomorrow ~10 am ET on empty schedule.

**[P2] ▸ Double-post vectors** — `util.py:14`, `posting.py:155`, `scheduler.py:431`
`retry_urlopen` retries non-idempotent POSTs (a committed-but-lost-response post/DM re-sends up to 2×); and
`_deliver_and_lock_pick` posts *then* persists the `bluesky_posted` flag, so a crash between them re-posts on restart.
Fix: don't retry non-idempotent POSTs; persist an "attempted" marker before posting.

**[P2] ▸ Transient MLB-API error at game-final crashes the daemon** — `scheduler.py:1402` → `picks.py:604`
`resolve_daily_slot_results` → `check_hit` raises uncaught after retries; recovers via restart *except* in the
UTC-window above (games go final 10 pm–1 am ET). Poll-loop errors are contained — this is specifically the resolve step.

**[P2] ▸ `missed_pick_alert_min` is dead config** — `scheduler.py:1473`
Read into a local and never used; `orchestrator.example.toml` documents it as a pre-first-pitch alert. The only
signal on a delivery failure is the end-of-day `post_failure` DM, hours after the window closed. Fix: implement or delete.

**[P3] ▸** `orchestrate()` ignores `pick_delivery` → manual run public-posts a DM/private pick (`orchestrator.py:335`);
`heartbeat_watchdog._pulse` skips the systemd ping if the file write fails (`heartbeat.py:113`); failed mid-game
reply never retried (`scheduler.py:1389`); repo's `scripts/bts-scheduler.service` is the stale Pi5 unit (no
watchdog, `Restart=on-failure`) — rebuilding a box from the repo installs a daemon that dies after one day.

---

## Health & alerting

**[P1] ✓ realized_calibration runs on the biased streak-proxy path in prod** — `scheduler.py:1808` + `realized_calibration.py:134,195`
The scheduler's `run_all_checks(...)` passes neither `data_dir` nor `today` (✓ confirmed: `run_all_checks` *accepts*
both, they default None). With `data_dir=None` the check falls to `in_bucket.append((p, 1 if result=="hit" else 0))`
— the streak-result proxy whose own docstring says it misattributes DD-day primary hits and inflates +6.6pp → "+14pp".
The 8/15/25pp thresholds were recalibrated for the PA-frame path production never executes, even though
`pa_2026.parquet` exists on the box. **This plausibly drives the daily realized_calibration CRITICAL DMs — they're
partly an attribution artifact, not pure distribution shift** (refines `feedback_realized_calibration_alert_handling`).
Fix: pass `data_dir=Path("data/processed")` (+ `today`) in the scheduler call.

**[P1] ▸ contest_state gap is calendar-days, not settlement-steps → guaranteed false CRITICAL after any no-pick day** — `contest_state.py:77-85` (health)
`gap_days = (latest_resolved_pick − source_date).days`, CRITICAL at `>1`. Both anchors freeze over off-days, but the
*calendar* delta between them inflates: the first played day after an n-day no-pick gap yields `gap≈n` → false
"STALE; picks frozen" CRITICAL. The All-Star break (~5 weeks out) fires this deterministically. The 58c9adc
level-aware fix reduced the *nightly* false-CRITICAL but not this class. Fix: compare `source_date` to the
*second-latest resolved pick* (settlement-step lag), not calendar days.

**[P1] ▸ Crashed health check returns `[]` — dead smoke detector** — `runner.py:44-50`
`_safe_run` catches Exception, logs, returns `[]` — a check that raises every night produces no alert (journalctl
only). Four checks additionally self-swallow in their own `check()`. Fix: `_safe_run` should emit a CRITICAL
"check X crashed" alert.

**[P1] ▸ Day-keyed Tier-1 checks no-op on post-midnight / early-finish nights** — `scheduler.py:1808` + `post_failure.py:38`
`today=date.today()` + a single EOD run: result polling legitimately ends anywhere from ~16:30 ET to the 5 am cap, so
an EOD after midnight checks day D+1 (no pick exists → `[]`) and an EOD before 22:00 ET permanently suppresses
`post_failure` via its time guard. A delivery failure on a West-Coast-pick night is never alerted (pick_delivery,
postponed_pick, fallback_defer, analytics_artifacts_missing all affected). Fix: pass the scheduler's `date` as `today=`.

**[P1] ▸ Wedged-but-pinging scheduler fools the external monitor** — `heartbeat.py:96-127`
`heartbeat_watchdog` spawns a daemon thread writing `state=RUNNING` + `notify_watchdog()` every 60s *regardless of
body progress*, and wraps the full prediction cascade (incl. network). A hang inside → `check_heartbeat` sees fresh
"running" forever, systemd is fed, no pick is made, EOD never runs, the `*/5` hc-ping is unconditional → total
silence. Fix: pulse only on a monotonic progress counter the body must bump.

**[P1] ▸ Failed CRITICAL DM is not retried** — `alert.py:183-198`
On `send_dm` failure (after 3 in-call attempts) the path writes `status="failed"` and returns — no queue, no retry.
Persistent conditions re-fire tomorrow, but day-keyed one-shot CRITICALs (pick_delivery etc. for day D) are lost on a
Bluesky outage during that EOD. Fix: on the next EOD, prepend "yesterday's health DM failed (N CRITICAL)".

**[P2/P3] ▸** `predicted_vs_realized` stat-power gates broken (`n14` counts the whole 35-day lookback; `min_days_28d`
never enforced — `predicted_vs_realized.py:128`); `since_deploy_iso` is approximate three ways (committer-offset vs
UTC lexicographic compare; %cI is commit-time not deploy-time; a canary rollback moves HEAD's date backward → the
rolled-back model's picks pass the filter — `realized_calibration.py:94`); no-games day skips the whole health block;
`streak_validation` accepts booleans for `streak` (`isinstance(True,int)` — `:47`); Tuesday memory digest can void
the night's alerts on a type-corrupt row.

**Silently-inert-right-now check:** `pooled_training` (config-gated; pooled cron doesn't run) and intermittently the
post-midnight day-keyed checks above. Everything else is active. `mdp_policy_alignment` is a *true positive* today.

---

## Security & supply chain

**[P1] ✓ `.env` + `data/{health_state,leaderboard,hetzner_results,lineup_posting_times}` are NOT gitignored** — `.gitignore:1-6`
✓ `git check-ignore` returns NOT IGNORED for all five. `.env` on the box holds every secret (Bluesky app password,
R2 keys, healthchecks URLs). Not leaked today (deploy does `git reset --hard`, never pushes from the box), but one
reflexive `git add -A && git push` on the box publishes the lot to the public repo. Fix: add `/.env`, `.env.*`, and
the four dirs to `.gitignore`.

**[P2] ✓ Dashboard binds `0.0.0.0:3003` with zero app-layer auth — protected only by an out-of-band firewall** — `web.py:1549`
✓ The bind is all-interfaces and no route checks any token/cookie (grep clean). The hero card renders today's pick
*before* it posts to Bluesky (`web.py:1030`). **Mitigated today: 3003 times out from the public IPv4** (✓ external
probe) — a Hetzner cloud firewall or Tailscale ACL blocks it. But there is **no firewall config in the repo**, so the
guarantee is invisible and a single console change un-protects pre-publication picks + the `/api/audit-progress` SSH
fan-out. Defense-in-depth fix: bind `tailscale0`/`127.0.0.1`, or add basic-auth.

**[P2] ▸ `/api/audit-progress` — path traversal + absolute `seeds_file` + unauth SSH fan-out** — `web.py:1415-1426`, `audit_progress.py:44`
`dir`/`seeds_file` are unsanitized: `dir=../../..` is a directory-existence oracle; a found `boxes.json` triggers
root-SSH to every listed IP (8-wide, 20s each) using the prod key, and the IPs come back in the JSON. `provider` *is*
whitelisted (no RCE), and it's tailnet-gated while 3003 is firewalled. Fix: require auth; reject `..`/absolute paths;
confine via `Path.resolve().is_relative_to(root)`.

**[P2] ▸ Leaderboard username → path-traversal write** — `scraper.py:287` + `storage.py:65`
Arbitrary public usernames (`models.py:26` only checks `min_length=1`) are used directly as `{username}.parquet`
with `mkdir(parents=True)` → `username="../snapshots/2026-06-09"` could clobber a snapshot. High bar (needs a top-N
streak + MLB allowing `/`), but the code trusts the string. Fix: sanitize to `[A-Za-z0-9_-]` or hash.

**[P2] ▸ Leaderboard scrape aborts entirely on MLB schema drift** — `scraper.py:280-305`
Only `httpx.HTTPError` is caught; `parse_leaderboard_response` and the per-user `int(entry["userId"])` are outside
the try, so one `KeyError`/`ValidationError` kills the whole scrape (no parquet) — contradicting the "one bad user
won't abort" docstring. Fix: wrap parse + per-user in `except Exception: log; continue`.

**[P3] ▸** real email in the public-repo scraper User-Agent (`endpoints.py:77`); `appleboy/ssh-action@v1` floating
tag holds the root SSH key — pin to a SHA (`deploy.yml:24`); `exclude-newer=2026-04-11` is ~2 months stale (blocks
security patches — `uv.lock` is hashed + `--locked`, so just advance deliberately); dead unbuildable `Dockerfile`
(refs untracked `scripts/fly-*.sh`); `/api/live` `date` param traversal/parse oracle (`web.py:1455`); MLB names
interpolated into HTML without `html.escape` (defense-in-depth; the feared leaderboard-username XSS is **not**
present — that endpoint is JSON-only and carries no usernames); no `permissions:` block in deploy.yml.

**Confirmed clean:** no live secrets in the tracked tree or history (swept); all secrets sourced from env/keychain;
`cron-setup-hetzner.sh` hard-refuses a hardcoded healthchecks URL; tracked `.npz` load `allow_pickle=False`; no
`shell=True` with user input on the web path; deploy.yml has no script-injection and rolls back to the correct SHA.

---

## Tests & CI

**[P1] ✓ No automated test execution exists** — `deploy.yml:19-120`
Only workflow; triggers on push-to-`deploy` + dispatch; steps are reset / `uv sync` / restart / canary
(`systemctl is-active` + HTTP 2xx). No pytest anywhere. A commit breaking streak math, the DD rule, or contest
precedence ships as long as the process boots. Fix: a `test` job on push/PR (ubuntu, `uv sync --locked --extra
model` — the lightgbm manylinux wheel works on Linux, no libomp problem — `pytest -m "not slow"`), deploy
`needs: test`. Public repo → free Actions minutes. (1668 not-slow tests ran in 105s here.)

**[P1] ✓ "lightgbm-optional" is not implemented — absence is broken-red, not skip** — `predict.py:8`, `conformal.py:22`
No `importorskip`/skipif/`collect_ignore`. ✓ This session: 5 modules error at collection and 57 tests fail at
runtime, all `OSError: libomp.dylib` — and a collection interrupt runs **zero** tests, so the local loop is
broken-red (not falsely-green as assumed). `importorskip` wouldn't even catch the `OSError`. The core inference path
(`predict.py`, 829 lines) has **no automated home**: contract tests only locally; real-path tests run only if a
human invokes pytest on Hetzner, which nothing automates. Fix: per-module `try: import lightgbm / except
(ImportError, OSError): pytest.skip(allow_module_level=True)`, + install libomp on the loaner for local model tests.

**[P2] ▸ `simulate/blend_model_cache.py` (244 lines) has zero tests** — a stale/corrupt-cache bug silently changes
backtest results, and ship decisions are backtest-driven. Add round-trip + invalidation tests (key includes
seed/params/feature-cols).

**Coverage is otherwise excellent:** health is 1:1 per module; contest math is value-pinned (BOTH-must-hit,
saver window incl. boundaries, override>auto>legacy with injected `now`); dated prod-incident regressions are
reproduced as tests; hermeticity is strong (network mocked, decision clocks injected). Thin spots: `simulate/cli.py`,
`evaluate/backtest.py` (one import each).

---

## What's genuinely solid

- **Leakage discipline** in features is real: ~25 features uniformly aggregate to (entity, date) before `shift(1)`,
  structurally solving the doubleheader-leak class; `sync.py` does SHA-256-diffed uploads + atomic manifests +
  shrink guards; `build.py` asserts schema match with an auto-derived version.
- **The falsification harness is honestly adversarial:** LOSO with fold-local dependence params, per-bin `rho_pair`
  with a fail-closed bin-shift contract, and a DEFENDED gate that requires CI-lower ≥ half-headline so a too-wide CI
  can't "defend" by impotence. The hard-to-get-right math (MDP backward induction, CE-IS likelihood ratios, Wilson,
  BH/BY, Murphy decomposition, Poisson-binomial tails) is textbook-correct, and known independence violations are
  *measured*, not assumed away. ⚠ One caveat: `ope_eval.py` V_replay vs V_pi compares different horizons (180-day
  V_pi vs a short holdout-window replay) → structural disagreement even for a perfect model (`ope_eval.py:170,350`).
- **Watchdog discipline is otherwise complete** (every other >60s wait is wrapped); result-polling contest logic is
  correct (both-picks early-success, `_earliest_pick_game_et`, void-as-terminal); the a436827 game_pk-normalize fix
  genuinely closes the NaN/type double-down hole.
- **fetch-contest-streak is properly fail-safe** (identity + prior-account + currentness gates, atomic writes, never
  overwrites the good value on failure), and the staleness *freeze* really does propagate into the decision
  (`streak=max(model,contest)`, `allow_double=False`), not just into a health alert.
- **Regression-test culture** is the best the test auditor reported seeing at this scale — value-level asserts on
  real dated incidents, not tautological mock-echoes.

---

## Notable cross-cutting theme

Several findings share one root: **single end-of-day health run + day-derived-from-UTC + restart-any-time**. That
trio is behind the evening-restart-abandons-night bug, the post-midnight day-keyed-check no-ops, and the no-games-day
thrash. Fixing the date basis to ET and making the scheduler resume an unfinished prior-date state on startup would
retire several P1/P2s at once.
