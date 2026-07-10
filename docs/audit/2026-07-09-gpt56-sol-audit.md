# GPT-5.6 sol external audit — 2026-07-09

**Setup:** Codex CLI 0.144.0, model `gpt-5.6-sol` (released 2026-07-09), `model_reasoning_effort=xhigh`, full repo access (`/Users/eric/projects/bts` @ 41dbda7) + read-only ssh to bts-hetzner (@ 115c0cf). Directed by Claude (Fable 5) with a neutral adversarial brief: assume defects, 7 scope dimensions, settled-decisions fence, mandatory self-refutation per finding, ≤15 findings. Runtime ~24 min. Tamper-check clean (repo HEAD/status unchanged; box units/crontab/timer untouched; no restarts).

**Raw report:** 15 findings + could-not-verify list + wildcards. Preserved verbatim below the triage. Claude independently verified the evidence lines for F1–F4, F6, F9, F10, F15 against source (all reproduced exactly) and ran the F2 production-impact check.

## Triage verdicts (Claude, post-verification)

### Tier A — confirmed, cheap, high-leverage fixes (recommend implementing)
- **F1 (HIGH, confirmed live twice: 7/08 and 7/09):** check-pick-entered treats `alerted` as terminal for the day — one DM, then never re-verifies entry or escalates. Both real incidents this week follow the pattern (7/08: DD leg never entered → +1 not +2; 7/09: entry happened only after an out-of-band warning). Fix: `alerted` → nonterminal; keep re-checking every 15 min until `confirmed` or cutoff; throttled escalation DMs (e.g. T−30/T−15); health source for "committed pick with no confirmed entry near cutoff".
- **F3 (HIGH):** `save_state` uses bare `write_text` (scheduler.py ~468) for the exact file the daemon parses crash-unprotected at startup — *after* it has already written the heartbeat and signaled READY/watchdog. Torn write → 30s crash-restart loop that stays externally green (check_heartbeat: any fresh `running` = healthy; restart-spike check runs only at EOD inside the crashing process; deploy canary = `is-active` at +30s). Fix: `atomic_write_text` (helper already exists in util.py), corrupt-state recovery path with its own alert, READY only after init, external NRestarts/PID-churn check in cron.
- **F4 (MEDIUM):** calibration / projected_lineup / post_failure / disk_fill swallow their own exceptions and return `[]`, structurally defeating `_safe_run`'s promised CRITICAL-on-crash (its docstring cites audit H3 — these catch-alls bypass that exact fix). Fix: let unexpected exceptions reach `_safe_run`; distinguish healthy / insufficient_data / evaluation_failed.
- **F6 (MEDIUM):** realized_calibration resets its sample on EVERY deploy stamp (11 deploys 6/29–7/8) → `min_bucket_n` unreachable during active dev → the check is effectively dead exactly when regressions are most likely. Fix: reset on a production-regime fingerprint (model_pickle_sha256 + policy_npz_sha256 + FEATURE_COLS + probability basis — already stamped on every pick file) instead of wall-clock deploy time.
- **F15 (LOW, one-liner):** `bts preview` computes "tomorrow" in UTC → run manually 20:00–23:59 ET it writes day-after-tomorrow. Fix: ET-based date + boundary tests.

### Tier B — confirmed, Eric's call (design/infra decisions)
- **F2 (HIGH mechanism, zero realized harm):** early-lock gate checks only the PRIMARY's lineup confirmation; the selected DD can be projected. 11 production days delivered projected DDs (4 locked 51–133 min early). Impact check: all 11 DD players actually played (9 hit / 1 miss / 1 unknown-slot); no scratch/void ever realized. Residual risk = tail scratch/displacement. Cheap gate if wanted: require both slots confirmed for normal early lock (final-fallback path exempt); + scheduler-level test.
- **F5 (MEDIUM):** R2 sync excludes `data/picks/` entirely (decisions, contest ledger, delivery IDs, skip-shadow markers, manual saver state — 6.7MB); no box-level backup exists (`find` for backup/restic/borg/snapshot = 0). Box loss = irrecoverable operational state (saver flag is deliberately not inferable). Fix: encrypted versioned backup of data/picks + key health_state, restore drill. (Aligns with post-MacBook-loss backup posture; Hetzner provider snapshots unverified — console check.)
- **F8 (MEDIUM):** R2 "manifest-last" isn't transactional — stable keys overwritten before manifest replacement; interrupted sync leaves old manifest pointing at new bytes; `verify_manifest` checks metadata/age only. Fix: content-addressed keys + atomic manifest publish + object-hash verification.
- **F11 (MEDIUM, cheap):** deploy.yml runs mutable-tag actions (`checkout@v4`, `setup-uv@v5`) on the same runner that later receives the root SSH key (only ssh-action is SHA-pinned); no `permissions:` block. Fix: SHA-pin everything + `permissions: contents: read`.
- **F12 (MEDIUM):** live systemd units are hand-maintained snowflakes; tracked `scripts/bts-scheduler.service` is materially stale (Type=simple, `/home/stonehengee` path, on-failure/300s vs prod notify/watchdog/always/30s). DR from repo = wrong unit. Fix: track real unit templates + read-only drift check.
- **F13 (LOW):** daemon result-polling (until 05:00) and the 01:00 cron scorer share an unlocked read-modify-write on streak/pick state (only saver has flock). No overlap observed in July logs; narrow but real. Fix: shared flock around reload→resolve→update→save.
- **F14 (LOW):** static_snapshots growing ~180–195 MB/day uncompressed (1.1GB now, disk 7%); no retention policy. Fix: gzip/content-address + retention + bytes/day watch.
- **F7 (MEDIUM as filed, likely accept):** `POST /saver/transition` has no auth beyond tailnet membership; absent Origin accepted by design (tested). Precedent: F9-2026-07-01 dashboard-auth was WON'T-FIX ("nothing sensitive at risk") — but this endpoint MUTATES state that feeds the MDP. Decide: extend the won't-fix, or add a mutation token.

### Tier C — methodology (present, don't auto-action)
- **F9 (CHALLENGES SETTLED DECISION — properly labeled, new evidence real):** the estimated_pa profiles still condition on the REALIZED starter (groupby-first pitcher_id from the outcome frame) and realized participation (11.33% of rows dropped, measured over 912 days) — production ranks a noisier pregame set than the eval basis. Doesn't reopen PIT; proposes a bounded sensitivity using already-captured production slates + probable-pitcher data. Verified against backtest_blend.py:625–648 — the mechanism is exactly as described. Eric decides whether the sensitivity is worth a session.
- **F10 (MEDIUM):** skip-shadow verdict = repeated nightly 95% Wilson looks vs a point 0.744 whose derivation lives only in a dead `/tmp` script reference. Even before fixing the peeking statistics: version the breakeven derivation into the repo. Then: pre-registered N or confidence sequence; stratify on regime changes.
- **Wildcards:** slate_auc counts n_days pre-join (lag can make "20 days" < 20 resolved days — telemetry nit); park_drag shadow cache keyed on mtime:size not content hash (shadow-only, not promoted).

### Falsified by the auditor (negative results worth keeping)
History-safe secret scan: 3,454 reachable blobs, zero key/token/JWT matches, zero .env paths. Dashboard binding exactly tailnet-only (100.100.43.24:3003). park_drag export uses strict-prior `allow_exact_matches=False`. All 17 existing decision.json files pass invariant checks.

### Could not verify (guardrail-blocked; human/console steps)
R2 object integrity + restore drill; Hetzner console snapshots/firewall; GitHub token permissions + deploy-branch protection + secret scoping; crash-loop reproduction (staging only); fast test suite (sandbox blocked uv cache — run `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -m "not slow" ...` outside sandbox).

## Fix-batch adversarial review round (same day, gpt-5.6-sol, xhigh)

The Tier-A implementation branch was itself adversarially reviewed by a fresh
GPT-5.6 instance (read-only, full repo). 12 findings; dispositions:

**Fixed on the branch (8):**
- **#1 (HIGH)** `present_unverified` was terminal `confirmed` — a wrong player hidden by a crosswalk gap ended the day's checking. Now: only exact `match` is terminal; unverified presence re-verifies every run; a missing slot COUNT is a `mismatch` regardless of crosswalk coverage (new check in `pick_entry_status`).
- **#2 (HIGH, would have shipped F6 dead)** the regime fingerprint included `model_pickle_sha256`, but the blend retrains DAILY (box-verified: sha differed 7/06→07→08 while policy+env stayed stable across a deploy) → one-day pools. Fingerprint is now `(policy_npz_sha256, feature_env_hash)`. Residual documented: predictor-code-only changes don't reset (future: explicit regime-version stamp).
- **#7 (MED)** partial/missing stamps on the NEWEST pick no longer silently adopt an older pick's regime — falls back to the wall-clock deploy filter.
- **#3 (HIGH)** single 20-min churn window missed slow (~10-min-cycle) crash loops → multi-horizon windows (20m/+3, 60m/+3, 180m/+4).
- **#10 (LOW)** churn auxiliary state hardened: `{"samples": null}`, naive timestamps, malformed entries can't crash the monitor; whole churn block wrapped so the liveness ping always goes out.
- **#4 (MED)** `pick_entry` + `scheduler_state_integrity` added to ALWAYS_ATTENTION_WARN_SOURCES (they were journal-only); integrity lookback 3→7 days to survive multi-day breaks.
- **#5 (MED)** per-file `OSError` no longer swallowed in calibration/projected_lineup lookback scans — content corruption skips, filesystem failures propagate.
- **#8 (LOW)** window boundary off-by-one: exactly first_pitch−5 is now outside the window (entry already locked; no "0 min to submit" DM).
- **#12 (LOW)** `--dry-run` no longer quarantines (mutates) a corrupt live state file.

**Deferred (3):**
- **#6 (MED)** postponed-game awareness in the entry checker + pick_entry source (nag DMs / false EOD WARN for a selection that will Pass) — needs a product call on status-API integration; pre-existing behavior, escalations amplify it mildly. → Eric.
- **#9 (LOW)** lookback off-by-one (31 calendar days) — pre-existing, shared pattern across sources; cosmetic for a 30d window.
- **#11 (LOW)** DM-sent-but-marker-write-failed re-DM loop — outbox machinery is YAGNI for a disk-failure-only scenario.

Suite after both rounds: 1710 fast tests green (baseline 1651).

---

## Raw GPT-5.6 sol report (verbatim)

1. **[SEV: HIGH] [D1/D2] The manual-entry guard treats “warning delivered” as terminal and never verifies that the contest entry was corrected.**

   - **Evidence:** [cli.py:1593](/Users/eric/projects/bts/src/bts/cli.py:1593) loads the daily marker; [cli.py:1602](/Users/eric/projects/bts/src/bts/cli.py:1602) returns immediately for both `confirmed` and `alerted`; [cli.py:1668](/Users/eric/projects/bts/src/bts/cli.py:1668) writes `status="alerted"` immediately after sending the DM. Fetch/schema/auth errors also return quietly at [cli.py:1621](/Users/eric/projects/bts/src/bts/cli.py:1621). Live command:

     ```text
     {'date': '2026-07-09', 'status': 'alerted', 'reason': 'no_pick',
      'checked_at': '2026-07-09T18:00:03.575409-04:00'}
     check-pick-entered: DM sent to stonehengee.bsky.social
     check-pick-entered: already alerted for 2026-07-09
     check-pick-entered: already alerted for 2026-07-09
     check-pick-entered: already alerted for 2026-07-09
     ```

     Therefore the last actual contest check found no entry at 18:00 ET; later runs did not query again.
   - **Failure scenario:** No entry or a mismatched/missing DD is detected → one DM is sent → the owner overlooks it or only partially fixes the entry → every subsequent cron invocation exits before checking the account → the cutoff passes with no valid contest entry.
   - **Self-refutation:** I verified that DM failures remain retryable and that the mechanism successfully sent today’s warning. It survived because successful delivery of a warning is not successful entry into MLB, and no other health source closes that loop.
   - **Smallest safe fix:** Make `alerted` nonterminal. Continue read-only entry verification every 15 minutes until `confirmed` or cutoff; throttle repeat DMs at explicit escalation points such as T−30 and T−15. Health should flag any committed pick lacking `confirmed` near cutoff.

2. **[SEV: HIGH] [D1] The normal early-lock gate can deliver a double-down whose second hitter is still projected.**

   - **Evidence:** `should_lock` rejects only a projected primary at [strategy.py:223](/Users/eric/projects/bts/src/bts/strategy.py:223); otherwise a projected hitter merely participates in the probability-gap test at [strategy.py:226](/Users/eric/projects/bts/src/bts/strategy.py:226). Double selection independently takes the highest different-game candidate at [strategy.py:368](/Users/eric/projects/bts/src/bts/strategy.py:368). The scheduler passes only `daily.pick` as `top_pick` at [scheduler.py:887](/Users/eric/projects/bts/src/bts/scheduler.py:887). The existing test explicitly asserts that confirmed `.85` plus projected `.80` locks at [test_strategy.py:609](/Users/eric/projects/bts/tests/test_strategy.py:609).

     Production scan:

     ```text
     delivered_confirmed_primary_projected_dd 11
     dates 2026-04-06,2026-04-15,2026-04-18,2026-04-29,2026-05-01,
           2026-05-07,2026-05-10,2026-05-21,2026-05-22,2026-05-25,2026-05-28
     ```

     Four were locked 51.6–132.6 minutes before the earliest game, beyond the 35-minute fallback, proving this was not solely final-fallback behavior.
   - **Failure scenario:** Primary confirmed at .85, selected DD projected at .80, gap .05 → `should_lock=True` → both picks are sent and lock at the earlier first pitch → the projected DD is absent or displaced in the real lineup → one half receives no/poor PA opportunity and can reset the streak.
   - **Self-refutation:** The gap rule is valid for a single pick competing against projected alternatives, and `projected_lineup` health measures rolling frequency. It survived because the scheduler applies that single-pick rule unchanged to a selected double, while health only alerts above 30% over 14 days and does not prevent an individual unsafe commit.
   - **Smallest safe fix:** Before normal early lock, require both selected slots to be confirmed. Preserve any deliberate projected-data behavior only in the explicitly logged final-fallback path. Add a scheduler-level test where the projected contender is the selected DD.

3. **[SEV: HIGH] [D2/D3] A deterministic startup crash can refresh the heartbeat every 30 seconds and remain externally “healthy” indefinitely.**

   - **Evidence:** The daemon writes `RUNNING`, `READY`, and watchdog notifications before schedule/state initialization at [scheduler.py:1893](/Users/eric/projects/bts/src/bts/scheduler.py:1893); it does not load state until [scheduler.py:1931](/Users/eric/projects/bts/src/bts/scheduler.py:1931). `scheduler_state.json` is written with bare `write_text` at [scheduler.py:450](/Users/eric/projects/bts/src/bts/scheduler.py:450) and parsed without recovery at [scheduler.py:471](/Users/eric/projects/bts/src/bts/scheduler.py:471), despite the atomic helper explicitly documenting this crash-loop class at [util.py:11](/Users/eric/projects/bts/src/bts/util.py:11). The external monitor declares any sub-five-minute `running` heartbeat healthy at [check_heartbeat.py:59](/Users/eric/projects/bts/scripts/check_heartbeat.py:59). Production:

     ```text
     Restart=always
     RestartSec=30
     WatchdogUSec=30min
     NRestarts=1
     ActiveState=active
     ```

     Restart-spike health runs only at end of day at [scheduler.py:2358](/Users/eric/projects/bts/src/bts/scheduler.py:2358). The deploy canary merely checks `is-active` after 30 seconds at [deploy.yml:91](/Users/eric/projects/bts/.github/workflows/deploy.yml:91).
   - **Failure scenario:** Power loss or concurrent write tears `scheduler_state.json` → each launch refreshes heartbeat and signals ready → JSON parsing crashes → systemd restarts after 30 seconds → both heartbeat monitors stay green, the canary may pass during an active interval, and all lineup/delivery work is missed.
   - **Self-refutation:** Production is not currently looping—`NRestarts=1`, current heartbeat/state parse, and today’s decision is committed. It survived because watchdogs do not help when the process exits and restarts, and the only restart-counter check is unreachable inside this loop.
   - **Smallest safe fix:** Atomically write scheduler state, recover a corrupt file as a separately alerted initialization failure, and signal `READY` only after successful state initialization. Add an independent cron/host check for PID churn and `NRestarts` that does not run inside the scheduler.

4. **[SEV: MEDIUM] [D2] Multiple health checks suppress their own failures, defeating the runner’s promised CRITICAL-on-crash behavior.**

   - **Evidence:** The runner says a crashing smoke detector becomes CRITICAL at [runner.py:46](/Users/eric/projects/bts/src/bts/health/runner.py:46). But calibration catches every exception and returns no alert at [calibration.py:187](/Users/eric/projects/bts/src/bts/health/calibration.py:187); projected-lineup does the same at [projected_lineup.py:70](/Users/eric/projects/bts/src/bts/health/projected_lineup.py:70); delivery health treats unreadable pick JSON as no alert at [post_failure.py:43](/Users/eric/projects/bts/src/bts/health/post_failure.py:43); disk-health I/O failure returns no alert at [disk_fill.py:36](/Users/eric/projects/bts/src/bts/health/disk_fill.py:36). The generic five-minute cron ping at [cron-setup-hetzner.sh:60](/Users/eric/projects/bts/scripts/cron-setup-hetzner.sh:60) is unrelated to these evaluations.
   - **Failure scenario:** A dependency/schema/file-permission regression makes a check raise internally → it logs and returns `[]` → `_safe_run` sees a successful empty result → no DM is dispatched → the monitored failure remains invisible.
   - **Self-refutation:** The errors are recorded in journald, and checks that let exceptions escape are correctly converted. It survived because there is no journal-error alerting, while several production checks explicitly consume the exceptions.
   - **Smallest safe fix:** Let unexpected exceptions reach `_safe_run`, or return a CRITICAL `health_runner` alert. Represent `healthy`, `insufficient_data`, `disabled`, and `evaluation_failed` as distinct statuses rather than all as an empty alert list.

5. **[SEV: MEDIUM] [D4] R2 omits the irreplaceable live decision and manual saver state.**

   - **Evidence:** R2 sync includes `pa_*.parquet`, probable-pitcher lookup, and MDP policy only at [sync.py:152](/Users/eric/projects/bts/src/bts/data/sync.py:152) and [sync.py:188](/Users/eric/projects/bts/src/bts/data/sync.py:188). `data/picks/` is ignored at [.gitignore:8](/Users/eric/projects/bts/.gitignore:8). The saver file is explicitly the manual authority at [ARCHITECTURE.md:279](/Users/eric/projects/bts/ARCHITECTURE.md:279). Production commands returned:

     ```text
     6.7M /home/bts/projects/bts/data/picks
     find /home/bts -maxdepth 4 ... '*backup*|*restic*|*borg*|*snapshot*' | wc -l
     0
     ```

   - **Failure scenario:** Box/filesystem loss → repo and R2 restore model/data artifacts but not saver state, contest ledger, decisions, delivery IDs, skip-shadow records, or same-day lock state → the restarted system cannot determine saver availability or whether today’s DM already went out.
   - **Self-refutation:** Current streak can be fetched again from the contest profile, and much history can be reconstructed. It survived because saver use is intentionally not inferable, while same-day delivery idempotency and decision provenance are also absent from R2. Provider-level snapshots remain unverified.
   - **Smallest safe fix:** Add encrypted, versioned backup of `data/picks/` and essential `data/health_state/`, excluding `.env` and cookies. Perform a restore drill that demonstrates saver and current-day delivery recovery.

6. **[SEV: MEDIUM] [D6] Realized-calibration health resets on every deployment, including changes that do not alter the production model regime.**

   - **Evidence:** The check derives a universal deploy cutoff at [realized_calibration.py:87](/Users/eric/projects/bts/src/bts/health/realized_calibration.py:87), drops every pre-stamp pick at [realized_calibration.py:197](/Users/eric/projects/bts/src/bts/health/realized_calibration.py:197), and emits nothing below five bucket observations at [realized_calibration.py:234](/Users/eric/projects/bts/src/bts/health/realized_calibration.py:234). The runner always supplies that cutoff at [runner.py:177](/Users/eric/projects/bts/src/bts/health/runner.py:177), while deploy stamps every success/rollback at [deploy.yml:147](/Users/eric/projects/bts/.github/workflows/deploy.yml:147).

     Production stamp is `2026-07-08T04:39:53Z`. Reflog shows 11 deployments from June 29 through July 8. The latest deployment range added park-drag to the context/shadow stack but did not change production `FEATURE_COLS` or the MDP artifact. The only resolved post-stamp slots on July 8 were `.7461` and `.7415`, outside the monitored `[.75,.80)` bucket; July 9 was unresolved during inspection.
   - **Failure scenario:** Frequent docs/shadow/ops deploys repeatedly erase the accumulated sample → `min_bucket_n` is never reached → genuine production overconfidence produces no alert during the season.
   - **Self-refutation:** Resetting is correct when a deployment actually changes the production model or probability definition. It survived because the reset key is wall-clock deployment time, not model/policy/feature provenance already stored on pick records.
   - **Smallest safe fix:** Define a production-regime fingerprint from model hash, production feature set, probability basis, and policy hash. Reset only when that fingerprint changes; expose prolonged `insufficient_n` as health status.

7. **[SEV: MEDIUM] [D5] Any permitted tailnet peer can mutate the live saver flag without application authentication.**

   - **Evidence:** Missing `Origin`/`Referer` is explicitly accepted at [web.py:179](/Users/eric/projects/bts/src/bts/web.py:179), and the test requires that behavior at [test_web_saver.py:89](/Users/eric/projects/bts/tests/test_web_saver.py:89). `POST /saver/transition` performs the state mutation without a token/session at [web.py:1774](/Users/eric/projects/bts/src/bts/web.py:1774). Production binding is correctly tailnet-only:

     ```text
     LISTEN 100.100.43.24:3003
     tailscale0 inet 100.100.43.24/32
     ```

   - **Failure scenario:** A compromised or over-permitted tailnet peer sends a raw POST without `Origin`, using the visible current state as `expected_prior` → saver flips `active↔used` → the next live MDP decision uses incorrect saver availability.
   - **Self-refutation:** The dashboard is not bound to the public interface, and expected-prior/transition guards prevent arbitrary states. It survived because network membership is the sole authorization control; absent-origin acceptance deliberately permits non-browser mutation.
   - **Smallest safe fix:** Require a mutation token or identity-aware Tailscale proxy policy; reject absent Origin for browser form submissions. Keep the existing file lock and expected-prior guard.

8. **[SEV: MEDIUM] [D4] The R2 “manifest-last” protocol is not transactional because data objects are overwritten before the manifest.**

   - **Evidence:** Changed files overwrite stable keys directly at [sync.py:179](/Users/eric/projects/bts/src/bts/data/sync.py:179); only after all uploads does the code replace the manifest at [sync.py:221](/Users/eric/projects/bts/src/bts/data/sync.py:221). Restore downloads directly to the final path and deletes it on checksum mismatch at [sync.py:294](/Users/eric/projects/bts/src/bts/data/sync.py:294). `verify_manifest` checks only manifest metadata/age, not object contents, at [sync.py:310](/Users/eric/projects/bts/src/bts/data/sync.py:310).
   - **Failure scenario:** `pa_2026.parquet` upload succeeds, a later upload or manifest write fails → old manifest remains but its object key now contains new bytes → a fresh restore downloads an object whose hash disagrees with the old manifest and fails. Metadata verification still reports the old manifest as present/fresh.
   - **Self-refutation:** Each S3 object replacement is atomic, and the next successful sync will repair the mismatch. It survived because atomicity of individual objects does not provide multi-object/manifest atomicity, leaving the backup inconsistent during exactly the failure window it is meant to survive.
   - **Smallest safe fix:** Upload content-addressed/versioned keys such as `<sha256>/pa_2026.parquet`, then atomically publish a manifest pointing at them. Restore into a temporary path, verify, then replace. Integrity health must verify referenced objects, not only manifest age.

9. **[SEV: MEDIUM] [D6] CHALLENGES SETTLED DECISION — `estimated_pa` profiles still condition on realized participation and the realized opposing starter.**

   - **New evidence:** The prior decision calls `estimated_pa` “good enough” at [2026-07-06-strategy-model-lever-investigation.md:55](/Users/eric/projects/bts/docs/audit/2026-07-06-strategy-model-lever-investigation.md:55). But the implementation defines the starter as the first actual `pitcher_id` in the outcome frame at [backtest_blend.py:629](/Users/eric/projects/bts/src/bts/simulate/backtest_blend.py:629), retains only batters who actually faced that pitcher at [backtest_blend.py:636](/Users/eric/projects/bts/src/bts/simulate/backtest_blend.py:636), and obtains game outcomes from the same realized rows at [backtest_blend.py:643](/Users/eric/projects/bts/src/bts/simulate/backtest_blend.py:643). A light read of the captured five-season production profiles found:

     ```text
     ALL days 912 total 243533 kept 215942 dropped 27591 drop_pct 11.33
     ```

   - **Failure scenario:** Pregame probable starter or projected lineup differs from the eventual participants → production must rank the uncertain pregame slate, while the profile uses the eventual starter and removes nonparticipants → model/policy and 0.744-breakeven claims inherit a cleaner candidate set than live serving.
   - **Self-refutation:** Most removed rows are probably bench/pinch hitters production would also exclude, and confirmed lineups make much actual order information legitimately knowable. It survived because actual starter identity and participation still come from the outcome feed, and no sensitivity analysis establishes that the resulting policy metrics are invariant.
   - **Smallest safe verification:** Do a bounded sensitivity using already captured production slates/probable-pitcher data for available dates, changing only candidate eligibility and starter identity. This is not a point-in-time platform proposal.

10. **[SEV: MEDIUM] [D6] The skip-shadow’s eventual binary verdict uses repeated ordinary 95% intervals against an unreproducible point threshold.**

   - **Evidence:** The 0.744 derivation exists only as `/tmp/skip_qdelta.py` according to [2026-06-20-skip-policy-shadow.md:17](/Users/eric/projects/bts/docs/audit/2026-06-20-skip-policy-shadow.md:17). The code computes an ordinary Wilson interval at [skip_policy_shadow.py:46](/Users/eric/projects/bts/src/bts/skip_policy_shadow.py:46) and emits above/below verdicts whenever the repeatedly refreshed interval excludes 0.744 after `n≥30` at [skip_policy_shadow.py:212](/Users/eric/projects/bts/src/bts/skip_policy_shadow.py:212). Status is rebuilt nightly at [skip_policy_shadow.py:288](/Users/eric/projects/bts/src/bts/skip_policy_shadow.py:288). Current production status:

     ```text
     resolved=6 hits=4 rate=.6667
     Wilson CI=[.29999,.90323] breakeven=.744 verdict=insufficient_n
     ```

   - **Failure scenario:** The dashboard examines the interval after every new divergent day → repeated looks inflate the probability of eventually crossing one side by chance → a transient crossing is presented as “skip validated” or “skip costs streaks,” despite breakeven uncertainty and model-era changes.
   - **Self-refutation:** The current result is honestly `insufficient_n`, and the 30-day floor prevents very early calls. It survived because a fixed-N 95% interval is not time-uniform under unlimited nightly monitoring, and the derivation cannot currently be reproduced from repository artifacts.
   - **Smallest safe fix:** Version the breakeven derivation and inputs. Pre-register a terminal sample size or use a confidence sequence/e-process; compare against the documented `.742–.752` uncertainty band and stratify/reset on production-regime changes.

11. **[SEV: MEDIUM] [D5] Mutable actions execute in the deploy job before a pinned action receives the root SSH key.**

   - **Evidence:** Deploy uses `actions/checkout@v4` and `astral-sh/setup-uv@v5` at [deploy.yml:33](/Users/eric/projects/bts/.github/workflows/deploy.yml:33), while only `appleboy/ssh-action` is SHA-pinned at [deploy.yml:52](/Users/eric/projects/bts/.github/workflows/deploy.yml:52). That later step receives `username: root` and the private key at [deploy.yml:57](/Users/eric/projects/bts/.github/workflows/deploy.yml:57). Unlike the test workflow, deploy declares no repository `permissions:` block.
   - **Failure scenario:** A moved/compromised major action tag executes on the same runner → leaves a process or modifies runner state before the SSH step → captures the later root credential or abuses an unnecessarily broad default `GITHUB_TOKEN` → full box compromise.
   - **Self-refutation:** The SSH action itself is pinned, and earlier actions are not directly passed the SSH secret. It survived because all steps share the runner trust boundary, and the workflow does not enforce immutable revisions or explicit token least privilege.
   - **Smallest safe fix:** Pin every action to a commit SHA and add `permissions: contents: read`. Separately scope the deploy credential to the minimum remote operations feasible.

12. **[SEV: MEDIUM] [D4] The live systemd contract is not reproducible from the repository and already materially differs from the tracked unit.**

   - **Evidence:** Production unit from `systemctl --user cat bts-scheduler.service` uses `Type=notify`, `NotifyAccess=all`, `WatchdogSec=1800`, `Restart=always`, and `RestartSec=30`. The tracked unit instead uses `Type=simple`, the obsolete `/home/stonehengee` path, `Restart=on-failure`, and `RestartSec=300` at [bts-scheduler.service:7](/Users/eric/projects/bts/scripts/bts-scheduler.service:7). Hashes differ:

     ```text
     repo 3dc527a5... scripts/bts-scheduler.service
     prod 74a3cf3e... ~/.config/systemd/user/bts-scheduler.service
     ```

     No canonical dashboard unit exists in the repository.
   - **Failure scenario:** Disaster recovery or manual reinstall uses the tracked unit → wrong user/path and no notify watchdog → service fails or loses the monitoring behavior assumed by code and deploy canary. Conversely, future production edits can drift without review.
   - **Self-refutation:** The current production units are running with the intended settings, and ordinary deployments do not reinstall them. It survived because that makes the correct configuration a hand-maintained snowflake rather than reviewed infrastructure.
   - **Smallest safe fix:** Track Hetzner-specific scheduler/dashboard unit templates and nonsecret orchestrator configuration. Add a read-only hash/diff health check; installation remains an explicit operator action.

13. **[SEV: LOW] [D3] The daemon and 01:00 scorer can concurrently perform an unlocked streak/pick read-modify-write.**

   - **Evidence:** The daemon polls until 05:00 and updates streak before saving the pick result at [scheduler.py:1719](/Users/eric/projects/bts/src/bts/scheduler.py:1719) and [scheduler.py:1824](/Users/eric/projects/bts/src/bts/scheduler.py:1824). Cron starts independent scoring at 01:00 at [cron-setup-hetzner.sh:50](/Users/eric/projects/bts/scripts/cron-setup-hetzner.sh:50); it also updates streak then saves the pick at [cli.py:2003](/Users/eric/projects/bts/src/bts/cli.py:2003). `update_streak` reads and writes separate state at [picks.py:465](/Users/eric/projects/bts/src/bts/picks.py:465). The only `fcntl.flock` in `src/bts` protects saver state, not scoring.
   - **Failure scenario:** A late game becomes final around 01:00 → both processes load `daily.result=None` → both update streak before either’s result marker excludes the other → streak is doubled/lost depending interleaving.
   - **Self-refutation:** July logs showed no actual overlap; the latest midnight result finalized at 00:26. The 02:00 forward replay and contest anchoring usually repair local state. It survived as a narrow but genuine transaction race.
   - **Smallest safe fix:** Use one shared `flock` around “reload daily → resolve → update streak → save daily,” with a second result check after acquiring the lock.

14. **[SEV: LOW] [D4] Static-feed archival is adding roughly 180–195 MB/day with no retention or compression policy.**

   - **Evidence:** Content changes are stored as complete JSON snapshots at [static_capture.py:125](/Users/eric/projects/bts/src/bts/leaderboard/static_capture.py:125); the module contains no prune/retention path. Production:

     ```text
     static_snapshots total 1.1G
     players 879M / 267 files
     units   169M / 172 files
     July 8: players 162,214,439 bytes; units 30,711,733 bytes
     Filesystem: 301G total, 21G used, 268G available
     ```

   - **Failure scenario:** Half-hour full lookup changes continue indefinitely → archive and future backup/inspection costs grow without bound → eventual disk pressure affects picks, model artifacts, and JSON atomic writes.
   - **Self-refutation:** Disk is only 7% used, content deduplication is functioning, and this season alone will not exhaust 268 GB. It survived as an unbounded operational trajectory, not an immediate outage.
   - **Smallest safe fix:** Store compressed/content-addressed snapshots, reduce full `players`/`units` lookup retention to the minimum required cadence, and monitor bytes/day plus projected exhaustion date.

15. **[SEV: LOW] [D1] `bts preview` defines “tomorrow” in UTC, producing the day-after-tomorrow during evening ET recovery use.**

   - **Evidence:** The command promises tomorrow at [cli.py:1215](/Users/eric/projects/bts/src/bts/cli.py:1215), but uses `datetime.now(timezone.utc) + timedelta(days=1)` at [cli.py:1235](/Users/eric/projects/bts/src/bts/cli.py:1235). There is no test for the default-date timezone boundary.
   - **Failure scenario:** Operator runs preview at 21:00 ET on July 9, when UTC is already July 10 → code adds one day and writes July 11 → July 10 dashboard remains blank or recovery work targets the wrong slate.
   - **Self-refutation:** Normal cron invokes preview at 03:00 ET, where UTC and ET calendar dates agree, and `--date` avoids the bug. It survived for manual evening/recovery execution, which is exactly when defaults should be safest.
   - **Smallest safe fix:** Compute `(datetime.now(ET).date() + timedelta(days=1)).isoformat()` and test both summer 23:00 ET and winter 20:00 ET.

## COULD NOT VERIFY

- **Today’s final MLB entry:** External contest access was prohibited. A human should open the MLB app and confirm that both July 9 recommended slots were entered correctly; the last automated check at 18:00 ET observed `no_pick`, and later runs did not re-query.
- **FAST suite:** `uv run ...` was blocked before test startup:

  ```text
  Failed to initialize cache at /Users/eric/.cache/uv
  Operation not permitted
  ```

  Run outside the sandbox: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -m "not slow"`.
- **Crash-loop behavior:** Verifying it would require corrupting state or restarting services. Reproduce only in staging by truncating a staged `scheduler_state.json` and observing heartbeat, canary, and restart-counter alerts.
- **R2 object integrity/restoreability:** Guardrails prohibited external R2 calls. In an isolated restore directory, download every manifest-referenced object, verify hashes, and test restore after an intentionally interrupted multi-file upload.
- **Provider controls:** Check Hetzner Console for snapshots/backups and firewall exposure of port 22; check Tailscale ACLs for which identities can reach port 3003.
- **GitHub controls:** Check repository default workflow-token permissions, `deploy` branch protection, environment approvals, and SSH-secret scoping in GitHub Settings.
- **Concurrent-score race:** It needs a staged late-final game or synchronization hooks to force both scorers through the transaction simultaneously.

## D7 wildcard observations

- The most important unasked boundary was **remediation acknowledgement**: several state machines treat “message sent” as completion, even when the required real-world action is manual and independently verifiable. Finding #1 is the live consequence.
- Shadow cache identity uses `mtime_ns:size`, not a content hash, at [park_drag.py:123](/Users/eric/projects/bts/src/bts/features/park_drag.py:123). A same-size, mtime-preserving restore could reuse a shadow model trained against different bytes. Normal producer replacement changes mtime and this is shadow-only, so I did not promote it.
- `slate_auc` increments `n_days` before joining outcomes at [slate_auc.py:113](/Users/eric/projects/bts/src/bts/health/slate_auc.py:113), but gates only post-join row count at [slate_auc.py:158](/Users/eric/projects/bts/src/bts/health/slate_auc.py:158). Under outcome-data lag, “20 days” need not mean 20 resolved days. Current status was `n_days=21`, `n_rows=5133`, `auc=.57675`; no current trigger, so this remains a telemetry defect rather than a ranked failure.
- Falsified candidates: a history-safe secret scan covered 3,454 reachable blobs with zero matches for private-key, GitHub/OpenAI/AWS/Slack/JWT patterns and zero `.env` paths; production dashboard binding was exactly tailnet-only; park-drag export uses strict-prior `allow_exact_matches=False` at [park_drag_producer.py:252](/Users/eric/projects/bts/src/bts/features/park_drag_producer.py:252); and a production scan found zero invariant mismatches across all 17 existing `decision.json` files.
