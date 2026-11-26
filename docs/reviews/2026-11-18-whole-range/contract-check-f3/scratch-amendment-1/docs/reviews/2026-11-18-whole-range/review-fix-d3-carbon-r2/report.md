# Whole-Range Review (Round 2, Carbon): Change Set D3 (`d57f4013` vs `24019a9a`)

## Verdict
**BLOCK**

## Blocking findings

- **B1 — `on_signal` during `publication=beginning` contains an unchanged box and stops the healthy live scheduler whenever any prior deploy record exists in `/run/bts-deploy/attempt`** (`.github/workflows/deploy.yml:591-594`, entered from `:647-650`).
  - **State and inputs:** A box has completed at least one deploy since boot, so `/run/bts-deploy/attempt` exists (it is never unlinked between deploys) and holds that prior run's record — e.g. `bts_deploy_attempt_v1 100 1 9999… 8888… committed - - - -` with `HEAD=9999…`, `bts-scheduler` active and enabled, and `data/.heartbeat` `{"state":"running"}` (or, on a re-run `100 2`, holds `100 1 … contained sync verified verified verified`). A new deploy (`RUN_ID=200`, `RUN_ATTEMPT=1`, `TARGET_SHA=2222…`) passes preflight (`:615-645`, where `:624` confirms `! same_ids` under the exclusive flock without changing checkout or services), sets `publication=beginning` (`:647`), arms `trap on_signal HUP INT TERM PIPE` (`:648`), and receives a trapped signal during `publish begun` (`:649`) **before** `mv -fT` replaces `/run/bts-deploy/attempt` (e.g. during `stage-record`).
  - **Wrong outcome:** `on_signal` runs `read_record 15` (`:585`), which reads Run 100's untouched record (`rc == 0`, `! same_record`), proving `begun` was not published. Because `:591` tests only `if [[ $publication == beginning ]] && (( rc == 3 )); then` (true only on the first deploy after a reboot) and omits `(( rc == 0 )) && ! same_record` (unlike `publish()` at `:398-410`), `on_signal` falls through to `contain interrupted 130` (`:594`) on an untouched box: it disables and stops Run 100's live `bts-scheduler`, overwrites `data/.heartbeat` with `deploy_failed` (`step="interrupted"`, `target_sha="2222222"`, `old_sha="9999999"`) while `HEAD` is still `9999…`, sends a false `/fail` POST, and overwrites `/run/bts-deploy/attempt` with `200 1 … contained interrupted …` (or erases Attempt 1's `step="sync"`/`old_sha` provenance and double-pages on re-run `100 2`).
  - **Smallest fix:** At `.github/workflows/deploy.yml:591`, replace `if [[ $publication == beginning ]] && (( rc == 3 )); then` with `if [[ $publication == beginning ]] && { (( rc == 3 )) || { (( rc == 0 )) && ! same_record; }; }; then` (and add a `publish-begun:{}:unpublished` test with a prior `committed` record on disk).
  - **Demonstrated by:** `.codex-review/wr-d3/r2/probe_r2_audit_new_logic.py` (`sig_during_beginning_with_prior_committed` and `sig_during_beginning_rerun_after_contained` vs fresh-boot control `sig_during_beginning_fresh_boot` in `probe_r2_audit_new_logic.log`).

## Notes

- **N1 — Real Debian `systemd` pruned `ControlGroup=""` on recovery of already-contained attempts:** On Debian `systemd` (cgroup v2), stopping `bts-scheduler` prunes its empty cgroup and clears `ControlGroup=` to `""`. Deploy `contain` captures pre-stop `user_group` while active and logs `Scheduler stopped: verified.`, but when `Recover failed deploy via SSH` then runs on that `contained` record (`:1286`), the unit is already inactive (`user_group=""` and post-stop `group=""`), so R3's rule (`:474,:1166`) logs `Scheduler stopped: unverified.`, omits `containment re-checked`, and rewrites `attempt` to `unverified` (`debian_pruned_cgroup` in `probe_r2_audit_new_logic.log`; masked in `tests/test_deploy_workflow.py:306` because `SHIM` returns `ControlGroup="/user-sched"` while inactive). Matches Amendment 3 R3 verbatim; note for checklist A7.
- **N2 — Adjacent pre-existing phrase in `INCIDENT.md:126-127`:** Per Amendment 3/3b (`one sentence each`), `:127` appends `For scheduler activation after a restore, follow §4 and use §5's checked deploy.`, while `:126` still ends `then enable + start deliberately.` Readers executing sentence-by-sentence before `:127` could manually start units.
- **N3 — Surviving Round-2 workflow test mutations (`probe_r2_mutations.log`):** Three extra scratch mutations survived all 296 workflow tests (`212` fast + `84` slow): (a) dropping `s.st_uid != 0` in `fs_code` `lockfile`/`mark` (`:187,:204,:879,:896`; `root_fstat` always sets `st_uid=0`), (b) skipping `$previous_group` in `check_stopped` (`:475-477,:1167-1169`) when post-stop `$group` is a different non-empty path, and (c) dropping `same_record &&` in `on_signal` (`:587`; no signal test pre-populates a prior run's `committed` record).

## 1. Replay of Round-1 blocking findings
Ran `.codex-review/wr-d3/r2/probe_r2_replay_round1.py` (`probe_r2_replay_round1.log`) on `d57f4013` (plus Python 3.11.18 `LC_ALL=C` check on `capture-contest`). Every Round-1 blocking finding is fixed on its original inputs:
- **R1 (Codex 1):** Failed containment `clock` (once and always, in deploy and recovery) still stops, disables, writes `deploy_failed`, POSTs `/fail`, and records `contained`.
- **R2 (Codex 2 / Carbon B1):** Late signals (`790s`/`795s`/`805s`, during wait, canary query, and post-`enable`/stamp) and post-`begun` `read_record` failure (`126`) all run `contain interrupted 130` and never print `DEPLOY NOT STARTED`.
- **R3 (Codex 3):** Populated `/populated-sched` with post-stop `ControlGroup=""` and always-blank `ControlGroup=""` both print `Scheduler stopped: unverified.`
- **R4 (Codex 4 / Carbon B3):** `{other}`, `{one,two}`, and `[1-1]` in `.env` send zero `curl` calls and print `unverified.`; all three `curl` sites pass `--globoff`.
- **R5 (Codex 5 / Carbon B2):** `commit-record` staging/rename exit `17` prints `(17)`; staging/rename/read `124`/`137` and expired Phase-2 budget print `(timeout)`; `publish-committed` is in `PHASE_FOUR`.
- **R6 (Codex 6) & R7 (Codex NUL):** Checklist `A5` defers all config/cron edits into `B8a`'s stopped/boot-disabled sequence (`test_round1_activation_edits_follow_the_boot_hold`); embedded `\0` inside a valid canary prefix is rejected as raw bytes (`Check contest-state failed (query).`).

## 2. Audit of Round-2 changes beyond Amendments 3 & 3b
Diffed `d3d05abd..d57f4013` (5 files touched; RD8 shared helper prefix byte-identical) and ran `probe_r2_audit_new_logic.py`:
- **`on_signal` & `publication` states (`:576-595,:647-651,:756-760`):** Once `publication=begun` (`:651`), no path prints `DEPLOY NOT STARTED` and no path exits `0` without a confirmed `committed` record (`publication=committing` uses a fresh 15 s `SECONDS` budget at `:584-590`). However, during `publication=beginning` (`:647-650`), `:591` checks only `(( rc == 3 ))`, causing **B1** whenever a prior deploy's record exists in `/run/bts-deploy/attempt`.
- **Containment fallback clock (`:294-299,:486-558`):** `local budget_start=$SECONDS` + single `clock()` sample (`budget_origin=0` on failure) bounds `contain` to ~280 s without external clock dependence (`slow_failing_contain_clock` completed all sub-steps in `4.88 s`) and reserves cumulative slices (`90 s` user stop, `90 s` legacy stop, `40 s` proof, `20 s` marker, `30 s` POST, `10 s` `publish contained`).
- **Recovery waits (`:358,:778,:1050,:1264`):** `start_phase 1580` gives `run 1505 5 lock flock -w 1500 9` its full `1500 s` kernel `flock` wait plus `>= 50 s` for record/mark work, while `contain` resets its own `280 s` budget (`<= 1860 s < 35m` `command_timeout`).

## 3. New and changed operator sentences

| Location & new/changed sentence | Worst state | Literal operator action & risk |
|---|---|---|
| Workflow `on_signal` at `publication=beginning` (`deploy.yml:591-594`) emitting `DEPLOY FAILED at interrupted…` | **B1**: signal before `begun` rename on box with prior `committed` (or `contained`) record | Sees `DEPLOY FAILED at interrupted` after untouched live scheduler was needlessly stopped and `.heartbeat`/`attempt` overwritten -> **BLOCK (B1)** |
| Workflow `Check commit-record failed (<status>).`, `Scheduler stopped: unverified.`, `Failure ping…` (`:401-406,:474-478,:251,:547`), raw-byte canary line (`:171-180`) | Post-stamp commit `17`/timeout, blank cgroup with live child or Debian recovery (N1), glob URL, NUL in canary | Escalates per `INCIDENT.md §4` on accurate status / `unverified` -> Safe |
| `CLAUDE.md:149`: `**Re-enable next season:** follow checklist B8a's activation order and deploy inside an approved sleep window.` | Offseason box with boot link enabled and stale 2026 state | Follows B8a (stop + boot-disable before edits, contest flags, checked deploy in sleep window) -> Safe |
| `INCIDENT.md:110`: `Apply §4's incident checks and §5's checked deploy before restarting through the workflow.` | Dashboard down pre-cutoff or with unreviewed `deploy` ref / refused live state | Applies §4 sleep/pick checks and §5 same-SHA + `accepted` state checks before dispatch -> Safe |
| `INCIDENT.md:127`: `For scheduler activation after a restore, follow §4 and use §5's checked deploy.` | Rebuilt box mid-season with live config and unverified restored state | Whole-bullet reader uses §4 + §5 checked deploy; sentence-by-sentence reader hits `:126` `enable + start deliberately` first (see N2) |
| `docs/ops/2027-season-start.md:14` (`A5`): `Follow B8a's activation order; do not edit delivery, intent or cron before B8a says so.` | Reading A1–A5 while research scheduler is boot-enabled with today's observation | Leaves delivery, intent, and cron untouched until B8a; a reboot at A5 stays in private/research -> Safe |
| `docs/ops/2027-season-start.md:31` (`B8a`: `--picks-dir` note; `dm`/`enter` edits & `.env`+`cron-setup-hetzner.sh show/install` inside stopped/boot-disabled sequence) | Custom `picks_dir`, or reboot / cron tick mid-preparation | Passes `--picks-dir` to all 5 commands; boot link is disabled before `dm`/`enter`/cron edits so reboot cannot start live scheduler -> Safe |

## 4. Focused tests and Round-2 mutation suite
- Focused 6-file pytest suite (`pytest_focused_all.log`): **421 passed in 1098.39s** (`-m "slow or not slow"`).
- Scratch mutations (`probe_r2_mutations.py` / `probe_r2_mutations.log`, 22 total):
  - **All 14 Round-1 mutations (`1..14`, including all 4 Round-1 survivors `9_drop_run_id_check_in_same_ids`, `10_curl_follow_redirects_in_contain_post`, `11_break_real_lockfile_and_mark_in_fs_code`, `12_drop_1500s_flock_wait_in_recovery`): DETECTED.**
  - **All 5 required Round-2 mutations (`15_on_signal_not_started_when_read_fails_after_begun`, `16_drop_fallback_clock_in_contain`, `17_accept_blank_control_group_as_absence`, `18_drop_globoff`, `19_collapse_commit_record_status_to_1`): DETECTED.**
  - **3 adversarial Round-2 mutations SURVIVED** all 212 fast + 84 slow workflow tests: `20_drop_st_uid_zero_check_in_lockfile_and_mark`, `21_skip_previous_group_when_post_stop_group_changed_nonempty`, and `22_drop_same_record_check_in_on_signal_committed` (documented in N3).

## What I ran
- `~/bts-runs/yield_run.sh .codex-review/wr-d3/r2/pytest_focused_all.log .venv/bin/python -m pytest -m "slow or not slow" tests/test_deploy_workflow.py tests/test_deploy_canary.py tests/test_heartbeat_staleness.py tests/test_heartbeat_churn.py tests/test_lightgbm_optional.py tests/test_web_saver.py` -> **421 passed** (`1098.39s`).
- `~/bts-runs/yield_run.sh .codex-review/wr-d3/r2/probe_r2_mutations.log .venv/bin/python .codex-review/wr-d3/r2/probe_r2_mutations.py` -> **19/22 detected, 3 survivors**.
- Direct shim probes (`< 1 min`): `probe_r2_replay_round1.py` (incl. Python 3.11.18 `LC_ALL=C` check) and `probe_r2_audit_new_logic.py`.

## What I did not run
- The full repository fast pytest suite (`brief.md` forbids running it).
- Live SSH, GitHub Actions runners, Debian `systemd`/cgroupfs/cron, Tailscale, or network requests (the box is deleted and network/real host commands are forbidden).
