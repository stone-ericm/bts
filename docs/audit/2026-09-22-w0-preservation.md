# W0 preservation evidence — profiles backup, pinned diagnostic, preliminary checkpoint

**Date:** 2026-09-22 · **Plan items:** W0.1, W0.1b, W0.2a (`docs/superpowers/plans/2026-09-14-season-wrap-plan.md`) · **Register:** `docs/audit/2026-09-22-exposure-register.md` §D

## W0.1 — estimated-PA 24-seed profile run (`data/hetzner_results/mdp_estpa_run`)

| Fact | Value |
|---|---|
| Content | 146 files: 24 seeds × 5 seasons `backtest_{2021..2025}.parquet` (120) + 25 `audit_validation_split.json` + `boxes.json`; 7.29 MB |
| Before | existed on Eric's Mac ONLY (`/Users/eric/projects/bts/data/hetzner_results/mdp_estpa_run`); the box had only `audit_full_48seed_v2` |
| Copy | `rsync -a` Mac → box `~/projects/bts/data/hetzner_results/mdp_estpa_run/` (2026-09-22 ~10:33 ET) |
| Manifest | `mdp_estpa_run.sha256` (146 lines, generated on the Mac; sha256 of the manifest file `d63fefffff0a8868de4d21c1aa680e352f9ee4badff158bd2a8a01c5e359e7c6`); also placed on the box at `data/hetzner_results/mdp_estpa_run.sha256` |
| Mac vs box | `sha256sum -c` on the box: **ALL 146 FILES MATCH** the Mac manifest |
| Restic | archive set (paths `data/leaderboard`, `data/hetzner_results`, `data/external`); snapshots `1b49f56f…` (10:51 ET) and `a2781532…` (10:56 ET, current) |
| Restore test | `restic restore <snap>:/data/hetzner_results --target <empty dir>` → 146 profile files restored; `sha256sum -c mdp_estpa_run.sha256` on the restored copy: **ALL 146 FILES MATCH** |
| Retention | the archive set applies rolling `forget` (`--keep-daily 14 --keep-weekly 12 --keep-monthly 12`); the profiles stay in every future archive snapshot as long as the directory remains on the box. The **final** W0.7 snapshot additionally gets a tag-isolated backup (plan W0.7). |

Gotcha recorded: restic snapshots store paths with the repo prefix stripped (`/data/hetzner_results/...`), so `restic ls`/`restore` must use that form, not `/home/bts/projects/bts/...`.

## W0.1b — pinned restore diagnostic

| Fact | Value |
|---|---|
| Script | `scripts/audit/confirm_mdp_policy_replay.py` (no arguments; reads `data/models/mdp_policy.npz` + `data/hetzner_results/mdp_estpa_run`; realized-sequence replay of mdp_deployed / always_single / always_double / double_until_50 on 120 trajectories per policy, naive and different-game DD pairing) |
| Inputs pinned | profiles manifest above; `mdp_policy.npz` sha256 `66d154717ae51afb…`; `mdp_tail_policy.npz` `dc5d0c9924431c11…` (not read by this script; recorded for pairing); `PYTHONHASHSEED=0` |
| Run | Mac (`uv run python …`) and box (`.venv/bin/python …`), 2026-09-22 |
| Output | 20 lines; **sha256 identical on both machines: `bdbb2a091daa85d0ee9598eb1f3b70c438fe98cc39ad355759c91a5cc54a201b`** |
| Headline rows (for recognition only, not a new result) | naive pairing: mdp_deployed mean_max 18.03, reach20 30.8%; always_double 18.37 / 42.5%; different-game DD: 17.73 / 31.7% vs 17.85 / 40.8%; AD vs MDP on max_streak 56 wins / 13 ties / 51 |
| Acceptance | a clean reader reproducing this hash from the restored profiles has reproduced the named diagnostic; the script's known replay shortcuts (7/13 audit) make it a reproducibility check, not policy acceptance |

## W0.2a — preliminary evidence checkpoint (`prelim-20260922`)

| Fact | Value |
|---|---|
| Location | box `data/hetzner_results/season_2026_snapshot/prelim-20260922/` (inside the archive set) |
| Contents | `journal_bts-scheduler_2026-09-01_to_now.txt` (188 KB), `journal_live-forward_2026-09-01_to_now.txt` (40 MB), `config/` (`.bts-orchestrator.toml` + 9/14 snapshot, `crontab.current` + 9/14 snapshot, all `bts-*.service/.timer` units), `data/picks/**` (24 MB: pick files, `<date>/decision.json`, shadow + policy-shadow files, `account_state/*`), `data/health_state/**`, `data/validation/**` (60 MB incl. the decision-weighted live-forward roots and the research root), `data/models/*.npz` + `npz.sha256`, `deploy_history.txt` (checkout SHA `f1b07e9`, capture time, reflog of every deploy), `uv.lock.pinned`, `cron.log` |
| Manifest | `prelim-20260922.sha256`: **5,042 files, 125 MB**; manifest sha256 `55d3649d541ca73e…` |
| Restic | archive snapshot `a2781532…` (2026-09-22 10:56 ET) |
| Restore test | restored into an empty directory: **ALL 5,042 FILES MATCH** the manifest |
| Gotcha | `bts backup run` passes `--exclude "*.lock"`; the first attempt left `uv.lock` and three zero-byte runtime lock files out of the snapshot. `uv.lock` is stored as `uv.lock.pinned` (and is git-tracked at the deployed SHA anyway); the runtime locks were dropped from the staged copy. |
| Also | detached execution over SSH died twice at ~1 minute (session cleanup kills background children); the staging was completed step by step. `pgrep -f <script>` from inside an ssh command matches the ssh command itself — check for the process tree, not the pattern. |

## Not covered here (by design)
- The **final** snapshot (W0.7) after the 9/28 grab and a declared grading cutoff, with a tag-isolated restic backup and its own restore test.
- Local `streak.json` replay vs contest streak are both inside `data/picks/**` in the checkpoint; the ledger (W1.1) keeps them apart.
