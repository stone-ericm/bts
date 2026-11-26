# W0.7 final immutable snapshot — season 2026

**Date:** 2026-09-28 · **Plan item:** W0.7 (`docs/superpowers/plans/2026-09-14-season-wrap-plan.md`) · **Owner go-ahead:** Eric, 2026-09-28 · **Predecessor:** W0.2a preliminary checkpoint (`docs/audit/2026-09-22-w0-preservation.md`)

## Declared cutoff
**2026-09-28 08:00 ET** — the BTS Official Rules correction cutoff for the season's last round (1009, 2026-09-27); no 2026 regular-season Streak result can change after it. The post-cutoff MLB scoring check (09:46 ET) found 0 changes after the final board's 18:51:54 ET tabulation (`docs/audit/2026-09-28-scoring-change-check.json`), so `final_grab_20260927` is the final whole-field board. Unresolved items are listed in the snapshot's `README.md` rather than waited for.

## Staged copy
| Fact | Value |
|---|---|
| Location | box `data/hetzner_results/season_2026_snapshot/final-20260928/` (inside the archive backup set) |
| Staged | 2026-09-28 09:51:52→09:52:20 ET from checkout `4490aee`; `diff -rq` of every copied source against the staged copy right after the copy: no differences apart from the dropped locks |
| Contents | the W0.2a set refreshed — `config/` (orchestrator TOML + 9/14 snapshot, crontab + 9/14 snapshot, every `bts-*.service`/`.timer`), `data/picks/**`, `data/health_state/**`, `data/validation/**`, `data/models/*.npz` + `npz.sha256`, `deploy_history.txt`, `uv.lock.pinned`, `cron.log` — plus `data/leaderboard/**` (1.8 GB: daily corpus 5/01→7/04, 30-minute static captures, `final_grab_20260922`, `final_grab_20260927`) and a `README.md` (cutoff, contents, unresolved items, unavailable history, correction rules) |
| Journals | **widened to the full retained range** (earliest 2026-05-11; W0.2a kept 9/01 onward): `bts-scheduler` 1.36 MB, `bts-live-forward-capture` + `-resolve` 168 MB, `bts-leaderboard` 47 KB (5/11→7/04). Reason: journald will eventually vacuum them, and the W1.5 incident register needs 7/16, 8/11, 8/13 and 8/30 |
| Dropped | three zero-byte runtime locks (`data/picks/.scoring.lock`, `data/picks/account_state/saver_state.lock`, `data/health_state/backup_status.json.lock`) — the archive set runs `--exclude "*.lock"`; same handling as W0.2a |
| Manifest | `final-20260928.sha256`: **20,096 files, 2,077,838,460 bytes**; manifest sha256 `e3dce02475d4f17dbb46619da44139732c8a42882cf8d6336c72627c4f6af2f3`. Paths are relative to `season_2026_snapshot/` (the W0.2a manifest's are relative to the repo root) |

## Tag-isolated restic snapshot
| Fact | Value |
|---|---|
| Snapshot | **`1e88487fd6d9e86755f59cd3bfe0b38b30ea98a67cad01c71b8713b90eb59c1a`** (2026-09-28 09:53:25 ET), tags exactly `[season2026]` |
| Content | the whole `season_2026_snapshot/` directory — `prelim-20260922/` + `final-20260928/` + both manifests: 25,140 files, 2.18 GB processed, 160 MB new (6.6 MB packed; the rest deduplicated against the archive snapshots) |
| Path form | a single-path backup stores the full absolute path — `ls`/`restore` it as `/home/bts/projects/bts/data/hetzner_results/season_2026_snapshot`, unlike the ops/archive snapshots, which list `/data/...` |
| Retention | no `season2026` forget policy (plan: none before the 2027 activation review). Dry runs of both scheduled selectors, exactly as `backup.py` issues them: `forget --tag ops --keep-hourly 48 --keep-daily 30 --keep-weekly 26 --dry-run` → 1 group, keep 79, remove 0; `forget --tag archive --keep-daily 14 --keep-weekly 12 --keep-monthly 12 --dry-run` → 1 group, keep 26, remove 0. **The protected snapshot is selected by neither.** The weekly `bts backup prune` (`restic prune`) removes only unreferenced data |

## Restore test (acceptance)
| Fact | Value |
|---|---|
| Method | transient unit `bts-w07-restore-test`: `restic restore 1e88487f --target <new empty dir>` → `sha256sum -c` of both manifests inside the restored tree |
| Result | restored 29,543 files/dirs (2.032 GiB) in 15 s; restored manifest byte-identical to the staged one (sha256 `e3dce024…`); **ALL 20,096 final files MATCH; ALL 5,042 prelim files MATCH**; 25,140 files restored under `season_2026_snapshot/` |
| Cleanup | the restore-test copy was deleted after verification; the staged directory and both manifests stay on the box |

## Unavailable history (explicit; also in the snapshot README)
- `decision.json` exists from 2026-06-23 only (writer deployed 6/22).
- `feature_env_hash` exists from 2026-05-26 only; pick files before 2026-05-04 carry no provenance fields.
- User journals are retained from 2026-05-11 only.
- The daily leaderboard corpus stops at 2026-07-04; 7/05→9/21 has only the 30-minute static captures.
- The shipped reach-57 policy's own training profiles (`pooled_bins_run`) were lost with the 2026-06-05 MacBook.

## Next
P-03 (context-shadow closeout) and P-05 (tail-policy audit) read this snapshot (plan W1.4b).
