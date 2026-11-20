# Verdict

BLOCK on `46ce572d..0645be4b`.

## Blocking findings

1. **F1 — Inventory still certifies an ungated writer and a disabled monitor.** An unmarked row with a shell escape inside the command word passes the raw regex, while the shell reconstructs `reconcile`. A heartbeat behind a short-circuited OR also passes, and a gated `echo bts check-results` row counts as a real job. `scripts/install-deploy-guard-hetzner.sh:734–784`. Verify the canonical executable jobs and heartbeat, and fail closed on unreviewed command rows. I compared preflight results and used inert shell controls for the token and short-circuit semantics.
2. **F2 — A timed-out hook can stop a recovered scheduler.** After 60 seconds of transaction contention, the hook can read the old hold, then pause while the current owner clears it and the scheduler starts. It still stops the scheduler and records evidence for the old run. `scripts/install-deploy-guard-hetzner.sh:577–585`. Keep the fallback stop and matching-hold check serialized with state changes and activation. The fixture interleaved release immediately after the old record read and observed the subsequent stop.
3. **F3 — The effective unit can see a different state tree.** A loaded `RootDirectory` or data `BindPaths` override passes while presenting the expected argv, workdir and environment-file path. The scheduler's apparent `data/picks` and heartbeat paths can resolve to the restore tree while the owner checks the original. `scripts/install-deploy-guard-hetzner.sh:693–731,799–807,1313–1319`. Inspect the effective mount/root properties and require the template defaults. Both revisions accepted the shim and neither queried those properties.
4. **F5 — The canary accepts a runtime that dies before READY and rejects one that works.** `heartbeat_path="regular-file/.heartbeat"` passes, then `bts schedule` raises `FileExistsError` before READY. A working SSH tier with `platform="linux"` now fails canary, although the actual cascade uses its POSIX branch successfully. `src/bts/scheduler.py:1189–1196,1209–1217`. Check the heartbeat path using the scheduler's write semantics and share the cascade's platform interpretation. I ran the real CLI against both and the real cascade with a stand-in SSH result.
5. **F6 — ET rollover can still occur between the final clock read and unlink.** A scheduler that passed the dated canary at 23:59:59 can commit if the process is descheduled at `os.unlink` and resumes at 00:00:01. `scripts/install-deploy-guard-hetzner.sh:359–373`. Refuse rollover-adjacent activation/release and keep dated certification enforceable when admitting work. A clock change at marker fsync is contained, while an injected pause after the last sample still commits.
6. **F9 — A fresh `stalled` heartbeat can clear the hold on a live scheduler.** The scheduler emits `stalled` while its cascade is running. An earlier failed stop can leave that process alive, and hand release now accepts the fresh heartbeat and admits cron. `scripts/install-deploy-guard-hetzner.sh:609–628,1094–1097`. Verify the scheduler is inactive or failed with no MainPID before clearing, and retain the fresh heartbeat/window checks. The pinned hand-release fixture accepted `stalled` at age 10 seconds.

## Notes

- F2's matching `active`/PID/FLOCK sample still returns clear in a model where MainPID has already exited and a stale manager view and inherited-lock row both report its old PID. The Linux kernel lock accounting and manager timing for that pair need verification on the A7 systemd 252 rehearsal.
- F8's later-heartbeat path uses the bounded guard-record reader. A fresh valid heartbeat with >8 KiB of extra diagnostic data goes through ordinary liveness without a release record and gets a deploy `/fail` with one. Normal scheduler heartbeats exercised here were small.
- F11's claim is retried once. Two containment attempts timed out while a shared `state.lock` was held and neither posted. This leaves the later monitor or next preflight to surface the failure.
- F14 the guard now treats extra stdout alongside a valid classified canary line as a failed canary and discards the classification. The fixture injected stdout, and the command runner discards stderr separately.
- The F13 monitor comparison stopped after the HEAD code read and validated the custom record. My subprocess stand-in lacked `DEVNULL`. The trace establishes the custom read path, with the final ping and damaged-guard controls unverified.
- A7 still needs Debian systemd 252, its real FLOCK accounting and stop hook, Python 3.11, uv 0.11.6, power-loss durability and alert receipt. This review used process and manager stand-ins, temporary files and an offline transport guard.

## Round-1 findings

| Finding | Check re-run | `46ce572d` | `0645be4b` |
|---|---|---|---|
| Codex 1 / Carbon B3 | Quoted, module, nested and chained writer rows and commented monitor | Preflight accepted | Those inputs refuse at `crontab`. Remaining F1 cases above |
| Codex 2 / Carbon B2 | Deactivating owner with another lock holder, abandoned activation with stale target, failed UID lookup | Clear with the wrong lock holder. Stale target/UID can leave stop, page and result undone | Prior examples fence or contain. UID failure pages and records result. Late-hook race remains |
| Codex 3 | Wrong `ExecCondition` path, success status 1, alternate workdir and env file | Accepted | All refuse at `guard`. Unqueried namespace settings remain |
| Codex 4 | Legacy dashboard POST after arm at candidate reset | POST 303 and saver changed if old service left running | Owner stopped dashboard before reset. Manual pre-arm stop covers old handler before owner containment |
| Codex 5 | `cascade_budget_min="forty"`, absent `tiers`, absent DM recipient | Canary passes in all cases. Runtime raises at startup/selection or refuses delivery | Canary refuses all. Two other runtime discrepancies remain |
| Codex 6 | Real dated canary sample just before midnight, activation completes just after | Commit despite expired observation | Retains failed hold, stops and pages. Residual check/unlink timing above |
| Codex 7 | Runbook date write and readback using a configured nondefault picks tree, guarded revert requirement | Default tree changed. Configured tree refused. Protocol requirement absent | Configured tree changes and passes readback/canary. Guarded workflow requirement present |
| Carbon B1 | First scheduler sleeping heartbeat versus owner marker, then monitor | Marker after heartbeat, exit 1 and POST `/fail` | Marker before heartbeat, base success output and GET |
| Carbon B4 | Failed hold after stopped scheduler's old wake or stale running heartbeat | Refuses at `window` | Releases. Fresh running/waiting and wake in 29 seconds refuse |

## 1. Same-input replays

The table uses independent fixtures against Git archives of both revisions. The legacy dashboard handler and unguarded heartbeat monitor came from `b128854d`. No prior review scripts or logs were used as execution evidence.

## 2. Remaining inputs

- F1 and F3 are inventory and effective-unit gaps. F2 has a deterministic stop-after-release interleaving and a separate modeled stale liveness sample.
- The checked-in public example, its DM and research variants, legacy private alias and local tier pass. The canary left fixture bytes and mtimes unchanged. The Linux tier and bad heartbeat parent break parity in opposite directions.
- Dated canary decisions flip at the July and year-end ET boundaries. HEAD contains rollover during activation and release-marker publication and contains loss of the dated observation on the same day. The last clock/unlink interval still admits a commit.
- No-record F10 matches the legacy monitor with the guard directory absent or initialized and empty, including stdout, stderr, exit, ping method/suffix and churn bytes. F8 matches for a fresh first heartbeat and later stale heartbeat and pages on equal timestamps.
- F9 permits old/missing stopped-box heartbeats and a wake in 31 seconds. It refuses current running/waiting and wake in 29 seconds. The fresh stalled case above needs a live-unit check.
- F12's valid pending hold with a modeled owner returned 1 for the scheduler and 75 for each writer on both revisions. For F15, `46ce572d` refused after a failed Tailscale lookup. HEAD proceeded to the expected loopback HTTP probe through the stand-in.

## 3. Behavior outside Amendment 3

- The new canary output parser introduces a stdout framing requirement. F5 also enforces the validator's platform vocabulary on existing scheduler configurations.
- The unit verification installer now checks both scheduler and dashboard effective settings. Its constraints on the dashboard are broader than the scheduler-specific F3 wording.
- `require` and stop helpers now preserve interruption classification. This changes a TERM at a bounded child call from the current step refusal to `interrupted` and leaves the recovery path in control.

## 4. Operator text

- `CLAUDE.md:118–119`, A7 and B8c in `docs/ops/2027-season-start.md:16,33` stage the target installer and unit templates outside the checkout, under root control for the guard. The manual stop and inactive check fence the old saver endpoint. A pre-arm refusal leaves that manually stopped dashboard down until recovery. Schedule the stop inside the already verified window after checking deploy prerequisites.
- A7/B8c's reinstall-on-hash-change, reload, stale-unit and stale-fingerprint text tells the operator why to restage/recheck before shipping. Use the staged absolute script paths. Running A7's new `uv run --no-sync` test requires a separate synced target runtime before a first deploy from an older checkout. The text gives no staging procedure or working directory for that runtime and its relative picks paths. Add the isolated test procedure so no checkout or live state changes are needed.
- A7's systemd/unit ownership, completion-hook, uv, 3.11 and distinct-ping rehearsal sentences call for isolated checks and notification receipts. B8c preserves R's offline first-boot condition, the guarded-cron check and the prohibition on using the October checkout to initialize 2027. They authorize no live pick or entry.
- `INCIDENT.md:97–103` first reads state in the intended synced checkout, verifies prior-slate settlement, sets today's dated observation in the configured tree, checks it again and retries only in the verified window. An uncertain delivery or entry keeps the hold. Both documented CLI commands were exercised on fixture state.
- `INCIDENT.md:115–118,125–126,181–184` makes backfill wait for both grading and reconcile to clear, routes dashboard recovery through the verified window, and requires compatible guarded ship/dispatch/revert targets serving today's state. Put the new compatibility rule before the shell examples at `:146–179`, where a hurried operator encounters the push commands.
- `DEPLOY NOT STARTED` now carries the resolved prior SHA when available. The abandoned wording refers to completed containment, but stop can be unverified. Read the stop evidence before retrying. `DEPLOY FAILED at stop-dashboard` requires incident recovery without reset. `Activation failed after certification; the hold was kept` directs a hold and delivery check. Qualify the first sentence when the adjacent hold evidence is unverified.
- The classified canary line identifies ET date and readiness/refusal. Scheduler stop and `Failure ping received HTTP 2xx` communicate containment and HTTP receipt. The installer’s effective-unit or reload failure calls for checking/staging the loaded units before retry. None authorizes a manual send or entry.
- The monitor suppresses a young matching live hold, pages an abandoned/custom hold, and uses ordinary liveness output for a newer heartbeat. A hand-release success opens cron and boot admission and the runbook requires a renewed window check before starting; add an explicit stopped-unit check for F9.

## 5. Mutation check

Skipped at the lead's direction after the round ended in a BLOCK. The round-1 mutation set and 16 Amendment 3 variants were prepared without measuring which tests catch them.

## What I ran

I ran `git diff 46ce572d HEAD`, `git diff --check`, and the standalone local comparisons after 18:10 ET on both revisions (`comparison-46ce572d.log` and `comparison-0645be4b.log`). I did not run pytest or the mutation step. The tracked worktree and index were clean.
