# 2027 season-start checklist (season wrap deliverable)

**Status:** FINAL for the season wrap, 2026-10-04 (first version 2026-10-03). Delivering it closes the wrap. The decision memo `docs/audit/2026-10-04-2027-decisions.md` (FROZEN) leaves D1, D2, D4, D6 and D7 to Eric; approved or deferred changes, and any selected study's conditional prerequisites in §D, are added here when he records those decisions.

**Rule:** delivering this checklist closes the wrap. **Activating 2027 is a separate, explicitly approved event** (D6). The executor is named per item. "Claude" means a supervised Claude session; Eric's items need his login or decision.

## A. Required by the season-wrap plan
| # | Item | Executor | Acceptance evidence |
|---|---|---|---|
| A1 | **Artifact restore verified.** Restore models, policies and parquets from R2 into an empty directory. Hashes must match the box's live files, **including `mdp_tail_policy.npz`** (the W1.5 I-113 fix). | Claude | restore log plus a hash table |
| A2 | **Recipe/policy pairing.** The tail policy's recorded base sha equals the deployed `mdp_policy.npz` sha (sha `66d15471…`); `tail_policy` health source GREEN. Any base-policy rebuild requires `scripts/rebuild_tail_policy.py`. | Claude | `bts` health output; the two shas |
| A3 | **Current contest rules, calendar and state.** Re-fetch the 2027 official rules: §6 hit/void clauses, cutoff, double-down, saver. Diff them against `docs/audit/2026-09-29-incident-register-evidence/rules/section6.txt`, and record the season dates. | Claude, then Eric reviews | rules diff memo |
| A4 | **Authentication and entry checks.** Re-capture the cookies on the Mac (`scripts/capture_bts_cookies.py`, interactive), then verify `fetch-contest-streak` and `check-pick-entered` against the live 2027 account. | **Eric** (capture); Claude (verify) | a fresh `contest_streak.json`; a check-pick-entered dry run |
| A5 | **DM and cron configuration.** <ul><li>Restore `pick_delivery = "dm"` in `~/.bts-orchestrator.toml` (snapshot `.bak-20260914-season-over`).</li><li>Run `bash scripts/cron-setup-hetzner.sh install` (`set -a && . ./.env && set +a` first). That re-enables `check-pick-entered` and installs the C-03 07:40 reconcile line.</li><li>Restart inside a sleep window.</li></ul> | Claude, with Eric's approval at D6 | `crontab -l` diff against the 9/14 backup; the scheduler journal |
| A6 | **Rehearsal on a fixture day.** A dry run of a full day on a pinned past date: no DM, no contest call. | Claude | dry-run log; the decision.json shape |
| A7 | **Rollback.** Know the pre-activation SHA and config snapshot; check the canary and auto-rollback path. Revert = restore `pick_delivery = "private"` and comment out the cron line. | Claude | written rollback command set |
| A8 | **Explicit activation approval (D6).** | **Eric** | his recorded decision |

## B. Code and deploy items that must ship before activation
| # | Item | Executor | Acceptance evidence |
|---|---|---|---|
| B1 | **Deploy main's undeployed runtime code**, including the C-03 reconcile cutoff `ce6676d` (`src/bts/picks.py`, `src/bts/cli.py`). It needs the 07:40 cron line from A5. Use a deploy-gating review per the deploy rules. | Claude | the deploy run; the canary |
| B2 | **I-113 R2 tail-policy backup**: hotfix `3577e8b`, deployed 2026-10-03 and **verified 2026-10-04 03:07 ET**. That night's sync uploaded `models/mdp_tail_policy.npz` (sha `dc5d0c99…`; manifest 12 → 13 files); the R2 object exists and its downloaded bytes match; the restored tail validates against the restored base `66d15471`. The same verifier failed before the fix, so the check was shown in both directions. For 2027: confirm the R2 manifest still lists the tail policy. | Claude | manifest listing |
| B3 | **`BTS_LGBM_DETERMINISTIC`**: decide whether to flip it, only after a deliberate re-baseline cycle (CLAUDE.md). | Eric decides; Claude measures | paired-seed memo |

## C. Added by Eric after W1.5 (2026-10-03)
| # | Item | Executor | Acceptance evidence |
|---|---|---|---|
| C1 | **Hazard: no private-mode test patches `bts.posting.post_to_bluesky`.** Add a private-mode test that patches the transport and asserts no post, before any delivery-mode change or mutant run. | Claude | the test, red without the patch guard **Built 2026-10-06 (`04cde0a`):** `tests/test_private_mode_transport.py` patches both transports in three private configurations. Review and D7 are pending with the watchdog producers. |
| C2 | **Hazard: a legacy `shadow_mode = true`-only config is silently ignored and would post publicly.** Make the config loader refuse or warn on the legacy key, with a test. | Claude | the test; the loader change **Built 2026-10-06 (`04cde0a`):** `_pick_delivery_mode` refuses a `shadow_mode`-only config, so `run_day` fails at its first line. The box config is unaffected. Review and D7 are pending. |
| C3 | **Unrun current-defence certificates:** <ul><li>I-043 link 5: a faithful coarse-status-fallback restoration with a positive wrong-lock return;</li><li>I-084 link 1: a positive wrong-deferral-return planner spec.</li></ul> Both go through the frozen W1.5 runner, which needs a new tooling review if the tooling changes. | Claude | accepted runs plus reviewer decisions; register rebuild |
| C4 | **I-071 recertification** (links 1 and 2): a killing closure that fixes `time.monotonic()` elapsed time as well as the wall clock, so the decision is inside ruling 13; then fresh accepted runs. | Claude | accepted runs; register rebuild |
| C5 | **Sweep GitHub issues and PRs** (all states, paginated): source R3 of the incident-register design. W1.5 Phase 1 did not sweep it. | Claude | sweep table; new candidates disposed |

## D. Carried from the wrap (to be completed)
- **E107:** record the outcome of Eric's Healthchecks step (`docs/audit/2026-10-03-e107-healthchecks-ping-rotation.md`).
- **Phase 2 of the incident register** (Route R over box data) only if the decision memo needs it: register row X-20 and Eric's go-ahead.
- **Eric's rulings, 2026-10-04** (register §C; decision memo, last section):
  - **D1 = Option 2** (a better expected longest streak at a stated cost in the chance of 57). Any 2027 policy built on it needs a measured trade table approved under D7 before it affects picks. Building one is rank 4b, which Eric added to C1 on 10/04.
  - **D6:** no activation decided. Activation needs his explicit A8 approval and a named owner for entry and official-state checks.
  - **D7:** B1 is put to him after its gating review signs and before activation; B3 only after the paired-seed memo.
  - **D4: C1 APPROVED 10/04** (`docs/audit/2026-10-04-d4-cycle-proposal.md`): ranks 2, 4a, 3 and **4b**, plus the rank-1 prerequisites; tracked in `docs/sota_audit/2026-10-04-c1-cycle-index.md`. Set these up before the first 2027 capture:
    - rank 2's watchdog, reviewed and approved under D7 before activation if Eric plays;
    - rank 3's pre-lock plate-appearance count-forecast archive;
    - the rank-1 receipt log (the instrumentation item above);
    - the 4a and 3 registrations, with their 2027 fit/test splits, published as exposure rows before any 2027 outcome.
    - rank 4b: Eric's trade table (chance of 57 given up against expected longest streak gained) approved under D7 before any 4b policy affects picks; any 4b artifact must also pass the tail/base pairing check (A2).
  - The stable-user-id and witnessed-served-slate items stay conditional: their studies (cohort, consensus contexts) were not selected.
- **MLB's forecast for every player (found 2026-10-04 while building W2.3).** The static `players` sheet, captured every 30 minutes since 7/04, also carries `probabilityStarter` and `numberSelections` for all ~2,900 players, not only the ~27 most-selected per round. It has no round id, so which game a value refers to is unresolved. The frozen W2.3 design reads only the most-selected sheet. If W4 rank 1 is selected for 2027, its registration should decide, before any 2027 outcome, whether this field can be bound to a round (for example by capture time against the units' lock times) and whether to capture it at a stated cadence.
- **Instrumentation gaps named by the frozen wrap memos** (each only if the matching W4 candidate or field study is selected for 2027; set up before the first 2027 capture, since none can be repaired afterwards):
  - **Receipt log for static captures (W2.3).** Static captures keep only content-deduplicated sheets stamped at run start, so whether MLB's forecast was available before lock cannot be measured. Register a cadence and a per-fetch receipt for every attempt: fetch time, success or failure, HTTP status and a hash bound to the response bytes, including unchanged successful fetches. Receipts witness retrieval, not the provider's model-generation age.
  - **Stable user ids in the daily pick corpus (W2.2).** The daily corpus is keyed by username, so W2.2 had to assume, unverified, that a name kept one owner. Record the stable user id with each captured batch.
  - **Independently witnessed served slates (#87).** No 2026 date had an admitted ranked surface (23 lacked an independent witness, 68 a served slate). Witness each day's served slate before lock, for example by an external timestamped hash.
- **Static capture format:** the capture writer switched to `.json.gz` on 2026-07-10. Any new reader of `static_snapshots/` must list both forms (`scripts/audit/benchmark_bridge/core.py` `capture_files`); a plain `*.json` glob silently drops every capture after that date.
