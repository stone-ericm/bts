# 2027 season-start checklist (season wrap deliverable)

**Status:** first version, 2026-10-03. Items are added as the wrap proceeds. It is finalized when the wrap closes (season-wrap plan, "Deliverables").

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
| B2 | **I-113 R2 tail-policy backup**: hotfix `3577e8b` (shipped 2026-10, if the deploy completed). Confirm the 2027 R2 manifest still lists `models/mdp_tail_policy.npz`. | Claude | manifest listing |
| B3 | **`BTS_LGBM_DETERMINISTIC`**: decide whether to flip it, only after a deliberate re-baseline cycle (CLAUDE.md). | Eric decides; Claude measures | paired-seed memo |

## C. Added by Eric after W1.5 (2026-10-03)
| # | Item | Executor | Acceptance evidence |
|---|---|---|---|
| C1 | **Hazard: no private-mode test patches `bts.posting.post_to_bluesky`.** Add a private-mode test that patches the transport and asserts no post, before any delivery-mode change or mutant run. | Claude | the test, red without the patch guard |
| C2 | **Hazard: a legacy `shadow_mode = true`-only config is silently ignored and would post publicly.** Make the config loader refuse or warn on the legacy key, with a test. | Claude | the test; the loader change |
| C3 | **Unrun current-defence certificates:** <ul><li>I-043 link 5: a faithful coarse-status-fallback restoration with a positive wrong-lock return;</li><li>I-084 link 1: a positive wrong-deferral-return planner spec.</li></ul> Both go through the frozen W1.5 runner, which needs a new tooling review if the tooling changes. | Claude | accepted runs plus reviewer decisions; register rebuild |
| C4 | **I-071 recertification** (links 1 and 2): a killing closure that fixes `time.monotonic()` elapsed time as well as the wall clock, so the decision is inside ruling 13; then fresh accepted runs. | Claude | accepted runs; register rebuild |
| C5 | **Sweep GitHub issues and PRs** (all states, paginated): source R3 of the incident-register design. W1.5 Phase 1 did not sweep it. | Claude | sweep table; new candidates disposed |

## D. Carried from the wrap (to be completed)
- **E107:** record the outcome of Eric's Healthchecks step (`docs/audit/2026-10-03-e107-healthchecks-ping-rotation.md`).
- **Phase 2 of the incident register** (Route R over box data) only if the decision memo needs it: register row X-20 and Eric's go-ahead.
- The decision memo's approved and deferred changes (D1, D2, D4, D6, D7) are added here when it is put to Eric.
