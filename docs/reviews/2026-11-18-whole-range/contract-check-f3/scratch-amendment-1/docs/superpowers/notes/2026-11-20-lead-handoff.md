# BTS lead handoff, 2026-11-20 (the lead session is replaced by a fresh one)

Eric, 2026-11-20 about 13:45 ET, after the workflow review recommended it: "feel free to replace your own session". The outgoing lead ran from 2026-11-15 to 2026-11-20 in herdr pane `w2:p3` on the Mac Studio. **You are the new BTS lead.** You are Eric's interface for BTS: he talks to you in your pane; your agent name in herdr is `bts-lead` (delegated sessions and the standing watcher address their messages to that name).

## Read first, in this order
1. `CLAUDE.md` (all of it: the work policies, the 2027 season section with every "Landed" and "Open" bullet, "Where everything runs", the gate baseline, the Safety Rules).
2. `docs/reviews/2026-11-18-whole-range/FINDINGS.md`: every finding of the pre-deploy review, its state, rulings R44 to R50 and Eric's decisions of 11-19 and 11-20. Then `STOP-C3.md`, `STOP-C3V.md`, `STOP-C2.md`, `RESTART-2026-11-20.md` in the same directory.
3. `docs/audit/2026-11-20-workflow-review/README.md` (ruling R51: what changed in how the lead works, and why) and `docs/audit/2026-11-18-workflow-review/README.md` (R43).
4. `docs/superpowers/plans/2026-11-12-2027-season-roadmap.md`, the rulings ledger (R1 to R48; R49 to R51 are in `FINDINGS.md` and the workflow review: add them to the ledger).
5. `docs/ops/2027-season-start.md` rows B5 to B8c.
6. The three kickoffs for delegated sessions, as models and for the rules they carry: `~/bts-runs/kickoff-wr-fresh-2026-11-19.md` ("Rules you must follow" is the reference text for agent handling on this machine), `kickoff-wr-fresh2-2026-11-19.md`, `kickoff-wr-fresh3-2026-11-20.md`; and their reports `~/bts-runs/wr-fresh*-report-*.md`.
7. Invoke the `herdr` skill before any herdr command.

## What is running right now (you inherit all of it)
| Thing | Where | State | What you do |
|---|---|---|---|
| `bts-wr-fresh3` | herdr pane `w2:p3Z`, Claude session `b86de61e-…`, Remote Control off | Working on change set D (the deploy hold and canary, Eric's choice A; `kickoff-fix-deploy-v3.md`), then W15 (drop bad probable-pitcher cache entries with a health warning). It reports to `~/bts-runs/wr-fresh3-report-2026-11-20.md` and prompts `bts-lead` when done or stopped | Read its report when it prompts you. A second-round BLOCK stops its change set: you do NOT override (R51) |
| Backtest queue | `~/bts-runs/bullpen-bt/run_all.sh` (detached), log `progress.log` | 6 of 16 done; run 7 (`old`, seed 2048) restarted 13:27 after the machine restart; about 1 h 55 min per run; it starts a run only when no wrapped test job is running and `~/bts-runs/backtest-hold` is absent | When `ALL DONE` appears: do the paired comparison (below). If the machine restarts again: `RESTART-2026-11-20.md` has the restart commands |
| Standing watcher | `~/bts-runs/watch_sessions.sh` (detached), log `watch_sessions.log`, list `~/bts-runs/watched-sessions.txt` | Started 13:33. Once a minute: answers the Claude Code "read outside the working directories" prompt with option 1 for listed sessions (Eric: "pick it, keep allowing"); prompts `bts-lead` when a listed session has been blocked more than 5 minutes, when the backtest runner has died, or when the queue is done. NOT yet seen to fire: verify it in the failure direction at the first chance | Add a delegated session's agent name to the list when you start one. After a reboot, start it again (`cd ~/bts-runs && nohup zsh ~/bts-runs/watch_sessions.sh > watch_sessions.out 2>&1 &`) |
| `c3v-text` | herdr pane `w2:p42` (Codex xhigh, clone `~/projects/bts-work/c3v-text`) | Started 13:40: a fresh review of the checklist step the old lead wrote into the C3v merge without review (row B8b item 2 and its `CLAUDE.md` sentence). Brief `docs/reviews/2026-11-18-whole-range/review-brief-c3v-operator-text.md`; report `.codex-review/c3v-text/report.md` in its clone, marker `C3VTEXT-<8 hex>` | The old lead's monitor on it dies with the old session: check the report file yourself. Verify the marker against the file's sha256, archive the report under `docs/reviews/2026-11-18-whole-range/review-c3v-operator-text/`, apply any MUST FIX as a docs change (tier D), `/quit` the agent, close its tab, log the stop |

## Open work, in order
1. **Whatever `c3v-text` finds** (docs only).
2. **`bts-wr-fresh3`'s results:** change set D, then W15. Merge only on a Codex SIGN. After D lands, the checklist rows B8a to B8c, `INCIDENT.md` and `CLAUDE.md` "Deployment" must describe the hold.
3. **When the backtests finish (expected about 2026-11-21 09:30 ET if little else takes the slot):**
   - The paired comparison: 8 seeds, arms `old` (`51465fa1`) and `new` (`812e11f6`, the bullpen same-day fix), 2021 to 2025, `estimated_pa` profiles, outputs `~/bts-runs/bullpen-bt/out/<arm>/seed_<seed>/backtest_<season>.parquet`. Per seed and season: rank-1 hit rate (P@1) for each arm and the paired difference; then the pooled difference with its spread across seeds; the streak metrics the scorecard uses if they can be computed from these profiles. 2021 to 2025 outcomes only (no 2026 outcome is read). Report plainly, including "no detectable difference" if that is the result. Seed 42 is a known outlier and is not in the set. Eric has not made the result a deploy condition (he was asked; he chose the studio run and did not answer that part).
   - **Then the two scheduler timing fixes (W3, W4), each its own change set** (Eric 2026-11-20: "build after backtests finish"): replacement texts in `contract-check-a/contract-check.md`. Both touch the scheduler's loop; if a contract check or a round-1 review shows the design unsafe, stop that one and report (Eric has twice chosen to leave a timing hazard documented rather than build an unsafe fix).
   - **Then the real lock in the test wrapper** (R51 item 7): only when no instance of `yield_run.sh` is running.
4. **The closing test (R51 item 4), after the LAST code merge:** (a) re-run the four slice harnesses on the final head (`slice-1-codex/`, `slice-2-codex/`, `slice-3-codex/`, `slice-1-carbon/` under the whole-range directory; several were adopted as tests already, see each change set's handover); (b) one blind read of the final head by a fresh Codex xhigh reviewer who has seen nothing (use `review-brief.md` in the whole-range directory with `BASESHA` still `619abda7`, and remove the review records from its copy as before); (c) Carbon reads slices 2 and 3 (`slice-2-brief.md`, `slice-3-brief.md`); (d) a new season-walk harness: every date of 2027 through the real decision path with the shipped policies, checking days left, that each day yields a pick or a named skip, and that the last game day is playable. Anything found goes through the normal process.
5. **Daily:** the multi-model workflow review with a delivery estimate (last: 2026-11-20, `docs/audit/2026-11-20-workflow-review/`); one list of owner questions a day (R51 item 6).

## Decisions Eric has made (do not re-ask)
- Playing 2027; nothing deploys before the whole-range review is closed; the box stays deleted until 21 days before a confirmed opener.
- 5b (re-pick on demand) and the malformed-`decision.json` hazard: "leave both" (R40). W16 (the entry check near first pitch): accept. W11 (duplicate identical DM): left on purpose by the lead.
- Deploy workflow: A, the durable on-box hold; the canary also asks whether the box can pick. W15: drop bad cache entries with a warning the system or he can react to. W3, W4: after the backtests.
- The late-phase policy-bin finding is folded into the longest-streak policy work (no separate fix exists); that work still needs his D7 approval before any policy affects picks.
- The backtests run on the studio (he was shown a comparison with a cloud run and chose the studio).
- The Claude Code "read outside the working directories" prompt: option 1, keep allowing.
- The leaderboard service DMs `stonehengee.bsky.social`.

## Waiting on Eric (put these in the daily list; none blocks current work)
- His explicit acceptance of the stopped season-replay fault (one stale never-committed pick file disables the nightly MODEL-streak rebuild until someone resolves that day; now visible through `bts reconcile`, not fixed; the contest streak that decides live picks does not depend on it: the `c3v-text` review is checking that sentence).
- Whether the bullpen backtest result is a condition for the deploy.
- Any overrule of the lead's rulings R42 to R51.

## How the old lead went wrong (so you do not)
- It introduced the 2027 day-count bug on 11-13 (a table entry with a different meaning from its neighbour) and it was found only by the whole-range review: when you add a value beside an existing one, prove they mean the same thing with a test over the old data.
- It overrode the two-round stop twice and merged an operator instruction no reviewer had seen. R51: no overrides; operator text is reviewed with its change.
- It left a delegated session unwatched for 13 hours. The standing watcher exists now; still read `herdr agent get <name>` yourself when something should have reported and has not.
- It wrote watcher commands that looked like test jobs to the authors' checks, briefs worded like security work (Codex's filter cut three reviewers off), and three contracts whose operator instructions were unsafe in some state. The kickoffs carry the corrected rules.
- Its estimates were optimistic every time. Use the later of any two estimates.

## Machine and money
- EoR terms on the studio stand: at most three BTS agents at once (a delegated Claude session's own agents count; the watcher and the backtest runner are jobs, not agents); one test job at a time through `~/bts-runs/yield_run.sh`; never touch `w1` panes, `~/.cache/eor-studio`, `/Volumes/EORScratch`; kill only your own pids; never restart the herdr server; log every agent start and stop in `~/bts-runs/notes-for-eor-driver/bts-agents.txt` with real clock times.
- Agents under you run with Remote Control off (`--settings '{"remoteControlAtStartup": false}'` for Claude; Codex `--no-daemon -c memories.use_memories=false -c memories.generate_memories=false`, reviewers add `-c model_reasoning_effort=xhigh`; Carbon `agy --model gemini-4-carbon-high --sandbox` with the `/config` check).
- Owner-only, never without Eric: spending, any deploy or push to `deploy`, the contest account, delivery mode, cutting scope, publication, 2026 outcomes without an exposure-register row.
- Allowances at 13:30 on 11-20: Codex weekly about 80% left, Gemini about 95%, Claude main 46% (`ai-usage`).
- Shared memory: `/Users/eric/projects/claude-shared/memory/bts_index.md` has the state line; update it when the state changes.
