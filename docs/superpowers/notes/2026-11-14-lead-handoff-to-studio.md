# BTS lead handoff: MacBook to Mac Studio (written 2026-11-14 about 23:00 ET by the outgoing lead)

## Why you exist
Eric, 2026-11-14 night: "the studio is staying until january 7th. eor has already been told. we're still going to get the mini on the 18th. so with that in mind, let's move everything over to the studio". You are the BTS lead session, now on the studio. The previous lead ran on the MacBook (session name `bts-70`) and stops leading once the two items under "In flight" are handed to you.

Read, in this order: `CLAUDE.md` (work policies, safety rules, the 2027 season section); this note; `docs/reviews/2026-11-14-season-rollover/FRESH-START.md`; `docs/reviews/2026-11-14-season-rollover/design/RULING-R24.md` and `RULING-R25.md`; the roadmap's rulings ledger (`docs/superpowers/plans/2026-11-12-2027-season-roadmap.md`, R18 to R25).

## What is being built
The open blocker for Opening Night (2027-03-24) is which streak and saver the pick decision uses when last season's files are on the box. History: three rules by the lead failed review (R23, the lead writes no fourth); a design pass followed (Gemini Carbon designed, Codex attacked, the lead ruled: R24; Carbon wrote a build specification, Codex checked it and said NOT YET for the larger change sets, the lead ruled: R25). Everything is under `docs/reviews/2026-11-14-season-rollover/design/`.

Decisions by Eric that bind the build:
- An ordinary in-season day with a stale contest reading KEEPS PICKING on the last accepted streak, with a DM ("i prefer keep picking"). Scope as the lead read it: in-season only.
- Opening stretch before the switch to automatic: a streak is supplied for each contest date, else no pick and an alert. This is the build's default; he has been told and has not objected or confirmed.
- General rule, all projects: no headless game runs on a Mac where a game is being played with Game Mode on; if run there anyway, Game Mode off first (shared memory `feedback_game_mode_off_for_same_machine_engine_runs.md`).

## In flight at handoff (both started on the MacBook; the outgoing lead finishes each and pushes the result)
1. **Change set 1** (defensive reads, model-state isolation). Contract: `docs/reviews/2026-11-14-season-rollover/build/kickoff-cs1.md`. A Codex author is working in a MacBook worktree on branch `sb-cs1`. When it hands over, the outgoing lead pushes `sb-cs1` to origin and commits its handover as `docs/reviews/2026-11-14-season-rollover/build/handover-cs1.md`. Yours from there: read the diff yourself first; the gate on the studio; tier A review the same day it is taken up (one fresh Codex reviewer at the highest effort with memories off, and Gemini Carbon in parallel, each in an isolated copy cloned from a bundle, briefed to assume the author is wrong, to run their own probes and to look for false greens); at most two rounds; reproduce every Gemini finding before acting; adopt the reviewers' probes as tests; then merge and push `origin main`.
2. **Build specification, version 3.** Carbon is revising it under R25 (`design/brief-carbon-v3.md`). When it lands the outgoing lead commits it as `design/design-v3.md`. Yours from there: one more adversarial check by a fresh Codex reviewer (adapt `design/brief-codex-spec-check.md`: same questions, against version 3 and R25), then rule. Change sets 2 and 3 (the `streak.json` season tag; the saver mapping and warnings) may be commissioned once their sections survive the check; change sets 4 to 6 only after the check passes. Each goes to a fresh author who finds and runs every test touching the modules it changes, with a contract written as a before-and-after table against the real code (see the CS1 kickoff for the form). If the second check also fails, stop and report to Eric instead of starting a third round.

If either item is still unfinished when you start, check `git log origin/main` and `git branch -r` for the pushes described above before doing anything about it, and do not start a duplicate.

## Other open BTS work (none started tonight)
- The scheduler's extra attempts on branch `sr-lastcheck` (one open Carbon finding, and the planned-check reuse needs its own ruling); the blocked branch `sr-state` is reference only, do not merge.
- Carbon's audit findings not yet covered: F4 to F8 (`docs/audit/2026-11-14-season-rollover-audit-carbon.md`).
- The longest-streak trade table (R19), the day-one captures (R20), the study registrations draft, the roadmap checkbox tidy-up, a backup check of Mac-only folders.
- The daily multi-model workflow review with a delivery estimate is due under the work policies.
- Waiting on Eric: the EoR driver's question about a two-review Carbon trial; the E107 Healthchecks outcome; eight unacknowledged messages in the alerts mailbox (untouched); any overrule of R18 to R25; confirmation of the opening-stretch default.

## Operating on the studio
- The studio is EoR's run host and its driver (`eor-driver-studio`, herdr pane `w1:p2`) lives here. Never touch its panes, tabs or workspace. Use your own herdr workspace. Its terms for BTS batch jobs so far: one job at a time at nice 19 through `~/bts-runs/yield_run.sh`, paused while a `victoria3` process or `~/.cache/eor-studio/bts-hold` exists (never write there), BTS under 10 GiB, no new job under 15 GiB free. The outgoing lead told the driver that BTS's lead and agents are moving here and asked for its terms for interactive agents; look for its answer in `~/bts-runs/notes-for-eor-driver/` and in its pane before staffing more than two agents at once.
- Gate: the fast suite command is in `CLAUDE.md`. Off the MacBook a known set of 53 tests fails for environment reasons (`~/bts-runs/gate-failed.txt`); green means the failing set is exactly that list.
- Agents: Codex with `--no-daemon -c memories.use_memories=false -c memories.generate_memories=false`, reviewers also `-c model_reasoning_effort=xhigh`, Remote Control off. Gemini: `agy --model gemini-4-carbon-high`; before each use open `/config` and confirm Enable Telemetry off, Use AI Credits off, Non-Workspace Access off, Sandbox Mode on. These were verified on the MacBook only; this machine's settings may differ, so read them before the first prompt.
- `gh` is not installed on the studio; git works over ssh.
- Shared memory (`~/projects/claude-shared`) was diverged on this machine until tonight; it was merged and pushed at about 22:50 ET and is in sync.
- Allowances at 21:00 ET on 11/14: Claude main 20% of the week left, resetting in about a day and a half; Codex 51%; Gemini 86%. Lean on Codex and Gemini for the heavy passes. Check `ai-usage` before staffing.
- The MacBook keeps one thing: a passive watcher of its herdr server's scheduling priority (`~/projects/mac-bg-priority/`, runs until 2026-11-28). It is not BTS work; leave it.

## First actions
1. `git fetch origin && git log --oneline -8 origin/main`, check out `main`, sync the environment (`UV_CACHE_DIR=/tmp/uv-cache uv sync --extra model`), run the gate once to confirm the baseline on this commit.
2. Read the files listed at the top.
3. Take up the two in-flight items as their pushes arrive.
4. Tell Eric in plain words what you have picked up and what you will do first.
