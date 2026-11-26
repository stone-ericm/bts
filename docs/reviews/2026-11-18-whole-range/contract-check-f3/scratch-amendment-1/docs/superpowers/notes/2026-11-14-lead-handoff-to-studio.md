# BTS lead handoff: MacBook to Mac Studio (written 2026-11-14 at 22:42 ET by the outgoing lead; brought up to date at 23:05 ET)

## Why you exist
Eric, 2026-11-14 night: "the studio is staying until january 7th. eor has already been told. we're still going to get the mini on the 18th. so with that in mind, let's move everything over to the studio". You are the BTS lead session, now on the studio. The previous lead ran on the MacBook (session name `bts-70`) and stops leading once the two items under "In flight" are handed to you.

Read, in this order: `CLAUDE.md` (work policies, safety rules, the 2027 season section); this note; `docs/reviews/2026-11-14-season-rollover/FRESH-START.md`; `docs/reviews/2026-11-14-season-rollover/design/RULING-R24.md` and `RULING-R25.md`; the roadmap's rulings ledger (`docs/superpowers/plans/2026-11-12-2027-season-roadmap.md`, R18 to R25).

## What is being built
The open blocker for Opening Night (2027-03-24) is which streak and saver the pick decision uses when last season's files are on the box. History: three rules by the lead failed review (R23, the lead writes no fourth); a design pass followed (Gemini Carbon designed, Codex attacked, the lead ruled: R24; Carbon wrote a build specification, Codex checked it and said NOT YET for the larger change sets, the lead ruled: R25). Everything is under `docs/reviews/2026-11-14-season-rollover/design/`.

Decisions by Eric that bind the build:
- An ordinary in-season day with a stale contest reading KEEPS PICKING on the last accepted streak, with a DM ("i prefer keep picking"). Scope as the lead read it: in-season only.
- Opening stretch before the switch to automatic: a streak is supplied for each contest date, else no pick and an alert. This is the build's default; he has been told and has not objected or confirmed.
- General rule, all projects: no headless game runs on a Mac where a game is being played with Game Mode on; if run there anyway, Game Mode off first (shared memory `feedback_game_mode_off_for_same_machine_engine_runs.md`).

## In flight at handoff (state at 23:05 ET, 2026-11-14)
1. **Change set 1** (defensive reads, model-state isolation). Contract: `docs/reviews/2026-11-14-season-rollover/build/kickoff-cs1.md`. Handover: `build/handover-cs1.md` (author marker `CS1A-a29311ee`). Branch `sb-cs1` is on origin at `7cff54d5`: the author's commit `32b6a74b` plus one correction by the outgoing lead after reading the diff (an empty or unparseable `recorded_at` no longer counts as a date; the kickoff's wording had allowed it; test first). Not merged.
   - **Gate:** running on this machine since 22:59 ET in the worktree `~/projects/bts-cs1` (detached at `7cff54d5`) through the yield wrapper; output `~/bts-runs/gate.out.cs1-7cff54d5`, watch log `~/bts-runs/gate-cs1.log.watch`. Green means the failing set equals `~/bts-runs/gate-failed.txt`. The suite is now about 4,950 tests; the "32 seconds" in `CLAUDE.md` is out of date.
   - **Yours:** read the diff yourself (`git diff main...origin/sb-cs1`); when the gate is green, the tier A review the same day it is taken up (one fresh Codex reviewer at the highest effort with memories off, and Gemini Carbon in parallel, each in an isolated copy cloned from a bundle, briefed to assume the author is wrong, to run their own probes and to look for false greens); at most two rounds; reproduce every Gemini finding before acting; adopt the reviewers' probes as tests; then merge and push `origin main`.
2. **Build specification, version 3** is committed (`design/design-v3.md`, Carbon, marker `SRD5-54eff2d2`). Its check is running on this machine: a Codex reviewer named `srd6-codex` in herdr pane `w2:p2` (terminal `term_66048997df66b4f`, session `01a1dd56-b375-7650-9e22-b2459b1b40a8`), working in the isolated copy `~/projects/bts-speccheck` from the brief `design/brief-codex-spec-check-v3.md`. It will write `~/projects/bts-speccheck/design-v3-check-codex.md` and print `SRD6-<first 8 hex of its sha256>` then `DONE`.
   - **Yours:** watch it (compare that identity before every key or prompt; read the pane before answering any dialog; approve only read-only probes inside its copy); when it finishes, check the marker against the file, copy the report verbatim to `design/design-v3-check-codex.md`, commit, close the agent and its tab, and log the stop in `bts-agents.txt`. Then rule. Change sets 2 and 3 (the `streak.json` season tag; the saver mapping and warnings) may be commissioned once their sections survive the check; change sets 4 to 6 only after the check passes. Each goes to a fresh author who finds and runs every test touching the modules it changes, with a contract written as a before-and-after table against the real code (the CS1 kickoff shows the form, and its row 4 shows how a loose word becomes a wrong test). If this second check also fails, stop and report to Eric instead of starting a third round.

The outgoing lead logged that reviewer's start in `bts-agents.txt`. It counts as one of BTS's three agents.

Before acting on either item, check `git log origin/main`, `git branch -r` and `herdr agent list` so you do not start a duplicate.

## Other open BTS work (none started tonight)
- The scheduler's extra attempts on branch `sr-lastcheck` (one open Carbon finding, and the planned-check reuse needs its own ruling); the blocked branch `sr-state` is reference only, do not merge.
- Carbon's audit findings not yet covered: F4 to F8 (`docs/audit/2026-11-14-season-rollover-audit-carbon.md`).
- The longest-streak trade table (R19), the day-one captures (R20), the study registrations draft, the roadmap checkbox tidy-up, a backup check of Mac-only folders.
- The daily multi-model workflow review with a delivery estimate is due under the work policies.
- Waiting on Eric: the EoR driver's question about a two-review Carbon trial; the E107 Healthchecks outcome; eight unacknowledged messages in the alerts mailbox (untouched); any overrule of R18 to R25; confirmation of the opening-stretch default.

## Operating on the studio
- The studio is EoR's run host and its driver (`eor-driver-studio`, herdr pane `w1:p2`) lives here. Never touch its panes, tabs or workspace. Use your own herdr workspace. Its terms for BTS batch jobs so far: one job at a time at nice 19 through `~/bts-runs/yield_run.sh`, paused while a `victoria3` process or `~/.cache/eor-studio/bts-hold` exists (never write there), BTS under 10 GiB, no new job under 15 GiB free. Its terms for BTS's interactive agents are in `~/bts-runs/notes-for-eor-driver/2026-11-15-eor-driver-answer-interactive-agents.md`: read that file before staffing anything. In short: at most three BTS agents at a time; test runs through the yield wrapper; BTS's combined memory under about 16 GiB and no new job while free plus inactive memory is under 24 GiB; nothing under `/Volumes/EORScratch`, `~/.cache/eor-studio`, `~/Documents/Paradox Interactive/` or the EoR repository; **no `herdr agent prompt`, `send-keys` or `read` aimed at an EoR pane** (talk to the driver through files in that notes folder); never stop, restart or upgrade the shared herdr server, and close only BTS's own tabs and workspace (`w2`); kill only by your own recorded pids; no per-file or per-second pollers; and append one line to `~/bts-runs/notes-for-eor-driver/bts-agents.txt` each time you start or stop an agent (time, count, model), because BTS's Codex agents draw on the same weekly Codex ration as EoR's lanes.
- Gate: the fast suite command is in `CLAUDE.md`. Off the MacBook a known set of 53 tests fails for environment reasons (`~/bts-runs/gate-failed.txt`); green means the failing set is exactly that list.
- Agents: Codex with `--no-daemon -c memories.use_memories=false -c memories.generate_memories=false`, reviewers also `-c model_reasoning_effort=xhigh`, Remote Control off. Gemini: `agy --model gemini-4-carbon-high`; before each use open `/config` and confirm Enable Telemetry off, Use AI Credits off, Non-Workspace Access off, Sandbox Mode on. These were verified on the MacBook only; this machine's settings may differ, so read them before the first prompt.
- Pushing works through a deploy key (section below).
- Shared memory (`~/projects/claude-shared`) was diverged on this machine until tonight; it was merged and pushed at about 22:38 ET and is in sync.
- Allowances at 21:00 ET on 11/14: Claude main 20% of the week left, resetting in about a day and a half; Codex 51%; Gemini 86%. Lean on Codex and Gemini for the heavy passes. Check `ai-usage` before staffing.
- The MacBook keeps one thing: a passive watcher of its herdr server's scheduling priority (`~/projects/mac-bg-priority/`, runs until 2026-11-28). It is not BTS work; leave it.

## Push access (done 2026-11-14 23:00 ET)
Eric chose a key file for this repository only. The EoR driver created it on the studio (`~/.ssh/bts_studio_deploy_ed25519`, with its own pinned known-hosts file) and pointed `~/projects/bts` at it (`origin` = `git@github.com:stone-ericm/bts.git`, `core.sshCommand` in the repository's local config only); the outgoing lead added the public half to GitHub from the MacBook as deploy key 169928125, "studio deploy key (until 2027-01-07)", read-write. A dry-run push from the BTS workspace returned "Everything up-to-date". The driver's record: `~/bts-runs/notes-for-eor-driver/2026-11-15-eor-driver-bts-deploy-key-set.md`. Worktrees of `~/projects/bts` share that config; a separate clone does not and cannot push.

The key can push to any branch. A push to `deploy` starts the production deploy workflow: that stays owner-only, as `CLAUDE.md` says. `gh` is installed on the studio but its sign-in sits in a keychain that herdr panes cannot read, so `gh` does not work for you; use git.

## How the lead session on the studio is started (2026-11-14 23:12 ET)
In herdr workspace `w2`, pane `w2:p3`, from `~/projects/bts`:
`claude --model claude-fable-5-2 --permission-mode auto --remote-control bts-lead-studio --add-dir /Users/eric/bts-runs /Users/eric/projects/bts-work /Users/eric/projects/bts-cs1 /Users/eric/projects/bts-speccheck /Users/eric/projects/claude-shared` (add `--resume <session id>` to continue a conversation; the first one is `a0a09172-5155-4c5d-9355-28277749576d`).

The extra directories are there because the session otherwise stops to ask before every read outside the project folder. `~/projects/bts-work` is for new worktrees and isolated copies, so they are covered without widening the session to all of `~/projects`, which also holds an EoR checkout. The `fable` alias on this machine's Claude Code resolves to an older version, so the model is named in full.

## First actions
1. `git fetch origin && git log --oneline -8 origin/main`, check out `main`, sync the environment (`UV_CACHE_DIR=/tmp/uv-cache uv sync --extra model`), run the gate once to confirm the baseline on this commit.
2. Read the files listed at the top.
3. Take up the two in-flight items as their pushes arrive.
4. Tell Eric in plain words what you have picked up and what you will do first.
