# Evidence brief: the BTS lead's workflow, 2026-11-17 03:00 to 2026-11-18 21:40 ET

You are asked for an independent opinion on a WORKFLOW, not on code. Facts are below; the lead's own diagnosis is deliberately left out. Use only this file (and, if you have file access, the read-only paths it names).

## Setting
- One person (Eric) owns a baseball-streak contest bot ("BTS") that must be ready for a season opening 2027-03-24 (a labor lockout may move it). The production box is deleted for the offseason and is restored 21 days before a confirmed opener. Nothing can be deployed until one whole-range review by a reviewer who has seen none of the earlier rounds has been done. Eric alone decides: spending, deploys, anything on the real contest account, policy changes, cutting scope.
- A lead Claude session coordinates on a shared Mac Studio (also another project's run host: at most 3 BTS agents, one test job at a time, the fast test suite takes 26 to 29 minutes at low priority and is the commit gate).
- Process for a pick-path ("tier A") change: the lead writes a contract (a table of required behaviour) -> a fresh Codex agent checks the contract against the real code BEFORE anyone builds -> a Codex author builds test-first in a worktree -> the gate -> two fresh adversarial reviewers in isolated copies (Codex at highest effort, whose verdict is the gate, and Gemini Carbon) -> at most two rounds -> the reviewers' probes are adopted as tests -> merge, push. Three BLOCKs on one change set stop it: obligations are written down and a fresh session starts. Instrumentation ("tier B") gets tests, the gate and one daily batch review.
- The lead approves each agent's sandbox-escape dialogs by reading them (commits, unsandboxed suite runs).

## Agent log (every start and stop, real clock times)
2026-11-17 03:03 ET  start  cs4b-check (Codex, GPT-6.1-Sol xhigh, contract check for change set 4b, pane w2:pQ; Codex weekly 25% before)  BTS agents running: 2  [bts-lead]
2026-11-17 03:28 ET  stop   cs4b-check (finished, marker CS4BC-825d0be0, NOT YET; pane w2:pQ)  BTS agents running: 1  [bts-lead]
2026-11-17 03:28 ET  start  cs4b1-check (Codex, GPT-6.1-Sol xhigh, contract check for change set 4b-1, pane w2:pR)  BTS agents running: 2  [bts-lead]
2026-11-17 03:56 ET  stop   cs4b1-check (finished, marker CS4B1C-3528edd8, BUILD WITH THESE EDITS)  BTS agents running: 1  [bts-lead]
2026-11-17 03:57 ET  start  cs4b1-author (Codex, GPT-6.1-Sol, author of change set 4b-1, pane w2:pS, worktree ~/projects/bts-work/cs4b1; Codex weekly 25% before)  BTS agents running: 2  [bts-lead]
2026-11-17 05:17 ET  stop   cs4b1-author (handover CS4B1-58079f13, commit 28ae3aa9)  BTS agents running: 1  [bts-lead]
2026-11-17 05:18 ET  start  cs4b1-rev-codex (Codex, GPT-6.1-Sol xhigh, adversarial review round 1 of change set 4b-1, pane w2:pT; Codex weekly 24% before)  BTS agents running: 2  [bts-lead]
2026-11-17 05:19 ET  start  cs4b1-rev-carbon (Gemini 4 Carbon high via agy, sandboxed, same review, pane w2:pV; Gemini weekly 97% and 5-hour 100% before)  BTS agents running: 3  [bts-lead]
2026-11-17 05:37 ET  note   about 05:22 to 05:26: two BTS test jobs overlapped for a few minutes (a Codex reviewer's sandboxed first attempt had started its child before the wrapper's nice call failed, and the approved retry then ran beside it). Same cause as 
2026-11-17 05:43 ET  stop   cs4b1-rev-codex (review done, marker CS4B1RC-6450d271, SIGN)  BTS agents running: 2  [bts-lead]
2026-11-17 06:06 ET  stop   cs4b1-rev-carbon (review done, marker CS4B1RG-231ef231, BLOCK on two status-display cases; Gemini weekly 97% left)  BTS agents running: 1  [bts-lead]
2026-11-17 06:08 ET  start  cs4b1-r2 (Codex, GPT-6.1-Sol xhigh, round 2 of 2 on the lead's two fixes to change set 4b-1, pane w2:pW)  BTS agents running: 2  [bts-lead]
2026-11-17 06:20 ET  stop   cs4b1-r2 (round 2 done, marker CS4B1R2-adcb6bfb, SIGN)  BTS agents running: 1  [bts-lead]
2026-11-17 06:47 ET  start  cs4b2-check (Codex, GPT-6.1-Sol xhigh, contract check for change set 4b-2, pane w2:pX; Codex weekly 24% before)  BTS agents running: 2  [bts-lead]
2026-11-17 07:13 ET  stop   cs4b2-check (finished, marker CS4B2C-37166b5c, BUILD WITH THESE EDITS)  BTS agents running: 1  [bts-lead]
2026-11-17 07:15 ET  start  cs4b2-author (Codex, GPT-6.1-Sol, author of change set 4b-2, pane w2:pY, worktree ~/projects/bts-work/cs4b2; Codex weekly 24% before)  BTS agents running: 2  [bts-lead]
2026-11-17 08:37 ET  stop   cs4b2-author (handover CS4B2-4cdb8f49, commit 8494b6e6)  BTS agents running: 1  [bts-lead]
2026-11-17 08:39 ET  start  cs4b2-rev-codex (Codex, GPT-6.1-Sol xhigh, adversarial review round 1 of change set 4b-2, pane w2:pZ; Codex weekly 24% before)  BTS agents running: 2  [bts-lead]
2026-11-17 08:40 ET  start  cs4b2-rev-carbon (Gemini 4 Carbon high via agy, sandboxed, same review, pane w2:p0; Gemini weekly 97% and 5-hour 98% before)  BTS agents running: 3  [bts-lead]
2026-11-17 08:59 ET  stop   cs4b2-rev-codex (review done, marker CS4B2RC-9a4631ea, SIGN)  BTS agents running: 2  [bts-lead]
2026-11-17 09:48 ET  stop   cs4b2-rev-carbon (review done, marker CS4B2RG-5dbacad5, SIGN; Gemini weekly 94% left)  BTS agents running: 1  [bts-lead]
2026-11-17 12:44 ET  note   BTS idle for the herdr server restart: 0 agents besides bts-lead, 0 wrapper jobs; resume note ~/bts-runs/bts-lead-resume-note-2026-11-17.md  [bts-lead]
2026-11-17 12:52 ET  note   bts-lead resumed after the herdr server restart (pane w2:p3, new terminal term_6607c76590d5147); watcher and yield wrapper checked; 0 other BTS agents  [bts-lead]
2026-11-17 12:54 ET  start  cs6a-check (Codex, GPT-6.1-Sol xhigh, contract check for change set 6a, pane w2:p11; Codex weekly 24% before)  BTS agents running: 2  [bts-lead]
2026-11-17 13:07 ET  stop   cs6a-check (finished, marker CS6AC-75abeff3, BUILD WITH THESE EDITS)  BTS agents running: 1  [bts-lead]
2026-11-17 13:08 ET  start  cs6a-author (Codex, author of change set 6a, pane w2:p12, worktree ~/projects/bts-work/cs6a)  BTS agents running: 2  [bts-lead]
2026-11-17 14:14 ET  stop   cs6a-author (handover CS6A-f3768c8a, commit cad7c5ec)  BTS agents running: 1  [bts-lead]
2026-11-17 14:16 ET  start  cs6a-rev-codex (Codex xhigh, adversarial review round 1 of change set 6a, pane w2:p13; Codex weekly 23% before)  BTS agents running: 2  [bts-lead]
2026-11-17 14:16 ET  start  cs6a-rev-carbon (Gemini 4 Carbon high via agy, sandboxed, same review, pane w2:p14; Gemini weekly 94% and 5-hour 97% before)  BTS agents running: 3  [bts-lead]
2026-11-17 14:29 ET  stop   cs6a-rev-codex (review done, marker CS6ARC-f5ab9bf0, BLOCK on one dry-run date ordering case)  BTS agents running: 2  [bts-lead]
2026-11-17 15:12 ET  stop   cs6a-rev-carbon (review done, marker CS6ARG-9d5dc10f, SIGN; Gemini weekly 93% left)  BTS agents running: 1  [bts-lead]
2026-11-17 15:14 ET  start  cs6a-r2 (Codex xhigh, round 2 of 2 on the lead's fix to change set 6a, pane w2:p15)  BTS agents running: 2  [bts-lead]
2026-11-17 15:37 ET  stop   cs6a-r2 (round 2 done, marker CS6AR2-ae8c39f4, SIGN)  BTS agents running: 1  [bts-lead]
2026-11-17 15:53 ET  start  cs6a-r3 (Codex xhigh, confirmation round on one change to change set 6a made at the owner's request, pane w2:p16)  BTS agents running: 2  [bts-lead]
2026-11-17 16:01 ET  stop   cs6a-r3 (confirmation round done, marker CS6AR3-d5012db1, SIGN)  BTS agents running: 1  [bts-lead]
2026-11-17 21:38 ET  start  cs5a-check (Codex xhigh, contract check for change set 5a, pane w2:p17; Codex weekly 22% before)  BTS agents running: 2  [bts-lead]
2026-11-17 21:56 ET  stop   cs5a-check (finished, marker CS5AC-14861524, BUILD WITH THESE EDITS)  BTS agents running: 1  [bts-lead]
2026-11-17 21:57 ET  start  cs5a-check2 (Codex xhigh, second contract check for change set 5a, pane w2:p18)  BTS agents running: 2  [bts-lead]
2026-11-17 22:12 ET  stop   cs5a-check2 (finished, marker CS5AD-665d731b, BUILD WITH THESE EDITS)  BTS agents running: 1  [bts-lead]
2026-11-17 22:13 ET  start  cs5a-author (Codex, author of change set 5a, pane w2:p19, worktree ~/projects/bts-work/cs5a; Codex weekly 21% before)  BTS agents running: 2  [bts-lead]
2026-11-17 23:27 ET  stop   cs5a-author (handover CS5A-4a4cc6f3, commit 30956cdd)  BTS agents running: 1  [bts-lead]
2026-11-17 23:28 ET  start  cs5a-rev-codex (Codex xhigh, adversarial review round 1 of change set 5a, pane w2:p1A; Codex weekly 20% before)  BTS agents running: 2  [bts-lead]
2026-11-17 23:29 ET  start  cs5a-rev-carbon (Gemini 4 Carbon high via agy, sandboxed, same review, pane w2:p1B; Gemini weekly 99% and 5-hour 97% before)  BTS agents running: 3  [bts-lead]
2026-11-17 23:46 ET  stop   cs5a-rev-codex (report CS5ARC-611793a9, SIGN WITH EDITS: one test to add)  BTS agents running: 2  [bts-lead]
2026-11-18 00:11 ET  stop   cs5a-rev-carbon (report CS5ARG-4492ef03, SIGN; note: it ran its last pytest probes directly, not through the yield wrapper; no other BTS job was running)  BTS agents running: 1  [bts-lead]
2026-11-18 00:42 ET  start  cs5b-check (Codex xhigh, check of the 5b contract draft 1, pane w2:p1C; Codex weekly 20% before, resets 01:36)  BTS agents running: 2  [bts-lead]
2026-11-18 00:58 ET  stop   cs5b-check (report CS5BC-5d864fdb, NOT YET: a restart is not a safe check on demand)  BTS agents running: 1  [bts-lead]
2026-11-18 01:05 ET  start  cs4c-check (Codex xhigh, check of the 4c contract draft 1, pane w2:p1D; Codex weekly 20% before, resets 01:36)  BTS agents running: 2  [bts-lead]
2026-11-18 01:24 ET  stop   cs4c-check (report CS4CC-a2b02462, BUILD WITH THESE EDITS)  BTS agents running: 1  [bts-lead]
2026-11-18 01:26 ET  start  cs4c-author (Codex, author of change set 4c, worktree bts-work/cs4c, pane w2:p1E; Codex weekly 20% before, resets 01:36)  BTS agents running: 2  [bts-lead]
2026-11-18 02:34 ET  stop   cs4c-author (handover CS4C-8fca29ff, commit 9fc2ee6d)  BTS agents running: 1  [bts-lead]
2026-11-18 02:36 ET  start  cs4c-rev-codex (Codex xhigh, adversarial review round 1 of change set 4c, pane w2:p1F; Codex weekly 100% before)  BTS agents running: 2  [bts-lead]
2026-11-18 02:37 ET  start  cs4c-rev-carbon (Gemini 4 Carbon high via agy, sandboxed, same review, pane w2:p1G; Gemini weekly 97% and 5-hour 90% before)  BTS agents running: 3  [bts-lead]
2026-11-18 03:48 ET  stop   cs4c-rev-codex (report CS4CRC-da00ddcb, BLOCK: two findings) and cs4c-rev-carbon (report CS4CRG-0855ab2a, SIGN)  BTS agents running: 1  [bts-lead]
2026-11-18 03:51 ET  start  cs4c-r2 (Codex xhigh, round 2 of the 4c review, pane w2:p1H)  BTS agents running: 2  [bts-lead]
2026-11-18 04:26 ET  stop   cs4c-r2 (report CS4CR2-31771096, SIGN)  BTS agents running: 1  [bts-lead]
2026-11-18 04:30 ET  start  cs6b1-check (Codex xhigh, check of the 6b-1 contract draft 1, pane w2:p1J)  BTS agents running: 2  [bts-lead]
2026-11-18 04:48 ET  stop   cs6b1-check (report CS6B1C-93be1de9, BUILD WITH THESE EDITS)  BTS agents running: 1  [bts-lead]
2026-11-18 04:48 ET  start  cs6b1-author (Codex, author of change set 6b-1, worktree bts-work/cs6b1, pane w2:p1K)  BTS agents running: 2  [bts-lead]
2026-11-18 05:22 ET  stop   cs6b1-author (handover CS6B1-9916b749, commit 0e0c49a3)  BTS agents running: 1  [bts-lead]
2026-11-18 05:23 ET  start  cs6b1-rev-codex (Codex xhigh, adversarial review round 1 of change set 6b-1, pane w2:p1M)  BTS agents running: 2  [bts-lead]
2026-11-18 05:24 ET  start  cs6b1-rev-carbon (Gemini 4 Carbon high via agy, sandboxed, same review, pane w2:p1N; Gemini weekly 97% and 5-hour 98% before)  BTS agents running: 3  [bts-lead]
2026-11-18 06:15 ET  stop   cs6b1-rev-codex (report CS6B1RC-c402c1cc, BLOCK: three findings) and cs6b1-rev-carbon (report CS6B1RG-cd3a9de9, BLOCK: one finding, the same as Codex's second)  BTS agents running: 1  [bts-lead]
2026-11-18 06:18 ET  start  cs6b1-r2 (Codex xhigh, round 2 of the 6b-1 review, pane w2:p1P)  BTS agents running: 2  [bts-lead]
2026-11-18 06:54 ET  stop   cs6b1-r2 (report CS6B1R2-d66d4fac, SIGN)  BTS agents running: 1  [bts-lead]
2026-11-18 07:03 ET  start  cs6b2-check (Codex xhigh, check of the 6b-2 contract draft 1, pane w2:p1Q)  BTS agents running: 2  [bts-lead]
2026-11-18 07:19 ET  stop   cs6b2-check (report CS6B2C-83c3d2db, NOT YET: reading a malformed decision record as absent is unsafe at several callers)  BTS agents running: 1  [bts-lead]
2026-11-18 07:19 ET  start  cs6b2-author (Codex, author of change set 6b-2 version 2, worktree bts-work/cs6b2, pane w2:p1R)  BTS agents running: 2  [bts-lead]
2026-11-18 07:59 ET  stop   cs6b2-author (handover CS6B2-cd9defaa, commit 38d707bd)  BTS agents running: 1  [bts-lead]
2026-11-18 08:01 ET  start  cs6b2-rev-codex (Codex xhigh, adversarial review round 1 of change set 6b-2, pane w2:p1S)  BTS agents running: 2  [bts-lead]
2026-11-18 08:01 ET  start  cs6b2-rev-carbon (Gemini 4 Carbon high via agy, sandboxed, same review, pane w2:p1T; Gemini weekly 96% and 5-hour 100% before)  BTS agents running: 3  [bts-lead]
2026-11-18 08:19 ET  stop   cs6b2-rev-codex (report CS6B2RC-799d467e, BLOCK: two findings)  BTS agents running: 2  [bts-lead]
2026-11-18 08:46 ET  stop   cs6b2-rev-carbon (report CS6B2RG-b6d8554d, SIGN)  BTS agents running: 1  [bts-lead]
2026-11-18 08:50 ET  start  cs6b2-r2 (Codex xhigh, round 2 of the 6b-2 review, pane w2:p1V)  BTS agents running: 2  [bts-lead]
2026-11-18 09:26 ET  stop   cs6b2-r2 (no report: Codex's content filter ended its turn twice; its test run had finished clean)  BTS agents running: 1  [bts-lead]
2026-11-18 09:26 ET  start  cs6b2-r2b (Codex xhigh, round 2 of the 6b-2 review, second attempt, pane w2:p1W)  BTS agents running: 2  [bts-lead]
2026-11-18 09:33 ET  stop   cs6b2-r2b (no report: Codex's content filter ended its turn again after its mutation run)  BTS agents running: 1  [bts-lead]
2026-11-18 09:34 ET  start  cs6b2-r2c (Gemini 4 Carbon high via agy, sandboxed, round 2 of the 6b-2 review under ruling R38, pane w2:p1X; Gemini weekly 95% and 5-hour 98% before)  BTS agents running: 2  [bts-lead]
2026-11-18 10:16 ET  stop   cs6b2-r2c (report CS6B2R2G-d3450eb8, SIGN)  BTS agents running: 1  [bts-lead]
2026-11-18 10:20 ET  start  audit-f4f8 (Codex, author of the audit findings F4 to F8, tier B, worktree bts-work/audit-f4f8, pane w2:p1Y)  BTS agents running: 2  [bts-lead]
2026-11-18 11:29 ET  stop   audit-f4f8 (handover AUDITF-49135eb5, head 153ff8c2, seven commits)  BTS agents running: 1  [bts-lead]
2026-11-18 11:29 ET  start  batch-1118 (Codex xhigh, the day's adversarial batch review of the audit findings F4 to F8, pane w2:p1Z)  BTS agents running: 2  [bts-lead]
2026-11-18 11:48 ET  stop   batch-1118 (report BATCH1118-4dee6264, BLOCK: five findings in F4, F5, F7, F8; F6 clean)  BTS agents running: 1  [bts-lead]
2026-11-18 11:48 ET  start  audit-fixes (Codex, author of the fixes for the batch review's findings, worktree bts-work/audit-f4f8, pane w2:p10)  BTS agents running: 2  [bts-lead]
2026-11-18 12:24 ET  stop   audit-fixes (handover AUDITFX-068ac311, head 8cac32ea)  BTS agents running: 1  [bts-lead]
2026-11-18 12:25 ET  start  confirm-1118 (Codex xhigh, confirmation review of the audit fixes, pane w2:p21)  BTS agents running: 2  [bts-lead]
2026-11-18 12:40 ET  stop   confirm-1118 (report CONFIRM1118-eff6c9e1, BLOCK: three findings in F4, F5, F7)  BTS agents running: 1  [bts-lead]
2026-11-18 13:05 ET  start  confirm2-1118 (Codex xhigh, second confirmation review of the narrowed audit change set, pane w2:p22)  BTS agents running: 2  [bts-lead]
2026-11-18 13:16 ET  stop   confirm2-1118 (report CONFIRM2-174c3058, BLOCK: two findings; third BLOCK on the audit change set, which is stopped)  BTS agents running: 0  [bts-lead]
2026-11-18 13:46 ET  start  bts-audit-fresh (Claude, fresh session for the audit change set after three BLOCKs, Remote Control off, pane w2:p23; it may run ONE Codex agent of its own at a time)  BTS agents running: 2  [bts-lead]
2026-11-18 14:16 ET  start  audit-ready-rev (Codex xhigh, adversarial review of sb-audit-ready 58c0c2bd: F6, F8, P-03 pin; pane w2:p24)  BTS agents running: 3  [bts-audit-fresh]
2026-11-18 14:28 ET  stop   audit-ready-rev (report AUDITREADY-abb701f3, BLOCK: three findings, all in F8; F6, the P-03 pin and the pick path clean)  BTS agents running: 2  [bts-audit-fresh]
2026-11-18 14:28 ET  start  audit-f7-check (Codex xhigh, contract check of the F7 thin-capture contract, pane w2:p25)  BTS agents running: 3  [bts-audit-fresh]
2026-11-18 14:37 ET  stop   audit-f7-check (report F7CHECK-82436ed3, NOT YET: the receipt design needs issuance, publication and provenance changes; reviewer advises leaving the capture as on main)  BTS agents running: 2  [bts-audit-fresh]
2026-11-18 14:37 ET  start  audit-f5-check (Codex xhigh, contract check of the F5 producer contract, pane w2:p26)  BTS agents running: 3  [bts-audit-fresh]
2026-11-18 14:46 ET  stop   audit-f5-check (report F5CHECK-422a098c, NOT YET: reviewer advises keeping the producer as on main with checklist row B7)  BTS agents running: 2  [bts-audit-fresh]
2026-11-18 14:50 ET  stop   bts-audit-fresh (report delivered: part A review BLOCK on F8; both part B contract checks NOT YET; its records carried to main)  BTS agents running: 1  [bts-lead]
2026-11-18 14:51 ET  start  proptest (Codex, one-lane property-testing trial on a discovery branch that never merges, worktree bts-work/proptest, pane w2:p27; Eric's request relayed 11/18)  BTS agents running: 2  [bts-lead]
2026-11-18 14:53 ET  note   the proptest lane ran short sandboxed test runs through the wrapper while the lead's gate on audit-f6 was running (two test jobs briefly at once, 14:50 to about 15:10); the lane does not need approval for sandboxed wrapper runs  [bt
2026-11-18 15:05 ET  stop   proptest (report PROPTEST-4764d0de: 9 minutes; one confirmed violation, opp_bullpen_hr_30g)  BTS agents running: 1  [bts-lead]
2026-11-18 15:19 ET  start  bullpen-check (Codex xhigh, check of the bullpen-feature contract, pane w2:p28)  BTS agents running: 2  [bts-lead]
2026-11-18 15:32 ET  stop   bullpen-check (report BULLPENC-2df1e761, BUILD WITH THESE EDITS)  BTS agents running: 1  [bts-lead]
2026-11-18 15:33 ET  start  bullpen-author (Codex, author of the bullpen-feature fix, worktree bts-work/bullpen-fix, pane w2:p29)  BTS agents running: 2  [bts-lead]
2026-11-18 16:08 ET  stop   bullpen-author (handover BULLPEN-9c165618, commit 1135c217)  BTS agents running: 1  [bts-lead]
2026-11-18 16:10 ET  start  bullpen-rev-codex (Codex xhigh, adversarial review round 1 of the bullpen-feature fix, pane w2:p2A)  BTS agents running: 2  [bts-lead]
2026-11-18 16:11 ET  start  bullpen-rev-carbon (Gemini 4 Carbon high via agy, sandboxed, same review, pane w2:p2B; Gemini weekly 93% and 5-hour 98% before)  BTS agents running: 3  [bts-lead]
2026-11-18 16:23 ET  stop   bullpen-rev-codex (report BULLPENRC-a95616ed, SIGN)  BTS agents running: 2  [bts-lead]
2026-11-18 17:06 ET  stop   bullpen-rev-carbon (report BULLPENRG-5bd15fab, SIGN)  BTS agents running: 1  [bts-lead]
2026-11-18 20:52 ET  start  proptest2 (Codex, property-testing lane 2 on a discovery branch, pane w2:p2C) and leads-check (Codex xhigh, read-and-reproduce check of five suspected defects, pane w2:p2D)  BTS agents running: 2  [bts-lead]
2026-11-18 21:02 ET  stop   leads-check (report LEADS-51c6d9c6)  BTS agents running: 1  [bts-lead]
2026-11-18 21:05 ET  stop   proptest2 (report PROPTEST2-cae67e5a)  BTS agents running: 0  [bts-lead]
2026-11-18 21:36 ET  start  leads-rev-codex (Codex xhigh, pane w2:p2E) and leads-rev-carbon (Gemini 4 Carbon high via agy, sandboxed, pane w2:p2F): adversarial review round 1 of the two small fixes  BTS agents running: 2  [bts-lead]

## Landings and fixes on main (commit time, subject)
11-18 21:35 f7238991 docs(review): gate counts for the small-fixes review brief
11-18 21:10 dd5423d5 docs(review): round-1 brief for the two small fixes from the leads check; the skip-policy checkpoint instability is ledgered, not built
11-18 17:35 df454725 docs: the bullpen-feature fix landed (both round-1 reviews archived; CLAUDE.md note with what it does not fix)
11-18 17:35 812e11f6 Merge the bullpen-feature fix: opp_bullpen_hr_30g no longer depends on the same day
11-18 16:08 51465fa1 docs(review): bullpen-feature fix handed over (BULLPEN-9c165618, commit 1135c217 on fix/bullpen-same-day) and the round-1 review brief
11-18 15:17 e54c049e Merge audit finding F6: the research-mode contest fetch cron sends no DM
11-18 12:40 401ce66a docs(review): confirmation review of the audit fixes: BLOCK (F4, F5, F7); ruling R39 (the two alarm exceptions are withdrawn, not repaired); the brief for a second, narrowed confi
11-18 12:24 f2dfa9ce docs(review): fixes for the batch review's findings handed over (AUDITFX-068ac311, head 8cac32ea) and the confirmation review brief
11-18 11:48 dd664b0b docs(review): the day's batch review of the audit findings: BLOCK (F4, F5, F7 twice, F8; F6 clean); kickoff for the fixes (the F7 retry is removed, not repaired)
11-18 10:58 1ac91821 fix(deploy): the leaderboard service DMs stonehengee.bsky.social (Eric 2026-11-18; audit finding F6)
11-18 10:17 9ec9d905 Merge change set 6b-2: a malformed alert status file cannot stop the scheduler
11-18 08:49 0d7d4843 docs(review): round 1 of the 6b-2 review (Codex BLOCK on two findings, Carbon SIGN), contract amendment 1, and the round-2 brief for the fix (af15e511)
11-18 08:49 af15e511 fix(cs6b2): round-1 review: a FIFO swapped in between the type check and the open cannot block the status read or write
11-18 06:54 8fc8bdc1 Merge change set 6b-1: the morning state check
11-18 06:53 288d143c test(cs6b1): round-2 note: the morning fixture restores the policy cache it found
11-18 06:18 0df39dea docs(review): round 1 of the 6b-1 review (Codex BLOCK on three findings, Carbon BLOCK on one of them), contract amendment 1, and the round-2 brief for the fix (7efc3cea)
11-18 06:18 7efc3cea fix(cs6b1): round-1 review: an interrupt leaves the morning check, a malformed record is nothing found, a failed send does not reprint the refusal
11-18 04:26 90000157 Merge change set 4c: a pick delivered on a stale automatic reading is recorded and warned about
11-18 03:51 759670e1 docs(review): round 1 of the 4c review (Codex BLOCK on two findings, Carbon SIGN), contract amendment 2, and the round-2 brief for the fix (687f9d55)
11-18 03:51 687f9d55 fix(cs4c): round-1 review: interrupts inside the stale-reading work cannot stop or interrupt a delivery; no warning without the transport's identifier
11-18 00:38 d990c5f0 Merge change set 5a: a saved pick is delivered only on the state it was chosen on
11-17 16:54 58132b1c Merge change set 6a of the season-boundary build: the configuration cannot leave the live refusal off
11-17 15:53 9c13f7b0 fix: bts schedule resolves its default date before loading the configuration in every mode, as on the base
11-17 15:12 87c118f6 fix: a dry run of bts schedule resolves its default date before loading the configuration; adopt the reviewers' tests
11-17 10:13 0ade4db0 Merge change set 4b-2 of the season-boundary build: the live refusal and revoke
11-17 06:44 3c7eac7b Merge change set 4b-1 of the season-boundary build: the season switch and what the fetch writes
11-17 06:06 861c8aac fix: season-contract status: a ledger row that cannot be rendered is a bad line; a boolean saver season is invalid; adopt the reviewers' tests
11-17 03:00 a3430194 Merge change set 4a of the season-boundary build: a contest streak supplied for today

## Outcomes by change set in this window
| Change set | Contract check | Review rounds | Result |
|---|---|---|---|
| 4a dated contest observation | (earlier) | round 1 both SIGN | merged 11-17 03:00 |
| 4b-1 season switch | version 2 | r1 Codex SIGN, Carbon BLOCK (2 display cases); r2 SIGN | merged 06:44 |
| 4b-2 live refusal | version 2 | r1 both SIGN | merged 10:13 |
| 6a configuration gate | version 2 + 2 amendments | r1 Codex BLOCK (dry-run date order), Carbon SIGN; r2 SIGN; a third confirmation round after the owner asked for a different fix | merged 16:54 |
| 5a saved-pick check | two contract checks, version 3 | r1 Codex SIGN WITH EDITS, Carbon SIGN | merged 11-18 00:38 |
| 5b re-pick on demand | check: NOT YET (the design was unsafe) | none | design withdrawn; owner later accepted the residual risk |
| 4c stale-reading warning | BUILD WITH EDITS; the author stopped once on a conflict between two table rows | r1 Codex BLOCK (2), Carbon SIGN; r2 SIGN | merged 04:26 |
| 6b-1 morning state check | BUILD WITH EDITS | r1 Codex BLOCK (3), Carbon BLOCK (1 of them); r2 SIGN | merged 06:54 |
| 6b-2 alert status file | draft 1 NOT YET (half the design withdrawn) | r1 Codex BLOCK (a FIFO race), Carbon SIGN; r2: Codex's content filter ended two sessions with no report; Carbon SIGN accepted as the gate under a recorded ruling | merged 10:17 |
| Audit findings F4 to F8 (tier B, five unrelated fixes in ONE change set, NO contract check) | none | batch review BLOCK (5 findings), fix, confirmation BLOCK (3), narrowed, second confirmation BLOCK (2): stopped after three BLOCKs. A fresh session then got a fourth BLOCK on one piece and two NOT YET contract checks | only one of the five pieces (F6, a cron flag) merged; the other four left as they were, with the false alarms they were meant to silence documented as expected |
| Property-testing trial (one Codex lane, 9 minutes) | n/a | n/a | found a real leak in a production model feature |
| Bullpen feature fix (from that trial) | BUILD WITH EDITS (the check also built two candidate implementations and measured them) | r1 both SIGN, notes closed with tests | merged 17:35 (2 h 16 min from finding to merge) |
| Property-testing lane 2 and a five-lead defect check (two Codex lanes, about 10 min each) | n/a | n/a | one finding needing the owner (a policy artifact's late-season bins), two small fixes now in review, several ledgered |

## Other facts
- Every author commit and every unsandboxed suite run needed the lead to read and approve a dialog; the property lane asked five times in ten minutes to `git add`/`git commit` files in its own scratch directory.
- Twice a sandboxed agent's short test run overlapped the lead's gate run (the one-job rule), because agents can run wrapped jobs without approval.
- The lead's first verification of the merged F6 change with an ad-hoc harness printed nothing (wrong directory layout); it then relied on the adopted test that does the same check.
- The lead approved an author's change that hardcoded 2027 dates; a reviewer caught it.
- The lead's own data-only rule for one alarm (F4) was too loose; reviewers showed real failures it would have hidden.
- Reviewer disagreement in round 1 (Codex vs Carbon) on 6 tier-A change sets: 4b-1 (Carbon found, Codex did not), 6a, 4c, 6b-2 (Codex found, Carbon did not), 6b-1 (both), 5a/4b-2/4a/bullpen (neither).
- Allowances at 21:40 on 11-18: Codex weekly 92% left (it reset at 01:36 that morning after reaching 20%), Gemini weekly 93%, lead's Claude account weekly 57%.
- The lead worked through the night of 11-17/18 without a pause; the owner checked in three times in the window.
- Not yet done: the whole-range pre-deploy review (brief written 11-18 21:20; about 2,300 changed lines in src since the last deployed commit); measuring the bullpen fix's effect on backtest metrics (a multi-hour run); owner decisions on the policy-bin finding.

## Questions
1. Where is this workflow flawed? Rank by expected cost (hours lost, or risk of a wrong pick or a missed day in season).
2. What are the false greens: steps that look like assurance and are not?
3. What is wasted motion? What would you cut or batch?
4. What one or two changes would you make tomorrow, concretely?
5. A delivery estimate: given this pace and the remaining list (whole-range review and its fixes; owner decisions; restore and season-start checklist 21 days before the opener), is "deployable code by 2026-12-01" realistic, and what is the critical path? Give a date range with your reasoning.
Answer in under 70 lines, plain language, most important first.
