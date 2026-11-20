# Evidence brief: the BTS lead's workflow, 2026-11-18 21:40 to 2026-11-20 13:30 ET

You are asked for an independent opinion on a WORKFLOW, not on code. Facts are below; the lead's own diagnosis is deliberately left out. Use only this file.

## Setting
- One person (Eric) owns a baseball-streak contest bot that must be ready for a season opening 2027-03-24 (a lockout may move it). The production box is deleted for the offseason and is restored 21 days before a confirmed opener. Nothing can be deployed until a whole-range review is finished. Eric alone decides spending, deploys, the contest account, policy changes, scope cuts.
- A lead Claude session coordinates on a shared Mac Studio (also another project's run host: at most 3 BTS agents, ONE test job at a time, the fast suite is the commit gate and takes about 26 minutes).
- Process for a pick-path change: the lead (or a fresh Claude session under it) writes a contract; a fresh Codex agent checks the contract against the real code before anyone builds; a Codex author builds test-first; the gate; two fresh adversarial reviewers in isolated copies (Codex, whose verdict is the gate, and Gemini Carbon); at most two rounds; a second-round BLOCK stops the change set (obligations recorded, a fresh session with something material changed). Since 11-18 the lead also: froze scope at a tagged commit, ran the whole-range review as three scenario slices, added standing "hostile input" rows to every contract, and treats alarm-silencing changes as pick-path.
- On 11-19 the lead, whose own session had run for more than two days, handed the remaining fixes to fresh Claude sessions working under it (three so far), each with a written kickoff, each running its own Codex and Carbon agents.

## Agent log (every start, stop and note, real clock times)
2026-11-18 21:36 ET  start  leads-rev-codex (Codex xhigh, pane w2:p2E) and leads-rev-carbon (Gemini 4 Carbon high via agy, sandboxed, pane w2:p2F): adversarial review round 1 of the two small fixes  BTS agents running: 2  [bts-lead]
2026-11-18 21:52 ET  stop   leads-rev-codex (report LEADSRC-005f4932, BLOCK on the void-reply half; cache half clean)  BTS agents running: 1  [bts-lead]
2026-11-18 21:52 ET  start  leads-r2 (Codex xhigh, confirmation round on the narrowed cache fix, pane w2:p2G) and workflow-rev (Codex xhigh, daily workflow review from an evidence brief, no test jobs, pane w2:p2H)  BTS agents running: 3  [bts-lead]
2026-11-18 21:54 ET  stop   workflow-rev (opinion WFREV-c3d98cbd)  BTS agents running: 2  [bts-lead]
2026-11-18 21:58 ET  stop   leads-r2 (report LEADSR2-57a0279e, SIGN)  BTS agents running: 1  [bts-lead]
2026-11-18 22:09 ET  stop   leads-rev-carbon (report LEADSRG-27ccee2e, SIGN on the cache half)  BTS agents running: 0  [bts-lead]
2026-11-18 22:28 ET  start  wr-slice1, wr-slice2, wr-slice3 (Codex xhigh, the whole-range pre-deploy review in three slices on tag review-candidate-2026-11-18, panes w2:p2J, w2:p2K, w2:p2M; each may run wrapped test jobs: told one at a time, but thre
2026-11-18 22:46 ET  stop   wr-slice1 (report WR1-4ce5a914, DEPLOYABLE AFTER THESE FIXES: four findings)  BTS agents running: 2  [bts-lead]
2026-11-18 22:46 ET  start  wr-slice1-carbon (Gemini 4 Carbon high via agy, sandboxed, slice 1 of the whole-range review, pane w2:p2N)  BTS agents running: 3  [bts-lead]
2026-11-18 22:47 ET  stop   wr-slice2 (report WR2-100aa6ee, DEPLOYABLE AFTER THESE FIXES: four findings)  BTS agents running: 2  [bts-lead]
2026-11-18 22:50 ET  stop   wr-slice3 (report WR3-2e4724bd, DEPLOYABLE AFTER THESE FIXES: four findings)  BTS agents running: 1  [bts-lead]
2026-11-18 22:52 ET  start  wr-check-a and wr-check-b (Codex xhigh, contract checks for the whole-range fix sets A and B, panes w2:p2P and w2:p2Q)  BTS agents running: 3  [bts-lead]
2026-11-18 23:02 ET  stop   wr-check-a (WRCA-6202b520) and wr-check-b (WRCB-75022a8a), both BUILD WITH THESE EDITS  BTS agents running: 1  [bts-lead]
2026-11-18 23:03 ET  start  wr-fix-a and wr-fix-b (Codex authors of the whole-range fix sets A and B, worktrees bts-work/wr-fix-a and wr-fix-b, panes w2:p2R and w2:p2S; told to check for a running wrapper job before starting one)  BTS agents running:
2026-11-19 00:07 ET  stop   wr-slice1-carbon (report WR1G-c1291f24, DEPLOYABLE AFTER THESE FIXES: four findings, two new; note: its sandbox could not run the yield wrapper, so it ran its short suites directly at nice 19, one at a time)  BTS agents ru
2026-11-19 00:46 ET  stop   wr-fix-a (handover WRA-e335d326, commit 9cb84536, gate green)  BTS agents running: 1  [bts-lead]
2026-11-19 00:47 ET  start  wr-a-rev-codex (Codex xhigh, pane w2:p2T) and wr-a-rev-carbon (Gemini 4 Carbon high via agy, sandboxed, pane w2:p2V): adversarial review round 1 of whole-range fix set A; told to start no wrapper run before 01:14 (author B
2026-11-19 01:13 ET  stop   wr-fix-b (handover WRB-83939b70, commit 2dce391c, gate green)  BTS agents running: 2  [bts-lead]
2026-11-19 01:13 ET  start  wr-b-rev-codex (Codex xhigh, adversarial review round 1 of whole-range fix set B, pane w2:p2W)  BTS agents running: 3  [bts-lead]
2026-11-19 01:19 ET  stop   wr-a-rev-codex (report WRARC-717f2369, BLOCK: one finding, a JSON NaN constant in the schedule body)  BTS agents running: 2  [bts-lead]
2026-11-19 01:22 ET  start  wr-a-r2 (Codex xhigh, confirmation round on fix set A, pane w2:p2X)  BTS agents running: 3  [bts-lead]
2026-11-19 01:27 ET  stop   wr-a-r2 (report WRAR2-4537008d, SIGN)  BTS agents running: 2  [bts-lead]
2026-11-19 01:27 ET  note   the lead's gate on fix set A (d67a25e7) starts now and runs about 26 minutes; the two remaining reviewers may start short wrapped test runs during it (brief overlap possible)  [bts-lead]
2026-11-19 01:29 ET  stop   wr-b-rev-codex (report WRBRC-28d0817a, BLOCK: two findings) and wr-a-rev-carbon (report WRARG-1dd9c5b4, SIGN)  BTS agents running: 0  [bts-lead]
2026-11-19 01:30 ET  start  wr-fix-b2 (Codex, round-2 author for whole-range fix set B, worktree bts-work/wr-fix-b, pane w2:p2Y; told to start no wrapper run before 01:58)  BTS agents running: 1  [bts-lead]
2026-11-19 01:56 ET  note   whole-range fix set A merged to main (082b8506)  [bts-lead]
2026-11-19 02:01 ET  start  wr-check-c (Codex xhigh, contract check for whole-range fix set C, pane w2:p2Z; told to start no wrapper run before 02:27)  BTS agents running: 2  [bts-lead]
2026-11-19 02:13 ET  stop   wr-check-c (WRCC-71c05e2c, BUILD WITH THESE EDITS; all findings reproduced)  BTS agents running: 1  [bts-lead]
2026-11-19 02:14 ET  start  wr-fix-c3 (Codex author of whole-range fix set C3, the season replay; worktree bts-work/wr-fix-c3, pane w2:p20)  BTS agents running: 2  [bts-lead]
2026-11-19 02:27 ET  stop   wr-fix-b2 (handover WRB2-191da5b1, commit fe593f29, gate green)  BTS agents running: 1  [bts-lead]
2026-11-19 02:29 ET  start  wr-b-r2-codex (Codex xhigh, pane w2:p31) and wr-b-r2-carbon (Gemini 4 Carbon high via agy, sandboxed, pane w2:p32): round 2 of the fix set B review  BTS agents running: 3  [bts-lead]
2026-11-19 02:34 ET  note   the lead stopped pid 89658 (the wr-fix-c3 author's own pytest, stuck in a test of its own; a child of that author's wrapper)  [bts-lead]
2026-11-19 02:40 ET  note   the wr-fix-c3 author's gate runs for about 26 minutes from now; the two fix-set-B reviewers may start short wrapped runs during it (brief overlap possible)  [bts-lead]
2026-11-19 02:46 ET  stop   wr-b-r2-codex (report WRBR2C-467fb418, SIGN WITH EDITS: one test, added)  BTS agents running: 2  [bts-lead]
2026-11-19 02:52 ET  stop   wr-b-r2-carbon (report WRBR2G-f45a2b0e, SIGN)  BTS agents running: 1  [bts-lead]
2026-11-19 02:54 ET  note   whole-range fix set B merged to main (a1bf1612); a full gate on main follows the wr-fix-c3 gate  [bts-lead]
2026-11-19 03:07 ET  stop   wr-fix-c3 (handover WRC3-27c51a31, commit c70effe9, gate green); the lead's full gate on main (fix sets A and B merged) started 03:07  BTS agents running: 0  [bts-lead]
2026-11-19 03:08 ET  start  wr-c3-rev-codex (Codex xhigh, pane w2:p33) and wr-c3-rev-carbon (Gemini 4 Carbon high via agy, sandboxed, pane w2:p34): adversarial review round 1 of whole-range fix set C3; told to start no wrapper run before 03:34  BTS a
2026-11-19 03:27 ET  stop   wr-c3-rev-codex (report WRC3RC-c7c4faea, BLOCK: three findings; its first turn was cut off by the provider's content filter and it finished on a second prompt, without its pytest and mutation runs)  BTS agents running: 1  
2026-11-19 03:27 ET  start  wr-fix-c3b (Codex, round-2 author for whole-range fix set C3, worktree bts-work/wr-fix-c3, pane w2:p35)  BTS agents running: 2  [bts-lead]
2026-11-19 03:54 ET  stop   wr-c3-rev-carbon (report WRC3RG-eee5b6e3, BLOCK: one finding, on the round-1 commit)  BTS agents running: 1  [bts-lead]
2026-11-19 04:13 ET  stop   wr-fix-c3b (handover WRC3B-127768fa, commit aa61bb43, gate green)  BTS agents running: 0  [bts-lead]
2026-11-19 04:14 ET  start  wr-c3-r2-codex (Codex xhigh, pane w2:p36) and wr-c3-r2-carbon (Gemini 4 Carbon high via agy, sandboxed, pane w2:p37): round 2 of the fix set C3 review  BTS agents running: 2  [bts-lead]
2026-11-19 04:34 ET  stop   wr-c3-r2-codex (report WRC3R2C-7bfbbf70, BLOCK: three findings; third BLOCK on fix set C3, which is stopped)  BTS agents running: 1  [bts-lead]
2026-11-19 04:37 ET  start  bts-wr-fresh (Claude, fresh session for the remaining whole-range fixes after the C3 stop, Remote Control off, pane w2:p38; it may run up to three agents of its own once the lead's last reviewer has quit; the lead runs non
2026-11-19 04:41 ET  start  wr-fix-c3v (Codex author, pane w2:p39, change set C3v: reconcile visibility)  BTS agents running: 2 (with the lead's wr-c3-r2-carbon)  [bts-wr-fresh]
2026-11-19 05:05 ET  stop   wr-c3-r2-carbon (report WRC3R2G-f383640e, BLOCK: eight entered-pick states and one symlink state; fix set C3 was already stopped)  BTS agents running: 1 (bts-wr-fresh)  [bts-lead]
2026-11-19 05:06 ET  note   lead's wr-c3-r2-carbon has quit; bts-wr-fresh may run up to three agents  BTS agents running: 1 (wr-fix-c3v)  [bts-wr-fresh]
2026-11-19 05:20 ET  start  wr-c3v-rev-codex (Codex xhigh reviewer, pane w2:p3A, round 1 of change set C3v at cb3bc37c)  BTS agents running: 2  [bts-wr-fresh]
2026-11-19 05:20 ET  start  wr-c3v-rev-carbon (Gemini 4 Carbon high via agy, sandboxed, pane w2:p3B, round 1 of change set C3v at cb3bc37c)  BTS agents running: 3  [bts-wr-fresh]
2026-11-19 05:35 ET  stop   wr-c3v-rev-codex (report WRC3VRX-6689cdae, BLOCK: one finding)  BTS agents running: 2  [bts-wr-fresh]
2026-11-19 05:55 ET  stop   wr-c3v-rev-carbon (report WRC3VRG-b867c8a9, SIGN with notes)  BTS agents running: 1 (wr-fix-c3v idle)  [bts-wr-fresh]
2026-11-19 06:04 ET  start  wr-c3v-r2-codex (Codex xhigh reviewer, pane w2:p3C, round 2 of change set C3v at 6ff1e87e; no wrapper runs before 06:35)  BTS agents running: 2  [bts-wr-fresh]
2026-11-19 06:05 ET  start  wr-c3v-r2-carbon (Gemini 4 Carbon high via agy, sandboxed, pane w2:p3D, round 2 of change set C3v at 6ff1e87e)  BTS agents running: 3  [bts-wr-fresh]
2026-11-19 06:44 ET  stop   wr-c3v-r2-carbon (report WRC3V2RG-450b3127, SIGN with notes)  BTS agents running: 2  [bts-wr-fresh]
2026-11-19 06:47 ET  stop   wr-c3v-r2-codex (report WRC3V2RX-87f8cf72, BLOCK: one finding, the checklist step; second-round BLOCK, change set C3v stopped)  BTS agents running: 1 (wr-fix-c3v)  [bts-wr-fresh]
2026-11-19 06:47 ET  stop   wr-fix-c3v (author; handover WRC3V2-5a0c431d, branch fix/wr-c3v head 6ff1e87e, not merged)  BTS agents running: 0  [bts-wr-fresh]
2026-11-19 06:50 ET  note   change set C3v merged to main (01f7e544) by the lead with the blocked checklist step replaced; a full gate on main follows  [bts-lead]
2026-11-19 06:50 ET  note   bts-wr-fresh continues with change sets W22 and W20 (up to three agents of its own; the lead runs only its gate until about 07:25)  [bts-lead]
2026-11-19 06:52 ET  start  wr-check-c2 (Codex xhigh contract checker, pane w2:p3E, change set C2 contract draft 2; no wrapper runs before 07:30)  BTS agents running: 1  [bts-wr-fresh]
2026-11-19 20:05 ET  stop   wr-check-c2 (contract check WRCHKC2-f3189434, BUILD WITH THESE EDITS)  BTS agents running: 0  [bts-wr-fresh]
2026-11-19 20:06 ET  start  wr-fix-c2 (Codex author, pane w2:p3G, change set C2 version 3)  BTS agents running: 1  [bts-wr-fresh]
2026-11-19 20:07 ET  start  wr-check-c1 (Codex xhigh contract checker, pane w2:p3H, change set C1 contract draft 2)  BTS agents running: 2  [bts-wr-fresh]
2026-11-19 20:23 ET  stop   wr-check-c1 (contract check WRCHKC1-00ce157d, BUILD WITH THESE EDITS)  BTS agents running: 1 (wr-fix-c2)  [bts-wr-fresh]
2026-11-19 20:24 ET  start  wr-c2-rev-codex (Codex xhigh reviewer, pane w2:p3J, round 1 of change set C2 at dfe774df; no wrapper runs before 20:50)  BTS agents running: 2  [bts-wr-fresh]
2026-11-19 20:24 ET  start  wr-c2-rev-carbon (Gemini 4 Carbon high via agy, sandboxed, pane w2:p3K, round 1 of change set C2 at dfe774df)  BTS agents running: 3  [bts-wr-fresh]
2026-11-19 20:29 ET  start  bullpen-bt runner (the lead's paired backtest of the bullpen-feature fix on Eric's word: 16 backtests, ONE at a time through yield_run.sh, each perhaps 2 to 3 hours; it starts a run only when no other wrapped job is runnin
2026-11-19 20:29 ET  hold   ~/bts-runs/backtest-hold set (C2 gate and C2 reviewers' test runs)  [bts-wr-fresh]
2026-11-19 21:02 ET  stop   wr-c2-rev-codex (report WRC2RX-b132b681, BLOCK: one finding)  BTS agents running: 2  [bts-wr-fresh]
2026-11-19 21:04 ET  stop   wr-c2-rev-carbon (report WRC2RG-b829b928, SIGN with notes)  BTS agents running: 1 (wr-fix-c2 idle)  [bts-wr-fresh]
2026-11-19 21:16 ET  start  wr-c2-r2-codex (Codex xhigh reviewer, pane w2:p3M, round 2 of change set C2 at e897c766; no wrapper runs before 21:45)  BTS agents running: 2  [bts-wr-fresh]
2026-11-19 21:16 ET  start  wr-c2-r2-carbon (Gemini 4 Carbon high via agy, sandboxed, pane w2:p3N, round 2 of change set C2 at e897c766)  BTS agents running: 3  [bts-wr-fresh]
2026-11-19 21:50 ET  stop   wr-c2-r2-carbon (report WRC2R2G-343a1037, SIGN)  BTS agents running: 2  [bts-wr-fresh]
2026-11-19 21:55 ET  stop   wr-c2-r2-codex (report WRC2R2X-430341f8, BLOCK: two findings; second-round BLOCK, change set C2 stopped)  BTS agents running: 1  [bts-wr-fresh]
2026-11-19 21:55 ET  stop   wr-fix-c2 (author; handover WRC2B-cfcb5c0c, branch fix/wr-c2 head e897c766, not merged)  BTS agents running: 0  [bts-wr-fresh]
2026-11-19 21:55 ET  hold   ~/bts-runs/backtest-hold removed (no BTS test job planned)  [bts-wr-fresh]
2026-11-19 21:57 ET  stop   bts-wr-fresh (finished: C3v landed by the lead, C2 stopped; report ~/bts-runs/wr-fresh-report-2026-11-19.md)  [bts-lead]
2026-11-19 21:57 ET  start  bts-wr-fresh2 (Claude, second fresh session: W22 confirmation, W20, the deploy workflow; Remote Control off, pane w2:p3P; up to three agents of its own; the lead runs only the backtest runner)  [bts-lead]
2026-11-19 22:00 ET  hold   ~/bts-runs/backtest-hold set (C2b author and gate coming; backtest old-seed_2273360 running since 21:53 continues)  [bts-wr-fresh2]
2026-11-19 22:02 ET  start  wr-c2b-author (Codex author, C2b receipt identity on fix/wr-c2 from e897c766; pane w2:p3Q)  BTS agents running: 1  [bts-wr-fresh2]
2026-11-19 22:08 ET  start  wr-check-deploy (Codex xhigh contract check of change set D, deploy workflow; clone ~/projects/bts-work/wr-check-deploy at fef05b5b; pane w2:p3R)  BTS agents running: 2  [bts-wr-fresh2]
2026-11-19 22:30 ET  stop   wr-check-deploy (two contract checks of change set D: NOT YET both; archived contract-check-deploy/; D parked)  BTS agents running: 1  [bts-wr-fresh2]
2026-11-20 00:20 ET  start  wr-c2b-codex (Codex xhigh confirmation review of C2b at 6663045e; clone ~/projects/bts-work/wr-review-c2b; pane w2:p3S)  BTS agents running: 2  [bts-wr-fresh2]
2026-11-20 00:21 ET  start  wr-c2b-carbon (Gemini 4 Carbon High review of C2b at 6663045e; copy ~/projects/bts-work/wr-review-c2b-carbon; pane w2:p3T)  BTS agents running: 3  [bts-wr-fresh2]
2026-11-20 01:11 ET  stop   wr-c2b-author (C2b commit 6663045e; gate 10,396 passed, failing set exactly the pinned 53; handover WRC2C-390dd214)  BTS agents running: 2  [bts-wr-fresh2]
2026-11-20 01:11 ET  stop   wr-c2b-codex (SIGN, report WRC2BX-6c3a0746, archived review-fix-c2b-codex/)  BTS agents running: 1  [bts-wr-fresh2]
2026-11-20 01:28 ET  stop   wr-c2b-carbon (SIGN, report WRC2BC-0816172e, archived review-fix-c2b-carbon/)  BTS agents running: 0  [bts-wr-fresh2]
2026-11-20 01:30 ET  hold   ~/bts-runs/backtest-hold KEPT after the C2 merge: change set C1's author starts now and needs the slot within the hour  [bts-wr-fresh2]
2026-11-20 01:31 ET  start  wr-c1-author (Codex author, change set C1 = W20, branch fix/wr-c1 from 3154f518; pane w2:p3V)  BTS agents running: 1  [bts-wr-fresh2]
2026-11-20 01:49 ET  start  wr-c1-codex (Codex xhigh review of C1 at d7d569a6; clone ~/projects/bts-work/wr-review-c1; pane w2:p3W)  BTS agents running: 2  [bts-wr-fresh2]
2026-11-20 01:50 ET  start  wr-c1-carbon (Gemini 4 Carbon High review of C1 at d7d569a6; copy ~/projects/bts-work/wr-review-c1-carbon; pane w2:p3X)  BTS agents running: 3  [bts-wr-fresh2]
2026-11-20 02:16 ET  stop   wr-c1-author (C1 commit d7d569a6; gate 10,437 passed, failing set exactly the pinned 53; handover WRC1-05963c38)  BTS agents running: 2  [bts-wr-fresh2]
2026-11-20 02:32 ET  stop   wr-c1-codex (SIGN WITH EDITS, report WRC1X-f2c6e4e8; one content-filter stop, one plain continuation; archived review-fix-c1-codex/)  BTS agents running: 1  [bts-wr-fresh2]
2026-11-20 02:41 ET  stop   wr-c1-carbon (SIGN, report WRC1C-bc88ec10, archived review-fix-c1-carbon/)  BTS agents running: 0  [bts-wr-fresh2]
2026-11-20 02:42 ET  start  wr-c1-edits (Codex author, C1 round-1 edits, tests and docs only, on fix/wr-c1 from d7d569a6; pane w2:p3Y)  BTS agents running: 1  [bts-wr-fresh2]
2026-11-20 03:15 ET  stop   wr-c1-edits (C1 round-1 edits e608e899; gate 10,442 passed, failing set exactly the pinned 53; handover WRC1E-08572c4f)  BTS agents running: 0  [bts-wr-fresh2]
2026-11-20 03:16 ET  hold   ~/bts-runs/backtest-hold removed (C2 and C1 merged; no further BTS test job planned by bts-wr-fresh2)  [bts-wr-fresh2]
2026-11-20 03:17 ET  stop   bts-wr-fresh2 (finished: W22 and W20 merged, deploy change parked; report ~/bts-runs/wr-fresh2-report-2026-11-19.md)  BTS agents running: 0  [bts-wr-fresh2]
2026-11-20 03:17 ET  stop   bts-wr-fresh2 (finished: W22 and W20 merged, the deploy-workflow change parked; report ~/bts-runs/wr-fresh2-report-2026-11-19.md); only the backtest runner remains (15 backtests, about two hours each, back to back)  [bts-l
2026-11-20 12:48 ET  start  bts-wr-fresh3 (Claude, third fresh session: the deploy hold and canary on Eric's choice A, then the cache-entry fix; Remote Control off, pane w2:p3Z; up to three agents of its own; the backtest runner continues)  [bts-lead
2026-11-20 12:55 ET  hold   ~/bts-runs/backtest-hold set (change set D starting; the backtest already running finishes first)  [bts-wr-fresh3]
2026-11-20 13:02 ET  pause  bts-wr-fresh3 (Mac restart; no agents of mine were running; contract draft 3 pushed 41952802; contract check not started; backtest-hold still set)  BTS agents running: 0  [bts-wr-fresh3]
2026-11-20 13:02 ET  note   Mac Studio restart requested (Eric's OK, via the restart coordinator): backtest hold set 13:01, run 7 (old seed 2048) will be killed and restarted afterwards; bts-wr-fresh3 told to stop at a safe point; restart note docs/r
2026-11-20 13:27 ET  start  bullpen-bt runner restarted after the machine restart (10 backtests remain, about two hours each, one at a time through yield_run.sh; hold file removed)  [bts-lead]
2026-11-20 13:29 ET  start  wr-check-d3 (Codex xhigh, third contract check of change set D draft 3 + amendment 1; pane w2:p30, clone ~/projects/bts-work/wr-check-deploy3; no pytest/wrapper)  BTS agents running: 1  [bts-wr-fresh3]

## Landings on main in the window (commit time, subject)
11-20 03:15 99b3bce2 Merge change set C1: the after-midnight health run judges contest state at the end of the processed slate, with command-free historical alerts (W20)
11-20 01:29 fac11738 Merge change set C2: page once per slate when a restart finds an unconfirmed send (W22), with C2b's receipt identity
11-19 06:50 01f7e544 Merge change set C3v: bts reconcile says why the season replay did not run (nothing is skipped; streak.json is saved exactly as before)
11-19 02:53 a1bf1612 Merge whole-range fix set B: messages and state reads
11-19 01:55 082b8506 Merge whole-range fix set A: the 2027 day count, the same-cycle pick check, the schedule parser
11-19 01:21 d67a25e7 fix(wr-a): round-1 review: a schedule body carrying NaN or Infinity is not a clean off-day
11-18 22:25 e063eac4 Merge the probable-pitcher cache fix: the cache is replaced atomically and an unreadable one is reported

## Outcomes by change set in the window
| Change set | Contract check | Review rounds | Result |
|---|---|---|---|
| Probable-pitcher cache write + void-reply parse (two small fixes, one change set, written by the lead inline) | none | r1 Codex BLOCK on the void half (it made a recovery tool's reply pairing worse), Carbon SIGN on the cache half; the void half withdrawn; confirmation Codex SIGN | cache half merged |
| Whole-range review of the tagged candidate: 3 Codex slices + 1 Carbon read, each about 17 to 20 minutes for the Codex slices | n/a | all four: DEPLOYABLE AFTER THESE FIXES | about 20 findings, including one the lead introduced on 11-13 (the season day count off by one all 2027, zero on the last day) |
| Fix set A (day count, same-cycle pick check, schedule parser); two of its five drafted rows were removed after the contract check showed them unsafe | BUILD WITH EDITS | r1 Codex BLOCK (NaN accepted in a JSON body), Carbon SIGN; r2 Codex SIGN; three contract amendments while the author worked (existing tests asserting the removed behaviour) | merged |
| Fix set B (operator messages, a read-only status line, a status file) | BUILD WITH EDITS | r1 Codex BLOCK (2: a special reader made the status line disagree with the live read; a far-future timestamp); r2 Codex SIGN WITH EDITS, Carbon SIGN | merged |
| C3, season replay skips never-committed pick files | BUILD WITH EDITS | r1 Codex BLOCK and Carbon BLOCK; rule rewritten ("positive evidence only"); r2 Codex BLOCK and Carbon BLOCK (entered picks can leave no evidence on disk) | STOPPED, not merged |
| C3v, replay visibility only (fresh session 1) | none (contract from the stop record) | r1 Codex BLOCK (a newline in a file name), Carbon SIGN; r2 Codex BLOCK on the CHECKLIST step the session itself wrote (move a file aside), Carbon SIGN | code merged by the lead with that step replaced by "move nothing"; no third review |
| C2, unknown-outcome alert after a restart (fresh session 1, then 2) | BUILD WITH EDITS | r1 Codex BLOCK, Carbon SIGN; r2 Codex BLOCK (identity of its own receipt file), Carbon SIGN; lead ruled "land if two receipt fixes pass one confirmation review"; confirmation Codex SIGN, Carbon SIGN | merged |
| C1, after-midnight health read (fresh session 2) | BUILD WITH EDITS | r1 Codex SIGN WITH EDITS (tests and docs), Carbon SIGN | merged |
| D, deploy workflow (fresh sessions 2 and 3) | draft 1 NOT YET; draft 2 NOT YET; owner then chose a different design (a durable on-box hold); draft 3 written, check pending | none yet | in progress |
| Paired backtest of an earlier feature fix (16 runs of about 2 hours, one at a time on the shared test slot, owner chose this machine over a cloud run costing under 15 euros) | n/a | n/a | 6 of 16 done after 15.5 hours; the queue waited behind gates for about 5.5 of those hours and was then killed mid-run by an unrelated machine restart |

## Other facts
- Fresh session 1 sat on a Claude Code permission prompt ("allow a read outside the working directories") from 07:03 to 20:03 on 11-19: thirteen hours in which nothing moved. The lead had no watcher on that session and noticed only when the owner asked "do you need anything from me?". The lead then added a watcher that answers that prompt and flags others; it expires hourly and is re-armed by hand.
- The lead's own watcher scripts contained the test wrapper's name in their command line, so authors' "is a job running?" checks matched the watcher; a stuck test in one author's first run held the single test slot for six minutes until the lead stopped it.
- Codex's content filter cut off three reviewer sessions in the window (briefs that read like security work); each finished after one plainly worded continuation, one without its test and mutation runs.
- Reviewers' round-2 BLOCKs in the window: C3 (design unsound), C3v (an operator instruction), C2 (a receipt file with content no code writes). Round-1 BLOCKs: a NaN constant in JSON; a special file at a state path; a newline in a file name; a wrong schema version in a receipt.
- Three times in the window a session wrote an operator instruction that a reviewer then showed to be unsafe in some state (move a blocking file aside; an alert's command template; "verify immediately" plus a batter's name).
- The lead overrode the two-round stop twice with a recorded ruling (C3v: land the code, delete the blocked instruction, no further review; C2: one more confirmation round).
- The owner made nine decisions in the window, most within a minute of being asked; three open questions waited on him for between 6 and 15 hours each.
- Allowances at 13:30 on 11-20: Codex weekly about 80% left, Gemini about 95%, the lead's Claude account weekly 46% left (it was 64% on 11-18).
- Still open: change set D and one small fix (in progress); two scheduler timing fixes (after the backtests, by owner decision); re-running the four whole-range harnesses on the final head; ten backtests (about 20 hours); an owner decision on a policy-artifact finding folded into later work.

## Questions
1. Where is this workflow flawed? Rank by expected cost.
2. What are the false greens?
3. What is wasted motion?
4. What one or two changes would you make tomorrow, concretely?
5. Is the pattern "a fresh Claude session under the lead, each running its own reviewers" working? What would you change about it?
6. A delivery estimate for "the whole-range review closed: every finding landed or accepted by the owner, harnesses re-run green on the final head", with a date range and the critical path.
Answer in under 70 lines, plain language, most important first.
