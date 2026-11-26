# Evidence brief: the BTS lead's workflow, 2026-11-21 16:10 to 2026-11-26 00:40 ET

You are asked for an independent opinion on a WORKFLOW, not on code. Facts are below; the lead's own diagnosis is left out. Use only this file.

## Setting
- One owner (Eric) of a baseball-streak contest bot that must be ready for a season opening 2027-03-24; the production box is restored about 2027-03-03. Nothing deploys before a whole-range review is closed. Eric alone decides spending, deploys, the deploy workflow, policy changes, scope cuts.
- A lead Claude session (Opus) coordinates on a shared Mac Studio (another project's run host: at most 3 BTS agents, ONE test job at a time through a wrapper, the fast suite is the 25-minute commit gate). The lead hands each piece to a fresh Claude session under it ("delegated session"), which runs its own Codex (xhigh) and Gemini Carbon agents.
- Rules in force (rulings R51, R57, R64): a change set gets a Codex contract check, tests first, the gate, Codex+Carbon review, at most two rounds; a second-round BLOCK stops it; the lead may rule ONE follow-up per stopped piece (naming the defect class and changing the mechanism), and if that stops the owner decides; a delegated session stops at its third counted BLOCK; for operator-facing text a finding blocks only if following the text can cause a wrong pick, wrong state, a lost alert or a lost day (R64), else it is SIGN WITH EDITS.
- The last daily workflow review was 2026-11-21 16:10; none ran on 11-22 to 11-25 (the rule is daily).

## Agent log (every start, stop and note, real clock times; lines cut at 260 characters)
2026-11-21 16:10 ET  stop   workflow-rev3 (opinion WFREV3-acffcdd0) and the lead's Claude subagent; review written up as ruling R57 (f1dfb8f3); fresh6's one-agent cap lifted  [bts-lead]
2026-11-21 16:15 ET  start  wr-d3b-check (Codex xhigh contract check of D3 amendment 4; pane w2:p4P; clone ~/projects/bts-work/wr-check-d3b)  BTS agents running: 1  [bts-wr-fresh6]
2026-11-21 16:35 ET  stop   wr-d3b-check (BUILD WITH THESE EDITS, WRCHKD3B-6eb4b7a2; archived contract-check-d3b/)  BTS agents running: 0  [bts-wr-fresh6]
2026-11-21 17:50 ET  hold   ~/bts-runs/backtest-hold placed (D3b focused run and gate next, after the running backtest ends about 17:59)  [bts-wr-fresh6]
2026-11-21 18:36 ET  start  wr-d3b-codex (Codex xhigh review of fix/wr-d3 at 90a9b6f0, whole branch; pane w2:p4Q; clone ~/projects/bts-work/wr-review-d3b)  BTS agents running: 1  [bts-wr-fresh6]
2026-11-21 18:37 ET  start  wr-d3b-carbon (Gemini 4 Carbon High review of fix/wr-d3 at 90a9b6f0, whole branch; pane w2:p4R; clone ~/projects/bts-work/wr-review-d3b-carbon)  BTS agents running: 2  [bts-wr-fresh6]
2026-11-21 19:16 ET  hold   ~/bts-runs/backtest-hold removed (gate on 90a9b6f0 ended; reviewers' focused runs wait for the next free slot)  [bts-wr-fresh6]
2026-11-21 21:04 ET  hold   ~/bts-runs/backtest-hold placed (both D3b reviewers' focused runs start when the running backtest ends, about 21:12)  [bts-wr-fresh6]
2026-11-21 21:17 ET  slot   GO to wr-d3b-codex focused runs (hold kept for both reviewers' batch)  [bts-wr-fresh6]
2026-11-21 21:51 ET  slot   wr-d3b-codex runs done; GO to wr-d3b-carbon (hold kept)  [bts-wr-fresh6]
2026-11-21 22:43 ET  hold   ~/bts-runs/backtest-hold removed (both D3b reviewers' runs done 22:43)  [bts-wr-fresh6]
2026-11-22 00:30 ET  hold   ~/bts-runs/backtest-hold placed (D3b round-1 fix focused run and gate next, after the running backtest ends about 00:40)  [bts-wr-fresh6]
2026-11-22 01:10 ET  round  wr-d3b-codex and wr-d3b-carbon given round 2 (428cfa93)  BTS agents running: 2  [bts-wr-fresh6]
2026-11-22 01:17 ET  stop   gate on 428cfa93 stopped (studio reboot); hold removed; reviewers wr-d3b-codex and wr-d3b-carbon mid round 2  [bts-wr-fresh6]
2026-11-22 01:17 ET  note   Mac Studio reboot (Eric); restart note docs/reviews/2026-11-18-whole-range/RESTART-2026-11-22.md; backtest 14/16 done, D3b gate killed by the reboot, fresh6 told to stop at a safe point  [bts-lead]
2026-11-22 01:38 ET  hold   ~/bts-runs/backtest-hold placed (after the studio reboot: gate on 428cfa93 next)  [bts-wr-fresh6]
2026-11-22 01:38 ET  gate   rerun on 428cfa93 started after the reboot (~/bts-runs/gate.out.d3b-428cfa93)  [bts-wr-fresh6]
2026-11-22 01:38 ET  restart  studio rebooted; per RESTART-2026-11-22.md: bts-lead resumed in w2:p43 (session f5a20749, Opus 5.5, RC off); watch_sessions.sh restarted; bts-wr-fresh6 resumed in w2:p4M (session 02177685) and told to continue from its RESUME POIN
2026-11-22 01:39 ET  restart wr-d3b-codex resumed after the reboot (codex resume 01a20077-... --no-daemon, memories off, xhigh; pane w2:p4Q); its round-2 turn had been stopped by the provider's content filter at about 01:15; one plain continuation sent  BTS ag
2026-11-22 01:39 ET  restart  watch_sessions.sh replaced (lead): backtest alerts now separate ('runner down' again after each runner restart; 'ALL DONE' once), runner matched as an exact process, not pgrep -f (EoR driver's note); old pid 3677 stopped, copy kep
2026-11-22 01:40 ET  restart wr-d3b-carbon restarted after the reboot (agy --model gemini-4-carbon-high --sandbox --continue, its round-2 conversation; config rechecked; pane w2:p4R)  BTS agents running: 2  [bts-wr-fresh6]
2026-11-22 02:38 ET  slot   GO wr-d3b-codex round-2 runs (late: my slot script mis-expanded its path in zsh, the slot was idle under the hold 02:21 to now)  [bts-wr-fresh6]
2026-11-22 03:14 ET  slot   wr-d3b-codex round-2 runs done; GO wr-d3b-carbon round-2 runs (hold kept)  [bts-wr-fresh6]
2026-11-22 03:16 ET  hold   ~/bts-runs/backtest-hold removed (D3b decided by Codex round 2; Carbon finishes its current wrapper run, no more)  [bts-wr-fresh6]
2026-11-22 03:17 ET  stop   wr-d3b-codex (round 1 BLOCK WRD3BREV1-80afbadd, round 2 BLOCK WRD3BREV2-7862a17a; archived; tab closed)  BTS agents running: 1  [bts-wr-fresh6]
2026-11-22 03:47 ET  stop   wr-d3b-carbon (round 1 BLOCK WRD3BCARB1-0133d0ea, round 2 SIGN WRD3BCARB2-38d11bbf; archived; tab closed)  BTS agents running: 0  [bts-wr-fresh6]
2026-11-22 03:50 ET  start  r52-rev (Codex xhigh, review of the R52 comparison tool before it reads the real backtest outputs; synthetic probes only, niced, no wrapper runs; pane w2:p4S)  BTS agents running: 1  [bts-lead]
2026-11-22 04:05 ET  stop   r52-rev (round 1 BLOCK R52REV-6a640abd, round 2 BLOCK R52REV2-807bb7d1); the lead builds one follow-up (R57) with a fresh reviewer  BTS agents running: 0  [bts-lead]
2026-11-22 04:11 ET  start  r52-fu (Codex xhigh, fresh review of the R52 tool's follow-up, R57; synthetic probes only, niced; pane w2:p4T)  BTS agents running: 1  [bts-lead]
2026-11-22 04:19 ET  stop   r52-fu (follow-up BLOCK R52FU-821af121); R57: the R52 tool question goes to Eric  BTS agents running: 0  [bts-lead]
2026-11-22 07:45 ET  start  wr-w3-check (Codex xhigh contract check of W3 kickoff-fix-w3.md v1; pane w2:p4V; clone ~/projects/bts-work/wr-check-w3)  BTS agents running: 1  [bts-wr-fresh6]
2026-11-22 08:12 ET  stop   wr-w3-check (check 1 BUILD WITH THESE EDITS WRCHKW3-e55ad7d9, check 2 BUILD WITH THESE EDITS WRCHKW3R2-06c17839; archived contract-check-w3/; tab closed)  BTS agents running: 0  [bts-wr-fresh6]
2026-11-22 08:12 ET  start  wr-w3-author (Codex default effort, W3 author; pane w2:p4W; worktree ~/projects/bts-work/wr-fix-w3, branch fix/wr-w3 from ceab9854)  BTS agents running: 1  [bts-wr-fresh6]
2026-11-22 08:40 ET  start  wr-w3-codex (Codex xhigh review of fix/wr-w3 at f7a8f601; pane w2:p4X; clone ~/projects/bts-work/wr-review-w3)  BTS agents running: 2  [bts-wr-fresh6]
2026-11-22 08:41 ET  start  wr-w3-carbon (Gemini 4 Carbon High review of fix/wr-w3 at f7a8f601; pane w2:p4Y; config checked)  BTS agents running: 3  [bts-wr-fresh6]
2026-11-22 09:12 ET  gate   on 59b72c04 (W3 second commit) started  [bts-wr-fresh6]
2026-11-22 10:03 ET  round  wr-w3-codex and wr-w3-carbon given round 2 (c0917aa7); gate on c0917aa7 running  [bts-wr-fresh6]
2026-11-22 10:34 ET  stop   wr-w3-codex (round 1 BLOCK WRW3REV1-4a3c8bb9, round 2 BLOCK WRW3REV2-e40968a3; archived) and wr-w3-author (commits f7a8f601, 59b72c04, 82acab73, c0917aa7 on fix/wr-w3; not merged; W3 STOPPED); tabs closed  BTS agents running: 1  [bt
2026-11-22 10:36 ET  start  bts-wr-fresh7 (fresh Claude session, Remote Control off; pane w2:p4Z, session 2eca93ca; kickoff ~/bts-runs/kickoff-wr-fresh7-2026-11-22.md: W4 only; up to three agents of its own minus fresh6's Carbon until it stops; on the watcher 
2026-11-22 10:44 ET  stop   wr-w3-carbon (round 1 BLOCK WRW3CARB1-87bcb5ce, round 2 SIGN WRW3CARB2-d452e88e; archived; tab closed)  BTS agents running: 0  [bts-wr-fresh6]
2026-11-22 10:45 ET  start  wr-w4-check (Codex xhigh contract check of W4 kickoff-fix-w4.md v1 at 5ace3050; pane w2:p40, session 01a203ef; clone ~/projects/bts-work/wr-check-w4)  BTS agents running: 2  [bts-wr-fresh7]
2026-11-22 10:58 ET  stop   wr-w4-check (contract check NOT YET WRCHKW4-54104b18; archived contract-check-w4/; W4 STOPPED at the contract stage, nothing built); tab closed  BTS agents running: 0 of mine  [bts-wr-fresh7]
2026-11-22 12:58 ET  gate   lead's gate on main + the R52 tool and its tests (uncommitted) through yield_run.sh, tag r52tool  [bts-lead]
2026-11-22 13:23 ET  run    the R52 comparison, ONCE, on the untouched outputs (tool da78ce9e, --streak, JSON ~/bts-runs/r52-result.json) through yield_run.sh  [bts-lead]
2026-11-22 14:02 ET  start  bullpen-bt runner extended to 24 seeds (R59: 16 more per arm, about 61 h; DONE now holds sha256 receipts); watcher restarted so its ALL DONE alert re-arms  [bts-lead]
2026-11-23 13:19 ET  start  bts-wr-fresh8 (fresh Claude session, Remote Control off; pane w2:p54, session 0802189a; kickoff ~/bts-runs/kickoff-wr-fresh8-2026-11-23.md: the closing test, parts a to d; up to three agents of its own; on the watcher list; told nev
2026-11-24 11:35 ET  note   bts-wr-fresh8 found idle since 2026-11-23 13:25 (API connection lost mid-response; the watcher does not flag idle sessions); resumed by the lead  [bts-lead]
2026-11-24 11:36 ET  restart  watch_sessions.sh now also reports a listed session idle or done for over 30 minutes (every 2 h at most); tested in both directions on a copy; list trimmed to sessions with work in progress (bts-wr-fresh8)  [bts-lead]
2026-11-24 11:38 ET  start  wr-blind (Codex xhigh blind read of FINAL bc143131, closing test part b; pane w2:p55, session 01a20e6c; clone ~/projects/bts-work/wr-closing-blind, review records removed)  BTS agents running: 1 of mine  [bts-wr-fresh8]
2026-11-24 11:40 ET  start  wr-carb2 (Gemini 4 Carbon High, slice 2 read of FINAL bc143131, closing test part c; pane w2:p56; clone ~/projects/bts-work/wr-closing-carbon2, review records removed; config checked)  BTS agents running: 2 of mine  [bts-wr-fresh8]
2026-11-24 11:40 ET  start  wr-carb3 (Gemini 4 Carbon High, slice 3 read of FINAL bc143131, closing test part c; pane w2:p57; clone ~/projects/bts-work/wr-closing-carbon3, review records removed; config checked)  BTS agents running: 3 of mine  [bts-wr-fresh8]
2026-11-24 11:44 ET  queued part a (four slice harnesses + adopted tests on FINAL) via ~/projects/bts-work/wr-closing-harness/closing-a/launch.sh: hold set at 12:40, one wrapped job when the slot frees, hold removed when it ends  [bts-wr-fresh8]
2026-11-24 12:31 ET  restart  watcher: a session whose screen shows a running background shell or monitor is not counted as idle (false alarm on bts-wr-fresh8 at 12:30); a finished session with none still reported (tested)  [bts-lead]
2026-11-24 12:58 ET  stop   wr-carb2 (Carbon slice 2 read of FINAL: DEPLOYABLE AFTER THESE FIXES, WRCARB2-af73c550; 3 blocking + 3 owner items, all reproduced or classified by me; archived; tab closed)  BTS agents running: 2 of mine  [bts-wr-fresh8]
2026-11-24 13:00 ET  window  hold window started (backtest-hold set 12:40, slot free 13:00): wr-blind 30 mutation runs, wr-carb3 2 runs, then part a; hold removed when they end  [bts-wr-fresh8]
2026-11-24 13:33 ET  stop   wr-blind (Codex xhigh blind read of FINAL: DEPLOYABLE AFTER THESE FIXES, WRBLIND-81f2c64f; 7 blocking, 5 owner items; archived; tab closed)  BTS agents running: 1 of mine  [bts-wr-fresh8]
2026-11-24 13:41 ET  hold   lead set backtest-hold: studio data volume 14 GB free (< 15 GiB no-new-job floor); the run started 13:35 finishes, nothing new starts until space returns  [bts-lead]
2026-11-24 13:41 ET  stop   wr-carb3 (Carbon slice 3 read of FINAL: DEPLOYABLE AFTER THESE FIXES, WRCARB3-00adaa93; 2 blocking + 3 owner items; archived; tab closed; copy kept for the reproduction run)  BTS agents running: 0 of mine  [bts-wr-fresh8]
2026-11-24 14:09 ET  clean  lead removed 114 finished BTS review/check/worktree copies (~116 GB nominal; review notes archived in ~/bts-runs/archive/review-notes-20261124.tgz); data volume 14 -> 36 GB free; backtest-hold removed  [bts-lead]
2026-11-24 14:10 ET  queued second batch (gap2.sh: 4 slice-2 matrices, slice-2 statuses, 33 slice-3 mutants, Carbon-3 B1/B2 reproduction) via launch4.sh: hold at 15:20, one wrapped job when the slot frees (about 15:30), hold removed when it ends  [bts-wr-fresh
2026-11-24 14:30 ET  clean  lead removed the last 30 BTS copies with uncommitted files (Eric's yes; files archived in ~/bts-runs/archive/uncommitted-work-20261124.tgz) and the old review bundles; BTS work copies 154 GB -> about 5 GB; data volume 43+ GB free  [
2026-11-24 15:43 ET  done   bts-wr-fresh8 closing test delivered (CLOSING-TEST.md on main 82e2b22b; season walk branch audit/season-walk 42c5d581, not merged); my copies deleted; BTS agents running: 0 of mine  [bts-wr-fresh8]
2026-11-24 16:14 ET  note   bts-wr-fresh8 finished (closing test done); removed from the watcher list  [bts-lead]
2026-11-24 18:54 ET  start  bts-wr-fresh9 (fresh Claude session, Remote Control off; pane w2:p58, session 9eefebe5; kickoff ~/bts-runs/kickoff-wr-fresh9-2026-11-24.md: merge the season walk, then change sets E and G; up to three agents of its own; on the watch
2026-11-24 18:55 ET  start  bts-wr-fresh9: backtest-hold set 18:55 for the season-walk merge gate (worktree ~/projects/bts-work/wr-e, 63669c42); gate starts when the running backtest ends; hold removed when the gate ends; no agents of mine yet  [bts-wr-fresh9]
2026-11-24 19:08 ET  start  wr-chk-e (Codex xhigh, memories off, pane w2:p59): contract check of change set E in ~/projects/bts-work/wr-check-e; no wrapper runs  [bts-wr-fresh9]
2026-11-24 19:28 ET  stop   wr-chk-e done (WRCHKE-2dd1d06c, BUILD WITH THESE EDITS); copy deleted, tab closed  [bts-wr-fresh9]
2026-11-24 19:28 ET  start  wr-e-author (Codex, memories off, pane w2:p5A, session 01a2101a): author of change set E code rows in ~/projects/bts-work/wr-fix-e; short single-file runs direct at nice 19, longer runs through the wrapper one at a time  [bts-wr-fre
2026-11-24 19:40 ET  note   bts-wr-fresh9: two gates queued behind the running backtest, back to back in one hold (season-walk 63669c42, then E db52bd02; about 55 min together); hold removed when the second ends  [bts-wr-fresh9]
2026-11-24 19:43 ET  stop   wr-e-author done (db52bd02 on fix/wr-e; handover WRE-6dd6f65d); tab closed  [bts-wr-fresh9]
2026-11-24 19:43 ET  start  wr-rev-e-cx (Codex xhigh, memories off, w2:p5B, session 01a21027) and wr-rev-e-cb (Carbon High, agy --sandbox, w2:p5C; /config read: Telemetry off, Non-Workspace Access off, Sandbox on, AI Credits off): round-1 reviews of E at db52b
2026-11-24 20:34 ET  note   bts-wr-fresh9: both gates done, backtest-hold removed 20:33  [bts-wr-fresh9]
2026-11-24 20:49 ET  stop   wr-rev-e-cx (WREREV1-2f4ce4d0 BLOCK) and wr-rev-e-cb (WRECARB1-45cfd488 BLOCK) done; copies deleted, tabs closed  [bts-wr-fresh9]
2026-11-24 20:49 ET  start  wr-e-author2 (Codex, memories off, w2:p5D, session 01a21064): round-2 author of E in ~/projects/bts-work/wr-fix-e; direct short runs only, no wrapper  [bts-wr-fresh9]
2026-11-24 20:54 ET  stop   wr-e-author2 done (256ac958; WRE2-ff86db1c); tab closed  [bts-wr-fresh9]
2026-11-24 20:54 ET  start  wr-rev2-e-cx (Codex xhigh, memories off, w2:p5E) and wr-rev2-e-cb (Carbon High, agy --sandbox, w2:p5F; /config as required): round-2 (last) reviews of E at 256ac958; direct short runs only. backtest-hold set 20:53 for the E round-2 
2026-11-24 21:08 ET  note   bts-wr-fresh9: E round-2 gate cancelled before it started (E stopped: second-round Codex BLOCK); backtest-hold removed 21:08 (it was set 20:53 and never delayed a backtest start)  [bts-wr-fresh9]
2026-11-24 21:11 ET  note   lead ruling R63: E's messages-only follow-up Ea, then G, then H, by bts-wr-fresh9; restore documents to January  [bts-lead]
2026-11-24 21:13 ET  start  wr-ea-author (Codex, memories off, w2:p5G): author of change set Ea (R63) in ~/projects/bts-work/wr-fix-ea; direct short runs only. Still running: wr-rev2-e-cb (Carbon, E round 2)  [bts-wr-fresh9]
2026-11-24 21:18 ET  note   bts-wr-fresh9: backtest-hold set 21:18 for the Ea gate (7faef255), queued behind the backtest that started 20:34; removed when the gate ends  [bts-wr-fresh9]
2026-11-24 21:19 ET  stop   wr-ea-author done (7faef255; WREA-992bc022); tab closed  [bts-wr-fresh9]
2026-11-24 21:19 ET  start  wr-rev-ea-cx (Codex xhigh, memories off, w2:p5H, session 01a21080) and wr-rev-ea-cb (Carbon High, agy --sandbox, w2:p5J; /config as required): reviews of Ea at 7faef255; direct short runs only. Still running: wr-rev2-e-cb  [bts-wr-f
2026-11-24 21:35 ET  stop   wr-rev-ea-cx done (WREAREV1-21601b4a BLOCK); copy deleted. Ea gate cancelled before start, backtest-hold removed 21:35. Still running: wr-rev-ea-cb, wr-rev2-e-cb (Carbon)  [bts-wr-fresh9]
2026-11-24 21:39 ET  start  bts-wr-fresh10 (fresh Claude session, Remote Control off; pane w2:p5K; kickoff ~/bts-runs/kickoff-wr-fresh10-2026-11-24.md: Ea round 2 under the R64 verdict rule, then G, then H; on the watcher list; bts-wr-fresh9 stopped at its thi
2026-11-24 21:41 ET  start  wr-ea2-author (Codex, memories off, w2:p5M, session 01a21094): Ea round-2 author in ~/projects/bts-work/wr-fix-ea; direct short runs only. Still running from fresh9: wr-rev-ea-cb, wr-rev2-e-cb (Carbon)  [bts-wr-fresh10]
2026-11-24 21:48 ET  stop   wr-ea2-author done (1ed57c47; WREA2-45f41f73); tab closed. backtest-hold set 21:47 for the Ea gate (1ed57c47), queued behind the backtest that started 20:34 (fresh10_gate_when_free.sh); removed when the gate ends  [bts-wr-fresh10]
2026-11-24 21:48 ET  start  wr-rev-ea2-cx (Codex xhigh, memories off, w2:p5N, session 01a2109a): Ea round-2 review at 1ed57c47 in ~/projects/bts-work/wr-rev-ea2-codex; direct short runs only. Still running from fresh9: wr-rev-ea-cb, wr-rev2-e-cb (Carbon); my C
2026-11-24 22:03 ET  stop   wr-rev-ea2-cx done (WREAREV2-06022bce BLOCK); tab closed; copies wr-fix-ea, wr-rev-ea2-codex, wr-rev-ea2-carbon and the bundle deleted. Ea gate cancelled before start, backtest-hold removed 22:02 (held 21:47 to 22:02 behind a runnin
2026-11-24 22:05 ET  start  wr-chk-g (Codex xhigh, memories off, w2:p5P, session 01a210aa): contract check of change set G at f44c0e61 in ~/projects/bts-work/wr-check-g; direct short runs only  [bts-wr-fresh10]
2026-11-24 22:06 ET  stop   wr-rev-ea-cb done (WREACARB1-70c1a267 BLOCK); copy deleted, tab closed. Still running: wr-rev2-e-cb  [bts-wr-fresh9]
2026-11-24 22:08 ET  done   bts-wr-fresh9 stopped at its third BLOCK (season walk merged 63669c42; E stopped; Ea round 1 BLOCK, branch fix/wr-ea pushed for a fresh session); all my agents closed, copies deleted, no hold; BTS agents running: 0 of mine  [bts-wr-
2026-11-24 22:17 ET  stop   wr-chk-g done (WRCHKG-f7eb49c4, BUILD WITH THESE EDITS); tab closed, copy and bundle deleted  [bts-wr-fresh10]
2026-11-24 22:17 ET  start  wr-g-author (Codex, memories off, w2:p5Q): change set G author in ~/projects/bts-work/wr-fix-g (contract v2 91032f8c); direct short runs only. fresh9's two Carbon reviewers are gone  [bts-wr-fresh10]
2026-11-24 22:29 ET  stop   wr-g-author done (d6141728; WRG-e68f06f5); tab closed. backtest-hold set 22:28 for the G gate (7484b38d), queued behind the backtest that started 20:34; removed when the gate ends  [bts-wr-fresh10]
2026-11-24 22:29 ET  start  wr-rev-g-cx (Codex xhigh, memories off, w2:p5R, session 01a210c0) and wr-rev-g-cb (Carbon High, agy --sandbox, w2:p5S; /config as required): reviews of G at 7484b38d; direct short runs only  [bts-wr-fresh10]
2026-11-24 22:45 ET  note   bts-wr-fresh10: the G gate (7484b38d) started 22:34 when the backtest ended and is running through the wrapper (ends about 23:02); backtest-hold removed 22:45, so the backtest runner takes the slot when it ends  [bts-wr-fresh10]
2026-11-24 23:01 ET  stop   wr-rev-g-cx done (WRGREV1-3e405af7 BLOCK); tab closed, copy deleted. G gate on 7484b38d ended 22:58 (ran 22:34 to 22:58)  [bts-wr-fresh10]
2026-11-24 23:01 ET  start  wr-g-author2 (Codex, memories off, w2:p5T): G round-2 author in ~/projects/bts-work/wr-fix-g. Still running: wr-rev-g-cb (Carbon)  [bts-wr-fresh10]
2026-11-24 23:09 ET  note   bts-wr-fresh10: backtest-hold set for the G round-2 gate (30fbdc2d, in ~/projects/bts-work/wr-gate-g), queued behind the backtest that started about 23:00; removed when the gate ends  [bts-wr-fresh10]
2026-11-24 23:50 ET  stop   wr-g-author2 done (cfc62745; WRG2-52fdee67) and wr-rev-g-cb done (WRGCARB1-e1fc6e6e BLOCK, the same finding as Codex B1); tabs closed, Carbon's copy deleted  [bts-wr-fresh10]
2026-11-24 23:50 ET  start  wr-rev-g2-cx (Codex xhigh, memories off, w2:p5V, session 01a2110a) and wr-rev-g2-cb (Carbon High, agy --sandbox, w2:p5W; /config as required): G round-2 reviews at 8be73842; direct short runs only  [bts-wr-fresh10]
2026-11-25 00:07 ET  stop   wr-rev-g2-cx done (WRGREV2-005a099e BLOCK); tab closed, copy deleted. G round-2 gate cancelled before start, backtest-hold removed 00:04. G stopped (STOP-G.md). Still running: wr-rev-g2-cb (Carbon, for the record)  [bts-wr-fresh10]
2026-11-25 00:08 ET  start  bts-wr-fresh11 (fresh Claude session, Remote Control off; pane w2:p5X, session b8da0193; kickoff ~/bts-runs/kickoff-wr-fresh11-2026-11-25.md: Ea option 1, then G as Eric decides, then H); bts-wr-fresh10 stopped at its third BLOCK (G
2026-11-25 00:10 ET  start  wr-ea3-author (Codex, memories off, w2:p5Y): Ea follow-up author (items 8 to 11) in ~/projects/bts-work/wr-fix-ea at c511c724; direct short runs only. Still running from fresh10: wr-rev-g2-cb (Carbon)  [bts-wr-fresh11]
2026-11-25 00:15 ET  note   bts-wr-fresh11: backtest-hold set for the Ea gate (988708df, in ~/projects/bts-work/wr-fix-ea), queued behind the running backtest; removed when the gate ends  [bts-wr-fresh11]
2026-11-25 00:17 ET  stop   wr-ea3-author done (988708df; WREA3-615152f6); tab closed  [bts-wr-fresh11]
2026-11-25 00:17 ET  start  wr-chk-ea3 (Codex xhigh, memories off, w2:p5Z, session 01a21122): sentence-limited check of Ea at 988708df in ~/projects/bts-work/wr-chk-ea3 (clone of bundle wr-ea-988708df.bundle); direct short runs only. Still running from fresh10
2026-11-25 00:30 ET  stop   wr-rev-g2-cb done (WRGCARB2-1f53b815 SIGN; Codex's BLOCK is the gate); tab closed; copies wr-fix-g, wr-rev-g2-carbon and my bundles deleted. bts-wr-fresh10 has no agent, job, watcher or hold left (the backtest-hold present now is no
2026-11-25 00:34 ET  stop   wr-chk-ea3 done (WREACHK3-f254bd86 BLOCK); tab closed; copies wr-chk-ea3 and wr-fix-ea and the bundle deleted (branch fix/wr-ea pushed at be65e087). Ea gate cancelled 00:32 before it started, backtest-hold removed 00:32. No agents o
2026-11-25 00:39 ET  start  wr-chk-h (Codex xhigh, memories off, w2:p50, session 01a21137): contract check of change set H at 1b971c36 in ~/projects/bts-work/wr-check-h (clone of bundle wr-h-1b971c36.bundle); direct short runs only  [bts-wr-fresh11]
2026-11-25 00:46 ET  note   bts-wr-fresh11: backtest-hold set for the G gate (7b61bcea = round 2's code, in ~/projects/bts-work/wr-gate-g) with the Ea gate behind it under the same hold if ready; queued behind the backtest that started 22:59; removed when they
2026-11-25 00:47 ET  start  wr-ea4-author (Codex, memories off, w2:p61): Ea second follow-up author (items 12 to 14) in ~/projects/bts-work/wr-fix-ea; direct short runs only. Also running: wr-chk-h  [bts-wr-fresh11]
2026-11-25 00:52 ET  stop   wr-ea4-author done (3f725779; WREA4-007c9c6f); tab closed
2026-11-25 00:52 ET  start  wr-chk-ea4 (Codex xhigh, memories off, w2:p62): check of Ea's three texts at 3f725779 in ~/projects/bts-work/wr-chk-ea4 (clone of a bundle); direct short runs only. Also running: wr-chk-h  [bts-wr-fresh11]
2026-11-25 00:59 ET  stop   wr-chk-h done (WRCHKH-355187e6, NOT YET); tab closed, copy and bundle deleted. Running: wr-chk-ea4; G gate in the slot since 00:56  [bts-wr-fresh11]
2026-11-25 01:03 ET  stop   wr-chk-ea4 done (WREACHK4-0f926650 BLOCK); tab closed; copies wr-chk-ea4 and wr-fix-ea and the bundle deleted (fix/wr-ea pushed at e63afd0e). The Ea gate was withdrawn from the queue before it started; the G gate (started 00:56) is 
2026-11-25 01:23 ET  note   bts-wr-fresh11: G gate ran 00:56 to 01:21 (green); backtest-hold removed 01:21 (held 00:46 to 01:21). G merged (f3afc02f); copy wr-gate-g removed  [bts-wr-fresh11]
2026-11-25 01:24 ET  start  wr-chk-h2 (Codex xhigh, memories off, w2:p63): second contract check of change set H (contract version 3, df96733c, base main with G) in ~/projects/bts-work/wr-check-h2; direct short runs only  [bts-wr-fresh11]
2026-11-25 01:43 ET  stop   wr-chk-h2 done (WRCHKH2-60281746, BUILD WITH THESE EDITS); tab closed, copy and bundle deleted
2026-11-25 01:43 ET  start  wr-h-author (Codex, memories off, w2:p64): change set H author in ~/projects/bts-work/wr-fix-h (contract version 3 with amendment 1, aee4b77d); direct short runs only  [bts-wr-fresh11]
2026-11-25 01:52 ET  note   bts-wr-fresh11: backtest-hold set for the H gate (813935bc, in ~/projects/bts-work/wr-gate-h), queued behind the backtest that started 01:22; removed when the gate ends (or at once if a review BLOCKs first)  [bts-wr-fresh11]
2026-11-25 01:56 ET  stop   wr-h-author done (813935bc; WRH-3a41b9a5); tab closed
2026-11-25 01:56 ET  start  wr-rev-h-cx (Codex xhigh, memories off, w2:p65, session 01a2117c) and wr-rev-h-cb (Carbon High, agy --sandbox, w2:p66; /config: Telemetry off, Non-Workspace Access off, Sandbox on, Use AI Credits off): reviews of H at 91d96148 (code
2026-11-25 02:16 ET  stop   wr-rev-h-cx done (WRHREV1-afc700cf BLOCK); tab closed, copy deleted. H gate cancelled 02:15 before start, backtest-hold removed. bts-wr-fresh11 stops at its third BLOCK. Still running: wr-rev-h-cb (Carbon, H round 1); kept: wr-fix-h
2026-11-25 02:18 ET  note   the lead ruled (02:17) that Ea's two sentence checks do not count: bts-wr-fresh11 continues at one BLOCK
2026-11-25 02:18 ET  start  wr-h-author2 (Codex, memories off, w2:p67): H round-2 author in ~/projects/bts-work/wr-fix-h (amendment 2, b8b89b37); direct short runs only. Still running: wr-rev-h-cb (Carbon, round 1)  [bts-wr-fresh11]
2026-11-25 02:22 ET  note   bts-wr-fresh11: backtest-hold set for the H round-2 gate (in ~/projects/bts-work/wr-gate-h), queued behind the backtest that started 01:22; removed when the gate ends  [bts-wr-fresh11]
2026-11-25 02:24 ET  stop   wr-h-author2 done (84805852; WRH2-9ab60698); tab closed
2026-11-25 02:24 ET  start  wr-rev-h2-cx (Codex xhigh, memories off, w2:p68, session 01a21196) and wr-rev-h2-cb (Carbon High, agy --sandbox, w2:p69; /config: Telemetry off, Non-Workspace Access off, Sandbox on, Use AI Credits off): H round-2 reviews at ac2818e
2026-11-25 02:42 ET  stop   wr-rev-h2-cx done (WRHREV2-97dcc5b0 BLOCK); tab closed, copy deleted. H round-2 gate cancelled 02:41 before start, backtest-hold removed. H stopped (STOP-H.md). Still running: wr-rev-h-cb and wr-rev-h2-cb (Carbon); kept: wr-fix-h, w
2026-11-25 03:04 ET  stop   wr-rev-h-cb done (Carbon round 1 on H: SIGN WRHCARB1-9093c1f2, archived on fix/wr-h a2ca1766); tab closed, copy deleted. Still running: wr-rev-h2-cb  [bts-wr-fresh11]
2026-11-25 03:15 ET  stop   wr-rev-h2-cb done (Carbon round 2 on H: BLOCK WRHCARB2-97f55ca9, archived); tab closed, copy and bundles deleted. No agents of mine are running; no hold. Kept: wr-fix-h, wr-gate-h  [bts-wr-fresh11]
2026-11-25 03:48 ET  note   bts-wr-fresh11 waiting on Eric's answers (H and Ea); taken off the watcher list until it has work  [bts-lead]
2026-11-25 07:11 ET  start  r52-24 (Codex xhigh, short review of the R52 tool's 24-seed extension and receipt check before its one final run; pane w2:p6A; no reads of the parquet outputs)  [bts-lead]
2026-11-25 07:23 ET  stop/start  r52-24 stopped (round 1 BLOCK R52X24-4e586247, round 2 BLOCK R52X24B-1c631c11); r52-fu2 (fresh Codex xhigh, the one follow-up check of the 24-seed extension; pane w2:p6B)  [bts-lead]
2026-11-25 17:20 ET  gate   lead's gate on main+the R52 tool merge (188af0ab, not pushed), hold set  [bts-lead]
2026-11-25 17:20 ET  stop   r52-fu2 (follow-up check BLOCK R52X24C-80f033a2; Eric chose option 1: tightenings applied, no further review)  [bts-lead]
2026-11-25 17:20 ET  start  wr-h-author3 (Codex, memories off, w2:p6C): H final-fix author (amendment 3, Eric's acceptance) in ~/projects/bts-work/wr-fix-h; direct short runs only  [bts-wr-fresh11]
2026-11-25 17:26 ET  stop   wr-h-author3 done (dad28268; WRH3-f411ba9b); tab closed
2026-11-25 17:26 ET  note   bts-wr-fresh11: two gates queued behind the lead's R52 gate under ONE hold kept present: the Ea merge candidate fa8085ae (~/projects/bts-work/wr-merge-ea), then the H merge candidate 36fd3e35 on top of it (wr-merge-h); hold removed 
2026-11-25 17:27 ET  start  wr-chk-h2b (Codex xhigh, memories off, w2:p6D): contract check of change set H2 (contract version 1, 500b7b5f, on the merge candidate 36fd3e35) in ~/projects/bts-work/wr-check-h2b; direct short runs only  [bts-wr-fresh11]
2026-11-25 17:48 ET  gate   r52x24 gate on 188af0ab: 10,841 passed; failing set = the pinned 53 + test_screen::test_the_runs_load_no_conftest and ::test_the_mutated_target_is_the_one_change_a_run_executes, both FLAKY ("files changed during the run: .git/object
2026-11-25 17:48 ET  stop   wr-chk-h2b done (WRCHKH2B-a565ae66, BUILD WITH THESE EDITS); tab closed, copy and bundle deleted
2026-11-25 17:48 ET  start  wr-h2-author (Codex, memories off, w2:p6E): change set H2 author in ~/projects/bts-work/wr-fix-h2 (contract v1 with amendment 1, bc789868); direct short runs only. My two gates hold the slot since 17:46 (Ea candidate, then H candida
2026-11-25 17:56 ET  stop   wr-h2-author done (b12e144c; WRH2C-1bbad4b3); tab closed
2026-11-25 17:56 ET  start  wr-rev-h2c-cx (Codex xhigh, memories off, w2:p6F, session 01a214eb) and wr-rev-h2c-cb (Carbon High, agy --sandbox, w2:p6G; /config: Telemetry off, Non-Workspace Access off, Sandbox on, Use AI Credits off): H2 reviews at cb1fb507 (co
2026-11-25 18:16 ET  run    lead's bullpen streak-gap check (docs/audit/2026-11-25-bullpen-streak-gap/check.py) through yield_run.sh  [bts-lead]
2026-11-25 18:16 ET  stop   lead stopped its own streak-gap check (started while bts-wr-fresh11's gate held the slot, against the one-job rule); requeued behind it  [bts-lead]
2026-11-25 18:23 ET  stop   wr-rev-h2c-cx done (WRH2CREV1-0f2a7c8b SIGN WITH EDITS); tab closed, copy deleted. Still running: wr-rev-h2c-cb (Carbon)  [bts-wr-fresh11]
2026-11-25 18:31 ET  stop   wr-rev-h2c-cb done (Carbon on H2: SIGN WRH2CCARB1-0466f07c); tab closed, copy and bundle deleted. No agents of mine are running; my H gate holds the slot  [bts-wr-fresh11]
2026-11-25 18:40 ET  run    lead's streak-gap check started (slot free)  [bts-lead]
2026-11-25 19:11 ET  done   bts-wr-fresh11: Ea (226e97de), G (f3afc02f), H (d29031a0) and H2 (e68edf9e) are on main (ed27620a). Final gate 18:43 to 19:09 green (11,021 passed, pinned 53). All my agents closed, copies and bundles deleted, no hold  [bts-wr-fresh
2026-11-25 19:14 ET  start  bts-wr-fresh12 (fresh Claude session, Remote Control off; pane w2:p6H, session b7da6723; kickoff ~/bts-runs/kickoff-wr-fresh12-2026-11-25.md: change set F in five pieces; on the watcher list); bts-wr-fresh11 finished (Ea, G, H, H2 m
2026-11-25 19:18 ET  start  wr-chk-f1 (Codex xhigh contract check of F1, memories off; pane w2:p6J; isolated copy ~/projects/bts-work/wr-chk-f1; short direct runs only, no wrapper)  [bts-wr-fresh12]
2026-11-25 19:35 ET  stop   wr-chk-f1 done (WRCHKF1-d2c4d503 BUILD WITH THESE EDITS); tab closed, copy and bundle deleted  [bts-wr-fresh12]
2026-11-25 19:35 ET  start  wr-f1-author (Codex author of F1, memories off; pane w2:p6K; worktree ~/projects/bts-work/wr-fix-f1; short direct runs only)  [bts-wr-fresh12]
2026-11-25 19:48 ET  stop   wr-f1-author done (1d30de86, WRF1-f32c8c43); tab closed. Gate on cd6fbf41 starting (hold set for it)  [bts-wr-fresh12]
2026-11-25 19:50 ET  start  wr-rev-f1-cx (Codex xhigh round-1 review of F1 at 1d30de86, memories off; pane w2:p6M; isolated copy; short direct runs only). F1 gate holds the slot since 19:48  [bts-wr-fresh12]
2026-11-25 19:50 ET  start  wr-rev-f1-cb (Gemini 4 Carbon High, --sandbox; /config read: Telemetry off, Non-Workspace Access off, Sandbox on, Use AI Credits off; pane w2:p6N; isolated copy; foreground runs one at a time)  [bts-wr-fresh12]
2026-11-25 20:08 ET  stop   wr-rev-f1-cx done (WRF1REV1-d44caf34 BLOCK); tab closed, copy deleted. F1 gate cancelled 20:08 at 16% (hold removed). Still running: wr-rev-f1-cb (Carbon)  [bts-wr-fresh12]
2026-11-25 20:08 ET  start  wr-f1-author2 (Codex round-2 author of F1, memories off; pane w2:p6P; worktree wr-fix-f1; short direct runs only)  [bts-wr-fresh12]
2026-11-25 20:17 ET  stop   wr-f1-author2 done (15b27474); tab closed. start  wr-rev-f1b-cx (Codex xhigh round-2 review of F1, memories off; pane w2:p6Q; isolated copy). F1 gate on 15b27474 holds the slot since 20:16. Still running: wr-rev-f1-cb (Carbon, round
2026-11-25 20:24 ET  stop   wr-rev-f1-cb done (Carbon round 1 on F1: SIGN WITH EDITS WRF1CARB1-0bc79b95); tab closed, copy and bundle deleted. start  wr-rev-f1b-cb (Carbon High round 2 on 15b27474, --sandbox; /config read: Telemetry off, Non-Workspace Access o
2026-11-25 20:29 ET  stop   wr-rev-f1b-cx done (WRF1REV2-dffa862e BLOCK; F1 stopped); tab closed, copy deleted. Still running: F1 gate on 15b27474 (hold since 20:16) and wr-rev-f1b-cb (Carbon round 2)  [bts-wr-fresh12]
2026-11-25 20:51 ET  start  wr-chk-f1b (Codex xhigh contract check of F1b, memories off; pane w2:p6S; isolated copy; short direct runs only)  [bts-wr-fresh12]
2026-11-25 20:57 ET  stop   wr-rev-f1b-cb done (Carbon round 2 on F1: SIGN WRF1CARB2-80db83ef); tab closed, copy and bundle deleted. Running: wr-chk-f1b  [bts-wr-fresh12]
2026-11-25 21:06 ET  stop   wr-chk-f1b done (WRCHKF1B-0cc2cee6 BUILD WITH THESE EDITS); tab closed, copy deleted. start  wr-f1b-author (Codex author of F1b, memories off; pane w2:p6T; worktree wr-fix-f1)  [bts-wr-fresh12]
2026-11-25 21:22 ET  stop   wr-f1b-author done (717ac97f); tab closed. start  wr-rev-f1c-cx (Codex xhigh F1b review, pane w2:p6V) and wr-rev-f1c-cb (Carbon High, --sandbox; /config read: Telemetry off, Non-Workspace Access off, Sandbox on, Use AI Credits off; 
2026-11-25 21:38 ET  stop   wr-rev-f1c-cx done (F1+F1b round 1: SIGN WITH EDITS WRF1BREV1-9fe31ecf); tab closed, copy deleted. Running: wr-rev-f1c-cb, the gate  [bts-wr-fresh12]
2026-11-25 22:04 ET  stop   wr-rev-f1c-cb done (Carbon F1b: SIGN WRF1BCARB1-8b50179e); tab closed, copy deleted. F1+F1b merged to main a4e8278d. No agents of mine running, no hold  [bts-wr-fresh12]
2026-11-25 22:37 ET  start  wr-chk-f5 (Codex xhigh contract check of F5, memories off; pane w2:p6X; isolated copy; may read data/raw read-only)  [bts-wr-fresh12]
2026-11-25 22:50 ET  stop   wr-chk-f5 done (WRCHKF5-d88824e2 BUILD WITH THESE EDITS); start  wr-f5-author (Codex author of F5, memories off; pane w2:p6Y; worktree wr-fix-f5)  [bts-wr-fresh12]
2026-11-25 23:01 ET  stop   wr-f5-author done (5a80f0cc). start  wr-rev-f5-cx (Codex xhigh, pane w2:p6Z) and wr-rev-f5-cb (Carbon High --sandbox, /config read: Telemetry off, Non-Workspace Access off, Sandbox on, Use AI Credits off; pane w2:p60). F5 gate on 0f
2026-11-25 23:19 ET  stop   wr-rev-f5-cx done (F5 round 1 Codex BLOCK WRF5REV1-2b64c6ac); tab closed, copy deleted. Still running: F5 gate, wr-rev-f5-cb  [bts-wr-fresh12]
2026-11-25 23:21 ET  start  bts-wr-fresh13 (fresh Claude session on Opus 5.5, Remote Control off; pane w2:p71; kickoff ~/bts-runs/kickoff-wr-fresh13-2026-11-25.md: F2 check, F5 round 2, F2, F3, F4); bts-wr-fresh12 stopped at its third counted BLOCK, finishing 
2026-11-25 23:25 ET  start  wr-chk-f2 (Codex xhigh contract check of F2, memories off; pane w2:p72; isolated copy ~/projects/bts-work/wr-chk-f2; short direct runs only)  [bts-wr-fresh13]
2026-11-25 23:38 ET  stop   wr-chk-f2 done (WRCHKF2-f8ef3e6e BUILD WITH THESE EDITS); tab closed, copy deleted. start  wr-f2-author (Codex author of F2, memories off; pane w2:p73; bundle clone ~/projects/bts-work/wr-fix-f2; short direct runs only)  [bts-wr-fre
2026-11-25 23:47 ET  done   bts-wr-fresh12: F1+F1b merged (a4e8278d); F5 round 1 Codex BLOCK (session's third counted BLOCK), records on fix/wr-f5 30087081, FINDINGS on main 5e0167b5. All my agents closed, no hold; worktree wr-fix-f5 kept for round 2  [bts-wr-
2026-11-25 23:47 ET  stop   bts-wr-fresh12 done (third counted BLOCK); F5 handed to bts-wr-fresh13 for round 2  [bts-lead]
2026-11-25 23:49 ET  stop   wr-f2-author done (2e20fdfc); tab closed. start  wr-rev-f2-cx (Codex xhigh F2 review, memories off; pane w2:p74) and wr-rev-f2-cb (Carbon High --sandbox, /config read: Telemetry off, Non-Workspace Access off, Sandbox on, Use AI Cred
2026-11-25 23:51 ET  start  wr-f5-author2 (Codex round-2 author of F5, memories off; pane w2:p76; worktree wr-fix-f5; short direct runs only). Running: wr-rev-f2-cx, wr-rev-f2-cb, F2 gate  [bts-wr-fresh13]
2026-11-26 00:10 ET  stop   wr-rev-f2-cx done (F2 round 1 Codex BLOCK WRF2REV1-6475edec); tab closed. stop wr-f5-author2 done (84e2a0fc); tab closed. F2 round-1 gate cancelled at 00:07 (hold removed). start  wr-rev-f5b-cx (Codex xhigh F5 round-2 review, pane w
2026-11-26 00:16 ET  stop   wr-rev-f2-cb done (F2 round 1 Carbon BLOCK WRF2CARB1-7abb62c5, same finding as Codex's first); tab closed. start  wr-f2-author2 (Codex round-2 author of F2, memories off; pane w2:p79; bundle clone wr-fix-f2). Running: wr-rev-f5b-cx,
2026-11-26 00:24 ET  stop   wr-f2-author2 done (3b7d5a18); tab closed; round-1 review copies deleted. start  wr-rev-f2b-cx (Codex xhigh F2 round-2 review, memories off; pane w2:p7A). F2 gate queued behind the F5 gate (launcher fresh13_gate_after.sh, hold kept)
2026-11-26 00:27 ET  stop   wr-rev-f5b-cx done (F5 round 2 Codex BLOCK WRF5REV2-67ca3192; F5 goes to the lead, R57). F5 gate on 6eed95a4 cancelled 00:26; F2 gate on 44a4a922 started 00:26 (hold kept). start  wr-rev-f2b-cb (Carbon High --sandbox F2 round 2, /co

## Landings and records on main in the window (first-parent, filtered to merges, stops, rulings, decisions)
11-25 23:47 5e0167b5 docs(review): F5 round 1 Codex BLOCK; bts-wr-fresh12 stops at its third counted BLOCK
11-25 22:01 ba53b46a Merge change sets F1 and F1b onto main (the entry watchdog: a batter in the wrong game is not confirmed; a singleton lock and run limit on its cron l
11-25 20:49 52676b0e docs(review): bts-wr-fresh12 continues on Opus 5.5 after its Fable limit (Eric: 3)
11-25 20:44 fd83279f docs(review): ruling R67, F1b (F1 plus a singleton lock on the watchdog cron line) as F1's one follow-up
11-25 19:36 97b864f9 docs(review): F1 ruling on an unreadable capture (batter confirmed, game unverified; no false all-clear, no nag)
11-25 19:11 ed27620a docs(review): clock time corrected (H and H2 merged 19:11)
11-25 19:09 99b9a00a Merge remote-tracking branch 'origin/main' into land/main
11-25 18:41 e68edf9e Merge change set H2 onto main (the contest-state caps raised from 57 to 400; Codex SIGN WITH EDITS, Carbon SIGN)
11-25 18:41 d29031a0 Merge change set H onto main (the longest-streak rule from streak 57; Eric's acceptance of 2026-11-25)
11-25 18:22 f9e8d4ef docs(review): change set Ea merged (FINDINGS.md)
11-25 18:21 226e97de Merge change set Ea onto main (the operator messages; gate on fa8085ae: 10,892 passed, failing set the pinned 53 plus one test that passes alone)
11-25 17:53 419e8c22 docs(review): the bullpen backtest's final R59 test: PASS (mean +0.25 pt, lower bound -0.05 pt); the mean max streak is 0.74 lower with the fix (repo
11-25 17:20 188af0ab Merge audit/r52-24seeds: the R52 tool on R59's 24 seeds with receipt checks (Eric's option 1, 2026-11-25)
11-25 17:19 096700d9 docs(review): Eric's decisions of 2026-11-25 17:19 (option 1 on the R52 tool, H and Ea)
11-25 03:15 2b90fb23 docs(review): STOP-H.md, Carbon's two reviews of H noted (round 1 SIGN, round 2 BLOCK on the same class)
11-25 02:42 7a03b67d docs(review): change set H stopped after round 2 (Codex BLOCK WRHREV2-97dcc5b0); STOP-H.md
11-25 02:16 1b9fc3e7 docs(review): H round 1 Codex BLOCK WRHREV1-afc700cf recorded; bts-wr-fresh11 stops at its third BLOCK; H2 draft and cap ruling noted
11-25 01:22 f3afc02f Merge change set G: the monotone skip guard (R61; accepted as it stands by Eric 2026-11-25, STOP-G.md option 1)
11-25 01:03 8b7316de docs(review): Ea option 2 built (3f725779); its three-text check is a Codex BLOCK WREACHK4-0f926650; second STOP-EA.md addendum for Eric
11-25 01:00 119a72c4 docs(review): ruling R66 on H's contract check questions (keep R33; the stale-57 gap is a limit H2 removes; H never deploys without H2)
11-25 00:48 4078f2dc docs(review): correct two times the lead typed instead of reading the clock (R65 00:36, Eric's answers about 00:45)
11-25 00:45 a9a341b1 docs(review): Eric's answers of 2026-11-25 (Ea option 2; G accepted as it stands; streak 56 keeps the table's waiting)
11-25 00:36 71590310 docs(review): ruling R65, H's design (the closed-form rule from 57; H2 raises the 57 caps; streak 56 to Eric)
11-25 00:33 3fbf7d3f docs(review): Ea option 1 built (988708df); its sentence-limited check is a Codex BLOCK WREACHK3-f254bd86; STOP-EA.md addendum for Eric
11-25 00:29 b939a436 docs(review): G round 2 Carbon SIGN recorded (G stays stopped on Codex's BLOCK)
11-25 00:06 4befa3ae docs(review): change set G stopped after round 2 (Codex BLOCK WRGREV2-005a099e); STOP-G.md; bts-wr-fresh10 stops at its third BLOCK
11-24 23:00 939cdb13 docs(review): Eric chose Ea option 1 (apply the reviewers' replacement texts, gate, a sentence-limited check)
11-24 22:07 c363d269 docs(review): Carbon's round-2 review of change set E archived (BLOCK, documents only); STOP-E.md completed
11-24 22:03 f83e0d0b docs(review): Ea round 2 Codex BLOCK recorded; Ea stops (STOP-EA.md), the messages go to Eric
11-24 21:38 3bf0ec13 docs(review): ruling R64, a verdict rule for operator-text reviews (block only on a harmful outcome; imprecision is SIGN WITH EDITS)
11-24 21:36 cf354ebd docs(review): Ea round 1 Codex BLOCK recorded; bts-wr-fresh9 stops at its third BLOCK
11-24 21:11 b1abc2bf docs(review): ruling R63 after E stopped (Ea messages-only follow-up; the restore documents move to January with the Debian rehearsal)
11-24 21:10 49d9030e docs: CLAUDE.md, change set E stopped (what still holds on main); the season walk is on main
11-24 21:09 7f6ea918 docs(review): change set E stopped after round 2 (record, contract, check and reviews); the season walk merged
11-24 20:21 b215832b docs(review): Eric's objective from streak 56 on (the longest streak possible); ruling R62, change set H
11-24 18:55 63669c42 Merge audit/season-walk: the 2027 season walk (closing test part d; tier C, built test-first, run end to end in the closing test)
11-24 18:54 52336a5a docs(audit): why the shipped reach-57 table skips a bin while a weaker one plays (a fitting artifact; the model's scale is flat at the top); ruling R
11-24 17:45 b18326a0 docs(review): Eric's answers on the closing-test findings (monotone skip guard after an investigation; leaderboard timer step; watchdog covers every 
11-24 15:44 ea3564c9 docs(review): ruling R60, what the closing test's findings get (change sets E and F, checklist rows, owner questions, ledger)
11-24 14:30 4cba5fa5 docs: delete each review and author copy when its change set closes (154 GB of copies had accumulated and stopped every job on 2026-11-24)
11-22 14:02 bc143131 docs(review): ruling R59, the bullpen backtest extended to 24 seeds with one final R52 test and no interim look (Eric: "3")
11-22 13:25 07331dd5 docs: CLAUDE.md records the R52 backtest FAIL (the deploy waits for Eric)
11-22 13:25 281cd680 docs(review): the bullpen backtest against R52: FAIL (mean -0.13 pt, lower bound -0.77 pt; 2021 and 2023 below -1.0 pt); the deploy waits for Eric
11-22 12:58 2a20ea94 docs(review): Eric's decisions of 2026-11-22 (W3 and W4 accepted as documented; the R52 tool's remaining findings accepted), rulings R55 to R58, the 
11-22 10:58 a43c547c docs(review): W4 stop time corrected to 10:58 ET (taken from date; 11:05 was written by guess)
11-22 10:58 3d8b251f docs(review): W4 STOPPED at the contract stage (Codex contract check NOT YET, WRCHKW4-54104b18); STOP-W4.md, FINDINGS, CLAUDE.md, check archived
11-22 10:44 1e95b156 docs(review): W3 round 2 Carbon SIGN (WRW3CARB2-d452e88e) archived; STOP-W3.md records it
11-22 10:33 beb873ef docs(review): W3 STOPPED after round 2 (Codex BLOCK WRW3REV2-e40968a3); STOP-W3.md, FINDINGS, CLAUDE.md, handover and review archived
11-22 08:36 084d6925 docs(review): W3 ruling, after the loop today's behaviour (no alert or fallback suppressed by an abandoned reuse)
11-22 03:47 088b4c52 docs(review): D3b round 2 Carbon SIGN (WRD3BCARB2-38d11bbf) archived; STOP-D3B.md records it
11-22 03:16 750339df docs: CLAUDE.md records D3b stopped and the deploy workflow deferred to January
11-22 03:16 4c3d39d5 docs(review): D3b STOPPED after round 2 (Codex BLOCK WRD3BREV2-7862a17a); deploy workflow deferred to January (Eric's b); STOP-D3B.md, FINDINGS, CLAU
11-22 02:30 99ea7140 docs(review): Eric's decisions of 2026-11-22 (a stopped D3b defers the deploy workflow to January; the Mac mini only when needed)
11-21 16:13 a16523e7 docs(review): D3 contract amendment 4 (D3b, rulings R56 and R57): the class, one proof function, one signal rule, must-still-catch table
11-21 16:10 f1dfb8f3 docs(audit): the daily multi-model workflow review for 2026-11-21 (ruling R57: one follow-up per stopped piece, then Eric; no Carbon-only merge; hold

## Outcomes by piece in the window
| Piece | Kind | Contract checks | Review rounds | Result |
|---|---|---|---|---|
| D3b (deploy workflow follow-up) | deploy, tier A | none | r1 BLOCK both, r2 Codex BLOCK | stopped; Eric deferred the deploy workflow to January |
| Closing test (harnesses, blind read, Carbon x2, season walk) | test | n/a | n/a | done; 19 new findings; a 22-hour idle gap after a dropped API connection, unseen until the owner asked |
| E (operator texts) | text | 1 | r1, r2 BLOCK | stopped; split into Ea (messages) and Eb (restore documents, to January) |
| Ea (messages follow-up) | text | n/a | r1 BLOCK, r2 BLOCK, then owner option 1 check BLOCK, option 2 check BLOCK, then owner acceptance | merged on owner acceptance; findings per round 8, 7, 6, 5, 3, 1 |
| G (monotone skip guard; owner-approved policy change) | policy | 1 | r1 BLOCK, r2 BLOCK (both on pre-existing decision-record behaviour) | merged on owner acceptance |
| H (longest-streak rule from 57) | policy | 3 | r1 BLOCK, r2 BLOCK (a corrupt-artifact promise) | merged on owner acceptance; the gate then found a regression neither contract check nor review had run (fixed in 6 lines) |
| H2 (raise the 57 caps) | verifier loosening | 1 | r1 SIGN WITH EDITS, Carbon SIGN | merged |
| R52 comparison tool, then its 24-seed extension | result-deciding tool | n/a | BLOCK, BLOCK, follow-up BLOCK; extension BLOCK, BLOCK, follow-up BLOCK | run on owner acceptance; results: FAIL on 8 seeds, PASS on 24 seeds (owner chose more seeds) |
| F1 + F1b (watchdog wrong-game) | alerting | 2 | F1 r1, r2 BLOCK; F1b r1 SIGN WITH EDITS | merged |
| F5 (watchdog all day) | alerting | 1 | r1 BLOCK, r2 BLOCK | stopped; lead ruled follow-up F5b (not started) |
| F2 (saver-flag warning) | alerting | 1 | r1 BLOCK, r2 BLOCK | stopped; lead ruled follow-up F2b (not started) |
| F3, F4 (morning-report visibility) | alerting | F3 contract drafted, unchecked | none | not started |

## Other facts
- Delegated sessions started in the window: fresh8 to fresh13 (six). Each but fresh8 stopped at its third counted BLOCK; fresh9 to fresh13 averaged under two pieces each.
- Owner questions in the window: about 25, answered within minutes to about six hours, several overnight; most answered "1" (the lead's recommendation).
- The lead's own errors in the window: twice typed clock times instead of reading the clock; started an analysis job while another job held the single test slot (stopped within a minute); pushed a git commit during a delegated session's gate (which can make a test flaky); stated a hypothesis about the backtest that its own check then refuted.
- A delegated session hit its model's usage limit mid-task; the owner chose to switch it to another model; the default model for new sessions changed as a side effect.
- The single test slot: a backtest queue (deploy condition) shared it with gates until 11-25 07:41.
- Allowances at 00:40 on 11-26: Codex weekly about 98%, the lead's Claude account weekly about 39%.
- Open before deploy: F2b, F3, F4, F5b; the closing test re-run on the moved head; January: the restore documents, the deploy workflow, a Debian rehearsal.

## Questions
1. Where is this workflow flawed? Rank by expected cost.
2. What are the false greens?
3. What is wasted motion?
4. What one or two changes would you make tomorrow, concretely, for the remaining pieces (F2b, F3, F4, F5b, the closing-test re-run)?
5. Is "two rounds, one follow-up, then the owner" converging, or is it a long route to the owner's acceptance? What would you change?
6. A delivery estimate for "the whole-range review closed and the closing test green on the final head", with a date range and the critical path.
Answer in under 70 lines, plain language, most important first.
