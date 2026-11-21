# Evidence brief: the BTS lead's workflow, 2026-11-20 13:30 to 2026-11-21 16:10 ET

You are asked for an independent opinion on a WORKFLOW, not on code. Facts are below; the lead's own diagnosis is deliberately left out. Use only this file.

## Setting
- One person (Eric) owns a baseball-streak contest bot that must be ready for a season opening 2027-03-24 (a lockout may move it). The production box is deleted for the offseason and is restored 21 days before a confirmed opener. Nothing deploys until a whole-range review is closed. Eric alone decides spending, deploys, the deploy workflow's design, the contest account, policy changes, scope cuts.
- A lead Claude session coordinates on a shared Mac Studio (also another project's run host: at most 3 BTS agents, ONE test job at a time through a wrapper, everything pauses while the other project's game engine runs; the fast suite is the commit gate and takes about 26 minutes).
- Process for a pick-path change: a contract; a fresh Codex check of the contract against the real code before anyone builds; an author test-first; the gate; two fresh adversarial reviewers in isolated copies (Codex, whose verdict is the gate, and Gemini Carbon); at most two rounds; a second-round BLOCK stops the change set. Since 11-20 (ruling R51) the lead may not override a second-round BLOCK: what lands next is a materially revised piece with a fresh verdict, or the owner's recorded acceptance. Operator-facing text is reviewed with its change.
- The lead session itself was REPLACED by a fresh one at 13:38 on 11-20 (owner's word, after the previous review recommended it). The new lead hands each piece of work to a fresh Claude session under it (fresh sessions 3 to 6 in this window), each with a written kickoff, each running its own Codex and Carbon agents. A standing watcher script answers one known permission prompt and messages the lead when a session is stuck or the backtest runner dies.
- The new lead's own session was restarted once by the harness (about 03:00 on 11-21); its herdr name was cleared and it restored it by hand.

## Agent log (every start, stop and note, real clock times)
2026-11-20 13:30 ET  start  workflow-rev2 (Codex xhigh, the daily workflow review from an evidence brief; reads one file, no test jobs; pane w2:p41)  [bts-lead]
2026-11-20 13:33 ET  stop   workflow-rev2 (opinion WFREV2-195b1269)  [bts-lead]
2026-11-20 13:33 ET  start  watch_sessions.sh (the lead's standing watcher: one loop a minute, a few herdr reads; answers the known read prompt for delegated sessions and messages the lead when one is stuck or the backtest runner stops)  [bts-lead]
2026-11-20 13:34 ET  start  c3v-text (Codex xhigh, review of one operator instruction against the code; reading and small probes, no gate; pane w2:p42)  [bts-lead]
2026-11-20 13:38 ET  note   lead handoff on Eric's word: a fresh Claude session in pane w2:p43 is now 'bts-lead' (handoff note docs/superpowers/notes/2026-11-20-lead-handoff.md); the old lead in w2:p3 is 'bts-lead-old' and will run no further agents  [bts-lead-old]
2026-11-20 13:40 ET  note   new lead (pane w2:p43, Claude session f5a20749) has taken over as bts-lead; inherits bts-wr-fresh3 (with its wr-check-d3), c3v-text, the backtest runner and the watcher. Watcher's runner-down message path checked in the failure direction with a one-pass copy on fake paths (message reached bts-lead); i
2026-11-20 13:43 ET  note   c3v-text round 1: BLOCK (5 MUST FIX with replacement text, report C3VTEXT-0a080c55); lead applied the replacements, round 2 confirmation sent to the same agent  [bts-lead]
2026-11-20 13:46 ET  stop   c3v-text (round 2 SIGN WITH EDITS, C3VTEXT2-3cd91f81; edits applied, 83163cb6; tab w2:t42 closed)  BTS agents running: 1 (wr-check-d3, owned by bts-wr-fresh3)  [bts-lead]
2026-11-20 13:52 ET  hold   ~/bts-runs/backtest-hold set (change set D author about to start; the running backtest finishes first)  [bts-wr-fresh3]
2026-11-20 13:54 ET  start  wr-deploy-author (Codex author, change set D v5; pane w2:p44, worktree ~/projects/bts-work/wr-fix-deploy, branch fix/wr-deploy; design checkpoint first)  BTS agents running: 2  [bts-wr-fresh3]
2026-11-20 13:57 ET  stop   wr-check-d3 (contract checks r3 NOT YET, r4 BUILD WITH THESE EDITS; archived in contract-check-deploy/)  BTS agents running: 1  [bts-wr-fresh3]
2026-11-20 16:17 ET  start  wr-d-codex (Codex xhigh, change set D round-1 review of 46ce572d; pane w2:p45, clone ~/projects/bts-work/wr-review-d; no pytest before 16:45)  BTS agents running: 2  [bts-wr-fresh3]
2026-11-20 16:18 ET  start  wr-d-carbon (Gemini 4 Carbon high via agy --sandbox, change set D round-1 review of 46ce572d; pane w2:p46, clone ~/projects/bts-work/wr-review-d-carbon)  BTS agents running: 3  [bts-wr-fresh3]
2026-11-20 17:46 ET  stop   wr-d-codex (round 1 BLOCK archived; round 2 stopped twice by the provider content filter, no report)  BTS agents running: 2  [bts-wr-fresh3]
2026-11-20 17:47 ET  note   lead ruling: change set D round 2 may use one fresh Codex xhigh reviewer (plain wording) after wr-d-codex's two filter stops; Carbon-only fallback only on a Carbon SIGN, flagged for the closing blind read  [bts-lead]
2026-11-20 17:48 ET  start  wr-d-codex2 (fresh Codex xhigh, change set D round 2 on 0645be4b, plain-worded brief per bts-lead ruling; pane w2:p45; no test runs before 18:10)  BTS agents running: 3 (author, Carbon, this)  [bts-wr-fresh3]
2026-11-20 18:11 ET  stop   wr-deploy-author (change set D: commits f6bb8b5b, 46ce572d, 0645be4b on fix/wr-deploy; not merged)  [bts-wr-fresh3]
2026-11-20 18:11 ET  stop   wr-d-carbon (round 1 BLOCK, round 2 BLOCK; archived)  BTS agents running: 1 (wr-d-codex2 finishing)  [bts-wr-fresh3]
2026-11-20 18:11 ET  hold   ~/bts-runs/backtest-hold removed (change set D stopped; no further BTS test job planned by bts-wr-fresh3)  [bts-wr-fresh3]
2026-11-20 18:12 ET  stop   wr-d-codex2 (round 2 BLOCK, WRDREV2B-d5701b11; pytest/mutation not run at the lead's request)  BTS agents running: 0  [bts-wr-fresh3]
2026-11-20 20:33 ET  start  bts-wr-fresh4 (fresh Claude session, Remote Control off; pane w2:p47, session 6d0b0e0a; kickoff ~/bts-runs/kickoff-wr-fresh4-2026-11-20.md: W15 then D2; up to three agents of its own, lead runs none meanwhile; added to the watcher list)  [bts-lead]
2026-11-20 20:36 ET  start  wr-check-w15 (Codex xhigh contract check of kickoff-fix-w15.md at 59df4131; pane w2:p48, session 01a1fbbf; clone ~/projects/bts-work/wr-check-w15)  BTS agents running: 1  [bts-wr-fresh4]
2026-11-20 21:10 ET  hold   ~/bts-runs/backtest-hold set (W15 author starting; its focused tests and gate are next)  [bts-wr-fresh4]
2026-11-20 21:11 ET  start  wr-w15-author (Codex author of W15, contract v3 + amendment 1 at f02e6b9c; pane w2:p49, session 01a1fbdf; worktree ~/projects/bts-work/wr-fix-w15)  BTS agents running: 2  [bts-wr-fresh4]
2026-11-20 21:11 ET  stop   wr-check-w15 (W15 contract checks: NOT YET, NOT YET, BUILD WITH THESE EDITS; archived contract-check-w15/)  BTS agents running: 1  [bts-wr-fresh4]
2026-11-20 22:01 ET  start  wr-check-d2 (Codex xhigh contract check of kickoff-fix-deploy-v6.md at a0d369b3; pane w2:p4A, session 01a1fc0d; clone ~/projects/bts-work/wr-check-d2; lead's yes with conditions: stopped before W15 reviewers start, small niced probes while victoria3 runs)  BTS agents running: 2  [bts-wr-fresh4]
2026-11-20 22:34 ET  stop   wr-check-d2 (D2 contract checks: NOT YET x3; D2 STOPPED, STOP-D2.md; archived contract-check-d2/)  BTS agents running: 1  [bts-wr-fresh4]
2026-11-21 00:59 ET  start  wr-w15-codex (Codex xhigh round-1 review of W15 a924d748; pane w2:p4B, session 01a1fcb0; clone ~/projects/bts-work/wr-review-w15; no pytest before 01:35)  BTS agents running: 2  [bts-wr-fresh4]
2026-11-21 01:00 ET  start  wr-w15-carbon (Gemini Carbon high round-1 review of W15 a924d748; pane w2:p4C; clone ~/projects/bts-work/wr-review-w15-carbon; /config checked: telemetry off, non-workspace off, sandbox on, AI credits off)  BTS agents running: 3  [bts-wr-fresh4]
2026-11-21 01:27 ET  hold   ~/bts-runs/backtest-hold kept after the W15 gate for the round-1 reviewers' focused runs (lead's ruling; hard cap 03:00 ET)  [bts-wr-fresh4]
2026-11-21 01:48 ET  hold   ~/bts-runs/backtest-hold removed (W15 round-1 reviewers' focused and mutation runs done; lead's ruling)  [bts-wr-fresh4]
2026-11-21 01:53 ET  hold   ~/bts-runs/backtest-hold set (W15 round-2 RED and focused runs are the next wrapped jobs, after the backtest started 01:49)  [bts-wr-fresh4]
2026-11-21 04:34 ET  hold   ~/bts-runs/backtest-hold removed (W15 stopped after round 2; no further BTS test job planned by bts-wr-fresh4)  [bts-wr-fresh4]
2026-11-21 04:34 ET  stop   wr-w15-codex (W15 round 1 BLOCK, round 2 BLOCK; content filter stopped it twice in round 2 before its marker; archived)  BTS agents running: 0  [bts-wr-fresh4]
2026-11-21 04:34 ET  stop   wr-w15-carbon (W15 round 1 BLOCK, round 2 SIGN; archived)  BTS agents running: 0  [bts-wr-fresh4]
2026-11-21 04:34 ET  stop   wr-w15-author (W15 commits a924d748, d7c66234 on fix/wr-w15; not merged)  BTS agents running: 0  [bts-wr-fresh4]
2026-11-21 04:36 ET  start  bts-wr-fresh5 (fresh Claude session, Remote Control off; pane w2:p4D, session 165e69a5; kickoff ~/bts-runs/kickoff-wr-fresh5-2026-11-21.md: W15b then D3; up to three agents of its own; on the watcher list)  [bts-lead]
2026-11-21 04:37 ET  note   bts-wr-fresh5 estimate: W15b 4-6 h (gate may wait up to one backtest), D3 10-14 h after it; total about 14-20 h  [bts-wr-fresh5]
2026-11-21 04:44 ET  hold   ~/bts-runs/backtest-hold set (W15b gate next on 284dc97a; waits for the running backtest arm=old seed=1277948386)  [bts-wr-fresh5]
2026-11-21 04:54 ET  start  wr-check-d3 (Codex xhigh, memories off, no daemon; D3 contract check 1 of kickoff-fix-deploy-v7.md at 6e76f266; pane w2:p4E, session 01a1fd87; lead's approval of 04:53: no D3 author before W15b ends)  BTS agents running: 1  [bts-wr-fresh5]
2026-11-21 05:10 ET  note   wr-check-d3 check 1 NOT YET (WRCHKD3-0a5f16b5, archived 25dae50f); contract v2 c90a89ff; check 2 sent to the same session  BTS agents running: 1  [bts-wr-fresh5]
2026-11-21 05:23 ET  stop   wr-check-d3 (check 1 NOT YET WRCHKD3-0a5f16b5; check 2 BUILD WITH THESE EDITS WRCHKD3R2-bb07ae25; archived, amendment 1 at 748b2406; tab closed)  BTS agents running: 0  [bts-wr-fresh5]
2026-11-21 06:31 ET  start  wr-w15b-codex (Codex xhigh, memories off, no daemon; W15b review round 1 of fix/wr-w15 at 284dc97a, whole branch; pane w2:p4F, session 01a1fde0)  BTS agents running: 1  [bts-wr-fresh5]
2026-11-21 06:32 ET  start  wr-w15b-carbon (Gemini 4 Carbon High via agy, sandbox on, telemetry off, non-workspace access off, AI credits off; W15b review round 1 at 284dc97a; pane w2:p4G)  BTS agents running: 2  [bts-wr-fresh5]
2026-11-21 07:20 ET  note   W15b round 2 sent to wr-w15b-codex and wr-w15b-carbon at 0ce9e5ee (round 1: Codex BLOCK WRW15BREV1-c33f890f, Carbon SIGN WRW15BCARB1-5e2c92f0)  BTS agents running: 2  [bts-wr-fresh5]
2026-11-21 08:07 ET  stop   wr-w15b-codex (round 1 BLOCK WRW15BREV1-c33f890f, round 2 SIGN WITH EDITS WRW15BREV2-bcd5f739; archived; tab closed)  BTS agents running: 1  [bts-wr-fresh5]
2026-11-21 08:07 ET  stop   wr-w15b-carbon (round 1 SIGN WRW15BCARB1-5e2c92f0, round 2 SIGN WRW15BCARB2-80db5dda; archived; tab closed)  BTS agents running: 0  [bts-wr-fresh5]
2026-11-21 08:07 ET  hold   ~/bts-runs/backtest-hold removed (W15 merged as 812e8880; D3 author next, will reset the hold when its focused tests start)  [bts-wr-fresh5]
2026-11-21 08:08 ET  hold   ~/bts-runs/backtest-hold set (D3 author's focused tests and gate next)  [bts-wr-fresh5]
2026-11-21 08:08 ET  start  wr-d3-author (Codex default effort, memories off, no daemon; D3 author on fix/wr-d3 from 24019a9a in ~/projects/bts-work/wr-fix-d3; pane w2:p4H, session 01a1fe38)  BTS agents running: 1  [bts-wr-fresh5]
2026-11-21 08:27 ET  note   approved wr-d3-author 'uv python install 3.11' (contract F8/tests: real 3.11 check; writes uv's Python store outside the worktree, the contract's one exception)  [bts-wr-fresh5]
2026-11-21 09:22 ET  note   wr-d3-author committed d3d05abd on fix/wr-d3 (11 files, +2816/-146; staged diff read; one docs fix requested before it); gate started on the committed tree  [bts-wr-fresh5]
2026-11-21 10:10 ET  start  wr-d3-codex (Codex xhigh, memories off, no daemon; D3 review round 1 at d3d05abd; pane w2:p4J, session 01a1fea8)  BTS agents running: 2  [bts-wr-fresh5]
2026-11-21 10:11 ET  start  wr-d3-carbon (Gemini 4 Carbon High via agy; sandbox on, telemetry off, non-workspace access off, AI credits off; D3 review round 1 at d3d05abd; pane w2:p4K)  BTS agents running: 3  [bts-wr-fresh5]
2026-11-21 11:42 ET  note   D3 round 1 Codex BLOCK (WRD3REV1-0c037f35); amendment 3 (6cdee0c5) sent to wr-d3-author; Carbon round 1 still running  [bts-wr-fresh5]
2026-11-21 13:09 ET  note   D3 round 1 Carbon BLOCK (WRD3CARB1-6c74cbda; same defects as Codex plus test gaps); amendment 3b (079bd6c4) sent to wr-d3-author  [bts-wr-fresh5]
2026-11-21 13:50 ET  note   wr-d3-author round-1 fix commit on fix/wr-d3 (5 files, +520/-130; staged diff read); gate next  [bts-wr-fresh5]
2026-11-21 13:51 ET  note   D3 round 2 (last) sent to wr-d3-codex and wr-d3-carbon at d57f4013; gate running since 13:51  BTS agents running: 3  [bts-wr-fresh5]
2026-11-21 16:03 ET  stop   wr-d3-codex (round 1 BLOCK WRD3REV1-0c037f35, round 2 BLOCK WRD3REV2-a9fa724e; archived; tab closed)  BTS agents running: 2  [bts-wr-fresh5]
2026-11-21 16:03 ET  stop   wr-d3-carbon (round 1 BLOCK WRD3CARB1-6c74cbda, round 2 BLOCK WRD3CARB2-8c8dbe06; archived; tab closed)  BTS agents running: 1  [bts-wr-fresh5]
2026-11-21 16:03 ET  stop   wr-d3-author (commits d3d05abd, d57f4013 on fix/wr-d3; not merged; D3 STOPPED, STOP-D3.md)  BTS agents running: 0  [bts-wr-fresh5]
2026-11-21 16:03 ET  hold   ~/bts-runs/backtest-hold removed (D3 stopped; no further BTS test job planned by bts-wr-fresh5)  [bts-wr-fresh5]
2026-11-21 16:05 ET  start  bts-wr-fresh6 (fresh Claude session, Remote Control off; pane w2:p4M, session 02177685; kickoff ~/bts-runs/kickoff-wr-fresh6-2026-11-21.md: D3b, then W3 and W4 after the backtests; up to three agents of its own; on the watcher list)  [bts-lead]

## Outcomes by change set in the window
| Change set | Contract checks | Review rounds | Result |
|---|---|---|---|
| Review of one operator instruction the old lead had merged unreviewed (C3v checklist step) | n/a | r1 Codex BLOCK (5 MUST FIX with replacement text); lead applied the reviewer's text; r2 same reviewer SIGN WITH EDITS | landed, docs only, about 10 minutes of reviewer time |
| D, deploy workflow with a durable on-box hold (owner's choice A) | draft 3 NOT YET, draft 4 BUILD WITH EDITS | r1 Codex BLOCK (7) + Carbon BLOCK (4), all fixed; r2 Carbon BLOCK (3), Codex session filtered twice, a fresh plain-worded Codex BLOCK (6); every r2 finding in text parsing of crontab/units, a test override, a re-implemented validator | STOPPED (~5 h) |
| D2, the same hold rebuilt smaller (owner chose "rebuild smaller") | v1, v2, v3 all NOT YET (partial switches, reload-dirty unit state, inherited environments; "needs real systemd/Debian facts") | none | STOPPED at contract stage (~35 min) |
| W15, bad entries in a cache dropped with a health warning | NOT YET, NOT YET, BUILD WITH EDITS | r1 Codex BLOCK + Carbon BLOCK (fd leak, exit code, alert over the DM limit), fixed; r2 Carbon SIGN, Codex BLOCK on two narrow findings (a path in alert text matched an unrelated out-of-memory regex -> false CRITICAL; backticks) | STOPPED, then W15b (a small follow-up on top, fresh reviewers, lead ruling R55): r1 Codex BLOCK (another text path into the same regex), Carbon SIGN; r2 Codex SIGN WITH EDITS, Carbon SIGN | MERGED 11-21 08:05 |
| D3, the small deploy workflow change (owner's option after D2: no automatic rollback, recovery never skipped, page, a contest-state canary) | v1 NOT YET, v2 BUILD WITH EDITS | r1 Codex BLOCK (6) + Carbon BLOCK (3 shared), fixed; r2 Codex BLOCK (3) + Carbon BLOCK (1 shared): a signal in a narrow window stops a healthy scheduler; two places recovery can print "verified" where "unverified" is right; none can send a pick | STOPPED 16:03; lead ruled a small follow-up D3b (R56), fresh session 6 started 16:05 |
| Paired backtest (deploy condition since 11-20 14:05 by owner's word; 16 runs of about 2 h, one at a time in the shared test slot) | n/a | n/a | 11 of 16 done at 16:04 on 11-21 (6 done at 13:27 on 11-20). Idle 06:30 to 16:04 on 11-21 while the D3 gates and reviews held the slot; paused by the other project's engine about 21:15 to 00:50 |

## Other facts
- Owner decisions in the window: 5 (accept the season-replay fault; backtest is a deploy condition; no overrule of R42-R51; D's next attempt "1"; after D2, "3"). Answer delays ranged from about 2 minutes to about 4 hours (the D2 question was asked at 22:40 and answered about 03:00).
- The standing watcher's alert path was tested once in the failure direction (a fake runner-down case reached the lead); its blocked-dialog path has not been seen to fire. No session was found stuck in the window.
- Codex's content filter stopped at least five review turns in the window (deploy-workflow reviews mostly); each continued after one plain prompt or was replaced by a fresh plain-worded session.
- Sessions wrote clock times ahead of the real clock in their records twice (one commit message, one report heading); the lead did once in a message to the owner. All sessions now take times from `date`.
- Round-2 BLOCKs in the window: D (parsing), W15 (alert text), D3 (signal window, stop proof). Every round-2 BLOCK confirmed all round-1 fixes. Two of three stopped pieces were continued as small follow-ups with fresh reviewers; one (W15b) merged.
- Allowances at 16:10 on 11-21: Codex weekly 74% left, the lead's Claude account weekly 36% left (46% at 13:30 on 11-20; resets in about 2 days), the other Claude account 93%.
- Still open before any deploy: D3b (in progress); W3 and W4 (scheduler timing, after the backtests); 5 backtest runs (about 10 hours if the slot is free) and the paired comparison; the closing test (re-run four review harnesses, one blind read, Carbon on two slices, a season walk of every 2027 date); in January, the owner decides on a Debian rehearsal and the full on-box hold.

## Questions
1. Where is this workflow flawed? Rank by expected cost.
2. What are the false greens?
3. What is wasted motion?
4. What one or two changes would you make tomorrow, concretely?
5. Is "stop after round 2, then a small follow-up piece with fresh reviewers" working, or is it an override by another name? What would you change?
6. A delivery estimate for "the whole-range review closed: every finding landed or accepted by the owner, closing test green on the final head", with a date range and the critical path.
Answer in under 70 lines, plain language, most important first.
