# Kickoff: change set E, operator texts (version 1, 2026-11-24)

Ruling R60 (`FINDINGS.md`). Tier A because it changes what an operator does; reviewed per sentence under R51: each sentence the operator can read is checked for "what if he does exactly this in the worst ordinary state". Source findings: `CLOSING-TEST.md` (SW1's text, BL-N3, C2-B1's text, BL-B4, BL-B5, BL-B6, BL-text, BL-B3, C2-O1/BL-N4), with the blind reviewer's operator table `closing-test/codex-blind/operator-text.md`.

**Base:** branch `fix/wr-e`, cut from `63669c42` (`main` `52336a5a` plus the season-walk merge). Commit 1 on the branch (the session's, docs only) holds this contract and the documentation rows D1 to D12. The author adds ONE commit with the code rows E1 to E3 and their tests.

**What E is not.** It changes no decision, no pick, no alert level, source, key, dedup rule or timing, no file the pick path reads, and no policy lookup (change set G does that, afterwards). It does not touch `src/bts/cli.py`, `src/bts/day_outcome.py`, `src/bts/health/`, `scripts/cron-setup-hetzner.sh` or `.github/workflows/` (change set F and the January deploy work own those).

## Code rows (the author)

Placeholders in braces are the values the code already has at that point. "Old" is `63669c42`.

### E1. The skip notice (SW1's text, BL-N3): `scheduler.build_skip_summary`, `scheduler.format_skip_dm`, the `SKIP —` journal line in `scheduler.run_single_check`, `web.render_skip_banner`

Today the summary names the highest finite prediction of ALL predictions, even one a status update removed before the policy decided (BL-N3), and all three renderings say that candidate is "below" a bar whenever the action is a skip, although the shipped table skips some probability ranges above its bar (1,511 states, R61) and, with no bar, says "below the ~80% bar" or "below the pick bar" (SW1).

**The summary.** `build_skip_summary(predictions, streak, pick_bar=None, decision=None, candidate=None)`:
- It returns `None` in exactly the states it does today (the same checks on `predictions`, evaluated first, and the same catch-all). Whether a summary exists is behaviour (`_maybe_alert_missed_pick`, the once-a-day notice): it must not change in any state.
- `best_batter`, `best_team`, `best_p`: when `decision` is not None and `candidate` is a mapping whose `p_game_hit` is a finite number, they are the candidate's `batter_name`, `team` and `p_game_hit` (the selection's `primary_candidate`: the row the policy decided on). Otherwise as today (the highest finite prediction).
- A new key `basis`: `"decision"` when `decision` is not None, else `"no_decision"`. Every other key (`streak`, `bar`, `objective`, `reason`) is exactly as today. JSON-native types only.
- `run_single_check` passes `candidate=sel.primary_candidate` when `sel` is not None. Nothing else at the call site changes (no second state read, the same `streak` and `pick_bar` sources).
- `strategy.effective_pick_bar` returns exactly what it returns today in every state. E does not change it; a test proves it over the whole shipped table (below).

**Which wording.** One function decides the case from the summary, and all three renderings use it:
- K1 `reason` is set (the tail stop rule).
- K5 `basis` is `"no_decision"`.
- K2 `bar` and `best_p` are both finite and `best_p < bar`.
- K3 `bar` and `best_p` are both finite and `best_p >= bar`.
- K4 otherwise (no finite bar, or no finite `best_p`).
A summary without `basis` (written by the previous code and read back from `scheduler_state.json`, or built by hand in a test) is treated as `"decision"`.

**The DM (`format_skip_dm`).** `{b}`, `{t}`, `{p}` are `best_batter`, `best_team`, `best_p`; `{s}` is `streak`.

| Case | Old sentence(s) | New sentence(s) | Worst state if the operator acts on it |
|---|---|---|---|
| K1 | `BTS {date}: No pick — model's best is {b} ({t}) {p:.1%}; {reason}. Streak holds at {s}.` | `BTS {date}: No pick — the best eligible candidate is {b} ({t}) {p:.1%}; {reason}. Streak holds at {s}.` | A status update removed a stronger batter before the decision: the old text named him. New text names the one decided on. No action is invited |
| K2 | `BTS {date}: No pick yet — model's best is {b} ({t}) {p:.1%}, below the streak-{s} bar of {bar:.1%} (streak holds at {s}). Will pick if a confirmed lineup clears it.` | `BTS {date}: No pick yet — the best eligible candidate is {b} ({t}) {p:.1%}, below the streak-{s} bar of {bar:.1%} (streak holds at {s}). A later lineup check decides again.` | "Will pick if a confirmed lineup clears it" is false in the 1,511 states (a candidate that clears the bar can still be skipped), and on the day's last check. The new last sentence promises nothing |
| K3 | the K2 sentence (false: the candidate is not below the bar) | `BTS {date}: No pick yet — the policy skips at streak {s} although the best eligible candidate, {b} ({t}) {p:.1%}, clears the streak-{s} bar of {bar:.1%}: its table skips this probability range in this state (streak holds at {s}). A later lineup check decides again.` | He may think it is a fault and enter the candidate himself: the text states that the skip is the table's, and invites nothing. After change set G this case cannot arise from the shipped table; it must still be true if it does |
| K4 | `BTS {date}: No pick yet — model's best is {b} ({t}) {p:.0%}, below the ~80% bar (streak holds at {s}). Will pick if a confirmed lineup clears it.` | `BTS {date}: No pick — the policy skips at streak {s} and there is no probability bar for this state: either it plays no candidate here whatever the probability, or the bar could not be computed (best eligible candidate: {b} ({t}) {p:.1%}). Streak holds at {s}. A later lineup check decides again.` | Every bin skips (streak 56 with 74 or more days left), or `effective_pick_bar` failed and returned None: the sentence is true in both, before and after G. He waits for a pick that will not come that day: the streak holds, which is the safe side |
| K5 | the K2 or K4 sentence (false: no decision was made; the named batter may be ineligible and above the bar) | `BTS {date}: No pick yet — no candidate could be chosen at this check (none eligible with a usable prediction, or game statuses could not be read); the model's top prediction is {b} ({t}) {p:.1%}. Streak holds at {s}. A later lineup check decides again.` | He may enter the named batter himself although his game is closed or postponed: the text says no candidate was eligible, and invites nothing. When `streak` is None the sentence `Streak holds at {s}.` is omitted |

A non-finite `best_p` renders as `?` in place of the percentage.

**The journal line.** Old: `  SKIP — best {b} ({t}) {p:.1%} {below}; streak holds at {s}.` with `{below}` = the reason, or `below the streak-{s} bar ({bar:.1%})`, or `below the pick bar`. New, same prefix and ending, `best eligible` in place of `best` in K1 to K4, and `{below}` =
K1 the reason; K2 `below the streak-{s} bar ({bar:.1%})`; K3 `clears the streak-{s} bar ({bar:.1%}); the policy table skips this probability range in this state`; K4 `skipped with no probability bar for this state`; K5 the whole line is `  SKIP — no candidate could be chosen at this check (none eligible with a usable prediction, or game statuses unreadable); top prediction {b} ({t}) {p:.1%}; streak holds at {s}.`

**The dashboard banner (`render_skip_banner`).** Old: `No pick — model's best is {name} ({team}) {pct}, {below}. Streak holds at {streak}.` New, inside the same markup, with the same HTML escaping of every value read from the summary:
K1 `No pick — the best eligible candidate is {name} ({team}) {pct}, {reason}. Streak holds at {streak}.`
K2 `No pick — the best eligible candidate is {name} ({team}) {pct}, below the streak-{streak} bar ({bar:.1%}). Streak holds at {streak}.`
K3 `No pick — the policy skips although the best eligible candidate, {name} ({team}) {pct}, clears the streak-{streak} bar ({bar:.1%}): its table skips this probability range in this state. Streak holds at {streak}.`
K4 `No pick — the policy skips and there is no probability bar for this state (best eligible candidate: {name} ({team}) {pct}). Streak holds at {streak}.`
K5 `No pick — no candidate could be chosen at the last check (none eligible with a usable prediction, or game statuses unreadable); the model's top prediction is {name} ({team}) {pct}. Streak holds at {streak}.`
When `streak` is None, K2 and K3 say `the bar` in place of `the streak-{streak} bar`, and the last sentence is omitted. An empty or missing summary still renders the empty string.

**None of the new renderings may contain the words `below the` unless the case is K2** (the season walk's check looks for them).

### E2. The two stale-reading warnings (C2-B1's text, BL-B4): `scheduler._warn_stale_contest_reading`, `scheduler._morning_state_check`

Only the message strings change. The journal lines, the alert level, source, incident key, status files, `now_et_date`, the exception containment (BaseException after a delivery, Exception in the morning) and every condition are untouched. `{D}` is the date the code already prints (`warning_day` after a delivery, `date` in the morning).

| # | Old sentence | New sentence | Worst state if he does exactly this |
|---|---|---|---|
| S1 | `The pick was delivered using an automatic contest reading flagged stale ({detail}).` | unchanged | |
| S2 | `Check MLB.` | unchanged | |
| S3 | `If the streak or saver differs, use bts set-contest-streak --date {D} --streak N --best-streak B --saver S with the scheduler's picks directory and the values shown in MLB.` | `If the streak or saver shown there differs, bts set-contest-streak --date {D} --streak N --best-streak B --saver S (the scheduler's picks directory, the values shown in MLB) records it for {D} ET only; it does not change the pick already sent or the saver flag.` | He runs it on {D} with MLB's values after the pick went out: the observation is accepted for the rest of {D}; the delivered pick is not re-chosen, re-sent or refused (the contract check must confirm that nothing later that day refuses, re-sends or pages because of it). He mistypes a value: only {D}'s later reads see it |
| S4 | `Use the current ET date when you run it.` | `The command is refused on any other date: do not change its date after midnight ET. For the next day, wait until MLB has settled this pick, compare MLB with bts season-contract status, and file that day's dated observation before its first lineup check only if they differ.` | Old: read at 00:05 while the slate's game is still live, he re-dates the command and installs a pre-settlement streak for the new day, which then wins over a correct fetch (BL-B4). New: the literal date is refused after midnight; the next-day step is conditioned on settlement and on a difference. He files a correct next-day observation that equals the automatic reading: harmless |
| S6 | (none) | `If MLB shows the Streak Saver used and bts saver-state (same picks directory) does not say used, also run bts saver-state --use, or later automatic days will count a saver that is gone.` | Flag `active` or `not_earned`: it becomes `used` (from `not_earned` the command prints how to take it back). Flag `used`: he does not run it; if he does, `Saver is already used.` Flag `uninitialized`: exit 1 and the command's own instruction (`--init used --season Y`), which is right for that state. He runs it in the wrong directory: the default directory's flag reads `uninitialized` and the same exit 1 follows, the real flag is untouched (the words "same picks directory" are the guard). He misreads MLB and marks a saver used that is not: the policy plays as if it were spent until `--undo`; no pick is sent on a saver that does not exist. No later fetch can turn `used` back (`maybe_auto_earn_saver` never overwrites `active` or `used`) |
| S5 | `Check the latest fetch status.` | unchanged, last | |
| M1 | `Morning check for {D}: the automatic contest reading was stale at startup ({detail}).` | unchanged | |
| M2 | `This reading is eligible if it remains in use.` | unchanged | |
| M3 | `Check MLB.` | unchanged | |
| M4 | `If state still differs, use bts set-contest-streak --date {D} --streak N --best-streak B --saver S with the scheduler's picks directory and the values shown in MLB.` | `If state still differs, use bts set-contest-streak --date {D} --streak N --best-streak B --saver S with the scheduler's picks directory and the values shown in MLB; it applies to {D} ET only, is refused on any other date, and does not change the saver flag.` | Read the next day: the literal date is refused. Read at 10:00 on {D}: as today |
| M6 | (none) | the S6 sentence | as S6 |
| M5 | `Check the latest fetch status.` | unchanged, last | |

Each whole DM, with its envelope (`BTS health WARN attention:\n- [contest_state] `) and the longest detail the code can produce, stays under 900 characters (a test asserts it).

### E3. The two refusal DMs (BL-B5): `scheduler._refuse_late_delivery`, `scheduler._refuse_void_game_delivery`

Only the last sentence of each message changes; the journal lines, the archive, the state record, the alert sources (`late_delivery`, `postponed_pick`) and the status file are untouched.

| # | Old sentence | New sentence | Worst state if he does exactly this |
|---|---|---|---|
| L1 | `LATE DELIVERY REFUSED: {names} would have been sent {m} min after the {cutoff} cutoff.` | unchanged | |
| L2 | `Nothing delivered; the scheduler will re-pick from later games if any remain.` | `Nothing delivered; the scheduler will re-pick from later games if any remain, and a pick message then names the pick to enter.` | No later pick comes (no later check, a failed prediction: the accepted W3, W4 and 5b limits): the day has no pick and the streak holds; the missed-pick DM and the 08:30 report are the backstops |
| L3 | `Enter a pick manually if you want one now.` | `A pick entered in MLB by hand now can be contradicted by that later message; the entry check would then report a mismatch.` | Old: he enters another batter at once, the scheduler sends a different pick 20 minutes later and the entry check says "Fix it now!" (BL-B5). New: no invitation; if he enters one anyway he has been told what follows |
| P1 | `DELIVERY REFUSED for {date}: {games}.` | unchanged | |
| P2 | `Nothing delivered; the scheduler will re-pick from other games if any remain.` | `Nothing delivered; the scheduler will re-pick from other games if any remain, and a pick message then names the pick to enter.` | as L2 |
| P3 | `Enter a pick manually if you want one today.` | the L3 sentence | as L3 |

## Documentation rows (commit 1; the diff of `docs/ops/2027-season-start.md` and `INCIDENT.md` against `63669c42` is the old and new text, sentence by sentence)

| # | Where | What was wrong | What it says now | Worst state checked |
|---|---|---|---|---|
| D1 | checklist A5 | Read alone, it sets DM delivery with neither contest-state setting (`bts schedule` then exits 2) | Points to B8c and B8d for the order; the two settings first | A5 followed alone on a restored box |
| D2 | checklist A7 | "Revert = private and comment out the cron line" leaves `entry_intent = "enter"`: the installer and `bts entry-intent` refuse, and 08:30 pages an invalid intent | Private AND research together, the installer re-run, restart in a sleep window; what the next 08:30 says about a pick delivered that day | A revert in the afternoon after a delivered pick |
| D3 | checklist B4 | The drill moved a live `decision.json` aside: with the entry check running, that day's receipt is left unbound | The drill runs on a copy of the picks directory | The drill run during the entry window of a delivered pick |
| D4 | checklist B8a, lead-in and bullets 1 to 4 | "Flags before deploy" (B8a) against "deploy before flags" (B8c) | Each bullet names its step of B8c; the two settings before the deploy, DM and enter after it | Read in either order |
| D5 | checklist B8a, the stale-DM line | Same faults as E2 (one day only; the saver flag; re-dating after midnight); "cookies" with no rule | Matches E2 and the CLI's own back-off messages | As S3, S4, S6 |
| D6 | checklist B8b (1) | Describes the wall-clock judgment that C1 replaced | C1 as landed: the alert names the slate and prints no command; verify settlement first; a quiet morning proves nothing | Acting at 00:20 on the alert, or on the status command, before settlement |
| D7 | checklist B8b (3) | "Must not be allowed to roll back … (Eric's decision pending, W12)": the workflow rolls back by itself and the decision was made (deferred to January) | What the workflow does, when its rollback is safe, what to read before a deploy and after a failed one | A failed canary on the first deploy; a failed sync |
| D8 | checklist B8c | Five steps that the workflow cannot follow (it starts both services during step 2) and that put the settings after the deploy | The one order in seven steps, each possible with the unchanged workflow; a failed first deploy cannot leave October code able to send (delivery stays private until step 5) | A canary failure at step 3; a job that stops before the canary; a restore on a game day (then B8d) |
| D9 | checklist, new row B8d | Nothing said when delivery may be switched; a switch onto a slate with a private lock makes the entry check tell him to enter a pick chosen on last season's state (BL-B3) | Before the first slate; never onto a day with a private lock; what to do if it happened | The switch made at 15:00 on a day the private scheduler already locked |
| D10 | checklist, new row B8e | Nothing covered the leaderboard unit; its timer is `Persistent=true` and it logs in with the real account (C2-O1, BL-N4) | Disable and verify at first boot; never the installer; re-enabling is Eric's alone, after cookies and the unit file | The timer enabled in the snapshot and already fired; the operator runs the installer to "fix" the unit file |
| D11 | `INCIDENT.md` §1, §2, §4, §5 | "A bad deploy is self-healing"; "canary + auto-rollback protect you"; a redeploy as a dashboard restart; "known-good SHA" with no season bound | What the rollback does and does not cover (W12, W13, W17 as they stand); which commit a rollback lands on; no redeploy as a dashboard restart while a pick is in flight; "known-good" means this season | A cloud session ships a fix to a restored box; a redeploy at 18:20 on a game day; a blunt rollback to an October commit |
| D12 | `INCIDENT.md` §3 | "Re-capture cookies" on any auth signal, against the CLI's own messages (back off on a rate limit; wait on a transient failure) | Three cases keyed to the words of the failure message; re-capture only on an expired login, on the Mac as the CLI says | A 429 at login followed by a re-capture |

## Must still happen (prove each on the committed tree)
1. `build_skip_summary` returns None for exactly the inputs it does on the base, over a corpus that includes every unusable shape the base handles (no columns, all NaN, inf, empty frame, a raising object); for every other input the keys `streak`, `bar`, `objective`, `reason` are equal to the base's.
2. `run_single_check`'s return value is equal to the base's in every field except the summary's `best_*` values and `basis`, over the existing skip-visibility and tail-objective fixtures.
3. `effective_pick_bar` over every (streak 0 to 56, days left 1 to 180, saver) of the shipped table, both regimes: equal to the base. `strategy.py` has no diff, or a docstring-only diff.
4. `maybe_notify_skip` still sends at most once a day and never raises; `_maybe_alert_missed_pick`'s skip gate sees a summary in exactly the base's states.
5. A `scheduler_state.json` written by the base (summary without `basis`) loads and renders; one written by this commit loads under the base's `load_state` (extract the base's module with `git show 63669c42:src/bts/scheduler.py` into a temporary package, or show by reading that the loader does not validate the summary's keys, and say which).
6. E2: the stale warning is still sent exactly when the base sends it, once per ET date across the morning and delivery routes, with the same key and status file; an interrupt inside the delivery-side warning still cannot stop the send or change the return; the 4c and 6b-1 test files pass with only their expected-text helpers changed.
7. E3: both refusals still archive, record, clear the pick and page under their own sources; the EOD `late_delivery` check is untouched.
8. The season walk, run end to end on the commit: the same 4,599 runs, every action, bar and day count equal to `results/walk-bc143131.json.gz`; no "skip message" problem; the 841 formerly flagged runs now render K3 or K4. `walk.py` passes `candidate=` the way `run_single_check` does (the only change to the tool), and its known-answer test for a false "below" still fails when the false text is planted.

## Standing rows
Say in the handover how each was tested or why it cannot apply: an interrupt (`KeyboardInterrupt`, `SystemExit`) while a message is being built or sent; a malformed or non-regular file (the summary read back from a damaged `scheduler_state.json`; a summary holding strings, NaN, None or nested objects where numbers are expected: no rendering raises); midnight ET and the year boundary (the date in each message is the one the base prints; December 31 to January 1 for `{D}`); a failed or unconfirmed send (a failed skip DM is retried next cycle as on the base; a stale warning whose send raises is contained as on the base); a restart mid-step (the summary and `skip_notified_at` are inherited as on the base).

## Tests (first, RED on `63669c42`)
New file `tests/test_wr_e_texts.py`: every sentence in the tables above asserted as an exact string through the real functions (not through copies of the templates); BL-N3's case through the real `run_single_check` with a status update that removes the top batter; SW1's case through the real decision at a streak where the shipped table skips its top bin, and at streak 56 with 100 days left (every bin skips); K5 through a real `no_eligible` selection; the 900-character bound; an operator-text scan: no string literal in `src/bts/scheduler.py` contains `manually if you want`, `Use the current ET date`, `below the ~80% bar` or `below the pick bar`, and `src/bts/web.py` contains none of the last two. Existing assertions that may change, and only these: the expected-text helper in `tests/test_stale_delivery_cs4c.py` (`message`), the one in `tests/test_morning_state_cs6b1.py` (`warning`), and the text assertions on `format_skip_dm`, the journal line and `render_skip_banner` in `tests/test_scheduler_skip_visibility.py` and `tests/test_scheduler_tail_objective.py`. If any other existing assertion would have to change, STOP and ask.

## Known and not fixed (ledger; do not build)
- A K5 summary still sets `state.skip_summary`, which silences the missed-pick alert for that day and can spend the day's one skip notice before a real skip (base behaviour, kept by row 1).
- The non-live missed-pick DM still says `Deliver manually if you want to keep the streak alive` (private or research configurations only).
- `bts saver-state --use` just after midnight on January 1 acts on the new year's flag (BL-N1).
- The reverse saver disagreement (the flag says used, MLB shows it available) has no message.
- Other user timers of the restored snapshot (lineup collection, live-forward capture and resolve, shadow prediction) run October code until the first deploy; B8c covers the scheduler, the dashboard, cron and the leaderboard timer only.

## Handover
`HANDOVER-WR-E.md` in the worktree root (not committed): the commit; files and lines; for each sentence the test that pins it and whether it was RED on the base; the must-still-happen rows with what was run and the counts; the standing rows; anything that could not be met as written (then STOP and say so). Then print `WRE-<first 8 hex characters of the sha256 of HANDOVER-WR-E.md>` on its own line and `DONE`.

# Version 2 (2026-11-24 19:28, after the contract check `contract-check-e/contract-check.md`, `WRCHKE-2dd1d06c`: BUILD WITH THESE EDITS)

**The check's replacement texts ARE the contract.** Every paragraph of `contract-check-e/contract-check.md` marked MUST FIX replaces the matching part of version 1 above (the E1 summary rules, case rules, surface rules and K1 to K5; S3, S4, S6, M4, M6 and the E2 scope and length rule; L2, L3, P2, P3; must-still-happen rows 1 to 6 and 8; the permitted test changes and the operator-text scan). Version 1 remains for the "old" column, the worst-state reasoning and everything the check did not replace (row 7, the standing rows, the ledger, the handover). The documentation rows (D1 to D12 and the check's additions for checklist A4 and B8) are already applied in commit 2 of the branch with the check's texts; the author does not touch `docs/ops/2027-season-start.md` or `INCIDENT.md`. Where the check leaves a composition open, this section decides:

## E1, composed
- **Summary** (`build_skip_summary(predictions, streak, pick_bar=None, decision=None, candidate=None)`): the check's first E1 paragraph, verbatim. Keys added: `basis` (`"decision"` or `"no_decision"`) and `candidate_verified` (bool). `run_single_check` passes `candidate=sel.primary_candidate` when `sel` is not None.
- **Case order** K1, K5, K2, K3, K4, with the check's display parser (second E1 paragraph).
- **Sentences** (`{candidate}` is `Candidate used in the decision` when `candidate_verified` is true, else `Top model prediction (eligibility unverified)`; `{p}` is the probability to one decimal as a percentage, or `?`; `{bar_text}` is `the streak-{s} bar ({bar:.1%})` with a valid streak, else `the bar ({bar:.1%})`):
  - K1 `{candidate}: {b} ({t}) {p}. Reason: {reason}.` (exactly one terminal period: none is added when the reason already ends with `.`, `!` or `?`)
  - K2 `{candidate}: {b} ({t}) {p}, below {bar_text} using unrounded values.`
  - K3 `{candidate}: {b} ({t}) {p}, at or above {bar_text}.`
  - K4 `{candidate}: {b} ({t}) {p}. No probability comparison is available for this check.`
  - K5 `No candidate was chosen. Top model prediction (eligibility unverified): {b} ({t}) {p}.`
- **Surfaces.** DM: `BTS {date}: No pick at this check. ` + sentence + suffix. Journal: `  SKIP — ` + sentence + suffix, one line. Banner: `<div class="skip-banner"><strong>NO PICK AT LAST CHECK</strong><span>` + sentence + suffix + `</span></div>`, every interpolated value HTML-escaped. Suffix: ` Streak used at this check: {s}.` only for a valid streak (a JSON integer, not a boolean, not negative). Line breaks inside a displayed string become single spaces on all three surfaces.

## E2, composed
`DIR` is a literal placeholder in the messages. The delivery warning's body is used for both its DM and its `STALE CONTEST READING` journal line.
- **After a delivery:** `The pick was delivered using an automatic contest reading flagged stale ({detail}). Check MLB. If MLB differs from status (B8a), use bts set-contest-streak --date {D} --streak N --best-streak B --saver S --picks-dir DIR. DIR is the absolute scheduler picks path. The sent pick and saver flag stay as they were. Write only if the pick's slate and today's ET date both equal {D}. For a later date, follow B8b(1). On {D} ET away from midnight, if MLB shows this season's saver used, check bts saver-state --picks-dir DIR. For active or not_earned, run bts saver-state --use --picks-dir DIR. For uninitialized, see B8a. Check the latest fetch status.`
- **In the morning:** `Morning check for {D}: the automatic contest reading was stale at startup ({detail}). This reading is eligible if it remains in use. Check MLB. If MLB differs from status (B8a), use bts set-contest-streak --date {D} --streak N --best-streak B --saver S --picks-dir DIR on {D} ET only. DIR is the absolute scheduler picks path. This leaves the saver flag unchanged. On {D} ET away from midnight, if MLB shows this season's saver used, check bts saver-state --picks-dir DIR. For active or not_earned, run bts saver-state --use --picks-dir DIR. For uninitialized, see B8a. Check the latest fetch status.`
- Each dispatched body with its real envelope is under 900 Unicode characters at the longest detail the code can produce (the check measured 881 and 842); if your measurement differs, STOP and say so, do not shorten a sentence yourself.

## E3, composed
- **Late:** `LATE DELIVERY REFUSED: {names} would have been sent {m} min after the {cutoff} cutoff. This pick was withheld. A later check may produce another recommendation. Use a later official pick message only while the entry window is open. A manual entry can differ from a later recommendation. The entry check may report a mismatch if it verifies the entry in time.`
- **Postponed:** `DELIVERY REFUSED for {date}: {games}. This pick was withheld. A later check may produce another recommendation. Use a later official pick message only while the entry window is open. A manual entry can differ from a later recommendation. The entry check may report a mismatch if it verifies the entry in time.` (Corrected 19:31: version 2 first said "the same four sentences"; there are five, the same five as in the late message.)

## Scope and tests
- Files the author may change: `src/bts/scheduler.py` (the functions named in E1 to E3 and the one call site), `src/bts/web.py` (`render_skip_banner`), `scripts/audit/season_walk/walk.py` (pass `candidate=`; the K3 and K4 counts) and its test file, the new `tests/test_wr_e_texts.py`, and text-only expectations in `tests/test_scheduler_skip_visibility.py`, `tests/test_scheduler_tail_objective.py`, `tests/test_stale_delivery_cs4c.py`, `tests/test_morning_state_cs6b1.py` as the check's last paragraph permits. `src/bts/strategy.py` has no diff.
- The season walk on the commit: 4,599 runs equal to the saved records in action, bar and day counts; no skip-message problem; 229 K3 and 612 K4 among the 841 formerly flagged runs. Save the result as `scripts/audit/season_walk/results/walk-e.txt` and `.json.gz` (the commit id is not known before the commit).
- The must-still-happen and standing rows as the check rewrote them; say for each what was run.

## Amendment 1 (2026-11-24 19:33; the author stopped and asked, as instructed)
E1 changes which candidate a summary names, so an existing assertion on the `best_batter`, `best_team` or `best_p` of a summary returned by `run_single_check` may change to the selection's `primary_candidate` when the test's selection carries a valid one that differs from the top prediction. Known case: `tests/test_scheduler_tail_objective.py::test_run_single_check_skip_summary_comes_from_the_selection` (`best_batter` `Steven Kwan` becomes `B1`; its streak, reason and bar assertions stay). Any other assertion of this exact kind in the four named test files may change the same way; list each in the handover with the value before and after. Fixtures are not edited to avoid the change. Everything else in the permitted list is unchanged, and assertions on whether a summary exists, on `streak`, `bar`, `objective` or `reason`, or on any other field of the return, may NOT change.

## Amendment 2 (2026-11-24 20:48, after round 1: Codex BLOCK `WREREV1-2f4ce4d0`, `review-fix-e-codex/report.md`; Carbon BLOCK `WRECARB1-45cfd488`, `review-fix-e-carbon/report.md`)
Both reviewers confirmed the composed texts character for character, that nothing but words changed against the base, the renderers' behaviour on bad data (one case excepted), the season walk and the tests. Fourteen findings, each with a replacement sentence. The document findings are applied by the session in the commit that carries this amendment (Codex B1's checklist side, B3 to B8; Carbon B1 to B5 and notes N2, N3; Carbon B1's claim that the unit installer replaces a mask was confirmed by reading the installer and the templates' origin, its unlink behaviour was not reproduced on macOS, whose `install` refuses a `/dev/null` link; Carbon B3 and B4 confirmed by reading `posting.py` and `capture_bts_cookies.py`). Where both reviewers gave a text for the same sentence, Codex's is used and Carbon's condition added.

**The round-2 author** makes ONE new commit on top of the branch head (no amend), tests first:
1. **Codex B1 (the morning warning).** In `_morning_state_check` the sentence `If MLB differs from status (B8a), use bts set-contest-streak --date {D} --streak N --best-streak B --saver S --picks-dir DIR on {D} ET only.` becomes `If prior entries are settled or absent and MLB differs from status (B8a), use bts set-contest-streak --date {D} --streak N --best-streak B --saver S --picks-dir DIR on {D} ET only.` Nothing else in that message changes; the delivery message is unchanged. The length bound (under 900 with the envelope at the longest detail) still holds: measure it, and STOP if it does not.
2. **Codex B2 and Carbon B6 (impossible numbers).** The display parser treats a probability or a bar outside the closed range 0 to 1 as missing (so `?` and the K4 wording apply, as for any missing number). `build_skip_summary` is not changed. Test with `1e308`, `1.5e308`, `-0.1`, `1.0000001`, `2`, and the edges `0`, `0.0`, `1`, `1.0` (valid).
3. **Carbon N4.** A `reason` that is empty after line breaks are turned into spaces and the ends trimmed is treated as no reason (no `Reason: .`).
4. **Tests for the scratch changes no test noticed** (both reports, items 6 and N1): a summary with both a reason and `basis == "no_decision"` (K1 wins); a reason with a finite bar (K1 wins); the complete journal line asserted for K1, K3 and K4 through the real `run_single_check`; the unverified label asserted inside a K2 sentence; HTML escaping of `best_team` and of `reason` in the banner.
5. Rename the two tests whose names say a legacy summary "keeps old wording" to say what they assert; correct the comment above the summary call in `run_single_check` that says a skip means the best candidate is below the pick bar.
6. Files: `src/bts/scheduler.py`, `tests/test_wr_e_texts.py`, `tests/test_morning_state_cs6b1.py` (its expected-text helper), and the two skip test files for item 5. No document, no other source file. Handover `HANDOVER-WR-E-R2.md` (not committed): the commit, the diff stat against `db52bd02`, per item what changed and which test was RED first, the measured lengths; then `WRE2-<first 8 hex of its sha256>` on its own line and `DONE`.
