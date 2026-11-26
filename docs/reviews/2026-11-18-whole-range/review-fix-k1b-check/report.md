# K1b check at `0b9dd2d2`

## Verdict

**BLOCK** under R64. The new daily command can cost a day's pick when the scheduler's absolute picks path contains a space and the operator inserts it as printed.

## 1. B2 and B3 replays

- **B2, original case fixed.** At the 18:35 ET refusal on July 15, the new command wrote 57/60/used to the absolute `live-picks` directory. Live status accepted the row, `data/picks` stayed empty, and the 18:55 check sent one pick and recorded delivery.
- **B3, both original cases fixed.** The first check produced no predictions in the later-check replay. Filing after the 18:55 refusal recovered the 19:55 pick. In the fallback replay, revocation set aside the row after the 18:25 pick was saved. Filing after the 18:35 fallback refusal recovered the 18:55 pick. Each sent one pick and recorded delivery. [Replay log](replay-round-two-with-review-path.log)

## 2. R64 and ordinary states

- **B2 remaining failure.** The real scheduler read `data/live picks` relative to its service directory. I inserted that directory's absolute path at `DIR` as printed and passed the shell-split arguments to the real CLI between checks. It exited 2 with `unexpected extra argument (picks)`. The live observation stayed absent and the later check sent zero picks. Quoting the same path gave exit 0, one send and a delivered decision. [Directory log](operator-directories.log)
- **Default and relative paths.** The default `data/picks` and a custom relative setting both gave one delivered pick when resolved against the scheduler's working directory. The raw-relative control wrote under the operator's working directory, where the scheduler could not see it.
- **Timing.** The sole refused check had no later run or fallback. Following B8a left the row absent and sent zero picks. At the overnight refusal with a prior entry pending, B8a called for waiting. The CLI rejected the old date and would have accepted the new date in the control. Opening the earlier refusal during the recovered send left state alone and the send completed. The 5a guard refused a saved pick after correction even though status accepted the corrected row. It sent nothing and left the saved file intact. [Timing log](operator-timing.log)
- **Two operators.** The second identical filing left the observation and request log byte-identical. An interleaving where both writers reached publication produced two request rows and one correct observation, with no saver flag or model streak write. Each scenario sent one pick. [Timing log](operator-timing.log) · [Overlapping filing log](overlapping-filing.log)

## 3. Checklist and tests

- `git diff 4c301f7f 0b9dd2d2 -- docs/ops` and a byte-for-byte comparison found only the two approved replacements in B8a. The rest of the checklist matched. `src` had no diff. [Scope log](checklist-scope.log)
- `tests/test_wr_k1_r2.py` passed all 45 tests. Its new CLI test verifies the directory when the path arrives as one argument. `test_k1b_late_filing_covers_every_check_and_fallback` checks the wording, so its name should be read as text coverage. `tests/test_wr_f2_saver_flag.py` passed all 63 tests and still checks the previous B8a hash after reversing the approved edits. The assertions hold. [K1 log](test-wr-k1-r2.log) · [F2 log](test-wr-f2-saver-flag.log)

## Edits

- Replace the daily command and its description with “Run `bts set-contest-streak --date <today> --streak N --best-streak B --saver S --picks-dir "$DIR"` with N, B and S read from MLB. Before running it, set the shell variable DIR to the scheduler's absolute picks directory. Resolve a relative `picks_dir` against the scheduler service's working directory. Check the same directory with `bts season-contract status --picks-dir "$DIR"`.”

## Notes

- The unqualified status calls elsewhere in the B8a cutover steps predate K1b. The replayed 5a refusal and the loss after the last usable check are also inherited.
- The first replay invocation needed the archived helpers on `PYTHONPATH`. The rerun passed. The host rejected the `nice` adjustment and runs continued. The review left tracked files clean.

ran: real CLI and scheduler replays with a pinned ET clock and offline MLB, prediction and DM fixtures, checklist diff and byte comparison, both requested test files; did not run: live MLB or Bluesky, the full suite, or a gate.
