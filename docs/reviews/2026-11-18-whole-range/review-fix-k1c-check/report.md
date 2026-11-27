# K1c check at `2883190b`

## Verdict

**SIGN WITH EDITS.** The space case is fixed. Make the shell setup explicit and guard against an empty or unset `DIR`.

## 1. Space in the directory

- `replay_space.py` ran B8a's commands through the installed `bts` in `/bin/sh` and `zsh` from a separate operator directory. The scheduler used `data/live picks` relative to its service directory and refused at 18:35 ET. At that point there was no pick DM or live observation.
- Filing between checks wrote `57/60/used` to the live directory. Both CLI commands exited 0 and status accepted the observation. At 18:55 ET the scheduler sent one pick DM and recorded `delivered` in each shell. [Replay log](replay-space.log)

## 2. R64 and directory checks

- Both shells delivered the expected pick with default `data/picks`, a temporary mirror of the box path `/home/bts/projects/bts/data/picks`, and a path containing spaces, an apostrophe, double quotes and `$`. The CLI wrote only the dated observation and its request log in the service directory. The refusal alert reached the offline transport in each replay. [sh paths](directory-edges-sh.log) · [sh box](directory-edges-sh-rest.log) · [zsh paths](directory-edges-zsh-paths.log) · [zsh box](directory-edges-zsh-status.log)
- Unset, empty and other-shell `DIR` all filed in the operator's current directory. Resolving a relative setting against the operator's directory and expanding a literal `$` during assignment also filed in the wrong place. In every control, filing and status both exited 0, status accepted the wrong directory, the scheduler's directory remained refused, and the day got no pick. [sh controls](directory-edges-sh.log) · [zsh setup](directory-edges-zsh-setup.log) · [zsh paths](directory-edges-zsh-paths.log)
- I read the requirement to set `DIR` to the scheduler's absolute directory as applying to the shell that expands the command. All observed day losses skipped that prerequisite. The exact replays produced the expected contest record, alert and pick, so the new text has no R64 block. Both shells rejected an unset `DIR` when the proposed guard was present and delivered with the guarded commands after reading the literal path. [Replacement log](replacement.log)

## 3. Commit scope and tests

- `git diff 0b9dd2d2 2883190b` changes the daily sentence, updates its test and adds 13 prior-review files. The checklist matched byte for byte after the earlier review's exact replacement. `src/` and `tests/test_wr_f2_saver_flag.py` had no diff. [Scope log](scope.log)
- `tests/test_wr_k1_r2.py` passed 45 tests. Its CLI test really expands `live picks` in `/bin/sh`, prints NUL-delimited arguments and passes them to Click's `CliRunner`. The full shell launch of `bts` is covered by the separate replay. `tests/test_wr_f2_saver_flag.py` passed 63 tests. [K1 tests](test-wr-k1-r2.log) · [F2 tests](test-wr-f2-saver-flag.log)

## Edits

- Replace the daily command and its description with the following text.

  Resolve a relative `picks_dir` against the scheduler service's working directory. In the shell where you will run both commands, run `IFS= read -r DIR`, paste the scheduler's absolute picks directory as it appears on disk, and press Enter. Run `bts set-contest-streak --date <today> --streak N --best-streak B --saver S --picks-dir "${DIR:?Set DIR in this shell}"` with today's ET date and N, B and S read from MLB. Check the same directory with `bts season-contract status --picks-dir "${DIR:?Set DIR in this shell}"` and confirm the dated row and `live decision now: accepted`.

## Notes

- The older unquoted B8a status example exited 2 when given a literal spaced path in either shell. Bare status returned an automatic all-clear from another directory while live status refused. These older status calls performed no writes. B8a's pre-existing claim that the refusal names the filing command is also inaccurate. The DM directs the operator to B8a for live repair. [sh status](directory-edges-sh-rest.log) · [zsh status](directory-edges-zsh-status.log)
- The initial sh edge run reached its time limit after seven cases. The remaining three passed in a separate run. The host rejected the priority adjustment and the commands continued. Tracked files stayed clean.

ran: installed CLI and scheduler replays with a fixed ET clock and offline MLB, prediction and DM fixtures, directory controls, commit comparison and both requested test files; did not run: live MLB or Bluesky, the actual box, the full suite or a gate.
