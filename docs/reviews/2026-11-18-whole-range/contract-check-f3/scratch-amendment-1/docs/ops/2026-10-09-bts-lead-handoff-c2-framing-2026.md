# BTS lead handoff (2026-10-09): the C2 framing 2026 test, code phase stopped after a third BLOCK

You are the next BTS lead. The previous lead, bts-lead2 (pane `wA:pQ`, session `5f91aea8…`), stopped the code phase under the herdr manager's rule: a BLOCK in the fresh reviewer's round 1 was the third BLOCK on this code in one session, so a fresh lead takes it over.

**Read first, in this order:**
1. `CLAUDE.md`.
2. This brief.
3. The STATE line in `~/projects/claude-shared/memory/bts_index.md`.
4. The frozen design `docs/sota_audit/2026-10-09-prereg-c2-framing-2026-test.md`, frozen at `9b344d9`.
5. Register rows `C2-framing-2026-test`, `C2-framing-2026-design-r1` and `C2-framing-2026-design-frozen` in `docs/audit/2026-09-22-exposure-register.md`.
6. Row (e) of `docs/sota_audit/2026-10-06-c2-cycle-index.md`.
7. On the branch (below), the three code-review reports in `docs/audit/`: `2026-10-09-c2-framing-2026-code-codex-c1.md`, `-c2.md` and `-f1.md`.
8. The ledger `docs/audit/2026-10-09-c2-framing-2026-evidence/round2-tdd-ledger.md`.

Before anything else, run `git fetch origin && git log main..origin/main --oneline`. Tell the manager, projects-30 at `uds:/tmp/cc-socks/2107.sock`, that you have started.

## 1. Where things stand

**Eric's decision.** He typed "Option 1" in the lead's pane at 16:05 EDT on 10/09 (register row `C2-framing-2026-test`, his words only). It chose a 2026 out-of-sample test of variant A (catcher-grouped framing replacing `pitcher_catcher_framing`) against the baseline.
- A-posted decides; A-projected is a stress arm; B is left out.
- The exception to D3 = RESERVE, the 10/06 side-item 2027 scope and D2's 2027 validation, for this one contrast, is the lead's recorded interpretation of the option as posed. The manager ruled it covered.

**The design** is frozen at `9b344d9`: r1 SIGN WITH EDITS was applied verbatim, and r2 was a plain SIGN.

**main** is pushed through `793082e`, the design close. Nothing of the code is on main.

**The code:**
- It is on branch `c2f26-code`, local only and not pushed, in worktree `~/projects/bts-c2f26-code`. The tip is `08a3b79`: `fccb6b9` plus the f1 archive.
- The range `793082e..08a3b79` holds `src/bts/simulate/backtest_blend.py` (the hook and strict prediction), `scripts/audit/c2_framing/f26.py`, three test files and the evidence directory.
- The fast suite at `fccb6b9` passed: 4477 passed, 0 failed.

**The reviews:**
1. **c1**, session `c2-framing-2026-c1`, now wound down, used its two rounds:
   - round 1 BLOCK, seven findings, on `4fa1ba4`;
   - round 2 BLOCK, two findings, on `d13eae8`.
2. **The fresh reviewer f1**, session `c2-framing-2026-f1`, round 1 BLOCK, four findings, on `fccb6b9`.
   - It is still alive and idle in `wA:p15` (tab `wA:t15`), session `01a12310-54ce-7192-885b-a1eb6c5e2447`, pid 39337, memories off.
   - Its worktree is `~/projects/bts-c2f26-code-review`, at `fccb6b9`.
   - It has one round left under its two-round cap. Whether to use it, or wind it down and start another fresh reviewer, is the manager's call. Ask before touching it.

**Nothing has run.** Nothing has been merged or launched, and no 2026 file has been read or hashed. There was no box contact in the code phase. X-37, the source inventory, the preparation-read row, the inputs row and the allowance row do not exist.

## 2. What blocks: review f1's four findings (read the report for the reproductions)

- **F1. Probability order.** `validate_run` checks consecutive ranks but not that ranks follow `p_game_hit`. A coherent swap of ranks 1 and 3 changed P@1 and passed.
- **F2. Model settings.** Only the two deterministic LightGBM flags are checked. A manifest with `n_estimators=1, learning_rate=0.9` passed, and the aggregate only checks that the seeds agree. Compare the full effective recipe, including the blend configurations, with the reviewed constants.
- **F3. Cost and budget at acceptance.**
  - `validate_run` and `seed_allowed` do not enforce the first-walk-forward stop or validate the allowance in the manifest.
  - `launcher_problems` does not read PENDING or apply C1's own `launch.terminal_problems`: a TERMINAL whose budget contradicts PENDING passed f26 although C1 rejects it.
  - Revalidate with C1's rules, bind the budget, and enforce the stop before a seed can advance.
- **F4. The expect step.**
  - `expect` checks the gate and the preparation row, but not that its directories are the inventory's, that its pins are the prepared ones and the screen's historical ones, or that its output is new.
  - It wrote the same expectation twice.
  - Bind it to the prepared pins and paths with exclusive creation, or get a process ruling that narrows the code's claims.

**Also from f1:**
- **A ledger error.** The ledger says off-launcher CPU (preparation, expect, the aggregate and pre-launch validation) is "not counted against the allowance". That contradicts design §7, which charges these costs to the allowance. Correct it, and make those costs enter the effective total before any allowance or launch. The manager's ruling 3 set this as "recorded by hand, reported alongside, not counted"; raise the conflict with the manager.
- **The early stop's basis.** Every model carries `batter_pitcher_shrunk_hr`, which the feature pipeline fills with a 0.2195 prior. So a featureless row is not expected from the real pipeline, and the stop is defensive plumbing. The note should say that, not lean on the weather observation.
- **Weak test names.** "End-to-end" overstates stubbed workflow tests.

## 3. What the previous lead learned (use it)
- **The pattern behind all three BLOCKs: acceptance checked internal consistency.** Reviewers forged records that agree with each other.
  - Every acceptance check must re-derive what it accepts from trusted evidence: the pinned inputs, the reviewed constants, and the launcher's own records under C1's own rules.
  - Before the next review, list every field a run declares and say for each one which trusted source checks it. F1–F3 are fields nobody had listed.
- **Test-first.** The original run half was written before its tests (disclosed in the commit and the ledger). Every fix since then has a RED/GREEN record in the ledger. Keep that discipline: the manager requires a failing test, its RED command and assertion, and the GREEN run.
- **Mutation check** (`docs/audit/2026-10-09-c2-framing-2026-evidence/mutation_check.py`). It is a scratch worktree per run that purges bytecode caches. The latest ledger is 72 of 72 at `4c429b5`. Mutation runs are PAUSED by the manager until the disk margin is back: free space is about 51.5 GiB and the EoR floor is 50 GB.
- **Suites and git are serialized** (the manager's rule):
  - no git write anywhere in `~/projects/bts` or its worktrees while a fast suite runs, because the screen's mutant-runner test fails closed on `.git` changes;
  - no suite while a Codex reviewer that may run git is active, unless the suite runs in a separate clone;
  - send the manager one line when a fast suite starts and one when it ends.
  - The fast suite takes about 22 minutes here.
- **Codex reviewers:**
  - start each with `-- --no-daemon -c memories.use_memories=false -c memories.generate_memories=false`, confirm the argv with `ps`, and record it in the tuple message;
  - put the memory prohibition first in the brief and hand-over text, and ask for one command per tool call;
  - the manager clears each hand-over.

## 4. Gates and protocol (unchanged; the frozen design §8 and the manager's rulings)
1. A design-level SIGN of the code, then the merge after the manager clears it.
2. X-37 PREDECLARED, published together with `docs/audit/c2-framing-2026-source-inventory.json`. X-37's description cites that file and its sha256. This comes before any 2026 file is opened or hashed.
3. Register row `C2-framing-2026-prep-read`, recorded after X-37. Tell the manager before the first 2026 read.
4. On the box, from `~/projects/bts-c1` at the admitted commit: `prepare`, then `expect`, which runs off-launcher with its CPU recorded in the C2 index.
5. The pins go into `scripts/audit/c2_framing/admission_2026.json`, and row `C2-framing-2026-inputs` binds their digest.
6. **Eric's allowance**, in his own typed words in the lead's pane, verified by the manager in the prompt log, becomes row `C2-framing-2026-allowance`.
   - The `CAP_H` change in `scripts/audit/c1/ledger.py` must be in reviewed code. The manager ruled that a cap commit after a plain SIGN gets its own short verbatim review, not counted as a round.
7. Launches: one seed at a time, through the C1 launcher. The manager gets a heads-up before each. Never launch from 00:45 to 03:10 box clock.

## 5. Open with Eric
- **The compute allowance. He has not given one.**
  - The lead recommended 100 CPU-hours additional, 12 per seed and a 4.1 CPU-hour first-walk-forward stop, with 120 total as the alternative.
  - Eric asked for the pros and cons of a higher cap, and they were answered in the pane.
  - The estimate is 83 CPU-h: 2.75 per walk-forward, three arms, ten seeds. The effective prior shared total is 137.6901.
- **Finding 1 of design review r1** was put to him as information: the deciding arm uses the final-boxscore starting catcher as the stand-in for the posted lineup. No objection was received. His silence is not consent to anything beyond "Option 1".

## 6. Records still to write on main
- NOTE rows, in the lead's words:
  - the tests-after disclosure;
  - code reviews c1, c2 and f1, with their shas: `58a8dc00…`, `6613515c…` and `6261ad60…`;
  - f1's blind-stage deviation: its tests read the register as an opaque argument, and two Matplotlib caches landed outside scratch;
  - c2's incidental read of `mutants-5bb792f.tsv`;
  - the uv clone-mode disk finding.
- One clause in C2 index row (e) for the code-review sequence.

## 7. Paths
- **Code:** `~/projects/bts-c2f26-code` (branch `c2f26-code`).
- **Review worktree:** `~/projects/bts-c2f26-code-review`. f1's brief and report are under `.codex-review/c2-framing-2026-fresh/`.
- **Results directory** `~/projects/bts-c2-framing-results/2026-test/`, not in git:
  - raw RED/GREEN records in `tdd/`;
  - the design-review briefs and reports in `design-review-record/`;
  - c1's briefs and reports in `code-review-c1-record/`;
  - suite outputs and mutation logs;
  - `prompt-f1.md`.
- **Older worktrees** (`bts-c2-framing-s2-acceptance`, `-review`, `bts-c2-2a-review`, `bts-c2-framing-s2-review`, `bts-w15-evidence`, …) are kept as records. Removing them is Eric's call.
