# Review request: the 2026 framing test's studio runner (tier C, the result-deciding tool)

Round 1 of 1. You are a fresh reviewer. Work only inside this workspace (`/Users/eric/projects/bts-agy-ka`, commit `d2da185`). Do not edit tracked files.

## Off limits
Do not read anything outside this workspace. In particular do not read `/Users/eric/projects/bts` (another checkout of this repository, including its `.git`), `~/projects/bts-c2-framing-results/`, `~/bts-runs/` or `~/projects/claude-shared/`. Ten seeds of this test have already run; their outputs live outside this workspace and you do not need them: this is a review of code against a design.

## What to review
`scripts/audit/c2_framing/f26_studio.py` (about 270 lines): `prepare`, `run`, `calendar_2026`, `seed_rank1`, `aggregate`. It replaces only the launcher path of `scripts/audit/c2_framing/f26.py` and calls that file's science unchanged (`starter_proxy`, `lookup_entry`, `table_frame`, `TeamHistory`, `AsOfFraming`, `ArmTransform`, `expected_calendar`, `rank1`, `dispose`, `load_inputs`, and `screen.py`'s `framing_by`, `relabel`, `original_portion_labels`, `blend_configs`, `install_closed_inputs`). Tests: `tests/scripts/c2_framing/test_f26_studio.py`.

The frozen design is `docs/sota_audit/2026-10-09-prereg-c2-framing-2026-test.md` (§3 inputs, §4 the catcher by arm, §5 procedure, §6 dispositions). The declared deviations are in exposure row X-37 of `docs/audit/2026-09-22-exposure-register.md` (rebuilt `pa_2026`, Apple silicon, no C1 launcher, sequential jobs with pauses). Do not re-litigate those deviations or the launcher's removal; they are ruled.

## The question
Could this runner or its aggregate produce a WRONG DISPOSITION or wrong reported numbers for the design's test, given correct inputs? Concretely:
1. Does `run` compute the three arms as the design's §4–§5 say, on the same frame, with the arm override reaching only predicted-day rows? Compare with `f26.run` (lines ~1402–1524 of f26.py), which was written to the design: is anything that affects the computed predictions or labels different or dropped?
2. Does `aggregate` build x[s, t] = A-posted rank-1 hit minus baseline rank-1 hit, seeds in §5's order, dates ascending over the complete calendar, and hand it to `f26.dispose` correctly? Is A-projected only described, never deciding?
3. Completeness (§6 "Completeness first"): can the aggregate accept fewer than the ten registered seeds, a stopped or partial seed, seeds on different calendars or different inputs, or recorded rank-1 hits that the retained profiles do not reproduce?
4. `calendar_2026`: is the calendar the design's (sorted official dates with at least one original-portion batter-game in the frozen pa_2026)?
5. Anything in `prepare` that would make the starter-proxy table, the 2017–2026 lookup or the pins differ from the design's §3 for the same feeds.

## Method
Do not stop at reading the code and running the existing tests. Write and run your own adversarial probes: build synthetic seed directories and inputs and call the functions, trying to make `aggregate` publish a disposition for evidence that the design's §6 calls incomplete or invalid. Put probe scripts under `.codex-review/agy-ka/scratch/` inside this workspace. Run Python as `.venv/bin/python` (for tests: `.venv/bin/python -m pytest <file> -q -p no:cacheprovider`). You may run `tests/scripts/c2_framing/test_f26_studio.py`, `test_f26_rules.py` and `test_f26_hook.py`; do not run the whole suite.

## Rules for findings
Tier C under `docs/superpowers/notes/work-policies-2026-11-12.md`. Review behaviour and the stated contract. A finding about test-code or evidence-tooling quality is a NOTE, not a BLOCK, unless it makes a false claim about behaviour that no gate run or real-artifact check covers. A BLOCK must name a concrete input or state and the wrong output it produces, and say whether you demonstrated it by running something.

## Deliverable
Write your report to `/Users/eric/projects/bts-agy-ka/.codex-review/agy-ka/report-g1.md` with these sections: `Verdict` (SIGN, SIGN WITH EDITS, or BLOCK), `Blocking findings` (each: input/state, wrong output, file and line, required change, how demonstrated), `Notes`, `What I checked`. Keep it under 150 lines.

When the file is complete, print one line `AGYKA1-<first 8 hex characters of the sha256 of the report file>` and then `DONE`.
