# Adversarial review, confirmation round: change set 6a, one change made at the owner's request after round 2

You are a fresh reviewer; you have seen none of the earlier work. **Assume the lead is wrong.** Tier A. Change no tracked file; probes, logs and the report go in `.codex-review/cs6a-r3/` inside this copy; pytest only through `~/bts-runs/yield_run.sh .codex-review/cs6a-r3/<name>.log <command…>`, one script and one invocation per family, files created with your edit tool, no script edited after you ask to run it; if a wrapper run prints a `nice` warning, wait for it to exit. No network (audit hook), box or contest account.

This checkout is at the commit under review. The whole change set is `git diff 09d9b51f HEAD`. **Your subject is `git diff 87c118f6 HEAD`**: one commit by the lead that changes the order of two statements in the `schedule` command of `src/bts/cli.py` and its tests in `tests/test_live_gate_cs6a.py`.

## Background
`bts schedule` on the base (`09d9b51f`): resolve the default ET date, load the TOML, call `run_day`. Change set 6a added a gate between the load and `run_day` that refuses to start a configuration with DM or public delivery (or `entry_intent = "enter"`) when contest state is not required (contract: `.codex-review/cs6a-r3/kickoff-cs6a.md`, rows 3 and 4, and amendment 2 at its end). The reviewed version resolved the date AFTER the gate for a real run. Round 2 (`.codex-review/cs6a-r3/report-codex-r2.md`) signed it but confirmed that a real accepted launch whose load straddles midnight ET then schedules the next day. The owner asked for the base order back. The command now is: date, load, gate, `run_day`.

## Establish, by running the real code
1. **The command's behaviour now equals the base's in everything but the gate.** For `bts schedule` with and without `--date`, with and without `--dry-run`, on accepted, warned and refused configurations, with the clock crossing midnight ET before the date lookup, during the TOML load, and after the gate: the date `run_day` receives, the exit status, stderr, and every resource access, at `09d9b51f`, at `87c118f6` and here (real loader and real `run_day` with captured schedule fetch, notifications and sleep, as round 2's `schedule_round2.py` did; it is in `.codex-review/cs6a-r3/` for reuse). Every accepted case must equal the base; every refused real case must exit 2 with the row-3 message, never call `run_day`, and touch nothing but the configuration file and the clock.
2. **Nothing else changed:** `git diff 87c118f6 HEAD -- src` is exactly the reordering and its comment. The diff of the `schedule` function against the base is exactly the added gate.
3. **The tests:** which of the changed or added tests fail at `87c118f6` and why; can each fail (mutate a scratch copy: move the date lookup after the load; after the gate; make it conditional on `--dry-run`); was any assertion from the reviewed version removed or loosened beyond what amendment 2 withdraws (the author's refusal test no longer forbids the date lookup: is everything else it forbade still forbidden?). Run the touched test files under clocks in 2027 and 2028.
4. Anything else in `git diff 87c118f6 HEAD` you consider wrong.

A finding about test code is a NOTE unless it makes a false claim about behaviour that nothing else covers.

## Deliverable
`.codex-review/cs6a-r3/report.md`, under 45 lines: `Verdict` (SIGN, SIGN WITH EDITS, or BLOCK), `Findings`, `Tests`, `What I ran`. A BLOCK names a concrete input, the wrong production outcome, the file and line, the smallest fix and how you demonstrated it. When complete, print `CS6AR3-<first 8 hex characters of the sha256 of report.md>` on its own line and then `DONE`.
