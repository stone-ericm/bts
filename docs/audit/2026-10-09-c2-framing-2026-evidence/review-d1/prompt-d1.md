# Fresh design review d1: the trust-boundary note (revision 2) for the 2026 framing test's code (BTS, C2 side item (e))

**Do not search, open or read any memory file, earlier session record, rollout summary or anything under `~/.codex/`.** This includes `~/.codex/memories/`. If a startup step has shown you any such content, disclose exactly what in the report's first section and carry on.

**Read this whole file before doing anything else.** Run one command per tool call. Do not create, remove or modify any scratch directory under `/private/tmp/c2f26-*` except your own `/private/tmp/c2f26-d1-*`; leave `/private/tmp/c2f26-f2-review` alone.

**How to write, so a content filter does not withhold your reply:** describe every check in prose; keep code excerpts to the few lines a finding needs; the report is the deliverable.

## What you are reviewing

**You are a fresh reviewer of a design document, not of code.** Five code-review rounds by three sessions each found a new unenforced trust path at the run boundary of this test's code. The herdr manager ruled that round 5 starts with a design note that draws the trust boundaries of the whole chain, and that a fresh session reviews that note alone before any code is written. Your verdict is on the note. The code is unchanged since the last code review and is known to be blocked; do not re-review it, read it only to judge whether the note's bindings are implementable against it and against C1's real launcher.

**The checkout** is this directory, a detached worktree at `f4edff3b32899bbe3885f772cb056a5dbb209141`. Confirm HEAD and a clean tracked tree before you start and after you finish.

**The note:** `docs/sota_audit/2026-10-10-c2-framing-2026-trust-boundaries.md` (sha256 `b659f045678775ab6323ee824e9b526522e07950642f7313433c302322744dcc`). It is a code-phase note; the frozen design it serves, `docs/sota_audit/2026-10-09-prereg-c2-framing-2026-test.md` at `9b344d9` (sha256 `a0d2919d5f94b466832ec035ff8833a4d1c0aeb5ccd9c3ec005ea78f1352aeca`), does not change.

**The code the note is about:** `scripts/audit/c2_framing/f26.py` (the test's rules, admission, preparation, expect, run, validation, aggregate and launch wrapper), `scripts/audit/c1/` (the shared C1 launcher, guard, ledger and admission the note proposes to extend in a default-off combined mode), `src/bts/simulate/backtest_blend.py` (the hook), and the tests under `tests/scripts/c2_framing/` and `tests/scripts/test_c1_cycle.py`.

## Part 1 (your own review of the note)

Write Part 1 into the report before any Part 2 read. In Part 1 do not read: the code-review reports `docs/audit/2026-10-09-c2-framing-2026-code-codex-*.md`, the design-review reports `…-design-codex-r1.md` and `-r2.md`, anything under `docs/audit/2026-10-09-c2-framing-2026-evidence/`, any commit message, anything under `.codex-review/` other than this brief, or the register rows that summarize reviews. Check:
1. **Every boundary 0–15.** For each: does the stated binding rest on a trusted producer (C1 under its lock, the guard inside the unit, git, bytes fixed by a register row, a reviewed constant) or on something the caller controls (a path argument, a file the same process writes, agreement between two outputs of one process, a default where a refusal belongs)? Is the "missing or failed" rule a refusal, and is it stated at the right stage (before protected reads; before the claim; after it)?
2. **The reservation (boundaries 8 and 9).** Is a C1-written `RESERVATION_<unit>.json`, decided and written inside C1's `_locked` after sweep and reconciliation and before PENDING and `systemd-run`, with the account's byte length and hash at decision, a receipt with a trusted producer and a prior unit binding? Can a caller still obtain a genuine receipt without the combined decision, or run a payload that passes `run`'s receipt check without one? Does anything between decision and start remain unserialized? Read `scripts/audit/c1/launch.py` (`_locked`, `plan_launch`, `launch_lock`, `reconcile`, `sweep`) and `guard.py` for this.
3. **The account (boundary 7).** Are the phases, states, digest links, settlement rows, the one-owner CPU model, the tail reserve and the overhead reserve sound and implementable without a false refusal of the valid sequence (seed row → prepare → expect → ten launches → aggregate, with Eric-acknowledged failed attempts in between)? Can CPU be counted twice or not at all under the stated rules? Is the scope statement honest?
4. **The certificate (boundary 0), the register (boundary 14), the row order (boundary 3), the actual arguments (boundary 5), the stages of run (boundary 10), the final cap rule (boundary 13).**
5. **The map.** Does every finding of the five code reviews, as listed in the note's table, map to a boundary that addresses it? Name any mapping that does not.
6. **The C1 scope conditions.** Are the manager's conditions (a)–(e) in the note sufficient for a change to shared infrastructure, and does the note's design respect (a), additive and default-off, byte-for-byte unchanged behavior without the new flags?
7. **Implementability.** Can each item of "What round 5 implements" be tested as a behavioral RED against the current code before its fix, with controls that carry the legitimate history the new rules require?

## Part 2 (against the record)

Now read the two reports of the previous reviewer f2, `docs/audit/2026-10-09-c2-framing-2026-code-codex-f2.md` (its code review, BLOCK on `e0ff07c`) and `…-f2r2.md` (its design review of the note's revision 1, BLOCK on `abd2477`), and the earlier reports `-c1.md`, `-c2.md`, `-f1.md`, `-f1r2.md`. Check: does revision 2 close every point of `-f2r2.md`? Is every finding of the earlier reports mapped as the note claims? Does your Part 1 disagree with f2 anywhere, and if so, which is right?

## Constraints
- **Reads:** no `data/` reads; no `.env`, configuration or credentials; no memory files or session records.
- **Network:** none; no box, SSH or `gh`. Use `uv run --offline` if you run anything.
- **Writes:** no tracked edits; write only the report below and scratch under `/private/tmp/c2f26-d1-*`, which you may leave in place (say so) — do not ask to delete anything with `rm -rf`; no git commits, branches or worktrees; do not run `mutation_check.py`.
- **Escalation:** none. **Runs:** this is a design review; run code only to check implementability, on synthetic data.
- **Command form:** `PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run --offline python -B …`.

## Verdict and report
**Write the report to** `/Users/eric/projects/bts-c2f26-code-review/.codex-review/c2-framing-2026-design-d1/report-d1.md`. Sections, in order: **`## Verdict`** (first line exactly one of `**SIGN.**`, `**SIGN WITH EDITS.**` or `**BLOCK.**`; exactly one line `Reviewed-commit: f4edff3b32899bbe3885f772cb056a5dbb209141`; exactly one line `Review-kind: design, the trust-boundary note`; disclosures), **`## Part 1`**, **`## Part 2 (against the record)`**, **`## Edits`** (for SIGN WITH EDITS, one unified diff to the note that applies with `git apply`; otherwise "None"), **`## What was run`**.
**Meanings:** SIGN — the note is the plan; code is written against it test-first, and a separate fresh reviewer reviews note and code together as the merge gate. SIGN WITH EDITS — the plan once your diff is applied verbatim. BLOCK — a boundary still trusts the caller or a finding is unaddressed; the note is revised. This review has at most two rounds. No verdict approves a merge, a run, a compute allowance, a 2026 read or a production change; this review's SIGN is a design certificate and is not the admission's code certificate.
**When the report is complete,** print on its own line `C2F26D1-DONE-` followed by the first 8 hex characters of `shasum -a 256` of the report. On the next line, print `DONE`. Delete nothing afterwards.
