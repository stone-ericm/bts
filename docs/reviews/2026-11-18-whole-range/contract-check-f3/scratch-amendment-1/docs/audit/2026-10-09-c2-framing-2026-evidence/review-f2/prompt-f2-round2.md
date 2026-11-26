# Fresh review f2, round 2 (your last): an anchored design review of the trust-boundary note for round 5

**Do not search, open or read any memory file, earlier session record, rollout summary or anything under `~/.codex/`.** This includes `~/.codex/memories/`. If a startup step or a compaction summary has shown you any such content, disclose exactly what in the report's first section and carry on.

**Read this whole file before doing anything else.** Run one command per tool call. Do not attempt to remove any scratch directory, including `/private/tmp/c2f26-f2-review`; leave scratch in place and say so.

**How to write, so a content filter does not withhold your reply:** describe every check in prose; keep code excerpts to the few lines a finding needs; the report is the deliverable.

## What this round is

Your round 1 (BLOCK on `e0ff07ce6b26a99a594db9cdb34c86989cb991e6`, report sha256 `44b89775375fcee176a60411e7941ec6abf9c35232b2c99e716822c136602334`) found B1–B4 and the catcher-evidence coercion, the two empty selectors and the inventory and ledger inaccuracies. The herdr manager ruled that round 5 starts with a design, not code: a note that draws the trust boundaries of the whole chain, states for each boundary what binds it and what happens when the bound thing is missing or failed, and maps every finding of the five reviews (c1 twice, f1 twice, you) onto a boundary. **This round is an anchored design review of that note.** No code has changed since your round 1. A BLOCK here costs no code; a SIGN here is the plan the code is then written against, test-first, and a new fresh reviewer on the whole range is the merge gate.

**The checkout** is this directory, detached at `abd24777ed8d4b0cad4a1231d4064715a9d1e16e`, which is `e0ff07c` plus one documentation-only commit adding the note. Confirm HEAD and a clean tracked tree before you start and after you finish.

**The note:** `docs/sota_audit/2026-10-10-c2-framing-2026-trust-boundaries.md` (sha256 `c54063e79ddff737fa84015a53859f9135b23935f66c2544c3bae648388c8adf`). It does not change the frozen design (`docs/sota_audit/2026-10-09-prereg-c2-framing-2026-test.md` at `9b344d9`, sha256 `a0d2919d…`); it is a code-phase note.

## What to check

1. **Closure.** For each of your B1–B4 and your coercion finding: does the boundary the note assigns it to, with the binding and the refusal the note states, close it? Name the next coherent forgery the stated binding would still admit, if any.
2. **The caller.** Walk every boundary 0–15 and say whether anything there still trusts the caller: a caller-supplied path, a caller-writable file without a registered hash or an exclusive identity, a value compared only with another output of the same process, a "missing" that defaults instead of refusing.
3. **Completeness of the map.** Is every finding of the five reviews mapped to a boundary that actually addresses it (the table at the end; the archived reports c1, c2, f1, f1r2 and your round 1 are in `docs/audit/`)? Name any finding mapped to a boundary that does not address it, and any boundary with no stated failure rule.
4. **Implementability.** Does round 5's list implement the note's bindings as stated, in an order that lets each change be tested against a RED on the current code? Are the reservation record, its nonce in the payload's environment and its exclusive consumption by the C1 unit sound against C1's real launcher and guard (`scripts/audit/c1/`), including the launcher's lock and the `status` sweep the note relies on? Are the accounting rules (the ruled seed row, per-step completeness, `finally` charging with complete CPU, whole-process CPU for CLI steps) enforceable without a false refusal of a valid real sequence?
5. **Scope statements.** Are the disclosed limits (boundary 15) and the accounting scope (boundary 7) honest and complete?

## Constraints

As your round 1: no `data/` reads; no `.env`, configuration or credentials; no memory files or session records; no network, box, SSH or `gh`; `uv run --offline`; no tracked edits; do not run `mutation_check.py`; scratch under `/private/tmp/c2f26-f2-*` (leave it in place); no escalation; tests on synthetic data only, and only if a check needs them (this is a design review).

## Verdict and report

**Write the report to** `/Users/eric/projects/bts-c2f26-code-review/.codex-review/c2-framing-2026-fresh2/report-f2-round2.md`. Sections, in order: **`## Verdict`** (first line exactly one of `**SIGN.**`, `**SIGN WITH EDITS.**` or `**BLOCK.**`; exactly one line `Reviewed-commit: abd24777ed8d4b0cad4a1231d4064715a9d1e16e`; disclosures), **`## Findings`** (your B1–B4 and coercion closure first, then the caller walk, then the map, then implementability), **`## Edits`** (for SIGN WITH EDITS, one unified diff to the note that applies with `git apply`; otherwise "None"), **`## What was run`**.

**Meanings:** SIGN — the note is the plan; code is written against it. SIGN WITH EDITS — the plan once your diff is applied verbatim. BLOCK — a boundary still trusts the caller or a finding is unaddressed; the note is revised. This is your second and last round; no verdict approves a merge, a run, a compute allowance, a 2026 read or a production change.

**When the report is complete,** print on its own line `C2F26F2R2-DONE-` followed by the first 8 hex characters of `shasum -a 256` of the report. On the next line, print `DONE`. Do not delete anything afterwards.
