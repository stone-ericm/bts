# Fresh code review f2: the 2026 out-of-sample test of catcher-grouped framing (BTS, C2 side item (e)), whole range, Part 1 blind

**Do not search, open or read any memory file, earlier session record, rollout summary or anything under `~/.codex/`.** This includes `~/.codex/memories/`. If a startup step has already shown you any such content, disclose exactly what in the report's first section and carry on.

**Read this whole file before doing anything else.** Run one command per tool call. Until you have read it all, run no command that touches a path outside this checkout.

**How to write, so a content filter does not withhold your reply:** describe every check in prose (what you ran, where, the measured outcome); keep code excerpts to the few lines a finding needs; keep chat replies short; the report is the deliverable.

## What you are reviewing

**You are a fresh reviewer.** You have not reviewed this code before. Four earlier review rounds by two other sessions exist (c1 twice, f1 twice); Part 1 below is blind to all of them. **Your plain SIGN is the merge gate:** after it, the branch is merged to main and the shared admission gate admits runs under it. Nothing in your verdict launches, reads 2026 data, grants compute or changes production.

**The checkout** is this directory, a detached worktree at `e0ff07ce6b26a99a594db9cdb34c86989cb991e6`. Confirm HEAD and a clean tracked tree before you start and after you finish.

**The range** is `793082e93eb624970edbf32c974be07490ff84f5..e0ff07ce6b26a99a594db9cdb34c86989cb991e6`: everything the branch `c2f26-code` would merge (its merge base with main is `793082e`; main has since gained documentation-only commits that touch none of these paths). It implements the frozen pre-registration `docs/sota_audit/2026-10-09-prereg-c2-framing-2026-test.md`, frozen at `9b344d9`; the copy here is the same bytes (sha256 `a0d2919d5f94b466832ec035ff8833a4d1c0aeb5ccd9c3ec005ea78f1352aeca`). The changed files:
- `src/bts/simulate/backtest_blend.py`: `blend_walk_forward(predict_day_transform=None, strict_predict=False)` and its strict checks. The only change under `src/`.
- `scripts/audit/c2_framing/f26.py`: the test's rules, admission, preparation, expect step, run, validation, aggregate and launch wrapper.
- `tests/scripts/c2_framing/test_f26_hook.py`, `test_f26_rules.py`, `test_f26_run.py`.
- `docs/audit/2026-10-09-c2-framing-2026-evidence/`: the author's ledgers (rounds 2, 3 and 4), inventory, mutation script and ledgers, raw test outputs.
- `docs/audit/2026-10-09-c2-framing-2026-code-codex-c1.md`, `-c2.md`, `-f1.md`, `-f1r2.md`: the archived earlier reviews.

**The design's sections:** §3 the inputs and the starter proxy; §4 the catcher by arm, the as-of value, the missing reasons and the hook; §5 the procedure; §6 the dispositions; §7 the budgets (every CPU-second, on or off the launcher, is charged to the allowance); §8 the process.

**Gates set outside the design** (the herdr manager's process rulings; the author's account; weigh them as stated):
- The X-37 exposure row is recorded before any 2026 file is opened or hashed. It cites a source-inventory file by sha256, published in the same commit. The preparation-read row is also recorded before any 2026 read.
- A one-time, off-launcher `expect` step after preparation pins the expected catcher evidence.
- Nothing launches without Eric's typed compute allowance (register row `C2-framing-2026-allowance`); the code's `CAP_H` is still 165 and no allowance row exists yet, so the allowance gate refuses.
- The launcher reserves the full per-seed budget. No launch from 00:45 to 03:10 America/New_York.
- CPU spent outside the launcher (preparation, expect, the launch wrapper, the C1 launcher process, the aggregate) is charged to the allowance (design §7). The manager's rulings of 2026-10-09 fix the accounting: (a) the authoritative combined record is C1's `compute_ledger.tsv` (launcher-written) plus the test's `<OUT_ROOT>/off_launcher_cpu.jsonl` (append-only; written by prepare, expect, launch and aggregate, and once by the lead to seed it); (b) the seed row carries the prior off-launcher total the C2 index records and the box ledger does not: 19.04 s + 60.97 s = 80.01 s = 0.0222 CPU-h, source "C2 index rows of 2026-10-08 and 2026-10-09"; (c) a missing or malformed record is refused, fail closed; a step that fails after spending CPU still appends its row; (d) the additional allowance is enforced through Eric's raised shared cap (his row's cap == `CAP_H`), one cap.
- The preparation record `PREPARED.json` is bound by a register row `C2-framing-2026-prepared` (its sha256), recorded after the preparation read and before `expect`; another C1 invocation of a seed is refused unless a register row `C2-framing-2026-invocation-<unit>` ruled by Eric acknowledges it.

## Part 1 (blind): your own review of the whole range against the design

In Part 1, do not read: any commit message; anything under `docs/audit/2026-10-09-c2-framing-2026-evidence/` (the ledgers, the field inventory, the mutation script and ledgers, the raw outputs); the code-review reports `docs/audit/2026-10-09-c2-framing-2026-code-codex-*.md`; the design-review reports `…-design-codex-r1.md` and `-r2.md`; anything under `.codex-review/` other than this brief; the register rows that summarize reviews (`C2-framing-2026-design-r1`, `-design-frozen`, `-tests-after`, `-code-reviews`, `-round3`, `-suite-serialization`, `-main-push-slip`). Use commands that print no commit subjects, and disclose any you see. Write Part 1 into the report file before any Part 2 read. Check:
1. **The hook and strict prediction.** Does the transform reach only the predicted day's private copy, before the blend and both estimated-PA prediction calls? Are default callers bit-identical to the base commit's implementation? Does strict mode refuse exceptions and misaligned or non-finite scores in both calls, without refusing valid real predictions?
2. **Every rule against §3, §4 and §6.**
3. **Every gate.** Name any path to a run, a launch, an acceptance, an expectation or a 2026 read that bypasses its gate.
4. **Acceptance.** Build your own list of every field a run or the process declares (the manifest, results and units, the claim, the retained profiles, the catcher evidence, scorecards and diffs, the launcher's PENDING/TERMINAL/RECONCILED records, the preparation and expectation records, the off-launcher record, the register rows, the admission record) and, for each, say which trusted source checks it, or that nothing does. Can `validate_run`, `seed_allowed`, `aggregate` or `expect` accept a run or an expectation that is wrong, incomplete or not what the design registers? Look for coherent forgeries: records that agree with each other and that only a trusted source could refuse.
5. **False refusals.** Can a valid real run or a valid real expect step be refused? Check against the real C1 record layout (`launch.plan_launch`, `guard.py`'s receipt, `launch.reconcile`), the real predict functions, the real feature pipeline, the real `LGB_PARAMS` with and without the deterministic environment flag in the validating process, and the register grammar the gates parse.
6. **The tests.** Look for tests that could not fail, stubs standing in for the boundary that matters, and design rules with no test.

## Part 2 (against the record)

Now read the commit messages (`git log 793082e93eb624970edbf32c974be07490ff84f5..e0ff07ce6b26a99a594db9cdb34c86989cb991e6`), the evidence directory and the four archived reviews (c1 BLOCK on `4fa1ba4`, c2 BLOCK on `d13eae8`, f1 BLOCK on `fccb6b9`, f1 round 2 BLOCK on `d73693d`). Check:
1. Do the commit messages and the two test-first ledgers describe the work accurately?
2. Did every earlier finding close? Compare the author's `field-inventory.md` with your own Part 1 list: name any field it omits, any CHECKED row whose stated source the code does not apply, any REPORTING row a decision actually reads.
3. Are the mutation ledgers' claims supported? Note that the round-3 and round-4 mutants (M75–M114) have not been run (mutation runs are paused); judge their anchors and selectors statically and do not run `mutation_check.py`, which creates a git worktree.
4. Give one sentence on each of the author's disclosed limits (the probabilities and the retained ten need the day's predictions; exact probability ties are reported, not ordered; the lookup's and table's derivation from the raw feeds is the admitted preparation's work): is each honest and complete?

## Constraints

- **Reads:** no `data/` reads at all; no `.env`, configuration or credentials; no memory files or session records, as stated at the top.
- **Network:** none; no box, SSH or `gh`. Use `uv run --offline`.
- **Writes:** no tracked edits; do not run `mutation_check.py`; write only the report below, plus scratch under `/private/tmp/c2f26-f2-*`, which you delete when you are done. No git commits, branches or worktrees.
- **Escalation:** none.
- **Runs:** tests on synthetic data only.
- **Command form:** `PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run --offline python -B …` or `… pytest -q -p no:cacheprovider <paths>`.

## Verdict and report

**Write the report to** `/Users/eric/projects/bts-c2f26-code-review/.codex-review/c2-framing-2026-fresh2/report-f2.md`. Its sections, in order:
- **`## Verdict`:** its first line is exactly one of `**SIGN.**`, `**SIGN WITH EDITS.**` or `**BLOCK.**`; it holds exactly one line `Reviewed-commit: e0ff07ce6b26a99a594db9cdb34c86989cb991e6` and no other line may start with `Reviewed-commit:`; it discloses any memory or outside read.
- **`## Part 1 (blind)`:** written before Part 2's reads; say so.
- **`## Part 2 (against the record)`.**
- **`## Edits`:** for SIGN WITH EDITS, one unified diff that applies with `git apply`. Otherwise "None".
- **`## What was run`.**

**What each verdict means:** SIGN — the range can be merged and admitted as written. SIGN WITH EDITS — it can be once your diff is applied verbatim and a later plain SIGN confirms the edited commit (the shared admission gate admits only a plain SIGN). BLOCK — something must change that a verbatim diff cannot settle. This review has at most two rounds. No verdict approves a run, a compute allowance, a 2026 read or a production change.

**When the report is complete,** print on its own line `C2F26F2-DONE-` followed by the first 8 hex characters of `shasum -a 256 /Users/eric/projects/bts-c2f26-code-review/.codex-review/c2-framing-2026-fresh2/report-f2.md`. On the next line, print `DONE`.
