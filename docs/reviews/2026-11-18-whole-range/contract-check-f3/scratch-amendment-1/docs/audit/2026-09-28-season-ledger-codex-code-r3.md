## Round

code r3 — `41af8f3..d112b18`.

Reviewed revision: `079ca4760a5e447c7a9cd7b818e0702c90684109..d112b18c1359614417af0d27dcba69afdfef8a67`, against spec v4 and I1–I15, continuing the prior whole-branch review.

**The accepted r2 findings are resolved.** I found no demonstrated class (a) defect in this revision: no wrong fact from the unmodified compiler on the tested production-shaped inputs, and no false rejection in the requested boundary cases. One additional class (b) hardening opportunity remains below; it does not block this code review.

Tracked files remained unchanged. All review writes are under `.codex-review/season-ledger-code/`. No real data or real snapshots were read; no network, SSH, herdr, or other pane was accessed. Tests use synthetic fixtures, including the acquisition suite's scratch tree and mocked fetches.

## Replay

From the worktree root, each Python command below used:

```sh
PYTHONDONTWRITEBYTECODE=1 TZ=America/New_York TMPDIR=/Users/eric/projects/bts/.claude/worktrees/season-ledger-phase1/.codex-review/season-ledger-code/tmp
```

Commands following that prefix:

```sh
.venv/bin/python -m pytest -q -p no:cacheprovider tests/scripts/season_ledger
.venv/bin/python -m pytest -q -p no:cacheprovider --tb=short .codex-review/season-ledger-code/probes/test_review_r3_r1.py .codex-review/season-ledger-code/probes/test_review_r3_r2.py
.venv/bin/python .codex-review/season-ledger-code/probes/run_r1_mutants_r3.py
.venv/bin/python .codex-review/season-ledger-code/probes/run_r2_mutants_r3.py
.venv/bin/python .codex-review/season-ledger-code/probes/run_invariant_scale_r3.py
.venv/bin/python -m pytest -q -p no:cacheprovider --tb=short .codex-review/season-ledger-code/probes/test_review_r3_new.py
```

Hereafter `P/` means `.codex-review/season-ledger-code/probes/`.

### Suite and independent probes

- Repository suite: **164 passed in 0.89 s** (`P/r3-suite.txt`).
- Ported r1/r2 probes: **40 passed, 1 failed in 0.90 s** (`P/r3-ported.txt`). All **29 r2 cases pass**. The r1 ports are **11 pass / 1 deliberate contract failure**: `test_g4_counts_slot_objects_only` still expects the object-only rule superseded by `ecebb95`. The accepted non-null-presence ruling remains unchanged; this is not an open defect.
- Newly added probes: **32 passed, 1 failed in 1.22 s** (`P/r3-new.txt`). All boundary/normal-output controls pass. The sole failure is the explicitly class (b) round-streak corruption probe in New finding 1; it starts with a correct build, modifies checker arguments, then shows that the checker does not reject that hypothetical corruption.

The five formerly failing r2 cases now pass: `test_invariant_rejects_whole_decision_hidden_by_disposition` for single, double and skip; `test_invariant_rejects_fabricated_recipe_fit`; and `test_invariant_rejects_stale_contest_observation`. The other 24 r2 controls still pass, including shared batter/game identities, unrecorded games, delivered/private/skip behavior, and 411 observations of one slot.

All other r1 ports pass: missing decision-only double leg, swapped same-kind references, an entirely missing recipe, a dropped contest identity, evidenced contest-only grade, delivery from an otherwise unusable pick file, both malformed-delivery parameters, wrong schedule date, naive manifest mtime, and the complete compiled recipe set.

### Every port change and rename

- `test_review_r2_ported.py` → `test_review_r3_r1.py`. No test/helper/API names or assertions changed. One fixture disposition changed: in `test_invariant_rejects_an_entire_missing_recipe`, `not_selected` → `outside_season_window`, because that direct checker test passes `season_dates=[]`. This matches the newly enforced source-date contract. The path-derived `file_date` fixture enrichment from r2 remains.
- `test_review_r2_new.py` → `test_review_r3_r2.py`, **byte-for-byte unchanged**. No production symbol or assertion needed a rename.
- `run_mutants_r2.py` → `run_r1_mutants_r3.py`; witness file → `test_review_r3_r1.py`; scratch directory `mutants_r2` → `r1_mutants_r3`; log prefix `r2_mutant_` → `r3_r1_mutant_`; receipt `r2-mutant-results.json` → `r3-r1-mutant-results.json`.
- `run_new_mutants_r2_verified.py` → `run_r2_mutants_r3.py`; witness file → `test_review_r3_r2.py`; scratch directory `new_mutants_r2_verified` → `r2_mutants_r3`; log prefix `r2_verified_new_mutant_` → `r3_r2_mutant_`; receipt `r2-verified-new-mutant-results.json` → `r3-r2-mutant-results.json`.
- Neither mutant runner needed a mutation-anchor or replacement change.
- `run_invariant_scale_r2.py` → `run_invariant_scale_r3.py`; control module/file `_r1_control` / `reconcile_r1_control.py` → `_r2_control` / `reconcile_r2_control.py`; comparison modes `r1_check,r2_check` → `r2_check,r3_check`; output directory `r2_invariant_scale` → `r3_invariant_scale`. The r2 control is the exact `reconcile.py` obtained with `git show 079ca4760a5e447c7a9cd7b818e0702c90684109:scripts/audit/season_ledger/reconcile.py`.

### Mutation replay

All **six prior mutants are killed**, by both the repository suite and the independent witness. The runners copy source/tests into separate scratch trees and log the actual imported compiler path. The recipe fingerprint remains `5e9d74f2f9c3093d66bc7c9ab0a7028e5fb361cdb46bbb1b368cd71e7f297b3f`.

| Mutant | Repository suite | Independent witness |
|---|---:|---:|
| r1: omit evidenced contest-only row | 2 failed / 162 passed | 1 failed |
| r1: drop T24 after evaluation | 19 failed / 145 passed | 1 failed |
| r1: copy round grade onto contest-only slot | 1 failed / 163 passed | 1 failed |
| r2: omit decisions with an unrecorded primary game | 1 failed / 163 passed | 1 failed |
| r2: fabricate T24 fit/label | 19 failed / 145 passed | 1 failed |
| r2: use only first contest snapshot | 1 failed / 163 passed | 1 failed |

Receipts: `P/r3-r1-mutant-results.json`, `P/r3-r2-mutant-results.json`; individual logs use the prefixes listed above. These are replays of the previously demonstrated acceptance gaps, class (b), rather than evidence that those mutations existed in production code.

### Boundary cases

`P/test_review_r3_new.py` adds:

- **12 source-date controls:** decision-only, pick-only and agreeing decision/pick inputs on 3/24, 3/25, 9/27 and 9/28. The endpoints are included; adjacent out-of-season sources remain accounted with `outside_season_window` and produce no canonical selection. No false rejection.
- **9 directory/body-date conflict controls:** single, double and skip decisions, with an in-season directory naming another in-season or out-of-season body date, and a 9/28 directory naming an in-season body date. The existing path-derived `file_date` governs placement; both dates remain in occurrence `fields_json`, and the original body remains in `record_raw_json`. No crash or new rejection.
- **7 latest-observation controls:** equal instants expressed as `Z` and `-04:00` with different grades; later file line with an earlier timestamp; quarantined JSON between qualified lines; a later qualified omission; a later disqualified slot; latest source-null grade; and latest wrong-typed grade. Expected references, grades/statuses, counts, quarantine counts and omission flags all pass.
- **4 genuine frozen-rule fit branches:** synthetic fixtures produce T24 totals `(191,157)`, `(191,156)`, `(190,157)`, `(190,156)`. The new checker accepts `both`, `primaries_only`, `legs_only`, `none` respectively, and the compiled tally label is `hypothesis` only in the first case. No change to `RULES` was needed.

The date-conflict tests establish which source date the compiler uses; they do not establish the historical truth of a contradictory file. The approved plan already constructs days from path-derived `file_date` (plan lines 4150–4156; current `compile.py:125–146`). The production writer uses the same `date` variable in the record and destination (`src/bts/daily_decision.py:86,101`), so I have not demonstrated such a discrepancy through that writer. I do not infer a wrong canonical date merely by choosing one side of a synthetic contradiction as authoritative.

### Scale and output-check cost

Reused the sealed r1 scale fixture without regenerating it: **3,600 files**, **2,335 unit captures**, **271,483,445 uncompressed unit bytes**, **411 contest lines × 136 rounds × two slots**, **220,295 emitted occurrences**. The contest census covers **111,792 slot occurrences**.

Four fresh child processes alternate the exact r2 checker and the revised checker in the **same r3 compiler**, on the same bundle. No other review test job ran during these measurements.

| Checker | Check duration, two runs | Whole compile duration | Peak RSS, bytes |
|---|---|---|---|
| r2 control | 0.445 / 0.439 s | 4.745 / 4.284 s | 1,002,405,888 / 1,000,718,336 |
| r3 | 0.745 / 0.746 s | 4.594 / 4.570 s | 1,082,933,248 / 1,089,503,232 |

The revised check costs about **0.30 additional seconds** here. Paired peak RSS increases are **80,527,360** and **88,784,896 bytes**. Unlike r2's streaming identity set, this census retains decoded slot facts, so the memory increase is consistent with the implementation. These synthetic measurements do not establish production timing or memory limits. No failure occurred at this scale.

All **six outputs are byte-identical across all four comparison builds**. Receipts and hashes: `P/r3_invariant_scale/results.json`, `P/r3_invariant_scale/output-hashes.json`. This run measures the full revised checker, including JSON decoding and census retention, not decoding alone.

## r2 findings

1. **Resolved — whole decision hidden by disposition. Class (b).** `reconcile.py:324–336` derives obligations from emitted source facts and `season_dates`, independently of the assigned disposition. An in-season decision marked outside the window is rejected; another disposition cannot exempt its named selections. All three independent omission probes pass, the null-game decision control passes, and the omission mutant is killed. The new in/out-of-season controls show that legitimately excluded 9/28 decisions and picks still compile.

2. **Resolved — fabricated recipe fit and label. Class (b).** `reconcile.py:348–353` independently computes fit from the counts and published totals and validates its label; membership totals remain checked at `:358–361`. The fabricated-fit probe passes and its mutant is killed. The new four-branch fixtures also establish that real matching/partially matching counts are accepted and yield the expected compiled recipe-level label.

3. **Resolved — an older contest grade accepted as latest. Class (b).** `reconcile.py:379–393` derives each identity's latest `(recorded_at,line_no)`, first/last observation times, observation count and grade/state/round label from emitted qualified slot occurrences. `:400–409` binds the canonical contest reference and grade to that occurrence. The stale-reference probe passes and first-snapshot mutant is killed. Equal-time differing grades, reordered timestamps, intervening/later quarantine, later omission, 411 repetitions and null/malformed grades all pass their independent controls.

4. **r1 #1 remainder: resolved for the demonstrated identity, completeness and grade failures. Class (b).** The source-date obligation closes the whole-decision escape; the independent latest-observation census closes the stale-grade escape. The earlier lost-leg, swapped-reference, missing-contest-identity and contest-only grade probes remain effective, and all three r1 mutants are still killed. This is bounded acceptance evidence, not a claim that every canonical column is independently proved; the remaining round-streak hardening opportunity is identified below.

## New findings

### 1. SHOULD — optionally extend latest-observation checks to carried round streaks

- **Class:** **(b) acceptance hardening** against a hypothetical future compiler bug. **Not a demonstrated class (a) defect.**
- **Location:** `scripts/audit/season_ledger/reconcile.py:390–392,407–409`; the existing correct value transfer is in `scripts/audit/season_ledger/contest.py:6–7,30` and `scripts/audit/season_ledger/compile.py:217,236`.
- **Failure scenario:** the latest qualified occurrence reports `round_streak=10`; a future output-construction bug changes the match's `round_streak` and canonical row's `streak_after` to 11 while preserving the latest occurrence id, grade and observation metadata → the checker accepts 11. Its latest-value comparison currently covers `slot_result`, `slot_result_state` and `round_result`, but not this round fact.
- **Demonstrated:** `P/test_review_r3_new.py::test_acceptance_hardening_latest_round_streak_not_checked`. It first asserts that the **unmodified compiler correctly emits 10**, matching the occurrence. It then changes only those two output fields to 11 and expects rejection; the checker does not raise. There is no demonstrated input that makes the current compiler emit 11, and I did not claim this additional mutation survives the repository suite.
- **Change I would make:** compare the match's carried round streak to its independently selected latest occurrence, and the canonical `streak_after` to that same fact. The same small extension can cover the other carried slot/round fields already present in the decoded census. Add this corruption probe as a regression test.
- **Disposition:** nonblocking follow-up for the owner to accept or defer. This does not reopen the resolved stale-grade finding or call for another planned review round.

## Verdict

SIGN — the accepted findings are resolved; no current-output defect or false rejection was demonstrated, with one nonblocking class (b) hardening recommendation retained for owner disposition.
