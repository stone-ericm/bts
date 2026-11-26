## Round

code r2 — `41af8f3..079ca47`.

Reviewed revision: `0d0ea954b1f1bed97f094cacfb30aceb3179d6a2..079ca4760a5e447c7a9cd7b818e0702c90684109` (seven commits), against spec v4 and plan rev4/I1–I15. This continues the r1 whole-branch review.

The original failures are substantially repaired. The remaining blockers concern acceptance: the new checks and full suite still accept three demonstrated corruptions of canonical output or its interpretation. The unmodified compiler produces the expected results on the corresponding synthetic controls. I am not claiming these mutations exist in the checked-in compiler or that any real output is wrong.

Tracked files remained unchanged. All new files are under `.codex-review/season-ledger-code/`. No real data or real snapshots were read; no network, SSH, herdr, or other pane was accessed. Acquisition tests use their synthetic scratch tree and mocked fetches.

## Replay

Commands ran from the worktree root, with this environment on each Python invocation:

```sh
PYTHONDONTWRITEBYTECODE=1 TZ=America/New_York TMPDIR=/Users/eric/projects/bts/.claude/worktrees/season-ledger-phase1/.codex-review/season-ledger-code/tmp
```

Commands below follow that environment prefix. `P` in prose means `.codex-review/season-ledger-code/probes/`; the actual command paths are expanded.

```sh
.venv/bin/python -m pytest -q -p no:cacheprovider tests/scripts/season_ledger
.venv/bin/python -m pytest -q -p no:cacheprovider --tb=short .codex-review/season-ledger-code/probes/test_review_r2_ported.py
.venv/bin/python .codex-review/season-ledger-code/probes/run_mutants_r2.py
.venv/bin/python -m pytest -q -p no:cacheprovider --tb=short .codex-review/season-ledger-code/probes/test_review_r2_new.py
.venv/bin/python .codex-review/season-ledger-code/probes/run_new_mutants_r2_verified.py
.venv/bin/python .codex-review/season-ledger-code/probes/run_scale_r2.py
.venv/bin/python .codex-review/season-ledger-code/probes/run_invariant_scale_r2.py
```

**Measured results:**

- Repository suite: **160 passed**, initially 1.31 s; final unmodified-tree control **160 passed in 0.63 s**, saved in `P/r2-suite-final.txt`.
- Ported r1 probes: **11 passed, 1 failed in 0.51 s**, `P/r2-ported.txt`. The sole failure is the deliberately superseded G4 expectation.
- New probes: **24 passed, 5 failed in 0.62 s**, `P/r2-new-final.txt`. The failures are three variants of whole-decision omission, fabricated recipe fit, and stale contest evidence. Each failed because the unchanged checker **did not raise** for corrupted arguments. Each starts by compiling and checking an uncorrupted fixture successfully.
- All three original r1 mutants are now killed by both the repository suite and independent witnesses, with the same recipe fingerprint. Receipts: `P/r2-mutant-results.json`.

| Original mutant | Revised repository suite | Independent witness |
|---|---:|---:|
| Omit evidenced contest-only row | 1 failed / 159 passed | 1 failed |
| Drop T24 after evaluation | 15 failed / 145 passed | 1 failed |
| Copy round grade onto contest-only slot | 1 failed / 159 passed | 1 failed |

**Every port change/rename:**

- `test_review_r1.py` → `test_review_r2_ported.py`. No test, helper, production function, field, or assertion was renamed. The only semantic fixture adaptation imports `DAY_KINDS` and `re`, and adds path-derived `file_date` to parsed day records in `source_account` before `account()`. This mirrors `compile.py:125–129`; direct parser calls previously omitted this compiler enrichment. The new identity checks legitimately need it. All r1 assertions remain unchanged.
- `run_mutants.py` → `run_mutants_r2.py`; copied witness filename becomes `test_review_r2_ported.py`; scratch directory `mutants_verified` → `mutants_r2`; logs `mutant_<name>_{suite,probe}.txt` → `r2_mutant_<name>_{suite,probe}.txt`; receipt `mutant-results.json` → `r2-mutant-results.json`. No mutation anchor or replacement needed changing.
- The scale replay is a separate `run_scale_r2.py`: it reuses the existing r1 sealed bundles and writes `r2_scale_and_determinism/`, without regenerating input.

Ported test disposition (all names below are in `P/test_review_r2_ported.py`):

| Test | Result |
|---|---|
| `test_invariant_rejects_missing_decision_only_double_leg` | pass |
| `test_invariant_rejects_swapped_same_kind_references` | pass |
| `test_invariant_rejects_an_entire_missing_recipe` | pass |
| `test_invariant_rejects_dropped_contest_identity` | pass |
| `test_evidenced_contest_only_survives_and_uses_its_slot_grade` | pass |
| `test_malformed_delivered_pick_is_other_selection_commit_evidence` | pass |
| `test_malformed_delivery_signal_does_not_prove_a_clean_skip` | both parameters pass |
| `test_wrong_schedule_date_cannot_support_inference` | pass |
| `test_g4_counts_slot_objects_only` | fails as expected under `ecebb95`/r2 ruling |
| `test_naive_manifest_mtime_is_not_interpreted_in_host_zone` | pass |
| `test_compiled_recipe_set_is_complete` | pass |

The new mutant runner preserves separate source/test copies and logs the imported compiler path and fingerprint. Its final receipts are `P/r2-verified-new-mutant-results.json`. All three mutants retain fingerprint `5e9d74f2f9c3093d66bc7c9ab0a7028e5fb361cdb46bbb1b368cd71e7f297b3f`:

| New mutant | Repository suite | Independent witness |
|---|---:|---:|
| Omit decisions with an unrecorded primary game | 160 passed | fails: `unobserved_day` instead of selection |
| Overwrite T24 fit/label after evaluation | 160 passed | fails: fabricated `fit=both` |
| Pass only first contest line to `slot_history` | 160 passed | fails: `hit` instead of latest `not_hit` |

Initial new-mutant receipts are retained too. I then made the omission witness assert nullable fields with `.get()` so its failure reports the wrong row kind directly, and moved the repeated-slot grade assertion ahead of its observation-count assertion. The verified runner above reran all three copies with those clearer assertions. These were probe diagnostics changes, not production changes.

**Valid shapes and day behavior:** eight double-down cases passed, with and without contest evidence: same batter/different games, different batters/same game, identical batter/game across local roles, and unrecorded games. Ambiguous local attachment stays unlinked. Decision-only null-game input also passes. A contest-only date retains its day row and one contest row after 411 observations, both with and without a unit mapping. Twelve delivered/private/locked-unconfirmed/skip cases pass, including positive delivery taking precedence over another malformed field, private status without a positive signal, and malformed commit evidence blocking a clean skip. No false rejection was demonstrated in these controls. See `P/test_review_r2_new.py`.

**Scale and determinism:** reused the r1 fixture: 3,600 files, 2,335 unit captures, 271,483,445 uncompressed unit bytes, 411 contest lines × 136 rounds × two slots; 220,295 emitted occurrences. The r2 replay succeeded. Three small-fixture runs with different hash seeds, time zones and locales produced identical bytes for all six outputs (`P/r2_scale_and_determinism/determinism-results.json`).

To isolate the extra checker work, `run_invariant_scale_r2.py` loads the r1 checker from `git show 0d0ea95:scripts/audit/season_ledger/reconcile.py` into a scratch module, then alternates old/new checkers in four fresh child processes running the **same r2 compiler and sealed large bundle**. No other review test job was running during this comparison.

| Checker | Check duration, two runs | Whole compile duration | Peak RSS |
|---|---|---|---|
| r1 control | 0.138 / 0.142 s | 4.229 / 3.872 s | 999,636,992 / 1,019,281,408 bytes |
| r2 | 0.435 / 0.438 s | 4.145 / 4.245 s | 999,620,608 / 1,019,559,936 bytes |

The revised check adds about **0.297 s** here. Whole-build timings vary; these four runs do not establish a precise whole-build percentage. Peak RSS shows no material increase in this experiment. All six outputs are byte-identical across the four comparison builds. Receipts/hashes: `P/r2_invariant_scale/results.json` and `output-hashes.json`. This measures the complete revised check, including its extra JSON parsing, on synthetic scale; it does not measure production performance.

## r1 findings

1. **Partially resolved — output identity/completeness.** `reconcile.py:310–328` now checks date, slot, attached-pick identity and named decision legs; `:364–387` checks qualified contest identities, placement and grade/reference agreement. The four ported invariant probes pass, and the two contest-output mutants are killed. However, decision completeness still depends on an output-derived disposition, and the selected contest observation is not independently shown to be latest. New findings 1 and 3 demonstrate these remaining gaps.

2. **Resolved — an entire frozen recipe could disappear.** `reconcile.py:331–343` derives the required rule set from `RULES`, checks its frozen definition and requires every rule × universe slot. The original missing-recipe probe passes and the T24-deletion mutant is killed. New finding 2 concerns a separate unchecked interpretation of retained counts, not disappearance of a rule.

3. **Resolved — valid delivery lost when every pick slot quarantines.** `rows.py:164–165` falls back to `pick_file.file_fields`; `sources/pick_files.py:83–88` preserves those fields independently of usable slots. The original malformed-delivered-pick probe now reports `unresolved` plus `conflicted`, as required by I14. Fixed in `733e651`.

4. **Resolved — wrong-typed delivery/commit fields became evidence of absence.** `rows.py:38–39` returns `delivery_fields_unreadable`; `:190–198` prevents a clean skip for that signal or an unreadable scheduler commit flag. The file summary now retains mismatch names (`sources/pick_files.py:84–86`). Both original malformed-delivery parameters pass, as do the new normal-delivery/skip controls. The repository's no-usable-slot case also passes. Positive delivery still wins per I4. Fixed in `733e651`, with the added test in `079ca47`.

5. **Resolved — schedule body/query-date mismatch.** `sources/static.py:118–121` quarantines a date container that disagrees with the query filename, making the schedule incomplete; `acquire.py:37–40` rejects the mismatched fetch response. The original inference probe passes; the mocked acquisition and parser tests pass. The check correctly concerns the response date container, not a resumed game's `officialDate`. Fixed in `f27f705`.

6. **Resolved by the explicit pre-exposure ruling.** `ecebb95` and plan lines 2782–2784 pin non-null presence. The original object-only probe intentionally remains red; the repository test pins false, zero, list, object and null. `src/bts/picks.py:209,301–313` supports the normal producer shape (`Pick | None`, serialized with `asdict`). I did not verify the contents of real files. For malformed non-null values, the declared recipe counts a slot while canonical parsing quarantines it; that distinction is now explicit and preserves the frozen fingerprint. I have no concrete consequence that warrants rejecting this ruling.

7. **Resolved — naive manifest mtime depended on host time zone.** `compile.py:117–118` normalizes through the offset-required `utc_iso` before recipe evaluation. The original UTC/New York probe returns `[None, None]`; the repository mtime test and cross-process determinism replay pass. Fixed in `cd8fcab`.

## New findings

### 1. BLOCKER — a whole usable decision can evade the completeness check

- **Location:** `scripts/audit/season_ledger/reconcile.py:321–328`; `scripts/audit/season_ledger/compile.py:243–254`.
- **Failure scenario:** an in-season usable decision is omitted from day construction → the compiler leaves its occurrence `outside_season_window` and emits an ordinary `unobserved_day` → the new decision check skips it because it is not marked `canonical_decision`. Thus the output decides whether the source was required to appear. Removing only one leg is detected; removing the whole decision can pass.
- **Demonstrated:** `P/test_review_r2_new.py::test_invariant_rejects_whole_decision_hidden_by_disposition` accepts this corruption for single, double and skip decisions. The source occurrence remains fully accounted. A compile-only mutant filters decisions on `primary_game_pk is not None`; all **160 repository tests pass**, and compiling the supported null-game decision writes an unobserved day. `test_unrecorded_game_decision_is_retained` catches it. Mutation and logs are under `P/new_mutants_r2_verified/omit_unrecorded_game_decision/` and `P/r2_verified_new_mutant_omit_unrecorded_game_decision_*.txt`.
- **Change:** derive the expected in-season decision obligations from emitted decision facts plus `season_dates`, independently of assigned dispositions. Require every usable single/double decision's exact named rows and every usable skip decision's one referenced day row. Validate that `outside_season_window` actually means an out-of-season source date. Add an end-to-end decision-only null-game case and a whole-decision omission mutant.

### 2. BLOCKER — retained recipe counts can carry a fabricated matching verdict

- **Location:** `scripts/audit/season_ledger/reconcile.py:333–347`, `:529–541`; `scripts/audit/season_ledger/compile.py:304`.
- **Failure scenario:** T24 retains its frozen definition and correct membership/counts, but `fit` is changed from `none` to `both` and its label claims reproduction → all invariant checks pass → `recipe_labels` reports the 9/14 tally as `hypothesis` instead of `unrecoverable`. On the one-pick fixture, **1 primary / 0 legs** is accepted as matching published **191 / 157**. Checking the rule definition and sum does not validate the interpretation subsequently published.
- **Demonstrated:** `P/test_review_r2_new.py::test_invariant_rejects_fabricated_recipe_fit` does not raise. The compile-only `fabricate_t24_fit` mutant changes those two fields after `evaluate_rules`; all **160 repository tests pass**, with the unchanged recipe fingerprint. `test_compiled_recipe_fit_agrees_with_counts` fails. Evidence: `P/new_mutants_r2_verified/fabricate_t24_fit/mutation.json` and corresponding `P/r2_verified_new_mutant_fabricate_t24_fit_*.txt` logs.
- **Change:** check each summary's `fit` against its two verified totals and frozen published totals, and validate its label. Derive or validate the final recipe-level label from those comparisons. Add a compiled nonmatching fixture assertion for `tally_0914=unrecoverable`, plus this post-evaluation mutation test. This follows spec §8 and I9 without changing the frozen counting rules.

### 3. BLOCKER — correct slot identity can conceal use of an older grade

- **Location:** `scripts/audit/season_ledger/reconcile.py:364–384`; integration at `scripts/audit/season_ledger/compile.py:164–165`.
- **Failure scenario:** a slot is observed first as `hit`, later as `not_hit` → both the match's `last_obs_id` and the ledger reference point to the first occurrence → the checker verifies the right slot identity and agreement with that occurrence, but never establishes that it is the last qualified observation. Earlier and later occurrences have the same identity, so the set comparison also passes. The resulting `hit` contradicts spec §6's last-observed-grade contract.
- **Demonstrated:** `P/test_review_r2_new.py::test_invariant_rejects_stale_contest_observation` accepts the stale reference/grade on two observations. The `first_contest_snapshot_only` mutant feeds only line 1 to `slot_history`, retaining full source accounting; all **160 repository tests pass**. On 411 observations of the same identity, the compiled grade is `hit` instead of the final `not_hit`, and observation count becomes 1. `test_contest_only_and_many_reobservations_are_accepted[True]` catches the wrong grade. Evidence: `P/new_mutants_r2_verified/first_contest_snapshot_only/mutation.json` and corresponding verified logs.
- **Change:** while decoding qualified occurrence facts, independently retain each identity's maximum `(recorded_at, line_no)` and expected occurrence id. Check `matches.last_obs_id` and ledger references against it, with the associated typed grade/state. Check first/last times and observation count from the same census. Add a full compile fixture with changing/repeated observations, equal-time line ordering, and later omissions; the source parser/`slot_history` unit tests alone do not exercise this integration boundary.

## Verdict

BLOCK — r1 fixes pass their direct replays, but three surviving mutants show that the revised acceptance checks still permit missing decisions, false recipe matches and stale contest grades.
