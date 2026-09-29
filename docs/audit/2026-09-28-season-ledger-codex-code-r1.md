## Round

code r1 — range `41af8f3..0d0ea95` (`41af8f3e2b934563ec0c5914944de2e50aeea979..0d0ea954b1f1bed97f094cacfb30aceb3179d6a2`).

Part 1 was written before opening the earlier Codex reviews or `.superpowers/`. Authority: design v4, plan rev 4 (Global Constraints, I1–I15, Review Focus, Tasks 1–11), and the four fix commit messages. Task 13 was not executed. All evidence below is synthetic; no real snapshot, pick, contest, or ledger data was read. No network, SSH, herdr, commits, or tracked-file edits.

## Replay

Working directory: `/Users/eric/projects/bts/.claude/worktrees/season-ledger-phase1`. Probe paths below are relative to `.codex-review/season-ledger-code/`.

1. Read-only range checks:

   ```sh
   git log --format='%h %s' 41af8f3..0d0ea95
   git diff --stat 41af8f3..0d0ea95
   git show -s --format='%H%n%B' 84b77ff b703a06 dc80c74 0d0ea95
   git diff --name-only 0d0ea954b1f1bed97f094cacfb30aceb3179d6a2 -- scripts/audit/season_ledger scripts/audit/build_season_ledger.py tests/scripts/season_ledger
   ```

   Fifteen commits; 29 changed files, 3,874 insertions. The final command returned no paths. Read the reviewed package, its tests, binding documents, and the relevant production decision/pick serializers using `cat`, `sed`, `nl`, and scoped `rg` searches.

2. Baseline, before probes:

   ```sh
   TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 TMPDIR=/Users/eric/projects/bts/.claude/worktrees/season-ledger-phase1/.codex-review/season-ledger-code/tmp .venv/bin/python -m pytest -q -p no:cacheprovider tests/scripts/season_ledger
   ```

   **151 passed in 0.74 s**, exit 0, Python 3.12.13 / pytest 9.0.2.

3. Independent probes:

   ```sh
   TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 TMPDIR=/Users/eric/projects/bts/.claude/worktrees/season-ledger-phase1/.codex-review/season-ledger-code/tmp .venv/bin/python -m pytest -q -p no:cacheprovider .codex-review/season-ledger-code/probes/test_review_r1.py --tb=short
   ```

   **10 failed, 2 passed in 0.98 s**. These failures are assertions of the desired contracts, not expected-failure markers. Receipt: `probes/probe-results.txt`. The two passing controls verify evidenced contest-only output and the complete compiled recipe set on unmodified code. The earlier probe iteration had 9 failures / 2 passes; the final iteration adds the same-kind reference swap and strengthens the invariant inputs.

4. Isolated mutants:

   ```sh
   PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .codex-review/season-ledger-code/probes/run_mutants.py
   ```

   The runner copies only reviewed source/tests into `probes/mutants_verified/`, records the exact source replacement, prints the imported module path and fingerprint, and invokes the full existing suite plus an independent witness. All three mutants kept fingerprint `5e9d74f2f9c3093d66bc7c9ab0a7028e5fb361cdb46bbb1b368cd71e7f297b3f`.

   | Mutant | Existing suite | Independent witness |
   |---|---|---|
   | `omit_evidenced_contest_only` | 151 passed, 1.01 s | failed: expected one row, got zero |
   | `contest_only_round_grade` | 151 passed, 0.49 s | failed: `used_mulligan` instead of `not_hit` |
   | `drop_t24_after_evaluation` | 151 passed, 0.52 s | failed: T24 absent |

   Exact commands, exits, timings and logs: `probes/mutant-results.json`, `probes/mutant_*_{suite,probe}.txt`. An initial runner setup attempt used the wrong parent-directory index and stopped before copying source; it was corrected, and the successful runs verify their imports point into the isolated copies.

5. Cross-process determinism and scale:

   ```sh
   PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .codex-review/season-ledger-code/probes/run_scale_and_determinism.py
   ```

   All six output files were byte-identical for the same well-formed synthetic bundle under `(PYTHONHASHSEED,TZ,LC_ALL)` = `(0,UTC,C)`, `(7,America/New_York,C)`, `(9182,Pacific/Honolulu,en_US.UTF-8)`. The existing suite also passes relocated-root/discovery-order checks. This positive result does not cover the accepted naive-mtime case in finding 7.

   The scale fixture has **3,600 files**, **2,335 mixed plain/gzip unit captures**, **271,483,445 uncompressed unit JSON bytes**, and **411 contest lines × 136 rounds × 2 slots**. Total stored input: 152,525,808 bytes; contest JSONL: 15,199,191 bytes. Compile exited 0, emitted 220,295 occurrence rows, and took **4.406 s inside the compiler / 4.548 s process wall time**, with **1,006,862,336 bytes peak RSS** on this Mac. This is a synthetic resource measurement, not a prediction of box runtime or proof of real-source format coverage. Receipts and hashes: `probes/scale_and_determinism/{determinism-results,scale-results}.json`.

6. CLI smoke check:

   ```sh
   PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/audit/build_season_ledger.py compile --help
   ```

   Exit 0; lists `--bundle`, `--out`, `--code-sha`, and `--uv-lock`. Final deliverable checks compare all 29 reviewed files with `0d0ea95`, verify the required headings and probe receipts, and recheck the sealed Part 1 hash. Receipt: `probes/final-verification.json`.

## Part 1 findings

### 1. BLOCKER — output checks allow lost selections, missing contest rows, and wrong evidence links

- **Location:** `scripts/audit/season_ledger/reconcile.py:289`, `:297`, `:319`, `:338`; uncovered emitter: `scripts/audit/season_ledger/compile.py:222`.
- **Failure scenario:** a decision-only double is reduced to its primary → `check_sources` and `check_invariants` still pass because a canonical decision needs only one reference. Two single decisions can have their `decision_obs_id` values swapped across dates/batters → the same checks pass because reference validation checks only source kind. Qualified contest slots can disappear from the match/ledger outputs → source occurrences still have `contest_evidence` dispositions, with no required join to the output identities. The one-way uniqueness checks do not establish output completeness or correct identity.
- **Demonstrated:** `probes/test_review_r1.py::test_invariant_rejects_missing_decision_only_double_leg`, `::test_invariant_rejects_swapped_same_kind_references`, and `::test_invariant_rejects_dropped_contest_identity` all fail with **DID NOT RAISE** on current checks, after source checks pass. Additionally, the isolated `omit_evidenced_contest_only` mutant suppresses only evidenced contest-only ledger rows, and `contest_only_round_grade` copies the round result onto those rows. Both compile and pass all **151 existing tests**. The added contest-only witness passes on current code and fails on each mutant. These two emitter corruptions are planted defects, not claims about the current emitter.
- **Change:** independently require the expected selection identities for each usable single/double decision; verify references against date, slot, batter and game; require every qualified contest identity exactly once in the contest table and in either its linked selection or a contest-only row. Validate transferred grades against the referenced qualified slot occurrence. Add the evidenced contest-only raw-file fixture, including a round label different from its slot result. This closes the spec §5/§7/§8 guarantee beyond the source census.

### 2. BLOCKER — an entire frozen recipe can disappear without changing the fingerprint or failing checks

- **Location:** `scripts/audit/season_ledger/reconcile.py:301`; integration boundary: `scripts/audit/season_ledger/compile.py:255`.
- **Failure scenario:** remove T24 from both `summary` and `membership` after evaluation → the expected rule set is derived from that same shortened summary, so the output checks pass. The compiler writes 31 candidate rules with the original fingerprint. A matching alternative could be lost, changing the reconciliation claim.
- **Demonstrated:** `probes/test_review_r1.py::test_invariant_rejects_an_entire_missing_recipe` fails on current checks. The isolated `drop_t24_after_evaluation` mutant passes **151 tests**, keeps the exact predeclared fingerprint, and fails `::test_compiled_recipe_set_is_complete`. Logs and exact replacement are in `probes/mutants_verified/drop_t24_after_evaluation/` and the mutant receipts.
- **Change:** require summary rule IDs to equal the frozen `RULES` keys exactly once; derive expected membership from `RULES`, not summary; bind each summary's recipe/window/grading/targets to its frozen definition. Add an output assertion for all S1–S8 and T1–T24. The current fingerprint binds the evaluator, but does not certify that all of its results reach the output.

### 3. BLOCKER — quarantining every pick slot discards its delivery evidence in a selection conflict

- **Location:** `scripts/audit/season_ledger/rows.py:159`, `:174`; `commit_status` at `:74`.
- **Failure scenario:** a single decision names batter 101/game 5001; the surviving pick names batter 202 with a malformed game ID and carries `notification_sent=true, notification_id="dm-other"` → its only slot is quarantined. The decision row correctly becomes `finalization=unresolved`, but incorrectly remains `commit_status=committed_evidenced` with no other-selection delivery basis. `file_pick` is taken only from usable slot rows; the preserved `pick_file.file_fields` is consulted only by the skip branch.
- **Demonstrated:** `probes/test_review_r1.py::test_malformed_delivered_pick_is_other_selection_commit_evidence` runs raw files through compile and fails: expected `conflicted`, observed `committed_evidenced`. Both build checks passed.
- **Change:** pass the readable file-level delivery evidence into selection commit evaluation even when all slots are quarantined. An incomplete/different whole selection set must retain its delivery as conflicting evidence, while attaching no pick facts to the decision selection (I14 and spec §5).

### 4. BLOCKER — malformed delivery fields turn unknown evidence into a clean skip

- **Location:** `scripts/audit/season_ledger/sources/pick_files.py:84`; `scripts/audit/season_ledger/rows.py:180`–`:192`.
- **Failure scenario:** a normal skip decision plus a pick file with `notification_sent="true"` and a DM id, or `delivery_attempted="true"` → the typed value becomes null. The file is still considered readable/complete, `pick_delivery` returns `no_delivery_evidence`, and compile emits `skip_day`. Wrong-typed evidence is not evidence that no commit/delivery happened.
- **Demonstrated:** both parameterizations of `probes/test_review_r1.py::test_malformed_delivery_signal_does_not_prove_a_clean_skip` fail on current compile: observed `skip_day`, expected `unfinalized_day`. The occurrence retains the mismatch, but the day rule ignores it.
- **Change:** preserve field-state/mismatch information in `pick_file_state`; let unusable delivery/commit fields block a clean skip without treating their raw strings as positive confirmation. Apply the same principle to scheduler commit fields. This is a field-level gap beyond I15's whole-record quarantine cases and the final fix's correctly typed unconfirmed-delivery cases; the governing requirement is spec §5's “no committed pick.”

### 5. BLOCKER — a schedule body for another date can authorize an inferred official grade

- **Location:** `scripts/audit/season_ledger/sources/static.py:112`, `:129`; `scripts/audit/season_ledger/compile.py:154`; `scripts/audit/season_ledger/contest.py:150`.
- **Failure scenario:** `schedules/2026-08-20.json` contains `dates[0].date="2026-08-19"` and one TB game 5001. A local 8/20 selection names TB/game 5001; its contest unit has no capture → the parser ignores the body's date, labels the schedule complete for 8/20, and compile transfers the contest grade with `match=inferred, entry_status=confirmed`. The input does not evidence that team's unique game on the selection date.
- **Demonstrated:** `probes/test_review_r1.py::test_wrong_schedule_date_cannot_support_inference` fails after a successful full compile: `confirmed` instead of `unknown`, with the grade transferred.
- **Change:** validate the schedule date container against the requested date, preserving and quarantining missing/mismatched date evidence so inference is blocked. Validate the response in acquisition too, where practical. Compare the container's date; do not require a resumed game's historical `officialDate` to equal the query date. The fix commit's `dates`-list check alone does not bind the response to its request.

### 6. SHOULD — G4 counts malformed non-object values as double-down slots

- **Location:** `scripts/audit/season_ledger/reconcile.py:384`, `:420`; plan Task 9 at `docs/superpowers/plans/2026-09-28-season-ledger-phase1.md:2782`.
- **Failure scenario:** a file has a valid primary object and `double_down=false` → the source parser quarantines that leg as `slot_not_object`, but G4 counts it because the membership loop tests only `is not None`. T8 reports one leg although the declared rule is “every slot object counts.”
- **Demonstrated:** `probes/test_review_r1.py::test_g4_counts_slot_objects_only` fails on current compiled output: expected zero legs, observed one.
- **Change:** pin whether G4 counts objects or arbitrary non-null slot values. To follow the declared object rule, keep malformed occurrences accounted for but exclude them from G4 counts, with a reason. Add null/false/number/list/object cases and update the predeclared rule/fingerprint documents through the approved pre-run process. Do not silently reinterpret a frozen rule after exposure. No claim is made that these malformed values exist in the real snapshot.

### 7. SHOULD — accepted naive manifest mtimes make output bytes depend on the host time zone

- **Location:** `scripts/audit/season_ledger/reconcile.py:394`–`:400`; `scripts/audit/season_ledger/bundle.py:50`.
- **Failure scenario:** the same sealed bundle contains `source_mtime_utc="2026-09-11T01:00:00"` without an offset → `_mtime_after` calls `astimezone` on a naive datetime. Under `TZ=UTC`, S1's `mtime_after_recipe_date` is `false`; under `TZ=America/New_York`, it is null. The same inputs therefore produce different reconciliation bytes.
- **Demonstrated:** `probes/test_review_r1.py::test_naive_manifest_mtime_is_not_interpreted_in_host_zone` compiles the same bundle twice and fails with `[False, None]` instead of `[None, None]`.
- **Change:** reject naive/invalid manifest times at the bundle boundary, or normalize them to unknown with the existing offset-required time policy before recipe evaluation. The acquisition implementation writes offset-aware UTC mtimes, so this is an accepted-input validation gap; it did not reproduce on the well-formed cross-process fixture.

## Part 2 reconciliation

After writing Part 1, read all seven `docs/audit/2026-09-28-season-ledger-codex-{design-r1,design-r2,design-r3,design-r4,plan-r1,plan-r2,plan-r3}.md` reviews, `.superpowers/sdd/2026-09-28-season-ledger-phase1/progress.md` including every `Ruling:`, and its four final-fix messages. Part 1 is preserved verbatim; its SHA-256 is recorded in `probes/part1-seal.json`.

| This finding | Earlier finding / ruling | Disposition after reconciliation |
|---|---|---|
| 1 — output completeness and identity | Design r1 #3/#4/#8; plan r1 #1 and plan r2 #1 | The original **source-census** omissions are addressed by current tests/checks. The broader promise to reconcile occurrences with their correct outputs still fails: a decision leg or contest-only output can vanish, and references can name the wrong same-kind occurrence. The three independent invariant probes and two surviving emitter mutants establish the remaining gap. No ruling settles it. |
| 2 — missing entire recipe | Plan r3 #3/#4; internal deferred minor #10 (“fingerprint binds module RULES, not the argument”) | Wrong-source membership, individual missing rows, predicate edits and regex flags have current protection. The expected rule universe still comes from the produced summary. Internal #10 acknowledges an adjacent integration risk; the new T24 mutant makes a concrete failure outside the frozen evaluator while the fingerprint and full suite stay green. |
| 3 — lost conflicting delivery | Plan r3 #5; I14/I15 | The old **skip-branch** case now retains file-level delivery, and unusable decisions/history no longer read as absent. The single/double branch still drops the same file-level evidence when every slot is quarantined. Thus the earlier preservation requirement is only partly implemented. |
| 4 — malformed delivery → skip | Internal final #4/#5 and their explicit I15 extension | Correctly typed attempts and sent-without-id signals are fixed. Wrong-typed signals are a new uncovered case. The ruling expressly says an attempt is unknown, not no, and a clean skip asserts no committed pick; it supports preserving uncertainty here rather than settling the case against this finding. |
| 5 — wrong-date schedule | Plan r1 #7; internal final #3 | Missing game/team metadata and non-schedule response bodies are handled. Date binding is another remaining part of the earlier full-envelope validation requirement. Neither the endpoint description nor the retry ruling supplies that check. |
| 6 — G4 object contract | Plan r1 #10; Task 9's added G4 | The earlier requested candidate was explicitly an ungraded primary/DD **object** count. No later ruling authorizes counting `false` as an object. The implemented non-null rule needs reconciliation before the real run. |
| 7 — naive mtime | No earlier matching finding | New accepted-input determinism case. Plan r1's statement that no separate timezone defect was demonstrated then is not evidence against this probe. Internal minor #12 concerns per-input acquisition timestamps, a different issue. |

The four internal fix commits hold for their stated cases: impossible capture stamps are unknown; identity checks hash once per file; schedule fetch retries reject non-schedule bodies; correctly typed unconfirmed delivery blocks a clean skip. I do not reopen the rulings accepting per-build directories, the non-binding scheduler commit flag, unknown eligibility without an evidenced lock, null unobserved attempt times, or confirmed contest-only entry with unknown local role. The synthetic scale result supplies a bounded memory measurement; it does not certify the actual snapshot or box. Task 13's publication/backup/runner acceptance remains outside this code review's execution scope.

## Verdict

BLOCK — the green suite and current output checks do not establish the required accounting and grade-transfer guarantees.
