## Round

plan r1 — `docs/superpowers/plans/2026-09-28-season-ledger-phase1.md`, commit `1696394` (current HEAD verified), against spec v4, `docs/superpowers/specs/2026-09-28-season-ledger-design.md`.

This is a plan/code-fence review. No real data, snapshot, network, SSH, or unauthorized audit memo was accessed. The supplied source facts are accepted as inputs; synthetic counterexamples below are not claims that those records exist in the snapshot. The design r4 SIGN remains a design acceptance, not implementation acceptance.

## Replay

Extracted the plan's 33 Python fences into `.codex-review/season-ledger/replay-plan-r1/`, preserving the specified package layout and append order. Added the five package markers requested by the brief and the `Counter` import explicitly directed by Task 6 Step 3. The implementation was otherwise unchanged. `extraction-map.json` records the fence/file mapping.

From that replay directory, ran:

```sh
TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/Users/eric/projects/bts/.codex-review/season-ledger/replay-plan-r1/tmp \
/Users/eric/projects/bts/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/scripts/season_ledger \
  --basetemp=/Users/eric/projects/bts/.codex-review/season-ledger/replay-plan-r1/tmp/baseline
```

**Measured: 112 passed, exit 0**, before adding independent probes. The plan's final count is correct; Task 10's 111 plus Task 11's acquisition test gives 112. Python was 3.12.13 and pytest 9.0.2. This establishes that the fences compose locally, not that the acceptance conditions are sound.

Then added `tests/scripts/season_ledger/test_review_probes.py` and ran the same command with that test file as the target and `--basetemp=.../tmp/probes-final`.

**Measured: 14 failed, exit 1.** The probes assert the safer/spec-conforming behavior described below; each failure records the contrary behavior. Ten exercise source parsing, row/accounting rules, recipe membership or compilation; two directly probe eligibility's temporal contract; one mutates a rule to challenge the claimed freeze test; and one tests gzip fixture stability. The gzip probe does not test compiler determinism on identical input bytes.

Full outputs are preserved in `replay-plan-r1/baseline-result.txt` and `replay-plan-r1/probes-result.txt`. All generated fixtures and outputs are synthetic and remain inside the authorized review directory. Existing repository suites and box commands were not run.

## Findings

1. **BLOCKER — The acceptance check does not detect an omitted occurrence.** **Location:** Task 9 Steps 1 and 3 (`account`, `check_invariants`); Task 10 Step 3.

   **Measured:** parse one production file containing primary and DD, construct its accounting, delete only the DD accounting row, and run `check_invariants`. It accepts. The check compares file-path sets, checks duplicates among rows that survived, and requires a nonempty disposition. The primary still accounts for the file. Nothing compares the expected occurrence multiset with the actual one. Giving every parser-emitted row a disposition is not an anti-join; a parser dropping a nested record is also invisible.

   The existing omitted-occurrence test removes a whole single-occurrence file, so it cannot expose this defect. Arbitrary nonempty dispositions also pass without a corresponding canonical/observation target.

   **Change:** enumerate expected record/slot locators from source structure, retain their content hashes, and reconcile that multiset against emitted/excluded/quarantined occurrences and their referenced outputs. Validate disposition values and targets. Add deletion and duplication mutants inside a multi-slot file and a multi-line contest file. This is spec §§4, 8 and 11's acceptance boundary.

2. **BLOCKER — The promised lossless observations are neither lossless nor persisted.** **Location:** Task 2 Step 4; Task 3 Step 4; Task 4 Step 4; Task 10 Step 3.

   `parse_pick_file` drops production provenance `feature_env_schema_version`, `feature_env`, and `policy_decision`. Production `DailyPick` records these fields; `scheduler._selection_decision_meta` explicitly uses the saved policy decision on restart/fallback. **Measured:** a pick-only record containing `policy_decision.objective = emax_season_best` loses it and compiles with canonical `objective = null`. At minimum the recorded provenance must survive; a same-record canonical fallback also needs an explicit rule.

   **Measured:** absent `pick.pitcher_id` and explicit null produce the same value and the same `absent_fields`; that mask covers file-level fields only. Decision/state fields use `.get` without presence metadata. Scheduler parsing reduces `final_skip_candidate` to two identity fields, refusal records to archive names, and refresh records to a count, losing the source values specified in O4.

   The output schemas deepen the loss: occurrences contain only path, a locator populated with an opaque observation hash, kind/state/reason/disposition. They do not contain the parsed observations, their actual locators, or source hashes. Contest output retains the last slot values and a `changed` flag, not the earlier values. Archive metadata, earlier contest values, and saver-attempt details disappear from the compiled observation layer. The bundle keeps raw bytes, but using them requires re-parsing; it does not fulfill the stated observation-table contract.

   **Change:** persist O1–O7 observations, either in typed tables or a lossless raw-value envelope alongside normalized fields, with presence information and resolvable occurrence references. Preserve complete provenance and structured scheduler records. Test absent/null/zero at nested paths and round-trip earlier observations. Do not describe these parsers as lossless until this passes.

3. **BLOCKER — I3 attaches an unnamed day-level commit flag to a particular surviving preview.** **Location:** Task 8 Steps 1 and 3 (`commit_status`); I3.

   **Measured:** with no decision, an undelivered preview B, and `committed_pick_written = true`, B becomes `committed_evidenced` with basis `scheduler_commit_flag`. The state contains no candidate identity. It can be evidence that some selection committed while a different surviving file remains, not that B committed.

   This directly reopens design r4's resolved boundary: its SIGN explicitly said a date-level scheduler flag is not permission to attach another selection's commit evidence. The plan's explanation that the flag “names no selection” is correct; its fallback to the surviving file does not supply the missing binding.

   **Change:** require evidence binding the flag to that selection/revision. Otherwise preserve the state observation and leave the selection `unconfirmed`, unless its own decision or delivery proves commitment. Replace the test that pins the unsafe behavior.

4. **BLOCKER — A single/double selection-set conflict is silently resolved by per-slot equality.** **Location:** Task 8 Steps 1 and 3 (`day_rows`); Task 10 Step 3.

   **Measured:** decision says single A; surviving delivered pick file says A + DD B. The compiler emits A with `finalization = decision`, attaches the pick file's delivery, and leaves B merely `not_selected`. It never marks the conflicting whole-file view unresolved. The same primary makes `same` true, although the chosen action and selection set disagree.

   Spec §5 requires both conflicting views to be retained and prevents the conflicting pick file's delivery/probability/result from being attached as if it were the decision record. The current different-primary fixture misses action/leg conflicts.

   **Change:** compare the complete named selection sets and action before attaching whole-file facts. Preserve extra or missing legs as an unresolved alternate view. Add single-versus-double and same-primary/different-DD fixtures, including delivery and day-result assertions.

5. **SHOULD — I5 confuses a changed selection with missing history.** **Location:** Task 8 Steps 1 and 3 (`history_status`); I5.

   **Measured:** a complete archived A plus surviving B yields `known_incomplete`, even though A's content survives. The implementation asks only whether an observation names something outside the canonical set; it does not ask whether the referenced record is missing, or even whether the observation predates the canonical version.

   Spec §5's condition is evidence of earlier content that is gone. A thin evolution entry naming a lost A can satisfy that; a fully retained A does not by itself. A discarded DD can likewise be a retained observation rather than proof of missing content.

   **Change:** distinguish observed selection changes from evidenced missing versions. Resolve history references against preserved full records and their chronology; otherwise retain `unknown`. Never replace this with a `complete` claim.

6. **SHOULD — I2 treats a dropped previous round as a skip and invents a pre-round streak.** **Location:** Task 4 Steps 2 and 4 (`line_round_streaks`, `streak_before`); Task 10 Step 3; I2.

   **Measured:** line 1 includes round 969 with streak 5 and round 970 with streak 0; line 2 includes 969 and 971 but drops 970. For 971, the code emits `streak_before = 5`. It ignores the known intervening round and the fact that it is absent from the selected line. Under §6's same-line requirement, the value should be unknown.

   I accept the supplied description that the real lines carry the full season so far. I am not claiming an observed gap in that snapshot. But the design explicitly requires dropped-round handling, and the implementation accepts those lines; the supplied description cannot replace that contract with “every absence is a skip.” The existing fixture labels its gap a skip without evidence.

   **Change:** establish the previous entered-round identity, check its presence in the selected qualified line, and emit null when it is missing. Make the completeness assumption explicit where used; do not infer skip history from absence. Add the known-intervening-round omission fixture.

7. **BLOCKER — Incomplete team metadata can manufacture a unique game and transfer a grade.** **Location:** Task 5 Step 3 (`parse_schedule`, `team_games`); Task 6 Step 3; Task 10 Step 3 (`schedule_dates`).

   **Measured:** a schedule lists G1 with TB/BOS and G2 without team metadata. The parser quarantines nothing. The compiler calls the date complete; the team index silently omits G2; a TB selection on G1 gets `match = inferred` and a selection link. If G2 is the other TB game, the exact ambiguity the design protects against has disappeared.

   Non-object date entries and absent game lists can also be silently treated as empty. The supplied endpoint facts establish the normal hydrated shape, not an enforced completeness check for every acquired response. This is a synthetic malformed/partial-response hazard, not a claim about the snapshot.

   **Change:** validate the full schedule envelope and each game's required identity/team metadata before permitting inference for that date. Preserve/quarantine incomplete records and mark coverage incomplete. Keep counting postponed/cancelled/suspended games; that part of the current rule is correct.

8. **SHOULD — Eligibility is not consistently bound to the relevant time.** **Location:** Task 10 Steps 1 and 3 (`_eligibility`, refusal index); I6.

   **Measured:** a unit is postponed at 10:00, scheduled again at 20:00, and selected/locked at 21:00. `_eligibility` reports the old postponement because it selects the earliest pre-lock postponed observation and ignores the later contrary status. A second probe supplies a refusal without a usable time; it still emits `refused_evidenced` with null time.

   Refusals are also collapsed by date/slot/batter/game, without revision or attempted-selection time. Multiple attempts for the same identity can therefore attach the wrong refusal to the canonical record. The supplied facts say no refusal archives occur in this snapshot, so that branch is an implementation-contract defect, not an observed season issue.

   **Change:** retain the timed evidence, account for superseding status observations before lock, and bind refusals to the relevant attempt with a reason and valid timestamp. Conflicting or insufficient chronology should remain unknown. Add these cases through raw parsers as well as the helper. Do not derive eligibility from HOLD; the existing code correctly avoids that.

9. **BLOCKER — Reconciliation conflates count agreement with historical recoverability and omits the canonical-outcome side.** **Location:** Task 9 Steps 1 and 3 (`evaluate_rules`, `recipe_labels`); Task 10 Step 3 (`RECONCILIATION_SCHEMA`); I9/I12.

   The scorecard prose rerun is a hypothesis whether or not its current totals match. Conversely, reproducing one naive-tally total does not pin any historical record membership. `recipe_labels` calls the first case `unrecoverable` if neither number matches and the second `partial`. Spec §8 explicitly says the naive tally is unrecoverable if no candidate matches both totals; count fit and membership evidence must be separate axes.

   The membership schema has only `recipe_value`, with no canonical outcome column or selection/observation reference to support the required comparison. It also lacks per-record historical-membership status. **Measured:** a source outside a rule's window simply disappears from membership rather than receiving an exclusion reason; file-set mismatches and several archived file forms likewise disappear. The proposed total membership/exclusion view is therefore incomplete. An mtime comparison is suggestive chronology, not an independently established evidence interval or proof of membership.

   **Change:** implement the spec's scorecard/tally labels, preserve aggregate fit separately, and keep historical membership unknown absent independent evidence. Add canonical outcome and join identity alongside recipe outcome. Define the membership universe explicitly and emit exclusions with reasons so each candidate can be audited. Preserve raw times and label their evidential limits.

10. **SHOULD — The candidate list omits an obvious naive tally, and its freeze test does not pin the list.** **Location:** Task 9 candidate-rules block and Steps 1 and 3.

    S1–S8/T1–T18 are visibly predeclared combinations; I found no evidence of tuning against real counts. However, every T rule requires a result: G1/G2 require hit/miss and G3 requires non-null. None represents the natural naive rule “count every primary/DD object,” including ungraded previews. That candidate is worth declaring before the first real run. It is a hypothesis, not a recommendation to select the rule that fits.

    **Measured mutation:** change S2's file set to F3 and its window to 1900–2099; `test_candidate_rules_are_the_predeclared_lists` still passes. It checks all names, but only S1/T1 in full and portions of S8/T18. The claimed freeze protection is false. The equal-total fixture is also weak: its two rules include the same scored occurrences because their only different file is ungraded.

    **Change:** freeze every rule field and its predicate definitions in an independently checked artifact/digest before counting. Add the ungraded-count hypothesis with a written rationale now, if retained. Test two different included occurrence sets that have equal totals, and verify both alternatives and their distinct memberships survive.

11. **SHOULD — Scalar types are not qualified before fixed-schema serialization, and a failure leaves partial outputs.** **Location:** Task 4 Step 4; Task 10 Step 3; Task 13 Step 3.

    **Measured:** change one otherwise valid synthetic contest slot's `hits` to the string `"1"`. Qualification accepts it and invariants pass. The compiler writes the canonical and occurrence parquet files, then fails writing contest slots with `pyarrow.lib.ArrowInvalid: Could not convert '1' with type str: tried to convert to int64`.

    The source facts specify keys and null cases, not all scalar types. I have not established that any real value has this type. The producer carries external contest values, so the plan still needs an explicit boundary between preserved raw values, typed normalized values, and invalid shapes. Similar unchecked values reach other integer/string columns; list/dict values can fail even earlier in hashing or normalization.

    **Change:** preserve raw values and validate/normalize each typed field under a documented policy, quarantining unsupported shapes with reasons. Validate all output tables before publishing any of them and write a complete build into a fresh staging directory. Add string/bool/list/null/absent cases for fields whose types are not pinned by the supplied facts.

12. **SHOULD — The determinism fixture generates different source bytes across seconds.** **Location:** Task 10 Step 1 (`_units`, `_season_files`, determinism test).

    **Measured:** generate `_season_files()` with the gzip clock fixed at 1000, then 1001. Five gzip files differ because Python 3.12's `gzip.compress` defaults to the current mtime. The original determinism test constructs two separate fixture bundles, so crossing a second changes content hashes, observation IDs and manifest hashes. A byte mismatch is then expected even for a correct compiler.

    **Change:** use `mtime=0` for synthetic gzip creation and build one byte set that is reused/reordered/relocated. Keep the strict byte comparison. The existing 112-test pass is not evidence that this test cannot flake. UTC normalization and explicit ET recipe dates look sound in the inspected code; I found no separate demonstrated host-timezone defect.

13. **BLOCKER — The box's recorded code SHA does not identify all executed code.** **Location:** Task 3 Step 4 (production helper import); Task 11 Step 3 (CLI); Task 13 Steps 1–3.

    The archive ships only audit scripts and package markers. `sources/day_records.py` imports `bts.daily_decision`, which therefore comes from the box's production installation, while the audit entry point runs from `/tmp/ledger_code`. The CLI also defaults `--uv-lock` to the production working directory. Its `code_sha` is the Mac archive SHA. If production and reviewed main differ, this is a mixed-code execution recorded as one reviewed SHA; if the installed helper lacks an expected symbol, it can fail at import.

    **Change:** stage the required helper and dependency identity from the reviewed revision, or explicitly verify and record the production module's bytes and compatibility as a separate input. Bind the actual environment/lock to the run. Do this without installing over, updating, or restarting production. Task 13 currently avoids deploy commands and service restarts, but that alone does not establish a reproducible execution.

14. **BLOCKER — Task 13 can report completion after failed comparison, and its retry path is not executable as written.** **Location:** Task 13 Steps 1–5; Task 11 Step 3.

    `cmp ... || echo "DIFF ..."` consumes the failure, and the final `echo compared` returns success. The backup pipeline similarly returns `tail`'s status without pipefail, and its `uv` invocation omits the required `UV_CACHE_DIR=/tmp/uv-cache`. These are command-inspection findings; no box command was run.

    Compilation writes first into the public output location, before the second build or comparison, so a crash can leave the mixed output set demonstrated in finding 11. The instruction to fix an `InvariantError` and rerun from Step 1 reaches acquisition of the already sealed nonempty v1 directory, which correctly raises `FileExistsError`. The fixed `rm -rf /tmp/ledger_code` also overwrites a shared scratch name rather than a uniquely owned reviewed-code directory. Compile itself runs in plain SSH even though the plan requires long jobs in transient units.

    Finally, Step 5 adds X-19 after Step 3 has computed outcome comparisons and recipe totals. The supplied project instructions require an exposure-register row before a 2026 outcome read. This sequencing must be repaired before execution; it does not require reading that register during this review.

    **Change:** predeclare the authorized X-19 read and frozen recipes, stage code/output in unique directories, run and record each process exit status, fail on any byte/file-set mismatch, and publish only a verified complete output set. Check backup success and snapshot identity independently. On a compiler fix, reuse and verify the sealed bundle, then resume compilation; use a new bundle version only for an intentional new acquisition. No production source/config/service changes are needed.

15. **SHOULD — I4's text and its implementation give opposite values when delivery evidence conflicts.** **Location:** I4; Task 8 Steps 1 and 3 (`delivery`).

    I4 says `private_locked` gives false and `locked_unconfirmed` gives null, with a separate conflict flag. The implementation instead returns true whenever the bound pick has a positive delivery signal, then sets the conflict flag; the existing private-plus-DM test explicitly expects true. A task-local executor can faithfully implement either reading and disagree with the other.

    **Change:** state precedence explicitly in I4 and its Interfaces text. The implementation's positive bound-delivery precedence is consistent with §9's ordered predicate and conflict flag; qualify the false/null cases as applying when no positive bound pick-side evidence exists. Keep all competing evidence in the observation layer.

**Spec §3–§11 coverage and remaining fixture assessment**

| Spec | Tasks | Assessment |
|---|---|---|
| §3 acquisition/offline compilation | 1, 11, 13 | Present/hash/missing manifest behavior and offline compilation pass synthetic tests. Execution pinning, publication and retry defects: 11, 13–14. Acquisition discovers available local files; it cannot independently detect a lost expected nested file without an inventory. |
| §4 lossless occurrences | 2–5, 9–10 | Implemented only in part: dropped fields/presence distinctions, missing observation outputs, and missing occurrence anti-join: 1–2. |
| §5 day/selection/commit/history | 8, 10 | Basic day cases work; selection binding, whole-view conflicts and missing-history inference deviate: 3–5. |
| §6 contest evidence and matching | 4–6, 10 | Qualified histories, unit conflicts, pick-time team and duplicate-link demotion are present. Missing predecessor and partial-schedule hazards: 6–7. |
| §7 outcomes | 7, 10 | Closed normalization, no aggregate-to-DD-leg comparison, same-record single derivation and C-03 synthetic disagreement work. Incorrect attachment/matching upstream can still invalidate the comparison: 4, 7. |
| §8 acceptance/reconciliation | 9–10 | Core acceptance false green plus missing outcome/membership semantics: 1, 9–10. |
| §9 field/timeline contract | 8, 10 | Columns largely exist. Provenance, commit binding, eligibility chronology, typed values and delivery wording remain defective: 2–3, 8, 11, 15. |
| §10 outputs and records | 10, 13 | Named files are produced; their contents do not yet satisfy §§4/8, and publication/exposure sequencing needs repair: 2, 9, 11, 13–14. |
| §11 fixtures | 1–11 | The named baseline cases mostly exist, but nested omission, whole selection-set conflict, temporal supersession and changed included membership are not established by the current tests. Freeze and determinism checks are weaker than claimed: 1, 4, 8, 10, 12. |

Additional bounded conclusions: the fences and interfaces compose in the synthetic replay; I found no missing-symbol/task-order failure there. Routing handles the supplied AppleDouble, repair/manual archive and empty-unit shapes, and early null game identities have dedicated tests. Standard doubleheader/postponed-plus-played ambiguity, traded-player pick-time team, conflicting unit mappings and two contest slots linking one selection are covered conservatively. The partial-void fixture preserves separate leg labels, but its one-HIT/one-HOLD round claims a +2 increase, and the fixture calls a retained streak-10 round `void` despite the spec's round-label meaning; it should use coherent raw facts before serving as a streak-state example. A failed-append/overwrite fixture can legitimately be indistinguishable from a plain surviving record: expecting unknown there is the correct limitation, not proof that capture history was recovered.

## Interpretations

- **I1 — accept:** integer identities, offset timestamps and whole-line quarantine are conservative; null slot grades stay ungraded, while typed payload validation still needs finding 11.
- **I2 — reject:** nearest retained round is not necessarily the previous entered round when a round was dropped; preserve unknown for an unproved/missing predecessor.
- **I3 — reject:** a flag that names no selection cannot become selection-specific commit evidence merely because a pick file survives.
- **I4 — reject as written:** its unconditional false/null wording contradicts the code's positive-delivery precedence; clarify the no-positive-signal condition.
- **I5 — reject:** another retained selection proves a change, not that earlier content is gone.
- **I6 — reject as complete:** the source choices are sound, but superseding status evidence and timed/refusal-attempt binding are missing.
- **I7 — accept:** retain evidenced other-game contest facts separately and do not transfer their grade onto a local selection for a different game.
- **I8 — accept:** without evidence choosing between changed entries, demoting both links prevents an arbitrary authoritative grade.
- **I9 — reject:** matching aggregate counts does not establish partial historical membership; the spec gives different scorecard and naive-tally rules.
- **I10 — accept:** the bounded player lookup is an explicit coverage choice and unmapped identities remain unknown; report its omissions without claiming complete coverage.
- **I11 — accept:** a non-null unknown label may be observed as graded while normalizing to UNKNOWN and remaining incomparable; malformed value types still require validation.
- **I12 — reject:** disposition strings do not implement an occurrence anti-join, and the proposed recipe table omits the canonical-outcome and historical-membership contract.

## Verdict

BLOCK — the runnable 112-test plan still accepts missing occurrences and unsupported selection/game claims; repair the contracts and execution gates before implementation or any real run.
