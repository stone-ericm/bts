## Round

plan r2 — revision 2 of `docs/superpowers/plans/2026-09-28-season-ledger-phase1.md`, commit `ac72676` (HEAD verified), against unchanged spec v4.

No real data, snapshots, network, SSH, or other restricted audit documents were accessed. The supplied source facts were accepted without re-probing them. All executable evidence below uses synthetic fixtures in `.codex-review/season-ledger/replay-plan-r2/`. Operational commands were reviewed as text, not executed.

## Replay

Extracted all **34 Python fences**, with the prescribed package markers and append order, without implementation edits. The mapping is in `replay-plan-r2/extraction-map.json`.

From that replay directory:

```sh
TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/Users/eric/projects/bts/.codex-review/season-ledger/replay-plan-r2/tmp \
/Users/eric/projects/bts/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/scripts/season_ledger \
  --basetemp=/Users/eric/projects/bts/.codex-review/season-ledger/replay-plan-r2/tmp/baseline
```

**Measured: 127 passed, exit 0**, before adding review probes. Task counts match the plan: 12, 9, 8, 9, 5, 11, 6, 21, 36, 9, 1. The cumulative 126/127 expectations are correct. The fences and ordinary interfaces compose; I found no missing-symbol or append-order failure.

Added `tests/scripts/season_ledger/test_review_probes.py`, then ran the same environment/pytest command targeting that file and using `--basetemp=.../tmp/probes`. **Measured: 15 failed, exit 1.** Each asserts the contract discussed below and fails on the demonstrated contrary behavior. These are counterexamples, not evidence that malformed values occur in the real snapshot.

Independently changed `counts` in memory so G1 also counts `suspended`, then ran all original tests, excluding the review probes. **Measured: 127 passed, exit 0; the fingerprint remained `f06a87f528b0fb7855f1fb9d1f230dad51c04a6ad3e5e66544c184c3deebb9b0`.** Reproducer: `replay-plan-r2/run-predicate-mutant.py`, run from the replay directory with the environment above. This verifies a surviving semantic mutant; I did not adopt the plan's claimed 46-mutant result as evidence.

Full outputs: `baseline-result.txt`, `probes-result.txt`, and `predicate-mutant-result.txt` under the replay root. `run.sh.review-copy` preserves the runner text for inspection. An AST scan found no `bts` imports in the extracted runtime package. No repository-wide suite, shell runner, restic, or remote command was run.

## r1 findings

1. **partially resolved — occurrence accounting.** Task 9 Step 3 now catches a deleted DD leaf, a phantom locator within a routed file, duplicate locator rows, and missing ledger references. Those tests pass. However, the empty-parser mutant compiles successfully, dropping a contest line row still passes invariants, an extra source path is accepted, and parent exclusion can overlap emitted children. See new finding 1.

2. **partially resolved — lossless observation layer.** Tasks 2–3 now retain raw pick/decision/state/evolution records, nested state JSON, provenance and nested pick presence metadata; Task 10 persists these in occurrences. `pick_policy_objective` preserves the recorded pick-side objective without pretending it came from a decision. But slotted rounds lose wrong-typed raw round values, and static observations omit raw items entirely. See new finding 2.

3. **resolved — unnamed commit flag.** Task 8 Step 3 removes state from `commit_status`; the flag is a separate observation column. The executed `test_commit_flag_names_no_selection_and_never_commits_one` leaves the preview unconfirmed. A contest match and generic lock still do not prove commitment.

4. **partially resolved — whole selection-set conflicts.** Task 8 compares full parsed sets; the single/double and changed-DD tests pass. Task 10 marks both surviving file legs `unresolved_pick_file_view` and attaches no pick facts in the ordinary conflict fixture. But a quarantined leg is removed before comparison and can make the remaining set appear identical. See new finding 4.

5. **resolved — retained change versus missing history.** Task 8's `history_status(thin, retained)` checks evolution identities against retained full records. The executed test covers the same changed candidate with and without a retained archive; retained content now yields unknown. The failed-append case remains unknown, never complete.

6. **resolved — dropped previous round.** Task 4 derives the previous entered round from all qualified lines, then requires that round in the selected line. Its missing-previous-round test passes. This closes the known-intervening-round counterexample within the supplied full-season-history scope; it does not independently prove completeness beyond that scope.

7. **resolved — partial schedules creating uniqueness.** Task 5 quarantines malformed date/game containers and missing team abbreviations; Task 10 propagates any quarantine to `schedule_status = incomplete`; Task 6 blocks inference. The new incomplete-schedule test passes, and status filtering still does not remove postponed/cancelled games.

8. **partially resolved — eligibility timing.** Latest pre-lock status now handles superseded postponements, and null/after-lock refusal times are rejected when lock is known; executed tests pass. `_eligibility` nevertheless permits an unknown lock and a missing reason, contrary to the new I6 text/§9 respectively. Attempt identity also remains date/slot/batter/game only. See new finding 8.

9. **partially resolved — recipe semantics and membership.** Task 9's labels now follow §8; out-of-window/file-set exclusions and distinct equal-count memberships are tested. Task 10 adds canonical outcome and selection identity for attached production picks. But recipe rows for excluded shadow sources carry nonexistent occurrence IDs, and the join is outside the invariant checks. See new finding 5.

10. **partially resolved — candidate list and freeze.** G4 adds the ungraded object-count candidate; all rule data are hashed, so the old S2 table mutation is addressed. The digest does not cover predicate implementations, and the independent G1 mutation survives all 127 tests. See new finding 6.

11. **partially resolved — typed fields and partial output.** The original numeric-string stat is nulled/flagged, and all four Arrow tables are constructed before writing, closing the demonstrated partial-output failure on type conversion. However, recipe evaluation re-reads untyped grades, decision identity validation is incomplete, and integer range overflow still raises. See new finding 3. Fresh output directories prevent sequential overwrite; they do not make interrupted writes atomic (finding 9).

12. **resolved — gzip fixture clock.** Task 1's gzip builder pins `mtime=0`; Task 10 reuses `FILES` across reordered/relocated builds. The strict byte-equality test passes. No remaining creation-clock difference was found.

13. **resolved — mixed production helper identity.** Task 3 vendors the decision helpers and tests equivalence; the AST scan confirms no runtime `bts` imports. Task 10 records Python, pyarrow and an explicitly named environment-lock hash. The archive now contains the audit implementation it executes. This does not certify an unexecuted box environment, but the original hidden production-helper dependency is removed.

14. **partially resolved — operational gates.** Task 13 puts X-19 first, uses SHA-specific directories, compares exact file sets/bytes with failing Python exits, keeps failed builds, reuses the sealed bundle after fixes, and adds pipefail/UV_CACHE_DIR for backup. Remaining gaps: the runner never exits with its recorded status, monitoring can miss completion, output publication lacks a completion boundary, and backup verification is not bound to the just-created snapshot. See new findings 7, 9 and 10.

15. **resolved — I4 precedence.** I4 and Task 8 now agree: a bound positive delivery signal wins; private/unconfirmed status supplies false/null only without such evidence, with a conflict flag when appropriate. The private-plus-DM test passes.

## New findings

1. **BLOCKER — The census still accepts dropped records and does not enforce exactly-once coverage.** **Location:** Task 9 Step 3 (`account`, `expected_locators`, `census_gaps`, `check_invariants`); Task 10 Step 3.

   Four independent probes demonstrate distinct holes:

   - Replace the pick parser with `lambda *_: Parsed()` for a valid primary+DD file. The complete compiler succeeds. `account` invents an excluded `file` row with reason `no_records`; `census_gaps` lets that ancestor cover both real slots. The parser's failure determines the exemption from the independent check.
   - Delete the emitted `line=1` contest occurrence while retaining its slot. Invariants pass because only structural leaves are required, although §8 explicitly accounts for every contest line and this implementation stores line metadata there.
   - Add an accounting row for `picks/phantom.json`, absent from `files`. Invariants pass. The file check is one-sided, and census checking only iterates routed paths.
   - Add a file-level `excluded/no_records` row beside emitted contest children. Invariants pass; the same nested content is now covered by both the ancestor exclusion and emitted descendants.

   Duplicate exact `(path, locator)` rows are detected, which is useful but insufficient. Reference checking also proves only that an ID is emitted, not that the referenced occurrence has the expected kind/identity or that a claimed canonical disposition has a corresponding row.

   **Change:** independently establish valid empty-source cases; never infer them from empty parser output. Require each actual metadata record as well as nested leaves, reject extra source paths and overlapping disposition coverage, and reconcile a multiset rather than only sets of leaves. Run source/accounting checks before canonical construction, then check typed references and reconciliation joins after those outputs exist. Add these four probes to acceptance.

2. **BLOCKER — Normalization still destroys raw values in the compiled observation layer.** **Location:** Task 4 Step 4 (`parse_contest_ledger`); Task 5 Step 3 (`sources/static.py`); Task 9 Step 3.

   **Measured:** a slotted round with `streak = "RAW_STREAK"` compiles, but the string appears in no occurrence's `record_raw_json`. The line raw JSON excludes `predictions`; a round row is emitted only for slotless rounds; slot raw JSON contains only the slot. The typed value becomes null, so the original reported round fact is gone from the observation table.

   **Measured:** a units item with `feedId = "RAW_GAME"` compiles with null typed feed ID and no raw record. Static parsers never populate `record_raw_json`. Their comment explicitly falls back to the bundle, contradicting I12/I13's claim that raw values survive in occurrences. Quarantined rows similarly have no raw envelope; their bytes remain retrievable only from the bundle.

   **Change:** preserve round metadata alongside every relevant slot or emit an explicit raw round observation for every round. Preserve raw O7 values for the fields being normalized, including wrong-typed identities. State the representation for quarantined records clearly. Verify round-trip recovery of absent/null/zero and wrong-typed values from the observation output, not merely from the sealed input.

3. **SHOULD — I13 is not applied consistently, and invalid identities can become committed selections.** **Location:** Task 1 Step 3 (`typed`); Task 3 Step 4 (`parse_decision`); Task 9 Step 3 (`counts`, `evaluate_rules`); Task 10 Step 3.

   **Measured counterexamples:**

   - A pick `result` object is nulled/flagged by the parser, but recipe evaluation reads the raw object again and crashes at `value in GRADED` with `TypeError: unhashable type: 'dict'`. This runs even for rules that would exclude the file by path/window.
   - A decision primary with string `batter_id` remains emitted with null typed identity; `day_rows` can name and commit that null identity. I13 explicitly promises quarantine for untypable identity fields.
   - An object-valued decision `action` crashes in the acceptance set-membership test before the typed policy runs.
   - A synthetic `hits = 2**80` passes the Python integer predicate and raises `OverflowError` when converted to Arrow int64. The output tables are built first, so this no longer leaves the earlier r1 partial files.

   These malformed values are synthetic; the given facts do not establish that they occur in the snapshot. There is also a wording mismatch between I11's “any non-null” result and I13: a wrong-typed non-null contest result becomes null and is then labelled `matched_ungraded`.

   **Change:** validate identities and enum container types before downstream logic; carry raw and typed values separately into recipe evaluation and define how each recipe handles unsupported shapes. Bound integers to the output type, and distinguish invalid grades from genuine source nulls. Add parser-to-compiler tests for each policy branch.

4. **BLOCKER — Quarantine can turn a conflicting double-down file into an apparently matching single.** **Location:** Task 2 Step 4; Task 8 Step 3 (`day_rows`); Task 10 Step 3.

   **Measured:** decision selects single A; the delivered pick file contains A plus a DD object with an invalid batter identity. The DD is quarantined, leaving only A in `pick_rows`. `day_rows` declares the sets identical, emits `finalization = decision`, and attaches the file's facts. The valid primary already carries `has_double_down = true`, but that structural evidence is ignored by the set comparison. A wholly quarantined file can similarly be treated as no file.

   This is a new interaction between I13 and I14. A partial parse cannot establish whole-record agreement.

   **Change:** pass file presence, raw selection shape and parse completeness into the row rule. Only establish agreement after every expected leg is accounted for and its identity is usable. Otherwise retain an unresolved view and withhold file-level attachment. Test malformed primary/DD and whole-file quarantine, not only valid differing IDs.

5. **BLOCKER — Reconciliation IDs do not resolve for part of the declared universe.** **Location:** Task 9 Step 3 (`evaluate_rules`, `account`); Task 10 Step 3 (membership join and invariant ordering).

   **Measured:** compile a synthetic `picks/DATE.shadow.json` containing a pick. Accounting emits one excluded `file` occurrence. Every recipe emits `obs_id(path, slot=primary, hash)`, which does not exist in occurrences. `occurrence_disposition` is null, and compilation succeeds because invariant checking precedes recipe construction and never validates its references.

   This uses an expressly supplied real path/shape class: the facts include top-level and backup shadow pick objects. No malformed input is needed. Quarantined whole files and excluded sources pose the same parent-versus-slot identity problem. Null canonical selection/outcome is appropriate for these sources; a purported occurrence ID with no target is not.

   **Change:** define recipe-only slot identity separately from the resolvable source occurrence, or emit an accounted recipe observation linked to its excluded/quarantined parent. Carry the source state/reason as well as any canonical link. Check every membership reference and cardinality after the join. Do not force excluded shadow slots into production canonical rows to make the join pass.

6. **BLOCKER — The “rules and predicates” fingerprint omits the predicates.** **Location:** Global Constraints; Task 9 Steps 1 and 3 (`rules_fingerprint`, freeze test); Task 13 Steps 1 and 4.

   `rules_fingerprint` hashes rule dictionaries, recipe kinds, the graded-label set, and regex strings. It does not hash or otherwise bind `counts`, `slot_value`, `slot_inclusion`, or the universe/window evaluation logic. Its docstring and the exposure declaration overstate what is frozen.

   **Measured:** make `counts("suspended", "G1")` return true while leaving other behavior intact. The fingerprint stays exactly the predeclared value and **all 127 original tests pass**. The truth table never tries suspended. This is a behavior-changing survivor, not a no-op mutant. The production `DailyPick.result` vocabulary includes `suspended`; I did not inspect whether it occurs in the snapshot.

   **Change:** bind the complete recipe evaluator semantics to the predeclared artifact, for example with reviewed source bytes/AST plus rule data, or explicit versioned predicates expressed entirely in the hashed specification. Include all supported labels and excluded-universe cases in truth tests. Record the corrected digest before the first real run; do not treat the current digest as proof that predicates stayed fixed.

7. **SHOULD — `run.sh` logs failure but exits successfully, and monitoring has a race.** **Location:** Task 13 Steps 2–4.

   The runner ends with `echo "EXIT=$status"`, without `exit "$status"`. For a handled acquisition, compilation or comparison failure, the final successful echo makes Bash/systemd report success. The manual instruction requiring a zero log sentinel does prevent acceptance if followed and observed; the process status does not independently enforce it.

   The monitor starts `tail -n 0 -F` after unit submission. A quick import, fresh-directory or acquisition-preflight failure can finish before the tail attaches, leaving no new sentinel to observe. Fixed append log paths also lack a per-attempt delimiter/identity.

   **Change:** use an EXIT trap that prints the actual status and exits with it, including setup failures. Record a run ID and use a log/readback protocol that includes already-written output for that run. Preserve the exact byte/file-set comparison and required success marker. The quoted outer `EOF` and nested `PY` delimiters appear correctly formed on inspection; I did not run any shell/remote command.

8. **SHOULD — The new refusal rule still differs between text and code.** **Location:** I6; Task 10 Steps 1 and 3 (`_eligibility`).

   **Measured:** a refusal with a valid time and `locked_at = None` is accepted because the filter explicitly allows `locked is None`, although I6 requires a valid time before lock. A second probe supplies a known lock and a refusal with no reason; it is also accepted despite §9's reason-and-time contract. Sorting chooses a refusal by date/slot/batter/game and time, without establishing which attempted version it belongs to.

   **Change:** make the text and helper agree on unknown-lock behavior; preserve a refusal as an observation if temporal binding is unproved. Require a usable reason for the canonical claim and document the attempt-binding rule. Add these cases through the raw archive parser. The facts say no refusal archives exist in this snapshot, so this is a contract issue, not a measured season issue.

9. **SHOULD — A fresh directory is not a verified-publication boundary.** **Location:** Task 10 Step 3; Task 13 Step 4.

   Building all tables first fixes the measured conversion failure. A later write interruption can still leave a partly populated SHA directory at the advertised final location, and a failed second build or byte comparison leaves the first directory there. There is no completion manifest or atomic promotion that distinguishes an accepted build for consumers. The initial emptiness check and `mkdir(exist_ok=True)` also do not reserve the directory against concurrent writers, though the normal single-unit workflow reduces that risk.

   **Change:** write each attempt to a uniquely reserved staging directory, retain failed attempts there, and publish a verified directory or explicit completion record only after file-set/byte/fingerprint checks pass. Keep the nonempty-directory refusal. Do not overwrite a failed run merely to retry.

10. **SHOULD — The restic check does not verify the specific successful backup's bundle.** **Location:** Task 13 Step 5.

    The API calls match the inspected local `restic_env`/`restic_bin` signatures, and pipefail now preserves backup command failures. However, the check separately asks for latest archive snapshots and lists `latest`; it does not capture the successful backup's exact snapshot ID. A concurrent archive backup can change the target. Counting the manifest pathname as a substring proves neither exact path identity nor that its bytes match the sealed manifest being recorded. A different snapshot containing the same path can satisfy this check.

    **Change:** capture the snapshot ID returned by this successful backup, inspect that exact ID, and compare the backed-up manifest with the bundle's recorded hash. Verify the manifest-declared bundle members are represented in that snapshot. Keep this verification separate from production changes. No restic/network call was made in this review.

## Interpretations

- **I1 — accept:** integer identities, offset timestamps, explicit null results and whole-line quarantine are conservative for the supplied shapes; malformed grade types still need the I13 handling clarified.
- **I2 — accept:** a previous round seen anywhere must be present in the selected line before its streak is used; this closes the known dropped-round case within the supplied full-season scope.
- **I3 — accept:** the unbound scheduler flag is an observation, never selection-specific commit evidence.
- **I4 — accept:** the positive bound-delivery precedence and conflict flag now match §9 and the code.
- **I5 — accept:** a thin observation naming a version absent from retained full records supports missing history; a retained change alone does not.
- **I6 — reject as written versus implemented:** the before-lock requirement is bypassed for unknown lock time, and the reason/attempt binding needed by §9 is incomplete.
- **I7 — accept:** keep evidenced other-game contest facts separate and do not transfer them onto a different-game local selection.
- **I8 — accept:** unresolved multiple contest links must not choose an authoritative grade arbitrarily.
- **I9 — accept:** scorecard hypothesis, tally recoverability and aggregate fit are now separated as §8 requires; historical membership stays unknown.
- **I10 — accept:** the bounded players lookup is explicit and unmapped players remain unknown rather than guessed.
- **I11 — reject as unqualified:** “any non-null” conflicts with wrong-typed non-null results becoming null under I13; define status from validated grade evidence without presenting malformed input as source null.
- **I12 — accept as the required contract:** persisted observations plus complete census/reference checks fit §§4/8; findings 1, 2 and 5 show the code does not yet meet it.
- **I13 — accept as policy:** typed null plus preserved raw value and mismatch metadata is sound, with explicit nullable identities; validation, raw retention and numeric bounds still need repair.
- **I14 — accept:** whole-selection agreement is the right attachment condition, but agreement cannot be established from a partial/quarantined parse.

## Verdict

BLOCK — several r1 fixes hold, but occurrence completeness, raw-value preservation, whole-file agreement, reconciliation references and the recipe freeze still fail independent probes.
