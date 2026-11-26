## Round

plan r3 — revision 3 of `docs/superpowers/plans/2026-09-28-season-ledger-phase1.md`, commit **9d2bb13** (HEAD verified), against unchanged spec v4.

No real data, snapshots, restricted audit documents, network, SSH, herdr, or remote operations were accessed. Supplied source facts were treated as input. All executable probes used synthetic fixtures under `.codex-review/season-ledger/replay-plan-r3/`. Task 13 was reviewed as text only; its commands were not executed. No implementation, plan, spec, or production file was changed.

The remaining disagreements below go to Eric. This review does not prescribe another automatic review round.

## Replay

Extracted **34 Python fences into 28 files**, with the same package layout and append order as r2. Verified every fence verbatim, with one separator newline between appended fences. `extraction-map.json` records the mapping. Package markers were added as before.

Test commands ran from `/Users/eric/projects/bts/.codex-review/season-ledger/replay-plan-r3`. Each replay/test Python invocation used:

```sh
TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 \
TMPDIR=/Users/eric/projects/bts/.codex-review/season-ledger/replay-plan-r3/tmp \
/Users/eric/projects/bts/.venv/bin/python
```

Arguments and measured results:

| Run | Arguments after Python | Result |
|---|---|---|
| Unchanged plan, before review tests were added | `-m pytest -q -p no:cacheprovider tests/scripts/season_ledger --basetemp=/Users/eric/projects/bts/.codex-review/season-ledger/replay-plan-r3/tmp/baseline` | **135 passed**, exit 0 |
| Ported r2 probes | `-m pytest -q -p no:cacheprovider tests/scripts/season_ledger/test_review_probes.py --basetemp=/Users/eric/projects/bts/.codex-review/season-ledger/replay-plan-r3/tmp/ported` | **12 passed, 3 failed**, exit 1 |
| Ported G1 suspended mutant, before new native probes existed | `run-predicate-mutant.py` | **133 passed, 2 failed**, exit 1 |
| New native r3 probes, final version | `-m pytest -q -p no:cacheprovider tests/scripts/season_ledger/test_review_r3_native.py --basetemp=/Users/eric/projects/bts/.codex-review/season-ledger/replay-plan-r3/tmp/native --tb=short` | **10 passed, 16 failed**, exit 1 |
| Independent regex-option mutant, unchanged plan tests only | `run-regex-mutant.py` | **135 passed**, exit 0; fingerprint unchanged despite a changed total |

Python 3.12.13, pytest 9.0.2. Full saved outputs: `baseline-result.txt`, `ported-probes-result.txt`, `predicate-mutant-result.txt`, `native-probes-result.txt`, and `regex-mutant-result.txt` under the replay root. `run.sh.review-copy` preserves the unexecuted runner text. The first native-probe run had 21 cases (6 passed, 15 failed); the final 26-case version adds outcome, identity-agreement and routing coverage. No repository-wide suite was run.

**Port: every rename, and no other changes.** Verified the port against the r2 files by applying exactly these substitutions:

1. `test_candidate_rules_and_their_predicates_are_frozen` → `test_candidate_rules_and_their_evaluator_are_fingerprinted` in the imported/called test name.
2. Recipe membership assertion `m["obs_id"]` → `m["source_obs_id"]`. Accounting IDs remain `obs_id`.
3. `replay-plan-r2` → `replay-plan-r3` in the predicate runner's comment and basetemp path.

No fixture, assertion meaning, or function argument was changed. The passing ports are: empty pick parser; round raw value; unit raw identity; malformed recipe grade; bad decision batter; bad decision action; quarantined DD agreement; shadow occurrence reference; predicate fingerprint; refusal with no lock; refusal with no reason; int64 overflow.

The three failing ports are `test_drop_emitted_contest_line_is_detected`, `test_phantom_source_path_is_detected`, and `test_excluded_parent_and_emitted_child_are_not_double_accounted`. All stop at `TypeError: account() takes 3 positional arguments but 4 were given`, before their assertions. **This is a deliberate API change, not evidence that those three defects survive.** Task 9 Interfaces (plan lines 2636–2641) removes the dispositions argument, adds `assign_dispositions`, and separates `check_sources(files, routed, accounting)` from output checks. Three separate native equivalents use that interface and **all pass**. They preserve the original three corruptions and require `InvariantError` from the source check.

The old G1 mutation now changes the digest and fails both `test_candidate_rules_and_their_evaluator_are_fingerprinted` and `test_grading_truth_table_covers_every_label_shape`. The new outside-`_RECIPE_CODE` survivor is described in finding 4.

**Executability and counts.** Tasks 1–11 compose with their stated interfaces. Measured per-task counts are 12, 9, 8, 9, 7, 11, 6, 23, 38, 11, 1; cumulative **134 after Task 10 / 135 after Task 11** are correct. No missing symbol or append-order failure was found. The updated `pick_file_state`, `day_rows`, accounting, and outcome-status signatures are represented in the Interfaces blocks. Tasks 2–3's narrower batter-only quarantine description conflicts with I13's identity rule; finding 2 explains the resulting behavior. Task 13's operational gaps are below. The predicted full-repository count and real bundle-entry count were not measured under this review's restrictions. The plan's other mutant claims were not adopted as verification.

## r2 findings

1. **partially resolved — census and exactly-once accounting.** Task 9 `required_locators` now walks raw containers independently of parser output. The empty **pick** parser port and the three native equivalents above pass; dropped contest metadata and excluded-parent overlap are caught. The new all-kind census probe accepts fixtures spanning every `PARSERS` kind, including quarantined evolution slots and empty units/schedule lists. However, empty decision/state parsers, invalid present/missing states, extra excluded-file records, and duplicate occurrence IDs still pass. See finding 1.

2. **resolved — the original round/static raw-value losses.** Task 4 emits every round's raw metadata; Task 5 `_raw` retains raw values of normalized static fields; Task 10 persists them. Both ported raw-value probes pass, as does the plan's `test_raw_values_round_trip_through_the_occurrence_table`. I12 explicitly limits static raw records to normalized fields, with remaining content in the sealed bundle. The newly extended quarantine rule has a separate JSON-null defect (finding 6).

3. **partially resolved — typed policy.** Tasks 1–3 and 9 fix int64 bounds, malformed enum membership, bad batter identity, and unhashable recipe results; all corresponding ports pass. The new partial-decision/evolution probe confirms whole-decision quarantine for a bad chosen batter and slot-level evolution quarantine. End-to-end grade probes confirm I11's null/malformed distinction. Present wrong-typed `game_pk` still becomes an apparently usable null identity, including false whole-file agreement; finding 2.

4. **partially resolved — partial pick-file agreement.** Tasks 2/8/10 now carry raw slot presence and completeness. The quarantined-DD port passes; the plan's wholly unparseable-file test and my no-decision partial/whole-pick probe pass. A partial undelivered pick preserves only the usable batter as unconfirmed, with `pick_file_complete=false`; a wholly unusable pick becomes `pick_file_unparseable`. But wrong-typed game identities can still produce a false complete agreement, and the skip branch ignores wholly quarantined delivery evidence. Findings 2 and 5.

5. **partially resolved — recipe occurrence references.** Task 10's actual join now selects the correct slot or excluded/quarantined file ancestor. The shadow port passes. A new positive probe verifies the exact path, locator, state, reason and disposition for emitted slots, excluded shadow files, quarantined archive files and quarantined slots. The invariant only checks whether the ID exists: wrong-source links and missing membership rows still compile. Finding 3.

6. **partially resolved — recipe freeze.** Task 9 binds the named evaluator functions and catches the original suspended mutant (two original tests fail). Regex compilation flags remain outside the digest. A changed F1 total survives the unchanged fingerprint and all 135 original tests; finding 4.

7. **partially resolved — runner status/readback.** Text inspection of Task 13 confirms the original final-echo status bug and `tail -n 0` completion race are fixed: an EXIT trap preserves status, and run-delimited readback includes already-written output. Setup after trap installation has explicit failures, and compile/comparison failures remain nonzero. The readback is not actually bounded against a hanging SSH, run IDs can collide, and `.env` sourcing has unchecked-failure/nounset cases. Finding 7. No shell probe was run, as explicitly required by this prompt.

8. **resolved — refusal binding.** I6 and Task 10 `_eligibility` now agree. Both old refusal probes pass; the plan's direct helper test verifies a reason and time before a known lock and rejects the missing/late alternatives. These probes exercise the helper's refusal map, not an end-to-end raw refusal archive. I6 explicitly limits the claim to the named date/slot/batter/game, not a particular delivery attempt. No attempt identity is invented.

9. **partially resolved — output publication.** Task 10 reserves `out_dir` with exclusive `mkdir()` after all tables are built. The existing-directory test passes for empty and nonempty targets; by inspection, a concurrent creator causes failure before any table write. Task 13 compares both exact file sets, bytes and fingerprint before creating the acceptance marker. That closes the old unmarked partial-build problem, but an interrupted marker write can leave the presence-based boundary ambiguous (finding 8). Compiler tests were run; the runner was text-only.

10. **resolved for the stated verification scope — backup snapshot binding.** Task 13 captures a start time, requires exactly one archive snapshot since it, then uses that fixed `sid` for both manifest dump and member listing. Zero or multiple candidates stop; a concurrent archive snapshot therefore does not silently redirect a later `latest` lookup. It compares exact manifest bytes by SHA and exact declared-present member paths. The snapshots response is parsed as one JSON array; `ls --json` is parsed line by line, filtering file nodes so the header is excluded. Timestamp fractions are reduced to microseconds without dropping the offset. Local `src/bts/data/backup.py` confirms successful backups normally create a tagged snapshot and retain its summary ID; `backup_cli.py` exits nonzero when `ok` is false. This is text verification, not a restic compatibility or restore test. Member presence plus manifest identity does not prove restored member-byte integrity, and the revised text does not claim it does. Capturing the command's returned snapshot ID directly would also be reasonable but is not necessary to close the original `latest`/substring defect.

## New findings

1. **BLOCKER — File-record omissions and invalid occurrence identities still pass both checks.**

   **Location:** Task 9 `required_locators`, `census_problems`, `check_sources`, `check_invariants` (plan lines 3062–3187); Task 10 invocation order.

   **Demonstrated:**

   - Replace only `PARSERS["decision"]` with an empty `Parsed()` for a valid single decision, or do the same for a scheduler record with `committed_pick_written=true`. Both complete compiles succeed. The independent census returns `{"file"}` for these records, the same value used for genuinely empty containers; `excluded/no_records` is accepted whenever `required == {"file"}`. The decision/state evidence disappears. These are two native failing cases.
   - Change a present contest `line=1` accounting row to `declared_missing`, clearing its parsed/raw fields and disposition. Both checks accept it. Direct locator presence counts as coverage without validating the state against the present source bytes.
   - Append a second excluded accounting row at `slot=phantom` beneath an excluded shadow file. Both checks accept it. `census_problems` validates locators only for routed files.
   - Change a manual-archive parser's DD `obs_id` to its primary's ID while retaining both distinct slot locators. A complete compile succeeds with two occurrences sharing one ID. The duplicate check covers `(path, locator)` but never recomputes occurrence IDs from `(path, locator, content hash)` or checks their uniqueness.

   These are coverage and identity-check failures, not claims that current unmodified parsers spontaneously return empty results or duplicate IDs. The user-requested omission/duplication guard must detect such failures.

   **Change:** distinguish actual file records from an empty-container sentinel; validate the permitted state for each present/missing/excluded source and require direct coverage to be emitted or quarantined. Apply accounting shape validation to every bundle path. Independently verify ID/hash/kind binding and ID uniqueness. Keep the raw structural census independent of `Parsed`; the current separation is useful but its exemptions are too broad.

2. **BLOCKER — Wrong-typed game identities can become equal nulls and falsely establish complete agreement.**

   **Location:** I13/I14; Task 2 `parse_pick_file`/`pick_file_state` (803/844), Task 3 `parse_decision`/`parse_lineup_evolution` (1091/1154), Task 8 `day_rows` (2557).

   **Demonstrated:** separate parser probes supply `game_pk="bad"` with a valid batter to pick, chosen decision, and evolution records. All emit rows with null `game_pk` and a mismatch flag, without quarantine. An end-to-end probe uses the same batter with `game_pk="bad_file_game"` in a delivered pick and `game_pk="different_bad_game"` in the decision. Compilation succeeds with `finalization="decision"`: both unusable game IDs collapsed to None, the pick file is called complete, and file facts attach.

   This contradicts I13's “Identity fields that cannot be typed quarantine the record” and I14's usable, identical selection identities. Tasks 2–3 only enforce a usable **batter**, so an executor faithfully implementing their narrower Interfaces block reproduces the defect. Genuine source-null `game_pk` on the supplied early-season files remains an intentional supported case; this finding concerns a present malformed value, not those nulls.

   **Change:** explicitly enumerate identity fields and allowable source-null cases. Quarantine present malformed game identity before completeness/agreement, or retain an explicit unusable-identity state that cannot establish agreement or authorize attached file facts. Do not treat two failed conversions as evidence that selections agree.

3. **BLOCKER — The membership invariant proves existence, not the correct join or complete membership.**

   **Location:** Task 9 `check_invariants` membership check (3184–3187); Task 10 membership join and check (3907–3918).

   **Demonstrated:** compile a production pick and a shadow pick, then substitute the existing shadow file occurrence ID into every production membership row just before the invariant. Also substitute its state/reason/disposition, so the metadata agrees with the wrong target. Compilation succeeds: the ID exists, but names a different file. A second mutant drops the first membership row while retaining the unchanged recipe summary; compilation also succeeds. Both fail my expected-rejection probes.

   The ordinary join is correct on the positive four-shape probe. The defect is the advertised acceptance check, including absence of the cardinality check requested in r2.

   **Change:** verify exact `(rule_id, source_path, slot)` coverage and uniqueness against the independent universe. Require `source_obs_id` to name that file's slot occurrence or its valid covering file ancestor; compare state/reason/disposition and any canonical link. Reconcile included membership counts with the reported recipe totals.

4. **BLOCKER — A semantic edit outside `_RECIPE_CODE` still survives the freeze.**

   **Location:** Task 9 `FILE_SETS`, `rules_fingerprint` (3348–3355), its freeze test; Global Constraints and Task 13's predeclared digest.

   **Demonstrated:** recompile F1 with the same `.pattern` and `re.IGNORECASE`. For synthetic `picks/2026-08-20.JSON` containing a hit pick, S1's primary total changes **0 → 1**. The digest remains **`eadce48f7fbce5346cf9214289f8c079f1d7ccb907a50253a4c133e97c2ffc1d`**, and the independent mutant runner reports **135 passed**. Regex flags affect behavior but are absent from the fingerprint, which hashes pattern strings only. This changes no function in `_RECIPE_CODE`.

   The uppercase-extension fixture is a synthetic counterexample in the expressly declared any-suffix universe; I make no claim about such filenames in the snapshot. The original suspended mutant is now correctly rejected.

   **Change:** bind regex flags along with patterns, and review other evaluator configuration outside the named functions for the same problem. Alternatively bind a reviewed evaluator source artifact plus explicit configuration. Any corrected digest must be predeclared before a real count; an executor must not simply edit the pinned expected digest to make a failed test pass.

5. **SHOULD — Quarantine presence is preserved for pick files but lost in other day branches.**

   **Location:** Task 10 day input construction (3788 onward); Task 8 `day_rows` skip and no-evidence branches.

   **Demonstrated:** a valid skip decision plus a wholly quarantined pick with `notification_sent=true` and a nonempty notification ID becomes `skip_day`. The skip branch consults only surviving parsed pick rows; the readable file-level delivery signal disappears with the malformed batter. This does not establish which batter was committed, but it prevents establishing the skip shape's “no committed pick” condition. The appropriate result is unresolved/unfinalized evidence, without inventing a selection.

   Independently, a bad-batter single decision as the only local file, or a wholly quarantined evolution line, becomes `unobserved_day/no_evidence`. In both cases an evidence artifact exists and is quarantined in occurrences. The new pick-file presence mechanism handles the analogous wholly bad pick correctly, as a passing probe confirms.

   **Change:** carry per-date source presence and parse completeness for decisions, state and history, and retain readable file-level delivery facts independently of slot identity. Define quarantine-aware day reasons. Unknown content must not silently become absent evidence or establish a definitive skip.

6. **SHOULD — Parsed JSON null violates the extended quarantine raw-value rule.**

   **Location:** Task 1 `quarantine` (420–425), I13, its callers.

   **Demonstrated:** compile a pick file consisting of valid JSON `null`. It is correctly quarantined, but its occurrence has `record_raw_json=None`, not the JSON text `"null"`. `raw=None` is used both for a parsed null and for no decoded value, despite I13 permitting absent raw only for undecodable bytes/path-level refusals.

   **Change:** use a distinct sentinel for the omitted argument and serialize an explicitly supplied None as JSON null. Retain the current no-raw behavior for undecodable bytes. The sealed bundle still preserves the bytes; this is a violation of the promised occurrence representation, not irreversible source loss.

7. **SHOULD — The runner still has environment and monitoring cases outside its claimed guarantees.**

   **Location:** Task 13 runner setup/backup (4323–4364), run-ID creation and readback (4404–4417). **Text-only scenarios; none was executed.**

   - The EXIT trap correctly captures `$?` for valid invocations after installation, including a final Python comparison failure, and exits with that status. A missing run-ID argument fails at `${1:?run id}` before the trap/START marker; narrow the claim that setup is always covered or install a fallback identity first.
   - `set -u` remains active during `. ./.env`. A shell env file containing `OPTIONAL=$UNSET_OPTIONAL` terminates on expansion; the trap should report nonzero, so this is a compatibility stop, not false acceptance. Conversely, a missing/unreadable env file or an ordinary failing source command is not checked: without `set -e`, the following `set +a` masks its status and backup may continue with inherited credentials. The actual `.env` was not read. Explicitly check source status and decide whether env expansion is nounset-compatible; keep any shell-option change narrowly scoped.
   - The 360-iteration wait does not bound an individual `ssh`. A stalled connection or authentication prompt can block on iteration one indefinitely. Even completed slow probes add to the stated two-hour interval. Use noninteractive SSH with connection/liveness limits and an elapsed-time deadline; report timeout as a stop.
   - `<UTC-second>-<mode>` is not a unique invocation ID. Two callers starting the same mode in one second share the unit name, log delimiters and output paths. The second launch can fail while its readback still observes the first invocation. Generate a nonce, retain the launch result and use exact per-run markers or a per-run log. Do not interpret `NO EXIT LINE` solely as “never started or killed”: a still-running acquisition/backup can also exceed the readback limit.

   These remaining cases do not negate the successful status-propagation and completed-log-readback fixes.

8. **SHOULD — `ACCEPTED.json` is created directly at the presence-based publication boundary.**

   **Location:** Global Constraints line 46; Task 13 marker write (4355) and Step 4. **Text-only failure scenario.**

   File-set, byte and fingerprint checks all precede the marker, which is correct. But `Path.write_text` opens the final pathname before completing its contents. Termination or disk exhaustion during that write can leave an empty/truncated `ACCEPTED.json`. The global/Step 4 rule says its presence publishes the directory, while the runner has no successful exit and the receipt cannot be read. The compared six outputs may still be valid; the ambiguity is whether acceptance completed.

   **Change:** write and close a temporary acceptance receipt in the reserved directory, then atomically rename it to `ACCEPTED.json`. Consumers should require a valid receipt with the expected run/file metadata, not mere pathname existence. Preserve the current exclusive directory reservation, failed-attempt retention and required successful run readback.

## Interpretations

- **I1 — accept.** Typed contest identities, offset times, required result keys and explicit null results remain appropriate; whole-line quarantine avoids silently qualifying partial identities.
- **I2 — accept.** The previous entered round is determined across qualified lines, but its value must be present in the selected line.
- **I3 — accept.** An unnamed scheduler flag is an observation, not selection-specific commitment.
- **I4 — accept.** A positive bound delivery signal takes precedence, with a separate conflict flag. Quarantined file-level signals need preservation under finding 5.
- **I5 — accept.** A retained full version does not itself prove missing history. Unusable quarantined history should remain visible as uncertainty.
- **I6 — accept.** The revised known-lock/reason/time requirements match the executed probes. Selection-level refusal evidence is explicitly distinguished from attempt identity.
- **I7 — accept.** Keep an evidenced other-game contest slot separate from the local selection.
- **I8 — accept.** Multiple contest slots for one selection remain ambiguous; do not arbitrarily transfer a grade.
- **I9 — accept.** Historical hypothesis labels remain separate from aggregate fit. Finding 4 concerns enforcement of the predeclared evaluator, not these labels.
- **I10 — accept.** The bounded players lookup and explicit unknown mapping remain unchanged.
- **I11 — accept.** New end-to-end probes confirm source null → `matched_ungraded`; wrong-typed object → `unknown` with exact JSON in `contest_slot_grade_raw`; unfamiliar string → `graded` but no normalized comparison. `_outcome_status` and `grade_raw` compose correctly for these cases.
- **I12 — accept as the required contract.** Persisted occurrences, explicit static-field raw scope, and a real source occurrence for every membership row are appropriate. “Emitted occurrence” here must include table rows whose accounting state is excluded/quarantined. Findings 1 and 3 show the promised checks remain incomplete.
- **I13 — accept as policy.** Source-null identity exceptions must remain distinct from failed conversion. Tasks 2–3's batter-only validation and the raw-None sentinel do not implement the full stated policy (findings 2 and 6).
- **I14 — accept.** Whole-file completeness is required before agreement and attachment; unusable identities cannot establish it. The new no-decision whole-pick behavior is correct. Findings 2 and 5 identify remaining interactions, not a reason to weaken the rule.

## Verdict

BLOCK — demonstrated census, identity-agreement, membership-validation and recipe-freeze gaps remain; send these concrete disagreements to Eric for disposition.
