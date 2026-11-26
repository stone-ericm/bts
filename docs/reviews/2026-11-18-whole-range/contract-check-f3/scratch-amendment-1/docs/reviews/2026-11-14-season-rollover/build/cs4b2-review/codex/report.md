# Verdict

SIGN. Reviewed `8494b6e6` against `0bbfc8d4`, against the version-2 contract. No production contract violation or separate contract-row counterexample found. No tracked file changed.

# Blocking findings

None.

# Notes

- Test-only: `tests/test_season_contract_cs4b2.py:176` returns immediately for `required=True, today=False`, reporting a pass without testing anything. Remove that parameter cell or assert its refusal and switch read. Its behavior is covered by the selection matrix, so this is not a block.
- Existing health/decision disagreement: `src/bts/health/contest_state.py:75` and `:83` issue CRITICALs for a covered timestamp-only observation or a sufficiently future source date although the decision accepts it. Both behaviors match the base; these are coverage/data-quality alarms, neither false switch-refusal alarms nor missed switch refusals. Any policy reconciliation is a follow-up outside this diff.
- The explicit `covered_automatic` fixture can hide a schema/season/identifier refusal if incorrectly used in a future refusal test: it intentionally replaces that metadata. It is not autouse; its current migrations preserve the value/date corruptions under test. `final_checks.py` demonstrates both properties.

# Can an uncovered observation drive a live pick

No through `load_decision_streak_state(require_contest_state=True)`: the independent matrix has 101,088 cells, 92,892 exceptions and 8,196 accepted decisions. Every acceptance is a dated observation in effect or a covered automatic observation, and its complete serialized decision equals the base, including status, saver and best trust.

Axes: 12 switch states × 13 manual classes × 27 automatic classes × 3 model states × 8 saver cases. These include each corrupt-switch check, actual unreadable manual files, automatic schemas on manual filenames, misleading sources, wrong JSON types, completely undated schemas, timestamp-only observations, and a saver integer exceeding Python's decoding limit. Selected legacy overrides never pass live enforcement, including expired/undated fallbacks; ignored expired files beside covered automatic observations still permit that automatic selection.

Both reader settings match the base in every cell. All 101,088 optional decisions and all 101,088 `expected=False` health outcomes match exactly and never open the switch. Dated live reads and loader errors also bypass the switch. Other live reads open an existing switch once; absence is determined by `lstat`. Direct-helper tests separately enforce one snapshot and no observation reopening.

# Health

All exact alert comparisons pass: 53,004 changed expected-health outcomes are explained by row 5; 48,084 are identical. The decision changes in 52,224 cells. Loader errors retain priority; completely undated automatic observations now produce only their loader-error CRITICAL. No other dating-rule change was found.

On normal completion each refusal is appended exactly once, last, after coverage, manual and saver alerts. Missing observations produce the original missing-state CRITICAL followed by the refusal. The dated early return correctly bypasses enforcement; no other return loses a promised append.

There are 210 accepted-decision cells with existing coverage/data-quality CRITICALs (105 timestamp-only, 105 future-date). Another 6,420 refusal cells have a pre-existing post-loader health exception before the append, as expressly excluded by the contract. The real health runner turns the overlong-saver exception into its existing `health_runner` CRITICAL. No newly missed or false switch alarm found.

# Messages and revoke

Ran 25 reason/subcase setups and 50 recovery runs, extracting and executing the printed commands with a directory containing spaces, both quote characters and `&`. Filled dated templates work; the evidence-fetch/status/enable/fetch path ends accepted; intermediate states remain refused until repaired. Tested every corrupt check, schema diagnostic and season/identifier ordering, plus deliberate corrupt-switch removal, revoked repair, replacement by enable, and matching automatic repair. Legacy messages correctly require replacement of the winning manual observation. No incorrect or dangerous Opening Night instruction found within the stated limits.

`status`: 8,424 base/branch command comparisons, with and without `--season 2026`, preserve every old section and exit code after removing only the new live section. The new section equals the real actual-year decision/error. Audits and tree comparisons found no writes or creation, including missing directories and non-ContestStateError failures.

`revoke`: usage errors precede application reads. Audit events show switch replacement, manual rename, saver rename in order. Checked every prior switch state, absent/present sources, repeated revoke, real read-only-directory failure, injected permission/lstat errors, occupied file/directory/symlink/dangling/FIFO targets, dangling/directory sources and absent sources beside occupied targets. Every failure preserves the required completed prefix and stops before the next step. Existing tests also exercise interruption after each step and before printing, then retry.

Identifiers and `switched_at` carry only from wholly valid same-season automatic/revoked snapshots; other states get a fresh identifier and null time. Automatic observation, ledger and actual fetch-status bytes remain unchanged across all three failure stages and success. December-31 midnight crossing uses the entry year and shared entry-time stamp. Renamed evidence is retained; post-revoke live decisions refuse, the flag reads uninitialized, and status agrees. Filled dated/saver templates demonstrate their separate authorities.

# Operator sequences

`commands_v3.py` and `remaining.py` exercise season start, evidence-only fetch, enable before installation, switched fetch, restore without revoke (accepted named limit), revoke, new dated observation, re-enable, next-day old-ID refusal and fetch recovery. Each step records the real decision, exact health alerts and policy saver.

A fetch failing before observation replacement stays refused; a fetch failing after replacement can be accepted despite exit 1. December 31's last microsecond accepts the covered observation; January 1 refuses the other-season switch until enable and a new fetch. Persistent unexpired, expired, no-expiry and automatic-schema legacy files are exercised across fetch/enable/revoke/restoration; a winning legacy still refuses until replaced. Restoring an old automatic file after revoke/re-enable never restores coverage without the old switch. No additional failure sequence found.

# Callers

- `orchestrator.py:283`, `run_and_pick`: raises before strategy selection on refusal. The real strategy runs for accepted controls; slate observability can precede the refusal.
- `orchestrate` catches that initial error, sends its existing operator-error DM and returns false. Its later reads at `:362` (skip), `:386` (catch-up) and `:429` (new pick) propagate the error. The new-pick branch may already have saved a preview; none posts after refusal.
- `scheduler.py:1098`, delivery: alerts and returns false before delivery attempt, lock or decision commit. `run_single_check` catches selection errors and returns no pick; fallback refresh re-raises, with the in-loop/final catches alerting and withholding delivery. A later scheduled check can retry; no new wake/retry is introduced.
- A refusal installed before delivery blocks it; changing the switch after delivery's own successful read can still permit delivery (named snapshot limit). Already-delivered, crash-attempt and private cached-pick guards retain their existing no-read lock/record behavior. These are unchanged paths, not evidence that a required read accepted an uncovered observation.
- The config gate is scheduler-required OR health-expected. Both were exercised. Optional CLI `run` (including posting), `preview`, dashboard, scheduler display fallback at `:1723`, and research/shadow paths remain outside enforcement. The tracked example config leaves both flags false; actual box configuration was intentionally not accessed. Closing optional entry points/configuration gaps belongs to change set 6.
- CLI grading and scheduler result polling still update the streak; captured end-of-day skips and `day-outcome` work while the decision loader is forced to raise. The five config-derived decision call sites plus status cover all required readers found under `src`/`scripts`.

Independent caller probes ran 168 cases on each source, five delivery/guard sequences and three later-orchestrator reads. Refused attempts produce no pick transport, new scoreable decision or new lock; operator-error DMs remain possible. Existing locks and captured skip records are intentionally preserved as above.

# Tests and false greens

Every touched test file passes: **2,679 passed at 2027-07-15 and 2,679 at 2028-07-15**, using my clock/audit plugin. Against archived base source, the 1,581 new tests yield **982 failed, 599 passed**. The named list is `base-green-controls.json`: 588 selection-matrix controls (dated bypass or existing parser errors), four optional/dated cases (one vacuous as noted), five existing delivery/error-handling guards, one grading control and one real caller/fallback control. Availability failures are not evidence of revoke semantics, and positive controls needing a new switch-read spy can fail on base even when the returned decision is unchanged.

All 15 source mutants are detected: remove refusal; invert dated bypass; drop corrupt/manual/inactive-switch/missing/schema/season/identifier reasons; omit either missing-state or final health append; continue after each failed revoke step; overwrite occupied targets. Detailed failure counts are in `test-results.json` and `mutant-no-health-final.log`. Spies patch the production runtime import sites; actual-source paths are recorded in the independent evidence.

`migration-inventory.json` inventories all 40 changed test/helper functions. The defensive-read, CS1-review and decoder-limit migrations retain scan, bounds and corrupt-model assertions; required legacy acceptance becomes explicit refusal while optional/reader meanings remain, and dated recovery replaces the old live legacy recovery. The v1 required/optional equality is deliberately replaced by optional acceptance plus required refusal. CS4a's three-day sequence retains its optional intermediate control. The CS4b1 edge test now expects switch access only for required non-dated reads. Missing-required diagnostic changes match row 3.

The unlisted health migrations are justified: `tests/health/test_contest_state.py` retains coverage thresholds/order and adds the second missing-state alert; `test_saver_cs3.py` retains every saver-matrix, season, attention and command-repair assertion; `test_saver_cs3_review.py` retains attention/cooldown and mistaken-use/undo assertions. They add covered metadata for intended accepted controls. `tests/test_contest_state.py` changes only the required-missing message. No unrelated assertion was removed, loosened or made vacuous by a migration.

# Rows 1 to 9

1. Holds — matrix/oracle, helper snapshot/no-observation-I/O tests, ordered-reason mutants.
2. Holds — `commands_v2.py` completed message runs; `remaining.py` additional repairs and real fetch recovery.
3. Holds — 101,088 live matrix cells, exact accepted outcomes, real caller/delivery probes.
4. Holds — every optional cell and both selecting-reader settings equal base; no switch opens.
5. Holds — exact alert oracle, both append mutants, dated/loader/post-loader controls.
6. Holds — 8,424 real status comparisons, actual-year/live-error equality, audit/tree checks.
7. Holds — `commands_v3.py`, `remaining.py`, failure/interrupt/retry tests and source mutants.
8. Holds — retained archives, exact snapshot, entry clock, executed dated/saver templates.
9. Holds — preserved-function AST comparison and byte comparisons; selection, delivery, grading and skip controls.

# What I ran

- Read the three contracts, R24/R28–R31 and the full diff; used the handover only after independent execution. Archived base `src` with `git archive`, without touching the tracked tree.
- `matrix.py` produced both 101,088-cell raw records; **`matrix_verify.py` is the authoritative comparison**, zero unexplained differences. Its corrected oracle accounts for absent-switch lstat (no open event) and Alert.incident_key; the original comparison's harness failures are retained, not findings.
- `commands.py base`; completed branch status/messages in `commands_v2.py`; filesystem/sequences in `commands_v3.py`; `remaining.py`; `callers.py branch` and `base`; `final_checks.py` with `PYTHONPATH=.`. Evidence is the corresponding JSON/log files. Initial command-probe failures were snapshot placement/fixture-accounting errors; corrected runs passed.
- All pytest children ran serially through `yield_run.sh`: `run_test_family.py` (valid rerun in `test-family-valid.log`) and `test_edges_family.py`. The first run lacked its scratch parent; its setup errors are excluded. Waited for each wrapper to exit, including nice warnings; never overlapped pytest wrapper runs.
- Each probe installs the offline audit hook; fetch/auth, schedule, post and DM transports were captured. No network, box or contest account used. Did not rerun the whole fast suite. Final tracked and staged diffs are empty.
