# C2 side item (e) evidence: the framing screen's mutant ledger
- `mutants.json`: 26 mutants over `scripts/audit/c2_framing/screen.py`. They cover the feature (shift, min_periods, daily mean), the self-check, the variants and blend configs, each disposition rule, the pre-registered constants, the pinned inputs, the run's refusals and stops, the aggregate, and coverage.
- `mutant_runner.py`: the runner used for C2 step 2a. Since revision 9 it lives at `scripts/audit/c2_framing/mutant_runner.py`.
- `mutants.out`: the first run at `9637a44` gave 25 RED and F12 SURVIVED.
  - F12 swaps "both seasons above zero" for "either". Its test case also missed the practical size, so another rule masked it.
  - After the case was isolated, F12 is RED.
- **Result: 26 of 26 RED.** The tests were written after `screen.py`; this ledger is their strength check.

## Revision 2 (after review r1 BLOCK)
- **Runner revised** (r1 item 8): a mutant counts as RED only when pytest exits 1 with a `FAILED` line. Any other exit is INCONCLUSIVE, and the runner exits 1 if any mutant is not RED.
- **Ledger rebuilt: 54 mutants.**
  - F24 is retired, because its aggregate code was rewritten.
  - G1–G29 cover the new boundaries: aggregate refusals, seed order and release, labels, closed inputs, settings, the launch budget and the zero-variance t.
- **First revision-2 run at `307e7bc`:** 49 RED, 5 SURVIVED (G1, G2, G5, G9, G20).
  - G1, G2 and G9 were masked by later checks that refuse the same input with a different error.
  - G5 had no wrong-root case.
  - G20 was masked because the real park-drag attach also gives NaN when there is no table.
- **The fix:** the tests now require the specific check's message, add a wrong-root run and a consistently changed head, and spy on the table read. The resume run turns all five RED.
- **Result: 54 of 54 RED.**

## Revision 3 (after review r2 BLOCK; row C2-framing-review-r3)
- **Runner revised (r2 R2-4):**
  - RED only when pytest exits 1 with a `FAILED` line, no `ERROR` line, and a summary with no error, interruption, skip, xfail or xpass.
  - A clean pass is SURVIVED; anything else is INCONCLUSIVE.
  - Its classifier has its own regression tests in `test_screen.py`.
- **Ledger: 67 mutants.** 13 were re-anchored to the revised code. H1–H13 cover `validate_run`, the release naming seed 1's run, and the identity-based aggregate.
- **First revision-3 run at `fe5eb85`:** 58 RED, 5 INCONCLUSIVE and 4 SURVIVED.
  - **INCONCLUSIVE (F4, F20, G14, G22, G28):** each failed the test that targets its rule. It also errored in the shared three-run fixture's setup, which the strict runner does not count as clean. Each now names its targeted test, which fails cleanly. F20 and G22 fail `test_run_end_to_end` on their own assertions: the seed env and the feature settings.
  - **G4 (claim binding):** survived; it had no case isolating it. Now covered by a `CLAIM.json` rewritten to other bytes that still names its run and code.
  - **H6 (the ten pin names):** survived; the only pins case also broke the digest. Now covered by nine pins, consistently, with a digest that matches them.
  - **H8 (the admitted identity in the seed-1 release):** survived; there was no case. Now covered by a released seed-1 run made by another identity.
  - **G8: recorded equivalent.** `validate_run` binds each run's `inputs_digest` to its `input_pins`, so runs that agree on the digest agree on the pins (barring a sha256 collision).
- **Resume at `2f43386`:** the eight attributable mutants are RED. G8 survives, as recorded.
- **Result: 66 of 66 attributable RED; G8 equivalent.** The suite holds 75 tests.

## Revision 4 (after review r3 BLOCK; Eric's row C2-framing-review-r4)
**Runner revised (r3 R3-4):**
- Each mutant's intended set is what `pytest --collect-only` selects from its named tests on the unmutated source.
- A mutant is RED only when pytest exits 1 and every intended node reported PASSED or FAILED under `-rA`. At least one must have FAILED, with no `ERROR` line and a clean summary.
- An intended node that never ran makes the mutant INCONCLUSIVE, whatever the summary says. That covers fail-fast, a crash or a changed selection.
- `PYTEST_ADDOPTS` is removed from the subprocess environment, ini `addopts` are cleared, and one `--rootdir` is pinned for both passes.
- A test list holding any option is INVALID before anything is mutated.
- The runner's own tests cover all of this, including the reviewer's two-node case with inherited `-x`, which the old runner marked RED with the second test never run.

**Spec: 88 entries.**
- Every mutant now names node ids: no whole-file lists and no options, and every list collects. Most were narrowed to the tests that failed for them in the revision-3 run.
- G6, H8, H10, H11 and H13 were re-anchored to the revised code.
- The new mutants:
  - N1–N16: card recomputation, scoring settings, season evidence, P@1 coverage, the HEAD check, trusted identity and pins;
  - R1–R5: the runner's error, not-run, environment, option and exact-node rules.
- The parametrized damage and classifier cases have short ids.

**First revision-4 run at `cab2549`:** 84 RED, 4 SURVIVED (G8, H5, H12, R5).
- **H5 (identity completeness):** shadowed in the aggregate cases by the strict trusted-identity check. It is now isolated by a direct call whose admitted identity has the same empty field.
- **R5 (exact node matching):** no case had an executed node whose id merely starts with an intended one. It is now covered: `t.py::ab` ran, `t.py::a` did not.
- **G8 and H12: recorded equivalents.**
  - G8 (unchanged reasoning): each run's digest is bound to its pins.
  - H12: each run must equal the trusted identity, so validated runs cannot disagree on identity.

**Resume at `e1033c5`:** H5 and R5 are RED.

**Result: 86 of 86 attributable RED; G8 and H12 equivalent.** The permitted suite has 122 tests (97 framing, 25 shared admission).

## Revision 5 (after review r4 BLOCK; Eric's row C2-framing-review-r5)
**Runner revised (r4 R4-2 and the sandbox-only failure):**
- Outcomes now come from pytest's own reports, never its text. A plugin loaded with `-p` records each report (node id, phase, outcome) to a private file the runner creates, and writes an end record when the session finishes normally.
- A mutant is RED only when all of these hold:
  - pytest exits 1, and the session ended normally;
  - nothing errored in setup or teardown, and nothing was skipped;
  - every intended node has its own call report, and no unintended node ran;
  - pytest's summary counts agree with the records;
  - at least one intended node failed.
- Printed or captured output cannot add an outcome. The reviewer's case is now a test: a fake `PASSED` line plus a second test stopped before its body gives INCONCLUSIVE.
- **Sandbox fix:** the runner's root is pytest's rootdir and working directory (default: the repository), and the subprocess uses the runner's own interpreter. The runner tests give their scratch files a root of their own, so collection never leaves it. The round-4 failure came from pinning the rootdir to the repository while collecting tests elsewhere, which made pytest list directories outside the scratch area.
- **Stated boundary:** test code that deliberately tampers with the runner's records file or with pytest's internals from inside the run is out of scope.

**Validator (r4 R4-1, R4-3):**
- **`profile_problem`:** every retained row must be one the scorer scores. That means:
  - no missing value in date, rank, season, hit or probability;
  - an integer season equal to the unit's, and dates in the season;
  - ranks 1..n on every day;
  - hits 0/1, and probabilities in [0, 1].
- **Identity:** it must have exactly the five fields.

**Spec: 99 entries.**
- Re-anchored: H5, H13, R1, R2, R5, N7, N8, N9.
- New:
  - H14 (exactly five identity fields);
  - N17–N22 (the scored-row rules);
  - R6–R9 (normal end, summary agreement, the recording plugin, its end record).
- **Recorded equivalents, each over the validator's whole accepted domain:**
  - **G8:** each run's pins must equal the trusted pins, and its digest is bound to them.
  - **H12:** each identity has exactly the five fields and equals the trusted identity.
  - **N10:** every accepted unit is non-empty, of its own season, and ranked 1..n daily, so P@1 always covers both seasons.
- **Message-level kills (named here, not counted as distinct defences):** some mutants are refused by a later check with a different message. N17 is one: without its missing-value check, a nullable season or rank is refused by the dtype or ranks checks instead. The test pins the specific message, so the mutant is RED.

**Revision-5 run at `2bc188f`:** 96 RED, 3 SURVIVED (G8, H12, N10, as recorded).

**Result: 96 of 96 attributable RED; G8, H12 and N10 equivalent.** The permitted suite has 133 tests (108 framing, 25 shared admission).

## Revision 6 (after review r5 BLOCK; Eric's row C2-framing-review-r6, under the full rules)
**Runner revised (r5 R5-1):**
- "Ended normally" is pytest's own session status, never unconfiguration (which pytest also reaches after `pytest.exit`) and never text. The recording plugin also records:
  - the interrupt hook (`pytest.exit` and KeyboardInterrupt), the internal-error hook, and failed collection;
  - a session record written last (`trylast`) at `pytest_sessionfinish`, holding pytest's exit status and its collected and failed counters. If another session-finish hook aborts first, there is no record.
- RED requires all of these:
  - the session record, with no interrupt, internal error or collection error;
  - every intended node's setup, call and teardown reports;
  - the counters agreeing with the reports and the process return code.
- The text cross-check is gone.
- The reviewer's aborted-teardown case is a test, along with a later session-finish abort and an exit after the last teardown report. Run against the revision-5 runner, the first two fail; the aborted-teardown case was certified RED there.

**Validator (r5 R5-2):** probabilities must be numeric, never coerced. Any scorer exception while rescoring the retained profiles is a `RunInvalid` refusal.

**Spec: 108 entries.**
- Re-anchored: H13, R1, R6, R7, R9.
- New:
  - R10–R16: interrupt, its recording, setup/teardown completeness, internal error, collection error, exit status, the session record written last;
  - N23–N24: numeric probabilities, the rescoring refusal.

**Revision-6 run at `750086b`:** 104 RED, 4 SURVIVED (G8, H12, N10 as recorded; R11).
- R11 (the plugin's interrupt record) was masked in the aborted-teardown test by the missing teardown report. It is now isolated by `pytest.exit` raised after the last teardown report, where every report exists and the counters agree.

**Resume at `aa4e72c`:** R11 is RED.

**Result: 105 of 105 attributable RED; G8, H12 and N10 equivalent.** The permitted suite has 143 tests (118 framing, 25 shared admission).

## Revision 7 (after review r6 BLOCK; Eric's row C2-framing-review-r7: full rules, a redesigned runner)
**Runner redesigned (r6 R6-1, R6-2):**
- **Driver process:** the mutated run is a driver that calls `pytest.main()` in-process. It writes its evidence only after `pytest.main` returns, so an abort anywhere, including unconfiguration, leaves no evidence.
- **Sentinel:** at collection finish, after every conftest is loaded, the recorder registers a sentinel whose hooks are `tryfirst` wrappers.
  - Pluggy calls the last-registered tryfirst wrapper outermost. So its session-finish wrapper sees any inner hook's or wrapper's abort (R6-1), and its runtest-call wrapper records what each call itself raised.
  - Any plugin registered after the sentinel is refused.
- **Expected failures (R6-2):** reports keep their `wasxfail` status, and an intended node marked xfail, skip or skipif is never a killer.
- **Cross-checks:** report outcomes must match the sentinel's call observations, and pytest's counters must agree.
- **Stated boundary:** tampering with the runner's own objects, its evidence file or pytest internals is out of scope.
- **New tests,** all red against the revision-6 runner, which certified seven of them RED:
  - a session-finish wrapper abort and a tryfirst wrapper abort;
  - a non-strict XPASS and a runtime xfail;
  - a plugin registered during the run;
  - a makereport wrapper rewriting an outcome;
  - an unconfigure abort;
  - plus a 23-case classifier table over the evidence.

**Spec: 116 entries.**
- Re-anchored: H13, R6, R7, R8, R9, R11, R15, R16; R2 repointed.
- New, R17–R24:
  - sentinel registration;
  - late plugins;
  - xfail/skip markers and `wasxfail` recording;
  - report/observation agreement and the sentinel's call record;
  - exit status and session-finish completion.

**Revision-7 run at `310dfe6`:** 113 RED, 3 SURVIVED (G8, H12, N10 as recorded).

**Result: 113 of 113 attributable RED in one pass; G8, H12 and N10 equivalent.** The permitted suite has 157 tests (132 framing, 25 shared admission).

## Revision 8 (after review r7 BLOCK; Eric's row C2-framing-review-r8: full rules, the registration-hook fix)
**Runner (r7 R7-1).** A late registration is detected in three layers:
1. **Event:** every registration after the sentinel is counted through pytest's public `pytest_plugin_registered` hook. The count is kept even if the plugin later unregisters itself.
2. **Census:** the final registry comparison stays.
3. **Outermost check:** the sentinel checks, at each test call and at session finish, that it is the outermost implementation of that hook. That catches an outer wrapper even when its registration bypassed pytest's API.

The driver guarantee is stated precisely: evidence is written after `pytest.main` returns. An abort escaping it leaves none. Aborts pytest catches may leave evidence, and each is refused by an explicit rule.

**New tests,** red first against the revision-7 runner, which certified the reviewer's lifecycle RED:
- the register / outer-wrapper abort / self-unregister lifecycle;
- the producer's evidence for a briefly registered plugin, asserted on the bundle itself;
- a session-finish wrapper and a call wrapper, each registered through pluggy's base class and caught only by the outermost check;
- a classifier case.

**Spec:** 122 entries; new R25–R30 cover the event, its count, both outermost checks, the refusal and the outermost position.

**Revision-8 run at `e9a2483`:** 119 RED, 3 SURVIVED (G8, H12, N10, as recorded).

**Result: 119 of 119 attributable RED in one pass; G8, H12 and N10 equivalent.** The permitted suite has 162 tests (137 framing, 25 shared admission).

## Revision 9 (after review r8 BLOCK; Eric's row C2-framing-review-r9: the runner's threat model stated and enforced)
**Threat model** (pre-registration §9; the runner's docstring): test code that changes pytest's plugin system, wraps its hooks or aborts the session is outside the runner's scope. The runner refuses to run on any suite containing such code. The experiment and its result checks are not narrowed: `screen.py` is byte-identical to revision 8.

**Enforcement:**
1. **The gate, before anything is collected or mutated** (`scope_problems`).
   - It covers every file the run imports as test code: each named test's module and the `__init__.py` of each package above it.
   - Each file is parsed from its bytes, as Python will parse it, so a coding cookie cannot hide code.
   - Each file must stay within the reviewed vocabulary, which is the real suite's own imports, builtins, attributes, keywords and parameters (so fixtures).
   - `pytest` is allowed only as `pytest.fixture`, `pytest.mark` and `pytest.raises`.
   - Refused outright: hook-like and dunder bindings, rebound builtins, any `**` that is not a pure pass-through, `setattr` other than (object, literal allowed name, value), and `match` statements.
   - A refusal prints `REFUSED`.
2. **What loads** in every collection and run:
   - `-c /dev/null`, so no ini setting applies;
   - `--noconftest`;
   - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`;
   - no inherited `PYTHON*` or `PYTEST*` variable;
   - `-B -P -s`;
   - a bytecode cache prefix under `/dev/null`, where no file can exist.
3. **The change scan.** No file or directory under the root or the interpreter's import path may change from the ledger's start (ctime). The mutation targets must keep the ctime of the runner's own last write while pytest runs. This closes write-then-import.
4. **Child processes** reach the run only through the operating system. A signal is refused (interrupted, or no evidence); a written file is caught by the change scan. A debugger is out of scope.

**Defence in depth behind the gate.**
- The driver watches pluggy's `HookCaller._add_hookimpl` and `HookCaller._remove_plugin` through `sys.monitoring` once the sentinel registers. Every registration or unregistration runs one of them, through whatever API.
- A plugin that is neither pytest's own nor the runner's at the sentinel's registration is foreign.
- The older layers stay. The outermost check's limit (R8-1) is stated in the docstring.

**R8's required changes:**
1. **Ordering bound to the active dispatch: superseded.**
   - R8-1's lifecycle needs `pluggy`, `request` and a `pytest_*` hook, so the gate refuses it.
   - Run anyway, the pluggy watch refuses it as `plugin system changed`. The evidence records `[HookCaller._add_hookimpl, HookCaller._remove_plugin]`, with `late_plugins` 0 and `not_outermost` [] (the layers R8-1 escaped).
2. **Producer tests added:**
   - the before-yield session wrapper, with and without the abort;
   - the before-yield call wrapper;
   - an early plugin unregistered mid-run, which only the remove watch sees;
   - the after-yield controls are kept.
3. **The stale "abort anywhere" sentence: removed.**

**Replay** (`r8_witness_replay.py`): your R8-1 witness gives `RED` on the revision-8 runner (from git, `dcdbb71`). On revision 9 the gate refuses it (`import importlib.util`, `import pluggy`, `name pytest_sessionfinish`, ...), and run anyway it is `INCONCLUSIVE(exit 1, plugin system changed)`. Output: `r8_witness_replay.out`.

**Vocabulary audit** (`gadget_audit.py`, output `gadget_audit.out`).
- It walks breadth-first from every allowed module over the allowed attribute names, five levels deep, statically.
- It reaches only:
  - `pytest.fixture` and `pytest.raises`;
  - the benign modules `numpy.random`, `numpy.testing`, `pandas.testing` and `stat`;
  - `subprocess.run`, which starts children (point 4 above).
- It does not model instances or call results.

**Suite changes** (`eaf5be1`, no behaviour change):
- The runner is imported statically from its new path.
- The two source checks read the production files' text. Each was confirmed red against an edited production file, then restored.
- `__import__` and `A.hashlib` became top-level imports, and `**S.SCORING` became explicit keywords.

**The hostile scratch suites** of rounds 4 to 8 now serve two purposes.
- Each is asserted refused by the gate.
- Each still runs through `run_mutant` to exercise the dynamic layers.
- The conftest-based ones load through `pytest_plugins`, since conftests no longer load.

**Spec** (`d0d83d9`, fixed in `b9c3c81`): 188 entries.
- The 31 runner entries follow the runner to its new path.
- R3, R8 and R17 are re-anchored on the same behaviour.
- R31–R97 are new: what loads (R31–R38); the gate's imports, names, attributes, parameters, keywords, `setattr`, parsing and files (R39–R75); the change scan (R76–R88); and the two dynamic layers (R89, R91–R97).

**First pass at `d0d83d9`** (`mutants_r9_first.out`; 17 minutes; no change-scan alarm in 189 runs): 182 RED, G8/H12/N10 survived as recorded, and four entries were not RED. Each was fixed in `b9c3c81`:
- **R44 (from-imports only from allowed modules) SURVIVED** because of message-level masking. With the module rule gone, the name rule still reported `import os.system`, which contains the test's `import os`. The case now imports an allowed attribute name (`from os import name`).
- **R47 was INCONCLUSIVE** because its replacement left an empty `for` body. The replacement is now `pass`.
- **R87 (the runner's restore is its own) SURVIVED** because it only matters when a spec has two target files. The test now runs mutants across two targets and back.
- **R90 SURVIVED** because pluggy's `get_plugins()` already omits blocked names' `None`. The driver's redundant filter is removed together with its entry, instead of being recorded as an equivalent.

**Revision-9 run at `b9c3c81`:** 185 RED, 3 SURVIVED (G8, H12, N10, as recorded); no change-scan alarm in 188 runs; 17 minutes.

**Result:** 185 of 185 attributable RED in one pass; G8, H12 and N10 equivalent. The permitted suite has 234 tests: 209 framing and 25 shared admission.

## Revision 10 (after review r9 BLOCK; Eric's row C2-framing-review-r10: committed code only)
**The rule** (Eric's option A; pre-registration §9; the runner's docstring): a run may execute only installed packages, the checked test files and code committed in the root's repository. Anything else is refused, before the run and by a Python audit hook during it. Revision 9's threat model stands. Committed code is reviewed code: the runner trusts it as it trusts installed packages, and a certificate holds for the commit it ran on. Tampering with the interpreter's installation is out of scope. `screen.py` is byte-identical to revisions 8 and 9.

**R9-1, collection outside the root.** With `-c /dev/null`, pytest's collection cutoff was `/dev`, so collection walked every ancestor of the root.
- Red first: on the revision-9 runner, a package `__init__.py` one directory above a scratch root ran, and the mutant was still certified RED (`r9_witness_replay.py`).
- Every collection and run now passes `--confcutdir=<root>`.
- In a Codex `:workspace` seatbelt sandbox (`codex sandbox -P :workspace`, the profile that denies `/private/tmp/codex-daemon-501`), the revision-9 permitted suite gives 27 failed and 207 passed, the same 27 nodes as your r9 receipt; at revision 10 it gives 271 passed (246 framing, 25 admission) at `487f54c`.
- The one remaining failure in your confined subset (`test_the_run_environment_is_scrubbed`) was the positional `args[-1]` check against your appended option; the test now also requires `--confcutdir=/r`.

**R9-2 and R9-3, one boundary.**
1. **Before the run** (`boundary_problems`; a refusal prints `REFUSED`, like the gate):
   - the root must be the top of a git work tree with a commit;
   - inside the root, a run imports from the named tests' package roots (pytest puts each first on the path) and from any startup import-path entry inside the root (the repository's `src`, for the real root);
   - there, recursively through importable directories, every importable entry must be a regular file committed with its current bytes: never uncommitted or changed, never a symlink, never a compiled module (bytecode or an extension);
   - no top-level name there may be one an installed tree also provides (a look-alike). A directory counts only when it holds something importable, so a bytecode cache is never a look-alike.
   - The installed trees are the standard library, site-packages, and every startup entry outside the root (for a scratch root, the repository's own editable install). None may contain the root.
   - So an allowed import spelled like an installed module (`numpy`) can only load the installed module, and any other allowed import can only load committed code.
2. **During every collection and run** (the driver's audit hook, added before pytest is imported):
   - code compiled from a file may run only if the file is in an installed tree, or is in the root with the commit's bytes as compiled (the mutated target: the mutation's), reached without a symlink, under no installed name;
   - no bytecode is ever deserialized (sourceless bytecode can name any file as its source);
   - no compiled module or native library (`ctypes`) loads from outside the installed trees;
   - each refusal raises `ImportError` and is recorded in the evidence, so a refusal the test code catches still refuses the run (`INCONCLUSIVE(..., code outside the boundary)`).
   - Collection now runs through the same driver, since collecting executes the test modules; the intended nodes come from its evidence, not from text.
3. **Measured before the design**, over the real permitted suite with the run's bytecode prefix under `/dev/null`: no bytecode was deserialized, every extension module and `dlopen` came from site-packages, and every module ran from the installed trees or the repository. So the hook's refusals cost the real suite nothing.

**The change scan stays** as defence in depth: a file a child writes is refused by the hook if the run executes it, and by the change scan in any case.

**Your required changes:**
1. Collection bounded to the root, with empty configuration and no conftests kept: done; the permitted suite passes in the sandbox (above).
2. Every allowed import bound to installed or committed bytes: done by the look-alike and committed-bytes rules, before the run and in it. The imported-fixture control (a local `numpy.py`) is refused before execution, and run anyway it is refused by the hook (`r9_witness_replay.py`).
3. Linked imports: refused before the run (a symlink) and by the hook's real-path check during it. The ordinary-file control is kept (`test_a_file_written_into_the_root_during_the_run_refuses_it`).

**Replay** (`r9_witness_replay.py`, output `r9_witness_replay.out`): benign witnesses whose forbidden code writes a canary.
- R9-1: on revision 9 the ancestor's canary is written and the mutant is RED; on revision 10 the mutant is RED and the canary is not written.
- R9-2 (an uncommitted local `numpy.py`): on revision 9 the look-alike runs (canary written) and the mutant is RED. On revision 10 it is REFUSED (`numpy.py: not committed`, `numpy.py: shadows the installed module numpy`). Run anyway, it is `INCONCLUSIVE(exit 2, code outside the boundary)`, with the hook's record `shadows an installed module`, and no canary.
- R9-3 (a committed `numpy.py` link to a file outside the root, written by a child the test starts): on revision 9 the linked module runs and the mutant is RED. On revision 10 it is REFUSED (`a symlink`, the look-alike). Run anyway, it is `INCONCLUSIVE(exit 1, code outside the boundary)`, with `outside the root and the installed trees`, and no canary.

**First-pass simplifications** (`25fa653`), each to leave no guard a test cannot fail:
- The hook has no re-entrancy guard (none of its own calls raises an audit event), and it checks every executed code object compiled from a file, not only module bodies.
- The look-alike names are what the installed trees provide. Built-in and frozen modules are found before the import path, so the standard-library and built-in name lists added nothing.
- The object-id algorithm comes from the commit id's length (sha1 or sha256), replacing a format query no test could make fail.

**Vocabulary:** the suite gains `boundary_problems` and `_startup` (attributes) and `boundary` (a keyword), and loses `_pytest_cmd`. None is in the escape-route lists; `gadget_audit.out` is re-run (158 objects visited, one more than revision 9, from the runner module's two added attributes and one removed; the reachable callables are unchanged).

**Spec** (`487f54c`): 242 entries.
- R8 (the mutated run goes through the driver) is re-anchored to `_drive`.
- R98–R150 and R152 are new: the collection cutoff (R98); the boundary check's wiring, root, installed trees, walk, symlink, compiled, committed, differs and look-alike rules (R99–R130); and the audit hook's rules, its evidence and driver-based collection (R131–R152).
- R133 (the hook's "outside the root and the installed trees" reason) is a disclosed MESSAGE-ONLY entry: code outside the root is still refused as "not committed", so its test pins the message.

**Revision-10 run at `487f54c`** (appended to `mutants.out`; one pass; 19 minutes; no change-scan alarm in 242 entries): 239 RED; G8, H12 and N10 survived, as recorded.
- Every new entry, and the re-anchored R8, failed every test it names.
- Kill reasons spot-checked by hand: R99 lets the test module's own top-level code run (SURVIVED, never REFUSED); R124 lets a committed look-alike run, certified RED; R141 lets the run read RED with no refusal record; R133 is refused as "not committed", as disclosed.

**Result:** 239 of 239 attributable RED in one pass (R133 message-only, disclosed); G8, H12 and N10 equivalent.

**Suites:** the permitted suite has 271 tests (246 framing and 25 shared admission), all passing locally and in the sandbox at `487f54c`.
