## Verdict

**BLOCK.** The duplicate-name/null and callee-async defects are repaired in the measured r13 cases. A unique-name variation still produces a false positive event certificate: the reader unwraps a genuine cell-valued argument and certifies its contents as the argument. Both import-time and call-phase code replacement accept, with links **2/3/4**, although the sender receives a `CellType` object. The 391-pass suite and seven eligible prepared-spec acceptances do not cover that physical-slot mismatch.

Pins: code **e927a7584e0c8dd94785e6d37cf463eb8eb57604**; evidence **eb559f0966c8444eb12f3cc237c5bb016ea1b114**; plan **edb16ffd9ebe1ca500f5154e652c4d0cfac14677**. The evidence commit changes no `src`, `scripts` or `tests` bytes relative to the code pin (`P/evidence-bindings.json`). **P** denotes `/Users/eric/projects/bts/.codex-review/2026-09-29-incident-register/r8-probes/r14`. Source abbreviations at `e927a75`: **O** = `scripts/audit/incident_register/observer.py`; **C** = `scripts/audit/incident_register/certify.py`; **D** = `scripts/audit/incident_register/defence.py`; **T** = `tests/scripts/incident_register/test_r10_counterexamples.py`; **M** = `docs/audit/2026-09-29-incident-register-evidence/tooling/mutation_sweep.py`. **L** is `docs/superpowers/plans/2026-09-29-incident-register-phase1.md` at `edb16ff`.

Independent baseline: **391 passed**, no errors or skips (`P/baseline.xml`). CPython **3.12.13**, macOS arm64; offline, niced heavy runs. Reviewer scheduling probes completed before reviewer mutation checks. The external 211-mutant merged output was not supplied before this report; no whole-sweep verdict is claimed. The reported fast-suite count is supplied evidence, not an independent rerun. A green sweep would not remove finding 1.

## r13 findings status

| Finding | Status | Measured rerun and limit |
|---|---|---|
| #1: unread arguments become witnessed nulls; duplicate names confuse slot kinds | **PARTIAL** | The exact r13 defect is resolved. Actual null accepts; actual string, duplicate names at import and duplicate names swapped during the call phase reject. The adapted slot unit accepts refusal and verifies `_UNREAD`, rather than requiring an unsupported call to be read. All five pass (`P/test_slot_unread.py`, `P/slot-async-frame.xml`, corresponding measurements). O:142–143 refuses duplicates, O:1099,1109 preserves markers, O:722–723 refuses a reached marker. Unique names still permit the false cell-content witness in new finding 1. |
| #2: async boundary callee omitted when callers are synchronous | **RESOLVED** | All three reruns pass: ordinary generator accepts; coroutine and async generator reject and each has an async boundary record. O:1182 includes the callee flags; C:143–144 refuses that interval (`P/test_async_boundary.py`, `P/slot-async-frame.xml`). O89 is killed by both async cases, with its generator control passing. |
| #3: coverage omits the pre-gate mock receiver read | **RESOLVED for the policy mismatch** | C:66–67,94–101 and L:140–147 now admit the non-cell, frame-owned `self` identity read at O:1067–1069. Function extraction remains after O:1090. The two read-set nodes pass in the baseline and kill O90, but their broader claim remains incomplete: an extra read outside `_fast_local` survives them (new finding 2). |
| #4: point sample presented as excluding reviewed native attachment | **PARTIAL** | C:100–101, O:105–106,1158–1159 and L:139 acknowledge the gap. Arrow attachments are now measured, including an ordinary Python-file read. The bounded full census race did not hit its intended interleaving, so the complete-read premise remains open. See §The open item (#4). |

The rest of the requested r13 controls were rerun:

| Rerun | Result and what it measures |
|---|---|
| Frame matrix, three nodes | Pass: **18** owner/layout cases, **7** cell/free-variable/code-swap cases, and one actual monitoring callback. These preserve argument identity or return unread without creating a locals dict in the measured frames. They cover compiler-produced cell layouts, not every crafted metadata/bytecode combination (`P/test_frame_matrix.py`, `P/slot-async-frame-runs/test_frame_owners_and_complete0/measurement.json`, `test_cells_free_variables_and_0/measurement.json`, `test_frame_from_monitoring_cal0/measurement.json`). |
| Cached generator × application lock × worker, four nodes | Pass: both twins report zero observer-site finalizers and cycle timeouts; alone accepts, worker rejects (`P/test_single_thread_cache.py`, `P/slot-async-frame.xml`). |
| Prior finalizer/path/lock/race/serialization controls | **27 pass**. They include the three r11 release/coordinated/cycle shapes, twelve current-path units, lock-window and pending-store controls. They establish the named refusals/bookkeeping behavior, not complete-window purity (`P/controls-final.xml`, retained probe sources and measurements). |
| Persistent-worker/concurrent attribution, three nodes | Pass as rejection checks: each records 20,000 boundaries, no attributed alert, no observer error. Both scheduled frame variants print `matched []`; neither proves the intended interleaving occurred this round (`P/test_concurrent_attribution.py`, `P/controls-final-runs/test_state_sampled_before_chil0/measurement.json` and `...chil1/measurement.json`). |
| Thread-state churn, ten sequential cases | Zero false `_alone` readings in **58,957,733** checks; zero wrongful `_track` holds in **79,540,527** checks across watchers × lock × GC. All drivers finish without errors. This tests the point predicate, not a lease (`P/churn/summary.json`). |

The first combined slot/frame run has 15 passes and two intended new regression failures. The expanded controls run has **34 passes and three intended failures**: the two full false certificates and their direct slot unit. Both runs have zero errors/skips (`P/rerun-computation.json`).

Every r13 false green is disposed separately:

| r13 false green | Status | Current evidence |
|---|---|---|
| Normal reader/import checks prove slot-kind correctness and unread propagation | **PARTIAL** | New duplicate/unread tests close the exact r13 cases; the unique-name false certificate still passes the import layout check. O:161–186 remains a normal-layout check, not a per-code physical-slot proof. |
| Async-entry test covers a callee-only async frame | **RESOLVED as a claim** | T:675–694 adds the missing three-case regression, and O89 kills both async cases. The old async-entry test remains narrow. |
| Concurrent refusal test proves no earlier application read | **PARTIAL** | Coverage now admits `self`, but T:381–386 still tests final refusal, and T:737–748 misses reads outside its slot spy. The extra-read QA mutant survives both read-set nodes. |
| O50 historical killer proves the clear transition | **RESOLVED as a claim** | T:775–795 directly asserts current/chain state. O50 is killed by its cleared-alone and cloned-alone nodes. Worker nodes survive that mutant because the separate concurrent cut produces the same state; they do not kill O77. The old end-to-end clear test is still not a unique transition oracle. |
| Unmodified r12 frame-attribution schedule reproduces the race | **NOT RESOLVED** | Its line-107/`head` condition is obsolete. The retained current-`me` variant also records no match in this rerun. Both green rejection tests must be reported as non-hit schedules. |
| Clean endpoints/conformance prove callback-free observation or argument-value integrity | **RESOLVED as plan wording** | L:63 states the endpoint limitation. Fresh pair fields are clean; finding 1 nevertheless passes conformance with a wrong boundary value. C:13–16 still needs the narrower wording described below. |

## New findings

1. **BLOCKER — unique names still let the reader unwrap a real cell-valued non-cell argument.** O:151 infers physical cell status from name membership in `co_cellvars`; O:155 then reads `cell_contents`. The duplicate guard at O:142 cannot prove that the executed `MAKE_CELL` instruction created a cell in that slot. Public code metadata replacement can move the cell name while leaving the bytecode untouched.

   Minimal application shape:

   ```python
   sent = []
   def send(recipient, text):
       def capture():
           return recipient
       sent.append((recipient, text))
   code = send.__code__
   send.__code__ = code.replace(
       co_varnames=('other', code.co_varnames[0]) + code.co_varnames[2:]
   )
   send('eric', types.CellType('BTS health CRITICAL: x'))
   ```

   All local names are unique exact strings. `MAKE_CELL` still operates on slot **0**; metadata now associates `recipient` with slot **1**. Slot 1 contains the genuine argument cell. The reader mistakes it for the frame's argument wrapper and returns its string contents. Bytecode SHA256 is unchanged (`P/slot-proof.json`). This changes an application code object through the public constructor and function setter; it modifies no bytecode, interpreter memory, observer, or standard-library function. Crafted code and code replacement are admitted by L:132,135 and O:819–820.

   `P/test_unique_cell_kind.py` checks in each full execution that `transport.sent[-1][1] is value`; observed and plain twins both confirm the actual argument is a cell. Measured expanded cases (`P/controls-final-runs/test_unique_names_never_unwrap0..4/measurement.json`, `P/rerun-computation.json`):

   | Case | Actual argument | Result |
   |---|---|---|
   | Normal code, ordinary cell-valued argument | `CellType(ALERT)` | rejected: incomplete `builtins.cell`, correct control |
   | Relabeled code, string argument | `ALERT` | rejected: unread, correct control |
   | Relabeled code installed at import | `CellType(ALERT)` | **accepted: false alert-string witness** |
   | Relabeled code installed during the mutant call phase | same cell | **accepted: false alert-string witness** |
   | Compiler-produced merged cell argument, whose value itself is a cell | same cell | rejected: incomplete `builtins.cell`, correct control |

   Both false certificates link entry/branch/event **2/3/4**, have no observer error, and record value `BTS health CRITICAL: x`, category `alert`, SHA256 **b52be06344621a3326af32846ffd5e9ffb0a959b1607abdceae44e68a559a1f5**. Their purity endpoint fields are clean and the observed/plain failures conform. The direct unit records `layout_verified: true`, `reader_is_argument: false`, `reader_is_contents: true` (`P/controls-final-runs/test_slot_kind_is_not_proved_b0/measurement.json`). A unique name present in both locals and freevars preserves the argument in the separate passing control; that overlap alone was not a defect (`test_a_name_in_both_locals_and0/measurement.json`).

   **Smallest conservative fix:** refuse cell-slot decoding instead of inferring and unwrapping it by name, retaining `_UNREAD` through the existing consumer refusal. Adjust the import cell check to validate that refusal so it does not disable safe non-cell reads globally. Missing a cell-dependent call is permitted by ruling 10. To retain cell support, independently verify that the executed pre-`RESUME` cell-creation layout agrees with the slot being decoded; refuse mismatches or unsupported layouts. A locals-plus metadata kind alone is insufficient when metadata and unchanged instructions disagree. Keep both full failures, the slot unit, and all controls as regressions. I have not implemented either repair or measured its prepared-spec cost.

2. **SHOULD — the read-set pin can stay green while an application read occurs outside its spy.** T:705–709 records only `_fast_local` invocations. T:709 fixes `_alone` to a constant; T:720–724 substitutes binding/shared-code checks; T:716–731 manually invokes `_on_start` from artificial boundary frames. Its `worker` label does not mean it starts a worker. It is a useful slot-call-shape unit, not a complete application-read oracle.

   A reviewer QA mutant inserts `hidden_application_locals = frame.f_locals` immediately after the mock frame acquisition at O:1065, before receiver extraction and the gate. Both read-set nodes still pass with unchanged inventory: **SURVIVED**, rc 0, zero failures/errors/skips (`P/readset_evasion.py`, `P/readset-evasion/summary.json`, baseline/mutant JUnit). That inserted read violates C:90–91 and the stated concurrent argument rule, but it is a throwaway mutant, not a claim that the pinned implementation currently uses `f_locals`. Source bytes were restored exactly.

   **Minimum claim edit, L:142:** replace “pins the read set” with “pins the `_fast_local` invocation sequence; reads outside that helper are checked separately.” Make the same qualification in L:537,547 and T:700–701,739–741. Add a separate regression that detects a pre-gate frame-locals read, including the cached-reference-release shape; keep a source guard against `f_locals` as supplementary evidence. A cells-flag mutant alone cannot establish absence of other reads.

3. **NIT — the corrected inferred-cause qualification has not reached every description.** T:475–480 correctly distinguishes the measured false point predicate from an inferred unlink/free sequence. O:110–112 still says the earlier reader “read a short-lived thread's freed state.” The churn outcomes establish false readings, not that exact memory-lifetime event (`P/churn/summary.json`; r13 report's corresponding distinction).

   **Minimum edit, O docstring:** “A version that followed the head's successor reported ‘alone’ while another thread lived. Reading an unlinked or freed state is an inferred cause, not a measured unlink/free sequence.” Also replace C:16's GC inference with: “Automatic garbage collection stays off; reference-count finalizers remain possible, and endpoint fields do not prove callback-free observation.” This aligns the prose with L:137,139 and the open native item without asserting a new measured finalizer.

## False greens

The current listed test with a demonstrated new false-green mechanism is **T:737–748**, `test_the_slots_read_while_another_thread_is_alive`: the extra `frame.f_locals` read survives it. O90 does test its declared argument to `_fast_local`, but the helper-call spy cannot certify all reads. T:665–670's identity unit reaches its positional marker first, so it does not independently exercise its keyword accessor; the eight fresh consumer controls supply positional, keyword, fallback-stop, opaque serialization, receiver and ambiguity checks (`P/unread-consumers.xml`).

The original slot/import checks remain green during finding 1. That is the decisive acceptance false green: a measured wrong value, not an unsupported/missed call. O87 and O88 kill the duplicate and marker repairs while neither protects a read that erroneously succeeds with another object's contents (`P/mutants/summary.json`, O:151–155).

Two reviewer schedule greens are explicitly non-proofs. Both current concurrent frame schedules have `matched []`. The native census regression has **coordinated_releases 0** in both twins. Its passing no-finalizer assertion therefore does not test the required after-sample census-reference release. Keep those hit counts visible rather than presenting their green statuses as successful race coverage (`P/controls-final-runs/test_state_sampled_before_chil0..1/measurement.json`, `P/native-race-prebuilt-runs/test_reviewed_native_attachmen0/measurement.json`).

Conformance compares node states and failing exception/source locations (D:112–128); it is not an independent boundary-value comparison. Purity checks read endpoint fields (C:114–145). Finding 1 passes both checks while recording the wrong value. L:63's stated limitation is appropriate.

## Rulings

**Ruling 10 stands.** The two absence specs are refused before execution. All demonstrated new false certificates are event certificates. No false returned-value certificate was measured, and no missed call is presented as a defect (C:3–8,102–104; `P/prepared/summary.json`).

**Ruling 11 remains the relevant trust boundary.** `_lookup` checks exact string keys by iteration (O:529–540); `_identity` rejects a reached marker before `_safe` (O:718–726). The new blocker arises earlier, where the reader returns the wrong application value as a successful read. It does not require application equality or dispatcher execution. The public local-name constructor rejects a string subclass on this interpreter (`SystemError`, `P/code-names.json`); no application-key callback was measured there.

**Ruling 12 is only partly established.** Its revised concurrent mock exception matches the pinned function and mock branches for a false point predicate (O:1065–1078,1089–1095). Frame/code identity and live code operands are separately admitted metadata. Its no-frame-refresh implementation claim holds in the inspected source, but the argument-value claim fails for unique-name cell relabeling. Its complete-read/native-attachment condition remains open. L:137's retained historical sentence “A concurrent boundary is refused before reading locals, receivers or arguments” should be explicitly qualified by the subsequent mock receiver exception; the heading and L:140 already do so.

The rev 14 table accurately maps the concrete repairs and tests (L:545–548, T:627–768, M:274–279). L:545's broader claim that unavailable-on-reader-failure now holds at the consumer is true for `_UNREAD`, but does not establish successful-read correctness. L:547 and L:537 overstate the read-set pin as finding 2 shows. L:548 correctly leaves the owner item open. L:549's direct O50 test and T:475–480 cause qualification are implemented. L:552's own-review entry matches O:1101–1106,1074–1078 and T:751–768: keyword-only slot 1 precedes function varargs slot 2, and the specified container cases are now unavailable. O91/O92 are killed by the intended unread cases, while the read and unaffected-positional controls pass.

## The open item (#4)

**Recommendation: audit the reachable reviewed-native path first; do not call a creation-counter comparison a complete-read lease.** Arrow is not merely an attachment-capable header. The reviewed installation actually invokes Python from native workers through an ordinary Python-file input path.

The inspected reviewed-venv sources and headers are preserved with hashes and excerpts in `P/abi-evidence.json`. Paths below are relative to the removed `wt14/.venv/lib/python3.12/site-packages/`, unless a CPython include path is named:

- `pyarrow/include/arrow/python/common.h:108–125`: `PyAcquireGIL` uses `PyGILState_Ensure` / `PyGILState_Release`.
- `pyarrow/io.pxi:879–888` and `pyarrow/include/arrow/python/io.h:36–59`: Python file objects can back Arrow reads. `pandas/io/parquet.py:141–145,260–265` obtains a binary Python handle and passes it to `pyarrow.parquet.read_table`.
- `pyarrow/_fs.pyx:1293–1311,1582–1588`: `PyFileSystem` connects native file operations to Python handler methods. `pyarrow/_dataset.pyx:3947–3969` runs scanning/table reads without the GIL.
- BTS test call phases reach parquet reads: `tests/data/test_build.py:108,120` uses `pd.read_parquet` on synthetic test output; `tests/leaderboard/test_storage.py:138–149` calls `pq.read_table`. Production code also has `src/bts/data/build.py:54–63`. These are code reads; no BTS data corpus was opened.

`P/probe_native_file.py` builds only synthetic parquet bytes. `pq.read_table(PythonInput(...), use_threads=True, pre_buffer=True)` reads 20,000 rows and calls the Python `read` method once from the main thread and once from a native worker with state ID 2. The creation counter changes **1 → 3**. The native `BufferReader` control reads the same 20,000 rows, invokes no Python input callback, and leaves that counter **3 → 3** (`P/native-file.json`). No Python worker is created by this probe. This measures the mechanism used by Python-file-backed Arrow reads; it is not a claim that a particular prepared-spec interval has that worker.

The separate in-memory `PyFileSystem` scan observes **240** native-worker callbacks and **240** distinct worker-state IDs across 256 file callbacks; the counter changes **1 → 241**. It reports zero callbacks between iterator pulls, so it supplies attachment evidence, not the census race (`P/native-arrow-summary.json`, `P/native-arrow-verified.json`).

The bounded full observer/plain race attempt uses the reviewed Arrow package, a Python filesystem handler and a referrer carrying the sender's code. It asks the native callback to drop that referrer's other owner only while the main thread is in `_code_shared`'s census generator (O:761–764). The final valid run executes 1,280 sender calls, records 126 worker callbacks, zero coordinated releases and zero observer-site finalizers in both twins. It accepts one real alert witness, records 1,199 concurrent refusals, and has no observer error (`P/native-race-prebuilt-runs/test_reviewed_native_attachmen0/measurement.json`). **The intended interleaving was not hit; the gap remains unaudited.**

There were two invalid dependency-setup attempts, followed by an observed child segfault when synthetic parquet writing was still inside the observed call; that run is rejected with rc **-11**, with no certificate (`P/native-race-final-runs/test_reviewed_native_attachmen0/measurement.json` and `out/mutant.stderr.txt`). Moving only fixture-byte construction before observation allows the bounded scan above to finish. I have not established the segfault's cause or reported it as a false certificate. Neither the rejected setup run nor the passing non-hit race proves complete-read safety.

For the proposed counter, the local **3.12.13** headers declare `interp->threads.next_unique_id` as an ordinary `uint64_t`, followed by the newest-first thread-state head (`/Users/eric/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/include/python3.12/internal/pycore_interp.h:91–104`). The assessed arm64 offsets are counter **64**, head **72**; the probes corroborate counter movement and API-reported state IDs. `internal/pycore_runtime.h:87–104` declares an interpreter-list mutex, but these headers do not establish which synchronization protects counter increments, attachment ordering, or the semantics of an arbitrary foreign-memory snapshot. Those need the exact implementation or a supported native reader; they cannot be inferred from the field's name.

Under verified monotonicity/snapshot assumptions, before/after comparison could detect a newly created state even if it has already exited before the final list sample. That is an optimistic invalidation check. It still has these limits:

- It does not exclude attachment or prevent a reference-count finalizer that already ran while a temporary reference was released. Rejecting evidence afterward cannot undo application effects.
- Its interval must include all acquisition, use and release of temporary references, plus cleanup/error paths. A final check followed by reference release leaves a new unguarded tail.
- Equality does not address cached-frame synchronization or other same-thread reference mutation, nor unsupported direct native object access, other interpreters, counter wrap/reinitialization, or unsynchronized reads. GC/signals remain separately constrained by the model.

A pre-existing state is normally detected by `_alone`; reusing an already-linked state is not, by itself, a demonstrated bypass. I have not measured a false certificate from native attachment. The current seven eligible prepared specs also show no newly unavailable link or attachment-dependent closure. Consequently this remains **SHOULD/open**, not a second proven acceptance blocker or a purpose-unfitness claim. If Eric chooses exclusion, state it explicitly and recheck prepared closures before accepting them; the present reviewed-venv clause alone supplies no such exclusion (C:105–109). A true prevention mechanism must protect the complete reference lifetime, rather than merely detect a changed counter afterward.

## Answers

**Unread propagation:** for a correctly produced marker, receiver matching uses identity (O:1110–1115,1171–1177); an unread bound-method receiver cannot match. Keyword lookup returns the marker without converting it (O:501–515,529–540), and a found marker stops fallback. `_identity` returns unavailable before serialization or classification (O:722–726). Shared-code/other-receiver reasons bypass identity entirely (O:1118–1129,1183–1184). Returns use the actual return callback operand, not `_fast_local`; an opaque marker/object has an incomplete safe form and therefore an unavailable return category (O:326–350,1207–1213). Eight narrow controls pass (`P/test_unread_consumers.py`, `P/unread-consumers.xml`). The dangerous remaining path is an erroneous successful cell read, not marker substitution. The merged-cell and local/freevar controls pass; the unique-name physical-cell mismatch does not.

**Prepared cost at `e927a75`:**

| Spec | Result | Concurrent / unread / async records |
|---|---|---|
| I-0811-a | accepted | 0 / 0 / 0 |
| I-0811-b | accepted | 0 / 0 / 0 |
| I-0813-a | accepted | 0 / 0 / 0 |
| I-0813-b | refused: absence | no stages |
| I-0830-a | accepted | 0 / 0 / 0 |
| I-0830-b | refused: absence | no stages |
| I-0830-c | accepted | 0 / 0 / 0 |
| I-0830-d | accepted | 0 / 0 / 0 |
| I-0903-a | accepted | 0 / 0 / 0 |

No eligible spec loses acceptance due to #1 or #2 in this bounded set (`P/prepared/summary.json`). The synthetic ref is **d497766eb10927cbafd6dc33505f839ea53c4b01**. `P/frozen/equivalence.json` hashes **566** copied source/config/registry/spec/tool files against `e927a75`; every prepared spec's original bytes are separately hashed, and only its baseline ref is substituted. Inputs are byte-checked before/after execution. The pair and prepared synthetic worktrees were removed (`P/execution-binding.json`, `P/prepared/cleanup.json`). Synthetic commits are fixture setup, not BTS history changes.

**Evidence bindings:** archived pair output names `ref: e927a75`; predicate output binds the full resolved ref plus independently matching fixture/script hashes; closure's header includes `e927a75`, and its body is byte-identical to r13, SHA256 **fe2b524d576a932610fca6de7fe004cc05918ee9a9797832ca47f218ddf91121** (`P/evidence-bindings.json`). Fresh pair accepts **22/22**, with **40** controls, **40** imported BTS modules in each run, no unavailable import, 44 clean endpoint fields per run, and no concurrency/unread/async record (`P/pair-summary.json`). All eleven predicate cases rerun **ALL AS REQUIRED**, with source/fixture restoration verified (`P/predicate-rerun.log`, `P/predicate-rerun-binding.json`). The historical closure body is not a native-worker reachability audit, as the Arrow probe shows.

**Mutation QA:** all **211** anchors are unique and compile (`P/mutants/anchors.json`). Each scoped baseline is clean; each mutation uses the same node inventory, kills through an assertion failure rather than an error/skip, and is restored byte-for-byte (`P/mutants/summary.json`):

| Mutant | Scoped nodes | Measured killer / scope |
|---|---:|---|
| O87 | 1 | duplicate-name slot refusal |
| O88 | 1 | positional unread identity; keyword independently covered by reviewer controls |
| O89 | 3 | coroutine and async generator; generator control passes |
| O90 | 2 | both slot-spy nodes detect `cells=True`; not an all-read oracle |
| O91 | 3 | function varargs-unread case |
| O92 | 3 | both mock unread-container cases |
| O50 | 4 | clear-alone and clone-alone state assertions; concurrent nodes remain controls |

These seven clean kills are not a full 211-mutant sweep. A subsequent acceptance review still needs the merged sweep at the exact repaired code pin: clean, matching baselines/inventories and genuine assertion kills for every intended non-equivalent mutant, with no mutant-error, collection/error/skip substitute or unaccounted survivor. The present **BLOCK** is independent of that pending output.

Cleanup verified: `git worktree remove --force` removed `wt14` before this final line was written. Its path and Git registration are both gone; main tracked files remain clean (`P/cleanup.json`). The review used only code and synthetic test inputs, with no BTS corpus read, network operation, actual BTS commit or production action. The excluded worktrees and other panes were not operated.

DONE
