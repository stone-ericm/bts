## Verdict

**BLOCK.** Three full executions accept an alert certificate for a value the boundary never received. A generator's later `RESUME(0)` causes another start callback after its body has changed the argument slot. The observer reads that changed local as an initial argument. The 404-pass review baseline, prologue unit, and O93–O99 kills do not cover this case.

Pins: code **f0725032c27632a95a9102c3171f5b41e10dfbd1**, evidence **53b70939bb7dae7574e4b41467029cd9c3987ea1**, plan **7cf04ea8f2824ba5df4de3bc1dda2a15e6e0be3b**. Evidence changes no `src`, `scripts`, or `tests` bytes relative to code. **P** means `/Users/eric/projects/bts/.codex-review/2026-09-29-incident-register/r8-probes/r15`. At f072503: **O** = `scripts/audit/incident_register/observer.py`; **C** = `certify.py` in that directory; **R** = `runner.py`; **T** = `tests/scripts/incident_register/test_r10_counterexamples.py`; **M** = `docs/audit/2026-09-29-incident-register-evidence/tooling/mutation_sweep.py`. **L** = `docs/superpowers/plans/2026-09-29-incident-register-phase1.md` at 7cf04ea. Computed pins and hashes: P/evidence-bindings.json.

Smallest repair for the confirmed blocker: validate the monitoring start offset against the first supported entry `RESUME` before extracting boundary arguments or registering swapped code. Ignore/refuse a later start as a witness. Preserve ordinary generator and cell support, and retain the four-case regression below. This is a proposed repair; it has not been implemented or independently verified as an exhaustive validator of arbitrary bytecode and exception tables.

Independent baseline: **404 passed**, no errors or skips. Reused value run: **31 passed, three failed**, no errors or skips; the three failures are the new rejection expectations. Saved exception and value probes were not rerun. CPython 3.12.13, macOS arm64; offline, niced heavy runs (P/baseline.xml, P/value.xml, P/exceptions/summary.json). The external full 217-mutant merged output has not been supplied. A green full sweep cannot remove the measured blocker.

## r14 findings status

| r14 item | Status | Measured rerun and limit |
|---|---|---|
| #1: unique-name cell relabeling | **PARTIAL** | The exact prior defect is **resolved**: all seven reviewer unique-cell cases/unit controls pass. Both relabel false certificates are now refused; ordinary cell objects are retained or incomplete, rather than certified as their contents. O:173–176 checks executed-prefix slots against named slots. The new later-entry case still violates successful argument-read correctness (finding 1). P/test_unique_cell_kind.py; P/value.xml and corresponding measurements. |
| #2: read-set evasion | **PARTIAL** | The exact `frame.f_locals` insertion is **resolved**: the four no-dict/stale-dict × worker/alone nodes kill it. The invocation-sequence claim is qualified, and the AST guard covers named materialising APIs. A raw argument-slot conversion outside the helper survives all five checks (finding 2). P/readset-evasion/summary.json; P/readset-raw-evasion/summary.json; T:706–775,956–970. |
| #3: inferred cause / endpoint wording | **RESOLVED** | O:114–116 now distinguishes the inferred lifetime cause from the measured false predicate. C:12–17 expressly says endpoint fields do not prove callback-free observation and that reference-count finalizers remain possible. L:566 matches these edits. |
| Open #4: reviewed-native attachment | **PARTIAL / OPEN** | The function-watcher defect is removed, but `_alone` remains a point predicate. Fresh Arrow input controls reproduce native attachment; the full census race has zero coordinated releases. No false certificate from native attachment was measured. P/native-file.json; P/native-race-runs/.../measurement.json; L:142. |

| Requested rerun | Measured result and what it establishes |
|---|---|
| Unread-consumer controls | **8 pass**: positional, keyword, fallback-stop, receiver, opaque return/serialization, and ambiguity cases retain refusal. A correctly produced marker is unavailable, rather than a witnessed null or empty call (P/test_unread_consumers.py; P/value.xml; O:1121–1129,1130–1148). |
| Slot / async / frame / cached generator | **5 / 3 / 3 / 4 pass**. Slot tests allow safe refusal; ordinary generator accepts, coroutine and async generator reject; the frame matrix exercises 18 owner/layout and seven cell/free-variable/swap shapes plus an actual monitoring callback; cached-generator twins report zero observer-site finalizers and cycle timeouts, accepting alone and rejecting with a worker. These are bounded controls, not a proof for every code object (P/value.xml and the named probe measurement files). |
| Prior finalizer, path, lock, pending-store, serialization, and concurrent-attribution probes | Initial copied probes: **26 pass, two fail**, no errors/skips (P/controls.xml). The two failures were obsolete reviewer expectations: a concurrent call was still expected to accept, and the old lock-cycle test still expected an observer-site finalizer. Outputs instead show rejection, zero observer-site finalizers, zero cycle timeouts and watcher waits in both twins. Correcting only those assertions gives **2 pass** (P/referrer-fixed.xml; `_fixed.py` sources). The original results are retained. |
| Concurrent-attribution schedules | All three rejection checks pass, each with 20,000 boundaries and no attributed alert or observer error. Both frame schedules print `matched []`; they do not establish the intended interleaving (P/controls-runs/test_state_sampled_before_chil0..1/measurement.json and persistent-worker measurement). |
| Churn | **10 completed cases**, zero wrong samples/errors, with 8 watcher × lock × GC tracking combinations and two GC modes of the point predicate. They test a changing thread list, not prevention of attachment during an entire read (P/churn/summary.json). |
| Prepared-spec cost | **Seven eligible specs accept**, zero concurrent/unread/async records; I-0813-b and I-0830-b are refused as absence claims before any stage. No acceptance cost appears in these eligible cases. P/prepared/summary.json; details below. |

Every r14 false-green disposition:

| Prior false green | Status |
|---|---|
| Read-set unit admits an extra `frame.f_locals` read | **RESOLVED for that insertion; PARTIAL for all-read sensitivity.** The old insertion is killed; the raw-read regression survives. |
| Positional unread unit reaches its marker before exercising keyword access | **PARTIAL.** That unit still proves only its reached path. The eight independently rerun consumer controls cover keyword refusal separately; do not describe the positional assertion as a keyword test. |
| Slot/import greens during the unique-name mismatch; O87/O88 did not protect erroneous successful reads | **RESOLVED for the exact mismatch; PARTIAL for the general claim.** Seven old cases pass with correct dispositions, but later-entry extraction remains uncovered. |
| Concurrent frame races print `matched []` | **NOT RESOLVED as race coverage.** Both current schedules again report no match. Their refusal checks pass. |
| Native census race has no coordinated reference release | **NOT RESOLVED as complete-read coverage.** Current observed/plain twins each report 126 worker callbacks, zero coordinated releases, zero observer-site finalizers, 128 released objects and 1,280 sender calls. |
| Conformance / endpoint purity imply correct boundary values | **NOT RESOLVED as such an inference.** Those checks still accept finding 1. Their narrower recorded-state/endpoint claims are useful; they do not independently validate an argument witness (C:10–34; P/value-runs/test_a_later_resume*/measurement.json). |

## New findings

1. **BLOCKER — a later start callback certifies a changed local as an argument.** O:1080 receives `offset`, but O:1083–1148 does not check it. O:142–161 stops at the first `RESUME`; O:173–188 checks cell layout and unwraps the live slot. Prefix agreement therefore does not establish that this callback is the initial argument-reading point.

   The regression keeps a genuine compiler-produced merged cell argument and generator prologue. Its boundary appends the exact received value to both `received` and `sent`, yields, assigns the alert string to its local `text`, and yields again. Only the final `RESUME` argument changes from its ordinary resume value to zero. The resumed generator supplies another `PY_START`; there is no second boundary invocation receiving an alert. The helper asserts `received[-1][1] is value` and `sent[-1][1] is value` in both twins (P/test_resume_entry.py:7–32).

   | Case | Actual original argument | Certificate |
   |---|---|---|
   | Ordinary code | `CellType(ALERT)` | rejected; correct control |
   | Later `RESUME(0)`, installed at import | same cell | **accepted**, links **2/3/5** |
   | Later `RESUME(0)`, swapped during the mutant call | same cell | **accepted**, links **2/3/5** |
   | Same bytes via direct `types.CodeType(...)` | same cell | **accepted**, links **2/3/5** |

   In each accepted run the first boundary (seq 4) is incomplete `builtins.cell`; the later boundary (seq 5) certifies `BTS health CRITICAL: x`, SHA256 `b52be06344621a3326af32846ffd5e9ffb0a959b1607abdceae44e68a559a1f5`, category `alert`. No observer error occurs; observed/plain output confirms the cell argument. Evidence: P/value.xml, P/value-runs/test_a_later_resume*/measurement.json and stage event/stdout files. Source and exception-table construction are explicit in P/test_resume_entry.py:17–21,38–42.

   This uses the public code constructor/function setter and executable generator instructions, with no interpreter memory write, observer/evidence alteration, standard-library replacement, or new native component. Crafted code is expressly contemplated by L:135,138 and this review's prologue question. It is an in-model false positive, not a missed-call complaint. Even without cell decoding, a later start could read an already changed non-cell argument; blanket cell refusal would not address this start-point defect.

   **Smallest fix:** require the callback offset to be the independently verified first supported entry `RESUME` (including its expected entry argument), before boundary reads and `_swapped` registration. A later start may be ignored or unavailable under ruling 10. Keep the import, swap, constructor and ordinary controls as full-stage regressions; use correct actual entry offsets in artificial-frame units. The general “interpreter ran exactly these” assertion also needs qualification: this parser is a prefix-shape check, not a control-flow/exception-table proof (O:143–146; L:141).

2. **SHOULD — a raw pre-gate argument read evades all read-set checks.** T:715–719 observes only `_fast_local`; T:728–734,747–753 detects locals-dict creation/refresh; T:956–970 bans selected materialising APIs. None detects an address-to-Python conversion that leaves the locals dict untouched.

   The reviewer QA mutant, immediately after mock frame acquisition, reads its interpreter-frame pointer, obtains the mock argument-tuple slot address and calls `ctypes.cast(extra_slot, ctypes.py_object).value` before the thread gate. This reads an application argument while the `worker` unit has `_alone=False`; it bypasses the slot spy and creates/refreshes no locals dict. The four read-set nodes **plus the AST guard** remain clean: **SURVIVED**, rc 0, identical five-node inventory, no failures/errors/skips (P/readset_raw_evasion.py; P/readset-raw-evasion/summary.json and JUnit). Source is restored byte-for-byte.

   This is a test-sensitivity defect, not evidence that f072503 currently performs that inserted read. **Smallest QA fix:** retain this mutant as a required-kill regression and instrument/restrict address-to-Python frame-slot conversions outside the verified reader, distinguishing admitted mock `self` from arguments. Keep the current precise invocation-sequence wording; do not upgrade the locals-dict checks to an all-read oracle (L:145,565).

## False greens

The decisive acceptance false green is the original prologue/import/slot coverage while finding 1 is accepted. T:863–887 and O93–O95 test prefix shape, relabeling and extended operands; they do not test that the callback occurs at that prefix's first entry. A correct layout is not evidence that the live slot still holds the original argument.

The demonstrated new sensitivity false green is the combined four-node read-set test plus source guard under the raw-read mutant. O96/O97 correctly kill `frame.f_locals`; their success does not generalise to all application reads. T:719 and T:933 explicitly stub `_alone`; those “worker” units start no worker and claim no actual race.

The concurrency schedules' `matched []` and native race's zero coordinated releases are non-hit measurements. The native race accepts two actual alert witnesses and records 1,179 concurrent boundaries without an observer error; that does not exercise the requested census-reference-release interleaving (P/native-race-runs/.../measurement.json). Conformance and clean endpoint fields are not independent boundary receipts, as the wrong-value certificates demonstrate (C:10–34; P/value.xml).

## Rulings

**10:** retained. The new blocker is a positive event certificate; no false returned-value certificate was measured. Missing or unsupported calls are not findings (C:3–8,77,107–109).

**11:** the prior dictionary/equality rules remain enforced in the measured controls. The new blocker occurs before classification: a successful read returns a value from the wrong point of the invocation. It requires no application comparison/dispatch (L:128–138; O:173–188; P/value.xml).

**12:** not exactly established as written. Its no-`f_locals` implementation and the qualified mock `self` exception match the inspected code and old-evasion kills (L:140–150; O:1085–1095,1112–1132). The prefix cell-set comparison is implemented as stated, but “no slot misread” and “instructions the interpreter actually ran” overstate what follows: no callback-offset or exception-path check exists. Minimum text qualification after repairing finding 1: “The prefix check verifies cell-layout agreement at the supported initial entry offset; unsupported entry paths are not argument witnesses.” Do not present this as an exhaustive arbitrary-bytecode validator without additional enforcement.

The rev 15 mappings for relabeling, extended operands, locals dictionaries, wording and watcher removal match the commits and tests (L:564–568,572; O:142–191,943–964; T:836–970; M:283–289). L:565 is appropriately specific about the locals-dict checks; it does not establish all-read sensitivity. The historical r12/r13 table entries are historical, rather than descriptions of a still-present function watcher (L:541,556–557).

The own-review mechanism is independently supported: old observer `sorted` raises `SystemError` and records `func_watch`; current observer preserves `TypeError`; the separate full old stage sends the alert only when observed and is refused through observer errors. Saved trivial-watcher Arrow control: plain rc 0, watcher **rc 139**. This supports the proposed cause of r14 **rc -11**; it is not a fresh reproduction of that original process. The dictionary residual remains reasoning, not a demonstrated current false certificate (P/exceptions/*.json; P/legacy-full-runs/.../measurement.json; L:151,572).

## The open item (#4)

Audit the reachable reviewed-native path before treating a thread sample as protection for a whole read. A before/after creation counter can detect attachment afterward, but cannot undo a finalizer that already ran. If Eric chooses an exclusion instead, state it explicitly and check every prepared closure against it.

The recommendation is unchanged (L:142). In the reviewed pyarrow 23.0.1 installation, the fresh synthetic Python-file input reads 20,000 rows, invokes a worker callback with state ID 2 and changes the creation counter **1 → 3**. Its native-buffer control reads the same rows with no Python callback and counter **3 → 3** (P/native-file.json). O:108–109,1180–1182 correctly acknowledge the gap. The full race is still a non-hit, despite real worker activity (P/native-race-runs/.../measurement.json). Watcher removal removes a distinct exception-changing mechanism; it supplies no complete-read lease. No measured native-attachment false certificate or prepared closure depending on an excluded mechanism is claimed here.

## Answers

**Prologue and `co_code`.** Six independent first-entry controls preserve identity and leave the locals dict NULL: warmed ordinary code, generator, coroutine, `COPY_FREE_VARS` plus a merged cell argument, slot 299 with `EXTENDED_ARG`, and direct CodeType construction. Actual callback offsets equal first `RESUME` offsets **2,6,6,4,4,2**; in each `co_code` still exposes opcode **151**, entry argument **0**, under monitoring (P/probe_entry_shapes.py; P/entry-shapes/summary.json). A separate control warms an actual `BINARY_OP_ADD_INT` specialization. During monitoring its adaptive bytes expose `INSTRUMENTED_RESUME`, while public `co_code` still exposes `RESUME` 151 with argument 0 and the reader preserves the argument identity (P/probe_specialized.py; P/specialized/summary.json). The saved frame matrix covers additional compiler-produced cell/free-variable layouts. Coroutine readability here is a reader control, not permission to certify async calls; the async full controls still reject them.

The prologue parser does not inspect the later body or exception table and does not validate all allowed-opcode placement/counts. Finding 1 keeps the compiler-produced prologue and table unchanged while the body supplies a later entry event. Thus neither `replace` nor direct CodeType construction is inherently safe merely because the cell sets match. Code-table paths that bypass part of cell setup must be refused unless the reader can establish the physical layout at the witnessing entry; no such extra validator or broad correctness proof is claimed (O:142–161; P/test_resume_entry.py).

Inside the actual reader the object is exact CodeType before `co_code` access (O:167,173). The installed CPython header declares `PyCode_GetCode` equivalent to that attribute and returning a strong reference to bytes; `_PyCoCached` has an owned `_co_code` field (`.../include/python3.12/cpython/code.h:70–76,362–364` under `/Users/eric/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none`). Reading may allocate/memoise bytes and release temporary builtin references; it is not literally free of all decrefs. It does not dispatch an application getter on exact CodeType or drop a last application constant reference: the code retains its constants. A retained constant with a finalizer records zero finalizers across 1,000 reads/parser calls (P/entry-shapes/summary.json). That sample supports this ownership argument, rather than proving all observer paths callback-free.

**Remaining callback exposure.** The sole observer CFUNCTYPE registration is the dict watcher (O:85–90,957–964); the function watcher is absent. Holding watched namespaces (O:685) prevents their deallocation while held, but not modification, clear or clone. The installed `_PyDict_NotifyEvent` header dispatches watcher bits for a live dict (`internal/pycore_dict.h:155–175`); that declaration alone does not establish pending-error preservation. I found no concrete reviewed extension path mutating a watched namespace with a pending C error, and measured no current exception replacement through it. A Python handled-exception store preserves the same exception with no observer error (P/callbacks/summary.json), but is not a pending-C-error test. Do not infer complete safety from that control or from namespace liveness. The residual is open reasoning, not a new proven blocker (L:151).

`PY_START`, `LINE`, `PY_RETURN` and `PY_UNWIND` are Python monitoring callbacks, not observer-owned ctypes trampolines (O:1004–1009,1242–1254). A real current-monitor unwind records the exceptional exit while the caller receives the identical TypeError instance, with no observer error (P/callbacks/summary.json). `_on_unwind` passes the exception to `_exit`, which uses only its type for an exceptional return (O:1229–1233). These results distinguish monitoring from the removed function-watcher path; they are not an exhaustive native-error audit.

The bootstrap census is also a Python audit callback. Its own body compares the event name and increments an owned counter (R:39–51); it neither registers another ctypes callback nor reads application argument contents. The additional control records a newly installed hook as count 1 at the endpoint (P/callbacks/summary.json), which C:77–78 makes unavailable. Audit hooks installed before the trusted bootstrap and standard-library replacement remain explicit exclusions (C:110–115). Refcount finalizers, reviewed-native attachment, and the admitted mock receiver read are separate concerns; removing the function watcher does not discharge them.

**Swapped-code attribution.** `_swapped` reads held exact functions' `__code__` by identity only while alone, and registers candidate spec/receiver pairs (O:943–955). Registration is not acceptance: the start path rechecks shared-code count, receiver and live binding under the lock (O:1134–1148; `_is_current_call`, O:1194–1200). The full r5 swap node passes in the 404-node baseline (`tests/scripts/incident_register/test_r5_counterexamples.py:162`); the twelve fresh path-level controls pass. No additional false binding attribution from registration alone was measured. It nevertheless participates in finding 1 because a later start is accepted as a call start. The lock and call-time read also retain the explicit reviewed-native attachment limitation (O:692–697,1175–1182).

**Prepared cost and execution binding.** The fresh synthetic ref is **df327da579b28c69505ce67104349c3ccd184a54**, with **566** source/config/registry/spec/tool files hashed byte-for-byte against f072503. Files are checked before/after execution; each of nine original specs has its own original-byte hash and only its baseline ref is substituted. Synthetic fixture history does not change BTS history. The synthetic pair/prepared worktrees are gone (P/frozen/equivalence.json; P/prepared/summary.json,cleanup.json; P/execution-binding.json).

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

The fresh pair accepts **22/22**, with **40** passing controls, 40 readable imported BTS modules and 44 endpoints per twin, no observer error or concurrent/unread/async record. The independent stage headers identify observer SHA256 **c0892edcf58c99258e5302a6bf69e14cf580d33ab7f423e46f3be274ced527e4**, matching f072503. The value, control, corrected-referrer and native-race stage sources match that hash; the distinct legacy comparison matches its explicitly recorded e927a75 source. The saved exception artifacts and value.xml are hashed and reused, not rerun (P/execution-binding.json). The reported fast-suite count is supplied evidence, not an independent run in this review.

**Archived evidence and mutations.** Archived current pair output names f072503 and accepts 22/22. The predicate output names the resolved full pin and independently matching fixture/script hashes; all eleven are marked as required. The closure header includes f072503, and its body hash is unchanged, `fe2b524d576a932610fca6de7fe004cc05918ee9a9797832ca47f218ddf91121` (P/evidence-bindings.json). That text screen is not an audit of reviewed-native reachability.

All **217** anchors are unique and compile; O19 is absent (P/mutants/anchors.json; M:47–50,340–349). Each scoped baseline is clean, inventories match, and source bytes are restored. All seven new mutants are **KILLED** by assertion failures without errors/skips (P/mutants/summary.json):

| Mutant | Scoped nodes | Actual killer / limit |
|---|---:|---|
| O93 | 1 | prefix/name disagreement |
| O94 | 1 | unexpected prefix instruction |
| O95 | 1 | extended operand at slot 299 |
| O96 | 4 | all mock no-dict/stale-dict × worker/alone cases |
| O97 | 4 | all function no-dict/stale-dict × worker/alone cases |
| O98 | 2 | alone registration case; worker control passes |
| O99 | 2 | worker refusal case; alone control passes |

O19's retirement is correct: its old function-watcher branch is gone and O98/O99 test the replacement. Those two units measure registration under a stubbed point predicate, not a concurrent integration schedule (T:928–953). These are seven scoped kills, not a full-sweep verdict. A subsequent acceptance still needs a merged sweep at the repaired exact pin: clean matching baselines/inventories, genuine assertion kills for each intended non-equivalent mutant, no invalid anchor, error/collection/skip substitute, missing label or unaccounted survivor. The present BLOCK does not depend on that pending output.

Cleanup verified: `git worktree remove --force` removed **wt15** before this last line was written. Its path and Git registration are gone; main tracked files remain clean (P/cleanup.json). Review inputs were code and synthetic fixtures only: no BTS data corpus read, network/SSH/gh operation, BTS commit, production action, or operation of the excluded worktrees or other panes. Saved value and exception artifacts remain preserved.

DONE
