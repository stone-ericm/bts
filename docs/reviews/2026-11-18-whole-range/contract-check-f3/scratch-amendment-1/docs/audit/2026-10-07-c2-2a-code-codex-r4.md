## Verdict

**BLOCK.**

Reviewed-commit: 0e32f0d97c509ece3549e6b7c051ba66c43979ee

The original round-3 counterexamples are repaired, but the class is not closed. An excluded entry fault in the new hashing writer loses an available forecast. Two reached-but-untested sweep pairs publish a false calibration status. A hash-accounting interruption can publish a wrong cache digest with every required component present and no errors. A failed context start can similarly contaminate an enclosing run's model provenance.

The permitted suite passed: **282 passed, 1 skipped in 234.71 seconds**. My all-61-scenario first-reach sweep also passed: **660 faulted runs, 0 failed, all 61 plain checks equal to their goldens**. Neither green result contradicts the counterexamples: one point is excluded, two pairs are omitted, and the other combinations/state are absent from the scenarios. The sweep's digest oracle also has an independently reproduced false green.

Cost meets the numerical rules as explicitly amended by Eric, with the original failing attempt retained. Cost is not the reason for this BLOCK. HEAD is the requested detached commit; the tracked tree was clean before and after review.

This is Eric's fourth and final round. **A non-SIGN stops 2a for this cycle; there is no fifth round.** This report approves no merge, certification, deployment or activation. The W1.5 recertifications, a new blind whole-range review and Eric's D7 remain separate, unperformed gates.

## Findings

### R4-1 — Blocking: the excluded writer-entry fault loses the forecast before the forwarded write

**Locations:** `src/bts/serving_witness.py:311–319`; `tests/c2_2a/sweep.py:96–103` (`COMPUTATION`); `src/bts/model/predict.py:158`.

The sweep excludes the new `HashingWriter.write` function's **call event**, classifying it as the genuine forwarded-write operation. Those are different boundaries. Entry into this added Python wrapper occurs before either its provenance update or `self._f.write(b)`. Its internal guards cannot contain a failed entry, and `pickle.dump` has no ancillary-failure recovery around it. The prediction handler then treats this new failure as computation failure.

I used the unmodified sweep's own `record`, `inject` and `verdict` on `model_cold`, including the excluded point:

```text
call serving_witness.py:311 <- predict.py:158
```

The unfaulted traced scenario equals the deployed golden. The entry injection fires; the sweep's own verdict rejects the result for a deployed-surface difference:

| result | unfaulted / f882411 golden | entry-fault candidate |
|---|---|---|
| persisted slate | present | absent |
| observed selections | 1 | 0 |
| cache SHA-256 | `d3470c120db231435b3cbbec4c1ffe7e814c2903db2684493aba9f8a68520ff2` | empty-file digest `e3b0c442…` |
| handled failure | none at this boundary | `Prediction failed: sweep: call serving_witness.py:311` |

The file is already opened/truncated, but the underlying write never runs. This is a deterministic injected wrapper-entry failure, not a measured disk failure or allocator exhaustion. The deployed cache save uses the file's write method without this new Python wrapper. I do not accept treating failure to enter the added provenance wrapper as genuine underlying I/O merely because the method later forwards a write. `genuine_partial_write` faults the underlying write; it does not establish containment or baseline equivalence at this new entry boundary.

The other computation exclusions need that distinction too. Faulting the actual forwarded write or unpickle is a genuine operation failure; R10's assignment is the intended computation change. Their genuine-error scenarios support those operations. They do not justify silently exempting all preparation/invocation work that surrounds them. NOP-only lines perform no operation and are reasonable exclusions within the declared fault model.

### R4-2 — Blocking: an omitted pair says calibration failed before assignment when it actually succeeded

**Locations:** `src/bts/serving_witness.py:368,386–399,561–564`; `src/bts/orchestrator.py:173–177,195–198`.

`Calibration.applied` starts as false. If the hook that records successful assignment fails, that false is an unknown fact, not evidence that assignment did not happen. A subsequent genuine error sets `failure`; `record()` then publishes `status="failed", applied=false` instead of withholding the unknown state.

I ran two additional pairs in the existing `calibration_error_after_assignment` scenario, which already raises its genuine stderr `OSError` **after** probability assignment:

```text
call serving_witness.py:561 <- orchestrator.py:177
line serving_witness.py:564 <- orchestrator.py:177
```

For both, the trace fires and the entire deployed surface still equals the f882411 golden. The plain candidate says `status=applied, applied=true`. The faulted candidate persists **`status=failed, applied=false`**, with `calibration: OSError: golden: stderr failure after assignment`. The returned forecast remains calibrated. The error description also incorrectly loses the distinction that this happened after assignment.

The sweep's own verdict returns `ok=false`, specifically:

```text
serving.calibration.applied: False != plain True
serving.calibration.status: 'failed' != plain 'applied'
```

I mechanically verified that **both pairs occur in the retained run's reached inventory and neither was faulted**. They were instead faulted in plain scenarios where losing the hook produces unknown/null status. This is a concrete counterexample to the totality argument that a point's consequence does not depend on its scenario. Guarding the call preserves computation; it does not make the remaining metadata inference truthful. The signed §3.2 `failed` status denotes failure before successful assignment, and is not earned here.

### R4-3 — Blocking: equal byte counters can certify the wrong cache digest

**Locations:** `src/bts/serving_witness.py:322–326,521–537`.

The new writer hashes first, then increments `hashed`. An interruption between those operations can leave the hash mutated but the counter unchanged. A successful short write can independently leave the file position unchanged. Equality of the counters then does not establish that the hash represents the written bytes.

Independent unit reproduction:

```python
class ZeroSink(io.BytesIO):
    def write(self, b):
        return 0
# writer.write(b"abcdef"), with MemoryError at the line after _h.update(b):
#     self.hashed += memoryview(b).nbytes
```

The call returns the deployed result, 0, and writes no bytes. The fault at `:323` is contained. Nevertheless `hashed == tell() == 0`, and `digest()` returns SHA-256 of `abcdef` (`bef57ec7…`) rather than null. The actual empty-byte digest is `e3b0c442…`.

I then ran the real baseline and candidate prediction/calibration functions with the shared synthetic 40-pick fixture and a zero-byte successful cache sink. Only heavy feature/training/prediction leaves are stand-ins. A single trace exception interrupts the candidate's counter after hash mutation:

| result | f882411 | candidate |
|---|---|---|
| probabilities | `[0.65, 0.65, 0.65]` | identical |
| cache writes | one 52-byte argument, result 0 | identical |
| persisted cache bytes | empty | empty |
| cache SHA-256 | `e3b0c442…` | identical |
| published model digest | no witness | **`b089ecc6c5459dc5888bba9aea3ba268c7dec04fcfa5ddf9d2f645264c037ec0`** |

Real `save_slate` persists that wrong model digest. **All required witness/calibration components remain non-null**, both PA records are hashed, there are 40 pick inputs and bindings, calibration is applied, and both error lists are empty. There is no incomplete-part signal elsewhere to rescue this case.

This combines a bounded ancillary interruption with a successful short write, an explicitly covered contract shape. It is not evidence that a normal buffered file spontaneously returns zero or that real OOM occurred. The zero result is a permitted synthetic successful-short-write boundary; no computation retry or output corruption was manufactured. The defect is the false hash claim. `_h.update` and its separately fallible byte accounting are not an atomic certificate of written bytes.

### R4-4 — Conditional scoping defect: a failed nested context start changes the enclosing witness

**Locations:** `src/bts/orchestrator.py:129–132,201–204`; `src/bts/serving_witness.py:173–185,499–512,521–537`.

If `begin(witness)` fails, its handler leaves `token=None` but does not prevent subsequent hooks from using an already-current enclosing witness. The inner run therefore records its model/save facts into the outer run. Its own `run_end` seals/attaches its separate witness, then `end(None)` fails; the enclosing current witness survives with the inner facts.

I reproduced this with a real nested `predict_local` invocation from the outer fixture's prediction leaf. The outer run uses a cached model; the inner run trains into a different synthetic models directory, with calibration off. A single trace exception at the **second `begin` call event** (`:173`) prevents only the inner context start. Both sides otherwise execute their real functions; baseline functions come from actual local `git show f882411` source.

Both baseline and candidate return the outer calibrated `[0.65, 0.65, 0.65]`, with the identical 53-byte outer cache, SHA-256 `468b3af364abf1df8ba10f1dbcaa993c349cff4b8aab55812e8f52f2fc1d699a`. The candidate's outer persisted witness instead says **`source=trained`**, SHA-256 `b089ecc6…`, belonging to the inner 52-byte save. Every required component is present, with 40 calibration inputs/bindings and empty error lists.

This is a synthetic **nested-call** counterexample, not evidence that the current production call graph already re-enters `predict_local`. Nesting was expressly part of this review. A supported nesting/scoping claim must fail closed on failed admission rather than silently recording into another owner; otherwise the exclusion must be explicit. The sweep's fresh-interpreter isolation and current scenarios do not test this state.

The disclosed prediction-failure/failed-stop residue is a separate two-fault case. Its ordinary subsequent successful run is isolated correctly by the tested new context and reset. The residue is not attached, so that tested follow-up serves its own witness. But it is not inert: `current()` returns it and other hooks record into it. Besides the disclosed held-byte references, repeated direct calibration calls can accumulate pick/binding metadata in its lists. That growth is reasoning from the code, not a measured long-lived-process memory result. No claim that all unwitnessed callers always find a null context is justified after the disclosed failed stop.

### R4-5 — The sweep accepts an unchanged digest with a changed or absent part

**Location:** `tests/c2_2a/sweep.py:246–255`; `tests/c2_2a/test_sweep_verdict.py`.

The sibling-digest rule checks canonical identity only when the digest **differs from the plain digest**. If the faulted run keeps the old digest but changes its published part by losing information, recursive comparison accepts both independently:

```text
plain:   samples = [{p: .77, file_sha256: "aaa…"}], digest = canon(plain samples)
faulted: samples = [{p: .77, file_sha256: null}],   digest = the unchanged old digest
witness_only_loses(...) = []
```

It also returns `[]` for `map=null, map_sha256=the unchanged plain map digest`. Both independently written assertions fail. The new tests reject a *changed* wrong digest or a changed digest without its part, but omit the unchanged-digest cases.

This is a measured false green in the **oracle**, not a claim that one of the retained 3,394 production runs actually emitted those stale digests. The permitted ordinary gate checks canonical identity for its complete scenarios; the sweep's verdict does not call those scenario-specific witness checks. Every non-null sibling digest must bind to its own published part, regardless of equality with the plain value. The legitimate `c0adb12` allowance for hashing a lossy published part is reasonable; this implementation of that allowance is incomplete.

### R3 disposition, rebinding and the revised interfaces

The replay against an exported `dfbf286` is **4 of 4 RED in 2.72 seconds**, with the intended assertions: failed helper entry loses the pipeline forecast or returns raw calibration, and failed flag update publishes the affected PA part. The imported package path was independently verified to be the owned export's `src/bts`.

The literal `collect`/flag-helper injections have no target in this candidate. The ported tests instead exercise the replacement class: six PA `hold`/`confirm` invocation faults, landed records plus failed confirmations, and cache parity. Those pass. My independent combined landed-record/lost-confirmation/lost-error check also passes: pipeline PA, calibration PA and pick-input lists are withheld; calibrated probabilities survive. Bindings remain independently earned where their own append/confirmation succeeds.

The ledger's record/held-object identity, confirmations and independent listings substantially improve PA/pick completeness. An omitted record, failed confirmation or lost listing leaves a required part null; loss of error recording does not restore it. Binding confirmation plus exact `(p,y)` equality protects the checked sample inventory. Those results repair R3's specific failure shapes. They do not cure the separate hash, status and context counterexamples above, so the broader class requirement remains unmet.

I inspected the full f882411-to-candidate production diff. The deployed computation statements in the four existing modules remain, apart from explicit `projected`; rebinding changes their operand objects, as disclosed. The passing gate/unit checks support held-buffer parsing, original-path fallback, default text encoding/strict decoding/universal newlines, one normal read, unreadable-pick skip, genuine parse/decode/unpickle failures, unchanged forwarding/truncation and genuine partial-write semantics. A lost held-object record still returns its buffer. Independent post-loop listings are conservative when files change or an unreadable pick has no confirmed record.

Replacing the proposed unshipped collector/`witness`/cache-hash parameters with guarded context hooks is a reasonable implementation change in the signed note: the deployed public interfaces are restored, and no frame provenance exists during calibration. The internal context's failed-admission behavior and writer integration still fail the intended behavior contract. Preserving a deployed source statement verbatim is not sufficient when the newly rebound object can fail or supply provenance from a different run.

### Sweep coverage, totality and the never-reached classification

I recomputed the retained run's counts: **660 points, 3,394 faulted pairs, 61 successful plain checks**, 15,498 reached non-excluded pairs and **12,104 untested pairs**. Its 426 plain-reached points are each faulted in a plain scenario. There are 234 fault-only points and 275 additional untested pairs at those points. The supplied pair-disclosure script reproduces its report. My own full first-reach run at the reviewed commit reproduces 660 successful faults and 61 successful plain checks; I did not rerun all 3,394 pairs.

This is useful systematic evidence within its model: first execution of a line/call-site point, one `MemoryError`, sometimes layered over a scenario's existing fault. It is not complete for arbitrary combinations, later occurrences, nesting or persistent context state. Module/class import bodies are also outside its executable-function model. The fault model and those limits must not be inflated into the signed universal containment claim.

The AST totality analysis reproduces **30 guarded call sites, 0 shape violations**. That validates its syntactic guard test, not semantic totality. It omits the wrapper entry reached through C pickle and does not prove monotonic witness facts. R4-2's two failed omitted pairs directly refute the offered inference that any tested point stands for every other scenario reaching it. Consequently **the provided coverage plus totality argument is insufficient to rely on for the omitted pairs**. This does not require a full cross product by fiat; targeted missing pairs already establish failure.

The retained `_compare` checks the declared deployed surface exactly, with the reviewed additional original-text read allowance. `_reread_once` only normalizes matching file inventories, each count equal or one greater, total extra at most one; other counts remain differences. For the decoder-fault scenarios `_compare` still imposes their separate all-files fallback rule. This normalization is bounded, but it does not itself bind the extra read to a particular preparation failure.

Ignoring `built_at` and error-list differences is reasonable for a loss comparison only alongside truthful null/required-part admission and package/recipe checks. The digest exception has R4-5. A loss-only relation alone cannot certify completeness, schema-valid combinations or unchanged actual assignment truth.

I reproduced the never-reached classification exactly: **124 lines, 52 executed by supplied unit coverage, 72 in exception handlers whose try body has successful injected runs, 0 other**. This is an accurate classification under the script's definition. It does **not** measure entry into or fault sensitivity of each of those 72 handler lines; nor does ordinary unit coverage inject a fault at its 52 executed lines. The README now states the first limitation correctly.

The `health.alert.send_dm` repair is correct: it patches the import-bound name on each scenario, and the sequential late-DM regression passes. I independently checked the retained hashes: only `cutoff_advancing_clock` changes among the 61 pre-fix goldens; all current hashes match. I inspected analogous imported names along the tested pick path and found no additional demonstrated transport-spy blind spot. The scheduler's import-bound `select_pick` is used by its shadow path, outside these scenarios; it is not proof of shadow-path observation. Filtered stderr and fake transports retain the earlier scope limits.

### Ledger: sound recorded kills, overstated retirement explanation

All **94 unique active entries** have exactly one valid replacement anchor. Combining the full run and resume yields **93 attributable RED**, each with a named `FAILED` test, and Q7 surviving as the recorded equivalent. The original five survivors are disclosed; the four non-equivalent ones have resumed kills. I found no falsely classified RED in that retained evidence and did not mutate tracked source.

Q1–Q40 test substantive core contracts: ownership/sealing, held-object identity, inventory/confirmation, context reset, source/digest admission, text semantics and binding/fit facts. H1–H34 test hook placement/omission, transport and several boundary guards. Their strength is sensitivity to their specific replacements. Removing one condition or recording call is not a demonstration of every state/fault combination. In particular Q12/Q13 do not cover R4-3's cancellation, H26 does not cover the genuine post-assignment pair, and H21 does not cover a failed nested start.

**Q7 equivalence is reasonable for this ledger's reachable invariants:** removing the last-record clause cannot turn an extra, unconfirmed record into a complete ordered ledger. The manual extra-record test remains incomplete either way. This is an invariant argument, not a RED kill. **O14/H34 combined mutants are defensible** because the intact masking layer hides each isolated change; the combined kills establish the combination's sensitivity. H31 separately kills the backstop's removal. Do not count the combined kills as independent evidence for each layer alone.

The retirement explanation is inaccurate. All 71 old anchors are absent from their old files, but **seven still occur exactly in `serving_witness.py`**: C5 (slot), C7 (date), C8 (identity/value binding), C9 (exact probability), C13 (`increasing`), C13b (y thresholds), O8 (unavailable status). The README's claim that none occurs anywhere in code is false. These operations moved; their semantic rules did not disappear.

I re-anchored the six non-equivalent replacements in an owned copy of `src` and ran targeted current tests: **C5, C7, C8, C9, C13b and O8 all produce named assertion failures**. O8 requires the no-sklearn case; an initial insufficient-support-only check unsurprisingly passed and was not counted as a kill. C13 retains the earlier default-True equivalence. Thus current tests protect these moved rules, but the recorded retirement reason and accounting need their actual disposition rather than “code absent.” This documentary issue is not the primary BLOCK.

### Cost: conditional pass under the explicitly amended method

I read the actual local register rows at main `c6e32e5b55d19686d8eec8b8f1693640d9344014`, including Eric's review/warm-up rulings and the manager's delegated scope/load rulings. I independently recomputed every summary from **120 initial phase rows plus 20 rows in each load attempt**, checked complete unique pair identities, five repeats and macOS byte units. All summaries match exactly. Source is unchanged from `e36e038`; reviewed tests are unchanged from `41b0e8b`. The shared phase driver and whole-run bench are unchanged.

The phase adaptation is necessary and reasonable: direct load/save calls open a candidate witness; tail stand-ins populate the context rather than frame attrs. Assertions after measurement ensure the relevant witness work occurred. This measures the declared real phases with their declared stops/stand-ins, not an accidentally unwitnessed candidate. The other worktree's copied bench identity remains supplied evidence; I did not read it.

| final counted phase | maximum / median added RSS, decimal MB | maximum absolute control, MB |
|---|---:|---:|
| load, warm-up attempt 2 | 32.211 / 29.458 | 2.900 |
| cache | 9.617 / 8.258 | 2.851 |
| save | 0.836 / 0.147 | 3.817 |
| tail_off | 1.360 / 1.212 | 1.819 |
| tail_on | 16.269 / 12.468 | 4.162 |
| slate | 1.294 / 0.852 | 1.966 |

All counted controls are below the 125 MB aim and all final maxima below 250 MB. The first load series is correctly **UNRESOLVED** (control 516.145 MB). Attempt 1 is correctly **EXCEEDED**, not unresolved or silently discarded: maximum 292.159 MB with calm control 2.736 MB. Eric then explicitly changed the method after that failure. Warm-up commit `d106f54` is timestamped 18:25:33 EDT, before attempt 2's recorded 18:45:02 start; attempts are at least 30 minutes apart. The discarded baseline warm-up peaks at **2,281.259 MB**. Attempt 2 is the first calm-control attempt under that amended method and therefore final; no third attempt was taken.

**On its merits:** one fixed, recorded warm-up with an unchanged driver, preserved failure, precommitted amended attempt rule and finality avoids selecting a favorable candidate delta within that new series. Attempt 2's stable baselines and controls support a bounded pass under Eric's amended rule. It does not establish the proposed first-process mechanism: the initial series also had lows elsewhere, and the instrument is lifetime resident high-water RSS, sensitive to the disclosed pressure. The original pre-amendment rule failed; the amended PASS cannot be presented as an unchanged predeclared-method success or proof about cold-process memory on the box. The warm-up is a conditional measurement choice, not independently proven causal correction.

Whole-run timing was explicitly waived by the manager's delegated ruling, not measured at this candidate. Summing phase medians is not a mathematical proof of a whole-run median or maximum. The small measured phase increments, placement of witness work in those phases, and earlier whole-run evidence nevertheless provide a reasonable **bounded engineering argument under that ruling**, rather than a fresh whole-run acceptance measurement. As an additional check, the sum of maxima of measured paired elapsed increments is about 0.11 s using final load attempt 2 and all other phases (even including both tails); it is not an observed full-run maximum. Constructor work excluded from direct phase timing is represented in the real tail calls. These limits and the synthetic footprint/stand-ins remain material. Cost does not cure the demonstrated behavior defects and is not independently a reason for this BLOCK.

### Disposition of the five earlier author decisions

| decision | this revision |
|---|---|
| Move provenance and drop it before selection | Improved: no frame metadata during calibration, final attachment afterward; slate take/backstop precedes row extraction and selection. The passing copy-fault checks support that lifetime choice. |
| Genuine buffered-read OSError skips without retry | Retained through the unreadable-pick stand-in; checked genuine read/decode behavior passes. The artificial errno/message is not claimed identical to the original exception, which the deployed skip handler does not expose. |
| Witness v2, old-reader skip, park-drag allowlist | Unchanged producer boundary. Actual step-3 reader alignment/admission remains separate; the skipped old-reader compatibility test is not evidence of that admission. |
| Experiment parquet gains provenance attrs | **Changed.** Direct unwitnessed `run_pipeline` now records/attaches nothing, restoring the deployed output instead of adding those attrs. Source and unwitnessed-caller tests support the change; no experiment run/artifact compatibility measurement was performed. This guarantee presumes no open leaked/enclosing context. |
| Enabled empty predictions may have null status | Still bounded to the nonpersisted empty-frame case; now an unearned applied fact can also be null. Empty slates are refused and required null parts must be refused by the reader. That does not authorize R4-2's explicit false `applied=false` assertion. |

## Required changes

This is BLOCK, not SIGN WITH EDITS. No documentation-only unified diff establishes the missing behavior. These are conditions for any later implementation Eric chooses; **they authorize no repair, fifth round, merge or deployment in this cycle**.

1. Resolve the writer-entry containment gap or explicitly obtain a narrower behavior contract. A genuine forwarded-write failure and failure to enter an added provenance wrapper require separate evidence; the latter cannot be admitted by relabeling it as the former.
2. Earn calibration assignment truth independently of losing its capture hook. A failure after successful assignment must retain truthful applied state or withhold the unknown record, never assert failure before assignment. Add the two demonstrated existing-scenario pairs.
3. Make a save digest unavailable after interrupted hash/accounting state, including successful zero/short writes and cancellation between independent counters. Preserve the unchanged writes/returns without retry. Assert the actual persisted byte digest, not merely counter equality.
4. Prevent failed context admission from recording into another run; establish or explicitly bound nesting and leaked-context behavior. Update the totality/coverage argument around actual state consequences, with regression cases for the clean wrong-model witness.
5. Require every non-null samples/map digest to match its own non-null published part, including unchanged digest values. Correct the retirement explanation/accounting for moved rules and retain their demonstrated test sensitivity. Re-run evidence against any future authorized implementation; the current green sweep and ledger do not close the counterexamples.

The controlling ruling's consequence now applies: **stop 2a for this cycle**. W1.5 recertification, a new blind whole-range review and Eric's D7 are not performed or approved by this report.

## What was run

- Read the entire round-4 prompt before any outside-checkout action. No memory or external shared-history lookup. Confirmed exact detached HEAD, clean tracked tree and the 64-file round-3-to-round-4 diff. Read the complete production rewrite, signed implementation note, sweep, gate, relevant tests, ledger, retirements, bench/drivers, supplied raw evidence and local Git ruling rows.
- Ran the permitted suite with `TMPDIR`, matplotlib routing and pytest basetemp set before harness entry:

  ```text
  UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York OMP_NUM_THREADS=1 TMPDIR=/private/tmp/c2-2a-r4-review MPLCONFIGDIR=/private/tmp/c2-2a-r4-review/mpl uv run python -B -m pytest tests/c2_2a tests/test_slate.py tests/test_local_tier.py tests/model/test_calibrate.py tests/scripts/c1_r4a/test_provenance_stored.py -p no:cacheprovider --basetemp=/private/tmp/c2-2a-r4-review/pytest
  ```

  **282 passed, 1 skipped in 234.71 seconds, exit 0.** All 61 golden scenarios and the sequential health-DM regression pass; the skip is the disclosed old-reader/new-producer compatibility test.
- Exported `dfbf286` with local `git archive` into owned scratch and ran the supplied `r3_replay.py` with that export first on `PYTHONPATH`, explicit scratch temp/basetemp, and isolated collection: **4 failed in 2.72 seconds, exit 1**, reproducing both R3 contracts. Verified the imported package path points to the export.
- Ran the sweep at the reviewed commit, all 61 scenarios, eight workers, fresh interpreters, **without `--all-pairs-in`**:

  ```text
  TZ=America/New_York OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/uv-cache TMPDIR=/private/tmp/c2-2a-r4-review MPLCONFIGDIR=/private/tmp/c2-2a-r4-review/mpl uv run python -B -m tests.c2_2a.sweep --workers 8 --out /private/tmp/c2-2a-r4-review/sweep.jsonl
  ```

  **660 faults passed, 0 failed; all 61 plain checks equal their goldens; 124 never-reached lines; exit 0.** Separately ran the two omitted after-assignment pairs and the excluded writer-entry point with the sweep's own API. All injections fire and all three verdicts reject; the after-assignment pairs preserve the entire deployed surface while failing witness truth.
- Ran five independent scratch pytest checks: **4 failed, 1 passed in 2.62 seconds**. Two failures demonstrate wrong writer digests, two demonstrate the digest oracle false greens; the combined landed-record/failed-confirmation/lost-error check passes. Isolated collection produces harmless unknown-slow-mark warnings; these were assertion failures, not collection errors.
- Ran four real-function baseline/candidate comparisons for zero-write/hash interruption and failed nested admission. Both pairs preserve calibrated probabilities and actual cache bytes, while candidate real slate persistence records false model hashes with all required parts present and no errors. Baseline functions were extracted from actual f882411 source; only the shared fixture's heavy leaves are stand-ins. An initial standalone import-path setup error was corrected before collecting these results.
- In an owned copy of `src`, re-anchored and ran six relocated non-equivalent mutants with current tests: six named assertion failures. The preliminary O8 test used its unaffected insufficient-support case and passed; its distinguishing no-sklearn case was then run and failed. Copied source was restored; tracked source was never mutated.
- Recomputed all three cost summaries and checked 160 complete unique raw rows; checked active/retired anchors, full/resumed named ledger failures, all golden hashes, retained sweep pair inventories and omitted-pair identity. Re-ran the read-only totality, pair-disclosure and never-reached analyses and reproduced their counts. Did not regenerate baseline goldens, run the tracked-file-mutating ledger runner, rerun all 3,394 sweep pairs, or acquire new performance/box measurements.
- Inspected supplied fast-suite evidence: **4105 passed, 7 skipped, 70 deselected, 22 xfailed in 395.47 seconds**, exit 0; supplied c2_2a evidence: **229 passed in 227.36 seconds**, exit 0. These are author runs, not my broader-suite measurements.

All reviewer runtime/harness paths were routed under `/private/tmp/c2-2a-r4-review`, which was deleted at completion. No scratch-location exception occurred. Root policy copying was only the two files expressly permitted inside the golden harness; no separate root `data/`, `.env`, configuration or credential read occurred. No tracked edit, commit, push, escalation, network/box/SSH/gh operation, certification or deployment was performed.
