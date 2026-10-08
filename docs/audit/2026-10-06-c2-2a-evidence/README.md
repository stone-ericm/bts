# C2 step 2a evidence (design `docs/superpowers/specs/2026-10-06-c2-2a-serving-record-design.md`)

## Mutant ledger (§5.4)
- `mutants.json` — 59 mutants, one per §3 rule, plus the behaviour mutants B1 (attrs lost between the local tier and
  persistence) and D1–D3 (decision changes that leave slate and pick bytes as they were). Pinned at `ed5bebd` (58);
  C13b added after the first run (below).
- `mutant_runner.py` — applies each mutant, runs its named tests (golden scenarios selected with `-k`), prints each
  failing test, restores the file by hash.
- `mutants.out` — the runs:
  - first run at `ed5bebd`: 55 RED, 3 SURVIVED (C9, C11, C13);
  - C9 (a binding's `p` rounded) survived because every fixture probability had two decimals; C11 (`n_fit` from the
    inputs) survived because every fixture input yielded exactly one sample. New tests
    `test_a_binding_carries_the_exact_appended_float` and `test_n_fit_counts_samples_not_files` kill both (resume run);
  - C13 (`increasing` hard-coded True) is **equivalent**: the production constructor keeps sklearn's default
    `increasing=True`, so `increasing_` is True on any data (checked on decreasing data). C13b (the y thresholds taken
    from the X thresholds) replaces it as the map-field mutant and is RED.
- **Result: 58 of 58 attributable mutants RED; 1 equivalent.**

## Cost benchmark (§6.1)
- **Method as declared:** `tests/c2_2a/bench/bench.py` (one fresh process per run) and `cost/drive.py` (paired, alternating order, 5 repeats per case, plus a baseline-vs-baseline control). Footprints were declared before measurement in `cost/declared.json`:
  - big world: six parquets, 138.7 MB on disk and 1.76 GB decoded; a 9.30 MB cache (12 models plus the single); 156 picks, 438,963 bytes;
  - small world: real LightGBM training for the cold control.
- **Runs:** `cost/runs.jsonl`, 17:46–18:15 EDT 2026-10-06. Summary: `cost/summary.json`.

| case | paired Δtime median / max (s) | paired Δpeak RSS median / max (MB) | declared acceptance (≤2.0 s / ≤5.0 s / ≤250 MB) |
|---|---|---|---|
| cold_small | 0.01 / 0.44 | 7 / 10 | met |
| warm_off | 1.94 / 3.87 | 73 / **1,505** | **RSS max not met** |
| warm_on | 0.22 / 0.79 | −157 / −19 | met |
| warm_unavailable | 0.53 / 1.27 | −51 / 226 | met |
| control: baseline vs baseline | 0.25 / 0.90 | −329 / **304** | (noise floor) |

- **The RSS criterion is not resolvable by the declared method on this Mac:**
  - Whole-run peak RSS of the identical baseline code ranged 4.9–6.3 GB across 15 warm runs.
  - The five control pairs reached +304 MB, above the 250 MB limit, with no code difference at all.
  - The witness holds at most one parquet's bytes (about 25 MB) or the cache's (9.3 MB) during loading, and frees them before feature computation, where the process peaks at 5–6 GB.
- **Capture-phase probe** (`cost/load_probe.py`, fresh processes, imports outside the timed region): loading the six parquets the baseline way (path), the candidate way (read, sha256, `BytesIO`), and through `pyarrow.BufferReader`.
  - The candidate way took 1.12–1.17 s against 1.05–1.13 s.
  - Peak RSS was 2,645–2,682 MB against 2,632–2,663 MB.
  - The frames were identical to the path parse.
  - A first version of this probe imported `bts.model.predict` inside the timed region for the candidate only, and wrongly showed +0.7 s and +100 MB. That was import cost, corrected here.
- **Contamination, disclosed:** warm_off repeats 0–1 (17:46–17:49) overlapped the author's own test runs on this Mac. Their candidate totals (50.6 s twice) are the two largest time pairs. Time still meets the acceptance.
- **Status:** under the design's predeclared rule, warm_off's RSS maximum is not met, which stops 2a before code review. It goes to the owner side as a ruling on measurement validity (C2 index; register row to follow). The limits are not relaxed.

## Phase-level memory method (row C2-2a-cost-method; manager's ruling, declared before any run)
**Why:** the whole-run memory comparison above is declared **uninformative** for the 250 MB limit, in both directions. It neither passes nor fails 2a. The time results above stand as measured. **The 250 MB limit is unchanged.**

**Instrument:**
- Each run is one fresh process: `tests/c2_2a/bench/phases.py measure --phase P`, run from the side's repo root with the file copied unchanged.
- Imports and the input copy happen before the measured section. The process's peak RSS (`ru_maxrss`, bytes on macOS) is read at the phase's end.
- Every phase runs the side's real code. Only the named stop or stand-in differs, and it is identical on both sides.
- Inputs are the big world of `cost/declared.json`, plus a full slate of **351 rows**: the largest production slate of 2026 (box, read 2026-10-06; median 270).

**The phases:**
1. **load:** `run_pipeline`'s PA loading of the six parquets. `compute_all_features` is replaced by a stop that records the peak.
2. **cache:** `predict_local`'s cached-blend load (9.30 MB). `run_pipeline` is replaced by a stop at its entry. The stop is a `BaseException`, so it escapes `predict_local`'s `except Exception` and the phase driver catches it (corrected after code review r1; the earlier text said the handler returns None).
3. **save:** `save_blend` of the representative 9.30 MB blend to a fresh path. The blend is built before the measured section.
4. **tail_off:** `predict_local` after `run_pipeline`, which is replaced by an instant return of the 351-row predictions frame. On the candidate it carries the attrs the real `run_pipeline` sets. Calibration is off, so the candidate builds and attaches the witness. Runs through `predict_local`'s return.
5. **tail_on:** as tail_off, with `BTS_USE_CALIBRATION=1`. That adds the calibration PA read (`pa_2026`), the fit over the 156 picks, the application and the witness.
6. **slate:** `save_slate` of the 351-row frame: no attrs on the baseline (as at f882411), and the realistic witness from a tail_on-shaped build on the candidate. The witness is built before the measured section.

**Pairs:**
- 5 repeats per phase, baseline (the f882411 worktree) and candidate, alternating which goes first.
- A same-code control per phase: 5 baseline-vs-baseline pairs.

**Rules:**
- A phase counts only if its control's max |Δpeak| is comfortably under the limit (aim ≤ 125 MB). A control above 125 MB but below 250 MB does not count either: that phase also returns to the manager (`CONTROL_ABOVE_AIM`; the summarizer was aligned after code review r1).
- A phase whose control is ≥ 250 MB is **UNRESOLVED** and returns to the manager. It does not pass.
- A counted phase passes when its maximum paired Δpeak (candidate − baseline) is ≤ 250 MB.
- If any phase exceeds 250 MB, 2a stops as designed and the redesign goes to Eric.
- Phase times are reported, without acceptance weight; time was settled by the whole-run measurement.

### Phase-level results (run 2026-10-06 after `93f5478`; `cost/phases.jsonl`, `cost/phases_summary.json`)
| phase | paired Δpeak, max / median (MB) | control max \|Δ\| (MB) | verdict |
|---|---|---|---|
| load (six parquets) | 21.5 / 20.1 | 2.7 | PASS |
| cache (9.30 MB blend) | 9.2 / 9.0 | 2.1 | PASS |
| save (9.30 MB blend) | 0.4 / −0.0 | 3.2 | PASS |
| tail_off (witness build) | 1.3 / 0.9 | 1.2 | PASS |
| tail_on (calibration + witness) | 3.5 / 2.9 | 3.1 | PASS |
| slate (351 rows) | 2.0 / 0.8 | 1.5 | PASS |

- **Every control is within the aim** (≤ 125 MB); the largest is 3.2 MB.
- **The increments match the accounting:**
  - load holds one parquet's bytes while it is parsed (~20 MB);
  - cache holds the cache's bytes through unpickling (~9 MB);
  - the rest is within noise.
- **Phase times** (median, baseline → candidate, no acceptance weight):
  - load 1.51 → 1.58 s; cache 0.001 → 0.005 s; save 0.002 → 0.005 s;
  - tail_off 0.0002 → 0.005 s (the witness build); tail_on 0.436 → 0.442 s; slate 0.005 → 0.006 s.
- **Result: all six phases PASS; the 2a cost gate is met** under row C2-2a-cost-method's declared method.
- **Scaling note (reasoning, not measured on the box):** production loads ten parquets (231 MB). The load increment is the largest single file held at once (28 MB on the box), not the sum.

## Mutant ledger after code review r1
- **Runner (r1 F6):** RED only for pytest exit 1 with a `FAILED` line. Anything else is INCONCLUSIVE, and the runner exits 1 if any mutant is not RED.
- **Ledger:** 76 mutants (`mutants.json`).
  - 29 were re-anchored to the same rules in the revised code.
  - O12 was split into O14 (`save_slate` takes the witness before building rows) and O16 (`run_and_pick`'s backstop drop).
  - New: C18–C24, W2b, W11, W12, P13–P15, O11b, O13, O15, covering F1–F5's rules.
- **First run at `d7f8cec`:** 71 RED, 4 SURVIVED.
  - C13 is the recorded equivalent.
  - C19 and C22 (bindings completeness) were masked by the length check. A new test, `test_a_binding_that_raised_after_appending_is_still_withheld`, kills them.
  - O11 (taking the pipeline attrs) was masked by the `clear()` backstop. It became the combined mutant, provenance left on the frame. O11b removes the backstop alone, and the new `test_a_failed_pop_still_clears_the_pipeline_provenance` kills it.
  - O11's first combined replacement left an empty `try`. The strict runner reported it INCONCLUSIVE, and it was fixed and re-run.
- **Result: 75 of 75 attributable mutants RED; C13 equivalent.**

### Phase-level re-run at the revised candidate (after code review r1; `cost/phases_r2.jsonl`, `cost/phases_r2_summary.json`)
**Method:** the same declared method and driver, run at `340dcef`, the code reviewed in round 2. The baseline is the f882411 worktree.

| phase | paired Δpeak, max / median (MB) | control max \|Δ\| (MB) | verdict |
|---|---|---|---|
| load | 98.6 / 21.5 | 1.1 | PASS |
| cache | 11.0 / 10.1 | 0.8 | PASS |
| save | 3.5 / 0.0 | 3.2 | PASS |
| tail_off | 1.1 / 1.0 | 2.8 | PASS |
| tail_on | 5.2 / 3.8 | 3.3 | PASS |
| slate | 1.1 / 1.0 | 1.5 | PASS |

- **Load outlier:** load's maximum is a single pair. Its median matches the first run (one parquet's held bytes). A Codex review session was running tests on the same Mac during this re-run; the pair is reported as measured.
- **Verdict:** all six phases pass the unchanged 250 MB limit, and every control is within the 125 MB aim.

### Whole-run timing re-run at the revised candidate (`cost/runs_r2.jsonl`, `cost/summary_r2.json`)
**Method:** the same declared driver, run at `340dcef`'s code, 20:02–20:35 EDT, with the author running nothing else on the Mac.

| case | paired Δtime median / max (s) | acceptance ≤ 2.0 / ≤ 5.0 |
|---|---|---|
| cold_small | 0.28 / 1.24 | met |
| warm_off | −0.33 / 0.54 | met |
| warm_on | −0.48 / −0.15 | met |
| warm_unavailable | −0.50 / 0.34 | met |
| control: baseline vs baseline | 0.62 / 1.65 | (noise floor) |

- **Memory:** this run's whole-run RSS columns stay uninformative per the ruling. The control's maximum is +1,296 MB with identical code. Memory acceptance rests on the phase method above.

## Revision 3 (after code review r2; row C2-2a-review-r3, the third and final round)
**Scope (manager's ruling):** the R2-1 and R2-2 fixes only, each with the reviewer's failing check as a red-first test, plus mutants, gate scenarios, the ledger, a cost re-run of the touched phases and the fast suite.

**Fixes (`9ae58f0`):**
- **R2-1:** `_take_pipeline_provenance` allocates its parts inside its guard and returns None when they cannot be held. `predict_local` contains the invocation. If it fails or returns None, the frame's attrs are cleared before calibration (outside the except suite) and `run_pipeline provenance unavailable` is noted. The witness then shows model and inputs as null.
- **R2-2:** `_read_pa_parquet` takes a preallocated one-slot flag. A failed `collect` sets it False (a slot assignment, so no allocation), on both the buffered and the path-parse paths. `inputs_complete()` requires the flag, the list and the count, for the pipeline's inputs and for calibration's `pa_input`.

**Tests:**
- `tests/c2_2a/test_r2_counterexamples.py` holds 8 tests.
  - The 6 that reproduce the reviewer's two checks were red at `e22a905`. The allocation fault is injected by a `sys.settrace` hook at the matched source line, as the reviewer did.
  - The other two each measure one layer separately:
    - the helper's own containment (O17);
    - the record count when the flag write itself is lost (P23).
- **Gate (`3fa1596`):** three new fault scenarios: `fault_provenance_allocation`, `fault_provenance_take` and `fault_pa_append_landed`.
  - The goldens were regenerated in full at f882411 with the harness copied in unchanged: 60 scenarios, a clean tree, and the 57 earlier goldens byte-identical.
  - On the candidate, the gate passes 64 of 64.
  - Against `e22a905`'s `src`, all three new scenarios fail. Under the allocation fault, the old code makes no pick (MemoryError) where the baseline picks.

**Ledger:**
- **Spec:** 91 entries. O4, P13 and O15 were re-anchored to the new signatures (rules unchanged). There are 15 new mutants: O17–O23 and P16–P23.
- **Full run at `3203267`:** 89 RED; C13 is the recorded equivalent. P23 survived, because the flag withholds on every single-fault path. O18 and P18 were killed only by their gate scenario, because their `-k` filter deselected the named unit tests.
- **Resume at `3cbe848`:** O18 and P18 were re-run with exact gate node ids, and P23 against the new two-layer test. All three are RED.
- **Result: 90 of 90 attributable mutants RED; C13 equivalent.**

**Fast suite (`fast_suite.out`, at `5523443`):** 4057 passed, 7 skipped, 68 deselected (the slow gate scenarios), 22 xfailed, 0 failed.

### Touched-phase re-run (`cost/phases_r3.py`, `cost/phases_r3.jsonl`, `cost/phases_r3_summary.json`)
**Method:**
- The declared driver `cost/phases_drive.py` was imported unchanged and restricted to the three phases the fixes reach:
  - load: the PA reads and their flag;
  - tail_off and tail_on: `predict_local` after `run_pipeline`, the contained take, and on tail_on calibration's PA read and record check.
- cache stops at `run_pipeline`'s entry, save is `save_blend`, and slate is `save_slate`. None of them reaches the changed code.
- **Code measured:** the candidate's `src` at `5523443`, identical to `9ae58f0`'s production code.
- **Baseline:** the f882411 worktree, whose bench and harness files are byte-identical to the candidate's.
- **Run:** 21:33–21:35 EDT, with nothing else running on the Mac. Before it, the author's mutant ledgers and fast suite had finished. During it, the author only edited files in another worktree; no tests ran.

| phase | paired Δpeak, max / median (MB) | control max \|Δ\| (MB) | verdict |
|---|---|---|---|
| load (six parquets) | 20.5 / 19.6 | 3.2 | PASS |
| tail_off (witness build) | 0.6 / 0.4 | 1.0 | PASS |
| tail_on (calibration + witness) | 6.0 / 1.4 | 4.3 | PASS |

- **Every control is within the aim** (≤ 125 MB).
- **load** is one parquet's held bytes, as before. Round 2's single 98.6 MB pair was a concurrent-load outlier and did not recur.
- **Phase times** (median, baseline → candidate; no acceptance weight): load 1.56 → 1.62 s; tail_off 0.000 → 0.005 s; tail_on 0.440 → 0.453 s.
- **Not re-run:**
  - cache, save and slate: none reaches the changed code;
  - the whole-run timing: the fixes add only constant-size statements, and time was settled by the round-2 re-run.
- **Result: the touched phases PASS the unchanged 250 MB limit under row C2-2a-cost-method's method.**


## Revision 4 (after code review r3; row C2-2a-review-r4, the fourth and final round)
**Scope (Eric's ruling, "bts - A"):** one fourth and final code round with a repair that closes the class R3-1 and R3-2 belong to. Two requirements:
- completeness is earned, never assumed (PA inputs, pick inputs, bindings);
- an automated per-line fault sweep, checked against the deployed baseline.

The usual rules apply: counterexamples red first, the sweep, the ledger, the fast suite, then the review round. A non-SIGN stops 2a for this cycle.

**The class repair** (`6f8a3b7`; completed in `e36e038`; the design's "Code review r4 implementation note"):
- **Deployed statements unchanged.** `predict.py`, `calibrate.py`, `orchestrator.py` and `slate.py` differ from f882411 only by added lines (plus R10's explicit `projected` flag).
  - Every added statement is a one-statement `try: … except Exception: pass` hook beside them.
  - Every hook in `bts.serving_witness` handles its own designed failures. So a caller's guard is reached only by an unforeseen fault, and a second fault stops there.
- **One read, through rebinding.** The deployed statements read what the hook hands them:
  - `pd.read_parquet(parquet)` parses a buffer over the hashed bytes;
  - `json.loads(f.read_text())` reads through a stand-in whose `read_text` is the C-level read of a text reader over the held bytes, with `Path.read_text`'s encoding;
  - an unreadable pick's stand-in raises `OSError` from C, so the deployed handler skips it with no second read;
  - `cached_blend = load_blend(cache_path)` unpickles the held bytes;
  - `pickle.dump(blend, f)` writes through a forwarding hashing writer.
- **The witness lives in a context variable,** not in frame attrs.
  - Only `predict_local` makes it current: after its cache load, until it attaches `attrs["serving"]` and resets it.
  - Every other caller finds none open and runs exactly as deployed.
  - Nothing is attached before calibration, so r1 F1 and r2 R2-1 cannot recur.
- **Earned completeness.**
  - A file's record counts only once the statement that consumed it has run and the object consumed IS the held one. It is checked against an independent listing of the files consumed.
  - A binding counts only once its append returned, with values exactly the sample's.
  - The cache's hash counts only when the held loader was used, and the save digest only when the bytes hashed equal the bytes the file reports written.
  - No flag marks a part incomplete, so no combination of faults can make an incomplete part look complete.

**Red first:**
- **Your counterexamples** (`r3_replay.py`, output `r3_replay_dfbf286.out`). R3-1 and R3-2 replayed with your own injections against the revision-3 code at `dfbf286`:
  - the landed-then-raised `collect` with its error lost;
  - then `MemoryError` at `_mark_incomplete`'s `call` event (R3-1), or at its `ok[0] = False` line (R3-2).

  **4 of 4 RED**, reproducing your table:
  - R3-1, pipeline: no forecast. R3-1, calibration: raw `[0.86, 0.81, 0.74]` instead of `[0.65, 0.65, 0.65]`.
  - R3-2: both affected parts published with their hashes.

  Revision 4 has no `collect` and no flag helper, so these injections have no target in it. Their class is covered there by `tests/c2_2a/test_r3_counterexamples.py` (the hook-invocation faults and landed-plus-failed confirmation) and by the sweep.
- **The sweep at `dfbf286`** (`sweep/red_r3_dfbf286.jsonl`). It ran on the six scenarios that reach R3-1's and R2-2's code: `model_cached`, `model_cold`, `calibration_on`, `fault_pa_append_landed`, `fault_calibration_pa_buffer` and `fault_parquet_buffer`.
  - **116 of 326 points failed:** 107 changed the deployed output against the f882411 golden, and 9 published a witness that says more than the unfaulted run.
  - The failures include R3-1 at your injection point: the `call` of `_mark_incomplete` in `fault_pa_append_landed`.

**The per-line fault sweep** (`tests/c2_2a/sweep.py`; its docstring states the fault model):
- **Fault points:**
  - every executed line the candidate adds or changes relative to f882411 in the five pick-path files (`git diff -U0`);
  - every new function's `call`, once per call site.
- **Injection:** `MemoryError` from a `sys.settrace` hook, one point at a time, the first time it executes in the run.
- **Scenarios:** all 61 gate scenarios, in three classes:
  - plain: 21, with no injected fault and no genuine failure;
  - designed fault: 29;
  - genuine failure of a deployed operation: 11.

  `main()` refuses a scenario list that is not exactly the gate's.
- **Which runs:** each point is faulted in the first scenario that reaches it (plain first, cheapest first). Per the manager's ruling `C2-2a-sweep-scope`, it is also faulted in every one of the 11 model and calibration plain scenarios, and `day_dm`, that reaches it.
- **Isolation:** every run, recording or faulted, is in a fresh interpreter. No module state carries from one run into the next, and a fault cannot leave state behind.
- **Plain checks:** each scenario's traced run without a fault must equal its golden. Otherwise none of its points could be judged.
- **Excluded:**
  - lines whose bytecode is only `NOP` (`try:`, `pass`), where nothing can fail;
  - five computation points: the held unpickle (×2), the forwarded write (×2) and R10's flag. These are listed with the genuine-failure scenario that covers each, not swept.
- **Verdict:**
  - The whole compared surface equals the f882411 golden (`test_golden._compare`: predictions, selection, locks, transports, files, cache, slate rows). The one reviewed fallback, a faulted held pick read re-read once from its path (r1 F2), is normalised.
  - The slate's witness may only lose information: every part is null, or equal to the unfaulted candidate run's.
- **Two gaps found and fixed before this run:**
  - Earlier runs covered only 43 scenarios, under a stated reason (their own trace hooks) that was true for none of the 18 left out. Those 18 include `day_prediction_failure`, which reaches the prediction-failure stop.
  - Running each scenario in a fresh process exposed a gate blind spot, equal on both sides. `bts.health.alert` binds `send_dm` by name at import, and the harness spied only on `bts.dm`. So in one process, later scenarios' health DMs were never observed, and the f882411 golden for `cutoff_advancing_clock` lacked the late-delivery DM it sends. Fixed red-first (`a910c11`, `test_every_scenario_observes_its_own_health_dms`). Goldens regenerated at f882411 with the harness copied in unchanged: 60 of 61 identical, and that one gained its DM. Gate: 222 passed.
- **Runs:**
  - **Uncommitted revision-4 tree** (`sweep/diagnostic_r4_uncommitted.jsonl`; its header records HEAD `dfbf286`): 19 of 636 points failed, because the hooks were not yet total (a confirmation, a binding, the listing hooks). All were fixed before `6f8a3b7` was committed.
  - **`7b2c078`** (43 scenarios): 640 points, 0 failed.
  - **`e36e038`** (43 scenarios, `sweep/full_r4_e36e038.jsonl`): 656 points, 0 failed.
  - **All 61 scenarios, `9116776`** (`sweep/diagnostic_r4_9116776.jsonl`): 3,394 runs, every plain check equal to its golden, **19 failed, all one case**.
    - The case: in `calibration_insufficient_support`, a fault at the first pick's hold leaves the deployed read on the path (one read, as designed). That sample's binding publishes a null file hash, which is a loss. But `samples_sha256`, the hash of the published samples, necessarily changes, and the verdict required every published value to be null or equal to the unfaulted run's.
    - First-reach runs never saw it, because in `calibration_on` the first pick yields no sample.
    - The fix is in the verdict, not the code (`c0adb12`, red first in `tests/c2_2a/test_sweep_verdict.py`). A changed `samples_sha256` or `map_sha256` stands only as exactly the canonical digest of the faulted run's own published part, whose own loss is checked beside it. This is the identity the gate already checks on every scenario. A digest that matches neither, a part that gained or changed a value, and a digest without its part are all still refused.
    - That scenario re-swept with all its pairs: 334 runs, 0 failed.
  - **Final, `41b0e8b`** (`sweep/full_r4_41b0e8b.jsonl`; 18:46–19:57 EDT): all 61 scenarios (21 plain, 29 designed fault, 11 genuine failure), 660 points, 3,394 faulted runs, each in a fresh interpreter. **0 failed**, and all 61 plain checks equal their golden.
  - `src` is identical between `e36e038` and the reviewed commit, and `tests` between `41b0e8b` and the reviewed commit; the commits after `41b0e8b` change evidence only.
- **Pair coverage** (`sweep/pairs_disclosure.py` → `sweep/pairs_r4.txt`). A pair is a point together with one scenario that reaches it. Of 15,683 reached pairs, 185 are at the five computation points (covered by genuine-failure scenarios instead), leaving 15,498. **3,394 were faulted; 12,104 were not.**

  | class | reached pairs | faulted | not faulted: a plain scenario also reaches the point | not faulted: fault-only point | computation |
  |---|---|---|---|---|---|
  | plain (21) | 5,097 | 3,160 (3,156 in the 12 all-pairs scenarios, 4 at first reach in the other 9) | 1,937 | 0 | 77 |
  | designed fault (29) | 8,202 | 208 (first reach) | 7,779 | 215 | 81 |
  | genuine failure (11) | 2,199 | 26 (first reach) | 2,113 | 60 | 27 |

  - **Points a plain scenario reaches: 426, and every one is faulted in at least one plain scenario.** The script checks this and exits 1 otherwise; a copy with one point's plain runs dropped printed 425 of 426 and exited 1. So the 11,829 uncovered pairs at these points are pairs whose point was faulted, without an earlier fault, in a plain scenario.
  - **Fault-only points: 234.** No plain scenario reaches them; only a designed fault or a genuine failure does. Each is faulted once, in the first such scenario reaching it, and `sweep/pairs_r4.txt` lists each one with that scenario.
  - **The uncovered pairs only a designed fault or a genuine failure reaches: 275** (215 in designed-fault scenarios, 60 in genuine-failure scenarios). These are the pairs at fault-only points beyond the one faulted. For them, the totality argument below is the whole case.
- **The totality argument** (`sweep/totality_map.py` → `sweep/totality_map.txt`, generated from the AST). Why a point faulted in one scenario stands for the others that reach it:
  - **Every call site is guarded.** All 30 statements in the four deployed files that reach the witness sit in a `try` whose only handler is `except Exception` with a body that only passes or assigns a constant or a name. Each such `try` holds one statement, except `predict_local`'s first, which holds the witness module's import and the witness's construction; its handler sets all three names to `None`. The map exits 1 on any other shape, which was checked by removing one guard.
  - **The map lists, per hook, the statements outside any internal `try`.** Only these can carry a failure to the call-site guard; every other failure is handled inside the hook.
  - **A fault anywhere in a hook leaves the deployed statement beside it with the deployed value.** A rebinding hook that raises leaves the name unbound to the held object, so the deployed statement reads the path once, as deployed. A recording hook that raises loses only its own record, which earned completeness then withholds.
  - So the consequence of a hook fault does not depend on which scenario reached it. That is reasoning, checked per point in the first-reach run and per pair in the all-pairs scenarios, not a proof over the pairs not run.
- **Lines no unfaulted run reached** (`sweep/never_reached_r4.py` → `sweep/never_reached_r4.txt`): **124**, which is `e36e038`'s 125 less `orchestrator.py:142`, reached by one of the 18 scenarios added since. They were classified against the c2_2a unit tests' line coverage at `41b0e8b` (`sweep/unit_coverage_41b0e8b.json`, `COVERAGE_CORE=sysmon`, because the settrace-based tests would switch a settrace-based tracer off part-way):
  - 52 are executed by the unit tests;
  - 72 are lines of an exception handler whose innermost `try` body the sweep fault-injected, with every injected run equal to the golden. The classification does not show that each injected run entered that handler;
  - 0 are handlers whose `try` body was not injected, and 0 are other.

**Ledger** (`build_spec_r4.py` → `mutants.json`, 94 entries; `mutants_retired_r4.json`):
- **Kept: 20 entries** whose anchors and tests survive the repair (W1, W3–W12, P12, B1, S1, S2, D1–D3, O14, O16).
- **Retired: 71 entries** (the 69 others, plus W2 and W2b for the removed `collect`). None of their anchors occurs anywhere in the code (checked).
- **New:** Q1–Q40 for the witness core and H1–H34 for the hooks beside the deployed statements.
- The builder checks that every anchor is unique and every named test collects.
- **Full run at `c007271`** (`mutants_r4.out`): 89 RED, 5 SURVIVED.
  - **Q21** (the cache hash's identity check) and **Q27** (the withheld-digest error): their named tests never reached the distinguishing case. New unit tests pin each hook's contract.
  - **O14** (the take's `pop`) is masked by `save_slate`'s backstop drop. **H34** (the warning's guard) is masked by `save_slate`'s call-site guard. Each now runs combined with its masking layer, as r1's O11. The backstop alone is H31.
  - **Q7** (`confirm`'s last-record clause) is the recorded equivalent. It can differ only after a record is appended outside `hold`/`confirm`, and such a record can never be confirmed, so `complete()` is false either way.
- **Resume at `6c9f0d7`** (`mutants_r4_resume.out`): O14, Q21, Q27 and H34 RED; Q7 survives as recorded.
- **Result: 93 of 93 attributable mutants RED; Q7 equivalent.**

**Fast suite** (`fast_suite_r4.out`, at `a916853`, whose `src` and tests equal the reviewed commit's): **4105 passed, 7 skipped, 70 deselected (the slow gate tests), 22 xfailed, 0 failed.** The first run, at `acfe983`, gave 4098 passed and 69 deselected; the difference is `test_sweep_verdict.py`'s 7 tests and the slow `test_every_scenario_observes_its_own_health_dms`.

**The c2_2a suite with the golden gate** (`c2_2a_suite_r4.out`, at `a916853`): **229 passed**, including all 61 gate scenarios against the f882411 goldens (first run at `acfe983`: 221; since then `a910c11`'s test and `c0adb12`'s 7).

### Phase-level memory re-run (`cost/phases_r4.jsonl`, `cost/phases_r4_summary.json`)
**Method:** the declared driver `cost/phases_drive.py`, unchanged, on all six phases, because the repair rewired every one of them.
- The bench (`tests/c2_2a/bench/phases.py`, `bb51d81`; the copy in the f882411 worktree is byte-identical) was adapted to the context-scoped witness, on the candidate only:
  - load and save call `run_pipeline` and `save_blend` directly, so they open a witness as `predict_local` does (with none open, the hooks record nothing);
  - the tail stand-in leaves in the open witness what the real `run_pipeline` records.
- Each candidate phase then checks that its witness work happened. With the opening disabled, load fails loudly (checked).
- **Code measured:** the candidate's `src` at `e36e038`; the baseline is the f882411 worktree.

**First run** (`cost/phases_r4.jsonl`, `cost/phases_r4.log`, `cost/phases_r4_summary.json`; 16:41–16:43 EDT, with nothing else of the author's running):

| phase | paired Δpeak, max / median (MB) | control max \|Δ\| (MB) | verdict |
|---|---|---|---|
| load | 307.2 / 30.5 | **516.1** | **UNRESOLVED** |
| cache | 9.6 / 8.3 | 2.9 | PASS |
| save | 0.8 / 0.1 | 3.8 | PASS |
| tail_off | 1.4 / 1.2 | 1.8 | PASS |
| tail_on | 16.3 / 12.5 | 4.2 | PASS |
| slate | 1.3 / 0.9 | 2.0 | PASS |

- **The five counted phases pass** the unchanged 250 MB limit, with every control within the 125 MB aim.
- tail_on grew from r3's 6.0 MB maximum to 16.3 MB. The likely reason (reasoning, not measured separately): `current_pa = _sw.calibration_pa(current_pa)` keeps calibration's held PA buffer (`pa_2026`, 13.3 MB in the bench world) referenced by `predict_local` until it returns, so the buffer is still alive through the fit and the witness build. r3 freed it when its reader returned. It is still far under the limit.
- **load was UNRESOLVED.**
  - Its pairs were +307.2, +28.5, −182.4, +247.0 and +30.5 MB, and its same-code controls were +1.4, +183.7, +77.9, +516.1 and −0.3 MB.
  - The Mac was under heavy memory pressure (`cost/conditions_r4.txt`): about 20 GB in the compressor, about 1.5 GB free, and 1.16 M pages swapped out. The pressure came from the owner's own applications, and identical baseline runs peaked anywhere from 2,011 to 2,590 MB.
  - The two pairs taken while the controls were calm (+28.5 and +30.5) match r3's single held parquet (median +19.6, max +20.5).
  - Under the method, an UNRESOLVED phase returns to the manager. **The manager's ruling (row `C2-2a-cost-r4-load`):** re-run load with the declared driver unchanged, at most three attempts spaced apart, gated on the control only (within the 125 MB aim), every attempt reported here, never selecting on the candidate's delta. If no attempt has a calm control, load goes to the review UNRESOLVED with this disclosure.

**Load re-run attempts** (`cost/phases_r4_load.py`, the declared driver imported unchanged and restricted to load; `cost/load_attempt.sh`; `cost/load_r4_a<N>.*`, each with its conditions before and after):
- **Attempt 1** (18:15 EDT, `fbca22a`, no warm-up; memory free 46%):
  - Control calm (max |Δ| 2.7 MB), so it counts under row `C2-2a-cost-r4-load`.
  - Pairs: +292.2, +26.3, +32.1, +30.0 and +28.9 MB. **By the declared rule, EXCEEDED.**
  - Raw peaks: the candidate's five runs were 2,616–2,619 MB. The f882411 code's 15 runs (5 baseline, 10 control) were 2,587–2,591 MB on 14 of them. The one exception was the first process of the series, a baseline, at 2,324 MB. So the +292 pair is one low baseline reading, and the other four pairs (+26 to +32) are one held parquet, as in r3.
  - The driver always runs a baseline first in repeat 0, and the controls run third and fourth in each repeat, so a low reading of the series' first process would land on pair 0 and on no control.
  - **That cause is a hypothesis, not established.** The 16:41 series had low readings at many positions, not only the first: baseline r0 2,029; candidate r0 2,336; control_a r1 2,404; candidate r2 2,405; control_a r2 2,511; baseline r3 2,372; control_a r3 2,011; control_b r3 2,527. That fits sporadic under-reads under memory pressure at least as well.
- **Eric's ruling (row `C2-2a-cost-r4-warmup`, relayed by the manager): B, re-measure.**
  - The warm-up was added AFTER this failing attempt. It is one discarded f882411 process before the series, with its peak recorded and reported below. It was committed before any run (`d106f54`); the driver is otherwise unchanged, and the rule is unchanged.
  - At most two more attempts, spaced at least 30 minutes after 18:15. The first calm-control attempt under the warm-up is final, and EXCEEDED there stops 2a.
  - If neither attempt is calm, attempt 1 stands: EXCEEDED.
- **Attempt 2** (18:45 EDT, `d106f54` with the warm-up; memory free 52%; `cost/load_r4_a2.*`):
  - The discarded warm-up process peaked at **2,281 MB**: low, as the hypothesis predicted. That is consistent with it, not proof.
  - Then the f882411 code peaked at 2,587–2,590 MB on all 15 runs, and the candidate at 2,616–2,620 MB on all 5.
  - Pairs: +28.0, +29.5, +29.3, +31.6 and +32.2 MB. Controls: −2.5, +2.9, 0.0, −2.6 and +2.3 MB.
  - **Control calm, so this attempt is final: PASS** (max 32.2 MB, against the 250 MB limit).
  - Attempt 3 was not run.
- **Result: all six phases PASS** under the method as ruled (rows `C2-2a-cost-method`, `C2-2a-cost-r4-load`, `C2-2a-cost-r4-warmup`). Load's steady cost is one held parquet (+28 to +32 MB, against r3's +19.6 median).

### Whole-run timing re-run (`cost/runs_r4.jsonl`, `cost/summary_r4.json`)
**Method:** the declared driver `cost/drive.py` and bench `tests/c2_2a/bench/bench.py`, both unchanged. The bench's timing wrapper for the removed `_attach_serving_witness` simply finds nothing to wrap.

**Not re-run (row `C2-2a-cost-r4-load`, the manager's ruling):** the run was started at 16:43 EDT and stopped before its first measurement. Its warm runs peak at 5–6 GB, so on this Mac under the pressure recorded in `cost/conditions_r4.txt` they would swap. Its acceptance has no control gate, so swap noise could have produced a spurious failure. The files of the stopped run were removed.

**Why time is settled without it.** All of the witness's work lies inside the six phases measured above:
- load: the PA files' hold, hash and listing;
- cache: the cache's hold and hash;
- save: the hashing writer;
- tail_off and tail_on: calibration's PA and pick reads, the fit witness and the build;
- slate: the take and the envelope.

The phases' median elapsed deltas (candidate − baseline) are:

| phase | baseline → candidate (s) | Δ (s) |
|---|---|---|
| load | 1.585 → 1.631 | +0.046 |
| cache | 0.0012 → 0.0049 | +0.004 |
| save | 0.0018 → 0.0050 | +0.003 |
| tail_off | 0.0002 → 0.0049 | +0.005 |
| tail_on | 0.437 → 0.443 | +0.006 |
| slate | 0.0049 → 0.0055 | +0.001 |

That totals about +0.06 s, against the predeclared ≤ 2.0 s median and ≤ 5.0 s maximum. Round 3 skipped the whole-run timing on the same reasoning. The round-2 whole-run re-run met every case.
