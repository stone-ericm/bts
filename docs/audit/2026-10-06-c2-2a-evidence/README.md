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
