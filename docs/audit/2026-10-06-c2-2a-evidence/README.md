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
