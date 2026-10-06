# C2 (b) step 2a design: 4a's serving record and R10's explicit lineup state (production, pick path), revision d2

**What this is:** the design for C2 item (b) step 2a (register row C2-step2-split). It is production code in the pick path:
- reviewed until SIGN: at most 2 design rounds, then at most 2 code rounds, in C2;
- then Eric's D7;
- deployed before 2027 contest date 1 (by 2027-03-10).

**This is revision d2,** the last design round. It answers design review d1 (`docs/audit/2026-10-06-c2-2a-design-codex-d1.md`, REVISE, required changes 1–5).

**What it serves:** 4a's registration (`docs/sota_audit/2026-10-04-prereg-c1-calibration.md`, FROZEN) and its X-E1 freeze rules. Each 2027 capture must carry the serving recipe, model and inputs that produced it, recorded in the serving process, not reconstructed later. 4a's study code reads them in step 3.

**It resumes from:**
- the serving witness `e66b440` (reverted from main in `6a419a6`);
- 4a code review r2's item S (`docs/audit/2026-10-06-c1-r4a-codex-r2.md`, S1–S3);
- r2's R10 producer half.

`src/bts` on main is byte-identical to the deployed `f882411` when 2a starts.

**Not in 2a:**
- rank 3's pre-lock run archive (step 2b);
- 4a's study code (step 3);
- any change to what is predicted, selected, locked, deferred or delivered.

## Changes from d1, by required change
| d1 required change | Where |
|---|---|
| 1. an executable S3 recovery rule | §3.0: buffered-first, original-path fallback, contained provenance |
| 2. an S1 save-operation contract | §3.1: a hashing writer around the unchanged `save_blend`; no new full-buffer allocation; open/truncate order kept |
| 3. S2's sample and map contract | §3.2: ordered sample bindings, `n_fit`, canonical map state, the optional API |
| 4. the full behaviour gate | §5: pinned baseline, the compared surface, scenarios, transport, decision mutants |
| 5. cost acceptance and certification | §6: predeclared limits and their consequence; the certification rules |

## 1. The rule this design keeps
**Observability only.**
- **What it never changes:** what the pipeline computes, what the cache file holds, what is selected, locked, deferred or delivered, or when. Every witness failure is contained, and an available forecast is never lost or altered.
- **Genuine failures keep their exact current semantics:** a model load, parse, training, save or prediction error raises, or `predict_local` returns `None` through its existing handler, exactly as today.
- **When the two conflict, the forecast wins,** and the witness records incomplete provenance.

## 2. Schema: `bts_slate_v3`
- **The change:** the slate envelope gains `serving` (the witness, or null), and every row's `projected` is an explicit boolean (§4).
- **Why v3:** v2 has been deployed since 2026-10-06 (`f882411`) without `serving`. A new version gives 4a's reader an unambiguous boundary: a 2027 capture must be v3 with a witness, and a v2 file is refused for 2027 rather than read as "witness missing". Existing v2 readers are unaffected, since v2 files keep their tag.
- **Unchanged:** `ROW_COLUMNS` (only the `projected` value changes, from null to false, for posted lineups); last-write-wins per date; swallow-and-log persistence.

## 3. The witness `bts.serving_witness` (from `e66b440`, amended)
The parts are as in `e66b440`:
- `recipe`: file and serving-function hashes, plus a fingerprint;
- `env`: an allowlist of recipe-flag values; other `BTS_*` names only, never values;
- `packages`, `model`, `inputs`, `calibration`, `errors`, `schema`.

### 3.0 The recovery rule (S3)
Every new capture follows one pattern. A provenance failure never replaces the computation.

| Step | Buffered path | On a failure of the buffered path | Witness |
|---|---|---|---|
| **PA parquet** (`run_pipeline`, each file) | `raw = path.read_bytes()`, then `pd.read_parquet(io.BytesIO(raw))` | **any** exception in reading or parsing the buffer → the original `pd.read_parquet(path)`. A genuinely bad file raises the original error there, unchanged | `{file, bytes, sha256}`. sha256 is null with an error when hashing fails (the parse still uses `raw`) or when the original path was used ("parsed from path, not from the hashed buffer") |
| **cached blend** (`predict_local`) | `raw = cache_path.read_bytes()`, then `pickle.loads(raw)` | any exception → the original `load_blend(cache_path)`, whose errors propagate exactly as today | the cache's sha256, or null with an error |
| **trained blend** (`run_pipeline`) | §3.1 | — (the save itself is unchanged) | the hash of the bytes written, or null with an error |
| **calibration PA parquet** (`predict_local`) | as for the PA parquet | the original `pd.read_parquet(current_pa)` | as above |
| **calibration samples and map** | §3.2 | a witness failure never changes the samples, the fit or its application | null parts plus errors |

- **Containment boundary:** each hash, metadata extraction, list append, canonical serialization, `build()` call and `attrs` assignment runs inside its own `try/except Exception`. `MemoryError` is an `Exception`.
- **Errors go into a local list;** appending to it is itself wrapped.
- **No new exception escapes** `run_pipeline` or `predict_local` from the witness. The only exceptions that can escape are the original computation's.
- **No retraining because a cache hash failed,** and no substitution of raw for calibrated probabilities because calibration provenance failed.
- **`calibration.applied`** says whether the returned forecast is calibrated, set from the actual assignment, even when provenance is incomplete.

### 3.1 The model (S1)
- **The cache/train decision stays where it is** (`run_pipeline`'s `if cached_blend:`), with the existing `_model` pop. `run_pipeline` gains one keyword, `cached_blend_sha256=None`, which only `predict_local` passes.
- **The provenance comes from the branch actually taken:**
  - `{"source": "cache", "sha256": cached_blend_sha256}` when the cached object is used;
  - `{"source": "trained", "sha256": <written bytes' hash>}` when it trains and saves;
  - `{"source": "trained_unsaved", "sha256": null}` when it trains with no save path (the shadow model and backtests).

  So an empty or falsy cache trains and is witnessed as trained (S1's counterexample).
- **The save is unchanged:** `save_blend(to_save, path)` still makes the parent directory, opens `path` for writing (truncating it **before** serialization) and calls `pickle.dump`.
- **The only difference:** the file object is wrapped in a `HashingWriter`. Its `write(b)` updates a sha256 inside its own `try` and then forwards `b` unchanged to the real file, returning the real write's result.
- **So the hash is of exactly the bytes written,** and no full serialized buffer is allocated. A serialization or write error behaves exactly as today: same exception, same truncated file. A hash failure only nulls the provenance.
- **The witness is returned as `predictions.attrs["serving_model"]`.** Setting it is wrapped. `predict_local` never re-reads the cache path.

### 3.2 Calibration (S2)
**`_resolve_pick_outcomes(..., bindings=None)`:**
- **The loop is unchanged:** every `2*.json` in sorted order is read before its date or result is checked, with the same filters, the same skips and the same order.
- **One read per file:** each file is read once (`read_bytes`) and decoded with `bts.picks._read_text_bytes`, which equals `read_text` (measured equal by the d1 reviewer, including Unicode and CRLF).
- **Bindings:** when `bindings` is a list, the resolver appends, for every sample it appends, in the same order: `{"file", "file_sha256", "date", "slot": "pick" | "double_down", "batter_id", "p", "y"}`.
  - **A hashing failure** gives `file_sha256 = null` plus an error. **It never drops the sample.**
  - **The return value is unchanged:** a list of `(p, y)`.

**`fit_calibrator_from_picks(..., witness=None)`:**
- **The return value is unchanged:** the calibrator or `None`, with the same thresholds and fallbacks.
- **When `witness` is a dict, it is filled with:**
  - `n_fit` = the number of selected `(p, y)` samples;
  - `samples` = the bindings;
  - `samples_sha256` = the sha256 of their canonical JSON;
  - `status` ∈ {`fitted`, `insufficient_support`, `no_sklearn`};
  - for a fitted map, `map` = `{"X_thresholds": [...], "y_thresholds": [...], "increasing": bool, "out_of_bounds": "clip", "y_min": 0.0, "y_max": 1.0}` and `map_sha256`.
- **Canonical JSON** is `json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)`. Floats are written as Python's shortest round-trip `repr`. A non-finite value is an error, never a substitute.

**The serving record's `calibration`:**
- **Fields:** `{enabled, applied, status, pa_input, n_fit, samples, samples_sha256, map, map_sha256, errors}`.
- **`status`** ∈ {`off`, `no_pa_file`, `insufficient_support`, `no_sklearn`, `failed` (the existing outer handler's genuine calibration failure), `applied`}.
- **Valid combinations:**

| status | enabled | applied | required parts |
|---|---|---|---|
| `off` | false | false | none |
| `no_pa_file` | true | false | none |
| `insufficient_support` | true | false | `pa_input`, `n_fit`, `samples` |
| `no_sklearn` | true | false | none |
| `failed` | true | false | errors |
| `applied` | true | true | every part |

  Missing required parts are incomplete provenance, which step 3's reader refuses.

### 3.3 Limits kept from r2
- The witness does not retain every live schedule or lineup response, so it is not complete replay evidence for every prediction input.
- The recipe hashes files on disk when the witness is built. The loaded code and the files can differ only between a deploy's checkout and its restart.

## 4. R10: explicit lineup state
- `_fetch_game_slots` sets `slot["projected"] = is_projected` for every slot: true for the prior-game fallback, false for a posted lineup.
- **Consumers checked (2026-10-06):** selection, gating and fallback read `projected_lineup` (from the `PROJECTED` flag text), never this key. The flag is appended only when the value is true. The key reaches only the predictions frame and the slate rows.
- **Test:** the real `_fetch_game_slots` on mocked schedule and feed responses with one posted and one projected lineup goes through `predict` and `save_slate`, and the persisted rows are `true` and `false`.

## 5. Gate (red first, then green)
### 5.1 Baseline and environment (pinned)
- **The baseline commit** is `f882411` (deployed).
- **The generator:** `tests/c2_2a/golden/generate.py`, run from a worktree at `f882411`. It writes the golden files, their sha256s, the generator's own sha256 and its command into a committed manifest.
- **Fixed environment:**
  - `TZ=America/New_York`;
  - `OMP_NUM_THREADS=1`;
  - the `uv.lock` package versions;
  - fixed wall clocks (the scheduler's `_now_et`, the pick and decision writers' clocks);
  - fixed fake transports.
- **Training leaves are stubbed** to a fixed tiny model on both sides, so LightGBM thread nondeterminism cannot make the comparison flaky. **One separate test** trains a real tiny LightGBM blend through the new save path and checks the written bytes equal `pickle.dumps`, a round-trip load, and identical probabilities.

### 5.2 What is compared
For each scenario, the new code must reproduce the golden outputs exactly:
- **the predictions:** ordered row identities `(batter_id, game_pk)`, and every value selection reads (`p_game_hit`, `p_game_blend`, `p_hit_vs_starter`, `p_hit_vs_reliever`, `est_pas`, `starter_pas`, `reliever_pas`, `flags`, the derived `projected_lineup`);
- **the cache:** its bytes after the run, and after each injected save or serialization error;
- **the `SelectionResult`:** action, primary, double-down partner and runner-up;
- **the scheduler's outcomes:**
  - eligibility and lock/defer results (`plan_fallback_action`);
  - the fake transport's call log (order, recipient, text);
  - the pick file and `decision.json` bytes;
  - delivery at cutoff − 1 s and refusal at exactly the cutoff.

**The only intended differences** are the slate's schema tag, `serving`, and posted rows' `projected=false`.

### 5.3 Scenarios
- **Model:** cached; cold train; an empty cached dict.
- **Calibration:** off; on; on but unavailable (insufficient support).
- **Prediction failure:** the tier returns `None`.
- **Day shape:**
  - an MDP skip;
  - private mode;
  - the fallback path with a cached pick;
  - a scheduler restart (a state reload mid-day).
- **Injected faults, one per boundary in §3.0:**
  - parquet buffer allocation (`MemoryError`), parquet hash, cache buffer, cache hash;
  - the `HashingWriter` hash, a save serialization error, a calibration buffer;
  - a sample-binding append, the map extraction, `build()`, the `attrs` assignment.

  Each must give the baseline forecast, calibration result and selected behaviour, with the provenance part null and its error recorded.

### 5.4 Transport and mutants
- **The `attrs` transport test** keeps `predict_local`, `run_cascade`, `run_and_pick` and `save_slate` real, and patches only lower computation and I/O leaves. It reads the written slate.
- **Behaviour mutants** (each must fail on an assertion):
  - an `attrs`-loss mutant between the local tier and persistence;
  - **a decision mutant** that changes a lock, defer or delivery result while leaving slate and pick bytes unchanged.
- **The ledger** is pinned before review, with one mutant per §3 rule: the buffered and fallback branches, each containment, the `HashingWriter` forward and hash, the provenance branch, each sample field and order, `n_fit`, map canonicalisation, the status table, the explicit `False`, the v3 tag.

### 5.5 What equality establishes
Golden equality is evidence for the tested deterministic scenarios only. The real-model test, the injected faults and the cost measurement (§6) cover the rest. None of this is a claim about the installed box.

## 6. Production safety and certification
### 6.1 Cost: predeclared acceptance
- **What is counted:** the complete synchronous witness cost:
  - the PA and calibration parquet buffers and their hashes;
  - holding the cached blend's bytes through `pickle.loads`;
  - the `HashingWriter`;
  - the recipe scan (every `.py` under `bts/model`, `bts/features`, `bts/data`, plus `inspect.getsource` of the serving functions);
  - the package-version queries;
  - the sample bindings;
  - the `attrs` transport;
  - the envelope serialization.
- **The method:**
  - paired baseline (`f882411`) and candidate runs of `run_and_pick` in fresh processes on the Mac, with identical synthetic inputs at realistic sizes: six PA parquets of about 26 MB compressed each, and their decoded frames;
  - the cases: a cold train (real training, small), a warm cache, calibration off, on and unavailable;
  - the phases are timed separately (parquet load, cache load, save, calibration, witness, slate, selection), with buffer lifetimes noted;
  - five repeats per case, reporting the median and the maximum of the incremental time and incremental peak RSS.
- **The acceptance, fixed now:** for every case, the incremental wall time is ≤ 2.0 s median and ≤ 5.0 s maximum, and the incremental peak RSS is ≤ 250 MB.
- **The relation to production headroom (reasoning):**
  - the scheduler's cascade budget is 12 minutes (`cascade_budget_min`, CLAUDE.md), so 5 s is under 1%;
  - the box showed 14 GB available on 2026-10-06 15:16, so 250 MB is under 2%.
- **The consequence:** exceeding either limit stops 2a before code review, to revise the capture (for example chunked hashing of the path read). Mac numbers are reported with their method, and are never stated as box behaviour or deadline invariance.

### 6.2 Certification
- **Which certificates:** the affected W1.5 current-defence certificates are identified from the final diff and call graph at the final candidate. Do not assume they are the seven re-run on 10/06.
- **How:** they are re-run with the frozen tooling `f453283`.
- **Acceptance** requires both the runner's `accepted` and a recorded reviewer `accept`, with the evidence retained.
- **What Phase 1 cannot do:** it certifies no missing-delivery absence (ruling 10).
- **What certification does not replace:** the behaviour and cost gates, the whole-range deploy-gating review (a fresh session, Part 1 blind) and Eric's D7.

**Deploy:** in a sleep window, with `entry_intent = "research"` unchanged. Nothing here changes activation.

## 7. What step 3's reader must check
These are named here only so that the producer and the reader agree; they are reviewed with step 3:
- refuse a 2027 capture that is not v3 or has no witness;
- require the full recipe, flag, package and component coverage and §3.2's calibration combinations;
- treat any witness error or null required part as incomplete provenance;
- permit a genuine registered unavailable-calibrator fallback (`no_pa_file`, `insufficient_support`, `no_sklearn`).
