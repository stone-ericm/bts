# C2 (b) step 2a design: 4a's serving record and R10's explicit lineup state (production, pick path)

**What this is:** the design for C2 item (b) step 2a (register row C2-step2-split). It is production code in the pick path: reviewed until SIGN (at most 2 design rounds, then at most 2 code rounds, in C2), then Eric's D7, and deployed before 2027 contest date 1 (by 2027-03-10).

**What it serves:** 4a's registration (`docs/sota_audit/2026-10-04-prereg-c1-calibration.md`, FROZEN) and its X-E1 freeze rules. Each 2027 capture must carry the serving recipe, model and inputs that produced it, recorded in the serving process, not reconstructed later. 4a's study code reads them.

**It resumes from:**
- the serving witness `e66b440` (reverted from main in `6a419a6` with 4a's C1 deferral);
- 4a code review r2's item S (`docs/audit/2026-10-06-c1-r4a-codex-r2.md`), findings S1–S3 with their "smallest fix" text;
- r2's R10 producer half: production must persist `projected=false` for a posted lineup.

**Not in 2a:**
- rank 3's pre-lock run archive (step 2b);
- 4a's study code (step 3);
- any change to what is predicted, selected or delivered.

## 1. The rule this design keeps
**Observability only.**
- No prediction, selection, delivery, decision or cache behaviour changes. Every witness failure is contained, and an available forecast is never lost.
- Model-loading and prediction failures keep exactly their current semantics.
- The witness costs work in the pick path (one extra hash per input buffer); §6 measures it.

## 2. Schema: `bts_slate_v3`
- **The change:** the slate envelope gains `serving` (the witness, or null), and every row's `projected` is an explicit boolean (§4).
- **Why v3, not an optional field on v2:** v2 has been deployed since 2026-10-06 (`f882411`) without `serving`. A new version gives 4a's reader an unambiguous boundary: a 2027 capture must be v3 and carry a witness, and a v2 file is refused for 2027 rather than read as "witness missing". v2 files stay readable to every existing reader.
- **Unchanged:** `ROW_COLUMNS` (the `projected` value changes from null to false for posted lineups); the last-write-wins file per date; the swallow-and-log persistence rule.

## 3. The witness `bts.serving_witness` (from `e66b440`, amended)
The parts are as in `e66b440`: `recipe` (file and serving-function hashes, plus a fingerprint), `env` (an allowlist of recipe-flag values; other `BTS_*` names only, never values), `packages`, `model`, `inputs`, `calibration`, `errors`, `schema`. The amendments:

**S1, the model witness follows the actual decision.**
- **The decision moves into `run_pipeline`.** It receives the cached blend's object together with the sha256 of the exact bytes `predict_local` read and unpickled (one read, as in `e66b440`). `run_pipeline` alone decides cache versus train (its existing `if cached_blend:`), so an empty or falsy cached object trains, and the witness says `trained`.
- **Training serializes once.** `pickle.dumps(to_save)` produces the bytes, they are hashed, and those same bytes are written to the cache path. `save_blend`'s `pickle.dump` writes the same bytes for the same protocol, which a test pins. The witness records the sha256 of that buffer, never a re-read of the path (the race in S1). The in-memory blend used for prediction is the object serialized.
- **The witness comes back with the predictions:** `predictions.attrs["serving_model"] = {"source": "cache" | "trained", "file", "sha256"}`. `predict_local` no longer guesses.

**S2, applied calibration's inputs are witnessed.** When `BTS_USE_CALIBRATION=1` and a calibrator is fitted:
- the current-season PA parquet is read once (`read_bytes`, hashed, parsed from that buffer), in place of `pd.read_parquet(path)`;
- `fit_calibrator_from_picks` takes an optional `witness` list. Each pick file it consumes is read once (`read_bytes`) and decoded with `bts.picks._read_text_bytes` (exactly as `read_text` would decode it), and `(file, bytes, sha256)` is appended;
- the fitted isotonic map's state (its threshold arrays, as canonical JSON) is hashed;
- the witness records `calibration = {enabled, applied, pa_input, pick_inputs, map_sha256, n_fit}`.

  **Supported combinations:** `enabled=false` → `applied=false` and no inputs; `enabled=true` with an unavailable calibrator → `applied=false`; `applied=true` → every input and the map hash present. Any other combination is an error in the witness, and the 4a reader refuses it.

**S3, every added step is contained.**
- Each added hashing or witness step (the parquet buffers, the training buffer hash, the calibration buffers, the map hash, `build`) runs in its own `try/except Exception` and records `null` plus its error. `MemoryError` is an `Exception`, which is r2's counterexample.
- **Error placement:** an ancillary failure in the base pipeline is recorded in the witness's `errors`; it is never raised out of `run_pipeline` or `predict_local`.
- **Unchanged failure semantics:** model loading, training and prediction keep their exact current behaviour (raise, or `None` from `predict_local`'s existing handler).
- **Tests:** a post-prediction hash failure, a calibration-buffer hash failure, and a `build` failure, each with an available forecast returned unchanged.

## 4. R10: explicit lineup state
- `_fetch_game_slots` sets `slot["projected"] = is_projected` for **every** slot (true for the prior-game fallback, false for a posted lineup), where it used to omit the key for posted lineups.
- **Consumers checked (2026-10-06):** selection, gating and fallback read `projected_lineup` (from the `PROJECTED` flag text), never this key. The flag is still appended only when true. The key reaches only the predictions frame and the slate rows.
- **Test:** a real `_fetch_game_slots` call on mocked schedule and feed responses with one posted and one projected lineup goes through `predict` and `save_slate`, and the persisted rows are `true` / `false`. This is the producer-to-reader path that r2 R10 found masked by a hand-written fixture.

## 5. Gate (red first, then green)
1. **On/off equivalence** (golden outputs from the pre-change code):
   - **The golden files:** a committed script, run once in a worktree at the deployed `f882411` (main's `src/bts` is byte-identical to it when 2a starts), writes the predictions frame's values (`p_game_hit`, `p_game_blend`, ranks, flags) plus the pick file and decision bytes for fixed fixtures. The test data is committed with its sha256 and the generator command.
   - **Training is stubbed** to a fixed tiny model on both sides, so LightGBM thread nondeterminism cannot make the comparison flaky.
   - **The new code must reproduce them exactly.** The only intended differences are the slate's schema, `serving`, and posted rows' `projected=false`.
   - **Paths covered:** the cached model, the train path, calibration off, calibration on, and a prediction failure.
   - **The real chain:** the witness must reach the persisted slate through the scheduler's actual path from `predict_local` to `save_slate`, not only a `predict_local` unit. It travels on `predictions.attrs`, which a copy or concat could drop. A test drives the scheduler's prediction step with leaves patched and reads the written slate.
2. **S1 counterexamples:**
   - an empty cached dict (the witness says `trained`, with the buffer's hash, which differs from the empty cache's);
   - the cache path replaced after save (the witness keeps the hash of the buffer used);
   - `save_blend`'s byte-equality with `pickle.dumps` for the protocol in use.
3. **S2 counterexamples:**
   - r2's case, where changing only the historical pick probabilities changes the applied forecast: the witness's pick input hashes must differ;
   - unsupported enabled/applied combinations are refused by the 4a-side validator's test, which is shipped in step 3;
   - calibration reads each pick file once.
4. **S3 counterexamples:** each added hash failure (including `MemoryError`) returns the same forecast with a null part and its error.
5. **R10:** the producer-to-slate fixture in §4.
6. **The mutation ledger** is pinned before review, one mutant per rule: the decision source, buffer hashing, each containment, each calibration input, the map hash, the explicit `False`, the v3 tag.

## 6. Production safety and certification
- **Cost:**
  - one `read_bytes` plus sha256 per PA parquet (about 26 MB each, 5–6 files), which replaces the path read;
  - one extra hash of the training buffer;
  - calibration buffers when enabled.
- **Measured before review:** wall time and peak RSS of `predict_local` before and after on the Mac, with real-sized synthetic parquets. The numbers are reported with their method; the box is not used.
- **Re-certification:** the change touches `orchestrator.predict_local`, `model.predict.run_pipeline` and `_fetch_game_slots`, `model.calibrate` and `slate`. Before D7, the affected W1.5 current-defence certificates are identified from the final diff and call graph, then re-run with the frozen tooling (`f453283`), as on 10/06.
- **Deploy:** a whole-range deploy-gating review (a fresh session, Part 1 blind), then D7, deployed in a sleep window with `entry_intent = "research"` unchanged. Nothing here changes activation.

## 7. What 4a's study code (step 3) must then check
4a's reader requirements are named here only so that the producer and reader agree. They are reviewed with step 3:
- refuse a 2027 capture that is not v3 or has no witness;
- require the full recipe, flag, package and component coverage and the calibration-state combinations (r2 R2);
- treat a witness error as incomplete provenance.
