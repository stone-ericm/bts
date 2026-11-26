## Verdict
**BLOCK.**
Reviewed-commit: dc58a32dcc9968fe268b04a19dde65fa2ea5c096

**Disclosure, before review findings:** My first tool batch read this prompt, searched `/Users/eric/.codex/memories/MEMORY.md` for `C2|framing|rank-4b|c1-r4b`, and listed `project_bts*.md` / `reference_bts*.md` filenames under `/Users/eric/projects/claude-shared/memory`. The two outside-checkout commands ran before I had read the prompt's prohibition. They returned registry excerpts and shared-context filenames; I did not open the listed shared files or any rollout summaries, and did not use those excerpts as evidence for this verdict. This review is therefore not strictly blind. Subsequent substantive evidence comes from this checkout and synthetic reproductions. One synthetic run's dependency import also created a Matplotlib fallback cache at `/var/folders/pb/9cjvyndj5plf63cdndwlj1gc0000gn/T/matplotlib-xy66uhrv`, outside the requested scratch prefix; subsequent probes set `MPLCONFIGDIR` inside the owned scratch directory. This incidental cache write is a further scope deviation, not a research input.

HEAD matches the requested detached commit. The tracked tree was clean at entry. The permitted suite passes **53 tests**, but it does not establish the registered execution or aggregation boundary. The formula and feature-list rewrites are correct; the input closure, three-seed disposition, one-claim scope, suspended-game handling, and several registration statements are not sufficient to admit a run.

This is round 1 of at most 2. No run, exposure admission, production change, or deployment is approved by this report.

## Findings

### B1 — The nine-parquet input closure is false; feature computation reads unpinned files, including 2026 raw feeds

**Source:** `screen.py:200–208,239`; `src/bts/features/compute.py:45–97,453–494,630–634`; `src/bts/features/park_drag.py:32–61,142–149,208–223`; registration §3, lines 33–36.

`load_inputs` correctly hashes each PA file once and parses those same bytes. Its filename whitelist excludes `pa_2026.parquet`. That does not close the complete run's inputs. `compute_all_features` unconditionally calls `_build_probable_pitcher_lookup()`, which:

- reads the unpinned relative file `data/models/probable_pitcher_lookup.json` when present;
- scans every season directory under relative `data/raw`, with no 2017–2025 restriction;
- reads uncached JSON feeds, including files in `data/raw/2026`, and can write the updated lookup cache.

The lookup supplies probable starters and pitching-team IDs used to compute `opp_bullpen_hr_30g`, one of the **16 baseline features**. The nine PA hashes and their exposure digest bind none of this. Ancillary paths are relative to the process's working directory, independently of `--data-dir`.

**Measured synthetic reproduction:** On the same 48 synthetic PA rows, real `compute_all_features` produced zero non-null bullpen values with an empty lookup, and 28 non-null values, all `1.0`, with a synthetic populated lookup. Both framing self-checks returned `identical=true`. Separately, the actual lookup helper read a scratch `synthetic_raw/2026/999000.json`, returned that game's lookup, and wrote a scratch cache. No real cache, raw feed, or PA data was read.

`compute_all_features` also reaches an unpinned park-drag CSV, whose path may come from `BTS_PARK_DRAG_TABLE`. That column is excluded from all three registered blends, so I have **not** demonstrated a ranking effect from park drag. It nevertheless contradicts the complete input-read inventory and permits additional input reads, including whatever seasons the table contains.

The demonstrated 2026 scan is a prohibited **read**, not proof that a 2026 outcome changes a historical feature: a distinct 2026 game ID normally will not match a historical PA. The demonstrated historical ranking-input dependency is the probable-pitcher lookup. The framing self-check cannot detect either issue.

### B2 — `aggregate` can issue a stage-one disposition from one seed, duplicate runs, or unregistered seeds

**Source:** `screen.py:136–154,284–294,304–309`; registration §5 and §6, especially line 86; `test_screen.py:146–148,200–211`.

`disposition` accepts any nonempty number of summaries. It uses `sqrt(n)` and a majority of `n`, rather than enforcing the registered three seeds and two-of-three boundary. `aggregate` trusts the supplied JSON seed fields and summary booleans, checks neither seed membership nor uniqueness, and requires no manifest or claim.

**Measured**, using synthetic `results.json` files with A deltas `(0.006,0.005)` and `passed=true`, and B deltas `(-0.004,-0.003)` and `passed=false`:

| Input to the actual aggregate | Returned seeds | A | B |
| --- | --- | --- | --- |
| One directory | `[2273360]` | positive | negative |
| Same directory three times | `[2273360,2273360,2273360]` | positive | negative |
| Two canonical seeds plus 42 | `[2273360,260991262,42]` | positive | negative |
| Four directories, including 42 | four seeds | positive | negative |

These directories had no `CLAIM.json` or `manifest.json`. The one-seed result directly contradicts the express rule that seed 1 alone cannot receive a stage-one disposition. Repeating one favorable seed manufactures `sd=0`, `t=inf`, and a favorable majority. Accepting arbitrary numbers also creates an unregistered stage-two decision rule.

Beyond sample size, results from different reviewed commits or input inventories can be mixed: aggregate never inspects their manifests or accepted identities. This is a consumption-boundary defect; green tests of the three happy-path JSON files do not cover it.

### B3 — One claim is enforced per caller-chosen root, not per registered seed; later-seed authorization is procedural only

**Source:** `screen.py:211–230,301–303`; `scripts/audit/c1/admission.py:267–299`; registration §4.1 and §6.

`--out-root` is arbitrary. Claim inspection and locking apply only beneath `out_root/seed_<seed>`. Changing the root bypasses the earlier claim without any owner invalidation. Neither `admission_gate` nor the shared launcher's generic command admission binds the framing output root.

**Measured synthetic reproduction:** The actual `run` function returned `0` twice for seed `2273360`, once under each of two scratch roots, leaving two durable claims and two `results.json` files. Admission, inputs, features, and the walk-forward were stubbed using the same seams as the author's end-to-end tests; claim handling, manifest persistence, scorecards, and result persistence were real. This does **not** demonstrate admission at the current commit, which correctly lacks X-35/admission. Source inspection shows that an admitted record would not fix the alternate-root gap: admission does not receive or validate `out_root`.

The code also accepts any of the three seed IDs immediately. It does not check that seed 1 finished, that its cost/projection was reported, or that Eric released seeds 2–3. The shared launcher does not implement that framing-specific decision either. The registration does impose that obligation on the lead; it should not be described as an enforced driver gate. An exact, reviewed launch procedure or wrapper must make its enforcement and provenance concrete before admission.

### B4 — The registered backtest uses the wrong suspended-game outcomes, and date shift alone does not prove event-time availability

**Source:** `screen.py:239,258–259,271–279`; `src/bts/simulate/backtest_blend.py:620–648,672–698`; `src/bts/data/build.py:26–52,156–161,189–196,317`; project scoring instructions supplied with this task.

The estimated-PA path builds `actual_hit` as `max(is_hit)` over **all** rows for a batter/game. Neither the driver nor that helper filters `is_resumed_portion`. The scorecard then treats those labels as BTS outcomes. The project's stated scoring contract excludes resumed-portion PAs while retaining them for training/features.

**Measured:** A synthetic batter/game with an original miss and a resumed-only hit returned `actual_hit=1`, `source_n_pas=2` from the actual `_starter_matchup_representative_rows`. Under `filter_out_resumed_portion`, its original-game contest outcome is a miss. The probability mode name `estimated_pa` does not correct its label basis. No estimate of the incidence or effect on the real 2024/2025 inputs was made.

There is also an availability caveat to the leak-free claim. `parse_game_feed` assigns the game's `officialDate` to every PA, including plays flagged as resumed from their later timestamps. Both the production pitcher feature and the catcher copy group solely by that date. A later resumed event can therefore enter history before the event actually occurred, even though `shift(1)` is correct algebraically.

**Measured framing counterexample:** Five original dates June 1–5 with CSR `0` give a June 6 catcher feature of `0`. Adding a synthetic PA that occurred September 1 but carries the suspended game's June 1 `date`, with CSR `1`, changes the June 6 feature to `0.1`. The current feature tests mutate the future according to `date`, so they do not test this availability boundary. This is inherited from the production date convention, not a new error in the grouping-key substitution. Parity against production cannot certify absence of a shared error.

The permissible conclusion is: **same measure and date-level shifting, conditional on the historical rows' availability semantics**. An unconditional leak-free claim on the registered real inputs has not been established. Retaining legitimate resumed PAs for training does not authorize admitting a future event before it became available.

### B5 — §5 contradicts itself about decision weight

**Source:** registration lines 56,62–76; `screen.py:119–125,144–149`; `src/bts/experiment/runner.py:164–196`.

The repo's screening rule really does use exact P(57) and mean-max-streak deltas when P@1 is neutral within 0.3pp. Its pass boolean then affects the positive and negative dispositions. Calling these metrics “reported but without decision weight” is false. The same list also includes per-season P@1, which is explicitly decisive, and CPU cost, which controls stops.

**Measured:** Keep the three seeds' P@1 deltas fixed at:

```text
seed 1: (+0.015, +0.015)
seed 2: (-0.001, +0.009)
seed 3: (-0.001, +0.009)
```

Their mean seasonal deltas are `0.0043333333` and `0.011`; `m=0.0076666667`, `t=2.0909090909`. Give seeds 2–3 neutral mean-max-streak deltas. With exact P(57) delta zero for both, only seed 1 passes and the disposition is **inconclusive**. Change only seed 2's exact P(57) delta to `+0.001`: two seeds pass and the disposition becomes **positive**.

For exactly three valid seeds, the principal inequalities otherwise match §5: strict positive seasonal means, `m>=0.003`, `t>=1.5`, two passes; negative only with `m<=0` and fewer than two passes. The zero-variance wording needs the missing `m=0` case: the implementation uses `t=0` then, not ±infinity. That edge does not make a positive by itself, but should be fixed before freezing the rule.

The no-multiplicity-adjustment statement is candid. These seeds measure algorithm-seed sensitivity on two consumed seasons; they are not three new season samples. The canonical seed file also says its positions come from an outcome-ranked, stratified historical baseline distribution. Taking its first three positions is predeclared here, but should not be presented as an outcome-independent random sample or full-range canonical-n10 coverage.

### B6 — Effective baseline feature settings are neither frozen nor recorded

**Source:** `src/bts/features/compute.py:26–42,157–184,327–329`; `screen.py:242–247`; `scripts/audit/c1/launch.py:70–71,382–389`.

`BTS_ROOKIE_GATE_K` and `BTS_PITCHER_HR_30G_MIN_PERIODS` are read when `bts.features.compute` is imported. They change baseline features, but the driver checks neither and records neither effective constant. The launcher loads the production environment file; the registration supplies no accepted values for these settings. I did not read that file and cannot verify its current values.

**Measured synthetic computation:** On the same PA frame and lookup, settings `(20,7)` gave 34 non-null pitcher rolling-HR values; `(0,10)` gave 28 and different batter rolling-HR values. Both framing self-checks passed. Thus the exact same admitted PA pins can lead to a different baseline/variant experiment without the manifest identifying the cause. Setting variables after import would not repair the already-built constants.

This does not invalidate the **feature-list** claim: the baseline is the current checked-out production `FEATURE_COLS`, with 16 columns. It invalidates treating that list, the PA pins, and the recorded LightGBM dictionary as a complete effective computation identity.

### B7 — Fixed 45-hour seed budgets cannot finish stage one at the stated estimate

**Source:** registration lines 79–86; `scripts/audit/c1/ledger.py:121–133`; `scripts/audit/c1/launch.py:361–369`.

At the note's estimated 30 CPU-hours per seed and starting total `0.48`, seed 3 reaches the launcher with `60.48` already used. Its mandated declared budget of `45` would exceed the cap: `60.48+45=105.48`. The launcher refuses that command even after the 50-hour acknowledgement. This is distinct from actual use at the 30-hour estimate, which would leave `9.52` under 100.

**Measured pure gate calls:** `gate(0.48,45,False) -> ok`; `gate(30.48,45,False) -> ok`; `gate(60.48,45,True) -> over_cap`. These are hypothetical ledger values from the note, not measurements of box usage.

The estimate is honestly marked unmeasured, and the residual `9.52` versus remaining planned `11.1` already exposes a `1.58`-hour shortfall at the estimate. But the fixed launch budgets create an additional refusal the note omits. Predeclare how an owner-approved remaining-seed launch budget is selected under the existing cap; do not silently change 45 or treat a 50-hour acknowledgement as a cap increase.

### Correct or bounded parts of the unit

- **Feature definition:** `framing_by` matches production's per-entity/date mean of PA CSR, then `shift(1).expanding(min_periods=5).mean()`. It gives equal weight to non-null daily means, not to pitches or PA counts across days. Missing catcher keys yield NaN; same-day doubleheaders merge before shifting. The copied borderline measure is a proxy, not an adjusted causal measure of catcher skill.
- **Self-check:** On its input frame it checks exact value equality, NaNs equal, and row count. With the reviewed left merge's unique entity/date keys, source inspection supports preservation of row order. It does not independently compare full row identities, exercise production on real inputs in this review, validate catcher attribution, close ancillary inputs, or test real event availability. The author's value-reference test builds its “production” column using the same `framing_by`; the independent reference test and source comparison help, but are different evidence from real-input parity.
- **Catcher approximation:** `_get_starting_catchers` selects the first boxscore player with catcher in `allPositions`, falling back to primary position. It does not establish that the player started or caught each PA. A synthetic boxscore with reserve ID 888 listed before starter ID 777 returns 888; reversing those listings returns 777. Consequently this screen measures history assigned to a postgame, game-level catcher proxy and mixes substitutions, pitching, umpiring, and game context. §8 admits approximation and the lack of a serving input, but should also state that the selected player need not be the starter. A future 2027 test needs an independently specified pregame catcher identity.
- **Variants:** A replaces the pitcher feature in its original position; B appends the catcher feature. The blend rewrite is the same rule as `runner.py:308–326`, including retention of third-tuple parameter dictionaries. Baseline retains the 12 current blend configs and their Statcast extras.
- **Probability basis:** All six calls explicitly pass `game_probability_mode="estimated_pa"` and `retrain_every=7`. Standard `top_n=10` is the called function's default. This avoids actual-PA-count compounding; it remains an offline diagnostic using retrospective participant/lineup data, with the outcome/availability limits in B4. The training window starts at 2019 and admits test-season training rows only from dates before the evaluated day.
- **Admission wiring:** The wrapper validates exactly nine lowercase-hex SHA-256 pins, supplies `pins_digest`, scope `catcher framing screen stage one`, and X-35 to the unchanged shared gate. Its closure covers package initializers, the C1 and framing scripts, `src/bts`, dependency files, and this design. The shared gate binds a plain SIGN's exact reviewed-commit field, archived report bytes, first publication of the structured exposure row, and unchanged tracked/untracked closure state. It does not bind ancillary runtime inputs or effective environment settings. The foreign-module check is a snapshot before subsequent BTS imports, not an ongoing import restriction; the prescribed clean CLI entry is an execution assumption.
- **Determinism and seed:** Deterministic/row-wise flags are built at import and are correctly checked in the actual `LGB_PARAMS`, in addition to the environment refusal. The seed is read dynamically by the backtest's `_rs()` at model creation; setting it in `run` is correct. A synthetic `_rs()` call returned registered seed `1746737973` after changing the environment, despite earlier imports. There is no demonstrated stale import-time seed bug.
- **First-unit stop:** CPU accounting includes the process's user/system CPU and reaped-child CPU. LightGBM thread CPU is charged to the process. A completed first unit above 7.5 CPU-hours stops before a second walk-forward; equality is permitted. This is a post-completion check, not an interrupt at 7.5. The implementation applies it to each seed's first unit, whereas §6 names the first unit of seed 1. The launcher supplies the cumulative hard budget separately. Make that stricter per-seed behavior explicit.
- **Outputs:** The success path saves a claim, manifest, six profile parquets, six unit records, per-variant/per-season P@1, pass reasons, secondary deltas, and CPU total. The stop paths use exits 3/4 and `STOPPED.json`. Profiles can be retained before a CPU stop; a stopped run is not a complete seed. Full scorecards/diffs are computed but not retained. The Monte Carlo default is 10,000 trials with seed 42 (`monte_carlo.py:130–135`); exact P(57) and Monte Carlo season-best metrics remain structural/bootstrap diagnostics, not independently validated contest policy value.
- **Limits:** Consumed evaluation, no deploy/production conclusion, missing pre-2019 catcher history, the historical comparison caveat, and the estimate's unmeasured status are plainly disclosed. Real parquet coverage claims, historical experiment performance claims, and current box cost totals were not independently verified because the review forbids those reads.

### Tests and mutant ledger

The permitted **53/53** includes 28 framing tests and 25 shared admission tests. The framing end-to-end fixture replaces admission, input loading, and production feature computation; its fake walk-forward confirms call arguments, not the real input closure, outcome basis, or training behavior. Existing duplicate-claim tests use one root; existing aggregation tests use three correct JSON seeds; none challenges the defective consumption cases above.

The checked-in ledger candidly records F12 surviving, then failing after its masking practical-threshold condition was isolated. All **26 mutation anchors occur exactly once** in the reviewed source. Comparing its original ledger commit `9637a447e8ef2f60555a7404c0dd27da816b3e7c` to HEAD showed only the README/output and the isolated two-line test change; the driver is unchanged. I did not run the ledger's mutation runner, because it edits tracked source. Its existing output supports the listed assertion failures, not exhaustive boundary coverage.

`mutant_runner.py:18–24` calls every nonzero pytest exit “RED”, including collection/import/setup errors. It also exits successfully after a SURVIVED mutant or invalid anchor. It is therefore not a machine-enforced “all named tests executed and failed for the intended reason” gate. The shown ledger has test-failure summaries, so I am not reclassifying its 26 recorded kills as infrastructure errors; that is a runner limitation to fix.

Missing boundary cases/mutants include one/two/four seed inputs, duplicate paths and seed IDs, foreign seeds, mismatched manifests/input digests, claims under a second root, later-seed invocation without the owner release, ancillary lookups/raw scans, import-time feature settings, resumed-only labels, and future resumed events assigned an old official date. Also add a real third-tuple config case, and a seed propagation check at the LightGBM constructor boundary rather than only the manifest's environment field.

## Required changes

This is a BLOCK requiring implementation and registration changes, not a conditional signature that can be cleared by prose edits alone. No tracked repairs were made.

1. **Close the input inventory.** Freeze and hash every outcome-affecting ancillary lookup actually consumed, restrict it to the registered historical seasons, include its bytes digest in the admission/exposure binding, and consume it without scanning or rewriting the live raw/cache directories. Explicitly bypass the unused park-drag read for this screen, or register and pin it too. Update §3 and the driver together. Test the actual feature-computation read boundary with synthetic 2026 and changed-cache canaries; framing parity must not be used as the input-closure test.
2. **Enforce stage-one aggregation.** Require exactly the three distinct registered seeds and three distinct resolved run directories before issuing any positive/negative/inconclusive disposition. Reject or explicitly return incomplete for partial stages; reject duplicates, foreign/extra seeds, stopped runs, mismatched reviewed/input identities, and missing or inconsistent manifest/claim/result relationships. Retain the scorecard/diff evidence needed to recompute the summaries and pass booleans instead of accepting caller-supplied booleans without checks. Add the counterexamples in B2 as tests and mutants.
3. **Bind claim scope and owner release.** Define and enforce one canonical persistent claim namespace on the box; production CLI output options must not bypass it. Provide a concrete reviewed launch procedure/wrapper binding that namespace, launcher budgets, deterministic environment, seed order, and Eric's recorded release of seeds 2–3 after the seed-1 cost report. Keep synthetic test roots behind an explicit test seam. Test that changing an output path cannot create a second eligible seed claim and that later seeds cannot be launched prematurely through the admitted entry.
4. **Correct contest labels and bound event availability.** Derive scoring truth from the already pinned PA bytes using `filter_out_resumed_portion`, joined by stable batter/game identity with an explicit missing/void rule; retain legitimate PA rows for appropriate feature/training history. Do not reread unpinned truth or fix this by deleting resumed PA indiscriminately from training. Establish how resumed-event availability is handled before claiming leak freedom; where the registered PA inputs cannot establish it, add a fail-closed stop or explicitly revise the screen's retrospective claim and obtain review of that bounded claim. Add resumed-only-hit and future-resumption history tests. State these label/date limits in §8.
5. **Remove the decision-weight contradiction without silently changing the rule.** State explicitly that per-season P@1 determines §5, that the two streak deltas affect §5 indirectly through the registered per-seed neutral fallback, and that CPU cost controls stops. Replace the “without decision weight” description accordingly. Specify `t=0` when both mean and standard deviation are zero. Disclose the first-three seeds' outcome-ranked canonical origin and the narrow algorithm-seed interpretation.
6. **Freeze effective feature settings.** Register the accepted rookie-gate and pitcher-min-period values; validate the effective imported constants before outcome-bearing work and record them in the manifest. If production values differ from source defaults, establish and document the intended baseline before exposure. Validate actual model seed/flag propagation in a synthetic constructor probe.
7. **Resolve the budget procedure.** Replace the unconditional 45-hour budget for every seed with an explicit owner-approved procedure that can respect the remaining combined 100-hour cap. Include declared-budget headroom as well as projected actual cost in the seed-1 report. Clarify that the 50 gate is checked between launches, can be crossed during a job, and does not raise the hard cap. Document the implemented first-unit stop on every seed, or align it with the registered seed-1-only condition.
8. **Make mutant outcomes accountable.** Treat collection/setup/tool errors as inconclusive infrastructure failures, require the intended test failures, and return failure if any selected mutant survives, has an invalid anchor, or fails to execute its intended tests. Add the boundary mutants above and retain evidence for the revised commit. A later plain SIGN of that revised commit remains necessary; one review round remains under this prompt.

## What was run

- Read the complete review prompt, registration, driver, tests, ledger runner/spec/output, and relevant in-checkout production/admission/launcher/scorecard/build sources. Initial outside-checkout reads and the incidental dependency cache write are disclosed at the start of this report.
- `git rev-parse HEAD` returned `dc58a32dcc9968fe268b04a19dde65fa2ea5c096`; `git status --short` was empty. Reviewed `git diff fd17fb8 dc58a32dcc9968fe268b04a19dde65fa2ea5c096`; `git diff --check` was clean. The diff contains only the nine subject files listed in the prompt and does not change shared admission or production code.
- Exact permitted command: `UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run python -B -m pytest tests/scripts/c2_framing tests/scripts/test_c1_admission.py -p no:cacheprovider` — **53 passed in 18.22s**.
- Synthetic reproductions under `/private/tmp/c2-framing-r1-i3r_o_ud`: `PYTHONPATH=. .venv/bin/python -B .../probes.py`, then `MPLCONFIGDIR=.../mpl PYTHONPATH=. .venv/bin/python -B .../more.py`. These exercised the actual aggregate/disposition/helper/feature/claim/output code as described above. The two complete synthetic `run` calls used fake walk-forwards; no `blend_walk_forward` was executed. Results are preserved numerically in this report; the owned scratch directory is deleted on completion.
- Read only the specifically allowed real data file `data/seed_sets/canonical-n10.json`. No real PA/raw/cache/external artifact, configuration, credential, `.env`, box, network, SSH, `gh`, live provider, or operational state read was performed for the substantive review. No escalation, tracked edit, commit, push, real-data run, or real walk-forward was performed.
- Verified all 26 current mutation anchors without modifying source. Driver SHA-256: `0ea2e10cf73b7b6ddcf7a71b29aa47c9b542a078780a0428efac7f932aa0f741`. The author's tracked-source mutation runner was not executed.
- Final report/receipt format, reviewed-commit uniqueness, final HEAD, unchanged tracked tree, report hash, and owned scratch removal are checked before emitting the completion marker.
