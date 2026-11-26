# Verdict

**SIGN.** Reviewed clean HEAD `e8dcbc6fc94307f7f6c4046ab5e1fd85e194b46f`. The three round-one false-disposition cases now refuse. I found no new wrong disposition or demonstrated false refusal introduced by the fix. This signs the code within the requested round-two scope; the sealed outputs were not inspected.

# B1

**Closed.** `validate_seed` now checks both arms' retained catcher evidence against `trusted()`'s recomputed catchers, values, side-games and PA counts, through `F.evidence_problem` (`f26_studio.py:273`). That function enforces the zero-identified A-posted refusal.

Ran `test_B1_no_finite_posted_catcher_value_refuses`: ten completed `ST.run` outputs with a valid no-candidate table now raise `RunInvalid` with “catcher contrast is unavailable.” Also ran the mismatched-catcher-evidence refusal. These replace the r1 zero-coverage acceptance probe with evidence actually written by the runner on the stubbed world.

# B2

**Closed.** `trusted()` derives `truth.calendar` from pinned `pa_2026`; `validate_seed` requires each manifest to equal it (`:247`–`:249`), and each arm's retained rank-1 rows must cover exactly that calendar (`:263`).

Ran `test_B2_a_date_missing_from_every_seed_refuses`: all ten manifests, profiles and recorded vectors consistently omit the last date, but the pinned input retains it. The aggregate now refuses with a calendar error. The frozen-file-only calendar test also passes.

# B3

**Closed for the r1 false-output cases.** Profile validation now precedes rank-1 reconciliation (`:257`–`:267`): binary hits, finite probabilities, rank structure/order, estimated-PA basis, and identities/original-portion labels against pinned inputs. Scorecards are recomputed and compared after timestamp/JSON normalization; aggregate summaries come from recomputed diffs and must match recorded summaries.

Ran the equivalent r1 probes: nonbinary hits made consistent with recorded vectors, NaN probabilities, and an unknown batter-game all refuse. Also ran refusals for inverted probability order, changed retraining settings, mismatched recorded hits, changed scorecards and changed summaries. Valid ten-seed evidence produced by `ST.run` aggregates successfully.

# New findings

No blocking finding.

Checked the requested exact-equality concern separately: using the synthetic world's nonconstant framing rates, computed expected as-of values before the framing merge, generated both arms' evidence after the merge, and round-tripped that evidence through Parquet. `F.evidence_problem` accepted all 80 side-games for each arm with exact value equality. This exercises the differing run/recomputation paths without opening season outcomes.

Also checked `_same_json` with integer dictionary keys, tuples, NaN, small floats and a nextafter float: JSON round trips and timestamp changes compare equal; a one-ULP numerical change compares unequal. The scoring simulation resets its RNG to seed 42, so rescoring does not depend on a previous call's RNG state. The valid-run tests exercised actual scorecard recomputation without a false refusal.

# Notes

- Ran `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/scripts/c2_framing/test_f26_studio.py tests/scripts/c2_framing/test_f26_rules.py tests/scripts/c2_framing/test_f26_hook.py -q`: **135 passed** (20 studio, 100 rules, 15 hook).
- The r1 reporting notes are applied: `total_cpu_h` sums seed `total_cpu_s`, walk-forward CPU has its own field, and `rank_ties_by_seed` is emitted. Seed CPU totals still exclude preparation and aggregation.
- The manifest check validates the registered procedure fields, feature settings, scoring, seed and deterministic flags. It does not compare the remaining LightGBM parameter entries or record/recheck the complete blend configuration. This is a metadata-validation limit, not a demonstrated changed-recipe path in the unchanged `ST.run`, which supplies the reviewed blend configurations and default model parameters; no new custody requirement is proposed.
- Stored `diff_*.json` files are not compared. The aggregate recomputes diffs and validates the summaries it publishes, so changing only a stored diff cannot change its disposition or reported summaries.
- Tests and independent float probes used synthetic data; feature computation/model training were stubbed in the runner fixtures, and their scorecards use 200 Monte Carlo trials. No full model run, whole suite, sealed-output read, source/test edit, commit or push was performed.
