# Verdict

**BLOCK.** The prediction path preserves the original science, but the aggregate drops scientific acceptance checks required by §6. It can publish a numerical disposition for an incomplete or invalid test. These are failures of the result-deciding tool's stated contract, not requests to restore launcher custody or improve test-code quality. I have not inspected the sealed runs and make no claim that they contain these states.

# Blocking findings

## B1 — An unavailable A-posted contrast receives a numerical disposition

- **Input/state:** Correct starter-proxy records identify no catcher on any predicted side-game, or every named catcher has fewer than five prior nonmissing daily rates. These are allowed missing-value inputs under §4. Every A-posted prediction-day value is NaN; other model features still permit finite predictions and complete profiles.
- **Wrong output:** `run` writes `results.json` and `aggregate` decides from the hits without enforcing the explicit §6 unavailable-contrast rule. A synthetic ten-seed aggregate with zero identified side-games/PA rows returned `positive`, `m=1.0`, instead of incomplete/refusal. Separately, the actual `table_frame` and `ArmTransform` accepted a valid no-candidate table and produced precisely zero finite values, confirming this input is reachable through the missing-catcher rule.
- **File and line:** `scripts/audit/c2_framing/f26_studio.py:187` records coverage but never gates it; `:224` and `:245` accept/decide without reading catcher evidence. The original check is in `f26.evidence_problem`.
- **Required change:** Before acceptance, validate the retained A-posted catcher evidence and require at least one finite value on a predicted PA. Zero coverage must take precedence over every numerical disposition. A run-time early refusal is also useful, but the aggregate must enforce completeness.

## B2 — Agreement between seeds substitutes for the complete pinned calendar

- **Input/state:** The correctly pinned `pa_2026` has scoreable rows on April 1 and April 2. All ten seed manifests and their three retained profiles consistently omit April 2; their recorded rank-1 vectors agree with those profiles.
- **Wrong output:** The aggregate accepts the shared one-day calendar and reports `positive`, `m=1.0`, `n_days=1`. Calling `calendar_2026` on those same pinned inputs returns both dates. §6 expressly says a shared omission is incomplete. The current runner derives its calendar correctly; the independent aggregate nevertheless accepts this partial-output state.
- **File and line:** `scripts/audit/c2_framing/f26_studio.py:221` trusts each manifest's calendar; `:235`–`:239` check only inter-seed agreement.
- **Required change:** Re-derive the expected calendar with `calendar_2026(inputs, pins)` at aggregate time and require each manifest and every retained arm to match that exact nonempty date set. Retain the existing duplicate/rank-1 checks.

## B3 — Reproducing recorded hits does not validate the retained profiles

- **Input/state:** All ten seeds cover the full pinned calendar and their recorded hits match `F.rank1` on their profiles, but a profile contains invalid hits, missing probabilities, or a batter-game absent from the pinned PA file. Each was probed separately against a valid synthetic control whose profile/identity checks pass.
- **Wrong output:** A-posted rank-1 hits of `2` against baseline `0` returned `positive`, `m=2.0`; NaN probabilities and unknown batter ids also returned `positive`, `m=1.0`. `F.rank1` merely casts hits to integers; it does not enforce binary labels, probabilities, identities or probability ordering. These invalid states must be incomplete under §6. Agreement between a file and its recorded vector cannot establish those properties.
- **File and line:** `scripts/audit/c2_framing/f26_studio.py:223`–`:228` replace the original scientific validation with only `F.rank1` and vector equality.
- **Required change:** Restore the scientific profile checks independently of the removed launcher: binary hits, finite bounded probabilities, valid ranks/order, registered estimated-PA basis, and identities/original-portion labels checked against pinned inputs (`S.profile_problem`, `F.identity_problem`, `F.order_problem`, or equivalent). Also preserve §6's registered-settings validation and reconciliation of reported scorecards/diffs/streak summaries to retained profiles; currently the latter are copied from `results.json` unchecked at `:228`/`:250`.

# Notes

- **CPU reporting:** `total_cpu_h` at `:251` sums only walk-forward unit CPU. It omits feature computation, relabeling/evidence writes and scoring already included in each seed's `results.total_cpu_s` (`:198`), plus preparation/aggregation. Sum the seed totals or label this field as walk-forward CPU only. This is a reporting NOTE, not a demand to restore C1 accounting or limits.
- **Ties:** §6 requires reporting rank ties. Unlike the original aggregate's `rank_ties`, this aggregate emits no tie counts. Include the counts from the profile ordering check. Daily delta ties and constant-bootstrap flags are already supplied by `F.dispose`.
- Reading the recorded pins verifies equality between seed declarations; it does not itself validate profile provenance. No additional custody mechanism is requested here—only the explicit scientific acceptance checks above.

# What I checked

- Reviewed at clean repository HEAD `d2da1853f66b7662b53bf2639a0bdc6755765403`; no source/test changes made. Read the frozen design, X-37 and the tier-C work policy. Did not search or read either forbidden results directory.
- Compared `prepare` with `f26._prepare`: same historical PA/lookup pins, game-id selection, starter-proxy/lookup extraction, table validation/order and canonical derived bytes for the same feeds. No additional scientific difference found; the declared rebuild/platform/launcher deviations were treated as ruled.
- Compared `run` with `f26.run`: same input closure, feature computation, self-check, build-keyed framing/as-of history, original-portion labels, three arm configurations, walk-forward arguments, strict predictions, profile readback and scoring. Moving the seed environment assignment before imports does not lose the seed: training reads it dynamically.
- Inspected the walk-forward hook: it runs after fitting on untouched available rows, transforms a day copy, and reaches both starter and reliever prediction paths. Cache and per-PA accumulation remain off by default; top-N defaults to 10.
- `calendar_2026` reads the pinned 2026 file alone and calls the correct original-portion calendar rule. This avoids the concatenated flag-dtype problem without changing the intended dates.
- Aggregate subtraction is A minus baseline, registered seeds are traversed in the literal order, and `F.rank1` orders by ascending calendar dates. `F.dispose` supplies the registered paired-day/bootstrap calculations. A-projected's disposition is removed and has no deciding weight.
- Fewer than ten registered seed directories, stopped/unfinished seeds, unequal seed calendars, differing pin declarations and rank-1 vectors not reproduced by the profiles are refused. Extra unregistered directories are ignored and do not enter the aggregate.
- Ran the three authorized files together with `UV_CACHE_DIR=/tmp/uv-cache uv run pytest ... -q`: **126 passed** (11 studio, 100 rules, 15 hook). No whole suite or model training run.
- Ran temporary synthetic aggregate probes with ten registered seeds, a real pinned two-date parquet and an otherwise valid profile control. Independently reproduced B1–B3; the existing profile/identity validators rejected the malformed-profile examples. Also exercised the real missing-catcher transform. These are synthetic failure-direction checks, not measurements of the sealed season results.
