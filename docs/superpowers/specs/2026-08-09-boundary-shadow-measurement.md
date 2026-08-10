# 2026-08-09 — Boundary-only shadow measurement (registration, amended r1)

**Status: AMENDED REGISTRATION AFTER PARTIAL UNBLINDING.** r0 (immutable at
`4800f7a`) was design-reviewed the same night (Codex r4, gpt-5.6-sol xhigh,
full repo + data-mirror access) and **rejected as implementation-ready**: its
state-reconstruction design contradicted production's contest-anchored streak
(2026-06-17 real-streak-anchoring design; local replay gives 15 where the
6/24 decision record says 13), most census rows carry no persisted state
(31/44 decision records are state-null; only the 13 MDP skips persist
streak/saver), `source=="mdp"` records post-clamp effective actions rather
than raw table lookups, and the live saver is operator-controlled
(`saver_state.json`, 6/18+) — not derivable from local results.

**Unblinding disclosure:** in the course of that review the reviewer computed
provisional B* boundaries and raw one-step diff counts (8 and 7 under a
plausible-but-invalid state path) from the data mirror. Those numbers are
VOID as results but are known to the designers, and they sit at r0's ≥8
follow-up threshold. Consequences, binding: (a) M1 is a **retrospective,
in-sample mechanism measurement** — it can characterize where boundary
rescaling changes table intent, and nothing more; (b) any verdict-bearing
evaluation must run on a stream untouched tonight — prospectively accumulated
decision records (see `bts_daily_decision_v2` follow-up) or a genuinely
held-out design; (c) the ≥8 threshold is retained ONLY as r0's inherited
operational work-investment trigger, explicitly arbitrary and now known to be
binding-adjacent. Lesson recorded: design reviews get schemas and code, never
the measurement data.

## Question (unchanged)

If ONLY the MDP bin boundaries were rescaled to the currently-served p scale
(action table untouched), where does table INTENT change on realized
decision-era states — a one-step disagreement census at deployed states, NOT
a counterfactual season (rows are unchained; a real counterfactual would
diverge in state after the first differing action and is out of scope).

## Artifact

`B*` = four boundaries at quantiles .2/.4/.6/.8 of a served primary-p sample.
Dedicated boundary function (NOT `compute_bins` — that inner-joins rank
pairs): inputs finite and in [0,1] else halt; pandas `linear` interpolation;
classification equality → upper bin (matching `lookup_action`/`np.digitize`
right=False); the four boundaries must be strictly increasing or the variant
is INVALID (no jitter, no dedup).

Variants (window and provenance crossed per review — one primary):
- **PRIMARY**: decision-record `primary.p_game_hit`, decision era
  (2026-06-23 → 2026-08-09; n=44).
- Sensitivity S1: flat-file rank-1 p on the same dates (34 dates carry both
  artifacts; three differ — 7/01 +.0181, 7/04 +.0572, 7/28 −.0254 — flat
  files are provisional, so S1 measures provenance alone).
- Sensitivity S2: flat-file rank-1 p, season-to-date (n=119; window +
  provenance jointly).
Sensitivities never trigger follow-up.

Stability (pre-specified, deterministic): leave-one-calendar-week-out and
endpoint ±7-day variants of the PRIMARY window, reported for boundaries and
diff counts. No new minimum-n rule (the point result is partially known; a
threshold invented now would be post-hoc).

## Census (M1): one-step table-intent disagreement

Rows: decision-era `decision.json` records, exact glob
`<picks>/YYYY-MM-DD/decision.json` with path date == body date and
`schema_version == "bts_daily_decision_v1"`.

Per-row state, by authority (never local pick-result replay):
- `state_source = recorded`: the 13 MDP skip records (streak + saver
  persisted).
- `state_source = ledger_asof`: commit rows whose decision-time (streak,
  saver) can be recovered from the box's authoritative streams — contest
  observation/ledger history and `saver_state` transitions — with an as-of
  timestamp at or before `finalized_at`. Implementation defines the exact
  recovery join; a row recovers only if both components resolve.
- `state_source = unknown`: everything else. Excluded from diffs; counted.

Diff on state-known rows only: `raw_deployed =
lookup_action(table, deployed_boundaries, state)` vs `raw_B* =
lookup_action(table, B*, state)`. Deployed-side parity: `raw_deployed` is
compared to the RECORDED action with clamp attribution (`allow_double`,
different-game executability, floor rules); mismatches attributable to a
documented clamp are reported (raw vs effective); an unattributable mismatch
HALTS. `source != "mdp"` rows (3 `unknown`: 6/29, 7/03, 7/12) are excluded
from diffs, listed.

Counterfactual actions are INTENT only. skip→double intent is flagged
non-executable-verifiable (skip records do not persist the second
candidate); no effective-action claim is made for it.

Reporting: exact streak + bands 0–2 / 3–7 / 8–9 / 10–15 / ≥16; deployed and
B* bin indices + occupancy; all six ordered action transitions per variant;
union and intersection of diff dates across variants; per-row
`policy_identity = verified | inferred | unknown` (pick-file policy-SHA
stamps where present; halt only on a CONFLICTING verified SHA);
`state_source` mix; coverage/exclusion reasons per row.

Outcome annotation: mechanism phase and outcome phase are SPLIT. The census
artifact is produced and committed with outcomes withheld. Annotation
coverage is honest: double→single gradable from the primary slot;
skip→single gradable from policy-shadow records; added second slots are
generally `unavailable` (never imputed; `decision.double_down` is a declined
candidate, not an executed slot). If follow-up triggers, its evaluation
design is registered BEFORE the outcome phase runs (gate-B prereg standards
— 2026-05-24 — govern any directional outcome claim: resolved-day minimums,
complete changed-slot outcomes, intervals).

## Follow-up trigger

≥8 unique, parity-passing, state-known dates with an intent diff in the
PRIMARY variant → a separately registered realized-sequence evaluation
design (7/13 standard; iid-DP prohibited) on a stream untouched tonight.
If ledger-as-of recovery proves infeasible and the state-known set stays
≈13, the census reports coverage-limited results and NO follow-up triggers —
insufficient support is not a null result.

## Artifact + harness contracts

Versioned script under `scripts/audit/`, TDD'd. Tests pin: the boundary
function (hand-built frames; at/below/above each boundary; duplicate-input
invalidity), clamp attribution (positive + halt cases), state-source
labeling, both parity behaviors, determinism (explicit `--as-of`, no
`date.today()` defaults), and artifact schema. JSON artifact under
`data/validation/`: `schema_version`, role ("mechanism census — no
production claim"), immutable parameters, per-file SHA256 of every input
consumed + one canonical manifest hash, registration commit (`4800f7a` r0 /
this amendment), execution commit + dirty-status, policy NPZ sha +
`season_length` + `SEASON_END_DATE`, package versions, UTC `generated_at`;
written via `atomic_write_text`; reproduce command records
`TZ=America/New_York`.

## Follow-up (separate, production): `bts_daily_decision_v2`

The durable fix for this entire class: persist on EVERY final decision
record — streak, saver, state source/status + observation identifier,
`allow_double`, and the executable second candidate on skips. Own small
spec + TDD; makes future censuses exact by construction. Not part of this
measurement.

## Non-goals (unchanged)

MDP re-solve; threshold changes; DD/`p_both` re-estimation; any touch of
`FEATURE_COLS`, the skip rule, delivery, or promotion criteria. Leakage
audit / nuclear test remain mandatory before any future promotion path;
none applies to this read-only census.

## r4 review trail

Rejected r0 blockers: contest-anchored streak (local replay invalid; 6/24 =
13 not 15); state-null commit rows; source=="mdp" ≠ raw lookup + unknowable
counterfactual executability; saver_state operator stream. Majors adopted:
era authority table, day-atomic saver semantics, one-step framing, crossed
variants, deterministic stability probes, quantile/tie/strictness contract,
stopping-rule disclosure, annotation coverage + phase split, registration
honesty, per-row policy identity, input/artifact contracts, state-geometry
bands. Survives from r0: boundary-only comparator shape (table fixed);
days_remaining production parity (census spans 97→50 days; the
season_length=180 clamp inactive); local saver band = inclusive 10–15
day-atomic; iid-DP ban. All review claims independently verified against
code/docs/mirror before adoption (contest_state.py, scheduler.py decision
writers, saver_state.py, 6/17 + 6/21 + 5/24 docs; decision census 41+3,
state-null 31, the three p-mismatch dates).
