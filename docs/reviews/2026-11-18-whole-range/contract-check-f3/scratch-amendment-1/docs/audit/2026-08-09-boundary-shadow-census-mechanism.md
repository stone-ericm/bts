# 2026-08-09 — Boundary-shadow census: mechanism phase (outcome phase WITHHELD)

Executed per the amended registration
(`docs/superpowers/specs/2026-08-09-boundary-shadow-measurement.md`, r0
`4800f7a` / r1 `6db921c`). Harness: `scripts/audit/boundary_shadow_census.py`
(10 tests). This document reports MECHANISM results only — where quintile
boundary rescaling changes the deployed action table's one-step intent at
realized decision-era states. **No outcome data was computed, read, or
annotated.** Rows are unchained (not a counterfactual season).

## Gates — both clean

- **Gate A (state authority): 13/13.** The ledger-as-of resolver (latest
  `contest_ledger.jsonl` observation with `recorded_at ≤ finalized_at`;
  saver = account flag, constant `active` since 2026-06-18) reproduced every
  recorded skip state exactly. The 28 `ledger_asof` commit rows therefore
  stand on a validated recovery rule, not inference.
- **Gate B (action parity): 41/41 `parity`.** Raw deployed-boundary lookups
  reproduce every recorded mdp-source action; zero clamp events in the era.

Coverage: 44 decision records (2026-06-23 → 08-09) = 41 mdp (13 recorded +
28 ledger_asof — **every mdp row state-known**) + 3 non-mdp excluded.
Policy identity: 34 rows `verified` via pick-file `policy_npz_sha256` stamps
(= loaded npz, `66d154717ae51afb…`), 10 `inferred` (skip days, no flat file).

## Boundaries (quintiles, linear interpolation; all variants valid)

| set | n | values |
|---|---|---|
| deployed | — | .7960 .8115 .8252 .8407 |
| PRIMARY (decision-record p, era) | 44 | .7431 .7745 .7828 .8063 |
| S1 (flat p, same dates) | 34 | .7690 .7823 .7946 .8170 |
| S2 (flat p, season) | 119 | .7363 .7586 .7771 .7895 |

The rescale drops the entire ladder ~4–5pp — the served-scale mismatch the
May gate memo documented, quantified at current scale.

## One-step intent diffs (state-known rows, deployed table held fixed)

| variant | n_diff | transitions | bands |
|---|---|---|---|
| **PRIMARY** | **8** | skip→single ×3, skip→double ×2, single→double ×3 | 3-7: 3 · 8-9: 2 · 10-15: 1 · ≥16: 2 |
| S1 | 5 | subset | — |
| S2 | 7 | subset + 8/07 | — |

PRIMARY diff dates: 06-26, 07-01, 07-04, 07-26, 07-27, 07-28, 08-01, 08-07.
Union = the PRIMARY set; intersection (all three variants) = 06-26, 07-01,
07-26, 07-28. skip→double intents are flagged non-executable-verifiable (v1
skip records carry no second candidate). Notable, stated mechanically: the
streak-8 skip days 07-28 and 08-01 flip to non-skip intent under
current-scale boundaries.

**Stability: every probe returns exactly 8 diffs** — all seven
leave-one-ISO-week-out variants and both ±7-day endpoint variants (first
boundary ranges .7281–.7665 across probes; the action-diff census does not
move). The mechanism result is insensitive to the window choices the design
review flagged.

## Follow-up trigger — FIRED, with its registration honored

primary_n_diff = 8 ≥ 8. Per r1 this threshold is the r0-inherited
OPERATIONAL work-investment trigger, disclosed as binding-adjacent after the
partial unblinding (the design review's provisional count under an invalid
state path was also 8). Consequences:

1. The **outcome phase remains withheld.** No retro outcome annotation is
   produced until a follow-up evaluation design is registered first.
2. Per the unblinding clause, any **verdict-bearing evaluation runs on a
   stream untouched tonight** — the `bts_daily_decision_v2` prospective
   records (exact state on every final action; accumulating from the first
   post-deploy decision) are that stream. The follow-up registration should
   define its evaluation on prospective data, with the retrospective era at
   most descriptive.
3. Nothing here authorizes a production change: gate-B history ("no live
   boundary knob") stands; leakage audit + nuclear test remain mandatory
   before any promotion path.

## Reproduce

`TZ=America/New_York uv run python scripts/audit/boundary_shadow_census.py
--picks-dir data/picks --ledger data/picks/account_state/contest_ledger.jsonl
--policy data/models/mdp_policy.npz --as-of 2026-08-09 --output <path>`

Session artifact (Mac mirror of box data, fresh through 08-09):
input-manifest `bc517e229dbe7ed5…`, policy `66d154717ae51afb…` (== box ==
pick-file stamps). The artifact schema is `bts_boundary_shadow_census_v1`;
atomic write; UTC-stamped; per-file SHA256 manifest.
