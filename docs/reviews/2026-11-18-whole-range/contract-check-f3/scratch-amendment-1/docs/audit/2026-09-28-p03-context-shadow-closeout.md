# Context-stack shadow v2 — closeout read (P-03)

**Date computed:** 2026-09-28 · **Plan item:** W1.4b · **Register:** X-06 → P-03 (`docs/audit/2026-09-22-exposure-register.md`) · **Data:** frozen W0.7 snapshot `final-20260928/` (manifest sha256 `e3dce024…`) · **Evidence:** `docs/audit/2026-09-28-p03-context-shadow-closeout.json` · **Reader:** `scripts/audit/p03_context_shadow_closeout.py`

## Protocol (quoted, unchanged)
- Register §B P-03: "**Code (`src/bts/shadow_eval.py`):** `SHADOW_STATUS_DEFAULT_MIN_DAYS = 30`; paired-day evaluation with Wilson intervals and a two-sided sign test on discordant days (`_sign_test_p_two_sided(prod_only, shadow_only)`); reconciliation via `check-results` on every exit path (2026-08-09 hardening)."
- W1.4b obligation: "report agreement, discordant outcomes, coverage and interval on the frozen eligibility; equal aggregates ≠ equivalence; no-promote stands unless the protocol says otherwise."
- Frozen eligibility = the status rule in `build_shadow_cycle_status`: only `context_stack_shadow_v2` shadow files count, each paired with the production pick file of the same date; a day is evaluable when both RECORDED results are hit or miss.

## Result (recorded results — the protocol basis)
| Measure | Value |
|---|---|
| Coverage | 2026-07-08 → 2026-09-18: 60 v2 shadow days, all paired with a production file, all shadow results resolved; `cycle_state = ready_for_manual_review` (≥ 30). The last 5 days (9/14–9/18) paired against private, undelivered production picks — still model-vs-model days. No shadow files after 9/18: the tail stop left no production pick from 9/19 |
| Evaluable paired days | **59 of 60** (8/13: the production file carries no result — the 8/13 silent-pass incident) |
| Production day hit rate | **32/59 = 54.2 %** (Wilson 95 % 41.7–66.3) |
| Shadow day hit rate | **32/59 = 54.2 %** (Wilson 95 % 41.7–66.3) |
| Shadow − production | **0.0 pp**, paired bootstrap 95 % **−10.2 to +10.2 pp** (10,000 draws, seed 57 — the code defaults) |
| Paired outcomes | both hit 28 · both miss 23 · **production-only hit 4** (7/08, 7/24, 8/08, 8/19) · **shadow-only hit 4** (7/09, 7/11, 8/16, 8/20) · sign test p = 1.00 |
| Decision agreement | same primary **43/60 (71.7 %)** · same unordered pick set **30/60 (50.0 %)** |

## Cross-check: recompute from cached game feeds
`build_shadow_backfill_manifest` (dry run, DD-aware, 0 API calls — every game from `data/raw/2026/`):
- **Shadow:** all 60 recorded results reproduce (0 would change).
- **Production:** two recorded results differ from the game feeds.
  - **8/13**: no recorded result → recomputed **hit**.
  - **8/20**: recorded **miss** → recomputed **hit**. The contest's official record agrees with the recompute (round 971: result hit, both legs hit, streak 6 → 8), and the feed has Chandler Simpson 1-for-4 with a 4th-inning single in a normal final game. The recorded miss was written by `bts reconcile` on 8/26 — see "Finding" below.
- Recomputed numbers: production **34/60 = 56.7 %** (44.1–68.4) vs shadow **33/60 = 55.0 %** (42.5–66.9); shadow − production **−1.7 pp**, paired bootstrap 95 % **−10.0 to +6.7 pp**; both hit 30 · both miss 23 · production-only 4 · shadow-only 3; sign test p = 1.00; agreement unchanged.

## Disposition
**No-promote stands.** Neither basis separates the two models: the recorded aggregates are identical (32/59 each) and the corrected ones differ by one day in production's favour, with intervals about ±10 pp wide. Per the protocol, equal aggregates are not equivalence — the models picked different primaries on 17 of 60 days and different pick sets on 30, and on the days their outcomes diverged the split is 4–4 (recorded) or 4–3 (corrected). Sixty days cannot tell a real ±5 pp difference from zero. The context stack (`CONTEXT_COLS`, v2 frozen 7/08) stays a non-production shadow; any 2027 revisit needs its own pre-registration and forward validation (D3). Register row X-06 moves to *closed — P-03 closeout read, no-promote*.

## Finding outside the protocol: `bts reconcile` flipped two true hits to misses (C-03)
Across the whole season the nightly reconcile (`picks.reconcile_results`, 8-day lookback) made exactly two "corrections", and both were wrong. **Both flipped Chandler Simpson primaries from hit to miss**, each at the day + 6 run:
- **5/10**, flipped on 5/16. Official: round 869 hit, both legs, streak → 3. Feed: Simpson 1-for-4.
- **8/20**, flipped on 8/26. Official: round 971 hit, both legs, streak → 8. Feed: Simpson 1-for-4.

Neither flip changed a production decision. The official streak was already 0 by 5/16 (a 5/12 miss), and the 8/26 recalculated local streak (11) equals the contest's 11; production state has come from the contest profile since June. What they do corrupt is the local pick files: both days read `result = miss` with `slot_results.pick = miss`. That affects any analysis that treats pick files as ground truth. The P-01 formal read is unaffected, because it counts `double_down` slots and both flips were primaries. Its two "primaries (context)" rows (through 9/08 and through 9/13) each contain both flipped days as misses, so each is 2 hits short: 83/109 and 87/114, not 81/109 and 85/114. Mechanism unknown — same batter both times, same lag — so it goes to the W1.5 incident register (failure-path fixture first). Recorded as C-03 in `docs/audit/2026-09-corrections-index.md`; W1.1 must take delivered-day results from the contest ledger, not the local pick files.

## Reproduce
On the box, from `~/projects/bts`: `.venv/bin/python scripts/audit/p03_context_shadow_closeout.py --out /tmp/p03.json` (defaults to the frozen snapshot; the cross-check reads the cached feeds in `data/raw`, read-only).
