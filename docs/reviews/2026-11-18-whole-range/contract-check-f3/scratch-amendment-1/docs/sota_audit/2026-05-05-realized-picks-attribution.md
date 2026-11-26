# Realized-picks attribution — SOTA #12 phase 3 (α P0)

**Date**: 2026-05-05 (snapshot 2026-05-05T16:16Z)
**Branch**: `feature/realized-picks-attribution-p0`
**Predecessor**: [Realized-picks calibration (2026-05-04)](./2026-05-04-realized-picks-calibration.md)
**Tracker**: [SOTA audit tracker](../superpowers/specs/2026-05-01-bts-sota-audit-tracker.md), area #12 phase 3
**Canonical artifact**: `data/validation/realized_picks_canonical_2026-05-05.parquet` (74 rows: 68 resolved, 6 pending)
**Scope**: docs + canonicalize-script extension + analysis. Not a deploy authorization.

> Per Codex agent-bus #203: extend the canonical artifact with per-game environment attribution so the strategic-question reframe (low-skill park-driven picks at predicted 0.65-0.80 realize HIGHER than predicted) becomes testable. P0 ships the env-leverage proxy `is_park_driven` derived from raw PA-frame signals; computed `park_factor` and `batter_skill_quartile` are deferred to P1. Same guardrails as the 2026-05-04 memo: this is an audit cut, not a production claim.

## Headline

| Cut | Cell | n (resolved) | gap | wilson_lo | wilson_hi | Verdict |
|---|---|---|---|---|---|---|
| A. post_pooled_mdp_pre_bpm × is_park_driven | park_driven | 5 | +0.136 | 0.231 | 0.882 | overconfident, wide CI |
| A. post_pooled_mdp_pre_bpm × is_park_driven | not_park_driven | 25 | +0.099 | 0.445 | 0.798 | overconfident, narrower CI |
| B. above × DD slot | park_driven | 1 | — | — | — | uninterpretable (n=1) |
| B. above × DD slot | not_park_driven | 14 | +0.227 | — | — | only interpretable DD env cell; overconfident |

**The strategic-question hypothesis** as stated (low-skill park-driven picks realize HIGHER) **is not directionally supported on this sample with this proxy**, with the explicit caveat that the relevant cell counts are n=5 (Cut A park_driven) and n=1 (Cut B DD park_driven). Within the post_pooled_mdp_pre_bpm stratum:

- Park-driven cells (Cut A) show overconfidence of similar magnitude to non-park-driven cells (+13.6pp vs +9.9pp) — the gap difference is well inside Wilson noise on n=5.
- The DD-slot +19.2pp signal from the 2026-05-04 memo decomposes (Cut B) as DD park_driven n=1 (uninterpretable) and DD not_park_driven n=14 (+22.7pp). The DD park_driven cell currently has insufficient support to test, so we cannot say whether the DD overconfidence "concentrates" anywhere — only that it is measurable in the not_park_driven cell.

Current support is insufficient to test the park-driven DD cell. The artifact + cut now exist; both cells become tractable as more post-bpm picks resolve.

## Headline refresh from 2026-05-04 → 2026-05-05

The post_bpm strict-current model stratum grew n=5 → n=7 (added 2026-05-02 + 2026-05-03 days). The headline gap moved from −26.2pp (5/5 hits, severely under-confident) to **+2.5pp (5/7 hits, calibrated within sampling noise)**. Two days of resolved post-bpm picks shifted the n=5 outlier toward calibration. The `inconclusive — sample-size-limited` framing remains correct: Wilson CI on 5/7 is [0.30, 0.95].

```
regime                         n hits   rate  mean_p     gap   Brier     BSS
------------------------------------------------------------------------------
post_bpm                       7    5  0.714   0.740  +0.025  0.2076  -0.017
post_pooled_mdp_pre_bpm       30   19  0.633   0.739  +0.105  0.2479  -0.068
pre_pooled_mdp                31   23  0.742   0.735  -0.007  0.1920  -0.003
```

## Methodology

### `is_park_driven` rule

The artifact extends the per-row schema with five new columns derived from the PA frame's per-game environment (`scripts/canonicalize_realized_picks.py`, env helper added in this PR):

```
pick_venue_id        (Int64, nullable)
pick_roof_type       (str, raw "Open" / "Dome" / "Retractable" / "Closed")
pick_weather_temp    (float)
pick_is_indoor       (BooleanDtype, NA when game env missing)
is_park_driven       (BooleanDtype, NA when game env missing)
```

Env is joined by `pick.game_pk` (not by `(batter_id, date)`) — this is the key change Codex bus #203 required. The PA frame's `venue_id`, `roof_type`, and `weather_temp` are game-level constants (verified: 0 of 443 game_pks in 2026 had multi-valued env), so a `groupby('game_pk').first()` table is the correct join surface. Joining on `(batter_id, date)` would have produced the wrong env on doubleheader days, where the same batter appears in two games with different venues / weather.

The boolean rule:

```
is_park_driven = (pick_venue_id == 19)
                 OR (pick_weather_temp >= 85.0 AND NOT pick_is_indoor)
```

`venue_id == 19` is Coors Field — structurally the highest park-factor venue in MLB (mile-high altitude, ball-flight aerodynamics fundamentally different from coastal parks). Hardcoded as a constant per Codex's guidance not to add a mandatory raw-game-JSON dependency to canonicalization. `weather_temp >= 85.0` follows the Statcast-era convention for "hot enough that ball flight measurably changes." Indoor games are excluded from the weather branch because dome-reported temperatures are typically synthetic / fixed and don't carry day-to-day environmental leverage. `pick_is_indoor` follows production's convention (`src/bts/features/compute.py:557-559`): roof_type lowercased and `isin({"dome", "closed", "retractable"})`.

### What this captures and does not capture

`is_park_driven` is an **environmental-leverage proxy**: does this game's environment plausibly elevate hit rate compared to a neutral game? It is **not** a feature-attribution measure. It does not test "did the model pick this batter because of park-environment features rather than batter-skill features" — that question requires SHAP / counterfactual model runs and is out of scope for P0.

Two consequences:

1. A pick of a Coors batter on a cold day flags `is_park_driven=True` regardless of whether the model's prediction was actually elevated by park-driven features. The cut therefore mixes "env-leveraged signal" with "happened to be at Coors."
2. Hot-weather games are flagged regardless of pitcher matchup, batter handedness, or other interactions. If the strategic-question hypothesis is really about a specific feature interaction, this proxy will dilute it.

The cut is informative for an upper-bound check: if even this loose proxy showed strong directional support for the under-confidence hypothesis, that would be evidence. The absence of directional support is weaker evidence — it could mean the hypothesis is wrong, or that the proxy is too noisy. P1 (computed park_factor + batter_skill_quartile) tightens the proxy.

### Excluded rows

Two pre_pooled_mdp picks (2026-03-29, 2026-03-30) have `pick.game_pk == None` in the source JSON (early-season picks before the scheduler reliably populated game_pk). Their env columns and `is_park_driven` are NA per the schema contract. These rows are reported in the `(env-NA, excluded)` line of Cut A but excluded from any cell totals. The post_bpm and post_pooled_mdp_pre_bpm strata had no env-NA rows.

## Cut A — regime × is_park_driven (resolved-only, env-attributed)

```
regime                       env                  n hits   rate  mean_p     gap  wilson_lo  wilson_hi
----------------------------------------------------------------------------------------------------
post_bpm                     park_driven          2    1  0.500   0.758  +0.258      0.095      0.905
post_bpm                     not_park_driven      5    4  0.800   0.732  -0.068      0.376      0.964
post_pooled_mdp_pre_bpm      park_driven          5    3  0.600   0.736  +0.136      0.231      0.882
post_pooled_mdp_pre_bpm      not_park_driven     25   16  0.640   0.739  +0.099      0.445      0.798
pre_pooled_mdp               park_driven          4    4  1.000   0.771  -0.229      0.510      1.000
pre_pooled_mdp               not_park_driven     25   17  0.680   0.730  +0.050      0.484      0.828
pre_pooled_mdp               (env-NA, excluded)   2
```

**post_bpm (n=7 attributed)**: park_driven n=2 with gap +25.8pp; not_park_driven n=5 with gap −6.8pp. Both Wilson CIs span more than 0.5 in width. No directional claim.

**post_pooled_mdp_pre_bpm (n=30 attributed)**: park_driven n=5 with gap +13.6pp; not_park_driven n=25 with gap +9.9pp. Gaps differ by 3.7pp, which is well inside sampling noise at n=5 (Wilson CI on 3/5 spans 0.65 in width). The cut does not directionally separate park-driven from not_park_driven calibration on this sample; both cells point to overconfidence within their respective wide CIs. **The cut is testable but currently uninformative for a directional verdict.**

**pre_pooled_mdp (n=29 attributed)**: park_driven n=4 went 4/4 (gap −22.9pp, under-confident) while not_park_driven n=25 was calibrated (+5.0pp). Reported for historical context only — different policy than current production.

## Cut B — regime × slot × is_park_driven (resolved-only)

```
regime                       slot          env                  n hits   rate  mean_p     gap note
--------------------------------------------------------------------------------------------------------------
post_bpm                     primary       park_driven          2    1  0.500   0.758  +0.258 exploratory
post_bpm                     primary       not_park_driven      2    2  1.000   0.722  -0.278 exploratory
post_bpm                     double_down   park_driven          0    -      -       -       -
post_bpm                     double_down   not_park_driven      3    2  0.667   0.739  +0.072 exploratory
post_pooled_mdp_pre_bpm      primary       park_driven          4    2  0.500   0.743  +0.243 exploratory
post_pooled_mdp_pre_bpm      primary       not_park_driven     11    9  0.818   0.755  -0.064
post_pooled_mdp_pre_bpm      double_down   park_driven          1    1  1.000   0.708  -0.292 exploratory
post_pooled_mdp_pre_bpm      double_down   not_park_driven     14    7  0.500   0.727  +0.227
pre_pooled_mdp               primary       park_driven          4    4  1.000   0.771  -0.229 exploratory
pre_pooled_mdp               primary       not_park_driven     11    7  0.636   0.745  +0.108
pre_pooled_mdp               double_down   park_driven          0    -      -       -       -
pre_pooled_mdp               double_down   not_park_driven     14   10  0.714   0.718  +0.004
```

Cells marked `exploratory` have n<5 and should not be interpreted directionally; they are reported for completeness so the resolved sample sizes are visible. Cells with `n=0` are absent.

### The DD-slot question

The 2026-05-04 memo identified +19.2pp DD-slot overconfidence in the post_pooled_mdp_pre_bpm stratum (n=15) as the most actionable finding from that cut. Cut B decomposes that signal:

- **DD park_driven**: n=1 (1 hit, gap −29.2pp by point estimate). Uninterpretable at this n.
- **DD not_park_driven**: n=14 (7 hits, gap **+22.7pp**). The only interpretable DD env cell in the stratum, and it is overconfident.

The DD overconfidence is measurable in the not_park_driven cell. The DD park_driven cell currently has n=1, which is too small to evaluate against the not_park_driven figure or to test directionally — we cannot say whether the +22.7pp overconfidence is specific to non-park environments, present in both, or absent in park environments. Two readings remain consistent with the available data:

1. **Structural DD-selection issue largely independent of park environment**: the strategy's DD selection produces overconfident picks in a way that does not require park-leverage to explain. Under this reading, the natural next investigation is DD-selection mechanics (lineup-position dependence, pitcher-matchup interactions, MDP-policy bin assignment for rank-2 candidates), not park-environment refinement.

2. **Park-driven DD cell is too small to evaluate**: with only n=1 in DD park_driven, the cell could land anywhere from well-calibrated to substantially overconfident as more rows resolve. Under this reading, we don't yet know whether park environment matters for DD overconfidence; we just don't have the data to rule it in or out.

**Both readings are consistent with the framing that the strategic-question hypothesis is not directionally supported on this sample with this proxy** — neither reading lands on "park-driven cells realize higher than predicted." The readings differ on whether the DD signal is park-environment-dependent at all; distinguishing them requires the DD park_driven cell to grow, which will happen as more post-bpm picks resolve.

### Primary-slot observation

Within the same stratum, the primary slot tells a cleaner story:
- primary park_driven n=4: gap +24.3pp (large, but exploratory).
- primary not_park_driven n=11: gap −6.4pp (well calibrated, narrower CI).

This is **consistent with the DD-slot reading 1**: the model's primary picks are well calibrated in non-park environments, and the under-/over-confidence noise in the park-driven cells is sample-size-limited. The +24.3pp gap on primary park_driven (n=4) cannot support an overconfidence claim with any confidence — Wilson on 2/4 is [0.07, 0.93].

## Interpretation guardrails

1. **`is_park_driven` is an env-leverage proxy, not a feature-attribution measure.** A null result against the strategic-question hypothesis does not mean park-leverage doesn't matter for calibration; it means *this proxy*, *on this sample*, doesn't show directional support.

2. **The strict-current post_bpm regime is still underpowered (n=7).** No cell-level claim is supportable under post_bpm. Re-run #12 phase 2 in 30 days as planned; the artifact + cut are reusable.

3. **Park-driven support in the post_pooled_mdp_pre_bpm stratum is n=5 (regime-level Cut A).** Wilson 95% CI on 3/5 spans 0.65 in width. The cut is testable but currently uninformative for a directional verdict; the stratum is a closed window (2026-04-15 → 2026-04-30) and will not grow.

4. **The DD-slot signal decomposes against park-leverage with only n=1 in DD park_driven.** This is the cell future cycles need to watch.

5. **Coors hardcode**: this P0 treats Coors as `venue_id=19`. If MLB ever renumbers venues (unlikely but possible), the rule silently fails-open (no Coors picks would be marked park-driven via the venue branch; the weather branch still works at any venue). A test on synthetic data covers the rule's logic; a production-data sanity check is deferred to P1.

## What this memo does NOT say

- It does NOT propose a deploy change. The methodology stack (#11/#13/#14) still has not authorized a production decision; this memo doesn't change that.
- It does NOT falsify the strategic-question hypothesis. It shows the hypothesis is not directionally supported on this sample with this proxy.
- It does NOT use computed `park_factor` (deferred P1) or `batter_skill_quartile` (deferred P1).
- It does NOT recompute features or re-run the model under a counterfactual where park-environment features are removed.

## What this memo establishes

- A canonical artifact at `data/validation/realized_picks_canonical_2026-05-05.parquet` with 5 new env columns and `is_park_driven` derived. Reproducible from the script's `--summary` flag against a synced production picks dir.
- Reproduction command (with current production picks rsynced to `/tmp/realized_picks_input/` and the freshest `pa_2026.parquet` at `/tmp/pa_2026_fresh.parquet`):
  ```
  UV_CACHE_DIR=/tmp/uv-cache uv run --extra model python scripts/canonicalize_realized_picks.py \
    --picks-dir /tmp/realized_picks_input \
    --pa-path /tmp/pa_2026_fresh.parquet \
    --output data/validation/realized_picks_canonical_2026-05-05.parquet \
    --summary
  ```
- The doubleheader regression covered by `tests/scripts/test_canonicalize_realized_picks.py::test_doubleheader_env_selected_by_game_pk`: same `(batter_id, date)` at two `game_pk`s, env joined by `pick.game_pk`. Catches the regression where a batter_id/date join would silently pull the sibling game's env on doubleheader days.
- A first-pass directional answer on the strategic-question hypothesis: **not supported on this sample with this proxy**. Future cycles can either tighten the proxy (P1) or extend the sample.

## What's next (recommendations, not commitments)

1. **P1: add `batter_skill_quartile`** to the canonical artifact. Career-PA-weighted hit rate quartile, computed at pick time from the PA frame. Combined with `is_park_driven`, this enables the full strategic-question cut (low-skill × park-driven specifically).

2. **P1: replicate computed `park_factor`** from `src/bts/features/compute.py:394-406` as a column on the canonical artifact. Use a continuous park_factor cut alongside the boolean is_park_driven; the boolean is a coarse threshold and may miss medium-park-leverage venues.

3. **Re-run #12 phase 2 + #12 phase 3 in 30 days.** post_bpm n at 30+ would let the strict-current verdict move from underpowered to a real signal, and the env cut would have power on that stratum (currently has n=2 park_driven).

4. **Track the DD park_driven cell as more post-bpm picks resolve.** The post_pooled_mdp_pre_bpm window is closed, but the post_bpm regime continues to grow. Watching whether DD park_driven picks land near the +22.7pp not_park_driven gap or at a different rate is the cheapest way to distinguish the two readings of the DD-slot signal. We do not forecast a specific timeline — depends on schedule weather and Coors visits.

5. **Investigate DD-selection mechanics.** If reading 1 (structural DD issue independent of park) holds with more data, the next investigation is the strategy's DD candidate selection — lineup-position dependence, pitcher-matchup interactions, MDP-policy bin assignment for rank-2 candidates. This shifts the frame from a calibration question to a strategy question.
