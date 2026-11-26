# Phase 0g — matchup-embedding ablation: RESULT (2026-06-15)

**Question (Codex-designed, Codex-reviewed):** Does a learned **batter × starting-pitcher identity interaction** add top-pick *decision* lift beyond (a) the production-proxy probability and (b) batter/pitcher ID quality marginals? This was the one same-data *representation* the resolution audit had not tested — every prior screen used rolling-aggregate features; this one uses raw identity in a factorization machine (collaborative-filtering style), the strongest argument for "trees can't pool across sparse matchup cells, an embedding can."

**Verdict (Codex-endorsed, scoped):** *For this dataset, target, offset, and daily top-pick evaluation, stop pursuing low-rank batter-ID × starting-pitcher-ID matchup embeddings; they show no decision-useful residual signal. This does NOT rule out smaller, non-low-rank, or pitch/handedness/arsenal-level matchup effects.*

## Design
- **Grain:** one row per batter-GAME, keyed `(batter_id, game_pk)` (doubleheaders kept separate). Pair = batter vs *primary opposing pitcher faced* (most-PA pitcher vs the batter's side).
- **Model (factorization machine, game level):** `logit(p_new) = OFFSET + ba[batter] + pa[pitcher] + <U[batter],V[pitcher]>`. OFFSET = `logit(GBM_pa game p)`, **fixed at coefficient 1**, so the trained terms fit only *residual* beyond the current model.
- **Ablation:** (1) offset only → (2) +ID bias marginals → (3) +dot-product interaction. **Decisive contrast = (3) vs (2):** does the interaction add beyond marginals?
- **Eval (decision-relevant, not global AUC):** OOF daily top-1/top-2/top-decile + switched-pick lift, paired bootstrap by DATE. Walk-forward: offset GBM trained season<s; embeddings trained on game rows [2020, s-1]; test s∈2021-25. Cold-start ids → zero (fall back to offset).
- **Two positive controls** (certify machinery BEFORE trusting the null):
  - `ctl` — interaction injected at game level (std 0.6 logit) → certifies optimizer/ID **plumbing**.
  - `ctl2` — interaction injected **only into PAs vs the starter** (std 0.8), then aggregated to game ≥1-hit → certifies **power to recover a realistic, diluted starter-matchup effect** (Codex's point: `ctl` alone matches the FM's hypothesis class exactly and is too easy).

**Codex caught a fatal bug in review:** the first version scored the controls against real `game_hit` instead of the synthetic labels — the certification would have been meaningless. Fixed (eval helpers take a label column) before the run that produced these numbers.

## Results (259,491 batter-games, eval folds 2021-25; vs-starter PA frac 0.589)
| arm | dot−bias top-1 | dot−bias switched | dot−bias decile | reading |
|---|---|---|---|---|
| **ctl** (game-level inject) | +0.0373 [+0.001,+0.074] | +0.0414 [+0.001,+0.080] | +0.0386 [+0.032,+0.045] | plumbing ✓ |
| **ctl2** (PA-level diluted) | **+0.1075 [+0.070,+0.145]** | **+0.1181 [+0.077,+0.160]** | +0.0411 [+0.035,+0.048] | **power ✓✓** |
| **real** | −0.0241 [−0.058,+0.008] | −0.0399 [−0.094,+0.015] | −0.0146 [−0.020,−0.009] | **NULL** |

Reg robustness (real, top-1 dot−offset): l2=1e-5 **−0.132**; l2=1e-4 **−0.027**; l2=1e-3 **+0.0066 [−0.019,+0.031] n.s.** — the interaction only stops *hurting* once regularized to ≈0; no setting yields a positive significant decision edge.

## Why this is a trustworthy null (not a power failure, not a bug)
- **Power is certified by `ctl2`**, which runs through the *identical* pipeline and recovers a realistic diluted matchup effect at +0.10–0.12 switched-pick lift. Same code, same dilution, same data structure — injected signal found, real signal absent.
- **The negative real decile is overfit, not a bug:** the controls share the exact code path (sign convention, descending sort, per-fold ID maps, date-bootstrap) and light up strongly positive. A sign/sort/mapping bug would have broken the controls too. The dot term is confidently mis-sorting some games after marginals — classic overfit-with-no-residual-signal.
- At the **pre-specified** `l2=1e-4`, the real top-1 upper CI is only **+0.009**: no evidence of an effect large enough to move the main top-pick decision region by even ~1 percentage point.

## Scope / what this does NOT claim (Codex anti-overclaim)
- It does **not** show "batter-vs-pitcher is noise" in general. It shows no **decision-useful low-rank identity interaction** beyond an already matchup-aware baseline.
- The offset already contains **`platoon_hr`** (handedness/platoon IS modeled — this is a residual-beyond-platoon test, not a refutation of platoon) and **`batter_pitcher_shrunk_hr`** (a hand-crafted *shrunk* batter-vs-pitcher matchup feature). So the learned embedding was tested against a baseline that already mines the matchup angle — and the learned low-rank version adds nothing the shrunk feature doesn't already have. That explains the null mechanistically.
- Power is certified for a **large, stationary, low-rank** effect. Smaller effects, **non-low-rank** effects, or effects that live in **pitch-arsenal / velocity / count-level** mechanisms (not in this feature set) are NOT ruled out — but any matchup effect big enough to break the flat top-pick region would have shown.
- `starter` = primary opposing pitcher by PA count (ex-post; mislabels openers/early hooks in a few % of games → adds noise, conservative). Global median-impute strengthens only the offset baseline → conservative for the hypothesis.

## Bottom line
The last untested same-data *representation* — learned identity embeddings — joins every other screen in the resolution audit as a clean null. The production model is already matchup-aware (platoon + shrunk-BvP features); a flexible learned interaction does not beat it. Within the tested representations, objectives, and model classes, the same-data top-pick predictor search is closed; the only remaining levers are new data modalities or the season-end-blocked decision-layer (Gate-B) work.

Scripts: `scripts/resolution_audit/phase0g_embedding_ablation.py`. Codex reviews: `/tmp/codex-phase0g-out.txt` (design/code), `/tmp/codex-phase0g-result-out.txt` (result). Data: `/tmp/resaudit_phase0g.parquet`.
