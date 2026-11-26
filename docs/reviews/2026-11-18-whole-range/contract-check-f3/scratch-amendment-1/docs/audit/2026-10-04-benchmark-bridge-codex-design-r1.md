## Verdict

**BLOCK.** The bounded bridge is feasible, but its binding, count-only contrast, attribution and exposure ordering are unsound as written. F1–F7 need the fixes below before implementation; F8–F9 fix the statistical and reporting contract before it freezes. None requires a point-in-time platform, a policy replay, a new candidate, or another historical experiment campaign.

Reviewed `main` HEAD `eacde2d07d053be71c4bcfbe2d0a1d8d9758ea90`; design sha256 `e9bfce39b73e5d9f34d21428454f591a2b2a08a1e6d6d92e53f56b7d42a9b955`. Evidence is code/docs/history inspection and in-memory synthetic Python probes against the actual functions. No `data/` reads, network, SSH, or tracked-file edits. The claimed 105-date census, actual pickle availability, live configuration and any real reproduction rate remain **unverified**.

Feasibility established: replacing `_fetch_game_slots` supplies slots to `predict`; the saved serving artifact includes the required single model under `_model`; both walk-forward modes accept a sufficiently large integer `top_n`; the PA log exists; the AUC and proper-score helpers are reusable. The oracle labels for A26/B26 are appropriate. Excluding resumed PA is appropriate for the specified target. Date-weighted within-slate AUC and paired whole-date resampling are usable with F8's explicit estimand and limits. The binary baseball outcome is distinct from BTS Pass settlement, which the design already acknowledges.

## Findings

### F1 — P1: Exact score equality rejects genuine selections; one matching selection does not bind a full slate

**Location:** `docs/superpowers/specs/2026-10-04-benchmark-bridge-design.md:20`.

`save_slate` uses pandas `to_json` with its default decimal precision (`src/bts/slate.py:53–56`). Picks retain the original float (`src/bts/picks.py:271`, `:313`); decision candidates also retain it (`src/bts/daily_decision.py:32–33`, `:103`). A native synthetic call saved `0.7654321098765432` as `0.7654321099`: exact equality was false, despite using the same input row. A floating-point-noise reproduction threshold also needs to allow this serialization error.

Conversely, another attempt can preserve the same primary score while changing other candidates, lineup states or game information. `bts_slate_v1` has no attempt/model identity. `run_and_pick` saves the slate **before** selection, which can reuse an existing locked pick (`src/bts/orchestrator.py:276–292`; `src/bts/strategy.py:479–494`). Matching one row therefore proves selection consistency, not that the file produced the final action. The pick-file/decision precedence is also unspecified, and research-only dates may have no final selected candidate.

**Smallest fix:** compare probabilities after the exact slate serialization, normalize identifiers, use decision-first precedence with conflicts recorded, and call the cohort **selection-consistent**, not bound. Preserve attempt/time provenance as unknown unless an existing witness establishes it. Give research-only/no-selection dates their own stratum; do not infer their model identity from a stale pick. No historical attempt-ID retrofit is needed.

### F2 — P1: A26-count is not count-only, and the following step changes ensemble aggregation too

**Location:** design `:30` and `:58`.

A26 is `1 − product(1 − p_i)` over the PA-level blend means (`src/bts/simulate/backtest_blend.py:556–579`). The proposed arithmetic mean removes within-game probability variation as well as changing N. With PA probabilities `[0.1, 0.5]` and **N_est = N_actual = 2**, the real helper returns `0.55`, while A26-count returns `0.51`. That four-point difference cannot be attributed to PA count.

B26 additionally averages **game probabilities across models**, unlike A26's average-PA-then-product order (`backtest_blend.py:703–715`). Its reliever context is frozen at the latest model-training window (`:676`, `:820–822`), and it has no serving opener adjustment. A26-count→B26 is consequently not a clean starter/reliever-context effect even after fixing the count formula.

**Smallest fix:** use `1 − (1 − A26) ** (N_est / N_actual)`, a geometric no-hit-rate normalization that exactly preserves A26 when the counts agree. Define N_est from the realized lineup slot, with the existing 4.0 fallback. Label the next transition context **plus ensemble/aggregation convention**, not context alone. If an isolated context effect is wanted, hold the aggregation convention fixed in a paired rescoring; it is not required for the minimal descriptive bridge. This count diagnostic needs A26's game score and `n_pas`, not the full PA log; remove that log unless another declared output needs it.

### F3 — P1: C-vs-D parity cannot turn B26-vs-C into an input effect

**Location:** design `:26–29`, `:40`, `:58` and `:96`.

B26 uses a fixed walk-forward recipe that refits every seven test dates and appends earlier 2026 outcomes (`backtest_blend.py:801–822`). C-frozen uses coefficients trained through 2025; C-served uses each archived daily artifact. These are different fitted models, training windows and update schedules. Even perfect C-served/D parity leaves B26→C-frozen confounded by coefficients/cadence as well as inputs. C-frozen→C-served also need not isolate daily learning if preprocessing, configuration or calibration differs.

The plan requires separating frozen coefficients from a **frozen daily-learning recipe**, not merely naming evolving production artifacts a fixed recipe (`docs/superpowers/plans/2026-09-14-season-wrap-plan.md:103`). Fixed seed/determinism does not make distinct training populations equivalent. Starting the walk-forward only at 6/11 would also change its training population and seven-date schedule.

**Smallest fix:** pin and name the A/B learning recipe; run its original whole-season calendar and restrict reported comparisons afterward. Describe C-served as the archived production sequence, stratified by recoverable recipe provenance. Explicitly label mixed transitions; withdraw pure-input and pure-learning attribution. An optional B-frozen rescoring using the *same* C-frozen single model/blend would isolate a useful input contrast without another training run. If omitted, declare that decomposition unresolved. Numerical parity is necessary supporting evidence, not sufficient causal identification.

### F4 — P1: A pickle hash and a date filter do not recover the serving information set or identify residual causes

**Location:** design `:26–27` and `:34–40`.

There are several concrete implementation and inference gaps:

- `model_pickle_sha256` is a **top-level** pick field, not `pick.provenance.model_pickle_sha256` (`src/bts/picks.py:234`, `:301–313`). Load `_model` separately and remove it from the dictionary passed as `blend`; saved artifacts contain it (`src/bts/model/predict.py:900–914`). `predict` always needs that single model, including for the common reliever probability (`:719–727`, `:767–770`). A synthetic injected-slot probe changed only that model and moved the same blend's score from `0.7376234157` to `0.8524131713`.
- Feature/env/code provenance matters independently of model bytes. The pick records `model_git_sha`, `feature_env` and its hash (`picks.py:146–154`). Current code/current env are not automatically historical serving. Optional calibration rewrites `p_game_hit` outside `predict` (`src/bts/orchestrator.py:118–145`).
- The slate omits pitcher hand and game context. Weather temperature is a production baseline feature. Opener detection changes the PA split and depends on the historical frame (`predict.py:689–694`, `:737–740`); it cannot be disabled casually or computed on the complete future frame. Post-game pitcher hand can also change the platoon input relative to the live/history fallback.
- `_build_feature_lookups` reproduces the lookup **algorithm**, not historical source availability. Serving uses the frame actually loaded (`predict.py:877–879`, `:916–917`). A final frame filtered by game date can contain later corrections, late-finalized games or resumed PA unavailable at that serve time. M3's mathematical parity test is not certification of those historical inputs. Feature computation also uses a mutable probable-pitcher cache (`src/bts/features/compute.py:45–99`, `:456`), and calls into the external park-drag artifact.
- Without a stored live lookup vector or live slot value, a residual cannot be verified as a lookup difference or post-game-vs-live field difference. Those are hypotheses. A planted classifier test cannot supply the missing historical witness.

**Smallest fix:** specify the artifact adapter, historical-frame boundary, opener behavior and provenance checks. Freeze/hash reconstruction inputs and use an owned cache/output directory. Call C a reconstruction from dated final-source records wherever historical availability is unproved. Treat residual classes as evidenced causes only with a direct witness or a controlled substitution; otherwise inferred/unknown. Report serialization, recipe/env, calibration and opener/context limitations. Keep unexplained residuals rather than building a PIT system.

### F5 — P1: Scheduled start alone is not serving eligibility, and the other metrics include already unavailable games

**Location:** design `:44`, `:48–49` and `:52–55`.

`_fetch_game_slots` includes games across statuses; the slate is written before selection (`predict.py:441–456`; `orchestrator.py:276–292`). Production selection additionally excludes postponed/cancelled/missing and warmup games (`src/bts/strategy.py:509–522`; `src/bts/picks.py:691–760`). The exact cutoff-game exclusion was introduced during this window on 8/30. A future scheduled start with a postponed or warmup status is a concrete counterexample to the proposed rule; native status-helper probes rejected both.

The final raw feed's start time does not establish the scheduled time/status then, and `written_at` is sampled after scoring, before a possibly later lock. Moreover, only rank-1 receives the time mask in this design: Brier, log loss, AUC and calibration include all hit/no_hit rows, including earlier games whose participation/status were already known. Those cannot all be presented as forecast quality on selectable candidates (plan ground rule 1.1). An argmax of D scores also does not reproduce D's archived final action on skip, reuse or research-only dates.

**Smallest fix:** apply a shared reconstructed eligibility mask to the primary forecast metrics, using historical time/status evidence where available. Where it is missing, call the rule a **scheduled-time surrogate** and report unknown eligibility; an all-row table is a separate diagnostic. Keep D's actual primary/action alongside the reranked diagnostic, with skip/research/unknown counts. Do not require DD/policy replay or claim selection parity from score parity.

### F6 — P1: Coverage losses are acknowledged, but the paired/reranking population and missing outcomes are undefined

**Location:** design `:17–19`, `:43–45` and `:52–63`.

A has no row for a nonparticipant. B also drops any batter-game with no PA against its inferred starter (`backtest_blend.py:620–653`), regardless of `top_n`. A native two-participant fixture retained one row and dropped the batter who faced only a reliever. Thus a large `top_n` cannot put every D candidate on every surface. Reporting missingness without fixing the comparison mask permits different pools/denominators to masquerade as paired effects; this was explicitly fixed by a joint mask in M3 (`docs/audit/2026-06-11-m3-serving-staleness.md:101–107`).

The three outcome labels also conflate a genuinely absent PA with an absent/unreadable/incomplete game source. `read_pa_for_bts_scoring` only filters existing rows; it does not establish game completeness or Pass settlement (`src/bts/data/build.py:44–64`). Dropping no_pa rows before ranking would silently replace a void winner with another batter. Separate nonvoid rate denominators can likewise create an unpaired top-1 difference.

**Smallest fix:** predefine per-pair identical scoreable/eligible candidate masks, preserve native-surface winners separately, and rank before inspecting labels. Define paired top-1 on dates where both chosen rows have known binary outcomes; count unilateral void/unknown days separately without reselection. Add outcome `unknown`, requiring complete game coverage before asserting no_pa. Retain the intended hit/no_hit target but name it baseball-event scoring, not BTS settlement; no_pa is not the exhaustive BTS Pass definition. Publish date and row denominators, overlap and exclusions. Define the missing-fraction denominator from an inventory of regular-season ET game dates; do not treat 105 as a verified fixed count.

### F7 — P1: X-21 gates too late and understates the new exposure

**Location:** design `:73–78` and `:82–86`.

The proposed runner performs A/B before registration gates the final outcome join. But those functions already compute `actual_hit`, return it in every profile and write `is_hit` in the PA log (`backtest_blend.py:523–538`, `:578–581`, `:643–646`, `:864–868`). A native mocked-training walk-forward returned `actual_hit=[1]` and PA-log labels `[0,1]` without any subsequent join. Training also appends earlier 2026 outcomes. The gate cannot be placed after these operations.

X-01 and X-09 cover production selected slots/scorecards, not all outcomes of all full-slate candidates. X-12 permits W1.2 diagnostic use but initially had unjoined research outcomes. The whole-season A/B API additionally emits labels for dates/candidates outside the proposed 105-date scope. The sentence “not new outcome dates” is neither a sufficient exposure argument nor proof that this read was already consumed.

**Smallest fix:** push X-21 before any outcome-bearing bridge execution or inspection, including A/B generation. Identify the new candidate-level exposure and the entire 2026 PA training/participation basis. Limit disclosed diagnostic outputs to the registered dates/rows, or explicitly register any broader outputs before execution. D3 already permits the W1.2 descriptive bridge (`docs/audit/2026-09-22-exposure-register.md:31`); no fresh candidate-testing permission follows. A new W4 registration alone cannot turn a bridge-motivated 2026 idea into prospective evidence: preserve D3's 2027 forward-validation requirement (`:63`).

### F8 — P2: The statistical recipe needs explicit weights, paired resampling and limits on reliability intervals

**Location:** design `:54–57` and `:62`.

Equal-date mean AUC is a defensible within-slate estimand, not an error; specify omission/counting of dates without both outcome classes. Reuse `_rank_auc` per date, not the health check's pooled implementation (`src/bts/health/slate_auc.py:154–165`). Brier/log-loss helpers average rows; changing to an average of daily means changes the estimand when slate sizes differ. The interval must recompute the declared statistic on jointly resampled date clusters, preserving repeated-date multiplicities and each pair's frozen population.

`proper_scoring.reliability_table` supplies candidate-bin Wilson intervals (`src/bts/validate/proper_scoring.py:98–100`, `:114–128`). Those are not date-cluster intervals and ignore same-day/shared-game dependence. About 100 available dates also does not guarantee 100 paired nonvoid dates or enough discordance; a narrow/degenerate conditional interval is not evidence of equivalence. Whole-date resampling handles within-date dependence, not dependence across successive dates, fitting uncertainty or selection bias.

**Smallest fix:** freeze weights and percentile resampling details; expose effective dates, candidate counts and discordance. Omit the helper's Wilson columns or replace them with date-cluster intervals. Label intervals conditional on fitted artifacts/support and exchangeable dates. Do not add a broad inferential framework or a hypothesis-test gate.

### F9 — P2: A26/D measures a restricted 2026 diagnostic gap, not a numerical decomposition of the README-to-live headline

**Location:** design `:8–12` and `:17–20`.

A26 reuses the actual-PA *method* on 2026 and restricts ranking to archived D candidates. It does not reconstruct the README's historical-season evaluation, its original artifact/configuration, or its original oracle participant pool. D's reranked forecast sample is also not the full-season delivered/contest population underlying 68–72%. The stated window limits do not cure the leading claim that each step explains that headline gap.

**Smallest fix:** describe a within-window score/decision comparison that diagnoses mechanisms. Carry the historical headline as noncomparable context; do not report a percent of the historical gap explained. This answers the bounded W1.2 question without re-running the old headline or expanding the missing historical slate population.

## Verbatim edits

Apply these replacements to the design before building. They are specification edits, not authorization to edit production, read box data, push, or execute the analysis.

1. **Replace §1's opening paragraph (line 8):**

> The README's historical backtest headline (top-1 about 86%) and the 2026 live result (about 68–72%) are not comparable estimates (rule 1.3). This bridge compares scores and reranked primaries within the reconstructable 2026 slate window. It diagnoses mechanisms behind the information-set mismatch; it does not reproduce the historical headline or assign a fraction of the headline-to-live gap to each step. Mixed transitions remain descriptive rather than causal decompositions.

2. **Replace §2's date and binding bullets:**

> - **Dates:** inventory the archived `bts_slate_v1` files dated 2026-06-11 through 2026-09-27; 105 is the provisional census, to be verified. Publish the denominator of all 2026 regular-season ET game dates and the missing-date fraction. The live-forward top-10 is frozen-code rescoring and is not substituted for served scores.
> - **Selection consistency:** use `decision.json.primary` where an authoritative decision exists; otherwise use the pick-file primary. Record conflicts between sources rather than choosing a favorable match. Match normalized `(batter_id, game_pk)` and the selected probability after applying the slate writer's exact pandas JSON serialization. This establishes selection consistency only; it does not bind the full slate to a prediction attempt or final lock. Record existing attempt/time witnesses when available, otherwise provenance is unknown. The primary selected-date analysis uses selection-consistent dates, with all available dates as a separately labelled sensitivity. Research-only/no-selection dates are a separate stratum, and stale selections do not supply their model provenance.

3. **Replace the C-served, C-frozen and A26-count table rows; add the paragraph below the table:**

> | **C-served, archived-coefficient reconstruction** | Load the archived `blend_<date>.pkl` only when its sha256 matches the selected pick's top-level `model_pickle_sha256` and its origin is consistent with the archived tier/provenance. Extract `_model` as the single model and pass the remaining members as `blend`. Check historical code, feature/env and calibration provenance. Build ordered lookups and opener history from the declared pre-date frame. Preserve the slate's lineup, pitcher, projected state and candidate identity; declare the source for every additional slot field, including pitcher hand and weather. Final-source reconstruction is not proof of historical input availability. Missing binding/provenance is reported, never replaced silently by a retrained model. | Inject slots into `predict()` without network or `run_pipeline()` refresh |
> | **C-frozen, fixed coefficients** | Train the single model and blend once on the declared 2019–2025 training pool, with deterministic settings fixed before importing training configuration; use the same reconstruction adapter and inputs as C-served wherever their provenance permits. | Same scoring path |
> | **A26-count, geometric count normalization (oracle)** | For `N_actual > 0`, `1 − (1 − A26) ** (N_est / N_actual)`. N_est uses the realized lineup slot and the existing lineup-to-PA map, default 4.0 for a missing slot. This holds the geometric per-PA no-hit rate fixed and equals A26 when the counts agree. | Derived from A26 game score, `n_pas` and the realized lineup slot; no PA log required |

> A26/B26 use one pinned daily-learning recipe, including code, feature/env configuration, seed, model set, training start and seven-test-date retraining calendar. Run the original full 2026 calendar and restrict reported outputs afterward. Record/reuse identical fitted models for their paired comparison. C-served is the archived production sequence, not automatically a frozen recipe; report recoverable recipe strata. A26-count→B26 changes matchup/context and ensemble aggregation conventions. B26→C-frozen changes inputs and coefficients/training cadence. C-frozen→C-served changes fitted models and may also change recipe/calibration. These are composite contrasts unless the other components are explicitly held fixed. A same-artifact B-frozen/C-frozen rescoring is optional; without it, the isolated input effect is unresolved. Neither recipe is a pristine 2026 holdout.

4. **Replace the reproduction-test text (lines 34–40):**

> **The reproduction test (C-served vs D).** Compare every reconstructable row on selection-consistent dates, using a predeclared absolute tolerance of 1e-9 for serialized game scores and separately reporting the maximum/quantiles of residuals and winner agreement. Score parity establishes numerical consistency on those rows; it does not prove attempt identity, historical availability, selection parity or causal attribution. Report missing artifact/provenance separately from numeric residuals. Residual explanations may include recipe/env/calibration, opener/PA-split, pitcher-hand or game context, lookup/source revision, and serialization. Call a cause verified only when supported by an archived live witness or a controlled substitution; otherwise label it inferred or unexplained. Projected/confirmed state alone is not a verified numeric cause. If C-served fails to reproduce D, attribution across that boundary remains unresolved; even successful parity does not isolate the composite B-vs-C transitions.

5. **Replace §§4–5:**

> **Outcomes.** Resolve `(date, batter_id, game_pk)` through `read_pa_for_bts_scoring`, excluding resumed PA. Labels are hit, no_hit, no_pa, or unknown. Assert no_pa only when the relevant game's PA source is complete; missing/unreadable/incomplete sources are unknown. Verify suspended-game exclusion support or mark the affected result unknown. Hit/no_hit are baseball-event labels, not BTS settlement labels. Report selected no_pa separately without treating it as a miss, and do not describe it as the exhaustive BTS Pass category. Rank before joining outcomes and never replace a winner because it is no_pa or unknown.
>
> **Eligibility and decisions.** Use a shared reconstructed eligibility mask for primary forecast metrics and reranking. Recover scheduled time and detailed candidate status at the observation time where existing evidence permits, including unavailable/warmup/cutoff exclusions and the applicable serving era. When only final raw start time and slate `written_at` exist, label `scheduled_start > written_at + 5 minutes` a scheduled-time surrogate; report eligibility as unverified and show that surrogate diagnostic separately. All-row scoring is a separate diagnostic that can include already unavailable games. Preserve the archived D primary and action alongside each surface's diagnostic argmax; count skip, research-only and unknown actions. Compare primaries only; DD selection and policy trajectories remain out of scope. Break exact score ties by archived D row order across surfaces.

6. **Add to §6 and qualify its adjacent comparisons:**

> For each pair, freeze the identical candidate pool with finite scores in both arms and the shared eligibility rule before outcome inspection. Report intersection/union overlap and exclusions by surface and reason. Keep native-surface winners and coverage separately; ranking on a common reduced pool is a conditional diagnostic, not the archived production action. AUC and proper scores use the same known hit/no_hit rows in both arms. Paired top-1 differences use the same dates where both previously selected winners have known hit/no_hit labels; report unilateral no_pa/unknown exclusions and never reselect. Publish row and date denominators for every table.
>
> AUC is the equal-date mean of tie-aware within-date AUCs; omit and count dates lacking either class. Brier, log loss and mean stated−realized residual are candidate-row-weighted; paired differences use the identical rows. Resample whole eligible dates jointly for both arms, with replacement, 10,000 times at a recorded fixed seed, preserving repeated-date multiplicities; recompute the declared statistics and take the 2.5/97.5 percentiles. Reliability tables use fixed bins on [0,1] and omit the helper's candidate-level Wilson intervals. Report effective dates and discordant-outcome counts. These intervals are conditional on the frozen artifacts/support and exchangeable date clusters; they do not include cross-date dependence, fitting uncertainty or selection bias. No equivalence or causal attribution follows from a narrow interval. Adjacent contrasts have the composite meanings stated in §3, and changing pairwise support prevents adding their effects into a single decomposition.

7. **Replace §8 and the registration/run ordering:**

> X-21 must be predeclared and pushed before any outcome-bearing bridge execution or inspection, including A/B generation, fitting, PA logs, joins and metric tables. It records the analysis recipe, full 2026 PA training/participation basis, registered slate dates/candidate identities, outcome definitions, support/eligibility rules, metrics and permitted descriptive outputs. Whole-season walk-forward is needed for the original fitting calendar, but disclosed diagnostic outputs are restricted to registered rows/dates; broader outputs require explicit prior scope. X-01/X-09 cover production selected-slot reads, not all newly examined slate-candidate outcomes. X-12 permits the research-only W1.2 diagnostic stratum. This is a new registered candidate-level descriptive exposure under D3, not a claim that all these outcomes were already consumed. Any model/policy candidate motivated by it requires its own registration and prospective 2027 validation; this bridge supplies no 2026 candidate test. Results may inform W1.3 and the decision memo within those limits.
>
> Before the transient analysis unit runs, verify X-21's published scope and freeze/hash its input manifest. Then generate the original-calendar A/B outputs, train C-frozen, score the declared rows, resolve outcomes and produce metrics. Use only owned caches and outputs under the run directory; isolate `_build_probable_pitcher_lookup` and external-artifact dependencies from mutable production caches. Do not refresh feeds or invoke live serving. Tests cover JSON probability precision, same-selection/different-slate ambiguity, `_model` extraction, count identity at N_est=N_actual, differing coefficients, opener/context/provenance residuals with and without witnesses, reliever-only participation, void/unknown winners, shared comparison masks, eligibility surrogates and repeated-date bootstrap weights. The memo reports reproduction first, then labelled contrasts, coverage/denominators, actual-action counts and the missing fraction.

8. **Add to §10:**

> Missing attempt identity, live input/status witnesses or recipe provenance remains missing. Final-source reconstruction and scheduled-time-surrogate metrics are diagnostics rather than demonstrated point-in-time forecasts. Composite transitions do not identify an isolated information-set effect, and pairwise support changes prevent telescoping the contrasts. The 105-date count and reconstructable-artifact coverage are verified at run time. The historical README/live headline remains noncomparable context.
