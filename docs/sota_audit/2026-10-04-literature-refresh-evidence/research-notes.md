# W3 Literature / SOTA refresh: research notes

Compiled 2026-10-03 by a Claude research subagent for the lead. These are working notes, not the W3 memo. Nothing under `data/` was read, and no ssh or repo edits were made. Repo context came from the plan (`docs/superpowers/plans/2026-09-14-season-wrap-plan.md` §W3), the SOTA tracker, `README.md`/`ARCHITECTURE.md`, and a grep of `src/` for field names.

**Evidence labels used below**
- **[V]**: I opened the primary page/PDF myself and quote or paraphrase it directly.
- **[M]**: bibliographic metadata verified (Crossref / RePEc / PubMed / arXiv listing). Abstract opened only where stated.
- **[S]**: secondary source only (news write-up, search-engine snippet, or a site quoting a paywalled primary).
- **[NV]**: could not verify. The source was blocked, paywalled, or not found.

**Our setting, for applicability judgements.** Tabular data with roughly 20–30 features and about 1M PA rows per season, trained on 2019+ (several million rows in all), with daily refits. There is strong within- and between-season drift. PA probabilities are aggregated to a game-level P(≥1 hit). What matters most is ranking within a daily slate and calibration in the top bins near the MDP's skip/double boundaries (about 0.76–0.84). Global calibration matters less. Production runs on CPU-only Hetzner. 2026 outcomes are reserved under D3, so a candidate motivated by them validates in 2027.

---

## Part 1: The five nominated items

### 1. TabM (Gorishniy, Kotelnikov, Babenko; ICLR 2025)

**Identity [V].** "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling". Authors Yury Gorishniy, Akim Kotelnikov, Artem Babenko. arXiv 2410.24210 (v1 2024-10-31, v3 2025-02-18). The arXiv comment reads "ICLR 2025". https://arxiv.org/abs/2410.24210 · code (Apache-2.0, `pip install tabm`): https://github.com/yandex-research/tabm

**What it claims [V].**
- One TabM network "efficiently imitates an ensemble of MLPs and produces multiple predictions per object". The implicit members are trained simultaneously and share most parameters (BatchEnsemble-style).
- MLP-family models, TabM included, are "stronger and more practical" than attention- or retrieval-based tabular DL. TabM is "the best performance among tabular DL models".
- Benchmark: 46 datasets with train sizes 1.8K–723K. Nine use "domain-aware" splits: the eight TabReD datasets with time-aware splits, plus Microsoft LETOR. All models are tuned per dataset, and GBDTs (XGBoost/LightGBM/CatBoost) are included.
- The README says the largest dataset in the paper had 13M objects. TabM is "relatively slower than MLPs and GBDT". Its paper hyperparameters use k=32 fixed (untuned), n_blocks 1–5, d_block 64–1024, AdamW with lr≈2e-3.
- Most per-split comparisons are in figures I could not extract as text. I therefore do **not** quote a TabM-vs-LightGBM margin on the time-split subset.

**Independent evidence on temporal splits [V].**
- *TabReD* (Rubachev, Kartashev, Gorishniy, Babenko; cited as ICLR 2025 in the TabPFN-3.5 bibliography; arXiv 2406.19380, https://arxiv.org/abs/2406.19380) finds that "evaluation on time-based data splits leads to different methods ranking, compared to evaluation on random splits". It also finds that "simple MLP-like architectures and GBDT show the best results" there.
- *BeyondArena* (Purucker et al., arXiv 2606.30410, 2026-06-29, https://arxiv.org/abs/2606.30410) has 142 datasets with IID, temporal and grouped splits. In its overall leaderboard (Elo on ROC AUC for binary tasks), TabM-default scores 1107, LightGBM-default 991 and LightGBM-tuned 1149. Tuned+ensembled TabM scores 1210 and tuned+ensembled LightGBM 1187. Per-regime Friedman/Wilcoxon (its Test E.5): "Tuned and ensembled neural networks (RealMLP) lead on temporal, grouped, medium, large…" Conflict-of-interest note: RealMLP's author (Holzmüller) is a BeyondArena co-author, which the paper discloses.

**Applicability to us.**
- The data scale is fine; the paper trained up to 13M objects.
- The headline neural gains in every benchmark come with tuning plus ensembling, and the plan forbids a sweep. Even so, the benchmark's default-vs-default gap favours TabM over default LightGBM (1107 vs 991 Elo, all datasets). That is relevant because the tracker describes our LightGBM as running on default hyperparameters.
- None of these benchmarks scores what we care about: within-slate top-1 rank and top-bin calibration after PA→game aggregation. All of them use ROC AUC for binary tasks.
- An MLP's PA probabilities would need a calibration check before the `1−∏(1−p)` aggregation. BeyondArena recommends post-hoc calibration for most models.
- Practical cost: training daily on CPU over several million rows is a real ops cost. Weekly refits or a GPU are likely needed (reasoning only).

**Suggested disposition.** This is the W4 rank-6 candidate (the single model-class challenger), and **my pick between TabM and RealMLP under the plan's frozen-default rule**. On BeyondArena's all-dataset leaderboard, TabM's default beats RealMLP's (1107 vs 1056). It adds only one new hyperparameter (k=32, fixed by the authors) and has a small Apache-2.0 package. *Caveat (reasoning only):* the default-vs-default numbers are all-dataset aggregates. I could not extract defaults on the temporal subset from the figures.

### 2. RealMLP / "Better by Default" (Holzmüller, Grinsztajn, Steinwart; NeurIPS 2024)

**Identity [V].** "Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data". Authors David Holzmüller, Léo Grinsztajn, Ingo Steinwart. arXiv 2407.04491 (v1 2024-07-05, v3 2025-01-15). The arXiv listing names NeurIPS 2024 as the venue. The v3 comment is "mention bug in XGBoost results". https://arxiv.org/abs/2407.04491 · code (Apache-2.0): https://github.com/dholzmueller/pytabkit

**What it claims [V].**
- It introduces RealMLP plus "strong meta-tuned default parameters for GBDTs and RealMLP" (LGBM-TD, XGB-TD, CatBoost-TD, RealMLP-TD). These were tuned on 118 meta-train datasets and tested on 90 disjoint meta-test datasets, all of 1K–500K samples.
- RealMLP "offers a favorable time-accuracy tradeoff compared to other neural baselines and is competitive with GBDTs". A combination of RealMLP and GBDTs with improved defaults "can achieve excellent results without hyperparameter tuning".
- The pytabkit README now recommends HPO plus Caruana ensembling with TabArena search spaces "for best results". So the authors' own best-results recipe is no longer "defaults only".

**Applicability.**
- It has the same benchmark/metric mismatch as TabM: random splits and generic metrics.
- In BeyondArena, RealMLP wins the temporal and large subsets only when tuned and ensembled. Its default Elo is 1056.
- The paper's most cheaply testable artefact for us is **LGBM-TD**, a frozen alternative default set for the model we already run. The plan rules this out with "no LightGBM re-tuning".

**Suggested disposition.** Not selected, because the plan says to pick one and the default-vs-default evidence favours TabM. Keep it as the alternate if the lead allows a tuning budget, since tuned+ensembled RealMLP leads BeyondArena's temporal subset. A lead decision is needed on whether "LGBM-TD as a frozen default" counts as forbidden re-tuning. I'd call it closed under the current plan boundary.

### 3. TabPFN v2 (Hollmann et al.; Nature 2025) and its successors

**Identity [V].** "Accurate predictions on small data with a tabular foundation model". Hollmann, Müller, Purucker, Krishnakumar, Körfer, Hoo, Schirrmeister, Hutter. *Nature* 637, 319–326 (published 2025-01-08). https://doi.org/10.1038/s41586-024-08328-6 (opened via https://www.nature.com/articles/s41586-024-08328-6)

**What it says about limits [V], quoting the paper:**
- Size: it "yields dominant performance for datasets with up to 10,000 samples and 500 features". The benchmarks used datasets with "up to 10,000 samples, 500 features and 10 classes".
- Its limitations section lists three items: "(1) the inference speed … may be slower than highly optimized approaches such as CatBoost; (2) the memory usage … scales linearly with dataset size …; (3) our evaluation focused on datasets with up to 10,000 samples and 500 features; scalability to larger datasets requires further study."
- Regression and classification are both supported ("supports regression tasks, categorical data and missing values"). Classification calibration uses a softmax temperature, T=0.9 by default.
- Time series and drift: the main text only names "specialized priors to handle data types such as time series" as future work. It also cites "Drift-resilient TabPFN: In-context learning temporal distribution shifts on tabular data" (Helli et al., NeurIPS 2024) [M, citation seen in the Nature reference list only; paper not opened]. Its single-sample latency: "For a dataset with 10,000 rows and 10 columns, our model requires 0.2 s (0.6 s without GPU)" per prediction, and it "is not optimized for real-time inference".

**Successors, which make "v2" stale [V].**
- TabPFN-2.5 (arXiv 2511.08667, 2025-11-11) handles up to 50,000 rows and 2,000 features.
- TabPFN-3 (arXiv 2605.13986, 2026-05-13) handles up to 1M training rows and claims gains "on time series". It ships a TabPFN-TS-3 checkpoint.
- TabPFN-3.5 (arXiv 2609.17895, v2 2026-09-22) is the current default and accepts up to 1,000,000 rows. Note the feature-limit mismatch: the README says 20,000 features, but the report's figure, as extracted, says 6,000.
- 3.5 claims to extend to "non-i.i.d. data with temporal or grouped splits". Its own BeyondArena result is narrower: "Tuned and ensembled MLPs retain the highest performance on grouped, temporal, and large datasets, but TabPFN-3.5 substantially narrows these gaps. The temporal … subset … is confounded with size."
- The BeyondArena paper itself concludes that TFMs (TabPFN-2.6 and TabICLv2 at in-context defaults) "fail to compete with traditional models on non-IID (temporal and grouped), large-scale…" data.
- **Licence [V]** (https://github.com/PriorLabs/TabPFN README; TabPFN-3.5 report §4):
  - TabPFN-2 weights are "Apache 2.0 with an additional attribution requirement".
  - The TabPFN-2.5, 2.6, 3 and 3.5 weights are "non-commercial". The 3.5 licence bars "commercial or production purposes … including … using model outputs as inputs to internal commercial decision-making".
  - Newer checkpoints need a Prior Labs login/token.

**Applicability.**
- PA level: not applicable. Several million rows exceed even 3.5's 1M-row limit.
- The plan's "compact game-level residual/reranking" idea is about 45K batter-games per season (my estimate, reasoning only). That fits 2.5/3/3.5 by size, but only v2's 10K-row-limited weights are licence-clean.
- Using 3.5 in a daily pick bot with prize money is plausibly "production" under its licence. I can't judge that legally; flag it for the owner.
- The benchmarks that cover temporal data say TFMs lag tuned conventional models there.

**Suggested disposition.** Not applicable at PA level, because of size and the evidence on temporal data. Park the game-level residual idea, as the plan already says, until W1.2 measures residual headroom. If it is ever run, treat it as a research-only evaluation, and settle the licence question before any production use.

### 4. TabArena (Erickson et al.; NeurIPS 2025 Datasets & Benchmarks)

**Identity [V].** "TabArena: A Living Benchmark for Machine Learning on Tabular Data". Erickson, Purucker, Tschalzev, Holzmüller, Desai, Salinas, Hutter. arXiv 2506.16791 (v4 2025-11-03). Comment: "Accepted (spotlight) at NeurIPS 2025 Datasets and Benchmarks Track." https://arxiv.org/abs/2506.16791 · leaderboard https://tabarena.ai

**What it benchmarks [V].**
- Scope: "Tabular classification and regression for independent and identically distributed (IID) data, spanning the small to medium data regime."
- It explicitly leaves for future work "non-IID data (e.g., temporal dependencies, subject groups, or distribution shifts)", very small data, and "large data".
- Inclusion criteria: "The dataset is IID, that is, a random split is appropriate", with 500 to 250,000 training samples.
- Splits: repeated outer CV, either 10×3-fold or 3×3-fold depending on size, stratified for classification.
- Metric: Elo computed from "ROC AUC for binary classification, log-loss for multiclass…, RMSE for regression".
- Findings: GBDTs are "still strong contenders", deep learning "caught up under larger time budgets with ensembling", foundation models "excel on smaller datasets", and cross-model ensembles advance SOTA but some neural models are "overrepresented … due to validation set overfitting".

**How temporal splits are handled [V].** They aren't; temporal data is excluded by design. The same team's *BeyondArena* (arXiv 2606.30410, details under item 1) is the extension with temporal splits. It distinguishes "temporal tabular tasks" (fit at t, deploy with a refit horizon) from time-series forecasting, which is exactly our framing. Its grouped/temporal setup also enforces non-IID inner validation splits.

**Applicability.** Not evidence for our setting. Our PA table is non-IID and, in TabArena's terms, "large". It can still serve as a methodology and implementation reference (model wrappers, search spaces, version tracking). BeyondArena is the better reference for picking a challenger.

**Suggested disposition.** Closed as evidence: it is IID-only with random splits, has a 250K-row ceiling, and scores binary tasks with ROC AUC. Keep the BeyondArena pointer as methodology context for W4 rank 6.

### 5. Online conformal prediction with decaying step sizes (Angelopoulos, Barber, Bates; ICML 2024)

**Identity [V].** "Online conformal prediction with decaying step sizes". Anastasios N. Angelopoulos, Rina Foygel Barber, Stephen Bates. *Proc. ICML 2024*, PMLR 235:1616–1630. https://proceedings.mlr.press/v235/angelopoulos24a.html · arXiv 2402.01139.

**What it claims [V]** (from the PDF):
- It uses a quantile-tracker update `q_{t+1} = q_t + η_t(1{Y_t∉C_t} − α)` with η_t ∝ t^(−1/2−ε).
- Worst case, for arbitrary sequences: long-run coverage 1−α ± C/T^(1/2−ε).
- Best case, for i.i.d. data: P(Y_T ∈ C_T) → 1−α, so coverage is "close to the desired level for every time point" when the distribution is stable.
- On drift, the authors say a decaying step "can be advantageous" when the data are stable. If "we detect a sudden distribution shift … we might want to increase the step size". Theorem 2 covers arbitrary step sequences, including resets.

**Applicability.**
- Our label is binary. Prediction sets over {0,1} carry almost no information, and coverage of a set is not a calibrated hit probability. This matches the plan's own boundary.
- A decaying step adapts *more slowly* over time, which runs against our strong drift unless resets are added. Adding resets turns it back into ACI-style fixed or adaptive steps, which tracker #8 already lists.
- It would only become useful if W4 rank 3 builds a count-distribution model for the number of PAs, N. Then interval coverage of N could be monitored with it.

**Suggested disposition.** Not applicable now: binary target, coverage ≠ calibration, and the decaying step is mis-matched to drift. Reopen only if a PA-count model from W4 rank 3 exists and needs a sequential coverage monitor.

---

## Part 2: The plan's search questions

### a. Public Beat the Streak models and write-ups (2026 and earlier serious ones)

1. **Garnett (2026), "Chasing $5.6 Million with Machine Learning: My Approach to MLB's Impossible Hitting Streak"** (Medium, *learning-data*): https://medium.com/learning-data/chasing-5-6-million-with-machine-learning-my-approach-to-mlbs-impossible-hitting-streak-7f888e1b9d00 **[NV]**. The URL was found, but Medium/Cloudflare blocked both my fetch attempts.
   - A search snippet [S] says P@100 = 84% (train 2021–23, test 2025) and 81% (train 2021–22, test 2024). The model publishes a daily top 10 at https://www.xwobiwan.com ("BTS Live: Beat the Streak Predictor", a JS app; I verified the page title and meta description only).
   - **Discrepancy to check:** our `README.md`/`ARCHITECTURE.md` cite "Garnett (2026), P@100=85%, P@500=77%". The tracker names the author "Kevin Garnett". I could not verify either the numbers or the first name against the article.
   - Relevance: it is the external benchmark the repo calls "current SOTA", so it needs a human read before the README keeps citing it. README hygiene is already on the W4 prerequisite list.
2. **Alceo & Henriques** [V]:
   - 2019: "Sports Analytics: Maximizing Precision in Predicting MLB Base Hits" (KDIR/IC3K 2019, SciTePress). https://www.scitepress.org/PublishedPapers/2019/83622/83622.pdf. Best model is an MLP with an "85% correct pick ratio", which is Top-100 precision.
   - 2020 extension: "Beat the Streak: Prediction of MLB Base Hits Using Machine Learning", in *IC3K 2019*, CCIS vol. 1297, Springer. https://doi.org/10.1007/978-3-030-66196-0_6 (author version opened: https://research.unl.pt/ws/portalfiles/portal/45749118/Beat_the_Streak_Prediction_of_MLB_Base_Hits_Using_Machine_Learning.pdf). On the new 2019 season the MLP "achieved an 81% correct pick ratio".
   - Relevance: a game-level academic baseline. Its numbers are not comparable to ours (different splits, no proper scores).
3. **David Pinto, Baseball Musings** [V]: daily BTS probability lists using log5 and a neural net that "puts the most weight on the three-year batter parameter", still running in 2026 (2026-04-01 post: https://www.baseballmusings.com/?p=159511). The method is described only loosely. Relevance: a long-running public forecast feed, but there are no as-of archives on our side.
4. **Sean Nickell, The Breakdown Point** [V]: 2025 scorecard published 2026-02-21 (https://www.thebreakdownpoint.com/p/the-scorecard-2025-beat-the-streak). It reports that the top-20 daily picks averaged ~68% predicted vs ~70% actual, and says the model is being rebuilt for 2026. No method details.
5. **Ryan McKenna, "Beat the Streak: Day Three"** (2015-08-14) [V]: http://www.ryanhmckenna.com/2015/08/beat-streak-day-three.html. This is the earliest serious public MDP formulation found. W(s,g) uses a skip option, assumes the best pick's p is Uniform[0.78, 0.85], always picks until streak > 8, and derives a threshold formula. It reports W(0,183) ≈ 0.000532.
   - Relevance: the same structure as our solver but with no double-down. Its iid-p assumption is exactly the one the 2026-07-13 run-structure finding warns about.
- Also seen but not serious benchmarks: Lucas Kelly's random forest on FanGraphs, 2021 (https://fantasy.fangraphs.com/not-impossible-just-improbable-beat-the-streak-is-back/); several student GitHub repos.
- Note: Eric's own repo `stone-ericm/bts` is public and dominates search results. Don't count it as an external source.

### b. MLB's `probabilityStarter` / the "% chance to hit" field

- **The only official description found [V]**, from the BTS FAQ (https://www.mlb.com/apps/beat-the-streak/frequently-asked-questions): "What is the '% chance to hit' feature about? We provide hit probability estimates from our unique prediction model. These numbers aren't guarantees, of course. Use your own judgment…" No methodology, inputs, target definition or update cadence is published.
- **The field name `probabilityStarter` appears nowhere public.** An exact-phrase web search returned nothing relevant [NV]. In-repo it appears only in `src/bts/leaderboard/static_capture.py` ("MLB's own probabilityStarter model", populated for today and tomorrow). Reading it as "P(hit) conditional on starting" is **speculation** from the name. W2.3 gate (ii) must establish the semantics empirically.
- **The official rules define the target, which matters for semantics [V]** (2026 Official Rules: https://www.mlb.com/apps/beat-the-streak/official-rules):
  - A Hit requires the pick be "credited with a hit … so long as your Pick had at least one (1) official at-bat or one (1) sacrifice fly".
  - A **Pass** applies if the pick has no official AB or SF or doesn't play, if all PAs are BB/HBP/interference/sac bunt, or if the game is suspended before a hit.
  - Activity in the resumed portion of a suspended game is never evaluated.
  - So the contest outcome is close to P(hit | ≥1 AB or SF, pre-suspension). A candidate semantics test is whether MLB's number behaves like that conditional or like an unconditional P(hit).
- Side note [V]: the rules list "Genius Sports Group" among BTS-associated entities. Nothing says who builds the "% chance to hit" model, so don't infer it.
- Relevance: this answers the plan's question only negatively. No published methodology exists, so the W2.3 benchmark can only report *association* unless target semantics are pinned down from the captures themselves. That outcome is already allowed for in plan gate (ii).

### c. Statcast bat-tracking metrics (2026): definitions, coverage, lag

- **Miss distance and swing timing (launched 2026-06-09)** [V], from Mike Petriello, "Which pitches miss bats by the most?" (https://www.mlb.com/news/explaining-the-new-miss-distance-statcast-metric) and the Savant leaderboard text (https://baseballsavant.mlb.com/leaderboard/bat-tracking/swing-timing-miss-distance):
  - Miss distance is "the distance (at moment of closest approach) between the top half of the bat and the ball, in inches." Only the barrel half counts. "Bunt attempts are excluded."
  - Coverage: "available beginning with the second half of the 2023 season", i.e. since the 2023 All-Star Game. The article puts the average miss at about 3 inches and says breaking/offspeed pitches miss by 5–6× more than fastballs.
  - The swing-timing categories are tied-up/centered/flail (centered = barrel within ±4 in horizontally), late/on-time/early (±7 ms), and over/lined-up/under (±2 in vertically). "Perfect" means all three; "flawed" means none.
  - Leaderboards exist for batters as well as pitchers. I verified `type=batter` rows with fields `n_swings`, `competitive_swings`, `miss_distance`, `flails`, `earlys`, `lates`, `unders` and so on.
  - Our pipeline already reads a per-pitch `miss_distance` column (`src/bts/features/swing.py`). The csv-docs page (https://baseballsavant.mlb.com/csv-docs) as fetched does **not** document it, nor `bat_speed` or `swing_length`. It does document `attack_angle`, `attack_direction`, `swing_path_tilt`, `intercept_ball_minus_batter_pos_{x,y}_inches` and `arm_angle`.
- **Swing path, attack angle, ideal attack angle, attack direction (launched 2025-05-20)** [V], from Petriello (https://www.mlb.com/news/new-statcast-swing-metrics-2025):
  - Swing path (tilt) is the bat-path angle "in the last 40 milliseconds prior to contact"; the MLB average is about 32°.
  - Attack angle is the vertical bat direction at impact (or closest approach on misses), averaging about 10°. "Ideal" means 5°–20°. Since the 2023 ASG, ideal-AA swings hit .272/.487 vs .250/.354 for others.
  - Attack direction is the horizontal direction at contact, averaging about 2° pull.
- **Squared-up rate** [V] (https://www.mlb.com/glossary/statcast/squared-up): a swing is "squared-up" if it achieves ≥80% of the maximum possible exit velocity given bat and pitch speed. The rate is reported both per swing and per contact; in April 2024, 25% of swings and 33% of contacts were squared up. Bat speed, swing length and blasts are glossary entries; blast = squared-up% × 100 + bat speed ≥ 164 (search snippet of the MLB glossary [S]).
- **Academic caution** [V]: Powers & Yurko, "Swinging, Fast and Slow: Interpreting variation in baseball swing tracking metrics" (arXiv 2507.01238, 2025-07-01). https://arxiv.org/abs/2507.01238. Swing timing relative to the pitch sets the point at which bat speed and length are measured. Batter intent varies with count and location, so raw swing metrics are confounded by context. They use a hierarchical skew-normal model and IV regression.
- **Availability lag:** I found no official statement of how quickly bat-tracking or miss-distance data reach Savant or the CSV [NV]. Our repo notes that Savant's search index lags same-night games. Measure the lag; don't assume it.
- Relevance to W4 rank 5:
  - Definitions and the coverage era (2023 second half onward) are now officially documented. A batter-side lagged feature has only about 2.5 pre-2026 seasons of history, and the 2026 season is reserved under D3.
  - The registration must fix the swing-conditioned denominator (whiffs with tracked miss distance, bunts excluded, `competitive_swings` vs `n_swings`) and the measured as-of lag.
  - The Powers & Yurko confounding argues for count/location-adjusted versions, or else for treating raw means as noisy.

### d. Calibration under regime shift / recalibration of drifting binary classifiers

1. **Gupta & Ramdas, "Online Platt Scaling with Calibeating"** (ICML 2023; arXiv 2305.00070) [V]: https://arxiv.org/abs/2305.00070.
   - Online Platt scaling (OPS) runs online logistic regression, via Online Newton Step, on the pseudo-feature logit(f(x)), i.e. it maps p to sigmoid(a·logit p + b) online. It "smoothly adapts between i.i.d. and non-i.i.d. settings with distribution drift" "without hyperparameter tuning". Calibeating adds adversarial calibration guarantees, and the same ideas extend to beta scaling.
   - Caveats in the paper: the regret is measured against the best *fixed* Platt map in hindsight. The baselines include windowed Platt scaling (WPS), the obvious simple comparator.
   - Relevance: this is the most directly usable method. It is a two-parameter, regularisable, online version of the plan's rank-4a "identity vs one regularized intercept map". Fixing a = 1 and learning only b gives the intercept-only online variant (my reading; the paper fits both a and b).
2. **Davis, Greevy, Lasko, Walsh, Matheny, "Detection of calibration drift in clinical prediction models to inform model updating"** (J Biomed Inform 112:103611, 2020) [M, abstract opened via PubMed 33157313; https://doi.org/10.1016/j.jbi.2020.103611].
   - Method: "dynamic calibration curves with optimized online stochastic gradient descent" plus "adaptive sliding windows" to detect increasing miscalibration. Alerts point to a recent window suitable for refitting.
   - Relevance: a monitor (tracker area #6/#8), not an alpha lever. It is model-agnostic.
3. **Hu, Tian, Yang, "Optimal Recalibration of an Online Predictor"** (arXiv 2607.19689, 2026-07-22) [V, abstract only]: https://arxiv.org/abs/2607.19689. (ε, ε²)-recalibration for Lipschitz proper losses in about ε⁻³ rounds.
   - Relevance: mostly theoretical. At ε = 0.02 that is roughly 125K rounds, far beyond our ~180 picks per season (my arithmetic). It confirms that OPS-style low-parameter maps are the realistic choice at our sample size.
- Context already in the tracker: Kull et al. 2017 Beta calibration, Venn-Abers, and ACI (Gibbs & Candès 2021) for coverage. Nothing new found that changes the plan's 4a design (identity vs one regularised map, held-out proper-score gate).

### e. Optimal stopping / MDPs with resets and milestone objectives

1. **McKenna (2015)** [V]: see (a). A BTS-specific DP with skip and reset, threshold policy, iid p.
2. **Xu & Mannor, "Probabilistic Goal Markov Decision Processes"** (IJCAI 2011) [V, PDF opened]: https://www.ijcai.org/Proceedings/11/Papers/341.pdf. Maximising P(cumulative reward ≥ target) is NP-hard in general but solvable in pseudo-polynomial time by **state augmentation** with accumulated reward. The optimal policy "may depend on accumulated reward, and randomization does not improve the performance". Relevance: the theoretical licence for our (streak, days, saver, best) state-augmented deterministic tables. Nothing new to implement.
3. **Bouakiz & Kebir, "Target-level criterion in Markov decision processes"** (J Optim Theory Appl 86:1–15, 1995) [M, Crossref only; abstract not opened]: https://doi.org/10.1007/BF02193458. The classic target-level / reachability criterion.
4. **Bergman & Imbrogno, "Surviving a National Football League Survivor Pool"** (Operations Research 65(5):1343–1354, 2017) [M, abstract opened on RePEc]: https://ideas.repec.org/a/inm/oropre/v65y2017i5p1343-1354.html. An elimination contest, structurally a one-loss streak. "Planning only partway through the season yields the highest survival probabilities."
- Relevance: no new method. The real open issue in our MDP is misspecified transition structure: realized run structure vs iid day types (`docs/audit/2026-07-13-…`). Nothing found addresses BTS run structure. Robust/DR-MDP (Iyengar 2005; Wiesemann 2013) is already in tracker #1.
- **One observation for the lead (reasoning only):** the rules award the $10,000 Top Streak Prize to the *highest* streak ≥ 20 among all entrants, split on ties. So the prize-relevant late-season objective is relative, P(our best ≥ the field's best), not E[season-best]. In practice the field's leader is far above 20 (the record is 51), so the tail policy's E[season-best] reads better as a personal objective than a prize objective. Mention this in the D-memo only if objectives are being revisited.

### f. Forecast combination with a public benchmark forecast

1. **Ranjan & Gneiting, "Combining Probability Forecasts"** (JRSS-B 72:71–91, 2010) [M, abstract opened via Crossref]: https://doi.org/10.1111/j.1467-9868.2009.00726.x.
   - "Any non-trivial weighted average of two or more distinct, calibrated probability forecasts is necessarily uncalibrated and lacks sharpness", so linear pooling needs recalibration. They propose the **beta-transformed linear pool**.
   - Their case study combines a statistical model with **National Weather Service** probability-of-precipitation forecasts, i.e. our own model plus an official public forecast, the closest analogue to ours plus MLB's "% chance to hit".
   - Relevance: this is the natural single combination rule for W4 rank 1. Never average p_ours and p_MLB linearly without recalibrating.
2. **Satopää, Baron, Foster, Mellers, Tetlock, Ungar, "Combining multiple probability predictions using a simple logit model"** (IJF 30:344–356, 2014) [M, Crossref metadata; abstract seen only via search snippet [S]]: https://doi.org/10.1016/j.ijforecast.2013.09.009. A one-parameter logit-space aggregator (geometric mean of odds plus extremising). Relevance: a logit-linear combination with one or two parameters is the low-variance alternative to the beta pool.
3. **Smith & Wallis, "A Simple Explanation of the Forecast Combination Puzzle"** (Oxford Bull Econ Stat 71:331–355, 2009) [M, abstract opened via Crossref]: https://doi.org/10.1111/j.1468-0084.2008.00541.x. Simple combinations beat estimated-weight combinations because of "finite-sample error in estimating the combining weights". Relevance: this supports the plan's "no blend/gate search on this window", with one rule fixed in advance.
4. **Harvey, Leybourne, Newbold, "Tests for Forecast Encompassing"** (JBES 16:254–259, 1998) [M, Crossref metadata only; content from general knowledge, not verified here]: https://doi.org/10.1080/07350015.1998.10524759. This is the standard test of whether forecast B carries information not in A, i.e. W2.3's "residual information" gate. For binary outcomes the analogue is a logistic regression of the outcome on logit(p_ours) and logit(p_MLB).
5. **Wang, Hyndman, Li, Kang, "Forecast combinations: An over 50-year review"** (IJF 39:1518–1547, 2023) [V abstract; arXiv 2205.04216]: https://arxiv.org/abs/2205.04216. A survey covering combination of probabilistic forecasts and time-varying weights. Background only.

### g. 2026 ball drag / carry: literature and MLB statements

1. **MLB Drag Dashboard** [V], from David Adler, "MLB baseball drag data now available" (2022-05-04): https://www.mlb.com/news/mlb-drag-dashboard-public-on-baseball-savant · dashboard https://baseballsavant.mlb.com/drag-dashboard. It shows the daily average drag coefficient of four-seam fastballs back to 2016, using the method of MLB's Home Run Committee scientists (Nathan/Kagan lineage).
   - Rule of thumb: −0.01 Cd ≈ +5 ft at 100 mph exit velocity.
   - "The bigger variation is from baseball to baseball within an individual season."
2. **MLB statement, 2025** [S], from Yahoo Sports (2025-06-13), citing The Athletic: https://sports.yahoo.com/mlb/article/mlb-acknowledges-increased-drag-on-baseballs-has-led-to-fly-balls-traveling-4-feet-less-this-year-181553407.html. Spokesperson Glen Caplin: "We are aware of an increase in average drag this season … There has been no change to the manufacturing, storage or handling of baseballs this year, and all baseballs remain within specifications." Fly balls were traveling about 4 ft less.
3. **MLB statement, 2026 (June drag drop)** [S], from Bleacher Report (2026-07-01) citing The Athletic (Sarris and Drellich; primary paywalled, [NV]): https://bleacherreport.com/articles/25449383-mlb-addresses-issue-game-baseballs-amid-tarik-skubal-comments-rise-hrs. MLB said: "we are aware of the recent reduction in drag. To be clear, there has been no change in the materials or manufacturing process"; "Rawlings and our scientists do not see any evidence to date that the yellow staining is related to this change in drag"; and variation is expected "throughout the season and between seasons". HR/PA was 2.8% before June and 3.4% in June.
4. **Independent 2026 analyses:**
   - Shaun Newkirk (Substack, 2026-06-24) [V]: https://theanalyticssay.substack.com/p/the-juiced-ball-is-probably-maybe. Using the Savant Drag Dashboard, drag was "touching near all time highs at the start of the year before just falling off a cliff in May and June".
   - Ballpark Pal (2026-04-09) [V]: https://www.ballparkpal.com/the-state-of-home-runs-so-far-in-2026.html. Its *early* read attributed low HRs to pitcher behaviour, not the ball, which was later overtaken by the June drop.
   - Patrick Dubuque, Baseball Prospectus, "The Erratic Flight of the Modern Baseball" (2026-07-07) [NV]: paywalled, title and subtitle only. https://www.baseballprospectus.com/news/article/108482/the-erratic-flight-of-the-modern-baseball/
- Relevance: official sources confirm both large within-season league-wide drag regime shifts (2025 high, June 2026 sharp drop) and large ball-to-ball variance. That supports keeping drag as *regime observability* (our `park_drag_delta` shadow). It adds no new alpha evidence: our 2026 screen was NULL, and drag mainly moves fly-ball carry and HR, while BTS hits are mostly singles (reasoning only).
- Unscreened variant (reasoning only): a league-wide daily Cd from the Dashboard in place of the venue rolling Cd. Any version is motivated by 2026, so under D3 it validates in 2027.

### h. Streak or pool contests (any sport) analysed with a public-consensus feed

1. **Clair & Letscher, "Optimal Strategies for Sports Betting Pools"** (Operations Research 55(6):1163–1177, 2007) [M, abstract opened on RePEc]: https://ideas.repec.org/a/inm/oropre/v55y2007i6p1163-1177.html. "Teams that are popularly perceived as 'favorites' gain a disproportionate share of entries." Modelling participant behaviour beats maximising correct picks "often by orders of magnitude" in pools with *relative* payoffs.
2. **Haugh & Singal, "How to Play Fantasy Sports Strategically (and Win)"** (Management Science 67(1):72–92, 2021) [M, abstract opened on RePEc]: https://ideas.repec.org/a/inm/ormnsc/v67y2021i1p72-92.html. Uses a Dirichlet-multinomial model of opponents' selections, fitted by regression, which is the closest analogue to modelling a `numberSelections` popularity feed.
3. **Simmons, Nelson, Galak, Frederick, "Intuitive Biases in Choice vs. Estimation: Implications for the Wisdom of Crowds"** (SSRN 2010, https://doi.org/10.2139/ssrn.1553935; journal version J Consumer Research 2011, [M] not opened) [M, abstract via Crossref]. In a season-long NFL experiment, "faulty intuitions led the crowd to predict 'favorites' more than 'underdogs' … even when bettors knew that the spreads disadvantaged favorites", and "the bias increased over time".
4. **Bergman & Imbrogno 2017** (see e) and **Decary, Bergman, Cardonha, Imbrogno, Lodi, "The Madness of Multiple Entries in March Madness"** (arXiv 2407.13438, 2024) [V abstract]: https://arxiv.org/abs/2407.13438. Elimination and top-heavy pools with an exact DP for the best entry's expected score.
- **No BTS-specific analysis of a most-picked or consensus feed was found** [NV, by search].
- Relevance, skeptically:
  - The BTS grand prize is an *absolute* threshold (57), split only on same-day ties. So the game-theoretic "contrarian value" result in items 1, 2 and 4 does not transfer, except to the small relative Top Streak Prize.
  - What does transfer is the evidence that crowd picks are biased toward favourites. That bias tilts against reading `numberSelections` as a forecast, and toward the #87 / W4 rank-7 framing: consensus may nominate a feature hypothesis, never a copying strategy.

---

## Part 3: Suggested dispositions (my suggestions; the lead decides)

| # | Item | Suggested disposition | One-line reason |
|---|---|---|---|
| 1 | TabM (ICLR 2025) | **Candidate W4 experiment** (rank 6, the single model-class challenger, frozen default k=32) | The better default among the two neural options on BeyondArena (1107 vs 1056 Elo) and above default LightGBM (991). Time-split evidence (TabReD) favours MLP-like models and GBDTs. Gate on within-slate rank and top-bin proper scores, not AUC. |
| 2 | RealMLP / Better by Default (NeurIPS 2024) | **Not selected** (alternate only if a tuning budget is allowed); LGBM-TD **closed** by the "no LightGBM re-tuning" boundary | Its temporal-subset lead in BeyondArena needs tuning plus ensembling. Its default lags TabM's. The plan says pick one. |
| 3 | TabPFN v2 (Nature 2025) and 2.5/3/3.5 | **Not applicable** at PA level; game-level residual idea **parked** | Limited to 10K rows (v2) and ≤1M rows (3/3.5) against several million PA rows. TFMs lag on temporal and large data (BeyondArena). Weights after v2 are non-commercial and the 3.5 licence bars "production" use. |
| 4 | TabArena (NeurIPS 2025) | **Closed** as evidence (methodology reference only) | IID-only by design with random CV, 500–250K training rows, ROC AUC for binary tasks. BeyondArena is the relevant temporal extension. |
| 5 | Online conformal, decaying steps (ICML 2024) | **Not applicable** (reopen if W4 rank 3 builds a PA-count model) | The binary target makes sets uninformative, coverage ≠ calibration, and a decaying step adapts more slowly under drift. |
| a | Public BTS models | **Closed** as a benchmark source; **README hygiene item** for the Garnett citation | No comparable as-of forecasts under our guardrails. The Garnett article wasn't opened and its numbers don't match our README (84%/81% P@100 in the snippet vs our cited 85%/77%). |
| b | MLB `probabilityStarter` | **Feeds W4 rank 1** (no change). The semantics gate must be settled empirically | No published method; the only official text is "our unique prediction model". The rules define Hit as ≥1 official AB or SF, with Pass otherwise. |
| c | Statcast bat tracking / miss distance | **Supports W4 rank 5** as registered. Add to the registration a measured as-of lag and the swing-denominator definition | Official definitions and a 2023-second-half coverage start are confirmed. No official lag statement exists. Swing metrics are context-confounded (Powers & Yurko). |
| d | Calibration under shift | **Inside W4 rank 4a.** If 4a is selected, register online/windowed Platt (OPS/WPS, intercept-only variant) as its single map. Davis 2020 goes to the monitor backlog | The low-parameter online map fits ~180 picks per season. Theory-optimal recalibration needs orders of magnitude more rounds. |
| e | MDPs with resets / milestones | **Closed** (no new method) | Confirms the state-augmented deterministic policy, which is already built. The open problem is run-structure misspecification, which no source addresses. Note the Top Streak Prize relative-payoff point if objectives are revisited. |
| f | Forecast combination with a public benchmark | **Supports W4 rank 1 design**: fix one rule a priori (beta-transformed linear pool or logit-linear pool) and use an encompassing-style residual-information test as the gate | A linear pool of calibrated forecasts is uncalibrated (Ranjan & Gneiting). Estimated weights lose to simple rules in small samples (Smith & Wallis). |
| g | 2026 ball drag | **Closed** for alpha; keep as regime observability | MLB confirms in-season drag shifts (2025 high, June 2026 drop). Our 2026 park-drag screen was NULL, and any league-drag variant validates in 2027 under D3. |
| h | Streak contests and consensus feeds | **Closed** as a method source; supports #87 framing (W4 rank 7 unchanged) | Contrarian results need relative payoffs, and the BTS grand prize is absolute. Crowds over-pick favourites, so consensus is a feature hypothesis, not a forecast to copy. |

---

## What I could not verify

- **Garnett (2026) Medium article.** The URL was located, but Medium/Cloudflare blocked me twice. Its P@100/P@500 numbers and the author's first name ("Kevin", per our tracker) are unverified. Our README's 85%/77% differs from the search snippet's 84%/81%.
- **`probabilityStarter` field semantics and MLB's model methodology.** Nothing is public beyond the FAQ sentence. Whether Genius Sports or anyone else builds it is unknown.
- **The Athletic's 2025 and 2026 drag reporting** (paywalled). The MLB quotes come via Yahoo and Bleacher Report. Baseball Prospectus's 2026 drag article is paywalled.
- **The availability lag of bat-tracking and miss-distance data** on Savant. No official statement was found.
- **Abstract text for** Satopää et al. 2014 (snippet only), Harvey–Leybourne–Newbold 1998 (metadata only; the content claim is from general knowledge), Bouakiz & Kebir 1995 (metadata only), Simmons et al. J Consumer Research 2011 version (the SSRN version was verified), and Drift-resilient TabPFN (Helli et al. 2024; seen only as a Nature citation).
- **TabM per-split numbers on the TabReD time splits, and BeyondArena's temporal-subset defaults.** These are in figures I could not extract as text, so the default-vs-default comparison above uses all-dataset aggregates.
- **TabPFN-3.5 feature limit.** The README says 20,000 features; the extracted report figure says 6,000. Not resolved; it is irrelevant at our feature count.
- **MLB technology blog, "Introducing Statcast 2023: High Frame Rate Bat and Biomechanics Tracking"** (403). The coverage start rests instead on Savant's own text ("second half of the 2023 season").
