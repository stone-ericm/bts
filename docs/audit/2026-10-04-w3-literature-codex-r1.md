# W3 literature / SOTA refresh — Codex review R1

## Verdict

**BLOCK.** The smallest fix is a documentation correction: restore the plan's conditional W4 gates, narrow the tracker dispositions to the tested hypotheses, repair the source attributions, and add the missing provenance. No experiment, outcome read, new candidate, or production change is needed to resolve this review.

Reviewed the memo and all three evidence files supplied at `1d2fb0cedb9b2a173ee4d2f6786ff7180fb781bf`. Main advanced concurrently to `cac7e917bac57377536681a32105ba5958628640`; the four under-review files are unchanged between those commits. The later W1.6 exposure corrections and branch-document archive are accounted for below. Memo SHA256: `cfc0f0904729fb531fde06c67c38aa069f2c1f5080c271f82b07231f355ca7ed`.

This is a repository/document and public-source review. Historical measurements below are **verified as reported in documents**, not independently reproduced. No `data/` file was read, no tests or experiments were run, and no tracked file was edited.

The checkout was initially clean. During verification, unrelated modifications appeared in the field-products design, MLB-forecast design, MLB benchmark core and its tests. This review did not edit or revert them; they do not change the four reviewed artifacts.

## Findings

### F1 — P1: Rank 4a becomes a different experiment without evidence for the change

**Locations:** memo:23,73,81; research-notes:176–185,246. The plan:191 specifies identity versus **one regularized intercept map**, with a slope challenger only if supported. The memo instead nominates two-parameter online/windowed Platt, demoting the intercept-only map to a variant, and says it fits our sample size.

[Gupta & Ramdas, §§2–2.4](https://arxiv.org/html/2305.00070v3) distinguish OPS regret against the best fixed Platt map from calibration guarantees supplied by the additional calibeating construction. Their WPS comparator periodically refits on accumulated history; that is not evidence for a rolling-window implementation. Their parameters were fixed after initial investigation. The paper does not establish useful performance with approximately 180 selected picks or prove the intercept restriction inherits all guarantees. The separate recalibration paper's asymptotic guarantee is not a power calculation for this candidate.

**Consequence:** the literature summary silently expands the selected model and its tuning choices. Restore the intercept-first gate; describe OPS as inspiration and any intercept-only adaptation as reasoning that still needs a frozen protocol and held-out evidence.

### F2 — P1: The forecast-combination literature does not justify “never a linear average” or a guaranteed repair

**Locations:** memo:21,75,80; research-notes:198–204. The notes label Ranjan & Gneiting **M / abstract only**, Satopää **M/S**, and the encompassing-test content **unverified general knowledge**. Those qualifications disappear from the final memo's firm method choices.

Independent mathematical checks refute the blanket wording: if two calibrated forecasts are identical, their linear average remains calibrated. A beta-CDF transform with parameters (1,1) is the identity, so merely choosing that family guarantees no repair. Estimating additional parameters can add finite-sample error. The relevant theorem's distinctness/nontrivial-weight assumptions cannot establish that a fitted nonlinear pool improves our forecasts, particularly when neither input's calibration and target equivalence has been established.

I attempted the cited [Ranjan & Gneiting DOI](https://doi.org/10.1111/j.1467-9868.2009.00726.x); the publisher access failed, including a 403 on the direct fetch. **Full-primary verification remains unavailable in this review.** Do not label it completed. [Smith & Wallis](https://doi.org/10.1111/j.1468-0084.2008.00541.x) is abstract-level evidence in the supplied notes, not a universally optimal combination rule.

**Consequence:** rank 1 should retain one later-nominated rule, with no search on W2.3 and no automatic recalibration guarantee. The binary residual-information check is a proposed analogue, not a verified direct application of the cited encompassing test.

### F3 — P1: Several tracker “negative” dispositions answer broader questions than the documents tested

**Locations:** memo:38–54. All 17 rows were compared with the tracker; the required six (#1, #2, #9, #13, #16, #17), and additional disputed rows, were checked against result/protocol documents. The table below records the supported scope and remaining defects. “No 2026 outcomes” refers to the documented experiment, not a proof that every related script never read them.

| Area | Supported state / disposition and basis | Exposure / trigger assessment |
|---|---|---|
| **1** | Built-and-measured **finite DR screen**, not a production DR solver. CVaR-over-streak was rejected for the objective; distributional DP remains unbuilt. `2026-05-06-dr-mdp-gap-result.md:17–54` reports a one-seed actual-PA gap below a **borrowed** harness half-width. `…-dr-mdp-gap-pooled-24seed.md:5–35` already reports a 24-seed raw screen, with path-derived seeds and uncertified provenance. This is a no-investment decision on those surfaces, not a powered null for robustness. Tail mechanism validation is not value validation. | Distinguish Gate A's live outcomes from Gate B's historical evaluation and served-p scale inspection. The reopening trigger must acknowledge the existing raw 24-seed result and require a certified/new surface, not imply that no 24-seed screen exists. |
| **2** | **Built-and-measured isotonic only**; the named Beta/Venn-Abers/spline comparison is unstarted. `2026-05-23-calibration-resolve-gate.md:29–37,63–73` gives n=158, slightly worse Brier, and its own n≥200 floor. Inconclusive is supported; this is not a negative result for the other map families. | X-27 now records the read. Retain explicit temporal cleanliness, target definition and positive Brier-improvement CI; reaching n alone is insufficient. |
| **3** | Unstarted / parked agrees with tracker:200–208. No measured negative is documented. | “None stated” is honest but is not the plan's specific reopening condition. Explicitly keep parked this cycle pending a separately approved diagnostic need. |
| **4** | E-values/e-processes unstarted; F10's checkpoint tests are adjacent work, not implementation of this target (tracker:210–218). | Attribute X-07 to that adjacent work. Do not imply it tested an e-process. Specify future sequential-testing need plus a registered valid construction, or no reopening this cycle. |
| **5** | Manifest and season-split infrastructure exist, with partial adoption (tracker:222–230). **“Phase D only” is false:** `2026-05-06-conformal-gate-v2-refresh.md:14,22–23` records a manifest-bound lockbox evaluation before Phase D. Full nested/purged adoption remains incomplete. | Historical evaluation; no documented live outcome read in these result documents. “Tracker” must be replaced by its actual adoption trigger; methodology completion is distinct from a pooled-policy win. |
| **6** | BOCPD unstarted; alert-only infrastructure is separate. `2026-07-08-park-drag-2026-screen.md:20–55,69–81` evaluates estimated-PA aggregation on **actual participant slates**, including substitutions; not the at-cutoff live stream. It supports no detected ranking gain for those definitions, not a universal drift-method null or a universal power floor. | X-02 covers 2026 consumption; identify monitors separately. Preserve observability and the D3 boundary on any new candidate. |
| **7** | BH/BY built-and-measured; e-BH unstarted. Zero discoveries in two named families is not a negative result for FDR methods or evidence that every feature is null. `2026-05-06-audit-verdict-fdr.md:12–29` uses two-season sign flips with minimum two-sided p=0.5: discoveries at q=.05 are structurally impossible. Tracker:282 also records a later segment-family survival problem, so blanket “0 survivors” is misleading. | Scope each family and live read. X-26 records the 5/10 refresh, not automatically every 5/05 computation. Valid e-values and a prospective family are the actual sequential-method trigger (tracker:252). |
| **8** | ACI/RCPS unstarted / parked is consistent with tracker:254–262; no evidence of failure of these methods. | Retain the binary-conformal prerequisite, while distinguishing a future PA-count coverage use from game-p recalibration. |
| **9** | Sequence/GNN target unstarted; nearby screens measured different hypotheses. Embedding null is restricted to low-rank ID interaction, with ex-post most-PA opponent selection (`2026-06-15-phase0g-embedding-RESULT.md:5–14`). Kcontact's +.0024 AUC arm is below its practical bar, with noisy top-1, not proof against smaller effects (`2026-06-15-kcontact-result.md:18–28`). Swing gate failure is **invalid for family inference**, not a feature null (`2026-06-13-swing-screen-gate-fail3.md`). Team-record is underpowered; P-03 is inconclusive/no-promote. | “Mixed” must resolve to the named screens' bases and exposure rows. Neither positive controls nor P-03 automatically trigger a new sequence model. W4 rank 5 remains conditional and separately registered. |
| **10** | Pooled policy measured **negative on its declared Phase D comparison**; predictive stacking remains unstarted. The historical iid-bin value difference −.063 and 0/100 positive seeds support rejecting that policy on that surface (`2026-05-08-phase-d-pooled-policy-outer-eval.md:14–38,56–68`). They do not falsify predictive stacking or establish serving value. Actual-PA is still labelled inferred; pin the profile producer before upgrading it. | No documented 2026 outcomes in this comparison. State the actual conditional reopening rule, not “tracker.” |
| **11** | Built-and-measured lower-bound gate; negative for the **two methods / six cells on the named canonical surface**, not all binary conformal methods (`2026-05-06-conformal-gate-v2-refresh.md:32–58`). | Historical actual-PA surface. Reopening requires corrected selectable-row validity and tightness with a nonempty ship set, not simply a new general paper. |
| **12** | Proper-scoring diagnostics are built; limited data do not falsify the scoring methodology. Candidate conclusions remain inconclusive. | X-19 is **counts only**, expressly no rates or model comparisons (register:25). X-26 now covers the refresh. X-21 is a registered descriptive bridge, not an alpha selection read. |
| **13** | V1 evaluator built; the **8.17% serving interpretation is invalid**, not “OPE negative.” The design `2026-05-04-bts-sota-13-ope-design.md` distinguishes the implemented model-based/terminal evaluation from fuller sequential estimators. `2026-07-06-strategy-model-lever-investigation.md` documents actual- versus estimated-PA mismatch. Estimated-PA outputs are projections, not measured jackpot frequency. | Basis must include the later estimated-PA comparator as well as old actual-PA. Realized replay is preferable for the exposed run structure, but not an unqualified trusted evaluator: plan:200 requires fixing/disclosing its clock and partnerless-double caveats. |
| **14** | CE-IS built under a declared simulator; its numerical rare-event estimate is conditional on that simulator. Invalid serving headline / unresolved transition structure is the proper distinction; the 7/13 evidence does not demonstrate a defect in Monte Carlo itself. | Separate original actual-PA iid basis from the later estimated-PA run-structure diagnostic. State a corrected-model validation trigger. |
| **15** | “No exploitable dependence” is contradicted by `2026-08-30-same-game-pair-correlation.md:33–51,66–74`: .9995 is **opposite-team** relative lift; same-team strata show approximately 1–2% positive relative lift. Marginals are batter-season rates, not production forecasts. Neither exact independence nor production action value was established. | X-28 now covers the live repeat-batter side-check. General dependence is unresolved; production-conditional pairing and temporal structure need their own valid replay before changing policy. |
| **16** | Built frozen candidate `5004b1c8`; eligible ≤113 <120, **paired comparison unread** (register X-11). Outcomes have been joined nightly, so “never read” needs that narrower qualifier. Inconclusive is correct. | `2026-05-08-fresh-audit-pre-registration.md:381–384` permits **pre-registering** a 2027 continuation; it does not contain one already registered. Keep no-peek until that continuation and its cross-season rules are frozen. |
| **17** | Tested legacy/model-class candidates failed to establish a dependable edge; the **whole area is parked**, not generally negative. `2026-06-15-resolution-audit-RESULT.md:3–5,27–28` expressly limits the conclusion to tested classes/objectives and notes untuned RF/ET. This is not a ceiling proof against TabM or all new representations. | Historical actual-PA evaluation, no documented 2026 read in those screens. One new challenger may be nominated under rank 6; nomination does not confer selection or a production gate. |

### F4 — P2: Benchmark attribution is wrong, though the four leaderboard point values match

**Locations:** memo:26–27,61–64. The primary [BeyondArena paper, Table D.1 and §E.5](https://arxiv.org/html/2606.30410v1) verifies the listed 1107, 1056, 991 and 1149 values. The claimed temporal leader and global “ROC-AUC Elo” description are wrong; the replacement text below specifies the corrections. A default aggregate ranking does not establish the ordering of the temporal subset's defaults or our production blend. Thus “RealMLP's temporal lead needs tuning” is stronger than the extracted evidence.

TabM may remain the **provisional single nomination**. This review does not demand testing both models or tuning LightGBM. However, “frozen default (k=32)” is not a fully frozen small recipe: the [TabM README](https://github.com/yandex-research/tabm#hyperparameters) says defaults depend on supplied arguments. Pin package/configuration/optimizer/refit budget if selected, as plan:170,193,198 already require.

### F5 — P2: The prize-winning probability is not the asserted payout objective

**Locations:** memo:28–31; verified-rules: inference; research-notes:194. The omitted conditioning and tie weight matter. See the [Official Rules, §§7–8](https://www.mlb.com/apps/beat-the-streak/official-rules).

**Reasoning, not a measurement:** let B be our season best, F the largest other eligible best, K the number of other co-winners tied at B, and E/G indicate our eligibility / no Grand Prize award. Expected Top Streak payment is

`10,000 × E[1{E ∩ G ∩ B≥max(20,F)} / (1+K)]`.

The probability of any share omits the denominator and can rank policies differently. An illustrative policy with a 40% chance of a ten-way share yields $400 expectation; a 20% chance of winning alone yields $2,000. These are synthetic arithmetic, not BTS estimates. No joint field/own-streak model or tie forecast exists in the reviewed material. A personal E[best] objective remains an owner choice; this source does not invalidate it or select a replacement D1 objective.

### F6 — P2: Required provenance is missing, and the source-strength claim is false

**Locations:** memo:6,10,36–54,95; tracker-inventory:5–25. Plan:164 explicitly requires a **frozen implementation reference**, specific reopening trigger, and **primary links in the final memo**. The table omits implementation refs except #16; “tracker” is not a specific trigger. The evidence table's `T:…` shorthand does not identify the actual frozen code/results. The final memo has no primary-paper links; moving them entirely into notes does not satisfy the requirement.

“Every external claim was checked against a primary source” contradicts the research pass's own M/S/NV labels. Preserve primary-page/full-text versus abstract-only versus secondary/unverified at claim level. In particular, the proposed binary encompassing test, blocked Garnett numbers and secondary MLB drag statements cannot acquire stronger status by condensation. An unsuccessful literature search does not establish that no public methodology, field-name reference, or lag statement exists anywhere (memo:71–72).

A compact provenance appendix is sufficient. For each row, give the code commit/blob or state “not implemented,” the result/protocol document, the measured surface/selection-time limitations, and the literal reopening condition. One repo revision can pin existing infrastructure, but it cannot retrospectively certify the code that produced an old result; missing producer pins must remain explicit.

### F7 — P2: TabPFN scope is treated as a hard cap and newer evidence is compressed away

**Locations:** memo:63,88; research-notes:68–82,240. The [TabPFN-3.5 report, Figure 1 and §2.2](https://arxiv.org/html/2609.17895v2) calls row counts recommended limits, not computational caps. It also gives a narrower temporal result on non-large tables than the memo's blanket “foundation models lag” statement. That qualification matters when the plan explicitly nominates a compact residual table. The notes' 6,000-versus-20,000 feature “mismatch” is explained by the figure caption's allowance for additional estimators.

The [official repository's licensing distinction](https://github.com/PriorLabs/TabPFN#license) between code and checkpoint weights supports keeping the licence question; do not call all weights simply Apache because the package is. Parking the game-level experiment pending W1.2 is justified by the plan. “Not nominated for our full multi-million-row daily PA fit” is a practical scope decision, not an impossibility proof. No owner licence ruling or production action is requested by this review.

### F8 — P2: The season-wide promotion claim is false; the open-item list now needs a dated closure

**Locations:** memo:17–18,39,49,52,56,90. `CLAUDE.md:77` and `ARCHITECTURE.md:46` explicitly record the pitcher-30g min-period change shipped 4/14; the plan:184 itself calls for multi-seed revalidation of shipped changes. “No model change was promoted” cannot describe the entire season. Restrict the assertion to the reviewed SOTA-cycle candidates and distinguish the invalid swing screen.

The table is labelled **as of 10/03**, so its historical absence statements need not be rewritten as though rows already existed then. But current open actions must acknowledge W1.6 updates during this review: `b8fa9d7` adds X-26/27/28 and C-05; C-05 explicitly concludes the 5/24 Gate B read used served probabilities rather than new 2026 outcomes. `b3df966` archives branch result documents, while code stays on those branches (`2026-06-unmerged-branch-results.md:3–11`). Update the memo and inventory with dated follow-ups, not a demand to register that Gate B read or merge code.

### F9 — P2: The bat-tracking search output omits a required item and conflates coverage with availability

**Locations:** memo:72,82; plan:176. The plan explicitly asks for squared-up alongside miss distance and swing path, plus coverage eras. Squared-up is in the notes but absent from the memo's mapping. Add a closed/no-additional-candidate disposition, with unknown coverage/lag marked unknown.

The miss-distance [launch article](https://www.mlb.com/news/explaining-the-new-miss-distance-statcast-metric) confirms launch on 2026-06-09; the [Savant leaderboard](https://baseballsavant.mlb.com/leaderboard/bat-tracking/swing-timing-miss-distance) confirms second-half-2023 historical coverage and excluded bunts. Historical retrospective coverage does **not** establish pre-launch point-in-time availability or a fixed ingestion lag. The existing rank-5 nomination can stand, conditional on measuring availability and freezing the precise tracked-swing denominator, missingness controls and practical effect bar.

## Verbatim edits

The replacements below address the disputed prose. Apply the same qualifications to the evidence notes' **suggested conclusions/inference**, while retaining their original source labels and raw observations. The table/provenance completion described after the blocks is also required; these prose edits alone do not earn SIGN.

### A. Replace Sources bullet 1 (memo:10)

> External sources were reviewed on 2026-10-03, with primary/full-text, metadata or abstract-only, secondary, and unverified labels in the evidence notes. The final claims retain those distinctions. BeyondArena's identity and headline were re-checked on 2026-10-04; a source's existence or abstract is not verification of every method claim attributed to it.

### B. Replace the season-change bullet and its sub-bullet (memo:17–18)

> None of the reviewed SOTA-cycle model candidates was production-cleared. This is not a claim that the season had no shipped model or feature changes: the pitcher-30g min-period change shipped on 2026-04-14 and still requires multi-seed revalidation. The 9/03 tail objective was an owner requirement; P-05 validated its mechanism, not its value or optimality. The nearby feature screens have scoped negative, inconclusive or underpowered results; the failed swing controls invalidate inference about its feature families.

### C. Replace the three “literature changes” items (memo:20–27)

> 1. **Rank 1:** retain the plan's order: establish target/availability semantics and shared as-of residual information in W2.3, nominate at most one combination rule afterwards, then validate on untouched later dates. A beta-transformed or logit pool is an option, not a guaranteed calibration repair; a simple fixed pool is not ruled out categorically. No combination search is permitted on the consumed benchmark window. The binary residual-information test is a proposed analogue that needs its own valid protocol.
> 2. **Rank 4a:** if selected, identity versus one regularized intercept map remains the experiment; fit a slope only with explicit support. Online Platt provides a possible implementation idea, not evidence that two fitted parameters work with our season-sized selected-pick sample. Any intercept-only adaptation, update rule and parameters are frozen before held-out evaluation; theory alone does not supply its power.
> 3. **Rank 6:** TabM remains the provisional single challenger nomination, conditional on the decision memo and a frozen small recipe/budget. BeyondArena's overall default comparison motivates it but does not verify temporal-default or BTS superiority. The paper's temporal result identifies tuned-and-ensembled **RealMLP**. Its aggregate scores combine binary ROC AUC, multiclass log loss and regression RMSE. Our gate remains temporal within-slate ranking and selected/top-bin proper scores on the identical serving contract, with paired seeds and agreed compute; no architecture sweep or LightGBM re-tuning.

### D. Replace the prize bullet and its inference (memo:28–29); retain the below-floor observation and decision-memo deferral

> **D1 input:** $10,000; highest eligible streak ≥20; inactive allowed; ties split; payable only if no Grand Prize is awarded ([Official Rules §§7–8](https://www.mlb.com/apps/beat-the-streak/official-rules)).
> Prize-winning probability, expected prize share and personal E[season best] are different objectives. If B is our best, F the largest other eligible best, K the number of other co-winners, and E/G our eligibility/no Grand Prize award, expected Top Streak payment is `10,000 × E[1{E ∩ G ∩ B≥max(20,F)} / (1+K)]`. This is a rules-based derivation, not a measured BTS value. We have not estimated the necessary joint field/own distribution. Whether to optimize a prize objective or personal season best remains D1's owner decision.

### E. Replace the nomination Reason cells for TabM, RealMLP, TabPFN and TabArena (memo:61–64)

**TabM:**

> BeyondArena Table D.1's all-task Elo point values are TabM default 1107, RealMLP default 1056, LightGBM default 991 and LightGBM tuned 1149 ([primary paper](https://arxiv.org/html/2606.30410v1)). Temporal-default ordering was not extracted. This aggregate is nomination evidence, not a comparison with our 12-model production blend. If selected, pin the package version, resolved full configuration and CPU training/refit budget; k=32 alone does not freeze the recipe ([TabM defaults](https://github.com/yandex-research/tabm#hyperparameters)).

**RealMLP:**

> Alternate to the provisional TabM nomination; no second challenger is added. The aggregate default comparison does not establish whether tuning is necessary for RealMLP to lead on temporal tasks. LGBM-TD remains outside the plan's no-LightGBM-re-tuning boundary.

**TabPFN:**

> Not nominated for the full multi-million-row daily PA fit. The compact out-of-time game-level residual idea remains parked pending measured W1.2 headroom. The 10K/1M row scopes are recommendations/evaluated regimes, not proven computational impossibility. TabPFN-3.5's report qualifies temporal performance on non-large tables and does not establish superiority for our task. Code and checkpoint-weight licences differ; later weights require an owner licence assessment before any production use ([report](https://arxiv.org/html/2609.17895v2), [official licence summary](https://github.com/PriorLabs/TabPFN#license)).

**TabArena:**

> IID random-split evidence; retain as a methodology/version/budget reference, not temporal baseball performance evidence. ROC AUC describes its binary tasks, not every task in the benchmark ([primary paper](https://arxiv.org/abs/2506.16791)).

### F. Replace the three disputed search findings (memo:71,73,75); append the bat-tracking qualification

**MLB methodology finding:**

> The FAQ supplies only a generic model description in the sources inspected. The search did not locate a public technical definition of `probabilityStarter`; its name does not establish conditional-on-starting semantics. The contest grading rules do not themselves define the forecast target. W2.3 must establish semantic/availability compatibility or report association only.

**Calibration under drift finding:**

> OPS is a possible low-dimensional update mechanism; its paper does not establish adequacy at our selected-pick sample size. Preserve rank 4a's identity-versus-one-regularized-intercept design, with a slope challenger only if supported, and a held-out proper-score gate ([Gupta & Ramdas](https://arxiv.org/html/2305.00070v3)).

**Forecast combination finding:**

> Abstract-level evidence warns that nontrivial linear pools of distinct calibrated forecasts can lose calibration; it does not prove a fitted beta/logit rule will improve our forecasts. Finite-sample weight-estimation error argues for a simple rule fixed before validation. Ranjan & Gneiting and Smith & Wallis remain abstract-level sources in this pass; the binary encompassing-style check is our proposed adaptation, not a directly verified theorem for this target ([Ranjan & Gneiting](https://doi.org/10.1111/j.1467-9868.2009.00726.x), [Smith & Wallis](https://doi.org/10.1111/j.1468-0084.2008.00541.x)).

**Append to bat-tracking finding:**

> Historical coverage is distinct from point-in-time availability; the inspected sources did not establish ingestion lag. Freeze the precise tracked-swing denominator and missing/permuted controls before labels. Squared-up means ≥80% of attainable exit velocity; rates may use swings or contacts ([official definition](https://www.mlb.com/glossary/statcast/squared-up)). It is closed as an additional candidate this cycle; metric-specific historical coverage and as-of availability remain unverified here. Swing-path/attack-angle coverage must likewise be recorded separately rather than inherited from miss distance.

### G. Replace §5 (memo:79–84)

> ## 5. What this means for W4 (reasoning only; selection belongs to the decision memo)
> - **Rank 1:** after the semantics/availability and residual-information gates, nominate at most one fixed combination rule; no search on the consumed window; validate on untouched later dates.
> - **Rank 4a:** identity versus one regularized intercept map; slope only with support; gate on held-out proper scores. An online update is a protocol choice, not a sample-size guarantee.
> - **Rank 5:** one frozen miss-distance/contact-suppression definition with metric-specific coverage, measured as-of lag, exact denominator, missing/permuted controls and a practical effect threshold.
> - **Rank 6:** provisional TabM nomination, one pinned small configuration/budget, identical feature/serving contract, temporal comparator and paired seeds. Selection is conditional; no architecture sweep or LightGBM re-tuning.
> - **Unchanged:** rank 2 ops validation uses its own failure/recovery fixtures; rank 4b policy-only changes use independent state/eligibility replay and downstream-value gates. Neither is replaced by a forecast-calibration gate.
> - **Not new candidates:** TabPFN full-PA fitting, TabArena-driven temporal selection, game-p online conformal, consensus copying, or another squared-up feature.

### H. Append a dated follow-up to §2/§6 and replace the obsolete current W1.6 action

> **2026-10-04 follow-up to the 10/03 inventory:** W1.6 registered the 5/10 refresh, 5/23 Gate A and 7/13 repeat-batter read retroactively as X-26, X-27 and X-28 (C-05; `b8fa9d7`). The 5/24 Gate B inspection used served probabilities, with outcome evaluation on historical folds, and does not require a new 2026-outcome row. Branch result/design documents are now archived on main with provenance (`b3df966`; `docs/sota_audit/2026-06-unmerged-branch-results.md`); their code remains on the original branches, and swing-escalation still has no result document. The earlier absence statements describe 10/03, not outstanding registration/archive work. X-19 covers counts only; X-21's bridge is predeclared and descriptive. These updates authorize no new candidate or outcome read.

### I. Complete the tracker provenance and bounded dispositions

These are the minimum table corrections, in addition to a frozen-reference/result appendix covering **all 17** rows:

- **#1:** use “finite DR investment screen did not clear its borrowed uncertainty bar; full DR solver unbuilt,” include the existing raw 24-seed report and provenance limitations, and give a certified/new-surface trigger.
- **#2:** use “built-and-measured isotonic only; other named families unstarted / parked; n=158 inconclusive.”
- **#5:** use “manifest/lockbox and opt-in season splits built; some measured adoption; full nested/purged adoption incomplete,” not “Phase D only.”
- **#6:** state estimated-PA aggregation on historical actual participant slates, with 2026 labels, not at-cutoff live evaluation; restrict the null to this screen.
- **#7:** use “classical FDR baselines measured; no discoveries in the named two families; underpowered inference, not a negative method result; e-BH unstarted.”
- **#9:** separate scoped embedding rejection, kcontact below-practical-threshold/noisy top-1, invalid swing-family inference, underpowered team-record, and inconclusive P-03. Record the actual basis and exposure per screen.
- **#11:** restrict negative to the tested lower-bound cells and state the selectable-row validity/tightness reopening gate.
- **#12:** identify X-19 as counts-only and X-21 as descriptive; do not turn either into calibration/model-comparison exposure.
- **#13–14:** separate implemented evaluators from the invalid serving headline; distinguish original actual-PA/iid projections from the later estimated-PA run-structure evidence. Replay remains conditional on documented limitations, not universally trusted.
- **#15:** use “no detected opposite-team same-game lift on the specified historical sample; small same-team lift; production-conditional and temporal dependence unresolved,” rather than “no exploitable dependence.”
- **#16:** use “paired comparison unread; ≤113 eligible <120; continuation must be separately preregistered before inspecting the accumulated target.”
- **#17:** use “negative/no-deployable-candidate for tested legacy classes/objectives; whole area parked; no general model-class ceiling established.”

For #3/#4/#8 and every “tracker” trigger, either spell out the existing tracker condition or state explicitly “no reopening in this cycle; a new owner-approved need/design is required.” This is a disposition, not a demand to build all 17 methods. Keep inferred bases and unavailable historical producer pins explicit; do not invent them to fill the appendix.

### J. Keep primary links in the final memo

Add a compact reference list or inline links for the five nominations and the central additional claims, retaining source-strength labels where applicable:

- [TabM](https://arxiv.org/abs/2410.24210) and [code/defaults](https://github.com/yandex-research/tabm).
- [RealMLP / Better by Default](https://arxiv.org/abs/2407.04491).
- [TabPFN v2](https://www.nature.com/articles/s41586-024-08328-6), [3.5 report](https://arxiv.org/abs/2609.17895), [checkpoint licence summary](https://github.com/PriorLabs/TabPFN).
- [TabArena](https://arxiv.org/abs/2506.16791), [BeyondArena](https://arxiv.org/abs/2606.30410), [TabReD](https://arxiv.org/abs/2406.19380).
- [Online conformal with decaying step sizes](https://proceedings.mlr.press/v235/angelopoulos24a.html).
- [Gupta & Ramdas](https://arxiv.org/abs/2305.00070); [online recalibration theory — abstract-only in the notes](https://arxiv.org/abs/2607.19689).
- [Ranjan & Gneiting — abstract-level, full text unavailable in this review](https://doi.org/10.1111/j.1467-9868.2009.00726.x); [Smith & Wallis — abstract-level in the notes](https://doi.org/10.1111/j.1468-0084.2008.00541.x).
- [MLB forecast FAQ](https://www.mlb.com/apps/beat-the-streak/frequently-asked-questions), [miss-distance coverage](https://baseballsavant.mlb.com/leaderboard/bat-tracking/swing-timing-miss-distance), [squared-up definition](https://www.mlb.com/glossary/statcast/squared-up), and the Rules link in the D1 passage.

Do not present uninspected abstracts, secondary reports, search negatives, or unavailable primaries as full-primary verification. The remaining limits can be frozen after R2; no third review round or additional experiment is required by these findings.
