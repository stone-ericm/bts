## Verdict

**BLOCK.** W1.2's specified table schema supports a bounded descriptive W1.3, but the proposed rules can reverse the optimism contrast, lose bootstrap multiplicities, classify missing or low-power evidence as weakening, and label explanations that their measurements do not test. The smallest fix is to narrow those labels and specify the consumer's statistical and selection adapters before freezing. The verbatim edits below require no new acquisition, surface rescoring, candidate, or historical experiment campaign. Missing evidence can remain explicitly unavailable.

Reviewed `main` HEAD `9e835de7d3ba1cb2ada779980f979fed12de1578`; W1.3 design SHA256 `7a721100a709b853d3ad73fbc6f9d686b49b4d344bf38146c96a7d995c3eb785`. Evidence: the specified plan, W1.2 frozen design and both design reviews, exposure register, W1.2 core/report/runner, drag producer/reader, relevant repository history, and read-only in-memory Python probes against the actual helpers. No `data/` reads, W1.2 outputs, SSH, network, or tracked-file edits. The accepted run, date census, actual coverage, reproduction rates and outcome estimates remain **unverified**. All numerical examples below are **synthetic**, not 2026 measurements.

The verdict concerns the W1.3 specification. It does not reopen W1.2's accepted design limits or certify the in-progress W1.2 implementation/run.

**Shared-checkout verification:** another session advanced HEAD during this review to `7cb44907f1cfde2f40f5de77295da645ccc2b3bb`. The W1.3 and frozen W1.2 design hashes remained unchanged. Inspected the intervening diffs: W1.2 added plain/gzip capture discovery (`4e4bf5a`), without changing the bootstrap, pair sign/support, reproduction classification or primary helpers discussed below. The exposure register added X-23; the cited X-02/X-03/X-09/X-12/X-21 rows were unchanged. Source line references below describe the inspected initial implementation; the runner's later references shift by one line after its capture-discovery import. No review claim relies on another session's run outputs. Tracked worktree and index were clean at verification.

## Findings

### F1 — P1: Contrast direction and common support are not executable as stated

**Locations:** W1.3 `:21`, `:31`, `:33`; W1.2 `report.py:58–91`, `core.py:87–93`; frozen W1.2 design `:62–64`.

W1.2's pair helper returns **later minus earlier**, under `top1_diff_b_minus_a`. A26 outperforming B26 therefore produces a negative interval. W1.3's arrows and “above 0” do not define the opposite sign its optimism rules need. A native synthetic case with A26 selecting a hit and B26 a miss on both dates returned difference **−1**, CI **[−1, −1]**: directly consuming the named helper would reverse E1.

W1.3 also calls for each surface's `_winners` within its own stratum, but W1.2's paired comparisons first freeze the finite-score intersection and rerank both arms there. These differ. A synthetic A26 winner was a hit on a row missing from B26; the native B26 winner was a miss, while the correctly paired common-support contrast was **0**. Native hit-rate subtraction would manufacture a one-point paired effect. The A26/B26 non-adjacent pair required by E1 is **not** in the runner's `CHAIN` or summary; compute it directly from the frozen table, rather than summing adjacent changing-support contrasts.

**Fix:** define reductions as earlier minus later, retain native winners only as separate diagnostics, and specify identical candidate/date support for every labelled contrast. The chain remains composite; neither it nor its sum decomposes the README/live headline.

### F2 — P1: The named bootstrap cannot give regrouped equal-date statistics new block identities

**Locations:** W1.3 `:22–23`, `:34–36`; W1.2 `core.py:101–110`, `:123–135`, `report.py:17–24`.

The core bootstrap concatenates sampled date groups but retains their original `date`. `equal_date_auc`, a daily residual-mean calculation, and `_winners` then regroup duplicate copies into one date. Row-weighted means preserve multiplicity; those regrouped equal-date statistics do not. On three synthetic dates with AUCs 0, 0, 1, seed 7's first draw was **[c, b, c]**. The actual helper plus `equal_date_auc` returned **0.5**, while the prescribed mean across three sampled blocks is **2/3**. W1.3's sentence “Each drawn copy ... is its own block” is not implemented by the named helper.

The median split itself is outcome-independent, but the design leaves the median date's assignment, half sizes, empty-class AUC dates, and half-specific resampling undefined. E5 also needs a refit in every draw, not ordinary candidate-independent logistic standard errors. “Count failures” does not specify which percentile distribution or labels survive them; the core helper simply applies `np.percentile` to all returned values and does not catch failed fits.

**Fix:** precompute one record per original date for equal-date statistics, then average the sampled records with multiplicity; or assign an occurrence ID to each concatenated block. Resample the two halves separately at their original sizes and jointly resample compared surfaces within each half. Define failed-draw handling before computation. Implement these in W1.3; no edit to the frozen W1.2 outputs is necessary.

### F3 — P1: Zero inclusion is treated as weakening or preserved ranking; multiplicity is unstated

**Locations:** W1.3 `:27`, `:31`, `:33–36`, `:38`, `:45`, `:65`.

E3/E4/E6 call an interval containing zero “weakened”; E5 calls an AUC-difference interval containing zero preserved ranking. Intervals such as **[−0.25, +0.25]** meet those rules while allowing large effects in either direction. Calibration intercept/slope intervals containing identity likewise mean identity was not rejected, not that calibration is good. This contradicts the plan's “not significant ≠ luck” and W1.2's prohibition on equivalence conclusions (`plan:117`; frozen bridge design `:64`). The eight explanation names also conceal several component flags, including an either-intercept-or-slope rule, plus multiple strata. Predeclaration and a primary stratum do not supply simultaneous 95% coverage.

**Fix:** zero/identity inclusion ordinarily means undetermined. Either predeclare and adjust the full primary inference family, or retain the existing simple intervals and explicitly demote all labels to unadjusted exploratory diagnostic flags. The latter is the smaller descriptive-only fix and is used below. Sensitivities may describe disagreement but cannot rescue an unavailable primary. E8 remains a conditional null-compatibility result, never affirmative evidence that luck explains the gap.

### F4 — P1: E3 does not test PA-opportunity forecast error

**Locations:** W1.3 `:33`, `:41`, `:69`; plan `:112`; frozen bridge design `:31`, `:35`; W1.2 `run.py:269–285`.

The plan seeks at-lock versus realized total/starter/reliever counts, out-of-time count scoring, and a count change with the PA hit model fixed. W1.3 supplies a realized-slot geometric normalization and a composite context/ensemble contrast. These measure oracle count dependence and mixed scoring conventions, not error in an at-lock forecast. Calling the lift “oracle” correctly excludes achievable improvement, but does not identify forecast error.

A native synthetic example with **actual N=6, estimated N=4** for every candidate changed probabilities from `[0.468559, 0.737856]` to `[0.3439, 0.5904]`; all winners stayed the same and the rank-1 difference/CI were **0 / [0, 0]**. The proposed “weakened” rule fires despite the two-PA count discrepancy. Conversely, realized-count oracle lift can occur with an unbiased but uncertain pregame count forecast. A26-count→B26 also changes matchup/context and ensemble aggregation, as the frozen W1.2 review already established.

**Fix:** retain the oracle count diagnostic and selected-player distributions, but mark the actual PA-forecast-error explanation **not testable here**. Do not reconstruct unavailable historical forecasts or claim B26's realized-slot estimate is at-lock evidence. Distinguish scoring `n_pa` from A26's `n_pas_actual`, particularly for resumed games.

### F5 — P1: E2's unexplained reconstruction gap is not observed serving drift; E4 accepts absent parity and improvement

**Locations:** W1.3 `:32`, `:34`, `:42`, `:66`, `:68`; W1.2 `run.py:228–244`, `:287–301`; frozen bridge design `:37`; bridge r2 finding N1.

The real classes are `exact_final_feed`, `inferred_weather_absent_at_serve`, and `unexplained`. A weather-blank match is explicitly an unwitnessed numerical sensitivity, not established historical weather. An unexplained gap can reflect final-source revisions, missing historical configuration or reconstruction defects. A ≥5% unexplained share is a reconstruction-discrepancy flag; it cannot establish live staleness or quantify its contribution to misses. Missing artifacts/provenance must remain outside the numeric denominator and visible in coverage. A zero-row reproduction table cannot weaken E2.

E4 requires only that **E2 is not consistent**, which includes unavailable reproduction and unexplained shares below 5%. That is not the plan's stable-parity evidence. Its two-sided residual rule also accepts reduced overprediction as evidence explaining deterioration. Equal-date AUC is computed but has no defined role in the label. Fixed coefficients do not fix evolving lookup inputs or slate composition; one evolving season remains the plan's stated weakening condition, regardless of a narrow interval.

**Fix:** name the reconstruction component precisely; preserve exact versus inferred classes and missingness. E4 may flag increased overprediction only on explicitly reported numerically reproduced support, conditional on that support and final-source assumptions. Other shifts are described with their sign. Neither diagnostic identifies conditional hit-model drift or makes the missing PA-level analysis available.

### F6 — P1: E5 lacks a valid fit/failure contract and does not establish “without ranking loss”

**Locations:** W1.3 `:35`, `:22`; plan `:114`; `pyproject.toml:17`.

“Logits clipped to [1e-15, 1−1e-15]” clips the wrong variable: the interval is for **probabilities before the logit transform**. Literal logit clipping would collapse many ordinary high probabilities. The unpenalized fit can be unidentified, completely/quasi separated, nonconvergent, or have singular information. A synthetic call to the locally available statsmodels Logit with three misses at p=.2 and three hits at p=.8 raised **`LinAlgError: Singular matrix`**. Constant p gives a rank-one design for two requested parameters. Merely obtaining finite optimizer output is insufficient.

Fitting intercept/slope on all evaluation labels is a calibration diagnostic, not held-out evidence that a newly fitted correction improves proper scores. The design omits the plan's original-forecast proper-score evidence even though W1.2 already supplies Brier/log loss and reliability. Finally, D−B26 compares different information sets/models; even a correctly paired AUC CI containing zero cannot establish stable ordering or a ranking-preserving remedy. The either-parameter rule also has multiplicity, covered by F3.

**Fix:** clip p, specify identifiability/separation/convergence checks and refitting in each bootstrap draw, and fail labels closed on invalid fits/draws. Carry original-score proper scores and reliability; do not report fit-on-the-same-rows improvements as held out. Report calibration and ranking as separate components. Without a predeclared material-loss/equivalence criterion, the joint explanation remains undetermined; no new calibration candidate is required.

### F7 — P1: E6 changes the plan's target and contradicts its own null limitation

**Locations:** W1.3 `:16`, `:36`, `:43`, `:67`; plan `:115`; drag producer `:219–274`; drag reader `:208–223`, `:243–271`.

**Verified in code and synthetic inputs:** the producer uses a venue/season tenth-date anchor, a 15-venue-game-date rolling mean, pitch-count shrinkage, and `merge_asof(... allow_exact_matches=False)`. Positive delta means greater estimated drag relative to the venue's anchor. With synthetic first-ten Cd=.3 and subsequent Cd=.4, June 12's delta was **+.00625**. Changing June 12's own source Cd did not change June 12's exported value, but did change June 13's value. There is no same-date source leakage in that construction. This does not establish historical file availability or absence of later source revisions. The daily producer is now in this repo; only the seed/history pipeline is external.

The mean over that date's registered venues is **not a league regime series**. Candidate repetition versus unique venue weighting is undefined; available venue sets and shrink weights change. Even with each venue's delta fixed forever at 0 or .02, venue sets `[1]`, `[1,2]`, `[2]` yield dated means **0, .01, .02** and Spearman **rho=1** with a matching composition-driven residual. No venue changed regime. W1.3's “all-known-candidate” residual also needs an explicit pool/common D/C-frozen row mask.

The plan requests independently dated regimes and league plus selected-player residuals. The continuous, changing-slate proxy does not supply them. The plan explicitly names a late-July reversion inside this window; the design supplies no provenance for its assertion that no independently dated in-window event exists. Do not assert existence or absence from this review; record the missing event-date/source witness. Do not select change-points from misses. Section 2 correctly says a null hit association is not evidence against a regime, while E6 labels precisely that null “weakened.” Rolling drag and seasonal residuals also have cross-date dependence/trends that iid-date intervals do not address.

**Fix:** identify this as registered-slate drag/residual association, define unique venue weights and missing-date handling, and preserve retrospective as-of versus live-availability limits. A null/opposite association does not weaken the existence of a ball regime. Mark the plan's full regime test unavailable with these inputs; keep the narrower correlation diagnostic, including selected-player residuals, without new event acquisition.

### F8 — P1: Served-primary re-derivation can create a selection on a skip date; E7's primary cohort cannot supply genuine skip comparisons

**Locations:** W1.3 `:14`, `:19`, `:37–38`; W1.2 `run.py:86–96`, `:180–183`; `core.py:31–48`; `daily_decision.py:85–95`.

`decision_primary` returns None for an authoritative skip; `selection_consistency` then falls back to the pick file. A synthetic skip plus a matching stale pick returned **`selection_consistent`, source `pick`, conflict false**. Thus the named helper sequence is not sufficient to identify the actual action's served primary. A skip record can carry a declined candidate, and neither it nor a stale pick is a selected primary. E8 must not simulate that candidate as an executed selection.

Once skip semantics are respected, genuine skip/no-selection dates do not belong to a selection-consistent selected-date cohort. E7's pick-versus-skip contrast therefore cannot be performed solely in the stated primary stratum. The real action values are `single`, `double`, `skip`, or unknown, not `pick`; “played” would additionally confuse a recommendation with contest entry. `day_meta` can supply action/date/slate size, but not evidence that Eric entered a pick.

E8 appropriately admits calibration circularity, but omits the **conditional independence across dates** assumed by independent Bernoulli draws. Marginal calibration alone is insufficient for that null variance. Its selected recommendation window also differs from the full-season/recipe brief counts; Wilson intervals on those carried numbers are descriptive iid-binomial intervals, not the date-block sampling model.

**Fix:** an authoritative skip blocks fallback; keep declined/stale candidates separate. Use registered all-date forecast diagnostics for E7's action comparison and expose research/unknown strata. Restrict E8 to genuine selected primaries with known outcomes and state both calibration and independence assumptions. Null compatibility cannot label luck as the explanation.

### F9 — P2: Exposure ordering is sound in intent, but the input/headline contract needs bounded exceptions and provenance

**Locations:** W1.3 `:6`, `:10`, `:12–16`, `:48`, `:52–56`, `:64`; exposure register X-02/X-03/X-09/X-12/X-21; W1.2 `run.py:314–327`.

X-29 is correctly placed **before computation** and names X-21's “within the design's limits” restriction plus X-02/X-03/X-09. It is presently a proposed gate, not an existing published row. Retain X-02's prohibition on a new 2026 park-feature fit and X-03's M3 rerun trigger; neither the new correlation nor numerical reproduction authorizes reopening those experiments. If E7 uses research-only dates, explicitly include X-12's overlap and the new descriptive use.

The design calls the inputs “all already produced” and the census exactly 105 despite the run being in progress and W1.2's count being provisional. Use the accepted run's actual registered inventory. The 96/141 figure is the **previously consumed X-09 brief scorecard**, not a verified W1.1 result; X-09 expressly says it is not a test result. Carry its Wilson calculation as an explicit exception to the registered-window output restriction, without recomputing full-season ledger outcomes.

The runner's manifest hashes slates, picks, decisions and PA files, **not final feeds or drag artifacts**. W1.3 can freeze/hash those newly used bytes, but cannot claim they are identical to the feed/table versions W1.2 read merely because W1.2 used those paths. Hash and validate the drag export and its sibling manifest independently; missing venue/drag values require coverage reasons, not zeros, forward fills or opportunistically changing denominators.

**Feasibility check:** all six score columns, `n_pa`, `n_pas_actual`, `lineup_realized`, `projected`, `sel_state`, and the three pools are real. `_winners`, `paired_pool`, `pair_metrics`, `date_block_bootstrap`, `decision_primary`, and `primary_of` are real. `day_meta` is in `summary.json` and includes `sel_state`, `action`, `n_rows`, and model provenance. Reproduction classes are **summary counts, not a row-level table column**; derive row classes from serialized C-served/C-served-weather-blank differences using the runner's precedence, if month/lineup breakdowns are needed. No extra scoring is necessary. Empty primary support must remain unavailable; diagnostics cannot replace it.

## Verbatim edits

Apply the following specification edits before building or computing W1.3. These are proposed edits only; this review does not modify the tracked design or authorize publishing/running the analysis.

### A — Replace §1's paragraph and §2's first two bullets

> This is a bounded diagnostic of the accepted W1.2 run's registered 2026 dates and candidates. The README's historical 86% headline and the previously consumed X-09 brief scorecard (96/141) are noncomparable context, not estimates on this window or measurements rebuilt here from W1.1. The eight plan explanations may overlap. Component diagnostics may be compatible with several explanations; they neither identify causes nor sum to the headline-to-live gap. Where the plan's required evidence is unavailable, that explanation is explicitly not testable here.

> - **The accepted W1.2 run, once available:** `table.parquet`, `summary.json`, and `manifest.json` from the run accepted by its memo. Read its actual `registered_dates`, candidates, coverage, six surfaces and pool definitions; 105 is provisional until that acceptance. Do not generate or inspect another run or disclose unregistered candidate outcomes.
> - **Selected primary:** read and verify the manifest-hashed decision and pick bytes. An authoritative `action == "skip"` means no selected primary and blocks pick fallback, even if a stale pick or declined candidate exists. Otherwise use the authoritative decision primary; use a pick primary only when no authoritative decision exists, retaining unknown action/provenance. Match normalized identity and serialized probability to the frozen table's D rows. Record conflicts and discrepancies with W1.2's recorded selection state; do not alter the accepted W1.2 files. Genuine selected primaries, declined candidates, research-only and unresolved selections remain separate. Numerical selection consistency does not prove attempt identity, delivery or contest entry.

In §2's ball-series bullet, replace “It is produced outside this repo” with:

> Its historical seed pipeline is external; this repo contains the daily producer. The final export is retrospective and source-date-causal by construction, not proof of the bytes or source revisions available live on each date. Record hashes of both the export and its sibling manifest.

### B — Replace §3 in full

> - **Strata:** preserve the accepted W1.2 `selection_consistent × pool_verified` forecast cohort as the primary diagnostic support, with its provenance limits. Selected-primary analyses additionally require a genuine selection under §2. Empty/undefined primary support is unavailable, never replaced by a sensitivity. Report `all_dates`, `pool_surrogate` and `pool_all` separately. E7's action comparison uses registered all-date forecast diagnostics, including separate research-only and unknown-action counts.
> - **Outcomes and scores:** known means hit/no_hit. Count no_pa/unknown without scoring them as misses or replacing a winner. Require finite probabilities in [0,1]; report invalid or absent scores as coverage losses.
> - **Support and ranking:** for each paired comparison, freeze the candidate intersection with finite scores in both arms and the declared pool before using outcomes. Rank both arms on that same pool with ties by archived D `row_order`. Evaluate top-1 on the same dates where both previously selected winners have known outcomes; count unilateral unknown/no_pa exclusions. Native-surface winners and their distributions are separate diagnostics. Do not subtract native hit rates or sum adjacent changing-support effects.
> - **Sign:** define an optimism reduction `G(U,V) = hit_rate(U) − hit_rate(V)`. W1.2's `top1_diff_b_minus_a` uses the opposite sign: negate its point estimate and map [lo, hi] to [−hi, −lo]. Compute the non-adjacent A26/B26 comparison directly on its frozen common pool.
> - **Weights:** residual and AUC half contrasts are equal-date; omit and count AUC dates lacking either class. Per-date all-candidate residuals average only the declared finite-score known rows. Recalibration fits and original proper scores are candidate-row-weighted. State each denominator and distinguish scoring `n_pa` from A26's `n_pas_actual`.
> - **Resampling:** use 10,000 whole-date draws, seed 20261004, jointly for arms sharing dates. For equal-date statistics, precompute each date's statistic and average the sampled date records with multiplicity, or give every drawn block its own occurrence ID before regrouping. The existing core bootstrap does not supply occurrence IDs. Refit logistic parameters in every draw. Do not recompute `_winners` or group by the original date in a way that collapses repeated copies.
> - **Half split:** sort the accepted run's registered dates, put the median date in the first half (`date <= median`), and use later dates in the second. Preserve that outcome-independent assignment in every draw. Resample dates separately within each half at its original size, jointly across compared surfaces. Report effective dates, omitted dates and coverage by half.
> - **Failures:** undefined statistics, invalid fits and failed draws are recorded with reasons and counts. Do not silently drop them or redraw until successful. A label using a percentile interval requires all 10,000 draws to be defined; otherwise its interval and label are unavailable/undetermined. Any explicitly shown successful-draw interval is conditional on fit success and supplies no label.
> - **Interpretation and multiplicity:** 95% intervals and the null envelope are pointwise, unadjusted exploratory diagnostics. The explanation table is not a family of confirmed findings, has no simultaneous 95% guarantee, and cannot rank the explanations. The several components and sensitivities increase opportunities for chance flags. Zero/identity inclusion does not establish equivalence, good calibration, absence of an effect or luck. Intervals condition on frozen artifacts/support and exchangeable date clusters; they exclude fitting-history uncertainty, selection bias and cross-date dependence, including overlapping drag windows and seasonal trends. Sensitivities do not upgrade a primary flag.
> - **Counts:** every statistic carries date, candidate-row and known-row denominators, with effective paired/discordant dates where relevant.

### C — Replace §4's label paragraph and eight-row table

> Labels refer only to the named measured component: **consistent** is an unadjusted directional diagnostic flag; **weakened** requires evidence contradicting that component, not merely an interval covering no effect; **undetermined** covers insufficient or ambiguous evidence. **Not testable here** identifies unavailable evidence for the full plan explanation. State the component and limitation beside every label; none establishes a cause, achievable lift, equivalence or a fraction of the historical gap. E2 is a deterministic reproduction diagnostic and E7 has no automatic label.

> | # | Explanation | Computation | Label rule and limit |
> |---|---|---|---|
> | E1 | Benchmark optimism | Carry the W1.2 chain with its composite meanings and paired reranking/discordance. Compute G(A26,B26) directly; report selected-player scoring-PA distributions. Also report the surviving C-served/D contrast. | The restricted optimism component is consistent when G(A26,B26)'s interval is wholly positive, weakened when wholly negative, otherwise undetermined. A surviving C-served/D gap weakens a benchmark-only account descriptively; do not automate “similar gap” from significance patterns or different supports. No numerical attribution to the headline gap follows. |
> | E2 | Serving drift / stale inputs | Reproduce the runner's serialized-score class logic: exact_final_feed first, inferred_weather_absent_at_serve second, otherwise unexplained, tolerance 1e-9. Report numeric coverage, missing artifact/provenance, signed and absolute residuals, and class shares by month/lineup. | At least 5% unexplained among genuinely reproducible rows flags an unexplained reconstruction gap, not measured live drift. Exact or sensitivity parity with persistent outcome residuals weakens the numerical-mismatch component conditionally, retaining the inferred-weather distinction. Empty support is not testable. Historical feature age, lineup/pitcher changes and causal staleness remain unavailable; carry M3/X-03 without rerunning it. |
> | E3 | PA-opportunity forecast error | Report G(A26,A26_count), paired score changes, A26_count/B26's composite contrast, and realized scoring-PA distributions. | The plan's at-lock total/starter/reliever forecast-error test is not testable here. A positive oracle reduction is compatible with an oracle count contribution, never an achievable forecast improvement. Zero top-1 change does not weaken forecast error. |
> | E4 | Conditional hit-model drift | Report second-half minus first-half equal-date residual and AUC under C-frozen, with B26 and D companions. Separately report C-served/D numeric reproduction coverage in each half. | Increased-overprediction on numerically reproduced support may receive a conditional consistent flag only when the residual contrast interval is wholly positive; retain exact versus inferred-sensitivity support and exclusions. A wholly negative interval flags reduced overprediction; zero inclusion or unavailable parity is undetermined. AUC changes are separately signed. One evolving season, final-source context and changing composition prevent identification of conditional hit-model drift; PA-level evidence is unavailable. |
> | E5 | Calibration without ranking loss | Carry original-score Brier/log loss and reliability from W1.2. Fit unpenalized logistic outcome ~ intercept + slope × logit(p) on each surface's declared known finite rows, clipping p before transformation. Bootstrap by refitting on resampled dates. Compute D−B26 equal-date AUC on identical scoreable known rows. | Flag the calibration component when a valid intercept or slope interval excludes identity (0 or 1), with the unadjusted multiplicity limit. Identity inclusion is undetermined, not evidence of good calibration. Report ranking separately; an AUC interval covering zero is inconclusive. Without a predeclared material-loss/equivalence criterion, the joint “without ranking loss” explanation is undetermined. No same-row fitted-score improvement is presented as held-out correction performance. |
> | E6 | Ball regime | Join exact venue/date values for unique registered venues, once per venue per date, with no zero imputation or forward/back fill. Name their equal-weight mean “registered-slate as-of drag,” not league drag. Use identical finite D/C-frozen known candidate rows for their per-date residuals; also report D rank-1 residuals without reselection. Report coverage and Spearman co-movement. | A wholly positive interval flags the narrow predicted association; zero inclusion is undetermined and a wholly negative interval contradicts that narrow direction. None weakens or establishes the existence of a ball regime. The full independently dated regime/league test is not testable from these inputs. Changing venue membership, shrinkage, retrospective source revisions, trends and temporal dependence remain limits. |
> | E7 | Selection / composition | Report D within-date versus pooled AUC, rank-1 versus all-candidate residuals, and registered all-date action diagnostics (`single`/`double` versus `skip`, with research-only/unknown separate). Report projected/confirmed flags and slate-size strata using the median frozen `day_meta.n_rows` over registered dates. | Descriptive only; no automatic label. Action is recommendation action, not contest entry/play. Genuine skip dates are not forced into a selected-date cohort. Report mean-p, outcomes and coverage together so availability/composition shifts remain visible. |
> | E8 | Sampling variation | Carry paired date-block intervals and discordant counts from W1.2. On genuine selected-primary dates with known outcomes, simulate 10,000 independent Bernoulli vectors using the selected D row's served probability, seed 20261004; report the observed count, expected count and central 95% envelope. Carry Wilson intervals on the previously consumed brief counts separately. | Report conditional null-compatible or null-incompatible, assuming both calibrated served probabilities and conditional independence across dates. Compatibility does not show that luck explains the gap; incompatibility does not identify drift. Calibration circularity, temporal dependence and the different brief population remain explicit. No selected-primary support is not testable. |

Append after the table:

> A logistic fit is invalid if either outcome class is absent, the intercept/logit design is rank deficient, complete/quasi separation occurs, optimization fails to converge, coefficients are nonfinite, or the fitted information matrix is singular/nonfinite. Invalid base fits are not testable and invalid bootstrap fits follow §3's failure rule. No penalty, alternative fit or successful-draw retry is substituted after inspecting results.
>
> For E6, a date's drag mean is available only if every distinct venue contributing to its declared candidate support has a finite exact-date value and venue identity. Report incomplete dates and candidate exclusions before correlations, use the same complete date support for the D/C-frozen co-movement comparison, and report selected-player support separately. The as-of construction is verified only as a source-date rule, not historical serving availability.

### D — Replace §4's first, third, fourth and fifth trailing rules

> - E1/E3's A26, A26-count and B26 use realized information and are oracle diagnostics. E6's exported drag is source-date-causal but retrospective; neither property establishes an achievable serving change.
> - Independently dated regime event/source witnesses are not included in these inputs. The plan names May 24, park rollout and a late-July reversion; this analysis neither invents their in-window dates nor claims none exists. It does not choose change-points from our misses. The continuous slate-drag association is a narrower diagnostic and leaves the plan's full regime test unavailable.
> - E8 conditions on calibrated probabilities and independent Bernoulli outcomes across dates. Its compatibility labels do not classify luck or model drift.
> - A sensitivity never supplies a missing primary label. Report disagreement and all coverage limits; do not substitute whichever stratum yields a directional flag.

Keep the one-evolving-season rule, with the E4 qualification in the replacement table.

### E — Amend §§5–6 and §8

Append to §5:

> `results.json` records the measured component, its diagnostic flag, the full explanation's unavailable/undetermined status where applicable, support definitions, raw pointwise intervals, all fit/draw failures, and reproduction-source classes. The memo presents these fields together and does not force eight causal verdicts or treat unexplained reconstruction discrepancies as live incidents.

Replace §6's X-29 declaration and scope bullets with:

> X-29 must be published and verified before any W1.3 outcome-bearing input inspection, computation or analysis unit begins. It names the frozen revised design and code, accepted W1.2 run and file hashes, actual registered dates/candidates, component computations, support/weight/sign rules, resampling/failure rules, strata, and pointwise exploratory interpretation. It records the exact newly used feed bytes, drag export and sibling manifest; W1.2's manifest alone does not bind those versions.
> - Disclose X-21's bounded reuse permission, X-02's earlier drag screen and prohibition on a new 2026 park-feature fit, X-03's carried M3 result and rerun trigger, X-09's previously consumed brief scorecard, and X-12's research-only overlap when E7 uses those dates. This read neither refits a park feature nor reruns M3.
> - Register the new PA distributions, half contrasts, logistic diagnostic parameters, drag associations and composition diagnostics, and the conditional selected-primary null, including their missing-evidence dispositions.
> - Newly computed candidate/outcome outputs stay within the accepted W1.2 registered dates and candidates. Wilson calculations on the already consumed X-09 brief counts are a separately labelled carried-context exception, not a new full-season outcome read or a W1.1 verification.

Retain D3's no-candidate/2027 prospective-validation rule. Replace §8's fixed census and expected-label sentences with:

> The window and usable sample sizes are the accepted W1.2 inventory and each computation's reported support, not an assumed 105 valid dates. Small samples can yield wide or degenerate conditional intervals, invalid fits, and unavailable primary analyses. Undetermined or not-testable results are acceptable. No label follows from a missing or low-power test, and the pointwise exploratory flags provide no overall error-rate guarantee.
