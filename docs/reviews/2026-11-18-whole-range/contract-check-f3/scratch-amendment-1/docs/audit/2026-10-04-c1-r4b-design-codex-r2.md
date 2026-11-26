# C1 rank 4b — final design review

## Verdict

**SIGN WITH EDITS.** Apply the four minimal edit groups below verbatim, verify their application mechanically, then freeze. No third design round is needed. They correct ambiguities in my round-one replacement text; they do not add a candidate, stress grid, acceptance threshold or owner decision.

Reviewed rev 2 at main 97d0e981de59904cd85b432210ae7a7cc8302c5f; registration SHA-256 1dccad3999237c706e93de730a39c9ceb03405fcf27f904bf87197ab30bbeb91. Every one of the twelve Markdown replacement blocks in E1–E7 occurs exactly once in rev 2. The archived round-one report is byte-identical to the original, SHA-256 e482be93df098fd5a3421dbb91400245b76dc619a1e233acb14a20f205282a98.

Main advanced concurrently to 4bdc049f3366726c4becc7acc0b33eec6da3999e. Final verification found the reviewed 4b registration, authorities and source code unchanged; the cycle index's 4b gate is unchanged. The tracked worktree remained clean.

The remaining corrections concern coherent availability-conditioned rates, zero-frequency refinement cells, Monte Carlo disposition precedence, and the final artifact's horizon/projection convention. With those corrected, the historical registration is executable and appropriately qualified under either answer to Eric's pending 2027 question. That answer remains Eric's: the design retains both a separately reviewed prospective contract and an explicit recorded amendment as alternatives. This signature decides neither one and approves no run, commit, push, deployment or activation.

Code/document inspection and small synthetic Python checks only; no data/ reads, historical replay, network, ssh or gh. No tracked file was edited. This is the last design review under the pace rule; subsequent code and result acceptance still have their already-declared gates.

## Findings

### Round-one dispositions

| Finding | Verified disposition in rev 2 |
|---|---|
| F1 — misleading decomposition / bundled positive | **Resolved:** lines 22–26 retain matched tail continuation and separate the adaptation, objective and switch contrasts. Lines 59–63 require both objective and switch evidence, with fixed regression limits. A1 has no separate ship pathway. |
| F2 — exposure and 2027 gate | **Resolved as a registration:** lines 5, 10, 30, 34, 70 and 74 disclose historical exposure and LOSO's nonchronological nature and preserve the two owner-authorized paths. Eric's choice is pending, not a reason to block this historical design's freeze. |
| F3 — final artifact not bound to the trade | **Resolved in intent:** lines 35 and 86–90 fix an all-five-season decision object before D7 and bind packaging to its hashes. Edit E3 supplies the last missing horizon and final-projection convention. |
| F4 — calendar, availability and classifier fidelity | **Resolved:** lines 31–42 and 75 fix calendar time, A0's own cutpoints/cap, legal demotion, coverage classification and equal application of fixes. Edit E1 makes the fitted availability kernel mathematically coherent. |
| F5 — comparable jackpot environment / dependence stress | **Resolved in intent:** lines 50–53 retain frozen classifiers, a common environment, explicit calibration/dependence scenarios and qualified paired jackpot differences. Edit E1 distinguishes a harmless zero-weight cell from an unsupported required rate. |
| F6 — m-independent validation blind spot | **Resolved:** line 54 limits the solver equality check and requires independent m-dependent path enumeration and the other declared boundary cases, including the whole-season phase-aware objective extension. These are future implementation requirements, not claims of completed verification. |
| F7 — effective sample, stress execution, guardrails | **Resolved in intent:** lines 44–48 and 57–74 freeze policies across stress, use season reporting, add both contrasts and regression dispositions, and avoid an independent-n=120 claim. Edit E2 makes the Monte Carlo rule and its precedence unambiguous. |
| F8 — provenance, consequences, links and overstated claims | **Resolved:** lines 4, 11, 29–32, 55, 78–84 and 93 fix the link, provenance, conditional information set, two visitation distributions, qualified headline and unmeasured compute sizing. Edit E4 removes an obsolete cross-reference. |

Two original limit bullets (“Information” and “P(57)”) were removed when the limits were replaced, beyond the literal instruction to replace the effective-sample and optimism bullets individually. Their substantive qualifications remain explicit in §§3, 5 and 8; this creates no remaining blocker and needs no restoration.

### R2-1 — correct the mixed denominators and zero-weight-cell treatment

**Locations:** lines 33 and 51. This defect came from my round-one wording.

Line 33 estimates p_hit on all primary-available days but p_both only on partner-eligible days. Those are different populations. On two synthetic dates, one partner-eligible primary/joint hit and one partnerless primary miss, the specified pair is p_hit=0.5, p_both=1.0. The actual _validate_rates in src/bts/simulate/tail_policy.py:144–155 rejects it. A naive one-day double value would be 2.0 streaks; legal execution on the two equally weighted types gives 1.0. Sharing this bad environment between A1 and A2 does not make it a coherent probability model.

Condition primary and joint rates on the same availability stratum, and integrate legal +1 versus +2 transitions in the optimizing kernels. This preserves a five-quality-bin raw-action table followed by the shared execution clamp; it does not add availability-specific candidate selection.

Separately, line 51 can be read as refusing a projection whenever any refinement cell is empty, including a cell with zero environment frequency. A common refinement naturally has impossible intersections and unobserved cells. The existing joint-environment precedent deliberately retains them at frequency zero (scripts/audit/dd_p_policy_value_sensitivity.py:173–197). A zero-weight cell contributes nothing and needs no conditional rate; a positive-weight cell needing an unsupported rate is different. Edit E1 fixes both distinctions and makes the no-primary case explicit.

### R2-2 — the Monte Carlo rule conflicts with deterministic equality and negative dispositions

**Locations:** lines 45, 64 and 66. This defect also came from my replacement text.

At Δ=0 there is no thinning uncertainty. A synthetic exact gain of +0.25 passes the registered ≥ +0.25 bar, but an inclusive “within two SEs” rule with MC SE zero calls that exact same value inconclusive. Conversely, zero sample variance from 200 stochastic replicates does not prove a boundary statistic is deterministic. The design also does not specify whether a sampled reach-20 regression within the ambiguity band is negative (line 66) or inconclusive (line 45).

Edit E2 specifies paired-replicate aggregation and MC SE, retains ordinary inclusive comparisons for deterministic cells, conservatively handles an estimated-zero-SE stochastic boundary, and gives incomplete/invalid/ambiguous evidence precedence over positive or negative classification. It leaves every numerical threshold, seed and replicate count unchanged.

### R2-3 — freeze the final horizon and projection environment rather than choosing them after results

**Locations:** lines 35, 39, 80 and 87.

Fold policies have their own season horizons, but an all-five-season artifact has no single “actual season horizon.” The final fitting rule does not choose one. Nor does “its own projection table” specify whether the final fit is projected on pooled rates, separate seasonal rates, a selected calendar, or a future 2027 horizon. Those choices affect the concrete trade Eric sees, despite the correct hash-binding requirement.

Edit E3 fixes the final table horizon from outcome-free historical calendars, fixes the final environment to the all-five-season pool, evaluates every historical calendar and reports all five plus their equal-season summary. These final projections remain exposed-fit model projections. It also makes later 2027 calendar evaluation use that same frozen table with its declared cap; it does not refit the artifact or settle the pending prospective gate.

### Checks that require no further edit

- **Parity symbols exist under exactly the registered names.** Imported the reviewed July module read-only: ANCHOR_DIFFGAME has the deployed-base, always-single and always-double anchors; ANCHOR_TOL is {mean_max: 0.005, reach20: 0.0005, resets: 0.05} (dd_p_policy_value_sensitivity.py:91–96). The original assertion adds 1e-9 (:838–843), so “the original implementation's tolerances” already specifies its arithmetic slack. I did not execute outcome-bearing parity.
- **The dependence scenario is finite and coherent.** For the r=8 transition, the order is calibration, analytic Δ, then the h reduction and joint-rate ratio. With availability-conditioned rates, the three categories remain nonnegative and sum to one; synthetic boundary/interior checks confirm this, including zero primary probability. At h=0 the transform equals the iid case. Updating r on unplayed primary outcomes makes it exogenous to policy actions; no-primary/no-opportunity resets r and executes skip. The scenario changes marginal calibration as disclosed. It does not identify the true dependence mechanism or resolve the true jackpot cost.
- **The matched tail does not invalidate the A1 solver equality check.** Once s+2d<57, every legal action still has zero jackpot value. Equality is required only on synthetic matching environments and an all-partners-available kernel, where the ordinary solver is an applicable oracle. The m-dependent and legal-demotion tests remain separately mandatory.
- **Common refinement does not re-bin a policy.** Each arm still looks up its original/fitted quality index, and both receive the same evaluated type distribution. The required identity is the policy's classifier mapping, not equal quintile occupancy in the evaluation population.
- **Final approval remains concrete under either owner answer.** A prospective contract can test the frozen final object; an explicit owner amendment can change the applicable validation gate without changing this historical recipe. Neither branch authorizes selecting another objective/bin scheme, removing stress scenarios or changing the artifact after approval.

## Verbatim edits

Apply these to rev 2. No tracked edits were made by this review.

### E1 — replace the “Rates” and “Comparable environment” bullets

Replace the §3 “Rates” bullet (line 33) with:

~~~markdown
- **Rates and legal fitting kernel:** within each phase and quality bin, condition primary and joint-success rates on the same primary/partner-availability stratum. On partner-eligible days estimate both p_hit and p_both from those same days, so 0 ≤ p_both ≤ p_hit ≤ 1. On primary-available partnerless days estimate their own p_hit; there is no joint outcome and the encoded p_both is zero. A primary-absent type has no quality bin and forces skip. Estimate type frequencies on evidenced calendar-opportunity dates, retaining observed primary absence; known no-opportunity calendar dates are handled separately as forced skips. Both optimizing solvers use the same fitted phase-aware iid type mixture: for a raw action at a quality bin, integrate its legal execution over availability, so a partnerless double has the single's +1 hit/miss transition and an eligible double has the joint +2 transition. The raw policy table remains indexed by quality, streak, best where applicable, days and saver; availability only applies the common execution clamp. Do not average partnerless outcomes into a +2 success probability, mix rate denominators, or select a different availability approximation after results. Known calendar no-opportunity dates are enforced in replay and projection; the optimizing type model's iid approximation is disclosed.
~~~

Replace the §5 “Comparable environment” bullet (line 51) with:

~~~markdown
- **Comparable environment:** for each fold, estimate rates and frequencies twice: from fitting seasons and from its held-out season. Classify both through frozen cutpoints. On primary-available dates use the common refinement of A0's saved bins, A1/A2's bins and partner availability, estimating primary and joint rates within the same type as in §3. Primary-absent types have no quality bin and force skip. Every arm receives this same phase-aware environment and executes through its own frozen classifier and the legal clamp. No policy's original or embedded training rates stand in for another arm's evaluation environment. Retain zero-frequency refinement/availability cells at zero weight; no conditional rate is required or imputed for them. This does not relax §3's fitting-quality-bin stop rule. A positive-weight type requiring an unsupported outcome rate makes the relevant projection unavailable. Evidenced calendar no-opportunity dates force skip and reset r without a quality/outcome draw. On other dates, environment types are sampled independently within phase in the iid case; the dependence grid changes outcome transitions through r. Report paired A2−A0 and A2−A1 jackpot differences in probability and percentage-point units; relative changes are unavailable when the reference is zero.
~~~

### E2 — replace “Randomness and precision” and the disposition paragraph

Replace the §4 “Randomness and precision” bullet (line 45) with:

~~~markdown
- **Randomness and precision:** use one outcome mask for every arm on a given seed/date/replicate; reuse uniform draws across Δ for nested thinning. Bind streams to stable seed/season identity, not filesystem enumeration. Report achieved marginal and primary-hit-conditional haircuts separately from season variation; Δ calibrates the thinning probability to fitting-season r_bar, so the held-out expected marginal reduction need not equal Δ. Average replicates inside each seed before season aggregation. For each stressed gate contrast, also form its statistic T_r separately in each of the 200 independent replicate sets: pair arms on the same masks, average seeds within each season, then weight seasons equally. Its Monte Carlo SE is sd(T_r, ddof=1)/sqrt(200); it measures thinning error conditional on the fixed historical inputs, not historical sampling uncertainty. For a stochastic gate, distance from its threshold ≤ two Monte Carlo SEs makes its disposition inconclusive. A sampled zero SE at exact equality is still inconclusive unless the gate statistic is proved mask-invariant. Δ=0 and proved mask-invariant gate statistics use the registered strict/inclusive comparisons directly. Replicate count and seed are not changed after seeing the result.
~~~

Replace the §6 paragraph beginning “Negative” (line 66) with:

~~~markdown
**Disposition precedence:** incomplete coverage, failed required validation or a Monte Carlo-ambiguous gate is **inconclusive** before applying numerical positive/negative rules; failed pre-result checks also stop the run under §7. Otherwise **positive** requires all six conditions above. Otherwise **negative** means either mean streak contrast is ≤0 at Δ=0 or a registered reach-20/individual-season regression guardrail is breached. All other cases are **inconclusive**. No disposition selects another candidate or permits retuning.
~~~

### E3 — replace the “Population and final fit” bullet and insert a final-projection paragraph

Replace the §3 “Population and final fit” bullet (line 35) with:

~~~markdown
- **Population and final fit:** average seeds within each season, then weight the five seasons equally. Freeze the final decision-object rule now: the same five-bin, 30-day-phase recipe fitted once on all 2021–2025 profiles, with D_final equal to the maximum opening-to-exclusive-end calendar-day horizon among the five outcome-free historical calendar manifests. The final action table covers d=0…D_final; lookup uses min(raw calendar days left, D_final), with the late phase at effective d≤30. Fold A1/A2 solves retain their held-out season's registered horizon under §4. The final artifact, manifest and separately labelled projections are produced before D7. Fold results evaluate the fitting procedure; they are not held-out measurements of that final all-season fit. A later calendar, including 2027, evaluates this same final table and cap without a policy re-fit; the applicable prospective validation contract and D7 remain separate.
~~~

In §8, after the paragraph beginning “A separate projection table” (line 80), insert:

~~~markdown
For the final decision-object table, fit one common-refinement evaluation environment on all five seasons pooled through A0's and the final object's frozen classifiers. Evaluate A0, final A2 and its matched final A1 on each of the five registered historical calendars, using their declared horizon caps and the full c×h×Δ grid. Report all five calendar projections and their equal-season summaries and paired differences. These use all-season fitted rates and policies and are exposed-fit projections, never held-out evidence; no calendar or rate version is selected after results.
~~~

### E4 — replace the obsolete code cross-reference

Replace the §10 “Code” bullet (line 95) with:

~~~markdown
- **Code:** scripts/audit/c1_r4b/, written test-first. The calendar/legal-demotion/m-state/A0-tail replay corrections, availability-conditioned fitting kernel, phase-aware whole-season E[best] solver and fixed-policy P(57)/r-state evaluator each get failing synthetic tests before implementation, including the independent validation cases in §5. Code review precedes outcome-bearing execution.
~~~
