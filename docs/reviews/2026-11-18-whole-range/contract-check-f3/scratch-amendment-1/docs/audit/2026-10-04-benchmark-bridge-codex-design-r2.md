## Verdict

**SIGN WITH EDITS.** Apply the small verbatim edits below, then freeze the design with its stated limits. No third design round, additional experiment campaign, or mandatory B-frozen surface is needed.

The substantive r1 fixes landed in the intended sections. The remaining issues are a residual-cause inference rule inherited from my own r1 wording and surviving original text that conflicts with the revised definitions. Both can be fixed entirely in prose. Missing historical witnesses, incomplete artifact coverage, composite contrasts and conditional intervals remain limits; they are not reopened as build requirements.

Reviewed `main` HEAD `ac554e9b5d947a800585a7acce6fa80a6df30310`; design sha256 `c511678a7714dbe36ac044d5a5e8b3fc8ef9f869f33130db64cf972bf5080b15`. The archived r1 report and the original report both have sha256 `878e66851c332cc9bbbffb38cfefdabc98db8d22b1c2032399adccae5a4907a3`. Verified that `src/`, `tests/`, `scripts/`, dependency declarations, the wrap plan and exposure register did not change between the reviewed revisions. This round inspected the revised text, application script, revision diff and relevant code; it did not rerun unchanged r1 probes. No `data/` reads, network, SSH, or tracked-file edits. Actual date counts, pickle availability and serving parity remain unverified.

At final verification, untracked `scripts/audit/benchmark_bridge/` and `tests/scripts/test_benchmark_bridge.py` had appeared in the shared checkout. They were neither inspected nor modified in this design review. The tracked diff remained empty and the reviewed design hash was unchanged; this verdict does not review the new implementation.

## F1-F9

All design references below are to `docs/superpowers/specs/2026-10-04-benchmark-bridge-design.md` at the reviewed revision.

| Finding | Disposition | Evidence in rev 2 and remaining limit |
|---|---|---|
| **F1 — selection binding** | **Resolved subject to E2** | Line 21 uses the slate serializer, normalizes identity, gives authoritative decisions precedence, records conflicts and explicitly limits the inference to selection consistency. Research-only dates no longer borrow a stale selection's model identity. Line 37 permits serialization error. The old “unbound slates” exclusion at line 82 must be replaced so it cannot impose the stronger binding criterion that the revised design withdrew. Full attempt identity remains unknown where unwitnessed. |
| **F2 — count-only contrast** | **Resolved** | Line 31 uses geometric no-hit-rate normalization and preserves A26 when the counts agree. Line 35 labels the next step as context plus ensemble convention. A26's row at line 30 no longer enables the unnecessary PA log. The count normalization remains an oracle diagnostic, not an achievable PA forecast or a clean context contrast. |
| **F3 — coefficient/input confounding** | **Resolved** | Line 35 pins the daily-learning recipe and full-season retraining calendar, requires identical fitted models for A/B, separates archived production from a frozen recipe, and labels mixed transitions. It explicitly leaves the isolated input effect unresolved without the optional same-artifact rescoring. Lines 37 and 62 prevent using parity or changing pairwise support to manufacture a causal decomposition. |
| **F4 — reconstruction/provenance** | **Resolved subject to E1 and E2** | Lines 27–28 specify the top-level hash, `_model` extraction, provenance checks, pre-date lookup/opener frame and slot sources. Lines 37 and 85 distinguish numerical parity from historical availability/identity; line 75 isolates caches and external dependencies. However, line 37's “or a controlled substitution” still lets an unwitnessed historical cause receive a verified label. E1 narrows that rule; E2 removes the surviving at-cutoff/as-of shorthand at line 13. Unrecoverable inputs remain limits. |
| **F5 — eligibility/actual decisions** | **Resolved subject to E3** | Line 43 separates historical eligibility evidence, scheduled-time surrogates and all-row diagnostics, and retains actual D actions alongside diagnostic argmaxes. Lines 60–62 apply shared eligibility/support. E3 reconciles the old metrics header with these rules and states that absent verified eligibility cannot be silently replaced by the surrogate in a primary forecast table. No selection parity is claimed from score parity. |
| **F6 — coverage/outcomes/pairing** | **Resolved subject to E2 and E3** | Lines 19–21 make the census provisional and define missingness/strata. Line 40 adds unknown outcomes, requires source completeness for no_pa and ranks before the join. Line 60 fixes common scoreable pools, paired nonvoid dates, no reselection and denominator reporting. E2 removes the table's surviving promise that `top_n` can retain every batter; E3 removes the header suggesting top-1 is selected from an outcome-filtered pool. Common-support results remain conditional diagnostics. |
| **F7 — exposure** | **Resolved** | Line 71 puts X-21 before fitting/A/B generation and all other outcome-bearing work, names the candidate-level exposure and whole-season training basis, restricts disclosed outputs and preserves 2027 prospective validation. Line 75 places the publication gate before the analysis unit. The previous claim that these outcomes were already consumed is gone. This is a specified future gate, not evidence that registration or execution has occurred. |
| **F8 — statistics** | **Resolved** | Line 62 specifies equal-date within-slate AUC, candidate-weighted proper scores, joint whole-date resampling with multiplicities, percentile intervals, omission of candidate Wilson intervals, effective counts and conditional assumptions. Cross-date dependence, fitting uncertainty and selection bias are expressly excluded from the interval's scope. No new inference gate is needed. |
| **F9 — headline attribution** | **Resolved** | Line 8 withdraws the historical-headline reconstruction and fraction-explained claim; lines 62 and 85 prohibit telescoping changing-support contrasts and keep the headline as noncomparable context. The bridge answers the bounded diagnostic question rather than reproducing full-season delivered/contest rates. |

## New findings

### N1 — P1: An unwitnessed substitution can still be called a verified historical cause

**Location:** `docs/superpowers/specs/2026-10-04-benchmark-bridge-design.md:37`.

“Call a cause verified only when supported by an archived live witness **or a controlled substitution**” is too permissive. This wording came from my r1 edit. A hypothetical weather, lookup or pitcher-hand substitution can move C to D's score without establishing that the substituted value occurred live. Several inputs and the fitted scoring path contribute to the same scalar (`src/bts/model/predict.py:737–779`); matching that scalar does not identify the historical input. Likewise, observing that a historical input differed does not by itself isolate its numerical contribution when other inputs differ too.

This is a **reasoning-only identification problem**, not a measured production incident. It would make the reproduction-classification report misleading even though the design correctly limits global causal attribution elsewhere.

**Smallest fix:** require independent historical evidence for a verified historical value/behavior and an isolated numerical comparison for its contribution. A substitution without the historical evidence is numerical sensitivity and retains an inferred/unexplained historical label. E1 makes this change without requiring acquisition of missing witnesses.

### N2 — P2: Surviving original labels conflict with the new surface, cohort and metric contracts

**Locations:** design `:13`, `:29`, `:46–47`, `:82`.

- Line 13 still calls C “at-cutoff” with “as-of lookups,” although lines 27 and 85 expressly allow final-source reconstruction with unproved historical availability.
- Line 82 still excludes “unbound slates,” although line 21's primary selected-date cohort requires selection consistency and can have unknown attempt provenance. Reading binding as a stronger eligibility requirement changes the population; reading it as proof of attempt identity overstates the evidence.
- Line 29 says `top_n` keeps “every batter.” The mode first removes batter-games without a starter matchup (`src/bts/simulate/backtest_blend.py:620–653`); `top_n` can remove truncation only after that selection. The revised coverage contract is correct, so this is a conflicting table promise rather than a requirement to add scores for unsupported rows.
- The surviving line-46 header places all per-surface metrics “on the shared hit/no_hit rows,” including top-1. Read literally, it filters the ranking pool by outcomes, contradicting lines 40 and 60. It also fails to distinguish verified eligibility from surrogate/all-row diagnostic pools. The detailed paragraphs are correct, but a builder should not have to choose which instruction governs the table.

**Smallest fix:** E2 updates those surface/cohort labels; E3 explicitly separates selection from outcome evaluation and names the metric populations. If no verified eligibility population exists, report that primary forecast table as unavailable and retain separately labelled diagnostics. No new historical evidence or broader build is required.

No other newly introduced problem would invalidate the specified bounded descriptive bridge. The edits landed in the appropriate sections; the stated unknowns and optional decomposition are accepted limits under the final-round pace rule.

## Verbatim edits

Only the following design text changes are required. This review does not edit the tracked design.

### E1 — Replace the two residual-cause sentences within §3's reproduction paragraph

Replace:

> Call a cause verified only when supported by an archived live witness or a controlled substitution; otherwise label it inferred or unexplained. Projected/confirmed state alone is not a verified numeric cause.

With:

> Call a historical residual cause verified only when independent archival evidence establishes the relevant live value or behavior and a comparison isolates its numerical contribution. A controlled substitution of an unobserved historical input establishes numerical sensitivity, not what occurred live; label its historical explanation inferred or unexplained. An archived input difference alone does not isolate its contribution when other inputs also differ. Projected/confirmed state alone is not a verified numeric cause. Missing historical evidence remains a stated limit rather than an acquisition requirement.

### E2 — Replace three surviving surface/cohort statements

Replace the C bullet in §1:

> - **C, at-cutoff reconstruction:** the served lineup and starter inputs and the as-of lookups;

With:

> - **C, archived-input reconstruction:** the slate's recorded lineup and starter inputs, with the declared pre-date lookups and final-source context; historical availability is verified only where existing witnesses establish it;

Replace the B26 table row in §3:

> | **B26, estimated-PA (oracle)** | `blend_walk_forward` over 2026, mode `estimated_pa`, with `top_n` set to keep every batter | one deterministic run |

With:

> | **B26, estimated-PA (oracle)** | `blend_walk_forward` over the original 2026 calendar, mode `estimated_pa`, with `top_n` large enough to retain every batter-game emitted by that mode; absent starter-matchup rows remain coverage losses | one deterministic run |

Replace the “Unbound slates” bullet in §10:

> - **Unbound slates** are excluded from the primary analysis.

With:

> - **Selection consistency:** selection-inconsistent dates are excluded from the primary selected-date cohort; research-only/no-selection dates are a separate stratum. Unknown prediction-attempt identity is reported as a limit and is not an additional cohort exclusion or an identity claim.

### E3 — Replace §6's per-surface heading and top-1 bullet; append one sentence to §5

Replace:

> - **Per surface, on the shared hit/no_hit rows:**
>   - top-1 hit rate (rank-1, eligible; void counted separately);

With:

> - **Per surface, with populations defined by §§4–5:** the primary forecast table uses the reconstructed verified-eligibility pool. Scheduled-time-surrogate and all-row pools produce separately labelled diagnostic tables. AUC, proper scores, residuals and reliability use known hit/no_hit rows within the declared pool. For each pool:
>   - select rank-1 among finite-score candidates before the outcome join; compute its hit rate on selected rows with known hit/no_hit labels, and report selected no_pa and unknown rows separately without reselection;

The remaining AUC, Brier/log-loss, residual and reliability bullets stay in place and use the known hit/no_hit rows of the declared pool, as specified in §6's existing weighting and pairing paragraphs.

Append to §5:

> If no verified eligibility population can be reconstructed, report the primary forecast table as unavailable with its coverage reason; publish the separately labelled surrogate/all-row diagnostics without substituting them for that table.
