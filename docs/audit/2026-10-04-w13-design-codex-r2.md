## Verdict

**SIGN WITH EDITS.** Apply the three minimal clarifications below, then freeze with the stated limits. No third design round, new input, surface rescoring, or experiment campaign is required. Two clarifications finish the mapping from a reported interval to its flag; the third restores the clipping constant and boundary counts lost in my round-one replacement. These gaps came from my r1 wording, not from the author's adaptation.

Reviewed HEAD `6d1ca9f638d3ed564173db618f86dcfd2579f0b7`; revised design SHA256 `b71283390e752d6a17c69dec44dc9c92c9ac4c196680dffdcac9251650523246`. The archived r1 report and original report both have SHA256 `c7eb37171a301c1dbdb7ac189d5716cd1e29d58d2c21d1629c87e0d3af1d5264`. Checked the revision diff, application script, revised text and actual W1.2 helpers/runner; ran read-only in-memory Python for application checks and the two remaining support/sign ambiguities. No tracked-file edits, `data/` reads, W1.2 output inspection, SSH or network. Actual census, accepted-run coverage and all 2026 results remain **unverified**.

During final shared-checkout verification, another session advanced HEAD through `6af086b4ea6b772e4ad89e412b4ae21828b49165` and `b4154a8`. Inspected the intervening W1.2 diff: it optimizes the pre-date history slice, without changing the helpers, fields or reproduction/flag logic reviewed here. A W1.3 core/runner and tests also appeared concurrently; they were not inspected in this design review. The reviewed design and archived r1 hashes remained unchanged; the tracked worktree and index were clean. This verdict reviews the design, not that new implementation, the concurrent optimization or any run outputs.

## Findings

### R1 dispositions and application check

Mechanically verified that every quoted replacement block A–E appears in the revised design, with the four D bullets checked individually around the retained E4 rule. The adaptation retains all of A's new provenance text and the original four-seam, strict-prior-date construction facts. It does not garble the producer attribution, sign or as-of qualification. The three author edits are consistent: §2 no longer claims an unfinished run already exists; §8 distinguishes the external seed pipeline from the repo's daily producer; §8 no longer presents B26 as at-lock PA evidence.

| R1 finding | Final disposition |
|---|---|
| F1: sign and common support | Resolved: explicit earlier-minus-later G, correct interval reversal, direct A26/B26 computation, common candidate/date support and separate native winners. |
| F2: repeated blocks and failures | Resolved as a specification: date-statistic multiplicities or occurrence IDs, half-specific resampling, bootstrap refits and fail-closed draw handling. The builder must implement the adapter; the existing helper has not changed to supply occurrence IDs. |
| F3: zero inclusion and multiplicity | Resolved within the adopted limit: zero/identity inclusion is inconclusive; all flags are explicitly pointwise exploratory, with no simultaneous error guarantee or sensitivity upgrade. This is not approval of confirmatory eight-explanation inference. |
| F4: PA forecast error | Resolved: full E3 is not testable; the surviving contrast is an oracle count diagnostic, with scoring n_pa distinguished from n_pas_actual. |
| F5: reconstruction/drift attribution | Historical attribution is correctly limited; E4 needs edit E1 below to bind its flag to its reproduced-row contrast. |
| F6: recalibration/ranking | Fit validity, refits, failure handling and separate ranking/calibration conclusions are resolved. Edit E3 restores the numerical clipping policy. The joint ranking-preservation claim remains undetermined. |
| F7: drag target/as-of | Source-date causality, retrospective availability, unique venue weighting, complete-date coverage and the unavailable full regime test are explicit. Edit E2 identifies the primary correlation that supplies the flag. |
| F8: skip/primary/null | Resolved: an authoritative skip blocks fallback; E7 has an all-date action diagnostic; E8 requires a genuine selection, calibration and conditional independence. Unknown provenance does not become contest-entry evidence. |
| F9: exposure/window/context | Resolved: accepted-run inventory replaces the assumed census, X-29 precedes outcome-bearing inspection, required overlaps and prohibited reopenings are disclosed, newly used source bytes are hashed, and the carried brief counts have an explicit bounded exception. |

### N1 — E4's flag must use a contrast computed on the reproduced rows

**Location:** revised design `:38`, together with `:19`, `:23–26`.

The computation cell asks for the C-frozen half contrast and separately reported reproduction coverage. The label cell requires a contrast “on numerically reproduced support.” The builder must not take an interval from the full primary population and use the existence of some parity rows as permission to flag it.

A synthetic four-row table, passed through the actual serialized-probability helper, had a **+0.85** C-frozen residual half contrast on the full population, but **0.0** on the two reproduced rows. The entire increase occurred on unexplained rows. These are synthetic point estimates, not 2026 results or a power claim. They show why the label's row mask must also be the interval's row mask.

**Required fix:** E1 freezes the numerical support from scores/classes, then evaluates the known outcomes on that support. It preserves full-support contrasts as separately labelled descriptions. Empty support in either half means an unavailable flag.

### N2 — E6's flag does not identify which correlation supplies its interval

**Location:** revised design `:40`, `:46`, `:52`.

The row computes drag co-movement with D, C-frozen and D rank-1 residuals, but says only “a wholly positive interval.” My r1 replacement omitted the original row's explicit primary `(a)` association. An implementation could select whichever companion points the expected way. On the same synthetic dates/outcomes, the actual Spearman function returned **+1.0** for D residuals and **−1.0** for C-frozen residuals. These are point correlations only; no valid confidence interval is claimed for that tiny fixture.

**Required fix:** E2 restores D all-candidate residual co-movement as the primary component; C-frozen and selected-player associations remain reported companions. Freeze venue membership from the declared scoreable pool before outcome filtering. This adds no computation or input.

### N3 — E5's numerical clipping policy was dropped

**Location:** revised design `:39`, `:44`.

Probability-before-logit clipping is now correct, but its epsilon and boundary-count reporting disappeared when my replacement table replaced the original row. Endpoint logits otherwise require an implementation choice that can materially change the fitted slope. E3 restores the original `1e-15` probability bounds and counts; the existing separation/identifiability/failure rules continue to govern. This is a small reproducibility clarification, not a new calibration analysis.

### Remaining builder obligations against the real W1.2 code

The revised design is buildable with these consumer-side adapters. It does not require modifying or re-running W1.2:

- The table supplies all six surfaces, `C_served_wblank`, `n_pa`, `n_pas_actual`, `lineup_realized`, `projected`, `sel_state`, `row_order` and the three pool booleans. `summary.json` supplies `registered_dates`, `day_meta`, reproduction coverage/status/counts and surface/pair metrics (`run.py:197–206`, `:251–253`, `:270–286`, `:288–327`). Join day metadata by date; `n_rows` and action are metadata fields, not candidate columns.
- There is no row-level reproduction-class column. Derive it from the frozen table using `serialized_probability(C_served) − D` and its weather-blank counterpart, tolerance `1e-9`; exact wins over inferred-weather when both match (`run.py:288–295`). Never turn the inferred class into a verified historical cause. Missing/nonfinite reconstruction is missing coverage, not an unexplained numeric observation.
- `CHAIN` lacks A26/B26. Compute it directly with the common-pool/reranking rules. `pair_metrics` uses later-minus-earlier; reverse only G's top-1 estimate/interval, not every score or proper-score contrast (`report.py:58–91`). E5's D−B26 AUC keeps its explicitly stated sign.
- `date_block_bootstrap` preserves copied rows but not copied date identities, and does not catch invalid fits (`core.py:123–135`). Use precomputed daily statistics with multiplicities or a local occurrence-ID adapter; implement half-specific sampling and the specified failure counts. Keep compared arms on the same draws. Do not reuse `_winners` inside a draw if grouping by the original date would collapse copies.
- The raw `decision_primary` → `selection_consistency` sequence still falls back to a pick on a skip (`run.py:93–96`, `core.py:39`). Apply §2's authoritative-skip guard before calling helpers. Validate/rederive selections from the manifest-hashed bytes; retain discrepancies with frozen sel_state and do not rewrite the accepted run. `day_meta` has no research-only flag or contest-entry witness. Classify research-only only where the allowed input evidence establishes it; otherwise retain unknown/unresolved status rather than inferring it from a missing pick or action.
- Restrict every fresh computation to the accepted registered inventory and report absent primary support as unavailable. Verify the accepted files and newly used feed/export/manifest bytes; the W1.2 manifest does not hash feeds or drag artifacts. The X-29 gate must run before reading outcome-bearing W1.2 inputs. The already consumed brief counts are carried from the named plan/X-09 context, not recomputed from ledger outcomes.

These are implementation obligations already within the revised scope. They should receive synthetic failure-path and support/multiplicity fixtures during the build/code review. No additional design campaign or missing-history acquisition is required. Surviving header shorthand about “each explanation” or “eight labels” is governed by §4/§5's explicit component flags, not-testable statuses and E7's descriptive-only row; it does not require eight categorical causal verdicts.

## Verbatim edits

### E1 — Append immediately after the eight-row table, before the logistic-fit paragraph

> For E4's flag, first freeze the primary `sel_state == "selection_consistent"` and `pool_verified` rows with finite C-frozen scores and an E2 class of `exact_final_feed` or `inferred_weather_absent_at_serve`, computed from the serialized-score tolerance rule. Then use their known hit/no_hit outcomes to compute the C-frozen residual half contrast and its interval. The full-primary contrast is a separate description and never supplies this reproduced-support flag. Report exact/inferred counts and exclusions by half. For this residual bootstrap, sample each half's dates having at least one known supported row at that half's original effective date count; preserve the registered-date median assignment. If either half has no such dates, the flag is unavailable/undetermined. The flag remains conditional on numerical reproduction and does not establish historical input parity.

### E2 — Replace E6's first label-rule sentence and append one sentence to its coverage paragraph

Replace:

> A wholly positive interval flags the narrow predicted association; zero inclusion is undetermined and a wholly negative interval contradicts that narrow direction.

With:

> Only the correlation between registered-slate as-of drag and the D all-candidate per-date residual on the primary support supplies E6's directional component flag: a wholly positive interval is consistent, zero inclusion is undetermined, and a wholly negative interval contradicts that narrow direction. C-frozen and D rank-1 correlations are separately reported companions and cannot supply or upgrade that flag.

Append to the paragraph beginning “For E6, a date's drag mean is available only if”:

> Freeze each date's venue membership from the declared primary pool's common finite D/C-frozen candidate rows before outcome filtering; average once per distinct venue, then evaluate residuals on the known rows and apply the stated complete-date rule.

The same declared recipe applies to separately labelled pool/cohort sensitivities; none supplies a missing primary flag.

### E3 — Replace E5's clipping phrase

Replace:

> clipping p before transformation.

With:

> clipping p to [1e-15, 1−1e-15] before the logit transformation and reporting the counts clipped at each boundary.

After these edits, freeze. Keep the expressly unavailable full E3/E6 tests, the undetermined joint E5 claim, conditional reproduction/seasonal reasoning, pointwise multiplicity limit and E8 calibration/independence assumptions visible in every report.
