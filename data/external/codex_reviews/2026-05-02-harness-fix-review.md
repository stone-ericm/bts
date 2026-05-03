# Codex Follow-up Review — Harness Fix Verification

**Date**: 2026-05-02 ~14:35 ET
**Model**: GPT-5.5 (high reasoning) via OpenAI Codex CLI
**Trigger**: Verify the 5 fixes made in response to the prior Codex review (commit `3a2c201`) actually addressed the original concerns and didn't introduce new bugs.
**Cost**: ~$1.50 of GPT-5.5 high-reasoning tokens
**Outcome**: BLOCK on three subtler issues. Fix 1 clean. Fix 2 partial (block-ordering artifact). Fix 3 partial (non-rejection DEFENDED gate at wide CI). Fix 4 partial (marginal inconsistency). Fix 5 partial (JSON field names still claim DR-OPE).

---

## Codex output (verbatim)

**Fix 1**
Verdict: addresses concern

Reasoning: Selecting `canonical_profiles = profiles[profiles["seed"] == canonical_seed]` restores binary `top1_hit/top2_hit` inputs for `pair_residual_correlation`, so the permutation null is no longer operating on seed-averaged fractional outcomes. Seed `42` is not inherently biased if seeds are exchangeable or if it is the production seed; the main residual concern is arbitrariness, so I would record the chosen seed and optionally run a seed-sensitivity diagnostic.

New issues to flag: none blocking.

**Fix 2**
Verdict: partial

Reasoning: The count logic now matches empirical bin frequencies: `_compute_bins_from_direct_profiles` assigns `frequency = mask.sum() / len(p)`, and nonempty bins partition the rows, so frequencies sum to 1 in this call path. Absorbing rounding drift into the largest-frequency bin is reasonable and minimizes relative distortion.

New issues to flag: the generated CE-IS profile is block-ordered by bin, not arranged like an empirical season. If `estimate_p57_with_ceis` is order-sensitive for streak probability, this can create a new synthetic-season artifact even though the bin counts are now right.

**Fix 3**
Verdict: partial

Reasoning: The old impossible `CI lower >= 0.05` gate is fixed, but `corrected_pipeline_lo <= headline <= corrected_pipeline_hi` is only non-rejection, not evidence that the headline is defended. Because DEFENDED is checked first, a very wide CI spanning both `< 0.5 * headline` and the headline returns `HEADLINE_DEFENDED` instead of `HEADLINE_INCONCLUSIVE`.

New issues to flag: high false-defend risk at low power. DEFENDED should require enough precision, or at least avoid DEFENDED when the CI also includes the broken/reduced region. `BROKEN` at half-headline is a defensible practical threshold; using one-third is a policy choice, not required. `is_conclusive` in JSON would be useful but is secondary to fixing the gate precedence/precision.

**Fix 4**
Verdict: partial

Reasoning: Reconstructing `new_p_both = p1*p2 + rho*sqrt(...)` addresses the double-counting bug; it treats `rho_pair_cross_game` as an absolute Pearson-style dependence knob instead of adding it on top of empirical `b.p_both`. Dropping empirical `b.p_both` is acceptable for v1 under a Bernoulli-pair Pearson approximation, because with fixed marginals the joint probability is determined by the correlation.

New issues to flag: `p1 = p2 = b.p_hit` is still wrong for rank-1/rank-2 pairs if rank-2 has a different hit marginal. Also, `p_both` is reconstructed from original `b.p_hit`, while the returned bin uses `p_hit=new_p_hit`; if PA correction changes `new_p_hit`, the corrected table can become internally inconsistent. This should take separate corrected rank-1/rank-2 marginals, or at minimum use the corrected marginal consistently.

**Fix 5**
Verdict: partial

Reasoning: `_run_terminal_r_mc_bootstrap` is a clear enough name, and the docstring accurately says this is terminal-reward MC, not DR-OPE. Keeping `rho_PA_within_game` as a documented no-op is tolerable for API compatibility, though still risky.

New issues to flag: the harness output still labels fields as `fixed_policy_dr_ope_p57` and `pipeline_dr_ope_p57`, so the external artifact remains misleading. `expected_block_length` is also still a dead parameter in `_run_terminal_r_mc_bootstrap`.

**Overall Verdict**
Block before Task 13. Fix 1 is clean; Fix 2 fixes frequency but may still have CE-IS ordering artifact. The real blockers are Fix 3's false-defend gate and Fix 4's corrected `p_both` marginal/invariant problem. Fix 5 should also rename the JSON fields before the real-data run so downstream artifacts do not keep claiming DR-OPE.

---

## Follow-up fixes applied (Claude, post-review)

Five additional fixes applied after this review:

1. **CE-IS profile shuffling** — added `np.random.default_rng(42).shuffle(ceis_profiles)` after the bin-block construction. Block-ordering would have artificially suppressed streak probability since high-prob days were concentrated at the end. Shuffling matches a real season's day-to-day variation.

2. **DEFENDED gate precision check** — added `corrected_pipeline_lo >= half_headline` clause to the DEFENDED condition. This rules out the false-defend case where a wide CI spans both the headline and broken-territory values.

3. **p_both internal consistency** — changed `p1 = p2 = b.p_hit` to `p1 = p2 = new_p_hit` so the bin's reported p_hit and the reported p_both share the same marginal assumption. v1 limitation noted: rank-1 vs rank-2 separate marginals deferred (not available in current bin format).

4. **JSON field rename** — `fixed_policy_dr_ope_p57` → `fixed_policy_terminal_r_mc_p57`, `pipeline_dr_ope_p57` → `pipeline_terminal_r_mc_p57`. External artifacts no longer claim DR-OPE.

5. **Dead `expected_block_length` removed** — from `_run_terminal_r_mc_bootstrap` signature and the two call sites in `audit_fixed_policy` / `audit_pipeline`. The day-bootstrap concept doesn't apply to trajectory-bootstrap.

All 152 tests still pass after the follow-up fixes.

## Open issues NOT fixed

- **Fix 1 seed-arbitrariness** — Codex suggested optional seed-sensitivity diagnostic; deferred (low priority).
- **Fix 4 separate rank-1/rank-2 marginals** — Codex flagged that using one bin-level marginal for both ranks is approximate; v1 limitation, deferred (would require bin format change).
- **JSON `is_conclusive` flag** — Codex suggested for downstream tooling; deferred.

## Lesson

Iterative review surfaces deeper bugs than any single review. The first Codex review caught BIG bugs (seed-averaging, cyclic CE-IS, miscalibrated gates). This second review caught SUBTLER bugs the first review couldn't see because the buggy code wasn't in front of it (block-ordering, non-rejection DEFENDED gate, marginal inconsistency, JSON labels). Each pass exposes a different layer.
