# Why the shipped reach-57 table skips a bin while a weaker bin plays (closing-test finding SW1), 2026-11-24

Asked by Eric: "confirm that's not indicating something more troubling under the hood". Lead's investigation, on this machine.

## The table
`data/models/mdp_policy.npz` (sha `66d15471…`; 58 streaks x 181 days left x 2 saver states x 5 quality bins; actions 0 skip, 1 single, 2 double; edges 0.7960, 0.8115, 0.8252, 0.8407). **1,511 of 20,996 states** have a bin that plays while a higher bin skips:
- pattern "bin 3 plays, bin 4 skips": 954 states with more than 30 days left and 202 with 30 or fewer (streaks 41 to 56 mostly; at streak 56 with 6 to 30 days left only bin 3 plays);
- pattern "bins 0, 2, 3, 4 play, bin 1 skips": 355 states, all with 30 or fewer days left (the late phase), at every streak.

## Why it can only come from the bins' fitted rates
In `solve_mdp` the value of skipping does not depend on the bin, and the value of a single pick is `p_hit * V(next) + (1 - p_hit) * V(miss)` with `V(next) >= V(miss)`, so it rises with `p_hit`; a double rises with `p_both` the same way. With rates that rise with the bin, the set of bins that play is therefore always "this bin and every higher one". A table that skips a higher bin while a lower one plays means the solver's fitted `p_hit` (or `p_both`) for the higher bin was LOWER. (Reasoning from the solver's equations; not a measured run.)

## What the data says (outcomes 2021 to 2025 only)
The table's own profiles (`pooled_bins_run`, 24 seeds) are lost. Two saved single-seed sets with the same probability basis as the table's edges (`actual_pa`, which inflates hit rates; `data/simulation/backtest_*.parquet` and `data/validation/mdp_dd_guardrail_profiles_2026-05-30/`), binned with the shipped edges by `compute_bins_with_boundaries`, and the late phase (last 30 dates) with its own quintiles as `build_pooled_policy` does:

| Set, phase | p_hit by bin 0..4 | p_both by bin 0..4 | days per bin |
|---|---|---|---|
| simulation, all | .776 .878 .870 .915 **.898** | .648 .733 .724 .774 .813 | 164 to 196 |
| guardrail, all | .791 .830 .883 .944 **.897** | .622 .681 .725 .815 .800 | 171 to 196 |
| simulation, late, own quintiles | .800 **.733** .800 .900 .933 | .633 .600 .700 .733 .833 | about 30 |
| guardrail, late, own quintiles | .833 **.700** .867 .900 .900 | .700 .500 .700 .800 .800 | about 30 |

## Reading
1. **The top of the model's probability scale is flat.** Above about 0.82 predicted, the realized hit rate stops rising (about 0.90 here, with hindsight-inflated probabilities); in both sets the top quintile hits less than the fourth. The differences (0.017 and 0.047, about 180 days a bin, standard error about 0.022) are within noise, so this says the model cannot rank among its strongest candidates, not that strong candidates are worse. It matches the live 2026 overconfidence (stated 0.765 against 0.68 to 0.73 realized).
2. **The odd cells are a fitting artifact.** The solver took each bin's observed rate as its true rate, with about 180 days a bin (about 30 in the late phase) and no constraint that rates rise with the bin, so sampling noise became policy. Not a code bug.
3. **The repair Eric approved** (never skip a candidate when the table would play a weaker one in the same state) imposes exactly that ordering on the play/skip decision and changes nothing else. It does not fix (1): that belongs to the longest-streak policy work (D7) and the 2027 live calibration measurement.

Limits: single seed, `actual_pa` stand-ins (the lost profiles may differ); the size of the effect on reach-57 probability is not measured.
