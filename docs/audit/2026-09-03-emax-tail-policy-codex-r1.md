# Adversarial review: E[max] tail objective

## Verdict

**BLOCK the proposed design.** The weakest claim is that extracting
`polAll[s, s, d, saver, q]` gives a sound E[season-best] policy while avoiding the
running-maximum state. It does neither: it is not the policy that `solve_emax`
evaluated after the first miss, and it can choose a strictly worse action than the
true `(s, m)` policy in both directions. The proposed streak-0 property also does
not establish the stated "keep picking through 9/27" behavior once the streak
rises.

The smallest safe decision is to define the product requirement first:

1. If **a pick on every remaining game date** is mandatory, make that a hard
   action constraint (`single` or executable `double`) and optimize a Markov tail
   objective inside it. Do not expect E[max] itself to imply no skips.
2. If occasional strategic skips are acceptable and the actual season maximum is
   the objective, carry `m` at runtime and accept that exact E[max] can rationally
   stop when the old maximum cannot be beaten.

The current proposal silently mixes those two incompatible requirements.

## Severity-ranked findings

### P0 — The frontier table is not a valid implementation of E[season-best], and the acceptance invariant is too weak (A, F)

`solve_emax` computes an action at `(s, m)` assuming all future decisions also use
the true successor `m` (`scripts/audit/skip_threshold_resolve.py:55-75`). Extracting
the action at `m == s` and then using the same extraction after a miss substitutes
`m=0` for the real prior maximum. Thus the deployed continuation is not the
continuation whose value made the extracted first action optimal.

A read-only exact DP using the proposed late pooled rates after frequency-weighted
PAV found **1,897 action disagreements** between the true `m=18` policy and the
frontier policy over valid states `0 <= s <= 18`, `1 <= d <= 28`, both saver states,
and five bins. Concrete counterexamples:

- `(s=17, m=18, d=8, saver=0, q=Q3)`: true E[max] chooses `skip`; the frontier
  chooses `single`. True-state action values are
  `[20.008737554, 19.947450175, 19.982853704]` for
  `[skip, single, double]`; frontier loses `0.061287379` expected best-streak units.
- `(s=10, m=18, d=5, saver=0, q=Q1)`: true E[max] chooses `double`; the frontier
  chooses `skip`. Values are `[18.000000000, 18.063997750, 18.093461108]`.
- `(s=17, m=18, d=4, saver=0, q=Q5)`: true E[max] chooses `double`; the frontier
  chooses `single`. Values are
  `[19.494725507, 19.657057623, 19.849027088]`.

Evaluating the extracted frontier policy under its *actual* continuation from the
current state also loses value: at `(s=0,m=18,d=25)`, exact E[max] versus the fixed
frontier policy is `18.109497053` versus `18.070189414` without a saver, and
`18.249740866` versus `18.186564208` with a saver.

The proposed `streak == 0` test does not prove "keep picking." On the smoothed
frontier table, an all-double-hit path beginning 9/04 has:

```
(s=0,d=24)  [double,double,double,double,double]
(s=2,d=23)  [double,double,double,double,double]
(s=4,d=22)  [double,double,double,double,double]
(s=6,d=21)  [skip,skip,single,double,single]
```

So after only three successful doubles, the table skips in the bottom 40% of its
modeled day types. The prompt's own unsmoothed preview similarly contains skips
from `s=5` onward. If "keep picking" literally means every remaining game date,
the design already fails its product requirement before code is written.

**Required fix:** do not ship frontier extraction as E[max]. Carry and validate a
running maximum if E[max] is the requirement. If picking every day is the primary
requirement, exclude `skip` in the unreachable regime (or make pick-count the
lexicographic primary objective) and choose a clearly named secondary objective.
Test all valid unreachable states and the post-clamp executed action, not only
streak 0.

### P1 — The proposed `p_both` inputs do not describe production's executable double (C, E)

The pooled-bin builder pairs rank 1 with rank 2 without considering game identity
(`src/bts/simulate/pooled_policy.py:174-194`). Production instead removes every
same-game row and uses the highest-ranked remaining different-game candidate
(`src/bts/strategy.py:366-375`, `src/bts/strategy.py:404-407`).

On the 3,600 late `(seed,date)` profiles:

- rank 2 is in rank 1's game on **753/3,600 = 20.9%** of days;
- production-shaped second choices are rank 2 on 2,847 days, rank 3 on 451,
  ranks 4-10 on 298, and absent on 4;
- raw prompt `p_both` by Q1..Q5 is
  `[0.5486, 0.4625, 0.5417, 0.5972, 0.6069]`;
- using the first executable different-game candidate gives
  `[0.5667, 0.4667, 0.5417, 0.5861, 0.5944]` (counting the four no-double days as
  non-successes).

This matters immediately because the proposed streak-0 policy doubles in every
bin. The solver also assumes the chosen double is available, while live strategy
silently downgrades `double` to `single` when `allow_double` is false or no
different-game candidate exists (`src/bts/strategy.py:162-165`). The resulting
executed single was never compared with the table's alternatives at that state.

**Required fix:** construct solver pairs with the same ranked-different-game rule
as production, model double availability (or solve an explicit no-double action
surface), and test the executed action after both operational clamps. At minimum,
report the sensitivity of the policy to the production-shaped `p_both` vector.

### P1 — “Optional keys” create a silent return to skip-forever, and the loader API is unspecified (C, F)

Production currently receives exactly a 3-tuple from `load_policy`
(`src/bts/simulate/mdp.py:164-171`), and `_load_mdp` retains only those three
objects (`src/bts/strategy.py:34-47`). Merely adding NPZ members cannot affect
`lookup_action`; changing that tuple breaks widespread unpacking (30 occurrences
across 25 Python files, including simulation/audit tooling).

There is a second, more dangerous path: `MDPSolution.save` writes exactly
`policy_table`, `boundaries`, `season_length`, and `optimal_p57`
(`src/bts/simulate/mdp.py:120-160`). Both `scripts/rebuild_policy.py:35-40` and the
public solve CLI (`src/bts/simulate/cli.py:190-212`) use it. Either writer can
silently erase the optional E[max] members. Because the proposed compatibility
behavior for a missing tail table is today's behavior, the failure mode is not a
startup error; it is **no more production picks**.

**Required fix:** keep the existing `load_policy` 3-tuple stable and add an
explicit versioned bundle loader for production, or use a separate tail artifact.
Whichever form is chosen, validate both tail keys, shapes, dtypes, finite/ordered
boundaries, action range, original-policy identity, and a schema/version field as
one atomic contract. In an unreachable live state, missing/partial tail data must
raise a visible startup/health failure or invoke an explicitly approved forced-pick
fallback; it must not silently use the old zero table. Update or fence every
writer capable of replacing `mdp_policy.npz`.

### P1 — The proposed source migration misses analytical meaning and can contaminate persisted skips (D)

Current exact consumers are:

- scheduler fallback skip classification:
  `src/bts/scheduler.py:564-573` (used at `2775-2801` and `2906-2932`);
- normal-cycle final-skip capture: `src/bts/scheduler.py:2580-2599`;
- fallback capture, which currently stores no source:
  `src/bts/scheduler.py:588-606`;
- EOD writer, which hardcodes `source="mdp"`:
  `src/bts/scheduler.py:678-700`;
- the three skip-shadow membership/consistency checks:
  `src/bts/skip_policy_shadow.py:129-145`, `148-179`, `204-227`;
- boundary-shadow census, which excludes any source other than exactly `mdp`:
  `scripts/audit/boundary_shadow_census.py:287-324`.

Adding a `source` default only at EOD is insufficient. A fallback E[max] skip must
copy its source into `final_skip_candidate`; otherwise the default `"mdp"` labels
it as a reach-57 skip and the nightly skip-policy shadow admits it. Conversely,
`mdp_emax` makes the boundary census call a genuine MDP decision
`excluded_non_mdp`.

`source` should identify the mechanism (`mdp` versus `heuristic`); objective is an
independent dimension. Keep `source="mdp"` and persist an explicit
`objective="reach57"|"tail_..."` through `SelectionResult`, both final-skip capture
paths, `final_skip_candidate`, and `decision.json`. Then make the reach-57 shadow
require `objective in {null,"reach57"}` for legacy compatibility. Because the
field affects analytical membership, a v3 decision schema is cleaner than adding
new semantics while continuing to call the record v2.

Source-agnostic consumers remain safe if `action`/`scoreable` stay unchanged:
`daily_decision.load_decision` validates action and scoreability but not source
(`src/bts/daily_decision.py:69-86`); check-results uses only
`is_scoreable_commit` (`src/bts/cli.py:2233-2242`); contest freshness scans settled
pick files (`src/bts/contest_state.py:151-185`); post-failure uses
action/scoreability (`src/bts/health/post_failure.py:48-57`); and the dashboard
uses the action plus `skip_summary` (`src/bts/web.py:286-294`,
`src/bts/web.py:1098-1113`, `src/bts/web.py:1151-1158`). Tests that currently pin
`source == "mdp"`, such as `tests/test_strategy.py:84-96`, will need objective
coverage rather than wholesale source rewrites.

### P1 — The new all-bin double policy increases exposure to an existing incomplete cutoff boundary (E)

`allow_double` is currently always true for both model-only and contest decision
states (`src/bts/contest_state.py:256-267`, `src/bts/contest_state.py:294-307`). A
streak-0 E[max] decision will therefore usually become a real double, not be
clamped.

A double changes the operational path materially:

- both selected lineups gate normal lock (`src/bts/strategy.py:232-243`,
  `src/bts/scheduler.py:1270-1318`);
- the earlier of the primary/DD games controls fallback and cutoff
  (`src/bts/scheduler.py:2672-2686`, `src/bts/scheduler.py:2867-2886`);
- projected DDs therefore make fallback delivery more common.

The last scheduler check is before `send_dm` (`src/bts/scheduler.py:970-986`), but
`send_dm` then performs authentication, handle resolution, conversation lookup,
and finally the non-idempotent send (`src/bts/dm.py:29-87`). A request can still
begin before cutoff and finish after it; the code itself calls this a residual
race (`src/bts/scheduler.py:975-977`). The deploy design increases the frequency
of the path without an end-to-end deadline-crossing test.

**Required fix:** add a production-shaped test covering a double with a projected
second lineup, the earlier DD cutoff, fallback refresh, transport latency, and
operator reserve. This is not a reason to put cutoff work inside the E[max] patch,
but it is a reason not to claim the first forced-double day is operationally
covered by unit policy tests.

### P2 — PAV is an unsupported modeling assertion, not a neutral cleanup (C)

Late-only data is the appropriate available scope: no MDP-unreachable state has
`d > 28`, and the available tail is late season. New estimated-PA boundaries are
also preferable to the shipped boundaries: the latter place 3,005/3,600 = 83.5%
of late rank-1 profiles in Q1 (occupancy `[3005,344,153,65,33]`), largely removing
quality discrimination.

What is not established is that the observed inversions are noise. A descriptive
5,000-replicate seed-cluster bootstrap using the fixed pooled late boundaries gave
Q2-Q1:

- `p_hit`: point `-0.04028`, bootstrap interval `[-0.07915,-0.00285]`, only 1.94%
  of replicates nonnegative;
- `p_both`: point `-0.08611`, interval `[-0.11822,-0.05255]`, 0/5,000
  nonnegative.

This is not a season-level confidence interval—the 24 seeds reuse five seasons—but
it does show the inversion is not a one-seed accident. PAV changes the inputs to
`p_hit=[0.69236,0.69236,0.73194,0.74167,0.80972]` and
`p_both=[0.50556,0.50556,0.54167,0.59722,0.60694]`. That effectively pools Q1 and
Q2 outcomes while retaining two labels.

**Required fix:** validate monotonic pooling out of sample by season/seed, and use
production-shaped doubles first. Fewer bins or an explicit Q1+Q2 merge is simpler
and more honest if validation supports it. Independently smoothing `p_hit` and
`p_both` also needs assertions `0 <= p_both <= p_hit <= 1` after smoothing.

### P2 — Artifact preservation and monitoring assertions are incomplete (C, D, F)

“Reachable region byte-identical” is not enough. Selection in that region also
depends on the original boundaries and season length. The composer should assert
full equality of every original member (`policy_table`, `boundaries`,
`season_length`, `optimal_p57`) against an input hash, then add new members. The
NPZ file hash necessarily changes, so future picks begin a new calibration regime:
pick provenance hashes the entire policy file (`src/bts/picks.py:145-176`) and
realized calibration groups by that hash (`src/bts/health/realized_calibration.py:176-192`).

`effective_pick_bar` builds probe values and reported floors from only
`mdp["boundaries"]` (`src/bts/strategy.py:80-101`). If lookup internally changes
to E[max] boundaries, the displayed bar can still be sampled/reported using the
reach-57 cutpoints. `mdp_policy_alignment` directly loads only `policy["boundaries"]`
and applies them to all recent picks (`src/bts/health/mdp_policy_alignment.py:90-149`),
so after the switch it monitors the wrong binning and mixes objectives.

**Required fix:** make boundary selection objective-aware in display and health
paths, and stratify monitoring at the objective/policy hash transition. Embed
input hashes, profile count/seed/season manifest, PAV method/version, and base
policy SHA in the composite artifact or a machine-checked adjacent manifest.

## Direct answers

### A. Objective

No: frontier extraction is unsound; the counterexamples and fixed-policy regret
above directly falsify it. Exact E[season-best] is not Markov on
`(s,d,saver,q)` because two histories with the same current streak and different
running maxima have different rewards and actions. A lexicographic P57-then-E[max]
objective reduces to the same E[max] problem once P57 is zero and retains its
cliff. A tie-break toward picking fixes the `s=0,m=18,d<=9` exact tie but not states
where skip is strictly better, such as `(17,18,8,0,Q3)` above. Discounting may
alter where skips occur but does not guarantee their absence.

I would carry `m` only if the accepted requirement is genuinely season-best
E[max]. The contest input already parses `best_streak`
(`src/bts/contest_state.py:88-115`), but `DecisionStreakState` does not carry it
(`src/bts/contest_state.py:46-60`), and it would need validation that
`0 <= s <= m <= 57`. If the real requirement is a pick every day, use a hard
no-skip constraint and a simpler Markov secondary objective such as expected
final tail streak.

### B. Reachability

Within the MDP model, `s + 2*d >= 57` is the exact reachability condition for
`0 <= s < 57`, positive `d`, and positive double-hit probabilities. Every action
consumes one day; the maximum increment is two; the saver only changes the miss
state and never adds hits (`src/bts/simulate/mdp.py:31-75`). Therefore `<57` is
zero-valued, while equality is not degenerate: the all-double-success path has
positive probability.

The shipped artifact confirms the boundary: `(s=0,d=28)` is all skip,
`(s=0,d=29)` is non-skip, `(s=55,d=1)` is double, and `(s=56,d=1)` is single.
`days_remaining` includes today; `d=1` on 9/27 is correct, and the existing
`d<=0` guard is correct (`src/bts/simulate/mdp.py:187-200`). Normalize
`d_eff=min(days_remaining, season_length)` before both objective selection and
indexing so the predicate and table use the same horizon.

This is exact only for the solver, not for live operational feasibility: the
predicate assumes an executable double every remaining day, while production can
downgrade doubles (`src/bts/strategy.py:162-165`). For the stated 9/03 slate and
model question there is no off-by-one; do not describe the helper as exact live
contest reachability without that caveat.

### C. Bins

Use the late estimated-PA distribution, not all-season rates and not shipped
boundaries. The latter collapse 83.5% of late observations into the bottom bin.
But do not ship the proposed raw-to-PAV transformation without held-out validation,
and do not use rank-2 `p_both` when production executes the first different-game
candidate. Two boundary arrays in one NPZ are mechanically fine only behind a
new bundle loader; changing the existing `load_policy` tuple is not.

### D. Source/action/persisted-state consumers

The full consumer inventory and effects are in the P1 source finding above.
`mdp_emax` is schema-tolerated but semantically inferior to `source="mdp"` plus a
separate `objective`. The latter makes skip-shadow exclusion explicit instead of
overloading a mechanism label, and it prevents boundary-census records from being
misnamed `excluded_non_mdp`. Carry objective/source through both normal and
fallback `final_skip_candidate` construction; default absent objective to
`reach57` only for legacy records.

### E. First E[max] day and carried state

On 9/04, `d=24`; the proposed frontier table requests `double` in every bin at
streak 0. `allow_double` currently does not stop it. A one-game/no-different-game
slate downgrades it to single; otherwise the F2 two-lineup gate, earlier-slot
cutoff, and fallback path all activate. Those paths need the end-to-end test
described above.

The 9/03 `final_skip_candidate` does **not** carry into 9/04: carry-forward requires
the same date (`src/bts/scheduler.py:1417-1437`). On a same-date restart, a genuine
new pick clears both `final_skip_candidate` and `skip_summary`
(`src/bts/scheduler.py:2600-2607`). I found no cross-day state misfire from today's
skip. The migration risk is mislabeled fallback/EOD source, not reuse on 9/04.

### F. Test gaps and most likely false green

Add these gates beyond the proposed tests:

1. Exact off-frontier brute-force states and evaluation of the *extracted fixed
   policy*, not only equality of the augmented solver to itself.
2. The accepted keep-picking invariant over every reachable live state after
   operational double clamps, including the all-success path to `s=6,d=21`.
3. Boundary-edge cases where reach-57 and tail cutpoints classify the same
   probability differently; objective-aware `effective_pick_bar`.
4. Old, new, partial, malformed, wrong-shape, wrong-dtype, and out-of-range NPZ
   bundles; a generic save/rebuild must not erase the tail contract.
5. Full preservation of all original members and base/input hashes, not only the
   reachable policy slice.
6. Normal-cycle and fallback E[max] skips through same-day restart and EOD write;
   verify skip-shadow exclusion and boundary-census classification.
7. Production-shaped different-game pairing, no second game, `allow_double=False`,
   projected DD F2 gating, earlier DD cutoff, fallback latency, and cached-pick
   suppression.
8. A deploy/startup probe that loads the committed production artifact and asserts
   the expected objective/action at `(s=0,d=24)` and both sides of the reachability
   boundary.

The single most likely false green is the streak-0-only property: it passes while
the shipped frontier table starts skipping after a successful run. Close behind
is a missing/dropped optional tail key silently restoring today's all-skip behavior.
The deploy workflow's canary checks service state and dashboard HTTP only
(`.github/workflows/deploy.yml:115-156`), so neither failure is detected by a
green canary.

Current baseline verification was green—123 targeted tests passed—but none contains
the proposed bundle/objective behavior, so that result is not evidence for this
design.

### G. Simpler alternative

For the literal owner outcome, use a **must-pick tail policy**:

- keep the shipped P57 artifact and lookup completely unchanged while
  `s + 2*d >= 57`;
- once unreachable, restrict actions to `single` and executable `double` and
  maximize expected final tail streak on `(s,d,saver,q)`; when double is not
  executable, single is authoritative;
- use late, production-shaped different-game rates;
- persist `source="mdp"`, `objective="tail_final_streak_no_skip"`;
- store the small tail table in a separate versioned artifact or code-owned
  constant so generic reach-57 writers cannot erase it.

That objective is not E[season-best], but it is Markov on the desired state, has no
running-maximum approximation, and satisfies the no-skip contract by construction.
If the label E[max] is non-negotiable, the irreducible design is full `(s,m,d,saver,q)`
state plus a no-skip constraint; frontier extraction is not a valid shortcut.

