# Round 2 adversarial design review: exact E[season-best] tail

## Verdict

**BLOCK as written.** The augmented `(s,m,d,saver)` objective is now the right
mathematical object, and the specified one-bin DP has no strategic skips in the
active tail. The weakest remaining claim is operational: the proposed forced-single
fallback is not yet placed above every route to the heuristic. If the *base* NPZ is
missing or unreadable, current control flow returns `mdp=None` and can still apply the
0.80 heuristic, recreating skip-forever on a 0.7336 late slate. Independently, one
plausible bad `best_streak` can make both the table and the fallback deliberately skip
the rest of the season.

Do not deploy until the unreachable-57 fallback is independent of both artifacts,
`m` has an explicit trust contract, and the runtime decision persists the actual
`m` and tail-artifact identity it used.

## Severity-ranked findings

### P0 — Base-policy failure bypasses the promised tail fallback (D)

The design says `_load_mdp` retains the three base objects and adds `tail`, then uses
forced single when tail is absent/invalid
(`2026-09-03-emax-tail-design-r2-prompt.md:78-88`). That does not cover failure of the
outer/base load:

- `_load_mdp` catches a missing base artifact, caches `{}`, and returns `None`
  (`src/bts/strategy.py:34-51`).
- `_mdp_action_from` returns `None` before it computes the date/horizon whenever
  `mdp` is falsey (`src/bts/strategy.py:54-69`).
- `decide_action` interprets that as permission to use the heuristic, whose first
  rule skips below 0.80 (`src/bts/strategy.py:150-161`).

Thus, on 9/04 with `p_game_hit=0.7336`, an absent/corrupt/unreadable base NPZ can
still return `skip`; the tail fallback is never reached. This is exactly the failure
the new contract claims to remove.

**Required change:** compute `d_eff` and objective selection before testing whether
the base bundle exists. In the unreachable regime, *every* artifact failure shape
(base absent, base corrupt, tail absent, tail corrupt, SHA mismatch, loader exception)
must resolve to:

```text
if min(target, s + 2*d_eff) <= m: skip
else:                              single
```

with a structured degraded reason and persistent health signal. The base and tail
loads must be isolated so a tail error cannot discard a valid reach-57 table, while a
base error cannot bypass the tail fallback. Add end-to-end `select_pick` tests with
the whole MDP absent and with the base NPZ malformed, a sub-0.80 primary, and
`s=0,m=18,d=24`; both must produce `single`, never the heuristic `skip`.

### P0 — `max(streak, best)` is an algebraic clamp, not a trust boundary (B, D)

The automatic fetch validates nonnegative integer values and
`seasonBestStreak >= activeStreak` (`src/bts/contest_fetch.py:182-196`). The generic
state parser does not preserve that contract: a non-integer best silently becomes
`None`, while a negative best or `best < active` is accepted
(`src/bts/contest_state.py:87-115`). More importantly, it cannot detect a plausible
but inflated best.

The source-selection cases are materially different:

- **Fresh auto observation:** it has the fetch-time internal invariant. This is the
  only ordinary path on which the proposed `max(s,best)` is enough.
- **Profile stale after a new high:** `best` is stale-low. `max(s,best)` may still be
  below the true maximum after a later reset, so the bot over-plays. It does not
  prematurely stop, but the decision is not exact E[season-best]. Freshness is only
  advisory; stale values are still returned for action selection
  (`src/bts/contest_state.py:270-309`).
- **Unexpired manual override:** it wins before the auto file is parsed
  (`src/bts/contest_state.py:136-147`). The CLI permits `--best-streak` to be omitted
  (`src/bts/cli.py:1351-1354`, `1421-1430`), and a hand-written file can contain an
  inflated integer. Omitted becomes the optimistic `m=s`; inflated becomes an early
  stop.
- **Expired manual override:** auto wins when present, but the expired manual is
  still returned if auto is absent (`src/bts/contest_state.py:145-148`). Its best must
  not authorize a stop merely because it is the last file left.
- **Model-only:** best is unknown, so `m=s` keeps playing for every positive `d`.
  That is the desired fail-open direction, but it must be labeled degraded rather
  than exact.

Concrete failure: `s=0,best=57,d=24` enters the tail (`48 < 57`) and both the table
and forced fallback stop for all remaining days. The outer `min(57, ...)` merely caps
an inflated value; it does not make it trustworthy.

**Required change:** hard-validate `0 <= streak <= best` when best is present,
normalize a legitimate above-target best to the solver cap, validate that the
observation belongs to the current contest season, and
carry a `best_status`/trust bit. Only a trusted best may authorize the terminal stop.
For missing, malformed, expired-only, wrong-season, or otherwise untrusted best, use
the optimistic degradation (`m=s`) and surface it in health and the decision record.
`m=min(target,max(s,best))` is the correct mathematical normalization *after* those checks.
Persist both the supplied best and effective `m`; otherwise an E[max] decision cannot
be reconstructed.

### P1 — The DP has exact ties, but the written stop predicate is uncapped and the
`1e-9` override is not an exact policy (A)

I independently solved the one-bin recurrence using the exact empirical rates
`p_hit=2641/3600` and `p_both=1984/3600`, over valid states `0 <= s <= m <= 57`,
`d=1..28`, and both saver values.

- In the tail (`s+2d < 57`), all **31,612** states with
  `s+2d <= m` chose skip.
- All **30,856** states with `s+2d < 57` and `s+2d > m` chose a play. The smallest
  `best_play - skip` margin was `5.684557891e-08`, at
  `(s=0,m=55,d=28,saver=0)`. There is no other exact-E[max] reason to skip in the
  deployed tail for these rates.
- Using the prompt's `s=0..56` convention, I reproduced exactly **12,256** raw
  skips in the reach-57 region (`s+2d >= 57`). In every one, the best play and skip
  were bit-equal in this float64 solve. At the prompt's example
  `(37,38,20,saver=0)`, `[skip,single,double]` was
  `[40.06307292, 40.01468527, 40.06307292]`. This is a structural stationary-horizon
  tie—double now and the optimal delayed continuation have the same value—not an
  observed `~1e-9` numerical deficit.

The declared stop formula omits the target cap. There are **1,624** valid cells where
`s+2d > m` but `min(57,s+2d) <= m`, all with `m=57`. For example,
`(s=56,m=57,d=1)` has values `[57,57,57]`: the season best cannot be beaten, but the
written formula forces a play. This cannot occur in the selected tail because the
tail has `s+2d < 57`; it still makes the artifact-wide builder/validator contract
false (`prompt:55-61`).

**Required change:** state the rule as
`skip iff min(target, s + 2*d) <= m`, or explicitly restrict the invariant to valid
tail cells. Do not use `abs(v_play-v_skip) < 1e-9` to alter an exact optimizer: it can
flip a genuine skip with regret below `1e-9` after a future rate/solver change and
then the artifact is no longer exactly optimal. For this table, prefer a play only
when it is itself an exact maximizer (for example, order play actions before skip
when taking argmax, or compare against the computed maximum without admitting a
strictly lower value). The active tail already satisfies the desired property, so it
needs no tolerance override.

### P1 — “Strict” artifact validation is sampled, a mismatch is accepted, and the
decision is not bound to the tail bytes (D)

The contract calls itself strict but checks stop consistency only on a sampled grid
and merely warns when the base SHA differs (`prompt:78-88`). A single unsampled
action byte can therefore turn a live active-tail state into skip. Exhaustive checking
is cheap: the full declared table has only `58*58*29*2 = 195,112` action cells, and
only 62,468 valid tail cells in the two stop/play partitions above.

A base SHA mismatch should make the tail invalid and select the safe fallback, not
continue under an unpaired release. Also validate with `allow_pickle=False`; encode
the profile manifest as canonical JSON or primitive arrays rather than an object
array; validate scalar cardinality/types, finite probabilities, Fréchet constraints,
frequency sum, shape/dtype/action range, `d=0`, and every valid applicable stop cell.

The separate artifact also falls outside existing provenance. `attach_provenance`
hashes only the path passed as `policy_npz_sha256`
(`src/bts/picks.py:145-176`), and every listed caller passes the base
`DEFAULT_POLICY_PATH`. A tail pick would therefore identify the unchanged reach-57
NPZ, not the bytes that chose its action. Decision schema v3 must carry a
`tail_policy_sha256` (and tail schema/solver identifier), and DailyPick provenance
must do the same for played tail days.

There is also an enum contradiction: the design declares objective values only
`reach57 | emax_season_best` (`prompt:23-25`) but names fallback decisions
`tail_fallback` (`prompt:87-88`). Either make `tail_fallback` a third accepted
objective everywhere, or retain `objective=emax_season_best` and add a separate
`degraded_reason`. Do not let a v3 writer emit a value its reader/health filters do
not recognize.

Finally, the deploy canary only checks service state and dashboard HTTP
(`.github/workflows/deploy.yml:115-156`); it never loads either artifact. The
committed-artifact contract test is necessary, but add a startup/health probe that
loads the deployed paths and records the tail SHA/error. Logging once inside the
lazy loader is not durable evidence that the protection is active.

### P1 — The four-call-site description hides two different state models, and
display/persistence can disagree with the decision (E)

The two CLI paths and the orchestrator already load `DecisionStreakState`, so they
can pass its best alongside streak/saver (`src/bts/cli.py:1157-1172`,
`1311-1325`; `src/bts/orchestrator.py:282-302`). The scheduler's fourth caller is
not equivalent: shadow prediction deliberately loads model replay streak directly
and never constructs contest decision state (`src/bts/scheduler.py:1737-1750`). The
module contract says simulation, shadow, and replay use `streak.json`
(`src/bts/contest_state.py:1-5`). Passing contest best there changes shadow semantics;
passing `None` makes its E[max] policy an optimistic `m=s` approximation. Choose one:

1. production-action shadow: use the full contest `DecisionStreakState` and stamp
   that provenance; or
2. model-replay shadow: track a separate model season-best, or explicitly exclude
   tail-objective shadow rows from policy comparisons.

Do not silently call the latter exact E[max].

Other missed consumers:

- CLI `run` and `preview` immediately unwrap `.pick_result`, discard the selection's
  objective/reason, and still describe every skip as “below threshold”
  (`src/bts/cli.py:1163-1183`, `1316-1329`). The scheduler-only
  `build_skip_summary` change does not fix these paths.
- The scheduler reloads decision state but retains only `.streak`, then calls
  `effective_pick_bar` without best (`src/bts/scheduler.py:1564-1577`). That can use a
  different state snapshot from the one that selected the action and necessarily
  computes `m` incorrectly when best exceeds streak. Use the already-resolved
  `SelectionResult`, not a second state read/lookup.
- Current v2 persistence records streak/saver/state source but no best/effective `m`
  (`src/bts/daily_decision.py:35-60`). Add those fields to v3 and thread them through
  both skip-capture dictionaries and all commit branches. Objective alone cannot
  reproduce the action.

A single structured policy result—`action`, `source`, `objective`, `reason`, `s`,
`m`, `d_eff`, artifact SHA, degraded status—should be produced once, carried on
`SelectionResult`, and reused for logs, the dashboard, fallback classification, and
decision.json. Appending new dataclass fields with defaults avoids breaking the many
positional `SelectionResult(...)` returns (`src/bts/strategy.py:294-362`,
`398-427`).

For D+1 preview, using the latest known contest best is coherent for a preliminary
pick, but the current-day result may still be unresolved. Preserve the state status
and effective `m` in the preview selection so the scheduler's later re-evaluation is
auditable; do not present it as final state.

### P2 — The profile set has exactly 150 dates, but a quintile is not 30 distinct
real days, and one bin removes decision-relevant signal (C)

Read-only recomputation over all 120 parquets / 24 seed directories found:

```text
pooled rows                           218,880
late rows                              36,000
late rank-1 seed-days                   3,600
distinct late dates by season  30,30,30,30,30
distinct (season,date) total              150
rank-1 rows per real date                  24
one-bin p_hit                 2641/3600 = 0.7336111
one-bin production p_both     1984/3600 = 0.5511111
```

So “the same ~150 days” is correct (it is exactly 150), but “a quintile is ~30 real
days” is not a literal partition. For the top-20/rest split at `0.792330`:

- top observations occur on 60 distinct dates;
- rest observations occur on 139 dates;
- 49 dates contribute to both groups because seeds select/classify different
  players on the same date;
- a date has a mean 2.03 distinct rank-1 batters across the 24 seeds (range 1..5).

The groups remain heavily shared-date dependent, but the effective sample size is
not obtained by dividing 150 by five. A date-cluster influence calculation gave
`p_hit(top)-p_hit(rest)=0.09514`, cluster SE `0.06744`, or 1.41 SE. This supports the
same “weak evidence” conclusion as the prompt's rough 1.2 SE, not its stated sample
mapping. Also, these are the last 30 season dates, not strictly September: ranges
include 2025-08-30 and extend as late as 2022-10-05.

One bin does lose something relevant to the owner's E[season-best] objective: the
current slate's observed quality no longer affects single versus double at all.
Using the proposed production-shaped two groups gives:

```text
                 rest       top20
p_hit          0.714583    0.809722
p_both         0.540278    0.594444
```

An exact two-bin solve disagreed with the one-bin action in **7,555 / 61,712**
active-tail `(state,current-bin)` cells. Example:
`(s=14,m=14,d=3,saver=1,top20)` has two-bin values
`[skip,single,double]=[16.204444,16.920022,16.898558]`; two-bin chooses single while
one-bin chooses double, losing 0.02146 expected best-streak units under the two-bin
point estimates. At the first-day state `(0,18,24,0)`, both bins and the one-bin
model choose double, so this does not change 9/04.

The limited five-season evidence can justify the owner's conservative one-bin
choice, but label it correctly: it trades away adaptive quality decisions to reduce
estimation variance. It is not an exact implementation of an objective that observes
today's model score; it is exact only under the assumed exchangeable one-bin world.

### P2 — A correct 9/04 action will not prove `best_streak` was threaded (F)

At `(s=0,d=24)`, both `m=18` and the missing-best degradation `m=0` choose double.
Therefore seeing the expected double on the first live day does not exercise the new
state dimension. The first-day acceptance check must assert the resolved/persisted
effective `m == 18`, objective, and tail SHA—not merely the action. If the artifacts
fail, the expected visible result is a **single** with degraded fallback status; a
silent double or heuristic skip is a contract failure.

This is the only new first-day-specific issue I found; the different-game downgrade,
two-lineup gate, earlier DD cutoff, and transport-latency residual remain exactly the
round-1 risks and should not be re-litigated inside this patch.

## Simpler design (G)

Keep the augmented state; it is irreducible for the chosen objective. Simplify the
rest:

1. Resolve reachability before artifact availability. A tail artifact error always
   enters the explicit single/terminal-stop fallback.
2. Solve the one-bin DP with exact argmax semantics. Do not rewrite near-optimal
   actions with `1e-9`; exhaustively assert the desired property over valid tail
   cells.
3. Treat best as `trusted integer` or `unknown`. Unknown/untrusted means `m=s` and
   may not authorize a terminal stop.
4. Return one structured policy decision and persist/reuse it everywhere. In a
   one-bin tail, `effective_pick_bar` does not need to probe boundaries: it is `0.0`
   for play and `None` for terminal stop, directly from that decision.
5. Keep the separate artifact, but hard-bind it to the base SHA and bind every live
   tail decision to the tail SHA.

That removes the tolerance policy, duplicate state reload, display re-lookup, and
the base-missing heuristic hole without changing the owner's accepted objective.

## Verification performed

- Repository HEAD: `3ec2fc5`; no tracked file was modified.
- Independent float64 DP over every valid one-bin state for `d=1..28`, both saver
  values, using the empirical integer rates above.
- Read-only scan of all 120 estimated-PA profile parquets, including production-shaped
  different-game pairing, date reuse, and the two-bin comparison.
- Exact production call-site/state/decision/provenance inspection at the cited lines.
