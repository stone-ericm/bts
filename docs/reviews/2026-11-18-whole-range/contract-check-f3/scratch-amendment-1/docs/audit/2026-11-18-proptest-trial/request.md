# Request for the BTS lead: a one-lane property-testing trial

Written 2026-11-18 by a Claude session on Eric's MacBook (not in herdr), delivered on Eric's word.
**Not urgent. Finish the audit change set's current step first; start this when you have a free agent slot.**

## Where this comes from

Eric read https://recursion.wtf/posts/agents-and-property-tests/ and asked whether it was useful to us.
The method: agents build property-test machinery (generators, a slow obvious reference, assertions)
on a discovery branch that never merges; each confirmed bug becomes a small standalone regression
test plus a minimal fix. A read-only scan of five of his repos followed. Eric's words in order:
"i'm interested in a scan", then "sure" to a single-agent BTS trial, then "go" to handing it to you.

Everything below about BTS code comes from one agent READING the MacBook checkout. Nothing was run.
Line numbers may differ on your head. Treat each claim as a lead to check, not a finding.

## The trial: one target, one lane

**Target:** `compute_all_features` (`src/bts/features/compute.py`, about line 133).
Its docstring contract: every feature for a plate appearance on date D uses only rows dated
strictly before D.

**Property (prefix invariance):** for a generated frame and any date D, the feature rows for date D
computed from the whole frame equal those computed from the frame cut at date <= D. Stronger form:
change `is_hit` and the pitch columns on every row dated >= D and assert the rows before D do not
change. This needs no reference implementation.

**Then, if cheap:** per-feature from-scratch loops for the rolling windows (`batter_hr_7g`, the
rookie shrinkage rule, expanding `platoon_hr` with its PA gate, the cumsum-minus-current step in
`batter_pitcher_shrunk_hr`).

**Things the reader flagged for whoever writes it:**
- Two hidden inputs: `_build_probable_pitcher_lookup()` reads `data/raw` relative to the working
  directory, and `park_drag.attach_park_drag` reads an external CSV. Both need monkeypatching.
- Windows count game-dates and cross season boundaries by design; `days_rest` spans the off-season.
  The property must encode that intent, not a per-season reset.
- Generators need collisions to mean anything: few batters, few pitchers, doubleheaders, dates
  packed around a season boundary. Read sample frames before trusting a green run.

**Rules for the lane:**
1. Hypothesis lives on a discovery branch only. `uv.lock` has `exclude-newer = "2026-04-11"`, so
   take a Hypothesis release from before that date. Nothing from the discovery branch merges.
2. Before believing a green result, plant a leak (for example, let one rolling window include
   date D) and confirm the property goes red. Record it, then remove the leak.
3. A confirmed violation becomes a plain pytest regression test (shrunk example, no Hypothesis
   import) plus a minimal fix, branched from main, through your normal review tier for
   feature-path code. An ambiguous one comes back to Eric as a question.
4. Your existing terms apply unchanged: the three-agent cap, the yield wrapper, the agent log.
5. Report what one run cost (agent time, tokens if visible) and what it found, including "nothing".
   Eric wants that number before deciding whether more targets are worth it.

## Other leads from the same read (your call whether any matter; all unverified)

Further property-test targets, in the reader's order:
- The streak and saver rules exist in about six places (`picks.py` `update_streak` and
  `_replay_season_streak`, `simulate/mdp.py`, `monte_carlo.py`, `exact.py`, `tail_policy.py`).
  A five-line fold could check them against each other. You are already deep in streak state for
  the season rollover, so this may overlap work in hand.
- The probable-pitcher disk cache in `compute.py` is keyed only by game id, never invalidated,
  ignores `raw_dir`, caches missing probables as None, and is written non-atomically.
- `skip_policy_shadow.py`: a 10-day lookback against 4-day checkpoint eligibility, with the hit
  checker called before the staleness rule. The reader infers a verdict documented as final could
  change after a cron outage of five or more days.
- `simulate/quality_bins.py` `compute_bins` drops empty bins, while `mdp.lookup_action` indexes by
  `len(boundaries) + 1`. Sizes would disagree if quantile boundaries tie.

Suspected plain bugs, no property test needed to check:
- Wind parsing differs between training and serving: `compute.py` (about line 527) matches
  "out to l f" / "out to r f"; `model/predict.py` (about line 665) does not.
- `parse_daily_pick` drops `shadow_model_version`. Harmless while that field is None.
- `format_result_reply("void")` writes "Void. Streak: N", which `parse_result_from_reply` does not
  read as a result.
- `scheduler.load_state` builds `SchedulerState(**data)` and quarantines on TypeError, so a state
  file with a newer field after a rollback would reset the day's state, `pick_locked` included.

## What to send back

One line to Eric when you pick this up, and the lane's report when it ends. The MacBook session
that wrote this will not be watching; Eric reads your pane.
