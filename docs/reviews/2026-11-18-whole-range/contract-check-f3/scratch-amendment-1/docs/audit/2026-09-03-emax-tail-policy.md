# 2026-09-03 — Tail policy: exact E[season-best] once 57 is unreachable

## Incident

On 9/03 the contest streak was 0 (the 9/02 double-down missed: Arraez miss,
McCarthy hit) with 25 game dates left. The scheduler logged
`SKIP — best Steven Kwan (CLE) 73.8% below the pick bar; streak holds at 0` on every
cycle. Root cause: `solve_mdp` maximises P(reach 57) and values EVERY action at
exactly 0.0 in any state with `streak + 2*days < 57`; `np.argmax` falls to index
0 = skip. `policy_table[0, d, :, :]` is all-skip for `d <= 28`. Without a change no
production pick would have been made again this season.

Owner requirement (verbatim): *"I don't mind if we stop picking after I can no
longer beat my top streak. at that point there's no value to be gained. my issue
was around stop picking after i can no longer hit 57."*

## Design (two Codex adversarial design rounds, files alongside this note)

1. **Regime from state alone.** `tail_policy.mdp_objective(streak, days)`:
   `reach57` iff `streak + 2*d_eff >= 57` (exact within the MDP model — every
   action consumes a day, max increment 2, the saver never adds hits; equality is
   reachable), else `emax_season_best`. Resolved BEFORE any artifact is consulted
   so an artifact failure can never fall through to the 0.80 heuristic
   (Codex r2 P0: a missing/corrupt base policy previously did exactly that).
2. **Tail objective = exact expected season-best** on the augmented state
   (streak s, running best m, days d, saver): a port of the validated
   `scripts/audit/skip_threshold_resolve.py::solve_emax` (values reproduced
   bit-for-bit in tests). Codex r1 killed the frontier shortcut (extract the policy
   at m == s): it is not the policy the solver evaluated after a miss, with exact
   counterexamples at m = 18 (e.g. (s=17, d=8) plays where the true policy skips).
3. **Stop rule, explicit:** skip iff `min(57, s + 2d) <= m`. Elsewhere ties among
   exact maximisers prefer a play (single, then double). Codex r2 verified
   independently that with the production rates no strategic skip exists in the
   tail (smallest play-over-skip margin 5.7e-8 at (0, 55, 28)) and that the
   skips the raw DP produced in the reach-57 region are exact structural ties.
4. **m at runtime** = the contest profile's `best_streak` (fetched 4x/day), with
   a TRUST contract (`contest_state.classify_best_streak`): trusted only if an
   integer, `streak <= best <= 57`, current season, from the auto fetch or an
   unexpired manual override. Untrusted/missing degrade to best = streak, which
   keeps picking. Only a trusted best can stop the account (Codex r2 P0: an
   inflated best would otherwise stop the season).
5. **One quality bin** (late-season production-shaped rates: p_hit 2641/3600 =
   0.7336, p_both 1984/3600 = 0.5511, the double being the first lower-ranked
   candidate in a DIFFERENT game, as production executes). The 24-seed profiles
   replay the same 150 real late dates; Codex r2's date-cluster calculation puts
   top-20% vs rest at 1.4 SE. This is a variance-reduction trade chosen by the
   owner after the preview, not an exact quality-aware objective; an exact
   two-bin solve disagrees on 12% of tail cells (none on 9/04).
   `scripts/rebuild_tail_policy.py --n-bins 2 --dry-run` shows the alternative.
6. **Separate versioned artifact** `data/models/mdp_tail_policy.npz`
   (`bts_tail_policy_v1`, 8.7 KB): table `[s, m, d<=28, saver, q]`, its own
   boundaries, bin rates, a profiles manifest, and the sha256 of the base policy
   it pairs with. `load_tail_policy` is exhaustive (no pickle, every key, dtype,
   shape, action range, every consulted cell against the stop rule) and refuses
   a base-SHA mismatch. Generic reach-57 writers cannot erase it.
7. **Every failure shape** (no artifacts, base corrupt, tail absent/invalid/
   unpaired, lookup exception) resolves to `forced_tail_action`: skip iff the
   stop rule, else single — never the zero table, never the heuristic — and is
   recorded as `degraded_reason`.
8. **Provenance:** decision.json schema v3 adds objective, best_streak,
   best_status, effective_best, tail_policy_sha256, degraded_reason; pick files
   add `tail_policy_sha256`. One `PolicyDecision` is produced in
   `strategy.resolve_policy_decision` and reused for the log line, skip
   summary/DM/dashboard, both end-of-day skip capture paths, and the commit
   writer (no second state read for display). The skip-policy shadow and the
   boundary census admit only `objective in {absent, reach57}`; the
   `mdp_policy_alignment` health check excludes tail days; a new `tail_policy`
   health source (always-attention) loads both deployed artifacts every run.

## Numbers

Policy at best = 18, saver off (D double, S single, – skip), columns streak 0..18:

```
   d   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18
  24   D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  S
  10   D  D  D  D  D  D  S  D  S  D  S  D  S  D  S  D  S  D  S
   9   -  D  D  D  D  D  D  D  S  D  D  D  S  D  S  D  S  D  S
   6   -  -  -  -  -  -  -  D  D  D  D  D  D  D  S  D  S  D  S
   3   -  -  -  -  -  -  -  -  -  -  -  -  -  D  D  D  D  D  D
   1   -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  D  D
```

9/04 (d=24): double at streak 0, doubles until the streak passes 18. From 9/19
(d=9) a streak of 0 can no longer reach 19 and the system stops, as intended.
The alternating cells near the end land exactly on best + 1.

Artifact: `mdp_tail_policy.npz` sha256 `dc5d0c99…f949f4`, paired with base
`66d15471…b93b4` (the pooled reach-57 policy from e1ebde9; its source profiles
`pooled_bins_run` are lost with the old laptop and it cannot be re-solved, which is
why the tail is stitched onto it rather than the whole table regenerated).

## Codex round 3 (code review) — dispositions

Verdict was BLOCK on three P1s; all seven findings were adopted
(`2026-09-03-emax-tail-policy-codex-r3.md`):

1. **Cached-fallback / restart-recovery commits lost the objective** (selection
   is None on those paths). Fix: `select_pick` persists the `PolicyDecision`
   ON the DailyPick (`policy_decision`); `_commit_decision_for_pick` and
   `_write_classification_decision` read it when no selection exists. No path
   re-reads contest state for provenance.
2. **Trust inferred from the filename.** Fix: `classify_best_streak` trusts by
   CONTENTS (auto schema + `mlb_bts_profile` source, or a manual schema with an
   unexpired override) and rejects a future `source_date` (ET).
3. **Base failure silently dropped the tail's SHA binding.** Fix: without a
   base hash the tail is unverifiable — the loader marks it unusable, the
   decision path itself re-checks `base_sha256 == tail.base_policy_sha256`
   (defence in depth against injected dicts), and the health source reports it.
4. **Loader accepted a non-optimal table / empty manifest.** Fix: the loader
   re-solves from the embedded rates and requires table equality; the manifest
   has a typed key contract with count/rate consistency.
5. **Two-open race + unbounded parse.** Fix: one `read_bytes`, sha of exactly
   those bytes, `np.load(BytesIO)`, 2 MB cap, `max_days == 28` exactly (base
   policy loaded the same way in `_load_mdp`).
6. **Nullable v3 objective.** Fix: the writer coerces a missing objective to an
   explicit `reach57` and refuses anything outside the enum; a v3 record with a
   null/invalid objective reads as `unknown` (excluded everywhere), while v1/v2
   keep the legacy reach57 default.
7. **Census boundaries built from tail rows.** Fix: quintile samples come from
   reach-57 decisions only (`_boundary_sample_decisions`).

Codex also independently regenerated the shipped table from its embedded
rates: 195,112 / 195,112 actions equal.

## Rollout + verification checklist

- Fast suite green; new tests: `tests/simulate/test_tail_policy.py`, `tests/test_tail_policy_r3_fixes.py`,
  `tests/test_tail_policy_strategy.py`, `tests/test_contest_state_best_trust.py`,
  `tests/test_daily_decision_v3.py`, `tests/test_scheduler_tail_objective.py`,
  `tests/test_tail_provenance.py`, `tests/health/test_tail_policy_health.py`,
  `tests/test_tail_policy_artifact_contract.py` (loads the COMMITTED pair).
- Deploy in the post-game idle window; the scheduler wakes 10:00 ET 9/04.
- **First-day acceptance is NOT "we saw a double"** (best 0 and best 18 both
  double at streak 0): check `data/picks/2026-09-04/decision.json` for
  `objective == emax_season_best`, `best_status == trusted`, `effective_best == 18`,
  `tail_policy_sha256 == dc5d0c99…`, `degraded_reason == null`, and the journal
  line `Policy: objective=emax_season_best …`. A single with a degraded reason
  means an artifact failed on the box; a heuristic skip is a contract failure.

## Known residuals (not in this patch)

- Stale-low best: the profile updates after settlement; between a new high and
  the next fetch `max(streak, best)` covers it while the streak stands, but a
  reset in that window over-plays slightly until the next fetch (<= ~6 h).
- Shadow-model path passes no contest best (model replay streak only): it runs
  the tail with best = streak (labelled `best_status == missing`), keeps picking,
  and its rows are not policy comparisons.
- The double-down cutoff residual race (send_dm after the last check) is
  pre-existing and unchanged; the tail doubles at streak 0 daily, as the reach-57
  policy did on 8/29–9/02.
