# batch1 notes (E01–E28, early season 3/29–4/14)

28 records drafted (E01.json … E28.json). No row was EX, so every row has a file. All 28 pass draft-mode
validation together against the records.py/record_schema.json revision of 13:03 (per-occurrence
contemporaneity). The only messages are cross-batch `related` ids: I-037 (E37), I-104 (E104), I-106 (E106),
I-206 (L06).

## Batch conventions and systemic caveats (read first)

1. **The deploy candidates for every fix before 4/10 are Fly.io runs, not production.** Run 24253126060
   (4/10 16:29Z, head 4a82cf8) is the first successful run of workflow "Deploy BTS to Fly" (fcedf71,
   `flyctl deploy`). Production at the time was the Pi5 scheduler/orchestrator with SSH worker tiers (Mac,
   Alienware) until the 4/11 Hetzner cutover. I followed the guide's rule (candidate_ancestry with that run)
   and added a note in each affected record. The production install time of these fixes is not evidenced:
   the Pi5 was updated by hand, and the workers ran their own checkouts until 5207a09 (4/04 19:38 ET) made
   them `git pull origin main` before each predict-json. The one exception is E17, where the Fly run is the
   affected host itself.
2. **Every deploy log in this window has expired.** Every `ref_basis` is `candidate`. Every fix's
   first_live_log is the 6/07 bound (`eaccaf8`, live by 2026-06-07T00:11:19Z), noted per record.
3. **Hetzner-era deploy runs were themselves unreliable until 4/12.** The 4/11 runs pulled nothing
   (I-018). Workflow-only commits triggered no run until 8193909 added deploy.yml to the paths filter. Until
   the 4/12 12:37 self-heal, a legacy system-level scheduler kept running beside the one each deploy
   restarted (I-020). Candidate runs 24310493193–24311344697 therefore bound "code on disk", not "code
   running".
4. **Reading commit bodies.** A body counts as a contemporaneous operator report only when it says the
   failure was seen: "Reproduced live", "Observed live", "today's pick", quoted error text, "was silently
   failing", "posting as a second production instance". Bodies that only describe the mechanism are typed
   `inference`, and those rows stay `unresolved_candidate`, each with an "alternative reading" note.
5. **Occurrence sources for 3/29–5/10 are thin.** There are no journals. The Pi5 is decommissioned (4/14),
   so its logs have unknown retention. The bot's public Bluesky feed is not an inventoried source (design §4).
   The missing_evidence items name these, plus B1 pick/state files.
6. **Axes.**
   - `delivery_mode`: unknown for E01 (no posting code before 85b61f1, 3/31) and for windows that start
     3/29–3/30 (E03, E06). Otherwise public, and unknown for the undated latent E26.
   - `policy_objective`: reach57 for pick-path records from the MDP deploy (68a87a7, 4/01 17:15 ET); unknown
     for earlier pick-path records (E03, E04, E06); none for grading, display and deploy records.
7. **Contract sources.** Several contracts can only be sourced from the fix itself or from post-fix docs
   (ARCHITECTURE updates), as E83 does. Where the 2026-04-03 scheduler design is the pre-existing contract,
   it is cited.

## Per record

- I-001 (E01) | observed_incident | B | 3/30 own-team pitcher as opponent for games under way. Changed from UC/P and class D→U: only rows of games under way were affected, and those are past the cutoff. delivery_mode unknown (no posting code on 3/30). "Surfaced" is thin.
- I-002 (E02) | unresolved_candidate | tier_pending | check_hit wrong game_pk/batter_id. The fix came 9 min after the 01:00 slot, but the body reports no live failure. Fails closed ("Check manually").
- I-003 (E03) | unresolved_candidate | tier_pending | no data refresh before live predictions 3/29–4/01. "Fixes stale features" could be read as a live report. Its fix introduced I-004.
- I-004 (E04) | observed_incident | A | predict-json stdout contamination on every tier + Alienware cp1252. Changed from UC/P. Tier A rests on code inference: the NOTE-to-stdout (since ab3db97) broke every tier whenever lineups were projected. No body says a scheduled run made no pick; if the lead needs that, the tier is pending.
- I-005 (E05) | unresolved_candidate | tier_pending | first scheduler day: skip gate plus game-level confirmation. Link 1 fix bd8bcd3 (4/12) is shared with I-022. May be a live report ("the picked team's lineup").
- I-006 (E06) | unresolved_candidate | tier_pending | projected lineup included substitutes (3/29 23:34 → 4/03). Same hazard class as I-009 (an extra projected row can hold the lock gap open).
- I-007 (E07) | observed_incident | B | duplicate public "Streak reset to 0" reply. Class A→U (public reply, not an operator alert). ev1 carries X-01. Same root cause as I-011, split kept.
- I-008 (E08) | observed_incident | B | reconcile aborted with HTTP 400 on the null-game_pk 3/29–3/30 files. Changed from UC. The failing run could be the 4/04 02:00 cron or a manual run. Per-date isolation in today's reconcile was not verified; no residual is claimed.
- I-009 (E09) | observed_incident | A | 4/04 pick never locked: a postponed game's projected batter held the gap open. How and whether the pick was finally delivered is not evidenced. The residual I-206 (no independent day-outcome watchdog) reaches production.
- I-010 (E10) | unresolved_candidate | tier_pending | restart did not recognise an externally posted pick. The skipped-polling consequence may never have happened (I-011's body says polling updated the streak on 4/04).
- I-011 (E11) | observed_incident | A | 1am cron double-counted 4/04 (local streak 4 vs 2, X-01). streak.json fed the MDP state (run_and_pick at 4cf7b1e). Correction of the state is not evidenced.
- I-012 (E12) | observed_incident | B | scorecard 'G?' labels. Changed from UC; the body says "showed … on the dashboard".
- I-013 (E13) | unresolved_candidate | tier_pending | no in-loop fallback; restart after the deadline skipped the post. The timing suggests a live 4/06 case. Link 2 was introduced by 0632976 and fixed 2.7 min later (split? link 2 as pre_ship_exclusion if never live).
- I-014 (E14) | observed_incident | B | premature "Final — pick missed" on a cross-game double (X-01 on ev1). Changed from UC. Its fix introduced I-024.
- I-015 (E15) | unresolved_candidate | tier_pending | cross-game double graded from one leg, plus the NameError that fix introduced. split? (a) one-leg grading, (b) the NameError, which under Restart=always would crash-loop polling on 4/07–4/08. Neither firing is evidenced.
- I-016 (E16) | observed_incident | tier_pending | live pipeline timeout from the 23K-feed scan (44fdc84 12:20 → e1aaf61 15:11 on 4/08). Changed from UC. Whether all tiers timed out at a check is unknown.
- I-017 (E17) | observed_incident | A | Fly shadow instance posted publicly as a second bot. Verified recovery for 4/11 comes from 634768e's escape analysis. Whether the Fly machine was ever stopped is not evidenced.
- I-018 (E18) | observed_incident | A | 4/11 Hetzner deploys green while pull and restart silently failed. Changed from UC/P. Tier A reads "loss of recoverability" as "the deploy silently didn't take effect"; only dashboard cosmetics were stranded (B arguable). cda8c1c's diagnosis conflicts with 8193909's.
- I-019 (E19) | observed_incident | A | every cron job dead since cutover (`source` under dash). The crontab was unversioned, so the mechanism has no code ref. Residual: deploys never install crontab (reaches production). The reinstall time is not recorded.
- I-020 (E20) | observed_incident | A | two scheduler units, orphan respawn. No duplicate post is evidenced. Residual (reaches production): the self-heal covers one path and unit_drift covers only user units; no process-count check was found.
- I-021 (E21) | near_miss_control_held | B | fallback deadline after the double-down's first pitch. Changed from OI/A: nothing was delivered late. The "control" was the operator's hotfix (13:10) before the 13:37 cutoff — not an automated control; OI/B if the lead counts the live wrong deadline as the deviation.
- I-022 (E22) | observed_incident | B | "0 new confirmations" log while sides flipped the pick. Log/state only.
- I-023 (E23) | observed_incident | A | shadow file overwritten by production (DISAGREES→AGREES) plus the load-path masking. Repair of 2026-04-12.shadow.json is not evidenced. Watchdog boundary "grading" is a forced fit (no research boundary).
- I-024 (E24) | observed_incident | B | no scorecard when a Preview primary was merged with a Live double-down; dead header code. Introduced by I-014's fix 0ba01e1.
- I-025 (E25) | observed_incident | A | fallback posted the 13:19 cached pick over a fresher candidate. EX? / design gap: the 4/03 design did not require re-prediction at the fallback, and the gap is a model-probability comparison. Residual: a failed refresh still delivers cached, by design (reaches production).
- I-026 (E26) | deployed_latent_defect | tier_pending | in-place sort changed the LightGBM bagging sample. EX? (model quality, design §1). The only evidence is a single-seed backtest, and the seed-42 outlier finding (142db4d) weakens it. The lead row's "live model silently changed" is not shown. Watchdog "delivery" is a forced fit.
- I-027 (E27) | observed_incident | A | 3am chain died on `bts data pull` without --start/--end (4/13, 4/14). The body's "CLI was updated to require" is wrong: required since d991a79 (3/29). Continuation of I-019.
- I-028 (E28) | observed_incident | A | green deploy with a failed pull on box-local uv.lock drift, about 14 min. Recovery is bounded by run 24428702189 succeeding under set -e. Tier A under the same reading as I-018 (B arguable). The operator's uv.lock reset before the fix pull is not recorded.

## Out of scope
None: no EX rows in this batch. E25 and E26 carry EX? flags for the lead.

## Cross-batch related ids
I-037 (E37, silent no-op deploys 4/21–4/28) from I-028. I-104 (E104, workers ran unpulled code) from I-004.
I-106 (E106, 947fce8 O(n²) dead code) from I-016. I-206 (L06, independent day-outcome watchdog) from I-009.
