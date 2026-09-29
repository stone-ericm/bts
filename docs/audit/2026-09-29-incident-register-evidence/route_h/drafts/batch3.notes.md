# batch3 notes (E57–E92, 6/06–9/22, excluding lead-owned plan-named rows)

26 records drafted; 2 rows out of scope (no file). All 26 pass draft-mode validation together; the only
messages are cross-batch `related` ids (listed at the end). Drafted by the batch3 drafter; E59, E73,
E64/E66/E67/E78 and E88–E92 were drafted by forks under the batch conventions below and reviewed here.

## Per record

- I-057 (E57) | observed_incident | B | nightly false CRITICAL 'STALE'. The body's "most nights" can't be literal: the check existed from 6/06 16:03 ET to the 6/07 fix, so at most the 6/06→6/07 EOD run fired. Route R should confirm the firing time. It is a spec defect (d383db5 required CRITICAL on any lag).
- I-058 (E58) | observed_incident | B | daily false "failed" DM. Only firings within 48 h of the 6/10 body count as the occurrence; earlier days (from the 6/07 deploy) go in recurrence.before_fix. The crontab install date is not evidenced.
- I-059 (E59) | deployed_latent_defect | A | fable5 audit batch, 24 links, split? (a)–(f) proposed in the record notes. Links 2 (daily realized_calibration CRITICALs on the biased path) and 3 (evening-restart day skip) are observed_incident candidates pending Route R. Links 18–20 (model/serving parity) and 22–24 (security hardening) may be out of scope. Tier A rests on residuals: post_failure 22:00 guard (drafter's code reading, unconfirmed by any body), I-206, I-202, I-204, I-205.
- I-060 (E60) | observed_incident | B | new CI test gate failed two deploys on the uv.lock/TZ mismatch. ref_basis log is for deploy.yml at the run's own head (cb7d00b). deployed follows the batch rule, but the workflow fix took effect in run 27283866776 (success).
- I-061 (E61) | observed_incident | A | 6/11 delivered pick never entered; no alert until 6/12. Link 1 is a human step (no code). The entry-check fix is v2 (a6ec548 + cron 720651a, code 7/04); the v1 check was disabled the same night (I-062). The cron activation date is unknown; the 7/06 DM (I-071) shows it running.
- I-062 (E62) | observed_incident | B | v1 entry check reported an entered pick as "not entered" (settled-only endpoint). Not stated whether this came as a cron DM or a manual run, so onset is date-bounded. Primary class A, secondary E (row had E/A).
- I-063 (E63) | observed_incident | A | decision/dashboard streak inflated by max(model, contest); fetch discarded the current activeStreak. The occurrence is the contemporaneously observed 6/16–6/17 state; the inferred 6/11→6/16 start is a before-fix recurrence (see convention 3). Streak values carry X-01. Link 4 ("daily false WARN past noon") may be anticipated rather than observed.
- I-064 (E64) | unresolved_candidate | tier_pending | [fork] 6/17 live-forward resolution stalled about 2 weeks on a suspended/resumed game. Every report comes ≥ 8 days after onset: the 7/01 audit, although contemporaneous with the 6/29 DM, is typed reported. Tier becomes A if the 6/17 production pick was on game 824912.
- I-065 (E65) | observed_incident | B | 6/18 correct skip gave no signal. The skip-vs-hang gap for a hung scheduler is left to I-206. Earlier silent skip days are not evidenced (Route R).
- I-066 (E66) | deployed_latent_defect | A | [fork] GH #144, 4 links, prelim UC. No pre-fix firing is reported, and 7/01 was the first live flip. The GH issue text was not read (no gh). Two accepted private-mode residuals were live from 9/14.
- I-067 (E67) | deployed_latent_defect | A | [fork] suspended-game grading used resumed-portion PA. Tier A comes from the clause-C residual (I-201). The fork added 793a093 and 63fd1d0, which the pack omitted. split? production grading vs research/monitor labels.
- I-068 (E68) | deployed_latent_defect | B | ungated 3am sync-to-r2. The harmful window (new SCHEMA_VERSION + ungated sync) contained no 3am run; the 7/01 audit shows the gate exercised. deployed basis operator_report (crontab), the only such case in the batch. Kept for restore coverage.
- I-069 (E69) | observed_incident | B | 7/01–7/02 first live flip: dashboard hid the SKIP banner and showed the declined pick; bar wording; history rows. No wrong entry is evidenced. Links 1–2 cite the installed tree 83a5cad as candidate; link 3 cites 48e8485 as log.
- I-071 (E71) | observed_incident | A | 7/06 premature "not entered" DM for an undelivered, later-deferred DD preview. Tier A only via corruption-class residuals verified unfixed at HEAD by code reading: load_decision has no internal-date check, is_scoreable_commit ignores action, and the checker reads {date}.json. Without them it would be B.
- I-072 (E72) | observed_incident | A | 7/08 DD leg never entered after a single 18:00 alert ('alerted' terminal); 7/09 is a second occurrence with no entry impact. The "+1 not +2" cost carries X-01. e5ef7ca is in the fix set (present_unverified escape, I-110). Detection latency is not_applicable because the detector fired before the violation instant.
- I-073 (E73) | deployed_latent_defect | A | [fork] gpt-5.6-sol audit batch, 10 links, split? (a)–(h) in the record notes. F2's 11 hazard days are reported only (auditor scan). F10 is the /tmp derivation loss, seen 7/10. d960958 fixed a claimed-but-missing R3 fix (see the fork's convention question 1).
- I-074 (E74) | unresolved_candidate | B | 7/10 shadow result stranded a month; prelim OI. The 8/09 bodies are 30 days after onset, so they are typed inference; Route R should read the stranded state. Link 2's wait loop needs a crontab install (not dated). split? 4f0257a also closes a latent old-date-scoring hazard. Prelim tier was A: B because shadow-only, no data loss.
- I-076 (E76) | deployed_latent_defect | B | calibration monitors blind to a chronic DD-leg shortfall; realized_calibration WARNs never reached the digest. SCOPE: probably EX (design §1, model quality/calibration, like I-081/EX2). Prelim OI changed because no operational firing exists. Watchdog boundary "none" with reason. Calibration figures not restated.
- I-078 (E78) | near_miss_control_held | B | [fork] stale provisional file on a skip day; the #144 gates held. The fork added the 7/01 flip as the establishing occurrence; 7/28 control-held is reported only. ev6 cites a claude-shared memory pointer (Route X, reported), which is outside the repo. UC if restricted to 7/28.
- I-079 (E79) | deployed_latent_defect | tier_pending | bare urlopen in discover_games. DOUBT: the fix body says "backfill-only, live pipeline never affected", but at 950c081^ the live cascade's season refresh (predict_local → run_pipeline → _refresh_season_data → pull_feeds) calls discover_games. Firing is unknown (Route R: journal "Prediction failed" urlopen errors 5/11–8/03, cron.log 3am pull failures).
- I-080 (E80) | deployed_latent_defect | B | decision.json v1 carried no decision-time state on commits and dropped the skip second candidate. Arguably a design limitation, not a defect (v1 met its documented 6/20 contract); exposed by the 8/09 census research need. Not relevant to W4; the lead may drop it. Watchdog boundary "preservation" is a forced fit.
- I-088 (E88) | unresolved_candidate | A | [fork] C-01 leaderboard parser stored the active streak in all_season/all_time rows. Firing evidence is C-01's 9/22 box-data confirmation (reported). Tier B is arguable. split? envelope validation.
- I-089 (E89) | unresolved_candidate | A | [fork] C-03 reconcile applied post-cutoff re-scoring (5/10, 8/20); fix ce6676d is main-only (not_deployed). Awaits the Route R §5.4(ii) read and cron.log "CORRECTIONS FOUND". X-01/X-19 tagged.
- I-090 (E90) | unresolved_candidate | A | [fork] 6/05 laptop loss of pooled_bins_run. Ruled IN scope (§2 "preservation/restore dependencies … on any host"), but borderline: mdp_policy.npz was never lost, only re-solvability. No repo record is written within 48 h.
- I-091 (E91) | deployed_latent_defect | A | [fork] ops state had no off-box backup 4/11–7/10. The window end (first successful restic ops snapshot) is not in the repo (Phase 2 B2). Tier A via residuals: data/validation/ is in no backup set; backup_freshness is silent when unarmed.
- I-092 (E92) | deployed_latent_defect | B | [fork] live-forward idled at pending_pick on no-pick days until D8. By-design coverage limit (P-04); the lead may drop it. The 11-of-79 count is reported only.

## Out of scope (no record)

- E70 | 7/04 authenticated deep scrape disabled (49a8446): a deliberate owner decision (MLB ToS §1(xi) / account risk to the streak), not a service failure (design §1/§2). It belongs with the leaderboard field-capture policy (docs/audit/2026-09-22-exposure-register.md owner decisions; docs/audit/2026-09-22-final-grab-runbook.md). The public static captures and own-account crons kept running.
- E81 | 8/09 mdp_policy_alignment bin-collapse WARN (chronic): a model/policy-quality signal, not a service contract deviation (design §1: the register measures no model quality or policy value). It belongs to the W1.4b due read "mdp_policy_alignment bin collapse" (plan 2026-09-14 §W1.4b, with EX2).

## Batch conventions applied (for the lead's harmonisation)

1. **run_id for the June log bound.** June fixes use deployed = log, sha 83a5cad, live_by 7/02 15:28:10Z, not_live_before 6/07 17:55:03Z, run_id 28601525167. That run's retained log first observed 83a5cad, as the pre-deploy SHA. No run has a logged deployed SHA 83a5cad: 28520680537 deployed head 83a5cad, but its log expired. Every record notes its candidate-ancestry run. The 7/01 audit also reports box HEAD 83a5cad and .last_deploy_iso 2026-07-01T13:26:19Z; this is used in notes only.
2. **Defective ref when fix^ is docs-only.** Where fix^ is a docs-only descendant of the installed tree, the cite is the installed SHA with ref_basis log, following the E83 precedent (404358d). This applies to E69 link 3, E71, E72, E74, E76, E79 and E80. src/ was verified identical in each case.
   - The E88–E92 fork instead added a second cite at the logged SHA.
   - An installed ANCESTOR of fix^ is not "that ref or a descendant", so in strict reading these would be candidate.
3. **Continuing conditions vs the 48 h rule (needs a ruling).** The validator measures contemporaneity from the EARLIEST onset. A report written live during a multi-day condition therefore cannot establish OI for that condition's start. Applied here:
   - **Discrete events** (daily DMs, per-day decisions): the ones within 48 h of the report are the occurrence, and earlier ones go in recurrence.before_fix (E58, E63).
   - **Pure continuing state** (stall, strand, stored rows): unresolved_candidate with missing_evidence (E64, E74, E88).
   - Fork D and the E88–E92 fork raised the same question independently.
4. **Latent defects.** Every deployed_latent_defect has occurrences: [], with the exposure interval in notes; verified_recovered is "not_applicable" or "unknown".
5. **Crontab-activated fixes.** Deploys never touch crontab.
   - E68 is dated by an operator report (the 7/01 audit).
   - E74 link 2 deployed is "unknown".
   - For E61/E62, the v2 entry-check cron is first shown running by the 7/06 DM (I-071).
6. **One event in several records.** The 7/01 first live flip appears in I-066 (verified_recovered), I-069 (display incident) and I-078 (near-miss occurrence).
7. **Scope doubts for the lead:**
   - I-076: EX candidate.
   - I-080 and I-092: may drop.
   - I-090: borderline.
   - I-059: links 18–20 and 22–24.
   - I-073: F15 (manual preview only).
8. **Outcome values.** Restated only where the incident is the value: I-063 (X-01), I-072 (X-01), I-089 (X-01, X-19) and I-073 ev3 (X-01). Candidate names, probabilities and streaks from memos are otherwise omitted.

## Cross-batch related ids

I-056, I-075, I-082, I-083, I-084, I-093, I-100, I-102, I-108, I-109, I-110, I-111, I-201, I-202, I-203,
I-204, I-205, I-206, I-207, I-208, I-209.
