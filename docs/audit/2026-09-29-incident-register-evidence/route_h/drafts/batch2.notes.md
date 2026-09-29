# batch2 notes (E29–E56, 4/15–6/06)

28 records drafted, all pass `validate(publish=False)` against the 13:04 validator/schema. The only remaining
errors are cross-batch `related` ids (listed at the end). No row in this batch was EX, so there are no
out-of-scope lines.

## Per record
- I-029 (E29): observed_incident, A. The streak value is what the incident is about, so ev1 carries X-01. A = local state impact, plus a residual: the fix's today-exclusion is still in `_replay_season_streak` at HEAD and is the root of L02 (I-202). policy_objective reach57 is inferred from code.
- I-030 (E30): observed_incident, B. Doubt: the run date is inferred (5 resolved pairs + shadow files from 4/10 → run on 4/14–4/15). If the lead wants an explicit date, this becomes UC.
- I-031 (E31): observed_incident, B. Recovery was verified by the next push-triggered run.
- I-032 (E32): observed_incident, A. Doubt: A rests on reading the false /fail pings as an alert storm (O1 Healthchecks history not read). The c43869d body says "19:05 ET" but was written 16:30 ET; I read it as UTC (a guess).
- I-033 (E33): observed_incident, B. Doubt: the prelim was A. The actual polling gap was 36–130 min because the fix was deployed within the hour, and final grades were intact.
- I-034 (E34): observed_incident, A. The Phase B WatchdogSec change was a box-only unit edit. The DH-recheck residual was closed later by ce5f51d.
- I-035 (E35): observed_incident, A. Doubt (my code reading): ddee3db's idle sleep probably never engaged, because the lookahead wake was always already in the past (see I-049). The HEAD no-op residual reaches production.
- I-036 (E36): observed_incident, B. Doubt: the body's present-tense example is treated as a live observation. The defect was live for about 15 min; borderline pre-ship.
- I-037 (E37): observed_incident, tier_pending. deploy_history.txt (B1 reflog) settles whether any 4/22–4/28 run was a silent no-op. **E37 caveat applies to every 4/22–4/28 deploy candidate (I-031..I-036).**
- I-038 (E38): observed_incident, B. Proposed split: E38b, the unexplained 0→1 restart (UC; no journal before 5/11). My guess is that it was the I-049 past-wake exit, which after 20:00 ET gives exactly one restart.
- I-039 (E39): observed_incident, B. Doubt: the prelim was A. It is a latency regression with one genuine alert, fixed in 16 min. Why the heartbeat aged despite 0830ef4 is unexplained; an unwrapped shadow run is plausible.
- I-040 (E40): observed_incident, A. Link 2 (the silent failure) credits bb7ccc1 (E3 alert, "was dead config"). The 4/29–30 delivery impact needs the pick files (Route R).
- I-041 (E41): observed_incident, B. Doubt: "fired daily false positives" names no instance. If the lead reads it as inference, this is DL B.
- I-042 (E42): observed_incident, A. Doubt: A comes only via B→A, from the regime-fingerprint residual documented at HEAD (it overlaps I-073/F6). watchdog `none` is used outside the display/pre-ship rule. No calibration numbers are repeated.
- I-043 (E43): observed_incident, A (5/05). Proposed split: (a) 5/05 stale pick, observed; (b) void grading + polling gate, 5/09, no production occurrence evidenced; (c) detailed-status follow-ups, 5/24–26, code-review/DL. The "p0" hotfix was merged 5/06 but not deployed until 5/09 03:34Z (the box was at a3bc4d3 on 5/08).
- I-044 (E44): observed_incident, B. It is `continuing`, and the same-night memo is contemporaneous with the 5/08 backfill mitigation. Doubt: the prelim was A. If a month of missing results before the 30-day promotion threshold counts as material, it is A. Consider merging with E95 (the backfill).
- I-045 (E45): deployed_latent_defect, B. Display only (not needed for watchdog/restore coverage). Proposed split into 3 display defects.
- I-046 (E46): deployed_latent_defect, B. No evidence that it fired. The 14 AppleDouble files in the W0.7 picks tree have no dates. Guess: they came with the 5/08 Mac-side backfill work.
- I-047 (E47): observed_incident, A, `continuing`. **Duplicate of E96 (I-096), so merge.** The symptom is only implied, by the same-day production mitigation (shadow_model=false).
- I-048 (E48): observed_incident, B. Doubt: the canary may have been a TRUE detection of the I-049 loop. If so, relaxing it in ac4a959 is an undocumented residual that is still in deploy.yml, and B→A would apply. Merge with I-049?
- I-049 (E49): unresolved_candidate, A (HEAD no-op residual). Needs a machine record of the end-of-day exit/relaunch cycle, from heartbeat.log (B2) or the journal (only from 5/11). If confirmed, the exposure window is 4/23–5/09.
- I-050 (E50): observed_incident, B. Proposed split into 7 live-forward defects. Two are observed (5/25 false CRITICAL; resolver unit failed); links 2, 3, 5, 6 have no located observation.
- I-051 (E51): unresolved_candidate, tier_pending (preview-only failure vs scheduler failure). cron.log and the 5/11+ journal can settle it.
- I-052 (E52): unresolved_candidate, tier_pending. The only evidence for the OOM kill is the test-fixture timestamps (5/21 13:53 ET). The 5/21 journal (B1) should settle it. Proposed split into 4.
- I-053 (E53): unresolved_candidate, A. The detail comes from the 5/28 memo, six days later (quoted journal lines, marked reported). The B1 journal and pick JSON should make it observed. It is arguably a decision-contract change (secondary P).
- I-054 (E54): unresolved_candidate, B. A mislabelled pre-5/23 WARN is likely but not located. If none is found, this is DL.
- I-055 (E55): observed_incident, B, `continuing`. An explicit same-day runbook entry.
- I-056 (E56): observed_incident, A, `continuing`. The 6/06 spec and the 365ea40 body are contemporaneous with the 6/06 hotfix. X-01 on ev1. Link 2 (the max(model, contest) freeze) was fixed by E63's 2b4ff1d; consider moving that link to I-063.

## Cross-cutting
- **Deploys:** no retained deploy log exists before run 27077693688 (6/07 00:11Z), so every fix in this batch is `candidate_ancestry`. Every record carries its log bound in notes.
- **Validator change:** the validator and schema were updated at 13:04, while I was drafting (the `continuing` contemporaneity rule, operator reports must be cited by an occurrence, latencies rounded outward, no null latencies). E44 and E56 moved from UC to observed under the `continuing` rule. Later reports about other occurrences are typed `inference`.
- **Mechanism refs:** these follow the parent-of-fix rule, with `ref_basis` candidate throughout. For E40 link 2 and E46 link 2, I also cite the code that was live at the incident. For E56 links 1 and 3, the parents (f5b60af, fa30ecb) are PR #142 branch commits that were never deployed.
- **Outcome content:** nothing about who hit, and no rates or calibration figures, is restated. The exposure tag X-01 is used only on E29 ev1 and E56 ev1.
- **Cross-batch related ids (filtered):** I-011 I-014 I-021 I-023 I-024 I-028 I-057 I-058 I-059 I-063 I-064 I-073 I-074 I-075 I-076 I-083 I-084 I-087 I-092 I-095 I-096 I-097 I-108 I-109 I-202 I-203.
