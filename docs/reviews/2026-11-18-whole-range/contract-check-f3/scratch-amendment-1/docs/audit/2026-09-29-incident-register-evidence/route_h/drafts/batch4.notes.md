# batch4 notes (E93–E111, L03–L09, EX2, EX3)

All 26 records validate in draft mode (0 errors, including against every other draft now in `drafts/`, so no dangling `related` ids).
Validator used: the updated `records.py` (per-occurrence contemporaneity ruling of 2026-09-29, cited-report rule, strict latency bounds).

## Records
- I-093 (E93) unresolved_candidate · B — changed from DL: the defective 03:00 line lived 45 min in the script (4/12 11:46–12:31) with no 03:00 run inside it, and before that the manual crontab died at `source`. It fired only if d5c466c was installed and not replaced before 4/13 03:00. Otherwise it is a pre_ship_exclusion. Includes a second defect in the same line (sync-to-r2 before preview).
- I-094 (E94) unresolved_candidate · tier_pending — split? The row's subject is not an incident. The day 1–3 pick files are 3/31 reconstructions because pick persistence shipped 3/31 18:22; that belongs in the memo's coverage inventory. The record covers the only defect in 1c2efaa: a lock keyed on game start, so re-selection could overwrite a posted pick, or a run on a no-file day could post again. Possible window: 3/31 18:49–22:08 ET. Firing is unknown.
- I-095 (E95) observed_incident · B — the same deviation as I-044, which a sibling drafted as UC/B. Merge them and align dispositions. OI here rests on the new contemporaneity ruling: the memo addendum was written 30 min after the repair ended the stranded state.
- I-096 (E96) observed_incident · A — duplicate of I-047 (the sibling says so too); merge. This row adds the operator mitigation (shadow_model=false) and the permanent 5/09 shadow gap. The date shadow_model was restored is unknown.
- I-097 (E97) observed_incident · B — the 5/10 live-forward day was lost; the integrity guard held (no post-outcome backfill). The capture-timer install date and the worktree move to a8632ce are not recorded.
- I-098 (E98) unresolved_candidate · A — the A tier comes only from the B→A rule (deploy timing is guarded by documentation only). The only source for "muddied read" is a 7/02 CLAUDE.md line. Facts: the 6/22 deploy was at 21:36 ET (run 27996097986), and no decision.json exists for 6/22. If the lead calls the residual generic (V9), the event is tier_pending.
- I-099 (E99) observed_incident · B — the fix is box-side config dated by 48e8485. Verified recovery would be the 7/02 deploy stop logging cleanly (Phase 2).
- I-100 (E100) pre_ship_exclusion · B (counted false) — changed from NM: both blocks were test-only defects (production code correct), so the gate held against no hazard. Alternative reading: list the gate as a working control. A third failed run, 27393707216 (6/12, log expired), has an unknown cause.
- I-101 (E101) pre_ship_exclusion · B (counted false) — changed from NM: no control is named in the sources. 3697512 was never installed (the retained log shows 3e1791c → 2ff2db9).
- I-102 (E102) near_miss_control_held · A — changed from OI: the only source is an 8/09 spec (46 days later), which reports that contest anchoring held. The A tier comes from the B→A rule on the local-vs-contest divergence. Hypothesis (unverified): the +2 is I-063's unentered 6/11 double-down. The lead may fold this into I-063.
- I-103 (E103) deployed_latent_defect · B — no watchdog or restore relevance; the lead may drop it. "Could not fire" rests on streak values (X-01) and the unknown pull time of the grading host.
- I-104 (E104) deployed_latent_defect · tier_pending — changed from UC: the 4/01 tier-failure commits give deployment evidence. The SSH tiers are retired, so the lead may drop it.
- I-105 (E105) unresolved_candidate · B — split? Four display-only defects; the lead may drop them or group them.
- I-106 (E106) unresolved_candidate · tier_pending — split? Part (b) of 947fce8 should be its own record: R2 sync omitted mdp_policy.npz, so a restore from R2 (Fly) silently fell back to the heuristic. That is restore-coverage, relevant to W4. The 30+ minute predictions were observed on Fly, not production.
- I-107 (E107) deployed_latent_defect · B — credential hygiene; the rotation (6/06) is the effective fix. The ping UUID is not reproduced.
- I-108 (E108) deployed_latent_defect · A — fable5 audit H5, fixed by the H5b set (6/12 candidate). Residual: the fix only alerts; nothing auto-recovers.
- I-109 (E109) deployed_latent_defect · B — run metadata rules out a firing in the gated era (shortest success→next-run gap 13.35 min against a 2.9–9.6 min reset interval; no cancelled runs).
- I-110 (E110) deployed_latent_defect · A — installs logged 7/04 04:10Z → 7/10 15:20Z. The masking needs an entered row the crosswalk cannot resolve. 7/08 took the "alerted" path (I-072). Residual: a present_unverified marker at the cutoff produces no EOD record (my reading of `pick_entry.py`).
- I-111 (E111) deployed_latent_defect · A — the A tier comes from the residual (the operator saver flag, an accepted risk). Inferred but not observed: during I-063's divergence the proxy returned False while the saver was available.
- I-203 (L03) deployed_latent_defect · A — unfixed; the §10.3 characterization fixture is owed or must be deferred at publish.
- I-204 (L04) deployed_latent_defect · A — unfixed; §10.3 fixture owed or deferred. The saver-reset consequence is my code reading.
- I-205 (L05) deployed_latent_defect · A — split? It bundles three independent 8/30 deferrals (F2 delivery_unknown, F10 feed files, F1 in-transport deadline).
- I-206 (L06) deployed_latent_defect · A — the backlog item E83 and E65 point to.
- I-207 (L07) deployed_latent_defect · A — link 1 exists only in the undeployed ce6676d (it reaches production with the 2027 deploy); links 2–3 are in the deployed reconcile. Split?
- I-208 (L08) deployed_latent_defect · B — recommend dropping: no cron or unit runs enrich-weather and no feature reads the weather columns. The "8/03 note" named by the row is not in the repo.
- I-209 (L09) deployed_latent_defect · A — the missed-pick half was closed 8/14 (224ddce). The entry-check gate is unchanged by design. Residual: local grading also skips such a commit (my code reading, not in any source).

## Out of scope (no record file)
- EX2 (I-302 if ever drafted) — "MDP policy maps live picks to Q0 (bin collapse)". Out of scope: it is a model/policy-quality question, and the register measures no model quality, calibration or policy value (design §1; §2 scope). It belongs to W1.4b's owed bin-collapse read (see also the 8/09 DD-tripwire memo, f23010f).
- EX3 (I-303 if ever drafted) — "research-audit cloud fleet: silent retrieve failure left zombie boxes (4/14)". Out of scope: research tooling (`scripts/audit_driver.py`, `audit_attach.py`), which is not part of the deployed service, its research streams or their preservation (design §2 scope). The shipped policy's lost profiles are E90's MacBook loss, not this fleet. It belongs to research-tooling hygiene (the 4/24 teardown-safety spec/plan, d3e5082/d868eff).

## Conventions and cross-batch points for the lead
- Fly-workflow candidates: the lead's concurrent sweep set `deployed` to `unknown` in E94, E103, E104, E105 and E106 and added notes; I kept those edits and trimmed my own overlapping notes.
- Latent links use `mitigated: none` and `verified_recovered: not_applicable`, following the lead's L01/L02; I-073's author used `unknown`.
- Deploys bounded only by the 83a5cad log (I-108, I-111) use run 28601525167, the first retained log showing 83a5cad, as I-063 does. The candidate run is in the notes.
- Sibling references to my ids are consistent: I-063 → I-102 (residual), I-073 → I-109/I-110/I-203/I-204, I-059 → I-108/I-111/I-204/I-205/I-206, I-089 → I-207, I-084 → I-205, I-047 ↔ I-096, I-044 ↔ I-095.
- Every record ends with the fixtures note. Regression tests named in fix commits are listed in the notes of I-095, I-096, I-101, I-103, I-108, I-110 and I-111.
