# GitHub issues and pull requests sweep (checklist C5, roadmap 0.4)

Date: 2026-11-14. Tier D record. Source R3 of the incident-register design (`docs/superpowers/specs/2026-09-29-incident-register-design.md`): GitHub issues and pull requests, all states. Phase 1 of the register did not sweep it.

## What was swept
- Every issue and pull request in the repository in every state, read with `gh issue list --state all` and `gh pr list --state all` (limit 1000; the totals are far below it): **2 issues** (both closed as completed) and **142 pull requests** (136 merged, 6 closed without merging), created 2026-04-20 to 2026-06-22. After 2026-06-22 work went straight to `main`, so nothing later exists in this source.
- A pull request counts as already covered when the register JSON (`docs/audit/2026-09-29-incident-register.json`, 119 records) contains its merge commit (first seven characters) or its number.

## Result
- **68 of 142** pull requests are cited by at least one register record.
- The other **74** were classified by hand: 31 research or validation tooling, 23 documentation or pre-registration, 4 operational features, 12 either closed without merging or describing something the register already records under another citation, 1 out-of-scope tooling note, and **3 that point at 2 candidates**.
- Both issues are covered or out of scope (table below).

## Candidates (listed, not certified)
**C1. The reach-57 policy's probability bins did not match the probabilities production served.** Pull request #109 (2026-05-23, `src/bts/health/mdp_policy_alignment.py`) says its preflight found that "all production primary picks have landed in Q0 against the current boundary scale"; #121 (2026-05-24, `docs/sota_audit/2026-05-24-probability-scale-pa-basis.md`) gives the cause: the policy was solved on profiles that compound the per-PA probability over the realised PA count, and production compounds over an estimated count. No register record covers it (searches for the bin, the PA basis and the scale return only I-090, the loss of the policy's profiles). Later analysis exists (`docs/audit/2026-06-29-skip-threshold-and-discrimination.md`, `docs/audit/2026-07-06-strategy-model-lever-investigation.md`; the profile-basis warning in `CLAUDE.md`). Stream would be production, disposition a deployed latent defect: the policy's day-quality input carried no information for an unknown span before 2026-05-23. Whether it changed any pick is not established here.

**C2. The live-forward capture timer failed when it re-checked a day whose artifact already existed.** Pull request #91 (merged 2026-05-11 UTC, `scripts/live_forward_capture_once.py`): the runner read the day's pick file before verifying an existing artifact, so the timer failed after a pick resolved or when the pick file was absent. Stream live_forward. The nearest records are I-050 and I-097; neither cites this defect. No effect on picks.

Neither is new information about 2027. C1 is a known and decided state: the 2026-06-29 follow-up kept the thresholds as they are, so the mismatch persists into 2027 unless the policy is rebuilt, and the profile-basis rule in `CLAUDE.md` is the standing guard for studies. It is listed because the register has no record of it. C2 was fixed by #91 itself, merged within hours of the capture automation (#90). Both go to register Phase 2 with the other deferred items (roadmap decision 8); the register tooling stays frozen.

## Note
**N1.** Pull request #19 fixed two `set -euo pipefail` patterns in a research provisioning script that aborted before the operator-facing error path could run. Research tooling, outside the register's streams; recorded here only because it is a silent-failure pattern.

## Limits
- The 74 uncited pull requests were classified from their titles. The opening 500 characters of the body were read for nine (#19, #65, #71, #82, #91, #117, #121, #134, #139) and every body was scanned for incident wording (incident, bug, regression, crash, silent, missed, corrupt, wrong, production); the scan surfaced #109 and nothing else operational.
- Review comments and commit-level discussion inside pull requests were not read.
- "Cited" means the string match above, not that the record's account of the pull request was re-checked.

## Issues
| # | State | Created | Title | Disposition |
|---|---|---|---|---|
| 7 | closed | 2026-05-03 | harness v2: per-bin rho_pair + within-fold dependence-parameter re-estimation | research enhancement request (falsification harness v2); not an incident |
| 144 | closed | 2026-06-21 | check-results scores undelivered preview/provisional picks on skip days (streak corruption) | recorded as I-066 and I-078 |

## Pull requests
| # | State | Created | Title | Disposition |
|---|---|---|---|---|
| 2 | merged | 2026-04-20 | Scheduler hardening: heartbeat monitor + dynamic fallback + sd_notify (Phase A) | cited by I-032, I-042, I-073 |
| 3 | merged | 2026-04-29 | Phase 2 keep-rule tightening + retro items 1-5 + pooled-prediction infra (rejected) | cited by I-042, I-059, I-073, I-202 |
| 4 | merged | 2026-04-29 | feat: promote batter_pitcher_shrunk_hr to FEATURE_COLS | cited by I-011, I-032, I-039, I-040 |
| 5 | merged | 2026-04-29 | fix(features): vectorize bpm cumsum — restore prediction-cycle latency | cited by I-039, I-073 |
| 6 | merged | 2026-05-03 | feat: falsification harness — Task 13 verdict HEADLINE_BROKEN | cited by I-110, I-034 |
| 8 | merged | 2026-05-03 | feat: harness v2.5 — nested factorial ablation, A-effect = null | research, validation or audit tooling and evidence (no operational stream) |
| 9 | merged | 2026-05-04 | feat(validate): SOTA #12 phase 1 — probabilistic forecast evaluation suite | cited by I-073 |
| 10 | merged | 2026-05-04 | feat(validate): SOTA #5 phase 0/1 — validation contract (lockbox + purged blocked CV) | research, validation or audit tooling and evidence (no operational stream) |
| 11 | merged | 2026-05-04 | feat(validate): SOTA #11 phase 0/1 — conformal gate v2 (binary-y validation methodology repair) | cited by I-073 |
| 12 | merged | 2026-05-04 | docs(sota-13): Phase 0 design memo — OPE policy-value evaluation | cited by I-009 |
| 13 | merged | 2026-05-04 | feat(validate): SOTA #13 phase 0/1 — policy-value evaluation over #5 manifest | research, validation or audit tooling and evidence (no operational stream) |
| 14 | merged | 2026-05-04 | docs(sota-14): Phase 0 design memo — CE-IS rare-event MC over #5 manifest | documentation or pre-registration only |
| 15 | merged | 2026-05-04 | SOTA #14 P0/P1 — CE-IS rare-event MC over #5 manifest | research, validation or audit tooling and evidence (no operational stream) |
| 16 | merged | 2026-05-05 | docs(sota-audit): falsification-harness synthesis memo (post-PR #15) | cited by I-050, I-097 |
| 17 | merged | 2026-05-05 | SOTA #12 phase 2 — realized-picks calibration on canonicalized stream | research, validation or audit tooling and evidence (no operational stream) |
| 18 | merged | 2026-05-05 | feat(picks): pick provenance v1 — model_git_sha + blend/policy artifact hashes | cited by I-043 |
| 19 | merged | 2026-05-05 | fix(scripts): v2.5 provision dry-run preflight + non-silent error paths | N1: research tooling defect, outside the register's streams |
| 20 | merged | 2026-05-05 | docs: refresh architecture validation and provenance notes | documentation or pre-registration only |
| 21 | merged | 2026-05-05 | feat(validation): add park-environment attribution to realized picks | research, validation or audit tooling and evidence (no operational stream) |
| 22 | merged | 2026-05-05 | feat(validation): add season-to-date batter skill quartile to realized picks | research, validation or audit tooling and evidence (no operational stream) |
| 23 | merged | 2026-05-05 | feat(validate): add BH/BY FDR baseline over realized-picks Cut C | cited by I-043 |
| 24 | merged | 2026-05-06 | fix(scheduler): regenerate stale postponed picks | cited by I-043 |
| 25 | merged | 2026-05-06 | feat(health): alert on stale postponed picks | cited by I-043 |
| 26 | merged | 2026-05-06 | [codex] Audit MDP objective and replay saved policies | research, validation or audit tooling and evidence (no operational stream) |
| 27 | merged | 2026-05-06 | Add DR-MDP gap measurement screen | research, validation or audit tooling and evidence (no operational stream) |
| 28 | closed | 2026-05-06 | Document DR-MDP gap evidence | closed unmerged; duplicate of #29 (says so in #29) |
| 29 | merged | 2026-05-06 | Document DR-MDP gap evidence | documentation or pre-registration only |
| 30 | merged | 2026-05-07 | Document pooled-seed surface inventory | documentation or pre-registration only |
| 31 | merged | 2026-05-07 | Screen pooled policy gap uncertainty | research, validation or audit tooling and evidence (no operational stream) |
| 32 | merged | 2026-05-07 | Measure pooled raw DR-MDP gap | research, validation or audit tooling and evidence (no operational stream) |
| 33 | merged | 2026-05-07 | Screen pooled gap determinism evidence | research, validation or audit tooling and evidence (no operational stream) |
| 34 | merged | 2026-05-07 | Refresh binary-y conformal gate evidence | research, validation or audit tooling and evidence (no operational stream) |
| 35 | merged | 2026-05-07 | Inventory nested CV lockbox adoption | documentation or pre-registration only |
| 36 | merged | 2026-05-07 | Design experiment runner nested split | documentation or pre-registration only |
| 37 | merged | 2026-05-07 | Add experiment runner nested season split | research, validation or audit tooling and evidence (no operational stream) |
| 38 | merged | 2026-05-07 | Update experiment split tracker status | documentation or pre-registration only |
| 39 | merged | 2026-05-07 | Reconcile SOTA tracker closeout status | documentation or pre-registration only |
| 40 | merged | 2026-05-07 | Add audit verdict FDR retrospective | research, validation or audit tooling and evidence (no operational stream) |
| 41 | merged | 2026-05-07 | Close out candidate-generation audit prerequisites | documentation or pre-registration only |
| 42 | merged | 2026-05-07 | Add real split-audit plan | documentation or pre-registration only |
| 43 | merged | 2026-05-07 | Add split-audit budget tiers | documentation or pre-registration only |
| 44 | merged | 2026-05-07 | Record split-audit capacity limits | documentation or pre-registration only |
| 45 | merged | 2026-05-07 | docs: record verified OCI split-audit capacity | documentation or pre-registration only |
| 46 | merged | 2026-05-07 | feat: pass split flags through audit driver | research, validation or audit tooling and evidence (no operational stream) |
| 47 | merged | 2026-05-07 | Prepare OCI canary launches | research, validation or audit tooling and evidence (no operational stream) |
| 48 | merged | 2026-05-07 | Tag audit provider provenance | research, validation or audit tooling and evidence (no operational stream) |
| 49 | merged | 2026-05-07 | [codex] Add raw profile audit launcher mode | research, validation or audit tooling and evidence (no operational stream) |
| 50 | merged | 2026-05-08 | [codex] close pooled-policy audit cycle | documentation or pre-registration only |
| 51 | merged | 2026-05-08 | [codex] preregister decision-aware audit cycle | documentation or pre-registration only |
| 52 | merged | 2026-05-08 | Add decision-weighted LightGBM candidate | research, validation or audit tooling and evidence (no operational stream) |
| 53 | merged | 2026-05-08 | Add candidate artifact comparison commands | research, validation or audit tooling and evidence (no operational stream) |
| 54 | merged | 2026-05-08 | Add live-forward candidate artifact export | cited by I-097 |
| 55 | merged | 2026-05-08 | [codex] Record decision candidate launch SHA | cited by I-095 |
| 56 | merged | 2026-05-09 | [codex] Fix shadow result reconciliation | cited by I-030, I-095 |
| 57 | merged | 2026-05-09 | [codex] Add shadow backfill analysis tooling | cited by I-045 |
| 58 | merged | 2026-05-09 | [codex] Refresh dashboard game progress with scorecard | cited by I-045, I-095 |
| 59 | merged | 2026-05-09 | [codex] Document shadow backfill quality dry run | cited by I-043, I-045, I-046, I-095 |
| 60 | merged | 2026-05-09 | [codex] Skip AppleDouble pick files in realized calibration | cited by I-046 |
| 61 | merged | 2026-05-09 | [codex] Add shadow production apply addendum | cited by I-095 |
| 62 | closed | 2026-05-09 | [codex] Scheduler live candidate artifacts | closed unmerged; live-forward research feature |
| 63 | merged | 2026-05-09 | [codex] Add shadow pick monitoring status | operational feature or visibility; no defect described |
| 64 | merged | 2026-05-09 | [codex] Add candidate artifact verifier | cited by I-032, I-047 |
| 65 | closed | 2026-05-09 | [codex] Guard shadow scheduler watchdog | closed unmerged; describes the shadow/watchdog incident already recorded as I-047 |
| 66 | merged | 2026-05-09 | [codex] Guard shadow scheduler watchdog | cited by I-032, I-039, I-047 |
| 67 | closed | 2026-05-09 | [codex] Add live-forward ops runbook | closed unmerged; documentation or pre-registration only |
| 68 | closed | 2026-05-09 | [codex] Add live-forward outcome resolver | closed unmerged; research tooling (body: non-prod) |
| 69 | merged | 2026-05-09 | [codex] Add live-forward ops runbook | cited by I-047, I-097 |
| 70 | merged | 2026-05-09 | [codex] Add live-forward outcome resolver | cited by I-067 |
| 71 | merged | 2026-05-09 | [codex] Avoid stale deploy SHA in runbook | documentation: a hardcoded deploy SHA in a runbook went stale; no wrong deploy recorded |
| 72 | closed | 2026-05-09 | [codex] Add BIP xBA to live scorecards | closed unmerged; dashboard feature (see #78) |
| 73 | merged | 2026-05-09 | [codex] Add BIP xBA to live scorecards | cited by I-043 |
| 74 | merged | 2026-05-09 | Handle postponed locked picks as void results | cited by I-043, I-045, I-047 |
| 75 | merged | 2026-05-09 | Treat schedule voids as final in result polling | cited by I-043, I-048 |
| 76 | merged | 2026-05-09 | Allow clean scheduler completion in deploy canary | cited by I-043, I-048, I-049 |
| 77 | merged | 2026-05-09 | Use schedule date for next wakeup | cited by I-035, I-048, I-049 |
| 78 | merged | 2026-05-09 | Render live xBA per scorecard PA | operational feature or visibility; no defect described |
| 79 | merged | 2026-05-09 | Normalize scorecard PA alignment | cited by I-045 |
| 80 | merged | 2026-05-10 | Fix resolved dashboard pick state | cited by I-045 |
| 81 | merged | 2026-05-10 | Refresh realized-picks calibration evidence | research, validation or audit tooling and evidence (no operational stream) |
| 82 | merged | 2026-05-10 | Add production-pick guard for live candidate artifacts | live-forward evidence guard (feature) |
| 83 | merged | 2026-05-10 | Pre-register live-forward comparison rule | documentation or pre-registration only |
| 84 | merged | 2026-05-10 | Add leaderboard clue audit tooling | research, validation or audit tooling and evidence (no operational stream) |
| 85 | merged | 2026-05-10 | Add leaderboard backfilled model audit | research, validation or audit tooling and evidence (no operational stream) |
| 86 | merged | 2026-05-10 | Add realized production leaderboard audit surface | research, validation or audit tooling and evidence (no operational stream) |
| 87 | merged | 2026-05-10 | Pre-register leaderboard mechanism mining | documentation or pre-registration only |
| 88 | merged | 2026-05-10 | Add live-forward provenance inventory | research, validation or audit tooling and evidence (no operational stream) |
| 89 | merged | 2026-05-10 | Export live-forward ranked surfaces | cited by I-097 |
| 90 | merged | 2026-05-11 | Automate live-forward capture | automates the live-forward capture whose manual miss is I-097 |
| 91 | merged | 2026-05-11 | Verify existing live-forward captures first | **candidate C2** |
| 92 | merged | 2026-05-11 | Constrain live-forward capture timer window | cited by I-045, I-097 |
| 93 | merged | 2026-05-12 | Fix dashboard scorecard unresolved results | cited by I-045 |
| 94 | merged | 2026-05-12 | Automate live-forward artifact resolution | cited by I-050 |
| 95 | merged | 2026-05-12 | Handle terminal voids in live-forward resolver | cited by I-050 |
| 96 | merged | 2026-05-12 | Allow legacy live-forward backfill verification | cited by I-050 |
| 97 | merged | 2026-05-13 | Guard live-forward capture against stale pick snapshots | cited by I-050, I-051, I-087 |
| 98 | merged | 2026-05-15 | [codex] Fix preview pick NaN pitcher IDs | cited by I-050, I-051 |
| 99 | merged | 2026-05-18 | [codex] Fix live-forward pick snapshot drift | cited by I-050 |
| 100 | merged | 2026-05-19 | Show Kenya time on dashboard | operational feature or visibility; no defect described |
| 101 | merged | 2026-05-20 | Send BTS picks by private DM | cited by I-017, I-052, I-073 |
| 102 | merged | 2026-05-22 | feat(health): analytics-job status tracking + shadow OOM loop breaker + two-tier WARN attention | cited by I-052 |
| 103 | merged | 2026-05-22 | [codex] Honor LGBM random-state env in live prediction | cited by I-052 |
| 104 | merged | 2026-05-22 | Dispatch shadow predictions out of process | cited by I-052 |
| 105 | merged | 2026-05-23 | [codex] Escalate inline shadow OOM visibility | cited by I-052, I-053 |
| 106 | merged | 2026-05-23 | [codex] Defer unsafe fallback picks | cited by I-053 |
| 107 | merged | 2026-05-23 | [codex] Align shadow unit environment (B.2 pre-enable prep) | cited by I-054 |
| 108 | merged | 2026-05-23 | [codex] Separate DD-pair shortfall from residual correlation (reframe same_team_corr) | cited by I-054 |
| 109 | merged | 2026-05-23 | [codex] Add MDP policy alignment health check | **candidate C1** (diagnostic health check) |
| 110 | merged | 2026-05-23 | [codex] Add calibration re-solve validation gate | validation gate for a policy re-solve (feature) |
| 111 | merged | 2026-05-23 | [codex] Add Gate B raw rebin measurement | cited by I-050 |
| 112 | merged | 2026-05-23 | [codex] Skip stale live-forward archives in resolver scan | cited by I-050 |
| 113 | merged | 2026-05-23 | [codex] Document alert disposition walkthrough | cited by I-052 |
| 114 | merged | 2026-05-23 | [codex] Recalibrate scheduler memory-growth alerts | cited by I-052 |
| 115 | merged | 2026-05-23 | [codex] Add canonical live-forward resolution health check | cited by I-050 |
| 116 | merged | 2026-05-23 | [codex] Handle final no-PA live-forward voids | cited by I-050 |
| 117 | merged | 2026-05-24 | Add fallback defer health visibility | operational feature or visibility; no defect described |
| 118 | merged | 2026-05-24 | Persist health DM delivery status | cited by I-043, I-059 |
| 119 | merged | 2026-05-24 | Fix postponed game candidate filtering | cited by I-043, I-050 |
| 120 | merged | 2026-05-24 | Fix live-forward outcome count backcompat | cited by I-050 |
| 121 | merged | 2026-05-24 | Document probability scale PA-basis mismatch | **candidate C1** |
| 122 | merged | 2026-05-24 | Add Gate B PA-basis rebin screen | research, validation or audit tooling and evidence (no operational stream) |
| 123 | merged | 2026-05-24 | Add Gate B walk-forward policy harness | research, validation or audit tooling and evidence (no operational stream) |
| 124 | merged | 2026-05-24 | Document Gate B estimated-PA walk-forward result | documentation or pre-registration only |
| 125 | merged | 2026-05-24 | Add Gate B fair-comparator rebaseline | research, validation or audit tooling and evidence (no operational stream) |
| 126 | merged | 2026-05-24 | Pre-register Gate B production metric rebaseline | documentation or pre-registration only |
| 127 | merged | 2026-05-24 | Measure Gate B production metric rebaseline | research, validation or audit tooling and evidence (no operational stream) |
| 128 | merged | 2026-05-24 | Inventory production-live boundary feasibility | documentation or pre-registration only |
| 129 | merged | 2026-05-24 | Document health WARN disposition | documentation or pre-registration only |
| 130 | merged | 2026-05-25 | Add feature-env fingerprint provenance | cited by I-043 |
| 131 | merged | 2026-05-26 | Fix postponed-game lock status fallback | cited by I-043, I-050 |
| 132 | merged | 2026-05-26 | Fix live-forward snapshot null provenance drift | cited by I-043, I-050 |
| 133 | merged | 2026-05-27 | docs: refresh alert disposition inventory | documentation or pre-registration only |
| 134 | merged | 2026-05-27 | docs: close postponed alert root fix | documentation closing the postponed-game fix already recorded as I-043 |
| 135 | merged | 2026-05-27 | docs: recast probability scale follow-up | cited by I-052 |
| 136 | merged | 2026-05-27 | fix: broaden restart spike alert wording | cited by I-052, I-055 |
| 137 | merged | 2026-05-27 | fix: keep dashboard health responsive | cited by I-052, I-055 |
| 138 | merged | 2026-05-29 | Drive live picks from contest streak state | cited by I-056 |
| 139 | merged | 2026-05-30 | Document early-lock timing audit and guard regressions | documentation and a regression test for the 5/22 early-game fallback already recorded as I-053 |
| 140 | merged | 2026-05-30 | Document DD guardrail surface gate inventory | documentation or pre-registration only |
| 141 | merged | 2026-05-30 | Evaluate MDP double-down guardrail | research, validation or audit tooling and evidence (no operational stream) |
| 142 | merged | 2026-06-06 | Contest-streak automation (Claude+Codex): auto-fetch real streak, never silently freeze | cited by I-001, I-002, I-003, I-004, I-005, I-006 … |
| 143 | merged | 2026-06-17 | Phase 1: Real-streak anchoring (fix 10-vs-8 inflation) | cited by I-102, I-056, I-063 |
| 145 | merged | 2026-06-22 | Skip-policy shadow + decision.json authoritative record (fixes #144) | cited by I-066, I-069, I-078, I-098 |
