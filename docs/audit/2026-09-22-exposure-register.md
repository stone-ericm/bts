# 2026 Analysis-Exposure Register + Protocol Inventory (season wrap W0.3 / W0.4 / W0.5)

**Created:** 2026-09-22 (plan approved by Eric the same day) · **Plan:** `docs/superpowers/plans/2026-09-14-season-wrap-plan.md` · **Rule:** plan §1.2 — no further 2026 outcome read runs without an entry here that names the candidate, its freeze, the outcomes already examined, the governing protocol and the allowed next look. New protocols written after the fact cannot retro-pre-register an analysis suggested by observed 2026 results.

Every later season-wrap memo cites its row here by **ID**.

## A. Outcome reads already taken on 2026 data (as of 2026-09-22)

| ID | What was read | Freeze / fitting data | 2026 outcomes examined | Governing protocol | Disposition | Allowed next look |
|---|---|---|---|---|---|---|
| X-01 | **Production picks + grading** — the live system itself | Recipe evolves daily (retrain); epochs per plan W1.1 | Every graded date 3/29→9/21 (contest ledger + local slot results) | none (operations) | ongoing | W1.1 ledger may read all of it; it is the fact base, not a test |
| X-02 | **Park-drag 2026 screen** (`docs/audit/2026-07-08-park-drag-2026-screen.md`) | Feature `park_drag_delta`, external as-of table | 2026 dates through ~7/06 (screen + backtest on 2026) | ad-hoc screen, one Codex round | NULL (observability, not alpha); shadow v2 armed | context-shadow closeout only (X-06); no new 2026 fit of park features |
| X-03 | **M3 serving-staleness audit** (`docs/audit/2026-06-11-m3-serving-staleness.md`) | frozen serving lookups vs fresh | 2026 dates through ~6/10 (39 discordant days) | paired fold design, v3 | −0.67pp CI[−3.4,+2.0], HELD | re-run only if rolling slate AUC ≥ ~0.61 (CLAUDE.md M3 trigger) |
| X-04 | **Strategy/model/PIT lever investigation** (`docs/audit/2026-07-06-…`), **skip-threshold + discrimination** (`2026-06-29-…`), **estimated_pa re-solve A/B** (`2026-06-10-…`), **DD-p policy value sensitivity** (`2026-07-13-…`) | estimated_pa 24-seed profiles (2021–25) + 2026 realized picks for context | 2026 realized rank-1 / DD legs through the respective dates | audit docs, Codex rounds | all HOLD / NULL | historical results carried into W1.2 bridge as priors, not re-derived; 2026 slices they touched are marked *consumed* |
| X-05 | **DD tripwire — interim look** (`docs/audit/2026-08-09-dd-tripwire-recompute.md`) | 7/12 measurement recipe (per-slot grading, legacy-single fallback) | DD legs 3/29→8/09 (n=57), primaries n=83 | 7/13 pre-registration (P-01) + 8/09 binding commitment | exploratory only; trigger NOT evaluated | **P-01 formal read at the first snapshot with n≥80 (crossed by 9/11)** — plan W1.4a |
| X-06 | **Context-stack shadow v2 status** (`bts shadow-status`, nightly) | shadow feature set frozen 7/08 | paired days through 9/21 (n=54 resolved at 9/14) | `shadow_eval` status rule (P-03) | prod 55.6% vs shadow 55.6%; no-promote | closeout memo under P-03 — plan W1.4b |
| X-07 | **Skip-policy shadow displays** (dashboard + `skip-policy-shadow-status`) | deployed skip rule (streak ≥8, p<0.796) | 16 divergent days, 15 resolved (through 9/21) | pre-registered checkpoints (P-02) | `insufficient_n`; 86.7% CI[62.1,96.3] displayed | only a P-02 checkpoint may produce a verdict; displays are monitoring |
| X-08 | **Boundary-shadow census** (`docs/audit/2026-08-09-boundary-shadow-census-mechanism.md`) | policy sha 66d15471; current-scale boundaries | 41 mdp decision records through 8/09 (mechanism only; outcomes WITHHELD) | registration r1 `6db921c`; trigger ≥8 intent-diffs fired | outcome phase withheld | follow-up evaluation design must be registered on the prospective `decision_v2` stream; none written yet |
| X-09 | **9/11 read-only scorecard** (memory `bts_index` 2026-09-11) | none (descriptive) | primaries 96/141, current recipe 79/109, DD legs 58/82, mean stated p 0.765 | none | descriptive; became the plan's headline | superseded by W1.1; may not be cited as a test result |
| X-10 | **7/03 me-vs-leaderboard** (memory `bts_index`) | tracked-user corpus through 7/03 | Eric's settled picks 3/29→7/03 vs tracked users | none (post-hoc) | selected-cohort description | W2 product 2 must be defined as an as-of cohort, not from these results |
| X-11 | **Decision-weighted `decision_weighted_lgbm_v0` live-forward stream** | candidate frozen `5004b1c8` (5/10) | pre-outcome slates 5/09→9/18 captured (68/79 decision days; 11 skip days missing); resolve step joins outcomes nightly — **no paired read has been taken** | P-04 | ready-but-unread (n to be counted) | P-04 read only if eligible n ≥ 120; otherwise `insufficient` |
| X-12 | **Research-only forecast captures (D8)** | production recipe, no pick | 9/19, 9/20, 9/21 (and forward through 9/27) — outcomes not yet joined | none; labelled research-only; ineligible for P-04 by construction | collected | W1.2 diagnostic use only; W1.1 joins outcomes offline |
| X-13 | **This plan's own drafting** (9/14) | — | read X-09, X-06, X-07 status lines + decision.json 9/01→9/13 | — | consumed | no candidate may be "pre-registered" whose motivation is these reads unless validated on 2027 forward data (D3) |

**Unexamined 2026 outcome windows** (as of 9/22): (a) D8 research captures 9/19→9/27 (forecast-record only); (b) private doubles 9/15→9/18 have been graded locally by `check-results` but not analysed; (c) 7/04→9/27 public `probabilityStarter` captures have never been scored. Under **D3 = RESERVE**, (a)–(c) may be used only by protocols already frozen before 9/22 (P-01…P-04) or by W1.2's descriptive bridge; any new candidate motivated by them validates in 2027.

## B. Protocol inventory — original texts (quoted; these govern the W1.4 reads)

### P-01 DD tripwire
- **Origin (7/13, `docs/audit/2026-07-13-dd-p-policy-value-sensitivity.md` L133–135):** "Tripwire: at the 7/12 doc's own accumulation checkpoint (~40 more legs, season-to-date n≈80-90), recompute the season-to-date leg gap over pick files (the 7/12 measurement); if it holds ≥10pp, rerun this script and [take the re-solve question seriously]."
- **Binding commitment (8/09, `docs/audit/2026-08-09-dd-tripwire-recompute.md`):** "the formal read happens at the first measurement snapshot with n≥80 season DD legs (~early-mid September at current cadence); the ≥10pp threshold and the escalation action are unchanged regardless of tonight's numbers; the formal read cannot be canceled or delayed; no further interim reads before it. Tonight's tails are descriptive only." Measurement = 7/12 recipe: per-slot grading, `slot_results` authoritative, legacy-single fallback for pre-slot files, exact tails Poisson-binomial.
- **W1.4a obligation:** identify the FIRST snapshot at which n≥80 (by the 7/12 recipe's inclusion rule), compute the gap there, apply ≥10pp; record the escalation disposition. If that snapshot cannot be reconstructed, record the deviation and label any later computation a reconstruction. The season-end update is a separate, labelled read.

### P-02 Skip-policy shadow
- **Code (`src/bts/skip_policy_shadow.py`):** `BREAKEVEN_P = 0.744`; `CHECKPOINTS = (30, 60, 90)`; `Z_CHECKPOINT = 2.394` ("Bonferroni-split alpha (0.05/3 two-sided)"); verdict "evaluated ONLY at these resolved-n checkpoints … computed deterministically from the FIRST c checkpoint-ELIGIBLE records in date order … A decisive look is terminal"; eligibility after `CHECKPOINT_ELIGIBLE_AFTER_DAYS`. Admits reach-57 MDP skips only (`is_reach57_mdp_skip`).
- **State 9/22:** 16 divergent days, 15 resolved → below the first checkpoint (30). **Season-end verdict = `insufficient_n`** unless resolved eligible n reaches 30 by 9/27 (it cannot: tail skips are excluded). Full alternative-trajectory replay is a separate, labelled analysis, not this protocol.

### P-03 Context-stack shadow v2
- **Code (`src/bts/shadow_eval.py`):** `SHADOW_STATUS_DEFAULT_MIN_DAYS = 30`; paired-day evaluation with Wilson intervals and a two-sided sign test on discordant days (`_sign_test_p_two_sided(prod_only, shadow_only)`); reconciliation via `check-results` on every exit path (2026-08-09 hardening).
- **W1.4b obligation:** report agreement, discordant outcomes, coverage and interval on the frozen eligibility; equal aggregates ≠ equivalence; no-promote stands unless the protocol says otherwise.

### P-04 Decision-weighted candidate (tracker #16)
- **Pre-registration (`docs/sota_audit/2026-05-08-fresh-audit-pre-registration.md` L297–309):** "1. At least `120` eligible resolved slate dates unless this floor is amended … 2. Positive point delta in `p_57_mdp`. 3. A one-sided candidate-better-than-production date-paired block-bootstrap …"; "Do not inspect paired `p_57_mdp` deltas, bootstrap intervals, or guardrail [metrics before the read]". Official verification requires production-pick parity (`production_pick_snapshot_required`, `--require-production-pick-snapshot`).
- **W1.4b obligation:** count eligible resolved dates from the official root only (research captures excluded); if <120 → `insufficient`; interpret `p_57_mdp` against the 7/13 evaluator findings (iid milestone values policy-dependent-wrong) and label any amended analysis.

### P-05 Tail policy (E[season-best])
- **Design (`docs/audit/2026-09-03-emax-tail-policy.md` §3):** "Stop rule, explicit: skip iff `min(57, s + 2d) <= m`"; first-day acceptance = decision.json `objective=emax_season_best`, `best_status=trusted`, `effective_best=18`, `degraded_reason=null`.
- **W1.4b obligation:** audit objective/state/provenance per date 9/03→9/27, stop behaviour on 9/19, delivered/entered/private separation. A few dates validate the mechanism, not the rates.

### P-06 Boundary-shadow follow-up (X-08)
- Registration r1 `6db921c` fired its ≥8 trigger; the outcome phase is **withheld** until an evaluation design is registered on the prospective `bts_daily_decision_v2` stream. No such design exists; writing one now would be motivated by consumed 2026 outcomes → under D3 it validates in 2027.

## C. Owner decisions recorded (plan W0.5)

| # | Decision | Recorded | Basis |
|---|---|---|---|
| D3 | 2026 data use | **RESERVE** — unread late-season outcomes stay with P-01…P-04; every new idea from this wrap validates prospectively in 2027 | Eric "not sure" 9/14 → methodology default adopted, reversible |
| D5 | Final-grab profile allocation | **SPLIT 150 / 150** — 150 from the top of the final `all_season` board, 150 from users present in the 2026-05-01 first snapshot (as-of cohort) | Eric "not sure" 9/14 → methodology default adopted, reversible before the grab |
| D8 | Research capture past the tail stop | **CONTINUE, separately labelled** — built + deployed 9/14, verified 9/19–9/21 | Eric 9/14 |
| — | Plan as a whole | **APPROVED** | Eric 9/22 "looks good to me" |

## D. Preservation evidence (plan W0.1 / W0.2a) — see `docs/audit/2026-09-22-w0-preservation.md`
