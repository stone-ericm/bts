## Verdicts

| Item | Verdict | Required disposition |
|---|---|---|
| A — W2.1 | **SIGN WITH EDITS** | Keep the census and survivor case series. Condition the prize inference, clarify that exact run dates are unavailable for all members, and complete the required declarations. |
| B — W2.2 | **SIGN WITH EDITS** | Attach the identity and acceptance conditions locally, including to our shared-date column and the difference; correct the streak-status attribution and complete support/coverage reporting. |
| C — #87 | **SIGN WITH EDITS** | Report the actual five-condition statuses, distinguish the empty testable family from the nonempty cell inventory, and correct the diagnostic's date-slot denominator. |

These are bounded memo edits, supplied verbatim below. No rerun, new acquisition, code change, threshold change or third review round is needed. Preserve the owner's maximum of two result-review rounds, then freeze with these limits.

Reviewed `main` at `8296b4311b884e7de4556bf08eb7d2fa8f3273c6`. Evidence: the three memos, frozen designs/protocols, both code-review rounds for each package, exposure-register rows, relevant implementation/history, and the three authorized copied JSON files. Read-only Python made 87 numerical/arithmetic assertions, all passing; it also independently counted the cell-level condition statuses and sensitivity support. That verifies transcription and arithmetic against the registered outputs, not the underlying observations or bootstrap execution. No `data/` reads, outside-checkout reads, SSH, network, `gh`, tracked-file edits, commit or push occurred.

The copied mining report's SHA256 is `7677719e8a0202a23eb3a8a7dc3b9530ad1be64ea05a23087b61535d2de5fc7a`, matching `COMPLETE.outputs["report.json"]`. Run ID, mode, code, X-22 commit, input-manifest hash and registration fingerprint agree between the copies. The current original protocol, amendment and archived mining R1 review match their run document pins. Both exposure commits are ancestors of HEAD; comparing the frozen package revisions with HEAD shows only the subsequent exposure-gate activation edits. A completion receipt remains the runner's attestation; no underlying source files or operational log were reread here.

During final verification HEAD advanced to `de7311b` with unrelated W1.3/decision-memo documentation. Git comparison verified that the three reviewed memos, frozen field design, mining protocols, exposure register and both packages were unchanged across that advance. The tracked worktree remained clean.

## A findings

**A1 — Medium: the prize sentence concludes more than the census establishes.** `docs/sota_audit/2026-10-04-field-final-leaders.md:34`. “Would go to that entrant” omits eligibility, the no-Grand-Prize condition, and the official close-of-entry-period determination. The preceding listing-not-award warning does not make that unconditional assignment valid. The checkout's W3 rules evidence (`docs/sota_audit/2026-10-04-literature-refresh-evidence/verified-rules.md`) supplies those conditions, but no copied result establishes that entrant's eligibility or an award. Use A-E1.

**A2 — Low: 144 is a result-status count, not the full count with unavailable exact dates.** Memo `:44–46`; `scripts/audit/field_products/streaks.py:87–108`. The output partitions the 150 users into 144 `dates_unavailable` and 6 `no_settled_round_reports_best`. Neither group has an exact reconstructed start/end date. State all 150 as unavailable and retain the two statuses without implying six recoverable runs. Explicitly restrict positive lower bounds to qualified complete all-hit rounds. Use A-E2.

**A3 — Medium: required declarations are incomplete; one provenance count is not in the review copies.** Memo `:5,49–54`; field code R2's “X-24/X-25 declarations.” X-14 and X-15 are not disclosed in the memo; incomplete-round streak increments, omitted-later observations and calendar-date semantics are also unstated. The raw slot-set witness must be distinguished from complete entered-round history. The quoted **1,366 source files** is absent from `field_results.json`; the freeze/manifest/log was not supplied. The runner stages sources before outcomes (`run.py:258–273`), but code inspection cannot verify that particular real-run count. Delete that count rather than seeking a new read. Use A-E3 and A-E4.

**Numerical reconciliation:** all substantive census and case-series numbers match the copy. Summing the distribution independently gives N=121,852, ≥20=2,186, ≥30=49, ≥40=0, maximum 39 with one entrant, and listing floor 1. There are no missing bests or qualification exclusions; every recorded census check is true. Own ID 50311, best 18 and rank 3,027 match; below/equal/above are 117,273/1,553/3,026 and independently yield [96.2421626, 97.5166596], correctly rounded to [96.2, 97.5]. A has 150 usable/raw-verified profiles, 18,168 witnessed rounds, 32,042 slots, 5 incomplete rounds and 18,163 complete rounds; 13,874/18,163 rounds is 76.3861%, correctly rounded to 76.4%. The 22,253/8,927/862 label counts and all-32,042 unknown-context count match. Revision/conflict/deleted-leg counters are zero.

The survivor-selected paragraph covers the entire A section and is adequate: these quantities describe the final-board-selected case series, not representative field behaviour or a copying/DD/skip effect. The census receives no sampling interval. The own percentile is an identification interval arising from ties, separate from stored rank. The historical C-01 and C-04/X-17 statements are cited documentary evidence, not remeasurement in this review.

## B findings

**B1 — Medium: the opening declaration is not a substitute for the required local conditions.** `docs/sota_audit/2026-10-04-field-cohort-comparison.md:8–10,18–25,29,36–43`; field code R2's declarations. The table labels only E as conditional on name attribution; “Ours” and “Ours − E” are unqualified. Our shared-date quantities also depend on the E-attributed shared-date selection, and production quantities depend on unchanged accepted bytes. Availability/activity and exclusions likewise need a conditional caption/label. The opening “attached to every E-attributed number” additionally sweeps in the extension, whose verified-ID attribution is expressly different. Keep the required paragraphs and identify exactly where each applies. Use B-E1 and B-E2.

**B2 — Medium: unmatched filenames are not verified non-E people.** Memo `:17` says 1,692 files “belong to users outside E.” `w22.identity.files_not_E` counts files not assigned by the name/filename binding rule; the disclosed lack of batch-ID evidence prevents converting that into a verified account-membership statement. Replace with “not assigned to E by this binding rule,” as in B-E2. Initial binding to 308 members precedes the additional 21 ownership-signal quarantines; make that sequence explicit.

**B3 — Low: the streak status is overgeneralized, and support details are omitted.** Memo `:39,44,48–60`. `w22.primary.window_streaks.status` contains 187 `no_complete_rounds` and 39 `no_window_rounds`, not 310 `no_complete_rounds`; the other 84 have no assessable daily history. Exact metrics remain unavailable for all. The shared E table omits its 2,795-round count; the extension omits 6,659 rounds and zero no-history responses, and states only a generic history-depth limit despite the registered depth inventory. Give those coverage details and point to the per-user artifact. Preserve the required Pass, omitted-later, completeness and context declarations. Use B-E2 and B-E3.

**Numerical reconciliation:** the manifest has 310 distinct IDs and SHA prefix `92f34179`; all recorded manifest/allocation checks pass. Initial bindings are 349 files/308 members, with 2 manifest-collision and 21 settled-identity-signal quarantines. Availability is 226 usable, 61 empty and 23 quarantined; activity is 187 observed, 39 none observed/unknown and 84 not assessable. Exclusions 247 void + 2 unlabelled + 3 identity-unresolved reconcile 6,895 to 6,643 usable slots. Our 72 slots comprise 44 primary/28 DD and 1 evidenced/71 inferred links; 18 not uniquely confirmed + 2 unconfirmed reconcile 92 selection rows to 72 included.

The all-date union has 64 dates: E 4,411/6,643=66.4%, [63.1,69.7], 187 users/3,690 rounds/63 contributing dates; ours 53/72=73.6%, [60.7,85.7], one user/46 rounds/46 dates. Shared support is 45 dates: E 3,349/5,047=66.4%, [62.7,70.0], 186 users/2,795 rounds; ours 52/71=73.2%, [60.3,85.5], one user/45 rounds. The unrounded ratio difference is 6.8831854 pp, correctly reported as +6.9 pp [−6.4,+19.6]. All copied interval records use 10,000 draws, seed 20261004 and zero failed draws. The extension has 194/310=62.6% usable histories, 116 budget omissions, zero fetch/parse/hash failures or no-history responses, and 44/44 usable E_in_A. Its pooled 8,517/12,292=69.3%, 150 contributing users, 83 dates and 6,659 rounds agree with the copy.

“Difference includes zero” is correct for the conditional descriptive interval. It does not establish equivalence, identical skill, absence of a practically relevant difference or a causal effect. The existing sentence refuses those conclusions and can stand once the conditions are local. The extension stays separate and partly outcome-determined; its verified final-grab ID attribution must not inherit the primary name assumption.

## C findings

**C1 — Medium: “every condition c1–c5 has empty support” is false for the reported cell inventory.** `docs/sota_audit/2026-10-04-mechanism-mining-result.md:35`; `scripts/audit/mining87/inference.py:195–239`. The empty dictionaries in `summary.fixed_testable_condition_status` describe only the empty subset of **testable** fixed cells. Each of the 48 observed fixed cells nevertheless has all five conditions evaluated. Independent counts from `streams.primary.cells` are:

| Condition | Pass | Fail | Unknown / unavailable |
|---|---:|---:|---|
| c1: ≥30 resolved disagreements | 0 | 48 | 0 |
| c2: lift ≥0.05 | 12 | 28 | 8 unknown |
| c3: BH q≤0.10 | 0 | 0 | 48 not testable |
| c4: all-tracked direction not contradictory | 26 | 10 | 12 unknown |
| c5: evidenced lock-available mechanism | 0 | 0 | 48 unknown |

All-tracked cells are comparisons, not nomination candidates. Sparse point lifts/directions remain descriptive; none rescues c1/c3/c5. This matters for A4's requirement to report the five conditions separately. Use C-E1, preserving coverage → primary/fallback → cell inventory/conditions/nomination → secondary reads.

**C2 — Medium: the unproven diagnostic is slot-weighted, not date-weighted.** Memo `:62`; `scripts/audit/mining87/report.py:171–184`. The denominator is the number of eligible resolved production **date-slots**, including DD, because the function uses `len(g)` without collapsing dates. The output's 15 is therefore not 15 distinct dates. The surface inventory separately records only 14 selection-consistent dates. The reported proportions are correctly rounded: rank-1 2/15=13.3% and top-10 7/15=46.7%. Keep the diagnostic outside primary top-N, decomposition/FDR and nomination, and correct its unit. Use C-E3; no new distinct-date calculation is required.

**C3 — Low: an empty candidate-comparison list does not mean the sensitivity has nothing to compare.** Memo `:39`. Both inventories have 94 cells and zero testable cells, but primary support sums to 145 units per cohort and sensitivity to 144 per cohort. One flagged unit is removed in each arm. There are no primary c1–c4 candidates, so `sensitivity_comparison=[]`; that is the appropriate scope of the empty comparison statement. Use C-E1.

**C4 — Low: state the concentration population and execution evidence precisely.** Memo `:9,61`; `report.py:156–165`; `run.py:330–335,435`. Concentration includes every public-consensus date-slot in the window, whether production-matched or not: 100 slot-1 and 100 slot-2 consensuses, versus 82/63 locked production units. The memo should make this different support explicit. Also stdout contains a JSON object with the final run directory, completion flag and mode, not literally only the directory. The supplied copies do not contain stdout; describe source behaviour and receipt attestation without implying a log was inspected. Use C-E2 and C-E3.

**Numerical reconciliation and claim limits:** all remaining quoted numbers match the registered report/receipt. Production 154 selections =145 locked (82/63) +9 unsupported; settlement 122 resolved +23 unknown; non-unit day rows 6/3/3. Public inventory 2,042 files, 102 empty, 2,809,077 rows, 27,139 outside-window, zero after-cutoff, 184,001 user-slot observations and 99 invalid IDs matches. The capture range is May 1–July 4. Fixed membership 22,800 tab rows/21,710 names, SHA prefix `0ce1968c`, 778 matching pick-file stems and zero observed sanitization collisions matches. Both cohorts select consensus for all 145 retained units; settlement is 142/1/2 fixed and 143/1/1 all-tracked. Surface coverage is 91 considered/0 admitted, with reasons 23 no witness/68 no slate. Agreement is 6/122=4.9% in each cohort, split 4/70 and 2/52.

Both streams have 94 nonempty cells (48 fixed/46 all-tracked), zero testable cells, null p/q throughout and zero statistical candidates/nominations. The maximum resolved-disagreement cell support is 13 fixed and 12 all-tracked. Forty fixed cells have positive disagreement support below 15; the other eight have zero resolved disagreements. The four paired rows reproduce exactly at the memo's precision: units/dates 119/72, 113/70, 120/73, 114/71; production 70.6/69.0/70.8/69.3%; consensus 74.8/73.5/76.7/75.4%; deltas +4.2/+4.4/+5.8/+6.1 pp with intervals [−5.4,+12.8], [−5.7,+13.4], [−5.0,+15.2], [−5.2,+15.8]. Bootstrap metadata is block 7/2,000 draws/seed 20260510 over ordered observed dates. All intervals include zero. Concentration medians 20.1553%/16.4666% and public-user medians 488/377.5 support the current rounded figures, subject to the support clarification.

The “no actionable mechanism” statement is **permitted by amended A4/X-22 with the stated information limits**. It is not a measured negative effect or disproof of useful leaderboard signal. The memo correctly distinguishes unresolved production, unresolved consensus, absent admitted surfaces and sparse cells, and explicitly notes that absent frozen mechanism records made nomination impossible independently of outcomes. Retain those qualifications. Public-log paired outcomes support no copying or pre-lock-availability claim; the current warning is appropriate. Missing ranked surfaces remain unavailable, not off-top-N failures. The context bins match A5. No candidate is introduced by these edits.

## Verbatim edits

Apply only the following memo edits; the reviewed numerical values otherwise stand.

### A-E1 — Replace A's Top Streak paragraph

> **The Top Streak prize, for D1:** the retained board had one entrant at 39, above the $10,000 Top Streak prize's floor of 20. Under the W3 rules evidence, the highest eligible streak at the close of the entry period wins that prize if no Grand Prize is awarded; eligible ties split it. This capture does not establish the entrant's eligibility, the official closing standings or an awarded prize. Our 18 was below the 20 floor.

### A-E2 — Replace A's Runs bullet and its sub-bullets

> - **Runs:** board season best is kept separate from run reconstruction. Exact run start/end dates and exact reconstructed streak maxima are unavailable for all 150 users because complete entered-round history is not witnessed. The output records 144 users with qualifying observed attainments but unavailable dates, and 6 with no qualifying settled all-hit round reporting their board best. Only observed-segment lower bounds from qualified complete all-hit rounds are reported in `w21_runs_A.parquet`; incomplete rounds contribute no claimed winning-round increment or complete-DD denominator.

### A-E3 — Replace A's Run line

> **Run:** `data/validation/w21_w22_field/327fcd6-20261004T171153Z/` (gate pin `327fcd6`), copied to `data/hetzner_results/season_wrap_outputs/w21_w22/`.

### A-E4 — Add these bullets to A's Limits

> - **Witness scope:** a verified raw final-grab response establishes its observed round slot set, not complete entered-round history or historical pick-time context. Daily-corpus DD frequency is unavailable without a completeness witness.
> - **Missing observations:** omitted-later rounds remain flagged positive historical observations; omission proves neither deletion nor complete follow-up. Unobserved calendar dates include dates without contest opportunity and are neither skips nor opportunity/activity denominators.
> - **Disclosed overlaps:** X-14; X-15/X-17 covered capture/equality only. C-01/C-04 apply as cited above; historical final-board equality is cited, not remeasured here.

### B-E1 — Replace B's opening blockquote with these declarations and qualification labels

> **I — primary daily corpus only.** Protocol deviation: the design's batch-specific stable-ID binding has not been established for the username-keyed daily corpus. For the fixed May 1-July 3 primary follow-up, candidate files are assigned using each frozen May 1 member's recorded name and filename aliases, under `stable_5_01_username_unwitnessed`. We assume every included appended batch in those files belongs to that member, without same-name/sanitized-name substitution by another account. This assumption is unverified. Manifest/visible-filename collisions and observed settled-identity changes trigger conservative quarantine; their absence does not establish identity. Partial identity witnesses may exist, but their coverage was not established for this analysis. All primary E-attributed quantities are conditional on this assumption.
>
> **A — acceptance bytes.** The production denominator is read from frozen current files in the named W1.1 accepted-build directory. The reader validates acceptance metadata, published build identity, compiler schemas and selected marginal counts. It does not independently verify equality of outcome bytes to those accepted earlier. Production rates and comparisons are therefore conditional on those current files remaining unchanged from the accepted build; a current input hash is run provenance, not proof of earlier acceptance-byte identity.

**Qualification labels below:** I denotes the unverified primary name-attribution assumption above; A denotes the unchanged-accepted-bytes assumption above. Frozen manifest membership/allocation counts are separately established. The final-backfill extension uses verified final-grab IDs and does not use I. These bootstraps do not quantify attribution or acceptance uncertainty.

### B-E2 — Replace B's sections 1 and 2

```markdown
## 1. The cohort and its availability

- **E membership:** the 310 distinct user ids in the frozen May 1 four-tab manifest (sha256 `92f34179…`). Every manifest, allocation and acquisition-record check passed. A, B, E_in_A and E_unfetched are acquisition labels, not cohorts.
- **Initial daily binding — conditional on I:** 349 files were initially assigned to 308 members by the name/filename rule; 2 members were quarantined for a manifest collision. An additional 21 members were quarantined for observed settled-identity changes. Another 1,692 daily files were not assigned to E by this binding rule; that is not verified non-E account identity.
- **Daily-history availability — conditional on I; all 310 retained:** 226 have usable daily history, 61 have empty files and 23 are quarantined.
- **Window activity — conditional on I:** 187 have observed graded slots in May 1-July 3; 39 have no observed window activity (unknown, not stopped or skipped); 84 are not assessable. Per-user availability, graded-slot counts and hit rates are in `w22_availability.parquet`, under the same I qualification.

## 2. The comparison, May 1-July 3

**Graded slot** means exact hit / not_hit; every other label is excluded and counted. `void` is a settled Pass under the W1.1 HOLD normalization, never a graded hit or miss.

- **E exclusions — conditional on I:** 247 void, 2 unlabelled and 3 identity-unresolved slots were excluded, leaving 6,643 usable of 6,895.
- **Our denominator — conditional on A:** 72 slots, comprising 44 primary and 28 double-down legs. All are committed-evidenced, confirmed and uniquely linked with an exact contest grade: 1 evidenced link and 71 inferred. Inferred game identity remains inferred. Of the window's 92 selection rows, 20 were excluded: 18 not uniquely confirmed and 2 unconfirmed.
- **Weighting:** pooled ratios weight prolific users and double-down days more; they are not mean-user skill estimates.

**Table qualification:** E-attributed counts/rates are conditional on I; production counts/rates are conditional on A. The joint date supports and bootstrap intervals, and our E-defined shared-date quantities and comparison difference, are conditional on both I and A. Brackets are conditional 95% intervals, not bounds on attribution or acceptance uncertainty.

| Support (conditional on I and A) | E (conditional on I) | Ours (conditional on A; shared subset also I) | Ours − E (conditional on I and A) |
|---|---|---|---|
| All-observed-date union: 64 dates | 4,411 / 6,643 = **66.4%** [63.1, 69.7]; 187 users, 3,690 rounds, 63 dates with slots | 53 / 72 = **73.6%** [60.7, 85.7]; 1 user, 46 rounds, 46 dates | — |
| Shared: 45 dates, with ≥1 usable graded slot in each arm | 3,349 / 5,047 = **66.4%** [62.7, 70.0]; 186 users, 2,795 rounds, 45 dates | 52 / 71 = **73.2%** [60.3, 85.5]; 1 user, 45 rounds, 45 dates | **+6.9 pp [−6.4, +19.6]** |

- **Conditional intervals (I and A):** 95% from 10,000 joint whole-date draws at seed 20261004, retaining repeated draws, with no failed draws. They assume exchangeable dates and exclude cross-date dependence, observation selection, missing histories and attribution or acceptance uncertainty.
- **Reading:** the conditional difference interval includes zero. This is not a skill ranking, an equivalence or a causal comparison.
- **Streak availability — conditional on I:** 187 members have `no_complete_rounds`, 39 have `no_window_rounds`, and 84 have no assessable daily history. Exact within-window maxima and attaining-run start/end dates are unavailable for all members. Daily positive streak bounds and DD frequency are unavailable without a completeness witness; incomplete rounds contribute no claimed winning-round increment or complete-DD denominator.
```

### B-E3 — Replace B's section 3, and add the two Limits bullets below

```markdown
## 3. Final-backfill extension, July 4-September 27 (separate; never appended to the primary)

- **Support and attribution:** E∩(A∪B) histories from the September 27 grab, attributed by verified final-grab user id. These quantities do not use the primary name-attribution assumption I.
- **Coverage:** 194 of 310 E members (62.6%) have usable histories; 116 were budget omissions. There were zero fetch/parse/hash failures and zero no-history responses. All 44 E_in_A are usable.
- **History depth:** the registered per-user inventory is in `results.json` under `w22.extension.coverage.history_depth`. Across the 194 usable histories, recorded first pick dates range from March 25 to April 17, last pick dates from April 11 to September 27, and observed round counts from 16 to 183. These are API-returned histories, not proof of complete follow-up.
- **Pooled:** 8,517 / 12,292 = 69.3% over 83 dates, 150 contributing users and 6,659 rounds.
- **No interval is reported.** Support is partly determined by final outcomes through allocation A/B; it does not represent an unselected continuation of E or the field.
```

Add to **Limits**:

> - **Witness scope and context:** a verified raw final-grab response establishes its observed round slot set, not complete entered-round history or historical pick-time context. Historical team/home-away composition is unknown.
> - **Omitted-later rounds:** retained, flagged positive historical observations; omission proves neither deletion nor complete follow-up.

### C-E1 — Replace C's section 3

```markdown
## 3. Decomposition cells, FDR and the five nomination conditions

- **Primary stream:** all 94 observed nonempty cross-product cells are retained: 48 fixed-cohort and 46 all-tracked. Every cell and its counts/conditions is in `cells_primary.parquet` and `report.json` under `streams.primary.cells`.
- **Testability:** none reaches 15 resolved disagreements; the maximum cell support is 13 fixed-cohort and 12 all-tracked. The realized testable BH/BY family is empty. No cell has an available p/q; there are no statistical candidates or nominations. Forty fixed cells have positive but sparse disagreement support; eight have zero resolved disagreements.
- **Five conditions, separately, over all 48 fixed-cohort cells:** sparse point effects and directions remain descriptive. All-tracked cells supply comparisons and are not nomination candidates.

| Condition | Reported status |
|---|---|
| c1: at least 30 resolved disagreements | 48 fail |
| c2: consensus-minus-production lift at least 0.05 | 12 pass, 28 fail, 8 unknown |
| c3: BH q ≤ 0.10 | 48 not testable; q unavailable |
| c4: all-tracked direction not contradictory | 26 pass, 10 fail, 12 unknown |
| c5: stated mechanism with evidenced lock-available variables | 48 unknown: no mechanism records supplied |

- **Tie-excluded sensitivity:** one flagged unit is removed per cohort, leaving 144 units per cohort. The inventory still has 94 nonempty cells and zero testable cells. The five-condition status counts above are unchanged. No primary cell passes c1-c4, so the candidate sensitivity-comparison list is empty; this is not a claim that the two support inventories are identical.
- **Outcome state:** `power_limited_no_testable_cells`.
- **Nomination:** none. **No cell met all five conditions, so there is no actionable mechanism for this cohort and window under the stated information limits.** Independently, condition 5 was unavailable by the frozen input choice, so this execution could not have nominated even with stronger outcome support.
- **Information limits:** all 145 fixed units lack an admitted surface; 23 have unresolved production settlement; 3 have unresolved consensus settlement; 40 fixed cells have sparse positive disagreement support. These are separate limits, not measured negative effects.

**This does not disprove signal** in a larger or better-instrumented sample. It is a power and instrumentation limit. Sparse descriptive lift/direction passages do not establish a mechanism.
```

### C-E2 — Replace C's Inputs and Pins/outputs bullets

> - **Inputs:** the copied completion receipt attests that consumed inputs matched the manifest (`f56917bc…`).
> - **Pins and outputs:** the runner writes execution pins before outcome-bearing loaders and emits the final run directory, completion flag and mode on stdout.

### C-E3 — Replace C's concentration and unproven-diagnostic bullets

> - **Consensus concentration:** this summary includes every public-consensus date-slot in the registered window, whether production-matched or not. The fixed cohort has 100 slot-1 and 100 slot-2 consensuses. Median selected-batter vote share is 20.2% for slot 1 (median 488 public users) and 16.5% for the legal slot-2 choice (median 377.5 public users). Concentration may indicate a publicly obvious batter class or a stale blind spot; it is not a success metric.
> - **Unproven served-slate diagnostic:** outside the registered primary, decomposition/FDR family and nomination stream, on 15 resolved production date-slots whose dates are selection-consistent but failed admission, the fixed-cohort consensus batter appeared at served rank 1 in 2/15 slots (13.3%) and in the top 10 in 7/15 (46.7%). These are slot-weighted diagnostics, not proportions of distinct dates or admitted top-N coverage.
