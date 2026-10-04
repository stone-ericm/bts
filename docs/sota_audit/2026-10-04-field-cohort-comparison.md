# W2.2 As-of cohort comparison: early manifest E against our production

**Date:** 2026-10-04. **Status:** FROZEN 2026-10-04 after Codex memo r1 SIGN WITH EDITS (B-E1–B-E3 applied verbatim by script; review archived at `docs/audit/2026-10-04-field-87-memos-codex-r1.md`). Pace rule: no further review rounds.
**Design:** `docs/superpowers/specs/2026-10-04-field-products-design.md` rev 2, FROZEN. **Code:** `scripts/audit/field_products/` at `793e987`, FROZEN. **Exposure:** X-25 (`b972cd9`).
**Run:** `data/validation/w21_w22_field/327fcd6-20261004T171153Z/` (shared with W2.1).
**Use:** descriptive only, under D3 = RESERVE.

> **I — primary daily corpus only.** Protocol deviation: the design's batch-specific stable-ID binding has not been established for the username-keyed daily corpus. For the fixed May 1-July 3 primary follow-up, candidate files are assigned using each frozen May 1 member's recorded name and filename aliases, under `stable_5_01_username_unwitnessed`. We assume every included appended batch in those files belongs to that member, without same-name/sanitized-name substitution by another account. This assumption is unverified. Manifest/visible-filename collisions and observed settled-identity changes trigger conservative quarantine; their absence does not establish identity. Partial identity witnesses may exist, but their coverage was not established for this analysis. All primary E-attributed quantities are conditional on this assumption.
>
> **A — acceptance bytes.** The production denominator is read from frozen current files in the named W1.1 accepted-build directory. The reader validates acceptance metadata, published build identity, compiler schemas and selected marginal counts. It does not independently verify equality of outcome bytes to those accepted earlier. Production rates and comparisons are therefore conditional on those current files remaining unchanged from the accepted build; a current input hash is run provenance, not proof of earlier acceptance-byte identity.

**Qualification labels below:** I denotes the unverified primary name-attribution assumption above; A denotes the unchanged-accepted-bytes assumption above. Frozen manifest membership/allocation counts are separately established. The final-backfill extension uses verified final-grab IDs and does not use I. These bootstraps do not quantify attribution or acceptance uncertainty.

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

## 3. Final-backfill extension, July 4-September 27 (separate; never appended to the primary)

- **Support and attribution:** E∩(A∪B) histories from the September 27 grab, attributed by verified final-grab user id. These quantities do not use the primary name-attribution assumption I.
- **Coverage:** 194 of 310 E members (62.6%) have usable histories; 116 were budget omissions. There were zero fetch/parse/hash failures and zero no-history responses. All 44 E_in_A are usable.
- **History depth:** the registered per-user inventory is in `results.json` under `w22.extension.coverage.history_depth`. Across the 194 usable histories, recorded first pick dates range from March 25 to April 17, last pick dates from April 11 to September 27, and observed round counts from 16 to 183. These are API-returned histories, not proof of complete follow-up.
- **Pooled:** 8,517 / 12,292 = 69.3% over 83 dates, 150 contributing users and 6,659 rounds.
- **No interval is reported.** Support is partly determined by final outcomes through allocation A/B; it does not represent an unselected continuation of E or the field.

## Limits
- **The attribution and acceptance-byte assumptions above.**
- **Pick logs** are observations, not proof of pre-lock timing.
- **Unobserved calendar dates** include dates without contest opportunity; they are neither skips nor activity denominators.
- **The extension's history depth** is whatever the end-of-season API returned. A profile with any pending slot made the grab fail closed.
- **Disclosed overlaps:** X-10 (the 7/03 me-vs-leaderboard read of this corpus) and X-01/X-09; X-19 covered counts only.
- **Descriptive only.** No causal effect, skill rank or equivalence; nothing is nominated.
- **Witness scope and context:** a verified raw final-grab response establishes its observed round slot set, not complete entered-round history or historical pick-time context. Historical team/home-away composition is unknown.
- **Omitted-later rounds:** retained, flagged positive historical observations; omission proves neither deletion nor complete follow-up.
