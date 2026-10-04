# W2.2 As-of cohort comparison: early manifest E against our production

**Date:** 2026-10-04. **Status:** draft for Codex review (analysis deliverable: at most two rounds, then freeze with the limits stated).
**Design:** `docs/superpowers/specs/2026-10-04-field-products-design.md` rev 2, FROZEN. **Code:** `scripts/audit/field_products/` at `793e987`, FROZEN. **Exposure:** X-25 (`b972cd9`).
**Run:** `data/validation/w21_w22_field/327fcd6-20261004T171153Z/` (shared with W2.1).
**Use:** descriptive only, under D3 = RESERVE.

> **Protocol deviation, attached to every E-attributed number below.** The design's batch-specific stable-ID binding has not been established for the username-keyed daily corpus. For the fixed May 1–July 3 primary follow-up, candidate files are assigned using each frozen May 1 member's recorded name and filename aliases, under `stable_5_01_username_unwitnessed`. We assume every included appended batch in those files belongs to that member, without same-name or sanitized-name substitution by another account. This assumption is unverified. Manifest/visible-filename collisions and observed settled-identity changes trigger conservative quarantine; their absence does not establish identity. Partial identity witnesses may exist, but their coverage was not established for this analysis. All primary E-attributed quantities are conditional on this assumption.
>
> **Acceptance bytes.** The production denominator is read from frozen current files in the named W1.1 accepted-build directory. The reader validates acceptance metadata, published build identity, compiler schemas and selected marginal counts. It does not independently verify equality of outcome bytes to those accepted earlier. Production rates and comparisons are therefore conditional on those current files remaining unchanged from the accepted build; a current input hash is run provenance, not proof of earlier acceptance-byte identity.

## 1. The cohort and its availability
- **E:** the 310 distinct user ids in the frozen May 1 four-tab manifest (sha256 `92f34179…`). Every manifest, allocation and acquisition-record check passed. A, B, E_in_A and E_unfetched are acquisition labels, not cohorts.
- **Binding** (conditional on the deviation above):
  - 349 daily files are bound to 308 members;
  - 2 members are quarantined for a manifest collision, and 21 for an observed settled-identity change;
  - 1,692 daily files belong to users outside E.
- **Availability, all 310 kept:**
  - 226 have usable daily history;
  - 61 have empty files;
  - 23 are quarantined.
- **Window activity:**
  - 187 have observed graded slots in 5/01–7/03;
  - 39 have no observed activity (unknown, not "stopped" or "skipped");
  - 84 are not assessable.

## 2. The comparison, May 1 – July 3
**Graded slot** means exact hit / not_hit; every other label is excluded and counted.
- **E:** 247 void, 2 unlabelled and 3 identity-unresolved slots were excluded, leaving 6,643 usable of 6,895.
- **Ours:**
  - 72 slots: 44 primary and 28 double-down legs.
  - All are committed-evidenced, confirmed and uniquely linked with an exact contest grade: 1 evidenced link and 71 inferred. The inferred game identity remains inferred.
  - 20 window selection rows were excluded: 18 not uniquely confirmed, 2 unconfirmed.
- **Weighting:** pooled ratios weight prolific users and double-down days more; they are not mean-user skill estimates.

| Table | E (conditional on the attribution assumption) | Ours | Ours − E |
|---|---|---|---|
| All observed dates (64) | 4,411 / 6,643 = **66.4%** [63.1, 69.7]; 187 users, 3,690 rounds, 63 dates with slots | 53 / 72 = **73.6%** [60.7, 85.7]; 46 dates | — |
| Shared dates (45, with ≥1 usable graded slot in each arm) | 3,349 / 5,047 = **66.4%** [62.7, 70.0]; 186 users | 52 / 71 = **73.2%** [60.3, 85.5] | **+6.9 pp [−6.4, +19.6]** |

- **Intervals:** 95% from 10,000 joint whole-date draws at seed 20261004, with no failed draws.
- **Assumptions:** they assume exchangeable dates and exclude cross-date dependence, observation selection, missing histories and attribution or acceptance uncertainty.
- **Reading:** the difference interval includes zero. This is not a skill ranking, an equivalence or a causal comparison.
- **Streaks:** within-window streak summaries are unavailable for every member (`no_complete_rounds`), because daily rounds have no completeness witness. Daily double-down frequency is unavailable for the same reason.

## 3. Final-backfill extension, July 4 – September 27 (separate; never appended to the primary)
- **Support:** E∩(A∪B) histories from the 9/27 grab, attributed by verified final-grab user id rather than by the primary's name assumption.
- **Coverage:**
  - 194 of 310 E members (62.6%) have usable histories;
  - 116 were budget omissions;
  - there were no fetch or parse failures;
  - all 44 E_in_A are usable.
- **Pooled:** 8,517 / 12,292 = 69.3% over 83 dates, 150 users with slots in the window.
- **No interval is reported, and the support is partly outcome-determined.** Allocation A came from final rank, which makes it an unselected continuation of neither E nor the field.

## Limits
- **The attribution and acceptance-byte assumptions above.**
- **Pick logs** are observations, not proof of pre-lock timing.
- **Unobserved calendar dates** include dates without contest opportunity; they are neither skips nor activity denominators.
- **The extension's history depth** is whatever the end-of-season API returned. A profile with any pending slot made the grab fail closed.
- **Disclosed overlaps:** X-10 (the 7/03 me-vs-leaderboard read of this corpus) and X-01/X-09; X-19 covered counts only.
- **Descriptive only.** No causal effect, skill rank or equivalence; nothing is nominated.
