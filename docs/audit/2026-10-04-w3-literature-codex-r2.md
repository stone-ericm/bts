# W3 literature / SOTA refresh — Codex review R2 (final)

## Verdict

**SIGN WITH EDITS.** Apply the bounded corrections below, then freeze with the stated limits. The W4 recommendations and literature qualifications now satisfy R1. The remaining defects concern the exposure inventory, the promised per-screen basis, and provenance locators; they require no experiment, outcome analysis, new candidate, or third review round.

Reviewed revision `e7e0829c3c0bb299ec4a6324d5a1b614d40db9cb`. Main advanced concurrently to `8de6d477a564b5c70518ba5bbdc60359c8a40d7d`; the memo, evidence files and tracker are unchanged from the review revision. Memo SHA256: `0d983d41598eb53b01c6994f3604deed146feb6f3c3cfcbb5d7b2d7b5b9bdd58`.

No `data/` file was read, no SSH or `gh` was used, no test or experiment was run, and no tracked file was edited. The two July audit JSONs under `docs/audit/` were inspected only for their `generated_by` and `git_head` identity fields, not to recompute or analyze outcomes. Historical measurements remain verified as documentary statements, not independently reproduced results.

## Findings

### R1 disposition and required appendix checks

- **Edits A–H and J landed.** Each replacement block from the archived R1 report is present; all 17 requested primary reference links are present. The extra qualifications in the evidence notes and the supersession notice in the inventory prevent the retained historical recommendations from being mistaken for current conclusions.
- **F1/F2/F4/F5/F7/F9 are resolved:** intercept-first rank 4a, conditional rank 1 without guaranteed pooling repair, correct benchmark attribution/metrics, separate prize objectives, bounded TabPFN scope/licensing, and squared-up/availability limits all remain intact. Selection belongs to the decision memo. No new public-source check was necessary to verify these unchanged R1 replacement blocks.
- **F3/F6/F8 are resolved subject to the edits below:** dispositions are scoped and history is distinguished from the dated W1.6 follow-up; the remaining table/provenance gaps must not be presented as completed checks.
- **Every literal tracker quotation matches the tracker text** after ordinary line-wrap normalization. The literal calibration-gate quotation matches its gate document, and the continuation quotation matches preregistration lines 383–384. The tracker locators, however, are one line low; see finding 2.
- **All 41 main-branch code references (39 distinct paths) match `git log -1 --format=%h -- <path>`.** All four named-branch code/path SHA pairs match their branch history and exist there. `swing-escalation` resolves to `5835245`. The five named health modules exist. Every explicitly cited result/protocol document exists; the abbreviated pooled-24-seed reference resolves, and the Gate B wildcard resolves to six documents. The fourth team-record branch contains the named code and its design/plan, as the follow-up states.
- **Positive producer statements are supported:** Phase C documents runner `ca71f06`; the DD-guardrail document reports manifest SHA `c68c7c0`; the calibration gate identifies deployed state `783986e`; the candidate freeze is `5004b1c8`. July identity metadata reports `96966f2e8f170e7378d475cb6b5a2de814c6cef2` and `69ef515461709b08a1580eb1a9ba1b3642764085`. This verifies what those records say, not historical runtime integrity. Missing code pins in the cited markdown must remain distinct from a claim that underlying artifacts contain no identity metadata.

### 1. P2 — Rows 13–14 incorrectly say “no” 2026 outcomes while citing analyses that consumed 2026 context

**Location:** memo:44–45; supporting appendix:104–109.

The rows now include the later 7/06 comparator and 7/13 sensitivity/run-structure interpretation. `docs/audit/2026-07-13-dd-p-policy-value-sensitivity.md:3–18` explicitly consumes live 2026 DD-leg evidence for its motivation and interpretation. Exposure-register X-04 (`docs/audit/2026-09-22-exposure-register.md:14`) explicitly covers that document and the 7/06 investigation: historical estimated-PA profiles **plus 2026 realized picks for context**, marked consumed.

The underlying historical OPE/CE-IS target evaluations need not have used 2026 labels. That narrower fact cannot support “no” for a row that also incorporates the later contextual analysis. This is the remaining decision-memo risk: already-consumed context could be treated as fresh validation. Cite the existing X-04; do not create a new read or registration.

### 2. P2 — Provenance text is sound, but its tracker line locators are uniformly one line low

**Locations:** memo:32–48,55–126; tracker-inventory:4.

For example, the exact DR reopening quotation is at tracker **189**, not 188; the classical-FDR trigger is **253**, not 252; and the model-class status/next action are **388–389**, not 387–388. The words match, so these are locator errors rather than invented triggers. The inventory's `T + 8` rule and appendix's “8 lines earlier” description must become **+9** for this revision. Preregistration and plan locators use different files and must not be incremented.

The appendix also uses bare “Producer pin: none” in several rows where only absence from the cited markdown was established. For example, the realized-picks FDR document describes output git-head metadata (`2026-05-05-realized-picks-fdr.md:127`). Add one convention sentence to prevent a missing documentary pin from being mistaken for proven absence of artifact identity.

### 3. P2 — Row 9 promises per-screen bases in the appendix, but the appendix contains no basis paragraph

**Locations:** memo:40,83–87.

“Per screen (appendix)” is not fulfilled by code paths and result titles. The sources distinguish historical residual/proxy evaluation, estimated-count aggregation on retrospective participant slates, and the live paired shadow. In particular, swing and kcontact compute `1-(1-p_pa)^PA_EST` (`src/bts/experiment/swing_screen.py:226–237,289–290`; `kcontact-screen:src/bts/experiment/k_contact_screen.py:316–337,685–686`), rather than establishing the production at-cutoff surface. The kcontact result reports a 2024 screen with 2025 untouched; the swing driver describes 2024H2. The team-record result reports 2024/2025, without certifying its aggregation basis. P-03 is the live 2026 paired stream.

Add the compact basis/limits paragraph below. It does not upgrade these screens into certified serving comparisons, does not infer missing producer provenance, and leaves unestablished exposures marked unestablished.

### 4. P3 — Two current-status sentences need their scope stated

**Locations:** memo:16,161.

The tracker now has a 10/04 update, so “was last updated on 2026-05-10” must refer to the period **before this refresh**. “W1.6: done” is broader than the completed exposure/archive items described in the follow-up; the same section still lists README hygiene as open. Narrow that completion statement to the items raised in this memo, without reviewing or certifying the rest of W1.6.

## Verbatim edits

### 1. Replace only the “2026 outcomes consumed” cells in §2 rows 13 and 14

**Row 13:**

> Historical v1 target evaluation: no 2026 labels. The later 7/06 investigation and 7/13 DD-p sensitivity interpretation consumed live 2026 context under X-04; that context is already consumed, not fresh validation.

**Row 14:**

> Historical CE-IS target evaluation: no 2026 labels. The later 7/13 sensitivity/run-structure discussion uses live 2026 context under X-04; the historical diagnostic is not a fresh 2026 validation read.

### 2. Correct the tracker locators and provenance convention

In memo §2 and §2a, apply the following simultaneous replacements to tracker locators only. Leave the preregistration `L381–384` and `plan:200` unchanged.

| Existing tracker locator | Correct locator at `e7e0829` |
|---|---|
| L188 | L189 |
| L196–198 | L197–199 |
| L208 | L209 |
| L218 | L219 |
| L230 | L231 |
| L238 | L239 |
| L252 | L253 |
| L260–262 | L261–263 |
| L262 | L263 |
| L412 | L413 |
| L276 | L277 |
| L291 | L292 |
| L302 | L303 |
| L317 | L318 |
| L330 | L331 |
| L28/L98 | L29/L99 |
| L343 | L344 |
| L362 | L363 |
| L387–388 | L388–389 |

Replace the §2a introduction with:

> Tracker locators refer to the file at `e7e0829`; the older inventory's `T:NNN` locators sit 9 lines earlier. A code sha is the last commit touching that path on main unless a branch is named. One repo revision pins existing infrastructure; it cannot certify the code that produced an old result, so a missing producer pin is stated, not filled in. “Producer pin: none” means no historical analysis-code pin was identified in the cited markdown; it does not establish that the underlying artifacts lack code-identity metadata. Recorded producer HEADs are documentary metadata, not independent verification of historical execution.

In tracker-inventory.md:4 replace `the current line is T + 8` with `the line at e7e0829 is T + 9`.

### 3. Add this bullet to appendix item 9, after its Results bullet

> *Evaluation basis / limits:* The embedding is a historical game-level residual test with ex-post opponent selection and a proxy baseline; its complete input/aggregation provenance is not certified here. Kcontact's documented 2024 screen uses lineup-slot PA estimates on retrospective participant slates; its result leaves 2025 untouched. The swing driver's documented screen is 2024H2 with lineup-slot PA estimates; failed controls invalidate family inference. Team-record reports 2024/2025 results; its aggregation basis is not certified here. P-03 is the live 2026 paired shadow stream (X-06). These historical screens are not certified at-cutoff production comparisons; the unestablished exposure/provenance limits in §2 remain in force.

### 4. Narrow the two status statements

Replace memo:16 with:

> No reviewed SOTA-cycle candidate has cleared a production gate. Before this 2026-10-04 refresh, the tracker had not been updated since 2026-05-10. The dated table and follow-up in §2 now record the intervening status.

Replace memo:161 with:

> **W1.6 items raised here:** the three retroactive exposure rows and the three-branch document archive were completed on 2026-10-04; see §2's dated follow-up and its separate team-record limitation. This memo does not certify completion of all W1.6 work. README hygiene remains open above.

After these edits, freeze without another review round. Retain the existing limits: unextracted temporal-default rankings, abstract/secondary/unverified source distinctions, unverified historical producer/basis details, D3's consumed-data boundary, and conditional selection rather than experiment or production authorization.
