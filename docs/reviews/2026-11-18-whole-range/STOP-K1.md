# STOP-K1: change set K1 stopped after round 2 (2026-11-26 13:55 ET, bts-wr-fresh15)

Branch `fix/wr-k1b` (pushed; code `34981a1a`, round 2 `bf3d4f02`, records to `4c301f7f`; the first push `fix/wr-k1` carries the same trees under a wrong commit identity, see `handover-fix-k1-r2.md`). Not merged; `main` unchanged.

## What K1 is
The live contest-state refusal DM is one short CRITICAL under 1,000 code points for every error (it was 1,146/1,147 for an absent/revoked 2027 switch with the box's path, over the chat limit), pointing to the journal, `bts season-contract status` and checklist row B8a; the four in-day alerts dedup under the slate's ET date (C3-2). Contract `kickoff-fix-k1.md` with Amendments 1 to 3; contract check `WRCHKK1-b7d4dc06` BUILD WITH THESE EDITS.

## The rounds
- Round 1, Codex xhigh **BLOCK** `WRK1REV1-986bc0af`: every length, date and scope claim held (peak 971 code points, 936 grapheme clusters; 36 captured runs equal to the base apart from the body and the dedup date); B1 (R64): B8a's "enter before the first lineup check" loses a late-start day's pick that filing between checks would give. Carbon stopped (R70 item 5).
- Round 2 (B8a sentence + date-forwarding tests; no src change), Codex xhigh **BLOCK** `WRK1REV2-2ea27ed3` (archive `review-fix-k1-codex-r2/`): round 1's B1 (original sequence), the Edit and the test gap are FIXED (eight single-site scratch changes all fail a test). Two defects of the same class, both because the DM now sends the operator to B8a instead of carrying the full refusal (whose dated template named the live path):
  - **B2:** B8a's daily command has no `--picks-dir`; with a non-default scheduler picks directory the observation lands in `data/picks`, status stays refused, the next check sends nothing. Fix: add `--picks-dir DIR` to the daily command and define DIR as the scheduler's absolute picks directory, the same used for status.
  - **B3:** the new late-filing condition names only the FIRST lineup check; a refusal at a later check or a fallback with a later check still available is not covered (two replays, zero picks; filing gives one). Fix (reviewer's text): "If a lineup check or fallback is blocked by unusable contest state, confirm this is still today's slate in ET and that MLB has settled every prior entry or confirmed its absence after its window closed. If a later lineup check or fallback refresh still has time to produce an enterable pick, file the verified dated observation between runs."
  Carbon round 2 stopped before its report (R70 item 5).
- Counted BLOCKs for K1: two (R57: the lead rules the one follow-up).

## Gate
On `4c301f7f` (round 2's tree; src identical to round 1's): started 13:47, still running at the stop; result appended when it ends.

## Ways forward (for the lead)
1. **Recommended: K1b, the two B8a sentences only** (B2 and B3 verbatim from the reviewer), each with a RED test reading the checklist, the F2 hash test extended as in Amendment 3, no src change; one fresh Codex xhigh check limited to those two sentences under R64; SIGN or SIGN WITH EDITS merges.
2. Put the live path back into the DM: costs length margin (the bound must then hold for a long configured path) and reopens the class round 1 found.
3. Accept K1 as it stands with B2 and B3 written as known limits (Eric).
