# Workflow review, Claude subagent (fresh, evidence brief only), 2026-11-18 about 21:50 ET

## 1. Flaws, ranked by expected cost
1. Nothing has run the nine change sets together. Each was reviewed alone, on a studio, with the box deleted. The season-start path (refusal, saved-pick check, morning check, stale warning, switch) only works as a sequence; the first real run of that sequence would be on the restored box inside the 21-day window. Likely in-season cost: a missed day from a silent fail-closed interaction, not a wrong pick.
2. "Both SIGN" is weaker than it looks. Round 1: Codex found defects in 4 change sets, Carbon in 2, 1 in common. With overlap that low, a crude capture-recapture estimate says about as many defects remain as were found. The sets that passed clean are the least examined, and they include 4b-2, the live refusal.
3. Alarm and verifier changes were filed as tier B. F4 to F8: five unrelated fixes in one change set, no contract check; about 4.5 hours, four BLOCKs, two NOT YETs, one merged cron flag; F6 was clean at 11:48 and waited until 15:17. Residue: known false CRITICALs in opening week.
4. A model feature fix merged before its effect was measured (the policy thresholds were tuned on the old values).
5. The lead ran about 42 hours without a pause and its errors cluster late (hardcoded dates approved, a loose F4 rule, a skipped contract check, five fixes bundled, a harness that printed nothing).
6. Round-1 findings repeat the same classes (interrupt mid-work, malformed or non-regular files, midnight date order, failed sends). The lesson is not fed back into contracts.
7. Scope has no stop: each ten-minute discovery lane adds work and moves the range the final review must cover.

## 2. False greens
- 6b-2 round 2: the gate was handed to Carbon, which had missed the FIFO race in round 1.
- Round-2 SIGNs cover the lead's own fix, not the change set as it now stands.
- The lead's dialog approvals (hardcoded dates passed; five prompts in ten minutes trains rubber-stamping).
- The F6 "verification": the adopted test, not a real-artifact check.
- "Closed out" for F4, F5, F7, F8: the defects are still there, relabelled as expected alarms.
- Short SIGN reviews (13 to 20 minutes on 4b-2, 5a, bullpen).
- The coming whole-range review: one reviewer on 2,300 lines is one more sample from the same low-overlap distribution.

## 3. Wasted motion
- Serial pipeline with a spare slot; merges wait 40 to 50 minutes on Carbon after Codex has finished.
- Per-commit approvals inside an agent's own worktree.
- Three review rounds on audit pieces whose design nobody had checked.
- Logging one-job-rule overlaps by hand (enforce with a lock in the wrapper).
- A docs commit per review round.
- Owner input arriving after review (6a).

## 4. Two changes for tomorrow
1. Freeze scope and rehearse before the whole-range review: park new discovery findings; land the small fixes; get the bullpen backtest and the policy-bin decision; tag a commit; on the studio run a scripted fake opening week against real artifacts (no observation, dated observation, late correction, enable, fetch, revoke, stale reading, malformed status file) and watch each alarm fire; only then launch the whole-range review on that tag, split into three or four slices by scenario with one fresh reviewer each.
2. Fix the contract template and the tiering: standing hostile-input rows in every contract (interrupt, malformed or non-regular file, midnight and year boundary, failed or unconfirmed send, restart mid-step); anything that silences an alarm or loosens a verifier is tier A, one piece per change set, with a contract check and a "must still catch" table; a hard stop for the lead each night.

## 5. Delivery estimate
"Whole-range review signed on a frozen commit by 12-01": about 70%, only if scope freezes now. Earliest 11-22, likely 11-25 to 11-29; 12-05 to 12-12 if the policy-bin finding needs an artifact rebuild, the bullpen fix moves metrics enough to need an owner decision, or discovery lanes keep running. Critical path: the owner's decisions, the frozen tag, the rehearsal, the whole-range review and its fixes. 12-01 is not the date that matters: code signed in December sits three months before the restore and any later commit makes the review stale; deployability can only be shown on the restored box (about 03-03 for a 03-24 opener). Plan the 21 days as the real test window; keep main frozen or re-review the delta before the restore.
