# C2 (2027 Cycle 2): cycle index

**Approved by Eric 2026-10-06 ~11:59 EDT** ("Approve as proposed (Recommended)", given directly and relayed by the herdr manager; register §C row **C2-approval**): `docs/audit/2026-10-06-c2-cycle-proposal.md` as written. On the rank-1 gap he ruled "Inside the watchdog (Recommended)" (row **C2-rank1-part-a**). **Started 2026-10-06.** Lead session: bts-lead2 (herdr `wA:pQ`).
- **Caps:** $0 of new spend; 15 CPU-hours of box research jobs (11.6 planned).
- **Reviews:** at most 2 Codex rounds per review unit. Without a SIGN, the item stops and returns with options; a third round needs a recorded ruling. Production code ships only on SIGN, then D7.
- **Stop rules:** proposal §6. **Owner gates:** D2 (independent acceptance), D7 (every production change and research result), D1's trade table for 4b.
- **Order:** one item at a time, in Eric's order: (a) → (b) → (c). An item is done for the order when its last code review signs and its D7 request has gone to Eric.

## How C2 jobs run on the box
- **The infrastructure:** C1's launcher, guard and ledger, unchanged (infrastructure review r4 SIGN on `3d6ae3f`; its rules are in `docs/sota_audit/2026-10-04-c1-cycle-index.md` §"How C1 jobs run on the box").
- **Naming:** `--name c2-<item>-<step>`, so units are `c1-c2-…` and C2's ledger rows can be summed on their own.
- **The cap check:** before each launch, C2's used hours plus the declared budget must stay within 15. A job not on the planned list, or any rerun, is reported to the manager first.
- **The shared ledger at the start:** 0.4452 CPU-hours in 16 rows, read 2026-10-06; all of it is C1.
- **The box worktree:** `~/projects/bts-c1` was at `1fa872a`, older than the signed launcher. It is re-pinned to each job's reviewed commit before its launch.

## Items

| Item | Starts from | Finish by | Status |
|---|---|---|---|
| (a) rank-2 watchdog, plus rank-1 Part A in unit 4 | W0 `c429c41` plus W0 r3's R3-1 to R3-4; W1 evidence map and spec draft; registration `docs/sota_audit/2026-10-04-prereg-c1-watchdog.md` and plan `docs/superpowers/plans/2026-10-06-c1-r2-watchdog-build.md`; rank-1 registration `docs/sota_audit/2026-10-04-prereg-c1-mlb-capture.md` Part A | every unit signed by 2026-11-30 | **ACTIVE from 2026-10-06:** unit 1 (W0 repair) built on branch `c2-watchdog` (design `docs/superpowers/specs/2026-10-06-c2-w0-repair-design.md`, fixes `4a845e8` + `808ae12`, evidence `docs/audit/2026-10-06-c2-w0-evidence/` on the branch: 28/28 mutants RED, watchdog 121, fast 4028). **Codex review r1 (12:33 ET, `d4c11ae`): BLOCK** (`docs/audit/2026-10-06-c2-w0-codex-r1.md`: R3-1/R3-3/R3-4 closed, R3-2 partly; new B1 starvation, B2 impossible "sent" metadata accepted, B3 split claim unvalidated, B4 fencing false green; reviewer disclosed a memory-registry search, so not strictly blind). Fixes `9e4d808` (36/36 mutants RED, watchdog 143, fast 4050). **Review r2 (last; 12:56 ET, `926726d`): SIGN WITH EDITS** (`docs/audit/2026-10-06-c2-w0-codex-r2.md`; E1 unsent message id, E2 one-component claim tests, M38–M40), **applied verbatim in `f1b948a`: unit 1 (W0) SIGNED.** Final evidence `539dd85`: 39/39 mutants RED, watchdog 163, fast 4070. W0 is the foundation only; no deploy until the whole watchdog signs and Eric's D7. **Unit 2** (W1 + W5 + W7): spec `docs/superpowers/specs/2026-10-06-c2-unit2-fast-job-spec.md` (branch, `b12c018`); **spec review d1 (13:30 ET): REVISE** (`docs/audit/2026-10-06-c2-unit2-spec-codex-d1.md`, 12 findings; reviewer again searched its memory registry before reading the ban, disclosed). Revision d2 `7efd888`; **spec review d2 (last; 13:55–14:27 ET): REVISE** (`docs/audit/2026-10-06-c2-unit2-spec-codex-d2.md`; no outside search this round). **Unit 2 STOPPED under the C2 review cap; options to the owner (2026-10-06 14:28 ET).** Central finding: multi-file W1 verdicts need production-capable writer closure (P3) or must stay pending/unverifiable. **PAUSED** (row C2-watchdog-pause; the manager's call under the 10/03 delegation, 14:29 ET): resume later, with the P3 decision first. |
| (b) rank 3, then the pick-path capture, then 4a | rank-3 code `5cca66e` plus r4's R4-1; 4a study `0f9ffb0` plus r2's R1–R4 and R10; serving witness `e66b440` plus S1–S3 | code signed and the rank-3 build accepted by 2027-01-31; capture deployed by 2027-03-10 | **ACTIVE from 2026-10-06 14:30 ET** (row C2-watchdog-pause): step 1, the rank-3 R4-1 fix |
| (c) 4b | code `3648120` / `ad29869` plus r3's required changes 1–8 | trade table to Eric by 2027-02-15 | waiting for (b) |
| (d) P3 / W-state | design draft with review d1 REVISE | — | deferred; reopens only per proposal §5 (a) |
| Rank-1 Part B (concordance diagnostic) | rank-1 registration Part B | after (c) | waiting; needs X-33 first |

## Compute ledger (C2 rows only)
| Date | C2 CPU-hours (cumulative) | Note |
|---|---|---|
| 2026-10-06 | 0 | cycle started; no C2 job yet |
