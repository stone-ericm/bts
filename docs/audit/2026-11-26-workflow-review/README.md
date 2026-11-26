# Daily multi-model workflow review, 2026-11-26 (overdue: none ran 11-22 to 11-25)

Two fresh reviewers (a Claude subagent and a Codex session at the highest effort, memories off) each read `evidence-brief.md` (agent log, landings and outcomes for 2026-11-21 16:10 to 2026-11-26 00:40 ET, without the lead's diagnosis): `opinion-claude.md`, `opinion-codex.md` (`WFREV4-d965d853`). Ran: one read of the brief each. Note: the Claude opinion reads the brief's "Codex weekly about 98%" as used; it is 98% LEFT.

## Where they agree
- **The owner's acceptance is doing most of the gating.** Of about eleven reviewed pieces only H2 and F1b signed; Ea, G, H and the R52 tool ran or merged on Eric's acceptance, most questions answered "1", several overnight. Acceptance is a valid decision but must not read as a pass.
- **Round 2 re-hunts instead of confirming:** G blocked twice on pre-existing behaviour; H's corrupt-artifact promise and F2/F5's state rules grew new classes each round; contract checks returned BUILD WITH EDITS and the pieces still blocked.
- **The closing test is stale:** its head has moved (Ea, G, H, H2, F1 merged, two of them policy); it must re-run on a frozen head.
- **"Passes alone" and "flaky" are weak:** the extra failures need a clean repeat on a frozen snapshot.
- **Supervision gaps were the lead's:** four missed daily reviews, a 22-hour idle session, hand-typed times, a job started on a held slot, a commit during a gate.

## Where they differ
- **Sessions.** Claude: count BLOCKs per piece, not per session. Codex: keep the coordinator across pieces; the third BLOCK is a pause to diagnose, not an automatic restart.
- **R52.** Claude: adding seeds after a FAIL is optional stopping, and the streak metric fell. Codex: the PASS answers the agreed criterion; keep the 8-seed FAIL, the accepted tool findings and the streak drop beside it (done in `FINDINGS.md` and `bullpen-r59-result/`).
- **Delivery:** Codex 11-28 to 12-01; Claude likely 12-02 to 12-06 (best 11-29, bad 12-12 to 12-20).

## What the lead adopts now (ruling R70)
1. **Round 2 is bounded.** The round-2 reviewer answers only: is each round-1 finding fixed, and did the fixes or the change set introduce a defect? A finding about behaviour the change did not introduce, or a new class elsewhere, goes to the ledger and the owner list, not a BLOCK (unless the change newly makes it reachable).
2. **Contracts carry concrete failure examples** (the sequences the change must catch and the ordinary ones it must leave alone) before an author starts, and the contract check runs them, with the existing tests that pin the touched functions.
3. **A session continues across pieces.** Its BLOCKs are counted per piece; the third counted BLOCK pauses the session for a diagnosis with the lead (does it need a fresh start, or is the piece the problem?), instead of stopping it automatically.
4. **One owner list a day, in daytime** (R51 item 6, which the lead broke): questions are queued and sent together around 10:00 ET unless something blocks all work; each acceptance question lists every open finding with its cost if wrong, and the record says "accepted" next to it, never "passed".
5. **Carbon stops once a Codex BLOCK has decided a round** (no reviewing "for the record" for an hour).
6. **Extra gate failures need a clean repeat:** a test that fails in a gate and passes alone is re-run five times alone and once in its file's order before the gate counts as green; no git writes, `git fetch` included, anywhere during a gate.
7. **The closing test re-runs once, last, on a frozen head** after F2b, F3, F4 and F5b: nothing merges while it runs, and what it finds is fixed only if it can cause a wrong pick, wrong state, a lost alert or a lost day; anything else is ledgered or put to Eric.
8. **Delivery estimate the lead uses: the later one, 2026-12-02 to 12-06** (critical path: F2b and F5b, F3 and F4, the frozen-head closing run, triage).
