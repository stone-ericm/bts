# Third check of the W15 contract (version 3)

Same rules as your first brief: change no tracked file, probes in `.codex-review/wr-check-w15/r3/`, no network, no pytest, bounded reads, copies of real files only.

This clone is at `2314d54f`. The contract `docs/reviews/2026-11-18-whole-range/kickoff-fix-w15.md` is now version 3. It adopts your round-2 design (preserve the original bytes and publish the observation, then the base rewrite with only usable entries; records kept per ET date with an identity; your replacement texts for P1, P2, P5 and P6). Three of your round-2 items are ruled OUT of this change set by the session's ruling W15-S1 (R2 sync behaviour, an explicit live status root through every feature caller, and two writers racing): they are ledgered with their cost. Treat W15-S1 as decided: classify those three as NOTE unless you find an ordinary sequence in which a pick is made on wrong features with NO alert and NO stderr line at all; then say so as a MUST FIX with the sequence.

Check only:
1. For each round-2 MUST FIX not covered by W15-S1: closed, closed differently but adequately, or open (probe where a sequence is involved: the unrelated bad member, new game, feed removed, fresh process sequence; an interrupt at each of the three renames; a late D run with a D+1 record on disk; two different observations on one day).
2. Anything new that version 3's evidence copy, dated records or identity rule makes wrong or ambiguous for an author (name the two readings).
3. The P6 texts and the worst-state table as rendered on this tree (`<preserved or "not preserved …">` included).

Deliverable: `.codex-review/wr-check-w15/contract-check-r3.md`, under 80 lines, verdict first (BUILD FROM THIS / BUILD WITH THESE EDITS with replacement texts / NOT YET with the smallest design change). End with what you ran and did not run. When complete print `WRCHKW15R3-<first 8 hex characters of the sha256 of contract-check-r3.md>` on its own line and `DONE`.
