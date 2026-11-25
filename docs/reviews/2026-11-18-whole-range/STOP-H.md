# STOP: change set H, the longest-streak rule from streak 57 (R62, R65, R66), after its second round (2026-11-25 02:41, `bts-wr-fresh11`)

**Not merged.** Branch `fix/wr-h` (pushed, head `4a28e521`): code `813935bc` (round 1) and `84805852` (round 2); contract `kickoff-fix-h.md` version 3 with amendments 1 and 2 and both rounds recorded; contract checks `contract-check-h/` (NOT YET, answered by R66) and `contract-check-h2/` (BUILD WITH THESE EDITS); reviews `review-fix-h-codex/` and `review-fix-h-codex-r2/`. `main`'s `src/` is unchanged by H: at streak 57 or more the policy still skips every day. Under R51 a second-round BLOCK is not overridden, and the session builds no follow-up of its own (R57).

## In plain words
The thing Eric asked for works and was never faulted. In both rounds the reviewers confirmed, with their own probes: from streak 57 the policy plays every day (a double with three or fewer days left, a single otherwise); the boundary is one constant; nothing below 57 changes (over 600,000 states compared with `main`, including every state G's guard changed); the real scheduler chooses, saves, sends and records these picks; grading moves the streak from 57 to 58; every reader of the decision record copes.

**Both rounds blocked on one promise in the contract: "if the base policy file is corrupt, play a plain single and say so in the record."**
- Round 1: a base file whose table was a single number, not a table, was not recognised as corrupt.
- Round 2 (after a check on the table's type, shape and values): a base file whose bin edges were words, not numbers, was not recognised as corrupt.

In both cases the pick that went out was the rule's normal pick (a double on the last day), and the record did not say the base was damaged. To get there the base file must be damaged in its contents yet still load, AND the tail policy must have been rebuilt against that damaged file (the rule takes its two rates from the tail, which is bound to the base file's hash), with the artifact health check silent. The rule itself never reads the base table.

## Why it does not converge as written
The promise makes H decide, by a hand-written list of checks, whether a base file "could answer a reach-57 lookup". Each reviewer finds one more way a file can load and still not answer one (round 2's reviewer also found that five of the existing clauses are noticed by no test). A third list would meet a third counterexample.

## Ways forward (the lead rules; anything that lands needs a materially revised scope with a fresh verdict, or Eric's recorded acceptance: R51)
1. **Eric accepts H with the reviewer's one-line edit applied (each bin edge must be a number), on a green gate, with no further review** (the session's recommendation). The residue is written as a limit: a base file damaged in some way the checks do not list, with a tail rebuilt against it, gets the rule's normal pick and a record with no degradation note. Cost if wrong: on such a day the pick is still the rule's own; only the record's "degraded" note is missing. About 45 minutes (author 10, gate 25 once the slot is free).
2. **A small follow-up, H3, that changes the mechanism instead of extending the list:** replace the checks by doing the thing they predict: inside the same `try`, run the real reach-57 lookup on the loaded base for every bin on the first and last day and both savers; any exception or non-action result is the forced single. One fresh Codex verdict on those lines. Cost: one more round (about 90 minutes) and it can find a lookup the probe set does not run.
3. **Narrow the promise:** the rule needs only a validated tail whose recorded base hash equals the hash of the base bytes read; the base's CONTENTS are not H's concern (the reach-57 regime and the health check own them). The base check is removed, the contract's H4 row is rewritten to "base missing, unreadable or hash-mismatched", and a fresh verdict is taken on the smaller change. Cost: one more round; a reviewer may hold that a base that cannot be used below 57 should not be trusted at 57.
4. **Leave `main` as it is:** a 57 streak stops playing, as today. H2 (raising the 57 caps) has no purpose without H.

## For the lead
- **No full gate has run on any H commit** (queued twice, cancelled at each BLOCK before it started). Whatever lands needs one.
- H2 is not started: draft contract `~/bts-runs/draft-kickoff-fix-h2.md`, cap ruled (one constant, 400).
- Carbon's reviews of both rounds were still running at 02:41; each is archived to the branch when it reports.
- Work copy `~/projects/bts-work/wr-fix-h` and gate copy `wr-gate-h` (at `84805852`, synced) are kept until the lead rules.
