# Change set 2: where it stopped on 2026-11-15, and what the next attempt changes

Written by the lead session after the third BLOCK on this change set. Under the work policies that ends this loop: the obligations are recorded here and the next attempt starts fresh with something material changed. Nothing is merged. The owner's condition for the confirmation round was that a block sends the change set back to him.

## State
- Branch `sb-cs2` on origin at `ee9fb870` (base `d8eb50d0` on `main`). Four commits: the author's (`3b4196be`), then three rounds of fixes by the lead (`e6d6f0a9`, `4cf73de7`, `a26958dc`, `ee9fb870`). Do not merge it as it is.
- Contract and its two amendments: `kickoff-cs2.md`. Handover: `handover-cs2.md`. Reports and probes: `cs2-review/` (`codex-r1`, `carbon-r1`, `codex-r2`, `codex-confirm`).
- Gate: green at `a26958dc` (5,090 passed, the 53 known studio failures only). A gate on `ee9fb870` was started and is moot.

## The three blocks
1. **Round 1 (Codex `CS2RC-7b337f8a`, Carbon `CS2RG-172d8223`).** The CONTRACT let a file that is not a JSON object count as "another season", so a dated scorer replaced corrupt state with an invented streak. Both reviewers rejected that. Carbon also found the reconcile rebuild broken for a deeply nested file. Fixed; the contract amended.
2. **Round 2 (Codex `CS2RC2-a30dcb54`).** The scheduler's four new season reads parsed the date with `date.fromisoformat`, which rejects an unpadded date the scheduler accepted before. Fixed by switching to `strptime`.
3. **Confirmation (Codex `CS2RC3-61bce5cd`).** `strptime` in turn rejects the compact (`20270324`) and week (`2027-W12-3`) forms that `fromisoformat` accepted, so those now fail in the three polling branches where the base returned `final`. The reviewer's proposed fix is to try one parser and fall back to the other.

## Why it failed three times
Blocks 2 and 3 are the same mistake made twice, by the author and then by the lead. Rows 6 and 8 of the contract put a season on four READS that are for display or for the shadow pick. At the base those reads took no date at all, so they accepted any string. Giving them a season meant parsing the date, and every parser is stricter than "any string": each fix chose a different parser and lost a different set of inputs. A third parser choice (try both) would still reject whatever neither accepts, where the base accepted everything; the confirmation's own notes list those remaining losses (whitespace, a time suffix, other separators, `None`).

The contract was wrong to treat "which season is this date" as a question that always has an answer at these sites.

## What the next attempt changes (material)
1. **The four scheduler reads and the CLI's void read never fail on the date.** One helper, in one place, returns the season of a date string or `None` when it cannot tell, trying `date.fromisoformat` then `strptime("%Y-%m-%d")`, and never raises. With `None` the read is the dateless read, which is exactly the base's behaviour for that input. So every input the base accepted is still accepted, every date the code can read is season-scoped, and no parser choice can lose an input again. The contract states this as a rule for rows 6 to 9 and as a test: for a matrix of date strings (padded, unpadded, compact, week, whitespace, a time suffix, other separators, junk, empty, a `date` object, `None`) each of those call paths returns what it returned at `d8eb50d0`, except that a parseable date reads the season-scoped value.
2. **`update_streak` stays strict** (row 3: a date that is not a valid ISO date string raises before any I/O). It writes, so it must know its season.
3. **The contract is rewritten as one document** with both amendments folded in (corrupt state is a loud error; the season-argument type check), instead of a table plus amendments.
4. **A fresh author** works from the rewritten contract ON TOP of `sb-cs2` (the reviewed parts are kept: three reviewers have run the season-boundary sequences, the corruption matrix and the changed-test audit against them), and changes only the date handling. Then a full review of the whole change set by a reviewer that has seen none of these rounds, as the policy requires after a stopped loop, with Carbon in parallel.

## Obligations carried over (every one to be closed or explicitly ruled on)
- Confirmation B2: compact and week ISO dates in the three polling branches (six regressions).
- Confirmation notes: the other inputs the polling display branches accepted at the base and reject now (closed by item 1 above if it is built as stated; the matrix is the proof).
- Add compact and week forms, and the rest of the matrix, to the scheduler tests (the 122-green result missed B2).
- Round 2 N3: the saver-zone miss passes an explicit saver to `save_streak`, so a file corrupted AFTER its successful reads is replaced. Same on the base. Rule on it or leave it with a ledgered reason.
- Round 1 N1 (both reviewers): `state/export.py` and `state/regenerate.py` drop the season tag, so a restored file reads as season-fresh until the next complete replay. Deferred to a later change set; keep it on the list.
- Round 2 N2: the reconcile summary line samples the clock again after the replay (display only at the ET New Year). Left as is by the lead's ruling; the reviewer agreed.

## What is not in doubt (three reviewers, run against the real scorers)
The season tag and reset on a readable file; the tagless-file rule; the once-per-date rule and the saver zone on a current-season file; reconcile's replay and stamp; the pp1 B1 marker protection on a tagless file; corrupt state as a loud error with the complete-replay rebuild; the changed existing tests (848 assertions unchanged, the incident-register file checked line by line); the clock pins.
