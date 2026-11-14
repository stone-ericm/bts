# Kickoff 3: one correction to item B (the lead's contract was too strict)

Your handover-2 implements kickoff-2 B as written: ANY game feed with a missing or unreadable `gameData.game.type` now raises, "whatever else is in the directory". That is stricter than the parent and it creates a new fatal path: `build_season` is also called by the prediction refresh (`predict._refresh_season_data`), so in season ONE cached feed with a bad body (the pull never re-downloads an existing file) would fail every build, abort the nightly chain and fail every prediction cascade: no pick, every day, until someone deletes the file. The parent skipped such a feed and built from the rest.

## Corrected contract for `build_season`
Behave exactly as the parent (`git show main:src/bts/data/build.py`) in every state EXCEPT these two, which are a clean no-op (return `None`, write nothing):
1. the season directory is missing or holds no game feed (weather sidecars are not game feeds);
2. every game feed in it has a readable game type and none of them is a requested (regular-season) type.

Everything else is the parent's behaviour:
- a feed with a missing or unreadable game type is SKIPPED, as the parent skipped it, with one stderr line naming the file (new, informational only);
- if the directory contains such a feed and NO feed of a requested type, that is not case 2: raise a clear error naming the unreadable file(s) (the parent also failed here, with its schema-drift error; an unreadable feed is not evidence that the season has no games);
- regular-season feeds that yield no rows, the seven-inning exclusion leaving nothing, truncated JSON: unchanged from the parent.

## Tests
Keep your malformed-metadata cases but split them by this rule: a typeless or `{}` feed BESIDE at least one regular-season feed builds from the regular feed(s), exits 0, writes the parquet, and prints the skip line naming the file; the same feed alone, or beside only non-regular feeds, exits non-zero naming the file and writes nothing. Add: for the in-season case, the built parquet's rows equal the parent's rows for the same directory (run the parent's `build_season` from `git show main:src/bts/data/build.py` in a scratch import, as you did for the scheduler reference). Also one test through `predict._refresh_season_data`: a season directory with regular feeds plus one `{}` feed refreshes normally (parquet written, marker written).
Mutation-check the three branches (skip-with-regular; raise-without-regular; no-op only when all types are readable non-regular).

Nothing else changes (A, C, D stand). Run the kickoff-2 focused set again and paste the last line.

## Deliverable
`/Users/eric/projects/bts-pp-c1/.codex-review/g2-fix/handover-3.md`, under 50 lines. Then print `G2F3-<first 8 hex characters of the sha256 of the handover file>` and `DONE`.
