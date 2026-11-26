# Adversarial review, round 3 (final check of one correction)

The checkout is now at `c339361`, one commit after the `dbb78b1` you reviewed in round 2. `git show HEAD` is the whole change: `src/bts/data/build.py` and `tests/test_data_build_empty.py` only. Your round-2 report is `.codex-review/agy-g3/report-g3-r2.md`. Same rules as before: read-only on tracked files, your own probes under `.codex-review/agy-g3/scratch/`, no commit or push.

## What changed and the contract
Your round-2 B1 and B2 were confirmed. The builder's `try/except` is gone: the base's two extraction lines and its membership test are back verbatim (`git show e3edbf58:src/bts/data/build.py` to compare), and the only additions are bookkeeping for the pre-season no-op. The contract:

| Feed shape | With a regular-season feed beside it | With only non-regular or no other feeds |
|---|---|---|
| untyped: `{}`, `{"gameData": {}}`, `{"gameData": {"game": {}}}`, a `game` object without `type`, `type` in `None`, `7`, `True`, `""`, `"   "` | skipped with one stderr line; the output equals a build of the same directory without that file (the base skipped these silently) | raises, naming the file; nothing written; an existing parquet untouched (the base also failed here, with a different exception) |
| fatal on the base: JSON `[]`, JSON `null`, `{"gameData": null}`, `{"gameData": {"game": null}}`, `type` `[]`, `type` `{}` | raises as the base does; nothing written; an existing parquet untouched | the same |
| readable non-requested type only (`"S"`, `"E"`, `"X"`), or no feed, or no season directory | (not applicable) | no-op: returns `None`, writes nothing, the CLI prints one stderr line and exits 0 (the base raised) |

A readable but unknown type string (your round-2 note 1) is deliberately in the third row: before a season's first game the directory holds only non-regular games, and refusing an unfamiliar code there would stop Opening Night's predictions.

The `parent_build` fixture is deleted; no test shells out to git.

## What to establish
1. Rerun your 32-case parity matrix against base `e3edbf58` and extend it if you can think of a shape it lacks. Report every case where rows written, exception or exit status, files touched, or an existing parquet's bytes differ from the base outside the third row.
2. Any state where the nightly chain now reaches `preview` or `sync-to-r2` on data the base would have refused, or a prediction cascade fails that the base's did not.
3. The changed tests: do they pass in this detached checkout with no `main` branch; would any still pass with the behaviour it names broken.
4. Anything else in `git diff e3edbf58 HEAD` you now consider wrong that you did not report before. Findings in code the diff does not touch are notes.

## Deliverable
`/Users/eric/projects/bts-agy-g3/.codex-review/agy-g3/report-g3-r3.md`, under 80 lines: `Verdict` (SIGN, SIGN WITH EDITS, or BLOCK), `Round-2 findings`, `Blocking findings`, `Notes`, `False greens`, `What I ran`. A BLOCK names a concrete input or state, the wrong production outcome, file and line, the smallest fix and how you demonstrated it. When the file is complete, print `AGYG3R3-<first 8 hex characters of the sha256 of the report file>` and then `DONE`.
