# Kickoff 4: one correction to `build_season`, and the tests that hid it

Round 2 of both reviews blocked `dbb78b1` on the same point. Reports: `.codex-review/g2-review/report-codex-r2.md` (B1 and its false-green note) and, for the test fixture, the excerpt at the end of this file. Nothing else in the branch changes: do not touch `predict.py`, `scheduler.py`, `day_outcome.py`, the cron script or any other test file.

## The defect
`src/bts/data/build.py` replaced the parent's type extraction with a `try/except (KeyError, TypeError)` that also swallows shapes on which the parent RAISED. Beside a valid regular-season feed, a feed that is a JSON list or `null`, or has `gameData: null`, `game: null`, or a `type` that is a list or dict, is now skipped and a partial parquet is written and published. On `main` each of those raises (`AttributeError` from the `.get` chain, `TypeError` from the set membership), nothing is written, and the nightly chain stops.

## The fix (tier A; RED first)
Restore the parent's two extraction lines and its membership test VERBATIM, and add only the bookkeeping the no-op needs. The loop body up to the doubleheader check becomes exactly:

```python
        # Filter by game type
        game_info = feed.get("gameData", {}).get("game", {})
        gt = game_info.get("type", "")
        if gt not in game_types:
            if not isinstance(gt, str) or not gt.strip():
                untyped_files.append(str(json_path))
                print(f"Skipping {json_path}: missing or unreadable gameData.game.type.", file=sys.stderr)
            skipped += 1
            continue
        eligible_games += 1
```

and the end-of-loop rule stays as it is now, on `untyped_files`: with no requested-type feed, raise `ValueError` naming every untyped file if there is one, otherwise return `None` and write nothing. Keep the missing-directory case a no-op (`None`), as now. Do not catch any exception from the three parent lines. Rename `unreadable_files` to `untyped_files` or keep the name, your choice; update the docstring to say: untyped means a missing type or a hashable value that is not a non-blank string; any other malformed shape raises as it always has.

Resulting contract, to test in full:
| Feed shape | With a regular-season feed beside it | With only non-regular or no other feeds |
|---|---|---|
| untyped: `{}`, `{"gameData": {}}`, `{"gameData": {"game": {}}}`, `{"gameData": {"game": {"pk": 1}}}`, `type` in `None`, `7`, `True`, `""`, `"   "` | skipped with the one stderr line; the output equals a build of the same directory without that file | `ValueError` naming the file; nothing written; an existing parquet untouched (bytes and mtime) |
| fatal on main: JSON `[]`, JSON `null`, `{"gameData": null}`, `{"gameData": {"game": null}}`, `type` `[]`, `type` `{}` | raises (`AttributeError` or `TypeError`); nothing written; an existing parquet untouched | the same |
| readable non-requested type (`"S"`, `"E"`, `"X"`) only, or no feed, or no directory | (not applicable) | `None`; exit 0 from the CLI; nothing written |

## Tests (`tests/test_data_build_empty.py` only)
1. RED first: add the six fatal shapes × companion in (`None`, `"S"`, `"R"`) and show the `"R"` cases fail on `dbb78b1` before the fix. Report the red count.
2. Split `UNREADABLE_FEEDS` into the two classes in the table. The existing test that expects exit 0 for null/list/unhashable shapes beside `"R"` is wrong and must change.
3. Delete the `parent_build` fixture. It runs `git show main:…`, which errors in any checkout without a local `main` branch (the deploy workflow checks out `deploy`) and compares the file with itself once this is merged. Replace the comparison with one that needs no git: build the directory with the bad file, build a copy of the directory without it, and assert the two frames are equal (`pd.testing.assert_frame_equal`), for every untyped shape.
4. No test may shell out to git or depend on a branch name.

## Deliverable
`.codex-review/g2-fix/handover-4.md`, under 60 lines: the diff summary, the RED count, the focused test command and counts (the same focused set as handover 3), `git diff --stat`, and anything you could not make match the table. Do not commit. When the file is complete, print `G2F4-<first 8 hex characters of the sha256 of the handover file>` and then `DONE`.

## Excerpt from the second reviewer (test fixture)
> `tests/test_data_build_empty.py:99-110` fails at setup with `CalledProcessError` (`fatal: invalid object name 'main'`), causing 8 pytest ERRORs on a detached checkout. Once merged to `main`, `main:src/bts/data/build.py` equals `HEAD`, turning `test_cli_typeless_feed_preserves_parent_rows_with_regular_feed` into a `HEAD == HEAD` tautology. It also restricts `bad_feed` to `{}` and a missing type key, excluding the shapes that break parent parity.
