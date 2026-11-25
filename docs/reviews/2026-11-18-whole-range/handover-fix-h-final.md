# Change set H, Amendment 3

Commit `dad28268` on `fix/wr-h`. Only `src/bts/strategy.py` and `tests/test_wr_h_longest.py` were committed. The longest-streak validator requires finite, strictly increasing numeric edges, rejects boolean edges, and rejects masked policy tables.

## Tests first

- On starting HEAD `35279dbe`, all eight malformed-edge resolver cases (strings, nested lists, complex values, NaN, infinity, boolean, equal edges, and decreasing edges), the masked-table resolver case, and the string-edge scheduler case failed. They returned or delivered a double where the test expected a forced single. The healthy controls passed.
- All four F3 cases passed on starting HEAD. With each corresponding clause disabled individually in an isolated Python process, the integer-horizon, positive-horizon, boundary-list, and ndarray cases each failed because the rule returned double.
- All 16 requested focused suites passed in three sequential bounded batches (232 + 96 + 221 = 549 tests). The environment rejected the niceness adjustment and the commands continued under their timeouts.

## Archived-base comparison on the commit

The existing test extracted `git archive 0a2bec10 src` into a temporary directory and asserted the imported `bts.__file__` for both trees. The existing grid completed in 13.78 seconds. Both sides matched over 640,451 cases, including 1,511 guard cells, raw days 0–271, and effective days 0–180.

SHA-256 `5ca5e0fe6fec880363ef2cee5925c81750485594c59c8a97621377d19973f35d`.

## Season walk on the commit

The end-to-end walk produced 4,818 records and exited 1 on the 612 known warnings. The fresh JSON matched the decompressed `walk-h.json.gz` byte for byte. The walk output matched the captured command output in `walk-h.txt`, which also contains an appended summary. The 219 added rows in the matching JSON comprise 207 singles, three doubles on September 24–26, and nine post-finale skips. Both saved results were preserved.

JSON SHA-256 `1bd04d1eef0ba76efda4566171a4fb40fba8b2b1e411c8e37c832fe684b96e72`.

## Contract details

NPZ serialization drops a masked array's mask, and the loader normalizes the serialized horizon to an integer and edges to a list. The tests inject the masked table, float horizon, tuple edges, and table-like object into the decoded context while retaining the validated tail re-bound to the base SHA. F3 names four isolated clauses. The rank check mentioned in the round-2 notes remains covered by the exact shape condition when the rank check alone is removed.
