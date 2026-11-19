# Whole-range C3 round-two handover

Commit `aa61bb43eeb20325500434eecc9a11b64607635b` on `fix/wr-c3` is the single new commit on top of `c70effe9`. No push.

## Diff stats

Against `c70effe9`:

```text
 src/bts/picks.py              |  84 ++++++--
 tests/test_wr_c3_corpus.py    | 449 ++++++++++++++++++++++++++++++++++++++++++
 tests/test_wr_c3_delivery.py  | 160 +++++++++++++++
 tests/test_wr_c3_reconcile.py | 121 ++++++++++--
 4 files changed, 782 insertions(+), 32 deletions(-)
```

Against `c2039541`:

```text
 src/bts/cli.py                          |  38 ++-
 src/bts/picks.py                        | 291 ++++++++++++++++-----
 src/bts/reconcile_receipt.py            |   3 +-
 tests/test_streak_replay.py             |  15 +-
 tests/test_streak_season_cs2.py         |   9 +-
 tests/test_streak_season_cs2b_review.py |   6 +-
 tests/test_wr_c3_corpus.py              | 449 ++++++++++++++++++++++++++++++++
 tests/test_wr_c3_delivery.py            | 160 ++++++++++++
 tests/test_wr_c3_reconcile.py           | 432 ++++++++++++++++++++++++++++++
 9 files changed, 1316 insertions(+), 87 deletions(-)
```

## Rules and compatibility

- Rule 1. [picks.py:868](src/bts/picks.py#L868) takes one guarded pick snapshot. [The terminal fold](src/bts/picks.py#L964) applies a miss or void from its result alone and uses the legacy `load_pick` decoder and hit increment for a hit. This applies through the locked ET replay date. Terminal results take precedence over delivery, scheduler and decision evidence, including a bad decision beside an otherwise foldable terminal pick. Correction reads retain their round-one validation.
- Rule 2. [picks.py:929](src/bts/picks.py#L929) skips an unresolved past pick only after validation and the applied-date check. A non-scoreable decision or, with no decision, a valid `chosen_on.required=false` stamp must establish eligibility. Each of the six delivery and attempt fields can veto a skip. A scheduler lock or unreadable scheduler state also vetoes it. The failure reason follows the date into the existing one-line CLI warning and receipt. The test table is in [test_wr_c3_reconcile.py:383](tests/test_wr_c3_reconcile.py#L383).
- A pre-5a refused preview without a preview stamp or decision still blocks. The operator can remove or archive the canonical pick named in the warning. The receipt's `outcome` for a blocked replay with a corrupt streak remains `completed`, with replay state `unavailable`.

## Corpus

[test_wr_c3_corpus.py:436](tests/test_wr_c3_corpus.py#L436) runs all 630 histories against observations captured from archived `c2039541` source. The independent run of the original reviewer generator also gave 560 equal observations and 70 differences.

| Reviewer category | Differences | Observed |
| --- | ---: | --- |
| (a) Expected replay changes | 21 | Refused preview, skip leftover, and current-day foreign-season marker, seven mixes each. Ambiguous past unresolved and suspended files and past foreign-marker files without skip evidence hold as on the baseline. |
| (b) Named blockers or baseline raises | 49 | Five malformed shapes formerly raised and now return a dated blocker, 35 histories. Fourteen nonterminal histories remain blocked under the checked C3 rules, seven for a missing pick with a scoreable decision and seven for an unknown current-day result. The baseline ignored these two nonterminal cases. |
| (c) Historical compatibility | 0 | All 35 former differences now match the baseline, including early regenerated misses and the four numeric-string ID shapes. |

All baseline-foldable terminal histories had identical streak and saver results. The synthetic 2026 legacy control preserved its `(13, False)` fold.

## Delivery counterexamples

The [round-one comparison](.codex-review/wr-c/round1/delivery-followup.log) showed that all four baseline runs returned unavailable and held `(10, True)` byte for byte. Round one corrected the later hit to a miss, replayed past the entered day, and wrote `(0, True)` in each case. [The round-two tests](tests/test_wr_c3_delivery.py#L117) keep that correction and verify a dated warning for `2027-04-18`, unavailable receipt state, retained streak display, and byte-identical `streak.json`.

| Counterexample | Round-two warning reason |
| --- | --- |
| DM accepted, acknowledgement lost, attempt marker cleared | `no decision or preview stamp` |
| DM persisted, scoreable decision write failed, EOD wrote a skip | `skip record contradicts a delivery` |
| Private delivery interrupted after scheduler lock and before decision | `scheduler state marks the pick locked` |
| Manual post accepted, interrupted before writing post marker | `no decision or preview stamp` |

## Fixture and hostile-input coverage

- No changes to the three pre-existing test files amended in round one. In the round-one C3 test file, invalid decision cases now use pending picks so they still test decision validation under rule 1. The interrupted rebuild fixture gives its unrelated past pending pick an explicit skip decision under rule 2. Both supported refused-preview evidence paths independently rebuild corrupt `streak.json` after each correction direction.
- The state table covers all six delivery signals, missing and malformed preview stamps, locks, conflicting decisions, corrupt and non-regular scheduler state, and a matching applied-date marker. Existing guarded pick and decision FIFO and corruption tests, special streak target, and corrupt blocked streak controls still pass.
- The real delivery probes cover interrupted and unconfirmed sends. The interrupted rebuild case keeps the correction and refolds on retry. The scorer-crash test verifies tagged and tagless markers across midnight, then retries when grading finishes. The New Year test verifies the locked ET replay date and ignores a foreign-season marker. The 08:00 correction cutoff and pp1 tests passed.

## Runs

- RED on `c70effe9`. `.codex-review/wr-c/r2-red-verified.log` recorded 341 failures and 344 passes, including 322 corpus regressions and 15 skip-evidence failures. Its four delivery cases reached the receipt assertion but shared a fixture parent. After isolating the picks directories, `.codex-review/wr-c/r2-red-delivery.log` confirmed all four RED at the missing warning. Earlier attempts exposed fixture import and fixture constant errors that were corrected before the committed run.
- Focused on committed `aa61bb43`. `.codex-review/wr-c/r2-focused.log` recorded 1,180 passes, including all 685 C3 cases, LightGBM optional and all web saver cases in one process.
- Single unsandboxed fast suite on committed `aa61bb43`. `.codex-review/wr-c/r2-fast.log` recorded 10,019 passes, 37 failures, 16 errors, 25 skips, six deselections and one xfail in 26:12. The 53 failure and error IDs matched `~/bts-runs/gate-failed.txt` exactly, with zero extra or missing IDs.
