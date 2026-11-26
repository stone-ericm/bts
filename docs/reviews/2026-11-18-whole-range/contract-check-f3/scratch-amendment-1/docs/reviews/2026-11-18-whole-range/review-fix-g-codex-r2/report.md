# Verdict

**BLOCK** on `8be73842` against `f83e0d0b`. The table changes exactly as specified. Two live scheduler paths write a scoreable record that disagrees with the pick they commit.

## Blocking findings

**B1. A guarded single records a second batter who was never committed.** On 2027-08-29, streak 1, 29 policy days, saver false, bin 1 at about 80.4%, the table skips and bin 0 supplies a double. With `allow_double=False` and batters 100 and 101 in different games, the DM and public post name only 100, and the private lock and pick file have one leg. All three fresh `decision.json` files say `action: single` and carry the correct guard, but name 101 in `double_down`. The saved-pick recovery of the same selection correctly writes `double_down: null`. A direct guarded single at streak 50 on 2027-03-24 showed the same phantom second leg.

`src/bts/scheduler.py:956` copies the ranked second candidate even when the strategy left the committed second slot empty. This violates Where item 3, “Test equality of guard, action and slot identities across fresh and recovered commits,” and the brief's wrong-leg criterion. Use the committed pick's slots for guarded records, including an empty second slot. `slots_and_journal.py` compares captured messages, persisted pick slots and records for all three modes. Its final assertion fails on HEAD. The author's fresh delivery test explicitly expects the ranked candidate in the committed slot at `tests/test_wr_g_guard.py:519`.

**B2. Final fallback delivers cached batter 100 and records declined batter 200.** On 2027-03-24 at 18:32 ET, streak 50 and saver used, save the guarded .96 single for batter 100. Restart with both policy files missing. The scheduled check gets no predictions. The final fallback gets a different slate led by batter 200 at .60, yielding a heuristic skip. The scheduler keeps and sends 100 once. The pick file retains its guard. The scoreable record says `single / delivered` but names 200, says `source: heuristic` and has `policy_guard: null`.

`src/bts/scheduler.py:2347` returns the cached pick with the declined refresh selection. At lines 947–956 the writer treats any decision on that selection as evidence for the committed pick. This violates Where item 3's requirement to take guard evidence from the current pick's decision or recover it from the saved pick. It also violates the standing row requiring the original guard whenever the saved pick actually commits. Use selection decision metadata only when the selection produced the pick being committed. For the cached path use that pick's slots and saved policy decision. `restarted_fallback.py` runs real `run_day`, the missing-artifact loader, selection, final fallback and record write with synthetic contest state and slate, a pinned clock, stubbed waits and captured transport. Its final assertion reports `(200, 'heuristic', None)`. Neither B1 nor B2 requires a bad table cell or a malformed pick file.

## Edits

**E1. Journal saver input for a negative index.** `src/bts/scheduler.py:862` treats `-2` as available. The shipped two-plane array uses plane 0 for that accepted index. `audit_controls.py journal` reproduces the mismatch at streak 50 with 180 policy days. The exact replacement line for the fixture is “POLICY GUARD — At streak 50 with 180 policy days left and saver input unavailable, the table skips bin 4. Bin 3 supplies single. Selected action is single.” Amend the prescribed `bool(int(...))` to render the resolved plane while preserving the lookup's accepted indices. Production state readers supply Booleans. The line explains the selection and provides no instruction that could cause an R64 outcome.

## Notes

- The resolver test at `tests/test_wr_g_guard.py:119` executes base `strategy.py` in HEAD's process, so its lazy import uses HEAD's raw lookup. The separate raw lookup test does load the base function itself. The independent grid used complete packages in separate interpreters.
- The normal guarded Warmup fallback sends and records the correct batter and guard. Its record still uses the no-decision selection's default `streak: 0` and `saver_available: null` alongside a saved pick chosen at 50 with saver false. A matching streak-50 ledger makes the census halt at Gate A before guard parity. Amendment 2 item 2 explicitly preserves those other fields. `commit-head.log` captures the conflict with the census acceptance requirement. The census's fixed 2026 calendar is also unchanged.
- In a 2026 census fixture with raw skip and a valid bin-4 override, changing `from_bin` from 3 to in-range bin 0, which also skips, still bypasses Gate B. `audit_controls.py` reproduces this evidence gap. The amendment's two explicit checks for bin bounds and raw skip pass. A known-state check of the actual bin and donor would also reject this fabricated provenance.

## 1. Table and dates

`table_grid.py` printed `bts.__file__` from the separate archived base and checkout packages. Per side it ran 398,924 raw lookups, 414,352 resolver calls and 21,808 direct bars. The 19 inputs include all boundaries, the float immediately below each upper edge, interiors, 0, 1, outside [0, 1] and NaN. Exactly 1,511 cells and 7,125 dated calls changed, with the specified action and guard every time. All raw results, unaffected resolver fields and bars agreed. Guard key and value types were checked with `type()`. Guarded decisions, actual pick files and commit records serialized to JSON. The regenerated CSV matched byte for byte (`77ec540475c0e4c43…`).

## 2. Production and offline calls

`entry_points.py` reran the archived live fixtures, using HEAD's imports and captured transports. `select_pick`, `run_and_pick`, the scheduler check, both full-day fallback sites, `bts run`, `bts orchestrate`, `bts preview`, shadow selection and the dashboard picked or displayed guarded single 100. The scheduler delivered and wrote the correct action and primary for the ordinary refresh fixtures. The direct bar at streak 50 on March 24 was 0.8252474190317034. Record exceptions are B1 and B2.

There are no remaining direct raw calls in `src/`. In `scripts/` they are `validation/compare_policies.py:13,19,20`, `oos_validation.py:31`, `audit_mdp_live_aggressiveness.py:125`, `gate_b_production_metric_rebaseline.py:290,299`, `evaluate_mdp_dd_guardrail.py:348,492` and `audit/boundary_shadow_census.py:243`. Their raw call inputs are unchanged, and the exhaustive raw lookup matched the base. `simulate/exact.py`, `simulate/pooled_policy.py`, `validate/ope.py` and `validate/ope_eval.py` use table cells directly. The census's new attribution happens after its raw lookup.

## 3. Double clamps

`slots_and_journal.py` exercised each clamp independently. Both kept `policy_action=double` and the original guard, executed single, saved one leg and said single in the journal and record. The disabled-double path also records a phantom leg (B1). The same-game clamp leaves the record's second slot empty. The executable guarded double chose 100 and 101 in that order, identical to a raw-double control with the same predictions and probability.

## 4. Commit and readers

`commit_lifecycle.py` ran the same five prior-commit transitions and the one-byte `ff` case on the full base and HEAD packages. A confirmed send after concurrent unconfirmed recovery and a private-to-DM commit both ended `delivered`. Refusal, orphan and same-date rerun all wrote the replacement batter. The undecodable record was replaced and the flag set. The inspected writer diff contained only the new parameter and its pass-through.

Failed refreshes and Warmup recovered the right committed slots through DM, public and private. A no-decision selection with unrelated candidate rows borrowed neither row. A removed second leg stayed absent and a replaced second leg recorded the replacement. The pick reader rejected malformed and missing required slot fields before a commit. The same Warmup slot recovery worked on an unguarded saved pick. A restarted heuristic refresh for a different pick caused B2.

`restart_records.py` exercised direct and real scheduler classification for durable attempts (`locked_unconfirmed`, no second send), delivered classification and the crash before the flag save. The pre-cascade check, classification and EOD preserved existing scoreable bytes. Actual EOD skips wrote `policy_guard: null`.

`record_readers.py` exercised a guarded play, unguarded play and skip through `parse_decision`, `day-outcome`, `check-results`, the skip predicate and the shadow writer. The plays graded and the skip alone entered the shadow. Base and rollback `619abda7` read HEAD's files. HEAD read actual files written by base. Census raw/parity returned `skip/guarded_play_excluded`, `single/parity`, and `skip/parity`. The three-record fixture moved skip membership from 2 to 1, and the five-record test moved it from 5 to 4 while retaining the guarded row's raw diff and excluding it from the trigger count. The Warmup state caveat is in Notes.

## 5. Refusals and other plays

`controls.py` produced identical base and HEAD observations outside the intended guard cases. Reachability 56, 57 and 58, target or above, zero days, heuristic and ordinary clamps all matched. A synthetic lower playing bin in a tail state still skipped. Missing, corrupt and directory artifacts used the same forced-tail actions. Tied edges inherited the unreachable raw donor as specified, and too few columns retained `IndexError` at the exact edge. Missing boundaries retained `TypeError`. Non-Boolean integer aliases selected their equivalent table plane and invalid values failed as before. An invalid donor code was passed over, while a queried invalid code retained its exception.

## 6. Standing rows

The real lookup was interrupted before a pick or decision existed. The focused test cut atomic replacement both with and without a prior pick file. No torn pick or new record appeared. The regular corrupt file and directory retained their loader errors, and the base loader's FIFO blocked until the 150 ms deadline on both packages. At 23:59 and 00:01 ET the year-edge counts were 0/0 and 269/180. Empty slates selected nothing, the synthetic January state guarded as expected, and the old automatic switch was refused after ET midnight.

A caught send failure cleared the marker and wrote no record. Both restart routes preserved the guard and sent no duplicate after an interrupted attempt. The focused tests also exercised the unchanged empty DM ID behavior and a private interruption before a record. The real restart after writing a decision restored the flag and preserved the bytes. Commands were timeout-bound.

## 7. Journal

`slots_and_journal.py` rendered the real journal for an unchanged decision, guarded singles with both saver values, the executable double and both independent clamps. It verified the streak, effective days including the 180-day cap, requested and donor bins, saver label, and both action words against the real selection and raw cells. The unchanged decision printed no guard line. Boolean and 0/1 cases were exact. E1 covers the additional negative-index case.

## 8. Season walk

The full walk exited 1 as expected. `walk_comparison_run.py` compared each of 4,599 records to `walk-bc143131.json.gz` and matched exactly 252 skip-to-single changes to the CSV (229 formerly flagged and 23 preseason). Each gained its primary and chosen-on stamp and lost its skip notice and bars. All 612 remaining warnings are on in-season all-skip rows. Zero other records or boundary results changed. Decompressed JSON matches `walk-g.json.gz` byte for byte. Combined streams match `walk-g.txt` before its appended comparison summary, including the sandbox's `nice` warning.

## 9. Tests and scratch changes

All 464 named tests passed across two runs. The first run passed 463 and the sandbox denied the dashboard's localhost socket bind. That test passed on its approved localhost-only rerun.

The suite caught 19 of 20 isolated changes. Caught were lowest donor, guard in tail, guard on an already-playing cell, drop guard on a clamp, raw production lookup, guarded offline lookup, omit the skip key, persist a guard on a skip, lose saved guard recovery, NumPy bin type, guard journal for an unchanged decision, restore the early return, take selection primary without a decision, convert a malformed donor, always print saver unavailable, accept an out-of-range census bin, accept a census guard over a raw play, drop the public guard, and invent a second slot on a saved single.

**Survived:** exclude an unguarded play over a raw skip while still rejecting malformed guard dictionaries. All 463 named non-socket tests passed on that scratch copy. `audit_controls.py` verifies that HEAD halts on the fixture. `census_mutation_control.py` verifies that the scratch copy excludes it. The focused file also stays green for B1 and B2. Its fresh single assertion expects the runner-up in `double_down`, and its fallback cases omit a declined selection belonging to a different batter.

Existing assertion changes were the two .96 calendar pins (2027-09-17 at 55 and 2027-09-21 at 56, both CSV rows with the .83 controls intact) and the saver test. Its new 2027-08-20, streak 9, bin 0, 38-day fixture is outside the CSV for both saver values and still runs the two state reads, comparing skip with single as Amendment 3 directs. The tuple stubs and guarded lookup stub preserve their callers' original assertions. These changes are authorized.

## 10. Other changes

Both shipped artifacts are byte-identical to the base. The v3 parser and delivery templates are unchanged. Production and test changes stayed within the contract's named paths. No tracked files were modified during this review.

ran: full archived-package grids and lifecycle comparisons, CSV regeneration, live entry points, slots and journal, restart/refusal/reader/census controls, full season walk, all named pytest files, one localhost-only rerun, 20 isolated scratch changes; did not run: the full fast suite, the shared wrapper, external network calls or a real contest account.
