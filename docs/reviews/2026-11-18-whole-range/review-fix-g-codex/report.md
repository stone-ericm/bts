# Verdict

**BLOCK** — `d6141728` at checkout `7484b38d`. The shipped table and calendar agree with R61. The scheduler can send the correct pick and leave an authoritative record for a different pick or delivery status.

## Blocking findings

1. **A real commit can leave the previous pick or delivery status in the record.** On 2027-03-24 at 18:32 ET, streak 50, saver false, p=.83, an in-flight DM for batter 100 sees a concurrent recovery write `locked_unconfirmed`. When the original transport returns a confirmed ID, the pick file shows delivery and base writes `delivered`, while G preserves `locked_unconfirmed` byte for byte. A private-to-DM transition at the delivery entry point preserves `private_locked` after one confirmed send. In three more runs the scheduler DM'd batter 200 and G retained batter 100 with `private_locked`: a late refusal archived the earlier private pick, an earlier process left its record while its pick was moved aside, and a same-date rerun began with only the old record. Base recorded 200 and `delivered` in all three. See `transitions-{base,head}.log` and `src/bts/scheduler.py:923`. Where §3 requires the current commit's slot identities and delivery status. Preserve bytes on recovery or retry of the same authoritative commit. Let confirmed delivery supersede `locked_unconfirmed`, and reconcile or refuse a distinct commit before letting an old record stand for it. The blanket immutability sentence needs this qualification.
2. **A guarded cached pick can be delivered with a null primary in `decision.json`.** On the same date and clock at p=.96, batter 100 is saved with G's guard. At fallback all candidate games report Warmup. The actual refresh returns the saved pick and a `SelectionResult` with no new decision or candidates. The DM contains 100, and the record says `single` and `delivered` with the guard, but its `primary` is null. Where §3 requires the slot identities to match fresh and recovered commits. At `src/bts/scheduler.py:957`, use `DailyPick` slots when a selection supplied no decision/candidate for that pick, as the existing `selection=None` branch does. Reproduced in `transitions-head.log`, `cached_warmup`. This path exposed a pre-existing candidate fallback defect on a pick that G newly makes.
3. **Unreadable prior bytes break the new commit write.** With the same unguarded .83 pick and an existing `decision.json` containing one byte `ff`, both versions delivered one DM. Base atomically replaced the bad file and set `committed_pick_written`. G left the bad byte and flag false, and the scoreability reader raised `UnicodeDecodeError`. The new read at `src/bts/scheduler.py:924` raises before `write_decision`, and the delivery wrapper swallows the failure. Where §3 requires a fresh commit record and protects scoreable history. Treat a decode failure as an unreadable record at this commit check, then take the existing atomic writer path. Reproduced on both package archives in `transitions-{base,head}.log`.
4. **A loaded malformed donor cell can raise where base safely skipped.** At streak 10, 100 days, saver false, p=.8 with stored boundaries `[.7,.8]` and row `[single,3,skip]`, the real NPZ loader accepts the file. Base resolves skip. G raises `IndexError` at `src/bts/simulate/mdp.py:231` before considering the valid lower donor, which can abort the selection. Where §1 preserves lookup exceptions and searches for a raw single or double. Check stored donor codes for the two playable actions before converting, keeping the existing lookup error for an invalid queried cell. See `controls-{base,head}.log`.

## Edits

- Journal at `src/bts/scheduler.py:862`: saver input `'0'` selects saver index 0 but prints `saver input available`. Use `bool(int(sel.saver_available))` for the label so the exact replacement fragment is `saver input unavailable` in this case. The reproduction uses a direct non-Boolean input and establishes no R64 operator consequence in an ordinary state. Both Boolean forms and all action clauses rendered accurately.

## Notes

- The base for the author's resolver and bar grid executes old `strategy.py` in a temporary module, but that module imports raw lookup from the active package. It does not independently verify the old package's resolver dependencies. The separate archived-package grid here does.
- `test_restart_after_decision_before_flag_save` tests safe replay of the same pick, then explicitly expects the old bytes from another direct commit. It never tests a confirmed send after unconfirmed recovery or a replacement pick after refusal. That green check coexists with B1.
- The test named `test_decision_round_trip_and_shadow` invokes the skip predicate, while the adjacent census test actually invokes the shadow writer. The round-trip alone cannot establish shadow behavior.
- The census accepts a structurally plausible but impossible `bin=999, from_bin=998` on a five-bin table, suppressing `raw=skip recorded=double`, which halts without the guard. Require in-range indices and check actual bin and donor provenance where state and calendar permit. See `census-guard-evidence.log` and `scripts/audit/boundary_shadow_census.py:333`. This affects audit evidence.
- Five lifecycle fixtures include direct delivery and overlapping recovery, as well as full scheduler reruns from orphaned records. An ordinary same-day restart with both files still present stops at the pre-cascade scoreable check (`scheduler.py:1867`).

## 1. All cells and all dates

`grid.py` imported `bts` separately from archived `f83e0d0b/src` and checkout `src`. Per side it ran 398,924 raw lookups, 414,352 resolver calls, and 21,808 direct bars. Nineteen inputs include all edges, the float immediately below each upper edge, midpoints, 0, 1, -.2, 1.2, and NaN. Raw lookup, all shared unaffected fields, and all direct bars matched. Exactly 1,511 cells and 7,125 dated calls changed, including repeats at the 180-day cap. All guard keys and values had built-in types. CSV regenerated byte for byte (`77ec540475c0e4c4…`).

## 2. Live calls and raw callers

`entry_points.py` ran real `select_pick` via `run_and_pick`, `run_single_check`, both `run_day` fallback sites, `bts run`, `bts orchestrate`, `bts preview`, and the shadow selection. With captured transports and a synthetic slate, all made or displayed guarded single 100. Both scheduler refreshes delivered it and wrote the corresponding record. Dashboard rendered that batter and no skip banner. Manual CLI and shadow saved the pick as designed. Direct `effective_pick_bar` was 0.83 or lower and matched base throughout the grid. Cached Warmup is B2.

Raw calls in `scripts/` are `validation/compare_policies.py:13,19,20`, `oos_validation.py:31`, `audit_mdp_live_aggressiveness.py:125`, `gate_b_production_metric_rebaseline.py:290,299`, `evaluate_mdp_dd_guardrail.py:348,492`, and `audit/boundary_shadow_census.py:243`. There are no direct raw callers in `src/` at G. `simulate/exact.py`, `simulate/pooled_policy.py`, `validate/ope.py`, and `validate/ope_eval.py` use table cells directly. The raw lookup matched base across the grid, the callers retain the same raw action inputs, and the census continues to show raw comparisons.

## 3. Double clamps

`clamps_journal.py` ran streak 1, 29 days, saver false, bin 1. With doubles off or all candidates in one game, `policy_action=double`, `action=single`, record `single`, one saved leg, same guard. With both conditions satisfied, it selected the same primary/second IDs (100/101) in the same order as a control raw double on the same slate. JSON serialization of the pick and record passed.

## 4. Decision records

`commit_paths_retry.py` checked guarded `delivered` via DM and public, `private_locked`, and failed-refresh cached delivery. All retained guard, action, and the primary slot. The five existing-scoreable transitions and corrupt decision path are B1–B3. Author tests exercised direct and classification recovery of a durable attempt as `locked_unconfirmed` without any added send, delivered classification, end-of-day byte preservation, and null guard on skips. `readers_retry.py` exercised an actual EOD skip alongside guarded and unguarded plays. Archived base and rollback `619abda7` loaded new files, head loaded files actually written by base, and all prior decision bytes survived the read paths. Real `parse_decision`, `bts day-outcome`, `bts check-results`, skip predicate/writer and census accepted all three fixtures. The plays graded and stayed out of the skip shadow. The skip stayed ungraded and in it. Census raw/parity gave `skip/guarded_play_excluded`, `single/parity`, `skip/parity`; the five-row census fixture reduced skip membership 5→4 while retaining the raw diff and excluding it from the trigger count.

## 5. Refusals and unrelated plays

`controls.py` compared both packages at `streak+2*days` 56, 57, and 58, target/over-target, zero days, missing and unpaired artifacts, and heuristic fallback. No unexpected play appeared. A synthetic reach-table hole below the tail boundary left the season-best stop intact. Tied `[.8,.8]` correctly inherited the stored unreachable bin 1. Two-column exact-edge lookup still raised `IndexError`, empty boundaries used bin 0, and `None` raised `TypeError`. Numeric non-Booleans mapped to the same saver indices and only yielded CSV-equivalent changes, while `2` and `'false'` still raised. The malformed donor exception is B4.

## 6. Standing rows

`controls.py` interrupted the real orchestration immediately after policy lookup and found neither a saved pick nor a decision. `test_interrupt_before_save_and_atomic_replace` cut the real `os.replace`: no torn file and prior bytes intact. Corrupt regular and directory artifacts degraded to the same forced single/skip; the unwritten FIFO waited for the 150 ms deadline on both packages. At 23:59 and 00:01 ET the Dec 31 and Jan 1 counts were 0/0 and 269/180. Neither picked on an empty slate; January's synthetic slate gave the expected capped guard. The old automatic switch was refused after ET midnight. The focused restart tests exercised caught failure (attempt cleared, no record), interrupted attempt (durable marker, no resend), empty DM ID, a private interruption before the record, and the crash between decision write and flag save. Classification and EOD preserved the latter record exactly.

## 7. Journal

`clamps_journal.py` rendered the real journal for unchanged, guarded single with each saver Boolean, executable double, and each independent clamp. Compared both bins to the raw table, streak and days to the selected state, and pre/post-clamp actions to the actual selected legs. The unchanged case emitted no guard line. The non-Boolean display mismatch is the edit above.

## 8. Season walk

Ran end to end (exit 1), then `compare_walk_saved.py` compared all 4,599 keyed records with `walk-bc143131.json.gz` and matched each change to the CSV. 252 skip→single, 229 flagged in season and 23 preseason. All changed rows replaced the skip notice and skip/bar fields with the selected primary and correct `chosen_on` kind. Exactly 612 warnings remain, all in-season all-skip states. Zero other differences and identical boundary checks. Fresh JSON exactly matched decompressed `walk-g.json.gz`. Stdout matched the prefix of `walk-g.txt` preceding its appended comparison summary.

## 9. Tests and scratch changes

Ran all named files, calendar replay, and both census files. The sandbox run had 258 passes and one local socket bind denial. The denied dashboard test passed on the approved loopback-only rerun (259 distinct passes). No full fast suite.

Eleven scratch changes failed a test: lowest donor, guard already-playing cells, lose clamp guard, raw production lookup, guarded offline lookup, omit skip key, write guard on skip, lose saved guard recovery, NumPy bin, extra unguarded journal line, and omit the defaulted skip key. Three changes survived all 258 non-socket tests: add the guard in tail when a donor exists, always log saver unavailable, and drop guard evidence on a public commit. The synthetic tail hole, the saver-true journal case, and public captured delivery respectively failed on those exact scratch copies (`*-control-mutated.log`).

The only existing expectation changes were .96 skips→singles at 2027-09-17/streak 55 and 2027-09-21/streak 56. Both are authorized CSV cells; their .83 controls still play. Other diffs only adapt `_mdp_action_from` stubs to tuples and the one strategy stub to the guarded lookup, all authorized. Existing assertions those stubs drive were retained.

## 10. Other diff checks

The production changes stayed within the named files, both shipped policy artifacts remained byte-identical to base, and the v3 parser and delivery templates stayed in place. The extra failure paths are B3 and B4. The census evidence gap is noted above. No tracked file was edited during this review.

ran: archived package grids and readers, CSV regeneration, independent scheduler/CLI/record/clamp/journal/standing/census probes, full season walk, named pytest groups and loopback rerun, 14 isolated scratch changes; did not run: the shared wrapper, full fast suite, external network or a real contest account.
