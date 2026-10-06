## Verdict

**SIGN WITH EDITS — P2 only, conditional on applying the exact patch specified below.** Reviewed detached HEAD `156f96f06c61bf37b986bbcaa62d2464e62180d5`, including its commit message and complete three-file diff. Authority is exposure-register §C row `C1-r2-p2-review-r3` (`docs/audit/2026-09-22-exposure-register.md:94`) and the final-round prompt. A, B and C/P1 remain settled and were not reopened.

D1 and D6 close their round-3 counterexamples. D3 closes the three original negatives, but its new ID-presence check does not establish that the returned grade belongs to that ID. Four real-producer synthetic cases still publish false coverage. The supplied exact, evidence-only edit closes those cases without changing grading, requests or application writes. Its four regression cases are red on this HEAD and green with the edit; the allowed 114-test suite passes both before and after the edit is substituted in memory. This is sufficient for a verbatim-edit signature, not an unconditional SIGN of the current tree.

I made no tracked edits. All test/probe fixtures and evidence are review-owned inside this checkout; network/DNS/client and cookie-loading leaves were denied before producer/test imports, then only synthetic client responses were supplied. Guard counts are zero. No runtime `data/`, credentials, configuration, `.env`, memory, external notes, remote services or live account were read. The signature permits P2 to proceed toward D7 after these exact edits; it does not approve deployment or establish installed receipts/current-defence re-certification. No further review round is requested.

## D1, D3, D6

| Item | Status at reviewed HEAD | Evidence |
|---|---|---|
| D1 | **Closed** for the reported lost-observation/prefix failures | `reconcile_receipt.py:245–248,272–277` applies every non-write degradation on the date to every slot. Independent `D1_lost_primary` controls and combined late/clock-step retest both uncover the primary and DD. |
| D3 | **Partially closed; closed with the exact edit** | Payload-owned integer game identity and agreement for hit/miss/suspended void now work; name-only absence is rejected. Remaining presence-versus-grade coupling is the new finding below. |
| D6 | **Closed** for the reported attribution and unavailable-evidence failures | `_guarded`, `reconcile_receipt.py:54–56`, binds production write-hook failures to their explicit date; `:284–302` distinguishes failed/missing evidence from the retained actual-save-exception case. |

**D1.** The earlier-primary hook fails at **08:00:01**, the clock steps back and the DD's correctly bound response arrives at **07:59:59**. The DD still has `basis: final_feed` and can have `clock_regression: false` because the earlier event was lost, but both slots now have `covered: false` and the date's observation degradation. A surviving maximum is no longer mistaken for a complete response prefix. The ordinary pre-cutoff source-loss control also uncovers both slots. `tests/test_reconcile_receipt.py:332,462,521` covers lost-feed, day-level hook and combined earlier-leg failure; these pass in the allowed suite. Write-hook failures intentionally do not remove separately established observation coverage.

**D3 original cases.** `_qualify` no longer obtains actual game identity from the URL (`reconcile_receipt.py:228,235–239`). It requires `type(actual) is int`, an ID occurrence, Final state and grade agreement; selected-game agreement precedes the suspended-void success return. Independent `D3_game_matrix` runs hit, miss and suspended void against own game `822934`, missing identity, wrong game `999`, and string/float/bool identities: 18 actual-producer cases. Correct-game integer positives are covered; wrong-game integer cases are `fallback_other_game`, uncovered; missing/noninteger identities are unqualified with no manufactured actual game. The name-only player-123 case is unqualified/uncovered. These are receipts published and rediscovered through the real P2 publisher, not planted receipt fixtures. The round-3 suite cases at `tests/test_reconcile_receipt.py:492–518` also pass.

The ID membership change is still insufficient to meet round 3's recommendation to qualify the decisive batter independently of the legacy name fallback. `_has_batter_id` sees an ID anywhere, while `_qualify` still calls `grade_pick_in_feed` **with the selected name**. It does not compare that grade with the grade established by the selected ID alone. The exact edit adds that comparison at publication; business grading is untouched.

**D6.** Independent `D6_write_hook` cases fail each actual production hook name (`write`, `write_intent`, `write_done`) with the default eight-date lookback. Every degradation is attributed to index **0 / 2026-08-20**, with `slot: null`, rather than stale index **7 / 2026-08-13**. Each affected date has `write.state: write_evidence_unavailable`, including the no-save write-decision hook case. Successful intent/done-failure corrections persist **miss**, complete the run, and retain their independently valid observation coverage. The deliberately missing completion event in a completed run is also unavailable evidence. The actual `save_pick` exception control still propagates the original OSError, leaves persisted **hit**, records `outcome: raised`, and reports `write_not_completed`, intended correction/completed false. Supplied intent/done tests at `tests/test_reconcile_receipt.py:546–561` pass. All current production write-hook calls pass the explicit date positionally.

Raw evidence is in `probe-results-r4.jsonl` and `probe-results-r4-candidate.jsonl` (37 labeled records each), produced by `probes-r4.py`. Labels identify each case and retain actual slot/write fields. The candidate file refers only to the exact `_qualify` substitution described below; it is not a claim that the tracked source was changed.

## Regression checks

**Independent execution:**

- Reviewed HEAD: **114 passed**, log `pytest-r4.log`, XML `pytest-r4.xml`, `R4_OFFLINE_GUARD_ATTEMPTS=0`.
- Exact candidate `_qualify` substituted in memory: **114 passed**, log `probe-run-r4-candidate.log`, XML `pytest-r4-candidate.xml`, guard attempts **0**. There is one harness-only pytest warning because the network-guard module was imported before pytest could rewrite its assertions.
- Four proposed regression cases executed directly through the real test/producer helpers: **4/4 red on HEAD; 4/4 green with the exact candidate** (`D3_verbatim_test` records). The direct cases are additional to, not part of, the 114-test collection.
- `git apply --check .codex-review/c1-r2-producers/c1-p2-verbatim-edits-r4.patch` succeeds. This checks applicability without applying it. Tracked status remains clean.

The allowed collection was exactly `tests/test_reconcile_receipt.py tests/test_receipt_io.py tests/test_reconcile_cutoff.py tests/test_scoring_lock.py tests/test_entry_receipt.py`. The run added the review-owned `-p r4_offline_guard`, disabled cacheprovider/plugin autoload, used an in-checkout basetemp, and set `UV_CACHE_DIR=/tmp/uv-cache`, offline/no-sync UV, bytecode suppression and ET. Code-substitution probes used **`python -B` with distinct fresh `PYTHONPYCACHEPREFIX` values**. The author's supplied **6/6 mutant kills and 3768-test full-fast pass were not independently rerun**; they are not substituted for the evidence above.

**D2 retained.** The code for ordered response instants, regression detection and prefix high water is unchanged by this fix (`reconcile_receipt.py:256–277`). The suite's late-primary/DD, late-selected-feed/on-time-fallback, single late-response step-back and all-pre-cutoff regression controls pass (`test_reconcile_receipt.py:304,350,366,471`). The combined D1 retest establishes that a lost earlier response no longer defeats the clock proof by leaving the later slot covered.

**D4 retained.** Intent precedes `save_pick`, completion follows its return, fresh locked selection is retained separately, and the changed-selection diagnostic remains visible (`picks.py:1232–1242`, `reconcile_receipt.py:163–173,298–299`). The actual-save failure and selection-change suite cases pass (`tests/test_reconcile_receipt.py:407,421`); the independent failed-save control confirms original exception/persisted-state behavior. D6 no longer labels a caught observer failure as that business-write failure.

**D5 retained.** The fix does not touch `picks.py`, `cli.py` or `receipt_io.py`; their diff from `3cbe960` to reviewed HEAD is empty. The live source hook still records raw response references and return instants, while hashing/parsing/qualification run after production decisions and writes. Independent with/without-receipt runs have equal ordered request lists, correction lists, actual pick bytes and streak fields other than the existing `updated` timestamp (`D5_baseline`). The no-receipt path still calls the resolver with exactly two positional arguments and no keyword (`D5_no_receipt_call`); the source-observer token is reset. The suite's delayed-hash/cutoff control passes (`tests/test_reconcile_receipt.py:441`). This closes the identified heavy-work interference; it does not claim zero instruction overhead.

**Shared publication retained.** `receipt_io.py` is unchanged. The allowed tests verify durable receipt-before-seal ordering, tombstone sync, refusal before/after rename, failed withdrawal plus failed tombstone with no seal, and seal failure. Discovery still requires sealed and untombstoned files. P2's CLI publication-failure tests preserve business exit/output while reporting unavailable evidence (`tests/test_reconcile_receipt.py:284`). The entry-receipt suite is a permitted shared-publication regression check, not a reopening of settled P1.

The exact candidate adds a second **ID-only, local, publication-time** call to the existing grader on already retained feed bytes. It adds no client request or authenticated access, leaves the observed result and stored pick untouched, and can only withdraw the identified unsupported coverage. A normal valid selected-ID hit/miss/void remains covered in the matrix and suite. Failed fetches, unchanged observations, no-pick/ungraded/past-cutoff dates, schedule voids, pending/not-attempted slots and original cutoff/locking tests also pass.

The existing re-certification disposition remains required at the eventual deploy candidate: the affected current-defence patches listed in `docs/ops/reconcile-receipt-v1.md` must be rerun with frozen tooling before claiming them current. This review neither ran those certificates nor modified their tooling, and does not report installed-producer or operational acceptance.

## New findings

**D3-R4 — High: an ID present elsewhere in the decisive feed can launder another player's name-derived grade into covered selected-ID evidence.**

At `reconcile_receipt.py:229–235`, `grade` is calculated with the selected name and `by_id` is calculated independently. `_boxscore_hit` searches away **ID then name**, followed by home **ID then name** (`picks.py:1029–1041`). It can return an away player's name-match result before reaching the selected ID on the home side. For suspended games, `_play_matches_batter` accepts either ID or name, so another ID's same-name PA can contribute to the selected result (`picks.py:976–981,1003–1020`). Checking ID presence elsewhere does not prevent either behavior.

Each following case used selected game **822934**, selected ID **802415**, another ID **123**, valid Final status, pre-cutoff source times and no degraded hooks. The real producer published and rediscovered a slot with `basis: final_feed`, `covered: true` at this HEAD:

| Fixture | Legacy returned/persisted slot grade | Grade established by selected ID alone | Evidence label |
|---|---|---|---|
| Away name-match has a hit; home selected ID has zero hits | hit | miss | `D3_id_presence_wrong_normal_grade`, `[1,0]` |
| Away name-match has zero hits; home selected ID has a hit | miss | hit | Same label, `[0,1]` |
| Suspended game: selected ID's pre-suspension PA is an out; different ID with saved name has a pre-suspension hit | hit | miss | `D3_id_presence_wrong_suspended_grade`, `miss` |
| Suspended game: selected ID appears only after resumption; different ID with saved name has a pre-suspension hit | hit | void | Same label, `void` |

These are synthetic counterexamples, not measured production events or permission to fix the production grader. They demonstrate that receipt qualification still accepts a name-derived result contradicted by the selected-ID evidence in the same bytes. No game-identity, publication, clock or request failure is needed.

**Disposition:** require the decisive feed's ID-only grade to equal the already observed slot result, retaining the existing legacy-grade test to identify the decisive response. ID-only absence or disagreement is `unqualified`, uncovered. This is the exact patch below; four red/green tests verify both refusal and unchanged business result. No other new material P2 finding was identified in this review.

## Required edits

Apply **all three hunks/files** in this exact review artifact:

`/Users/eric/projects/bts-c1-r2review/.codex-review/c1-r2-producers/c1-p2-verbatim-edits-r4.patch`

SHA-256: **`be3b9ca9364c9d7567129a699f1fab5ee1e2675357cd564d0e95684b8075c948`**.

The patch changes only `src/bts/reconcile_receipt.py`, appends the four parameterized regression cases to `tests/test_reconcile_receipt.py`, and adds the precise ID-bound grade rule to `docs/ops/reconcile-receipt-v1.md`. The appended tests' exact source is also retained in `proposed-tests-r4.py`. The only implementation change is:

```diff
                 grade = grade_pick_in_feed(resp, slot["batter_id"], slot["batter_name"])
                 by_id = ReconcileReceipt._has_batter_id(resp, slot["batter_id"])
+                grade_by_id = grade_pick_in_feed(resp, slot["batter_id"])
             except (KeyError, TypeError, AttributeError):
                 continue
             if grade is None:
                 continue                         # the batter is not in this feed
-            if type(actual) is not int or not by_id or not final or grade != slot["result"]:
+            if (type(actual) is not int or not by_id or not final or grade != slot["result"]
+                    or grade_by_id != slot["result"]):
                 return "unqualified", actual if type(actual) is int else None
```

The method's original legacy-grade call remains, so the receipt does not skip an unqualified decisive response and promote a later one. The added call omits the name and operates only at publication. The exact candidate was tested by compiling/substituting only this `_qualify` body in a fresh process; tracked files stayed untouched. The full patch's applicability was checked separately.

This **SIGN WITH EDITS** is bound to that patch, including its tests and contract text. It does not require another design or code round for those verbatim edits. Application, commit, certificate reruns and D7 deployment remain the authorized executor/owner's subsequent steps; none was performed here. If these edits are not applied, the reviewed HEAD still has the demonstrated D3 false coverage and is not signed as-is.
