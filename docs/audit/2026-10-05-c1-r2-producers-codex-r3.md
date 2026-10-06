## Verdicts

C: **SIGN WITH EDITS.** Apply the C1 source, schema and regression-test edits in `c1-verbatim-edits-r3.patch` verbatim. The exact edit and its red/green evidence are below. Unedited HEAD is not signed.
D: **BLOCK.** Close incomplete-prefix coverage and the remaining actual-identity gaps, and distinguish unavailable write-completion evidence from an unsuccessful save. This is P2's last round; the item goes to Eric.
B: code unchanged **yes**.

Reviewed detached HEAD `42277b55028c3643d9220885c329308290ecb38c`, including the messages/diffs of `6047f2f`, `cacf65b` and `42277b5`, the recorded P1 third-round ruling, revised schemas and registration's producer contracts. This report follows the current prompt's explicit allowance for a verbatim SIGN WITH EDITS on P1. It requests no fourth P1 round or third P2 round. D7, re-certification on the final candidate, installed-producer commissioning and A8 remain separate; nothing here approves a deployment, configuration change, activation or play.

The exact permitted command passed **244 tests in 9.71s** (output: `pytest-r3.log`):

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_entry_receipt.py tests/test_receipt_io.py tests/test_reconcile_receipt.py tests/test_entry_intent.py tests/test_private_mode_transport.py tests/test_cli_integration.py tests/test_reconcile_cutoff.py tests/test_scoring_lock.py tests/test_daily_decision.py tests/test_daily_decision_v2.py tests/test_daily_decision_v3.py -p no:cacheprovider
```

Independent synthetic evidence is retained in `probes-r3.py` / `probe-results-r3.jsonl` and `probes-r3-retest.py` / `probe-results-r3-retest.jsonl`, with their run logs. All client/network leaves were denied before task-module imports; the reused cron fixture's child was guarded too. Probes used synthetic temporary inputs, fake HTTP/auth/DM leaves and throwaway crontab shims. No real data/config/credential file, network/DNS, SSH, gh, real crontab, external memory or notes was accessed. No tracked source was edited, including for the proposed fix: its exact function was compiled from the retained proposal and temporarily substituted in memory. Probes ran with `python -B` and fresh `PYTHONPYCACHEPREFIX` paths. The prior reports and their results were preserved. The author's full fast-suite and mutant counts were not independently reproduced as a whole and do not clear the counterexamples below.

## Item C

**C1 — partially closed at HEAD; the remainder is closed by the required verbatim edit. High.** `src/bts/entry_receipt.py:99`.

The season inventory closes the original latest-file false green. Every matching filename-year capture is consumed and individually hashed; conflicting non-null feed/round pairs become `unit_unverified`. The round-2 two-feed counterexample now returns `all_confirmed: false` and names both consumed files (`C_conflicting_captures`). The suite covers conflicting history, another season and the same-round null-feed control.

The null rule is correct **for game identity**, but too broad for round identity. `scripts/audit/season_ledger/contest.py:70` independently retains every non-null round even when the feed is null; `:126` rejects a contradictory round set. `mining87/run.py:252` ignores null feeds when constructing its feed set, so a pre-binding null does not create a second game. It does not establish that a known contradictory round may be discarded from a round-qualified receipt.

Reproduced through the entry producer: capture unit 1 with `(feedId: null, roundId: 99)`, then `(feedId: 1, roundId: 7)`. Both files are recorded, yet the selected round-7/game-1 slot is confirmed. The earlier audit retains rounds `{7, 99}` (`C_null_feed_round`). The same-round control `(null, 7)` then `(1, 7)` correctly confirms and must keep doing so. Null means no asserted game; it does not erase an asserted round. Content deduplication remains acceptable for the mapping lookup, without treating the last stored-content timestamp as recent cron/account observation proof.

**Required verbatim edit:** apply the companion patch `c1-verbatim-edits-r3.patch`, SHA-256 `1f161e630c89f25d886373abace27457c9034e21bb1d66ff218064e366ce192e`. `git apply --check` succeeds at the pinned HEAD. It contains precisely these three changes:

1. In `src/bts/entry_receipt.py`, preserve all non-null rounds independently of feed presence:

```diff
@@
-    - **A null feedId** names no game: it contributes no identity, and contradicts none.
+    - **A null feedId** contributes no game identity. Its non-null roundId still participates in round
+      consistency, so a pre-binding capture cannot erase a round contradiction.
@@
     seen: dict = {}
+    rounds_seen: dict = {}
     files = []
@@
             seen.setdefault(u["id"], set())
+            rounds_seen.setdefault(u["id"], set())
+            if rnd is not None:
+                rounds_seen[u["id"]].add(rnd)
             if feed is not None:
                 seen[u["id"]].add((feed, rnd))
@@
     units = {uid: (next(iter(ids)) if len(ids) == 1 else (None, None) if not ids else None)
              for uid, ids in seen.items()}
+    for uid, rounds in rounds_seen.items():
+        if len(rounds) > 1:
+            units[uid] = None
     return units, {"files": files, "inventory_sha256": payload_sha256([f["sha256"] for f in files])}
```

2. In the schema's `qualification` row (`docs/ops/pick-entry-receipt-v1.md`), replace exactly:

> a null `feedId` names no game and contradicts nothing

with:

> a null `feedId` contributes no game identity, but any non-null `roundId` still participates in round-consistency checks

3. Append this exact test to `tests/test_entry_receipt.py`:

```python
def test_c1_a_null_feed_still_preserves_a_round_contradiction(monkeypatch, tmp_path):
    H._setup(monkeypatch, pending=[H._pending(100)], crosswalk={100: 1})
    units_snapshot(tmp_path, [{"id": 1, "feedId": None, "roundId": 99}], name="20260601T120000Z")
    units_snapshot(tmp_path, [{"id": 1, "feedId": 1, "roundId": 7}], name="20260612T120000Z")
    picks = picks_dir(tmp_path, batter_id=1)
    H._run(picks, IN_WINDOW)
    r = only(tmp_path)
    assert r["verifier"]["reason"] == "match"
    assert r["qualification"]["slots"][0]["state"] == "unit_unverified"
    assert r["qualification"]["all_confirmed"] is False
    assert len(r["observation"]["sources"]["units"]["files"]) == 2
```

That exact test fails against HEAD and passes with the exact proposed `load_units` function substituted. Four actual existing producer tests also pass with the proposal: conflicting feed history, same-round null/other-season handling, duplicate selection and the result-string domain (`C_verbatim_edit_red_green`, `C_verbatim_edit_control`). The proposal leaves the original verifier, requests, decision/DM/marker path and receipt inventory unchanged. This is a conditional signature for this precise edit, not permission for additional repairs or an assertion that the whole suite was run against physically patched source.

**C3 — closed.** `src/bts/entry_receipt.py:40`, `:69`; `tests/test_entry_receipt.py:519`.

The emitted row result is restricted to hit/not_hit/miss/void/null; other strings are nulled and named in `mistyped`. The synthetic token-in-result counterexample no longer appears in serialized output (`C_known_string_field`). The source hash remains opaque, and existing extra-key, nested-row, mistyped-field and exception-message controls pass. The result domain matches the existing resolved contest states. No legacy verifier behavior was changed by this sanitization.

**C6 — closed for the actual producers' fresh, distinct attempt paths.** `src/bts/receipt_io.py:52`, `:72`, `:119`; `tests/test_receipt_io.py`.

Sealed acceptance meets registration line 56 without admitting the earlier failed receipt. The receipt file is flushed/fsynced, renamed and its directory synced **before** a seal can become visible. Discovery now requires the seal and absence of a tombstone. The old directory-sync/removal/tombstone triple failure leaves no seal and no discoverable receipt; the producer retest confirms this (`C_publication_withdrawal`). Tombstone creation now attempts its own parent sync, fixing the prior namespace-durability omission.

A failed seal before its rename withdraws or invalidates the receipt and never creates acceptance. A failure **only** in the seal's directory sync is different: the receipt behind the already-visible seal has completed its durable publication. The auxiliary seal may be lost after a crash, which removes acceptance; it cannot make an incomplete receipt acceptable. An independent trace verifies receipt rename → successful receipt directory sync → seal rename → injected seal directory-sync error; publish returns successfully and discovery returns the complete receipt (`C_seal_directory_failure`). That disclosed loss of acceptance durability is fail-closed, not a weakening of the receipt-evidence guarantee. This is reasoning from the local rename/fsync contract and injected call ordering, not a measured power-loss test.

The safety argument assumes the actual producer's unique timestamp/UUID attempt name, not arbitrary reuse or overwriting of an already sealed path. The current CLI producers generate fresh attempt names and publish once. Consumers must use the revised sealed discovery rule; old unsealed files are unavailable. C's earlier double-refusal blocker therefore no longer blocks shared publication for D either.

**C7 — closed.** `src/bts/picks.py:477`; `src/bts/daily_decision.py:128`; `tests/test_entry_receipt.py:530`.

The held-byte text wrapper uses the same default text encoding, strict errors and universal-newline policy as `Path.read_text`. The malformed CRLF counterexample now gives the identical message and character position 7 in both loaders (`C_loader_newline_diagnostic`). CR-only and incomplete malformed inputs pass the new comparisons; the independent normal/CRLF-valid/invalid-JSON/invalid-UTF8/BOM value-or-exception comparisons still match. Exact raw bytes are retained for hashing. These are current-environment probes plus the matching standard-library decoding construction, not measurements across every installed locale/runtime.

**C8 — closed.** `src/bts/entry_receipt.py:135`, `:139`; `tests/test_entry_receipt.py:543`.

The one-row/two-identical-required-slots counterexample now returns two `unsupported_duplicate_selection` states with `all_confirmed: false` (`C_one_row_two_slots`). A used-row set also refuses reuse by another slot. Ordinary profile/pending duplication continues to collapse the same observed identity; expected duplicate selections do not turn into two confirmations.

**C2, C4 and C5 — no regression found.** The CLI entry body is unchanged since round 2. Pick and decision hashes still bind the exact read-once parsed/gated bytes, and the interleaving tests pass. The final response instant remains recorded before verification, including exact/late cutoff and earlier-DD cases. Entry hooks remain guarded reference/clock capture; mapping, hashes, extraction, qualification and serialization stay in protected publication after the authoritative decision/DM/marker path. The existing five computation-failure seams, failed hook and serialization tests pass. The new inventory work and the proposed round check add no live-path computation or request. No other new C finding was established.

P1 can proceed toward D7 only after the above patch is applied verbatim and its required checks pass, under the current prompt's conditional-signature authority. This does not sign P2 or the complete deploy/configuration candidate.

## Item D

**D1 — partially closed.** `src/bts/reconcile_receipt.py:42`, `:224`, particularly `:248` and `:251`.

The original lost-feed observation now produces `unqualified`, no manufactured schedule completion time, attributed degradation and `covered: false` (`D_missing_feed_source`; `tests/test_reconcile_receipt.py:330`). Day-level observation failures also uncover their slots (`:460`).

The remaining gap is an incomplete **earlier** slot in a later slot's clock-proof prefix. `high_water_at` includes the sources of earlier slots, but `slot_degraded` includes only failures of this slot or the whole day. A failed earlier source's time is absent from the maximum, and its degradation does not invalidate the later slot.

Reproduced through the real DD producer: primary response arrives at **08:00:01**, its recording hook fails, the clock steps back, and the fully bound DD feed arrives at **07:59:59**. Primary is uncovered, but DD is `covered: true`, `clock_regression: false`, with a fabricated-complete high-water bound of 07:59:59 (`D_late_earlier_source_lost`). Both feeds carry their actual game IDs. This is a synthetic recording/clock fault, not a measured production event. An ordinary earlier-source-loss control demonstrates the same omission without the deadline crossing.

Smallest fix: fail closed when any response observation required by the relevant day's status/slot prefix is incomplete. Conservatively invalidating coverage for the affected day is sufficient; alternatively propagate earlier observation failures through dependent slot prefixes. Preserve the legacy grading/write result. Add this combined lost-earlier-observation and clock-step producer case, not merely a single-slot lost-feed fixture.

**D2 — closed for complete observation histories; D1 still defeats the unconditional guarantee.** `src/bts/reconcile_receipt.py:237`.

For retained events, the day-wide regression flag and prefix high-water mark close the original multi-source step-back counterexample: the late 08:00:01 remains the high water, regression is true and coverage false (`D_clock_back_multiple_sources`). The suite adds DD, fallback and pre-cutoff regression controls (`tests/test_reconcile_receipt.py:348`, `:364`, `:469`). A later source cannot overwrite a retained late response into acceptance. The author's last-versus-max mutant is equivalent only under the enforced no-regression condition and a complete ordered relevant event list. Missing-event completeness is the D1 remainder; a maximum over surviving events alone cannot prove the full bound.

**D3 — partially closed; three actual-identity false greens remain. High.** `src/bts/reconcile_receipt.py:194`.

The prior Live fallback is now unqualified, and a Final feed explicitly naming another game is `fallback_other_game`, uncovered. This is verified by the retest and `tests/test_reconcile_receipt.py:384`, `:398`. Schedule voids now require the consumed schedule to contain the selected game in a void state. Counts are no longer used as a terminal basis.

The new qualification still overstates what its decisive payload establishes:

- **Missing game identity:** at `:213`, absent `gameData.game.pk` is replaced by the requested URL's game number. A feed with no own game identity publishes `actual_game_pk: 822934`, `final_feed`, covered true (`D_identity`, missing_game_identity). The schema explicitly promises identity from the feed's own field; a request's expected recipient is not an actually verified response identity.
- **Wrong-game suspended void:** at `:219`, the void return precedes game agreement. A Final selected-URL payload with its own game **999**, and only resumed/evaluable-excluded PA for the selected batter, publishes `suspended_no_evaluable_pa`, `actual_game_pk: 999`, covered true for expected game **822934** (`D_suspended_void_identity`). The correct-game suspended-void control also passes, so refusal need not break legitimate void coverage.
- **Name-only batter match:** qualification calls `grade_pick_in_feed` with the selected name. A bound selected-game feed containing only player **123** named Chandler Simpson, with no selected player **802415**, is graded by the legacy name fallback and published as selected-batter coverage (`D_identity`, name_only_batter_identity). Expected batter identity is copied into the receipt; the distinct actually observed player is not recorded or qualified. Matching a name is not the promised selected-ID proof.

Smallest coherent fix: require a valid payload-owned game identity; apply selected-game agreement to hit/miss **and** suspended void; and qualify the decisive batter by bound ID independently of the legacy name fallback. Unknown or contradictory actual identity must remain unqualified/uncovered, with expected identity separate. Do not change the legacy grader or add requests as part of this receipt fix. Replace the thin positive fixtures that omit their own game identity and add actual-producer missing-game, wrong-game void and same-name/different-ID negatives alongside valid hit/miss/void controls.

**D4 — closed for the two round-2 failures; new D6 below remains.** `src/bts/picks.py:1232`; `src/bts/reconcile_receipt.py:157`.

Write intent precedes the save, and completion is recorded only after `save_pick` returns. The original failed-save counterexample now reports `write_not_completed`, intended correction, completed false, run raised, and persisted hit (`D_write_failure`; `tests/test_reconcile_receipt.py:405`). Original exceptions still propagate. The selection-change interleaving now exposes observed batter 802415, written batter 99 and `selection_changed: true` (`D_write_selection_change`; `:419`). The preexisting race is reported rather than repaired, as required.

**D5 — closed for the reported hashing/timestamp/decision interference.** `src/bts/reconcile_receipt.py:105`, `:189`, `:224`.

The live source hook holds raw bytes and the return clock instant tagged with day/slot; hashing, parsing and coverage computation run at publication. The round-2 synthetic hashing delay now yields the same hit → miss correction and persisted miss with and without a receipt, and preserves the actual **07:59:59** response time (`D_source_processing_clock`). `tests/test_reconcile_receipt.py:439` supplies another delayed-hash comparison. Ordinary guarded event capture still has execution cost; this closes the identified fallible/heavy preparation ahead of the deadline, not a literal zero-overhead claim.

**New D6 — medium: recording failure is mislabeled as save failure and attributed to the wrong date.** `src/bts/reconcile_receipt.py:50`, `:166`, `:268`.

Inject a guarded `write_done` recording failure after a successful correction in the default eight-day lookback. The run completes and the persisted pick is **miss**, but its write says `write_not_completed`/completed false. The schema defines that as a save that did not complete and a raised run; neither happened. The degradation is assigned to day index 7 (**2026-08-13**, the last skipped target), not the corrected date **2026-08-20**, because `_guarded` uses the stale live observation index during Phase 2 (`D_write_done_observer_failure`). Consumers cannot locate the missing write evidence in the date's degradation record.

Smallest fix: attribute write-hook failures using their explicit date argument, and distinguish **completion evidence unavailable** from an original save exception/noncompletion. A missing recording event cannot assert what the business write did. Preserve original save exceptions and report successful saves only when their completion observation is qualified. Add guarded failures in write intent and write completion with multiple lookback targets; verify persisted bytes, run outcome, write evidence state and correct attribution separately.

**Shared publication and unchanged paths:** C6's sealed acceptance closes D's publication blocker too. The two source fetcher bodies and `_fetch_json` are unchanged since round 2; no observer still uses the original retry wrapper, timeout/read/JSON parse and propagates original business errors. Independent baseline comparisons again match ordered requests/results, and `reconcile_results(receipt=None)` still makes the exact two-positional/no-keyword call (`D_no_observer_baseline`, `D_no_receipt_call`). The observer token remains context-local and resets in `finally`. No new request, authenticated client, retry or cadence change was introduced. Existing lock/reload, cutoff and replay/streak-save semantics remain authoritative. Successful replay is recorded after `save_streak` returns. Public endpoint URLs, hashes and type-only errors remain the emitted source/security fields; no raw payload bodies, headers, cookies or exception messages are copied into receipts.

The 244-test green includes real producer hooks, unchanged-result, no-fetch, failure and cutoff cases, but most `feed()` positives omit `gameData.game.pk` (`tests/test_reconcile_receipt.py:47`). They therefore exercise the unregistered URL-identity fallback rather than prove the documented payload-owned identity contract. The new one-slot failure tests do not prove completeness of later slot prefixes, and a save-exception test does not prove correct treatment of a completion-observer exception. These false greens explain why supplied mutant kills do not clear D1/D3/D6.

D remains **BLOCK in its second and last round**. Its smallest revision is evidence-only: complete relevant observation prefixes, actual game/batter binding for every covered disposition, and correctly attributed unavailable write evidence. Freeze/escalate to Eric under the pace rule; do not automatically start another round or repair the original grader/selection race. A final candidate still needs the registered affected current-defence certificate reruns with tooling frozen at `f453283`; patch applicability is not re-certification.

## Item B

**Code unchanged, yes.** Between the round-2 signed HEAD `5c87de6ab6a83a9398f86171976a6dbdacdc0d75` and this pinned HEAD, both requested paths have an empty diff and identical Git blob IDs:

- `scripts/cron-setup-hetzner.sh`: `1f236bd7bea1a2122d8f4146e3faed8ff5e741e8` (mode 100755).
- `src/bts/entry_intent.py`: `be61e5bc9179bfe68b9c6723d7112f442cac581f` (mode 100644).

The prior B SIGN is retained. Items A and B are not reopened by this review.
