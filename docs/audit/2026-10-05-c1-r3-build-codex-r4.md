## Verdict

**BLOCK.**

Reviewed-commit: 5cca66e4d6c9213f85df843a26b55df40f435b04

Five of the six round-3 findings are closed. R3-5 remains partially closed: the new receipt checks validate only selected records after duplicate collapse, and partition terminal completions by outcome before checking conflicts. Malformed or incompatible witnesses can therefore disappear from validation while the actual build publishes certified outputs. The smallest remaining fix is to validate every supported request/completion identity before joins and reject incompatible completions for one attempt across all terminal outcomes.

The permitted command completed with **115 passed in 10.12s**:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/scripts/c1_r3 tests/scripts/test_c1_admission.py -p no:cacheprovider
```

I read the messages and scoped diffs of `b233585f4d16c2857c0999728c43cd831a0fd06d`, `da1143c3ad4deba5cf7b597a0a65239c8a6d7f8e` and the reviewed commit, and re-tested the round-3 counterexamples using the actual functions, independent raw-play PA counters, in-memory parquet buffers and intercepted filesystem/Git models. The admission replacement re-test calls actual `main`, `load_admission`, `admission_gate` and shared `admission_check`. Build probes use a one-season synthetic fixture (`SEASONS=(2023,)`); external filesystem/Git/lock state is modeled, while receipt validation, pin checks, real Arrow/pandas parsing, metadata verification and count/BF aggregation remain active. These checks establish executable behavior, not historical incidence or physical crash recovery.

No real `data/` input, network, SSH, gh, outside-checkout search or memory lookup was used. No tracked file was edited and no project commit or push was made. The supplied event-snapshot acquisition facts and author's 18-mutant red check are supplied evidence; I did not inspect or execute the outside-checkout mutant script. The launcher/guard and intervening watchdog/docs changes remain outside this review's scope.

Register §C row **C1-r3-build-review-r4** authorizes this last round and makes any verdict other than plain SIGN a deferral. **Rank 3 is therefore deferred for this cycle, with no fifth round.** This report does not authorize hash capture, X-34 publication, the build, repairs, commits/pushes or prospective/D7 activation. The required fix below records the residual defect; applying it does not reopen this cycle.

## Disposition of R3-1 to R3-6

| Round-3 finding | Disposition | Code/test evidence and inherited remainders |
|---|---|---|
| **R3-1 — review subject and whole-cell exposure (N1)** | **Closed** | `admission.py:105–127` binds SIGN to one full subject field, once in the whole report and inside its Verdict section. Prose mentioning a historical BLOCK no longer identifies the accepted subject. Exposure and invalidation descriptions use `fullmatch` (`:180,255`). Independent probes confirm a SIGN of R mentioning BLOCK Q accepts R but refuses Q, including for invalidation; a valid exposure prefix with a DENIED suffix refuses. Tests `test_c1_admission.py:77–102,114–139,178–208` cover exact verdict/subject, missing/duplicate/outside/short fields, conditional edits, suffixes and historical correction substitution. The inherited N1 admission and invalidation remainders are closed; no external-signature or remote-publication requirement is imposed. |
| **R3-2 — consume the admitted freeze (N6)** | **Closed** | `count_build.py:101–126` reads/parses/hashes admission bytes once and returns the admitted record plus accepted exposure-blob report identity. `main` passes those objects into `_run`, which does not reload (`:306,319,322–352`). In an independent actual-main Git/filesystem model, a second admission read would return B with different matching pins and a BLOCK report; only A is admitted. The file is read **once**; the manifest records A, A's exact admission-byte SHA and A's report; A's mismatching pin produces `STOPPED_incomplete.json`, with no results. Ordering is inventory → manifest → claim → parquet read → stop. Test `test_count_build_e2e.py:353–383` also checks the one-record behavior. The expected-pin/report-binding N6 remainder is closed. Hash-only capture before fitting remains an acceptable freeze design, subject to a real plain SIGN and X-34. |
| **R3-3 — incomplete/unsupported plays and invalid non-PA identities (N2/N3/N4)** | **Closed** | `count_meta.py:264–281` requires exact true completion for every relevant raw play, a supported result and positive builtin batter/pitcher IDs. Unsupported/invalid records produce problems before omission; `verify_game` incorporates those problems and the census quarantines them. Independent re-tests quarantine each old false green: interior incomplete/missing completion, unknown result, boolean non-PA batter and boolean non-PA pitcher. Valid ordinary and inning-ending CS controls certify with independent counters. Tests `test_count_build_parts.py:221–285` exercise those cases and bind the vocabulary to the committed snapshot. Together with unchanged official-total reconciliation and turn progression, this closes the N2/N3 and play-identity N4 remainders. |
| **R3-4 — contradictory player/pitcher witnesses (N4)** | **Closed** | `count_meta.py:128–176,192–199` validates player key/person identity before map insertion or BF use, rejects duplicate substitute ranks/declarations, different-side use of the same starting pitcher and a starter in the opposing lineup. Independent re-tests reject the shared starter, ID250/person999, conflicting substitute-person declaration and duplicate rank. Tests `test_count_build_parts.py:288–329` also retain full certification for a pitcher batting in his own lineup. Existing typed season/date/game/inning/status rules remain intact. The inherited N4 identity remainders are closed. |
| **R3-5 — receipt bytes, duplicates and availability (N5)** | **Partially closed** | `count_build.py:154–178,216–243` rejects unknown top-level kinds/outcomes, mismatched linked-response decoded bytes, the old differing-URL duplicate, malformed selected request IDs and reversed retrieval times. Independent probes confirm those decisions and preserve valid legacy receipts and identical repeats; modern positive/negative and reversed-time tests pass (`test_count_build_e2e.py:386–467`). However, dictionary equality can collapse a malformed float-ID request into a later valid integer-ID request, and conflict/attempt validation omits error completions and conflicts across completion outcomes (`:175–199`). Actual `_run` publishes certified outputs for these cases. R4-1 below is a remaining N5/R3-5 boundary failure. |
| **R3-6 — recursive ancestor creation and namespace durability (N7)** | **Closed** | `admission.make_run_dir` and `admission_lock` now create at most one directory level (`admission.py:226,287`), so missing ancestors refuse before lock/input work. Independent intercepted calls confirm missing-parent FileNotFoundError/SystemExit and no lock open; the existing-parent path requests no recursive mkdir and fsyncs the namespace and its acquisition parent in order (`:229–238`). Tests `test_c1_admission.py:212–228` cover missing ancestors, no created parent and the successful preexisting-parent control. This closes the N7 remainder while retaining the prior fsync fix. |

The inherited remainder accounting is therefore **N1, N2, N3, N4, N6 and N7 closed; N5 partially closed**. N8's durable accountable input-failure paths remain unchanged and covered by the passing suite; no N8 regression was found in this scoped change.

**Open-turn contract question:** yes, MLB's published base-running/non-plate-appearance flags, fixed to this committed metadata snapshot and excluding the pending ruling, provide an acceptable explicit supported vocabulary. Independent inspection found **74 distinct codes**, and `OPEN_TURN` exactly equals that selected set. The snapshot and constant are within the pinned executable closure. This is a declared conservative reader contract, not a measurement of historical feed coverage: membership alone does not certify a game. Completion, typed identities, turn/half progression, substitutions, official totals and parquet agreement still apply. Unrecognized codes and unsupported PA codes quarantine; this can cost coverage and trigger the >1% stop without changing the frozen production PA definition.

**Availability/refusal split question:** yes. When bytes/request identity are intact but the availability timestamp is unusable or precedes its valid request start, the eligible game is unavailable and belongs in the quarantine census; its rows must not enter the table or BF history. A conflicting identity/byte/attempt witness breaks the provenance anchor and should refuse inventory rather than become a permitted small omission. Exact duplicate witnesses may be tolerated. The problem in R4-1 is that the implementation does not enforce that distinction for all records.

## New findings

### R4-1 — Medium: validation after duplicate collapse and outcome partitioning leaves false-green receipt reconciliation

This finding is in the changed `count_build.py` receipt code and reintroduces the N5/R3-5 requirement to reject malformed or conflicting witnesses before joining/filtering them.

**A. A malformed request can be treated as an identical duplicate and overwritten.** `_witnesses` compares Python dictionaries with `!=` before assigning the last record (`count_build.py:175–177`). Python considers integer and equal-valued float identities equal. The exact integer game-ID check occurs later, on the **selected** intent/source (`:228–230`), rather than every raw request before the lookup is built.

Verified receipt sequence for game 7:

1. Intent a1, correct game-7 URL and valid started time, but `gamePk=7.0` (float).
2. Intent a1, otherwise identical, with `gamePk=7` (integer).
3. The valid legacy stored completion a1 for game 7, with correct path, stored/decoded hashes and retrieval time.

The first two raw records compare equal as dictionaries, despite the first violating the typed identity contract. `_witnesses` overwrites it with the second. `feed_inventory` returns an available entry, and the typed check only sees the second record. With the malformed float record alone, the existing test correctly refuses; combining it with an equal-valued valid duplicate bypasses that check. This is not a demand to reject truly identical duplicate receipts.

**B. Incompatible terminal completions are checked only within selected outcome classes.** `feed_inventory` builds witnesses for intents, response completions and stored completions separately (`:197–199`). `http_error`, `network_error` and `rate_limited` are allowed outcomes (`:150–162`) but never enter a completion-wide conflict/attempt check. `unresolved` treats every completion as a resolving tuple without first validating those identities (`:181–185`).

Verified independent examples:

- A valid legacy stored completion a1 for game 7 **and** a second completion a1 for game 7 declaring `outcome=http_error`, HTTP **404**. The stored receipt is selected and the incompatible terminal error is ignored. The supported legacy layout has one intent plus one stored completion for its per-game attempt; the modern layout has one response completion plus a separately identified linked store record. Neither layout makes those two conflicting terminal completions for the same attempt identical or consistent.
- An additional intent z for integer game **8**, followed by a `network_error` completion z with float `gamePk=8.0`, resolves z in `unresolved` by Python tuple equality. Its malformed game identity never reaches the later stored-feed typed checks. No unavailable/unresolved count or refusal records the mismatch.

**Measured synthetic build effect:** each negative sequence above was passed independently through actual `_run`, alongside a valid pinned in-memory PA parquet and receipt-bound synthetic feed for game 7. Filesystem reads/writes/Git were intercepted; receipt validation, stored/decoded hashing, actual Arrow schema and pandas parse, metadata verification and aggregation were not replaced. All three cases returned:

```text
return=0
eligible=1, certified=1, quarantined=0, rate=0.0
CLAIM.json, count_table.json, bf_starts.json and results.json published
BF start rows=2
```

The valid legacy control returned the same successful census. A guard that raises if `_run` reloads admission stayed untriggered, independently confirming R3-2's fix. These are fully certified outputs, not merely an unaccounted exception; the >1% quarantine rule cannot catch witnesses that never become reasons.

The passing new tests reject a differing-URL duplicate and a lone float-ID request (`test_count_build_e2e.py:435–467`), but do not combine the float with its equal-valued integer duplicate or combine incompatible completion outcomes. The supplied mutant-red result does not cover these additional compositions. I did not inspect real receipts, and infer no historical frequency or wrong numerical result from the synthetic example. The demonstrated failure is accepting a run with unsupported/contradictory provenance and falsely reconciled request state, contrary to the declared pre-aggregation source boundary.

## Required changes

The smallest residual fix is confined to receipt validation:

1. Validate **every** supported raw feed intent/request completion's exact positive integer game identity and string attempt identity before dictionary/set comparisons, lookup construction or `unresolved`. Preserve explicitly supported schedule metadata and the legacy/modern layouts; do not invent retrospective response receipts for the legacy capture. A malformed earlier duplicate must refuse even when a later valid record compares numerically equal. After typed validation, tolerate genuinely identical duplicates.
2. Check completion witnesses across **all** terminal outcomes for a given attempt, rather than only within response and stored partitions. Refuse distinct incompatible completions and validate error completions before they can resolve an intent. Preserve modern storage's separate store ID and its explicit `from_attempt_id` link. A completion-wide witness map, followed by outcome-specific selection from validated records, is sufficient; no new acquisition is needed.
3. Add meaningful negative checks for the float-then-integer duplicate, stored-plus-HTTP404 completion and malformed error completion resolving an otherwise unmatched intent. Retain positive controls for truly identical repeats, valid error completion/retry records with distinct attempt identities, valid legacy receipts, the modern response/store chain and counted reversed-time quarantine. Verify refusal before the claim and before any PA/feed read; do not silently accept these cases as ≤1% omissions.

No further edit is required for R3-1, R3-2, R3-3, R3-4 or R3-6 as re-tested here. Preserve the fixed count specification, conservative unsupported-event quarantine, official-total reconciliation, expected pins/same-buffer parsing, durable claim/stop publication and complete census.

**Disposition under the final-round ruling: deferred for this cycle; no fifth round.** The required changes are a record of the remaining defect, not authorization to repair, promote or run it in this cycle.
