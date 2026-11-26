## Verdict

**SIGN.**

Reviewed-commit: bbafbef11beb530b22e8ba8232eb874dbd4b1510

C2R1-1 is closed. All three required changes are met: supported receipt shapes are validated before joins, completion/intent identity comparisons include request kind and the appropriate primary ID, and actual-main negatives and writer-generated positives cover the required compositions. Fresh byte-read instrumentation confirms that all former false greens now refuse before creating a run directory, publishing a claim or reading PA/feed bytes. The actual legacy and modern producer controls still certify. No new defect in the change was found.

The permitted suite completed with **184 passed in 11.47s**. HEAD matches the requested pin and the tracked tree was clean before and after review. This round read the entire prompt first and performed no memory lookup or outside-checkout source search. It continues the same session as round 1; round 1's disclosed memory-registry search remains part of that round's provenance, so this is not a claim that the review sequence was strictly blind.

This is the last C2 code-review round. The plain SIGN supplies the reviewed-code verdict required for later admission; it does not itself authorize a build, hash capture, exposure row, deployment or activation. No real input or box state was inspected, and operational admission was not demonstrated.

## Findings

### C2R1-1: all three required changes met

| Required change from round 1 | Disposition and current evidence |
|---|---|
| 1. Complete the identity boundary for both request kinds while preserving actual writer shapes | **Met.** `count_build.py:172–199` validates every intent/completion before either witness map or reconciliation. Feeds require an exact positive builtin integer `gamePk` and null/absent `season`; schedules require an exact positive builtin integer `season` and null/absent `gamePk`. Nonempty string attempt/link IDs remain required. Fresh probes refuse non-null neutral-field values, including false/zero, on both roles, and malformed schedule primary IDs on both roles. Actual producer controls and null-versus-absence controls pass. |
| 2. Resolve by validated request kind and primary identity; reject incompatible completions; preserve completion uniqueness and modern links | **Met.** `_request_identity` (`:206–209`) returns `(kind, gamePk-or-season)`. `unresolved` (`:230–244`) compares that identity whenever a completion shares an intent's attempt ID, before marking it done. This applies to every completion outcome. Earlier duplicate checks (`:255–262`) protect the intent/completion maps before their use. Modern store IDs without a separate request intent remain valid, and their explicit link reaches the response/request as before. |
| 3. Actual-main negatives, direct byte-read instrumentation, producer schedule/feed/retry positive, selective error-conflict tests/mutants and attributable failures | **Met.** The added tests exercise all three compositions, including a schedule-declared intent with its game ID removed to isolate the join. `_refuses_before_claim` now instruments `Path.read_bytes` for both input roots. The producer-generated positive runs the real acquisition `main`; network-error/rate-limited conflicts are parametrized and have selective-omission mutants. `117db05` adds isolated bad-season cases for both roles. The supplied runner runs all selected tests and prints each failed test name; fresh standalone probes independently establish the important behavioral failures behind its REDs. Remaining coverage opportunities below are nonblocking. |

### Fresh actual-main reproductions

Each negative ran alongside a valid game-7 synthetic feed, a receipt-bound stored completion and a pinned PA parquet. The actual `main` → `_run` → inventory path executed. Fixture creation and initial pin preparation preceded instrumentation; after that, the spy covered actual PA/feed `Path.read_bytes`, and claim publication was also instrumented.

| Round-1 composition | Result at this pin | Run directories / claims / PA-feed reads / results |
|---|---|---|
| Integer-8 feed intent z; schedule-declared network-error completion z with `season=2023, gamePk=8.0` | `ProvenanceError: attempt z: a schedule receipt carries a game id: 8.0` | 0 / 0 / 0 / 0 |
| Schedule intent s for 2023; sole schedule error completion s for 2024; both game IDs null | `ProvenanceError: the completion for attempt s names a different request ('schedule', 2024) than its intent ('schedule', 2023)` | 0 / 0 / 0 / 0 |
| Legacy game-7 intent a1 re-declared as schedule/2023, keeping integer `gamePk=7`; unchanged feed stored completion a1 | `ProvenanceError: attempt a1: a schedule receipt carries a game id: 7` | 0 / 0 / 0 / 0 |
| Same re-declaration, but remove the intent's game ID so both records independently satisfy their shape rules | `ProvenanceError: the completion for attempt a1 names a different request ('feed', 7) than its intent ('schedule', 2023)` | 0 / 0 / 0 / 0 |

The last variant tests the discriminator/identity check independently of the stray-field rejection. I also paired a valid feed intent for numeric game **2023** with a valid schedule completion for numeric season **2023**, using the same attempt ID and otherwise appropriate null/absent fields. The current code refuses the differing request kinds before claim/reads even though the primary numbers match.

**Legacy writer-generated control:** loaded `20ffa8e:scripts/audit/c1_r3/acquire.py` from this checkout's Git object and ran its actual `acquire` with synthetic fetches and task-owned roots. Its successful intent and stored completion omit both `kind_of` and `season`, share the attempt ID and carry integer game 7 and the canonical feed URL. The build returned **0; eligible=certified=1; quarantined=0; two BF starts**, with results published. Instrumented order was **claim → PA read → feed read**. No response receipt was retroactively invented.

**Modern writer-generated control:** ran the current acquisition `main` against a synthetic schedule and feed, with a network error on the first feed attempt followed by a successful retry. Its real emitted sequence contained:

- schedule intent/response: `gamePk=null, season=2023`;
- linked schedule stored completion: omitted `gamePk`, separate store ID;
- feed intent/error and distinct retry intent/response: integer `gamePk=7, season=null`;
- linked feed stored completion: omitted `season`, separate store ID.

Those receipts passed actual build `main`: **return 0; eligible=certified=1; quarantined=0; two BF starts**, results published, with **claim → PA read → feed read**. The changed rules therefore do not falsely refuse either tested writer layout. Additional successful controls mix null with absence on neutral fields and explicit feed kind with the legacy default; genuine duplicate witnesses and a distinct completed error attempt also certify.

The inherited legacy `failed`/`invalid` outcome limitation recorded in round 1 is unchanged: `_receipt_problems` still supports the same outcome vocabulary. I have not broadened that policy, reclassified it as a new defect, or inferred the contents of actual historical receipts.

### No regression of R4-1, R3-1 to R3-6 and N1 to N8

| Earlier requirement | Fresh evidence / current disposition |
|---|---|
| **R4-1's raw typed identities and canonical duplicate comparison** | Exact float-7 intent followed by its integer twin still refuses at the type check; exact network-error completion with float game 8 still refuses. Both have zero run directories, claims and PA/feed reads. Canonical comparison remains unchanged; the differing-number-type duplicate test passes. Its wording now accurately describes canonical parsed-record comparison. |
| **R4-1's all-outcome completion map** | Exact stored-plus-HTTP404, stored-plus-network-error and stored-plus-rate-limited sequences all refuse with `conflicting completion receipts for attempt a1`, before claim/reads. The completion-wide map remains ahead of outcome-specific response selection. |
| **R4-1's positive controls and availability split** | Both actual writers and genuine-repeat/distinct-error controls pass. A fresh 120-game reversed-time control returns 0 with **119 certified, one quarantined, rate 1/120, 238 BF starts** and the explicit availability reason for game 7. The unavailable game's rows are excluded. The existing linked-response decoded-byte mismatch test passes. |
| **R3-1 / N1: exact report subject, affirmative whole-cell exposure/invalidation** | Shared gate unchanged; its full test file passes. Independent report probes accept only the declared full subject and refuse a historical BLOCK hash in the same Verdict section; a valid exposure cell with a DENIED suffix fails `fullmatch`. Conditional SIGN WITH EDITS and exact invalidation protections remain covered by the suite. |
| **R3-2 / N6: consume the admitted freeze, accepted report identity and pinned bytes** | Main still passes the same admitted record/identity into `_run` with no reload; this code is unchanged. The suite's admission-replacement test consumes A once, retains A's expected pin/report identity and refuses results. A fresh wrong-pin actual-main control retains the claim and writes `STOPPED_incomplete.json` naming `pa_2023.parquet`, with game 7 pending. |
| **R3-3 / N2/N3 and play-identity N4: completeness, supported turns and non-PA identities** | Reader/verifier unchanged. Fresh independent raw-play PA counters certify ordinary and inning-ending caught-stealing controls. Interior false/null/missing completion, unsupported open-turn result and boolean non-PA batter/pitcher each produce quarantine reasons. Official PA/BF totals of 27 against nine observed turns yield all four discrepancies. Existing turn/substitution/IBB/resumed/zero-PA tests pass. |
| **R3-4 / N4: contradictory player/pitcher witnesses and strict metadata** | Fresh probes reject a shared starter and ID250/person999 BF declaration. Suite retains opposing-lineup rejection, legal own-lineup pitcher, duplicate substitute-rank rejection and typed game/season/date/inning/status checks. Metadata code unchanged. |
| **R3-5 / N5: receipts, bytes and availability** | Closed for the recorded r4 and C2 round-1 findings. The exact negatives, writer controls, byte-binding tests and counted availability controls above pass. No newly demonstrated provenance false green in the actual changed code. |
| **R3-6 / N7: durable namespace and no recursive ancestor creation** | Shared code unchanged; fresh missing-ancestor lock probe refuses without creating the parent. Existing namespace creation, missing-parent and lock tests pass. No physical power-loss test is claimed. |
| **N8: durable accountable failure paths and complete census** | Fresh wrong-pin and missing-feed actual-main probes preserve the claim, publish no results and write source/error/pending-game stop records. Claim precedes the attempted input reads. Existing matching-hash bad-gzip, schema, persistent-claim/rerun and quarantine tests pass. |

Only `count_build.py` changed in executable source since the round-1 pin. The shared admission gate, acquirer, metadata/verifier/count/BF modules, all `src/bts`, dependency files and frozen registration are unchanged. Dynamic import inspection again finds only `bts`, `bts.data`, `bts.data.build` and `bts.data.schema` among loaded BTS modules. The unchanged parser helper/constants remain the build's production dependencies; intervening watchdog producers are still not imported or executed on this path. Count definition, conditioning, cap/overflow, smoothing, resumed treatment and >1% stop are unchanged and covered by the passing suite.

### New defects and test/mutant strength

**No new defect found in the change.** In particular, the new neutral-field constraints fit the actual successful legacy and modern writer outputs, including separately identified linked stores. Rebuilding a local intent map inside `unresolved` is safe on this path because typed validation and conflict-checking witness maps have already run; no incompatible duplicate is silently collapsed before those checks.

The supplied ledger is materially stronger than round 1's: 14 current anchors each match exactly once, no `-x` truncates selected tests, R1/R9 directly select the malformed error-completion example, R2b/R2c target selective error omissions, and R11 selects isolated wrong-season and kind-mismatch cases. `117db05` records and addresses the earlier season-check mutant survivor by removing the competing stray-game-ID issue from the test and testing both roles.

Fresh **in-memory**, actual-main mutant probes independently confirmed that removing all typing (R1), primary schedule-season validation (R6), primary feed-game validation (R9), network-error conflict participation (R2b), rate-limited participation (R2c), the identity join check (R11), or the feed-neutral-season restriction (R12) permits the corresponding unsupported/conflicting witnesses to publish certified outputs. These are behavioral controls; I did not execute a mutant pytest suite or edit tracked source.

Two nonblocking limits remain in the ledger's granularity:

1. **R10 is caught through exception-message placement in its selected compositions.** Removing the schedule stray-game-ID check still makes those cross-kind compositions refuse at the identity join, with `different request` instead of the required `schedule receipt` message. Thus this RED demonstrates enforcement of the earlier validation boundary, not acceptance under that mutant. I isolated the shape rule with a schedule intent and schedule error completion both carrying `gamePk=8.0` and the same valid season: actual code refuses before claim/reads; the R10 mutant certifies. That isolated behavioral negative would strengthen the stored ledger.
2. **Request kind deserves an equal-primary-number mutant/control.** The ledger removes the whole identity check, but does not selectively remove the kind component from `_request_identity`. A fresh in-memory mutation retaining only the primary number certifies feed-game-2023 versus schedule-season-2023; current code refuses it. Existing wrong-season and schedule-anchor tests use unequal primary numbers, so they do not independently establish this component. This is an optional coverage improvement, not a defect in the actual implementation or a condition on SIGN.

The tests now spy on input byte reads directly, correcting round 1's schema-spy limitation. Supplied failed-test names establish selection/attribution; they do not by themselves expose every assertion's failure reason. The independent probes above supply that distinction for the key rules. The green suite and RED ledger remain code evidence, not historical-feed coverage, real-input acceptance, crash-recovery or operational permission evidence.

## Required changes

None. C2R1-1 is closed without conditional edits. The two optional mutation-coverage improvements above are not conditions on this plain SIGN. They do not authorize changing the reviewed executable closure after admission.

## What was run

### Freshly reproduced in this round

- Read the entire round-2 prompt before any other command; read the round-1 report, both fix-commit messages, complete scoped diff, current changed reader and relevant writer code, and the supplied evidence files. Confirmed requested HEAD and clean tracked tree. Git comparison establishes the unchanged regression/closure paths above and that executable source/tests at the supplied evidence pin `117db05d8ee0e0eafbd28d5029b6e6ae4c701577` equal those at this reviewed pin.
- Ran exactly the permitted command:

  ```text
  UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run python -B -m pytest tests/scripts/c1_r3 tests/scripts/test_c1_admission.py tests/scripts/test_c1_r3_acquire.py -p no:cacheprovider
  ```

  Result: **184 passed in 11.47s**.
- Ran `.venv/bin/python -B` standalone probes in `/private/tmp/c2-r3-r2-Juc1Pw/`. They used actual build `main`, `_run`, inventory, schema/pin checking, metadata certification and aggregation; real locks and task-owned files; actual historical writer `acquire` and current acquisition `main`; synthetic fetch callbacks; and actual Arrow/pandas parquet serialization/parsing. PA rows were counted directly from raw fixture plays independently of `count_meta.extract`. Synthetic roots/seasons, accepted admission and clean-tree status were modeled. These probes do not show that operational admission of this checkout succeeds.
- Instrumented `Path.read_bytes` for both synthetic input roots after fixture/pin preparation and instrumented claim publication. Reproduced all C2R1-1 and exact R4-1 negatives, writer-generated positives, genuine repeats/retries, null/absence compatibility, typed-neutral/primary field negatives on both roles, reversed-time quarantine, equal-primary cross-kind refusal and durable wrong-pin/missing-feed stops. Independently checked selected inherited gate/metadata/namespace boundaries and loaded BTS modules.
- Checked all 14 ledger anchors against current source and loaded selected mutants only in memory. No author mutation runner, extra pytest command or tracked edit was performed. Arrow emitted sandbox CPU-cache-query diagnostics; probes completed without escalation.
- Removed all task-owned scratch after recording the observations here. Final report structure/subject, HEAD and clean tracked state were mechanically verified, then the receipt was computed from the final report bytes.

### Author-supplied evidence, inspected but not rerun

`docs/audit/2026-10-06-c2-r3-r41-evidence/mutants.out` records the 14-mutant run at `117db05`: every mutant RED, with each failed test named; R6 has twelve failures, R2 three, R9 four, R10 two, and R11 two failures plus one passing test. I inspected its ledger, runner and output without executing its tracked-source mutation workflow. `fast_suite.out` records **3940 passed, 7 skipped, 6 deselected, 22 xfailed**, exit 0, at the same pin. The broad fast suite was not rerun here.

No real `data/`, credential or configuration contents were inspected; no box, network, SSH, gh, escalation, commit or push was used. Only this report and task-owned scratch were written by the review, apart from the explicitly permitted test command's normal temporary files.
