## Verdicts

A: verbatim **yes**; no differences in A1, A2 or A3.
B: **SIGN.** B1 is closed.
C: **BLOCK.** Preserve conflicting lookup evidence, close the remaining row-disclosure and failed-publication gaps, and refuse unsupported duplicate selections. This is the last review round: the item goes to Eric for a decision.
D: **BLOCK.** Qualify actual terminal sources and complete observations, preserve cutoff violations across clock regressions, defer source processing, and report writes only after their actual outcome is known. The shared publication blocker also applies.

Reviewed pinned detached HEAD `5c87de6ab6a83a9398f86171976a6dbdacdc0d75`, the messages/diffs of `2769a7e`, `7da1b3b`, `2a2934a` and `5c87de6`, the revised schemas, plan P2 and registration lines 54, 56 and 59. No tracked files were changed. This round used checkout-only evidence and synthetic temporary inputs; no real data/configuration/credential file, external memory, network/DNS, SSH, gh or real crontab was accessed. Network client boundaries were denied before task-module imports in the independent probes; cron children also received those guards and used the tests' throwaway HOME and crontab shim.

The exact permitted suite passed **220 tests in 10.21s**; its output is in `pytest-r2.log`. Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_entry_receipt.py tests/test_entry_intent.py tests/test_private_mode_transport.py tests/test_cli_integration.py tests/test_reconcile_receipt.py tests/test_reconcile_cutoff.py tests/test_scoring_lock.py tests/test_daily_decision.py tests/test_daily_decision_v2.py tests/test_daily_decision_v3.py -p no:cacheprovider
```

Independent probes completed successfully and are retained as `probes-r2.py`, `probe-run-r2.log` and `probe-results-r2.jsonl` in this directory. Findings below identify synthetic counterexamples separately from source-derived durability limits. The author's external scratch mutant/isolation evidence was not inspected or independently reproduced as a whole. Green tests and applying certificate patches do not clear the counterexamples below. Re-certification with the frozen runner remains a final deploy-candidate gate; it was not run under this prompt's suite boundary. Nothing here approves deployment, D7, installed-producer commissioning, A8 or play.

## Item A

**A1, A2, A3 — applied verbatim, yes.** A mechanical comparison against the round-1 report confirmed the complete A1 test body, A2 condition and added parameterized test, and A3 documentation sentence. See `tests/test_private_mode_transport.py:3`, `:39`, `:61` and `src/bts/scheduler.py:891`.

The actual private-lock tests pass in the permitted suite. Independently forcing `_pick_delivery_mode` to `dm` now fails each of the three actual decorated tests and reaches the fake DM transport, rather than returning early for a missing recipient (`A_forced_dm`). The legacy private/shadow combinations and shadow-only refusals are covered by the suite. The claim remains the exact narrowed A3 claim about an eligible private recommendation lock; it does not assert that operational health DMs cannot occur.

## Item B

**B1 — closed.** `scripts/cron-setup-hetzner.sh:108` contains the round-1 replacement branch verbatim: grep status 1 is the expected empty result, and other filter failures refuse before installation/removal can invoke `crontab -`.

`tests/test_entry_intent.py:262` injects a grep error only on `-v`, for both install and remove, and checks nonzero exit, unchanged foreign lines and the crontab call log. Independent guarded invocation of those actual tests retained the planted foreign job byte for byte; both call logs contained only `-l` (`B_filter_error`). Empty, missing and BTS-only crontab controls also pass in the permitted suite. No new B finding.

The round-1 positive intent conclusions remain supported: exact enter/research values with no default, validation against the effective delivery resolver, refusal of missing/malformed/disagreeing configuration, and quoted validation invocation. The scheduler does not gain an `entry_intent` behavior change. Deployment still does not install cron; the documented migration and separate activation decision remain necessary. No installed configuration was examined.

## Item C

**C1 — partially closed: separate game qualification is implemented, but conflicting captures are discarded. High.** `src/bts/entry_receipt.py:94`, `:114`; `tests/test_entry_receipt.py:149`, `:163`, `:175`.

The old batter-only false confirmation is closed for unknown/wrong units: the legacy verifier remains separate, and consumers must require `qualification.all_confirmed`. Missing units, wrong game/round, conflicting identities within one file, unresolved players and a missing DD leg have fail-closed fixtures.

The latest-capture choice does not reproduce the earlier audits' ambiguity handling. `season_ledger/contest.py:70` retains every recorded feed/round identity and `:122` rejects contradictions. `mining87/run.py:252` consumes all pinned capture files, admits only a singleton feed identity and returns every consumed file's hash. P1 instead selects only `snaps[-1]` and drops all earlier evidence.

Reproduced: two synthetic captures bind unit 1/round 7 first to feed 9 and then to feed 1. A selected batter/game 1 becomes `all_confirmed: true`; the receipt names only the later capture (`C_conflicting_captures`). Registration line 58 says source ambiguity invalidates current confirmation, and the schema itself lists conflicting captures as unverified. A latest-file policy cannot resolve that contradiction without a separate authority rule.

Content deduplication alone does **not** make an old capture an invalid lookup: `static_capture.py:125` stores no new file when the raw bytes equal the previous capture. An unchanged mapping can remain useful. Its filename is the last stored-content time, however, not proof of a recent successful cron run or a fresh account observation. Do not impose an invented age limit or describe the stored timestamp as a freshness receipt.

Smallest fix: bind a declared inventory of applicable local unit captures, retain their consumed-byte hashes and reject conflicting feed/round identities. Alternatively, Eric must explicitly authorize an authoritative-latest interpretation that changes the frozen ambiguity rule. Add a real-producer conflicting-history case. No additional authenticated request is needed or authorized.

**C2 — closed for consumed-byte identity; exact text-reader semantics have a small new difference.** `src/bts/picks.py:459`, `:467`; `src/bts/daily_decision.py:112`, `:123`; `tests/test_entry_receipt.py:230`, `:252`.

The producer parses and gates on the bytes returned by the read-once loaders and hashes those same held bytes during publication. It does not reread the path for identity. The replacement interleavings now bind the consumed pick and decision, rather than the later file. The old `load_pick`/`load_decision` retain `Path.read_text()` and the shared parser behavior.

Independent comparisons with functions extracted from the pinned pre-C commit matched values or exception classes for normal, CRLF-valid, invalid JSON, invalid UTF-8 and UTF-8-BOM inputs; returned raw bytes were exact (`C_loader_baseline`). That is bounded evidence, not proof for every locale/runtime.

**New C7 — low: universal-newline handling changes malformed-pick diagnostics.** `load_pick_bytes` decodes raw bytes without the universal-newline translation of `read_text`. For `b'{\r\n"x":\r\n}'`, the old loader reports `line 3 column 1 (char 7)`/position 7, while the new producer reports `(char 9)`/position 9 (`C_loader_newline_diagnostic`). Both raise `JSONDecodeError`; no scoreability difference was observed. Thus the statement that the producer's old encoding/error semantics are *truly unchanged* is too strong. Smallest fix: use text decoding with the same encoding/error/newline policy as `read_text` over the held bytes, while retaining the original raw bytes for hashing; add the malformed CRLF/CR diagnostic comparison. This does not reopen the byte-binding defect itself.

**C3 — partially closed: extra keys and exception messages are excluded; a known string field remains unrestricted. Medium.** `src/bts/entry_receipt.py:37`, `:66`; `tests/test_entry_receipt.py:277`, `:288`, `:302`.

The explicit typed field allowlist removes the original arbitrary-key and nested-slot leaks. Mistyped values are nulled, bools do not masquerade as ints, and HTTP errors retain type/status without secret-bearing messages/URLs. Those round-1 counterexamples are covered by the permitted suite.

Reproduced payload drift: a valid pending row with `result: "SYNTHETIC_TOKEN_IN_RESULT"` publishes that string verbatim and still has `all_confirmed: true` (`C_known_string_field`). This is not evidence that the live API returns a token there. It shows that the categorical "Never stored ... any token" statement is not enforced for the one allowlisted response string. Smallest fix: emit only the explicitly supported result states (the existing `contest_fetch.RESOLVED` domain plus supported unresolved states), null and flag other values, and test a secret sentinel in `result`. Preserve the opaque payload hash and legacy verifier behavior.

**C4 — closed.** `src/bts/cli.py:1799` records completion immediately after the last fetch and before verification. `tests/test_entry_receipt.py:337` uses fixed clock state to cover a response just before cutoff with verification after it, equality at cutoff and a genuinely late response; `:346` covers an earlier DD cutoff. The prior post-verification timestamp defect is removed. The new D timestamp defect below is in a different observer.

**C5 — closed for the reported preparation-failure defect.** `src/bts/entry_receipt.py:148`, `:180`, `:200`, `:240`, `:281`.

The running hooks hold references/clock readings behind guards. Hashing, extraction, unit lookup, qualification and serialization occur inside protected publication after the original decisions, DMs and marker writes. `tests/test_entry_receipt.py:397` injects `MemoryError` at five preparation seams and compares DM count, status, reason, escalation and exit against the normal path; `:405` and `:413` cover a failed hook and serialization. The old observation-assembly failure no longer suppresses the original DM/marker. Original business exceptions remain authoritative. Publication availability is still limited by C6; reference-only hooks do not clear that separate contract.

**C6 — partially closed: ordinary withdrawal and ancestor syncing work, but the stated limit still permits failed publication to be accepted. High.** `src/bts/receipt_io.py:31`, `:44`, `:62`, `:84`; `tests/test_entry_receipt.py:423`, `:469`.

The original post-rename directory-fsync failure is now withdrawn when cleanup works. New directory levels have their parents synced, and the suite exercises write/file-fsync/rename/directory-fsync failures and cleanup/tombstone paths. These are meaningful improvements.

Two remaining results matter:

- Force post-rename directory-fsync failure and refuse final-file removal: a successful tombstone makes discovery return zero receipts. However, `_withdraw` fsyncs the tombstone file and never its parent directory. The probe traced **zero directory syncs after tombstone creation**. Its crash-durable namespace entry is therefore not established. This is source/trace evidence, not a power-loss measurement.
- Also refuse tombstone creation: the command reports `entry receipt unavailable`, yet `discover` returns one receipt with `all_confirmed: true` and `before_cutoff: true` (`C_publication_withdrawal`). The stated limit is reproduced, not merely hypothetical. Stderr is not a consumer-visible disqualification.

Under registration line 56, that limit is **not acceptable for an unconditional code SIGN**: failed/incomplete publication must be unavailable evidence. Smallest local fix for the first gap is to sync the tombstone's parent and test that failure. The double-refusal gap needs a qualification/closure rule that fails closed when publication completion is unavailable, or Eric's explicit decision to weaken the frozen guarantee. Another best-effort write on the same failing filesystem cannot by itself supply that guarantee. Do not silently sign the weaker contract. This helper is shared with D, so the same blocker applies there.

**New C8 — medium: one observed row can confirm two unsupported expected slots.** `src/bts/entry_receipt.py:120`.

A synthetic committed selection with identical primary and DD batter/game identities and only one pending row (`number: 1`) produces two confirmed slots and `all_confirmed: true` (`C_one_row_two_slots`). This is an invalid/unsupported selection counterexample, not a claim that the current scheduler emits it. Independent per-slot set lookups reuse the same row without checking selection cardinality. Smallest fix: explicitly refuse duplicate expected identities as unsupported, or implement a one-to-one observed-slot binding that cannot confirm two required entries with one row. Retain ordinary profile/pending duplication handling and add a producer fixture.

Marker compatibility, original retry/cadence sites and no-new-authenticated-fetch behavior remain supported by the diff and the passing legacy tests. The documented 56 entry runs/files per full enter day and restic path are configuration/source facts, not measured installed volume, retention or backup success. The producer revision samples checkout HEAD at publication and is correctly described as weaker than executing-artifact/deployment proof. C is **BLOCK in its second and last round**; remaining fixes or contract relaxation require Eric's disposition, not an automatic third review round.

## Item D

P2's required contract is stronger than recording the grader's return value. Registration lines 54/56/59 and plan P2 require per-date/slot dispositions, expected versus actually verified identity, consumed source identity/hash, final/evaluable or void basis, actual response time and next-day 08:00 ET cutoff, plus separate actual correction-applied/refused evidence. Pending/skipped/late slots and an empty corrections list cannot establish coverage. The schema and actual CLI/direct producer paths exist, but the following false greens prevent that contract from passing.

**D1 — high: incomplete source observation can still be covered.** `src/bts/reconcile_receipt.py:97`, `:125`; schema's consumer coverage rule.

Inject a guarded recording failure only for game-feed observations, leaving the real synthetic feed fetch/parse and schedule observation intact. The receipt has `degraded: ["RuntimeError"]`, an observed miss, `basis: "final_feed"`, **no slot sources**, and `covered: true`. It substitutes the earlier schedule timestamp for the missing feed timestamp (`D_missing_feed_source`). The documented consumer rule accepts observed/covered slots and does not require complete source evidence or an empty `degraded`.

Smallest fix: missing/failed source observation must leave the affected slot unverified/uncovered; schedule-time substitution is valid only for a successfully observed schedule-void basis. Require complete, applicable source qualification in the consumer rule and a fail-closed treatment of recording degradation. Bind failures to the affected date/slot so another slot cannot inherit a prior successful source. Add this actual-producer fault fixture.

**D2 — high: a later clock reading hides an earlier post-cutoff response.** `src/bts/reconcile_receipt.py:102`, `:137`; `tests/test_reconcile_receipt.py:299`.

Reproduced fallback chain: the selected-game feed completes at **08:00:01**, lacks the batter, and a wall-clock step back precedes the fallback schedule and decisive alternative feed at **07:59:59**. The receipt retains all three timestamps but overwrites `response_completed_at` with the last one; the day is observed and the slot `covered: true` (`D_clock_back_multiple_sources`). The current single-feed step-back test passes because there is no later source to overwrite the late time.

Smallest fix: preserve response event ordering and make a relevant cutoff crossing irreversible for coverage during that slot's resolution; a later wall-clock step back cannot erase it. Reject ambiguous clock regressions or use a qualified interval/high-water rule over the consumed observations. Keep the legacy arrival/write decision unchanged, as required. Add the multi-source fallback case and exact-cutoff controls. Per-slot `last_time < cutoff` alone does **not** meet line 59.

**D3 — high: terminal basis and actually verified game are guessed from source count.** `src/bts/reconcile_receipt.py:133`; `src/bts/picks.py:1066`.

The producer stores raw-byte hashes, but `slot_done` labels nonvoid returns as `final_feed` or `final_feed_fallback_search` solely from the number of recorded payloads. It never qualifies their parsed terminal status or binds the decisive batter/game identity separately from the expected slot.

Reproduced: selected final feed has no selected batter; the schedule lists another game as Final, while that alternative decisive feed says **Live (`L`)** and contains the batter with a hit. The preexisting fallback grader returns hit. P2 publishes `basis: "final_feed_fallback_search"`, `covered: true`, and the expected selected game although its decisive feed URL identifies the alternative game (`D_nonfinal_fallback`). No recording failure or degraded flag is needed.

Smallest fix: independently qualify the consumed decisive payload's final/evaluable status, actual batter/game and void basis, preserving expected identity separately. Unknown or contradictory schedule/feed/identity evidence is uncovered. Carry the actual grading basis rather than infer it from payload count. Test successful fallback, contradictory terminal state, absent batter and wrong game through the producer. This is a receipt fix; repairing the preexisting fallback grader is a separate scope/owner decision and is not authorized by this review.

**D4 — high: an unsuccessful save is labeled applied; the written selection can differ from the observed one.** `src/bts/picks.py:1214`, `:1226`, `:1231`.

`rec.write(..., "correction_applied")` and `"slot_results_updated"` execute **before** `save_pick`. With an injected `save_pick` OSError, the receipt's overall outcome is raised, but its per-date write says hit → miss was applied while the persisted pick is still hit (`D_write_failure`). The schema defines that state as an actual completed write, and treats coverage and write outcome as separate facts; a run-level exception does not correct the false per-write assertion.

A separate interleaving changes the selected batter from 802415 to 99 before the existing lock/reload. The legacy code applies the original proposal to the newly loaded pick, unchanged by P2; the receipt exposes only observed batter 802415 and `correction_applied`, while the persisted batter is 99 (`D_write_selection_change`). Its corrections list names the changed batter, but the write lacks actual selection identity. This probe demonstrates the existing race; it is not authorization to repair it.

Smallest fix: distinguish write attempted/failed from completed, emit applied/updated only after successful save, and retain failure state if save raises without swallowing that original error. Record the actual selection read/written under the existing lock separately from the observation selection; a mismatch cannot imply a correction of the observed/current selection. Add failed-save and changed-selection cases. P2 need not implement P3's whole multi-file snapshot to report these facts truthfully.

**D5 — medium: source hashing runs before both the claimed response timestamp and the business deadline check.** `src/bts/picks.py:682`; `src/bts/reconcile_receipt.py:98`.

The observer copies/encodes and hashes the body, then reads the clock. A synthetic source-hashing delay advances the clock from the feed's actual return at **07:59:59** to **08:00:01**. Without a receipt the original path applies hit → miss; with a receipt it discards that correction and retains hit, while claiming response completion at 08:00:01 (`D_source_processing_clock`). This is a reproduced timing-fault counterexample, **not** a measurement of real SHA-256 cost or production frequency.

Smallest fix: capture the response-return instant and held immutable raw bytes at the client return boundary, and defer hashing/qualification/serialization until protected publication after the original decisions/writes. Preserve original parsing and error propagation. Add fixed-clock tests that advance during receipt processing, comparing results, writes and streak against the no-receipt path. Guarding exceptions does not isolate computation time, and the current categorical behavior-neutrality claim is false at the cutoff.

**Publication:** D uses `receipt_io.publish` unchanged. The C6 ordinary-failure improvements and both remaining gaps apply to reconcile receipts too. Its statement that failed publication is never discoverable cannot be unconditional under the admitted double-refusal limit. D's publication tests cover ordinary failure seams, not durable tombstone creation or the demonstrated double refusal.

**Unchanged behavior that is supported, and its limits:**

- With no observer, `_fetch_json` performs the same `retry_urlopen(..., timeout=15).read()` and `json.loads(raw)` as the replaced inline sites, preserving retry wrapper arguments, raw parsing and original exceptions. Independent comparisons against pre-P2 functions matched return values and ordered request/timeout lists for detailed statuses and a three-request `check_hit` fallback (`D_no_observer_baseline`). The context default is None and its reconcile token is reset in `finally`.
- Source/call-site inspection covers check-results, scheduler, model and policy shadow callers, orchestrator, strategy and experiment paths: none sets the observer. The shared replacement does not add a request, retry, pacing rule or authentication site to them. This conclusion comes from the diff/call graph plus bounded baseline probes, not an independently executed full suite of every caller.
- `reconcile_results(receipt=None)` makes the exact original two-positional-argument call with no keywords (`src/bts/picks.py:1187`; `D_no_receipt_call`). Existing scoring lock, reloaded pick, cutoff checks, replay/refusal and streak-save path remain in place. `replayed` records saved only after `save_streak` returns. The source observer adds no direct production save. D5 nevertheless disproves unchanged receipt-enabled writes/decisions at the cutoff.
- `tests/test_reconcile_receipt.py:227` compares request lists, corrections, slots and streak with/without the new receipt on normal fixed-clock fixtures. It is useful normal-path evidence, but compares two variants of the new code and misses processing delay, incomplete observation, contradictory fallback and failed writes. The producer tests cover unchanged results, pending/not-attempted, schedule void, failures and late answers; their green state does not qualify the counterexamples above.

**Security and provenance:** the new source observer is attached to the existing public MLB schedule/live-feed requests. Its emitted source fields are constructed endpoint URLs, timestamps and hashes of held bytes; it does not serialize raw response bodies, request headers, cookies or exception messages, add cookie/session reads, or introduce an authenticated client. The receipt includes local selected batter/name/game, results, corrections and streak state as documented. This is a source-bound conclusion for the actual constructed requests, not a claim about arbitrary malformed identity strings. Producer revision is sampled at publication and is not deployment proof. The documented backup location and two scheduled reconcile runs per day are not evidence of installed receipts or successful backups.

D remains **BLOCK in its first round**. Its smallest coherent revision is evidence-only: complete source/identity/disposition qualification, irreversible late/ambiguous-clock exclusion, deferred receipt processing, truthful successful/failed write records and shared publication closure. Keep requests, legacy grading decisions, retry/cadence and existing writer semantics separate from any proposed repair. The seven listed current-defence mutant patches still require frozen-runner re-certification on the final deploy candidate; `git apply --check` is not that re-certification.
