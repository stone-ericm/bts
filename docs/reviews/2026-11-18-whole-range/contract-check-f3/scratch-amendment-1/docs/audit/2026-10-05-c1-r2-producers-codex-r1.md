## Verdicts

A: **SIGN WITH EDITS.** Apply A1 and A2 verbatim below; narrow the categorical transport claim as specified in A3.
B: **BLOCK.** B1 can erase foreign cron jobs after a filtering error.
C: **BLOCK.** C1–C6 leave identity, secrecy, timing, behavior-preservation and publication contracts unsatisfied.

Reviewed detached HEAD `aeb56dace06881fe3f56253df43229877282c603`, including the messages and diffs of `04cde0a`, `1c0e759` and `5f5217e`, the build plan, registration R4, producer contracts and gates 4–5. The exact permitted pytest command completed with **133 passed in 9.45s**, rather than the prompt's stated 84. Independent synthetic probes and their final results are retained in `probes-r1.py` and `probe-results-r1.jsonl` in this directory. The probes use fake auth/HTTP/DM leaves, temporary pick files and the existing cron tests' throwaway-HOME fixture; no real crontab was run. Findings below distinguish reproduced synthetic failures from source-derived limits. No tracked files were edited, and no deployment, configuration change or activation is approved. D7, subsequent commissioning and A8 remain separate.

Scope deviations: the batch that first loaded the prompt also searched an external memory registry and listed external topic filenames before its checkout-only restriction had been read. Those results were excluded from the review evidence. Two preliminary executions of the baseline-comparison probe retained the original public lookup functions in a copied globals dictionary and encountered a DNS failure; authenticated leaves were patched. The final probe binds those lookups to the fixture and denies HTTP at the client boundary. Only the corrected run supplies the baseline comparison reported below. No successful external response or credential-file read was observed in those probes.

## Item A

**A1 — Medium: the private-mode test survives a private-to-DM regression.** `tests/test_private_mode_transport.py:33` and `:43`.

Counterexample: patch `_pick_delivery_mode` to return `"dm"` and invoke the actual decorated test for each of its three configurations. All three pass. Each fixture lacks `bluesky.dm_recipient`, so `_deliver_and_lock_pick` returns False before calling the patched DM transport (`src/bts/scheduler.py:1045`). The test never checks the return value or lock state. Likewise, replacing the entire delivery helper with a no-op would satisfy its current assertions. This is a false green in the prerequisite itself, although the actual eligible private-lock branch currently avoids both pick transports.

Required exact edit: replace the body of `test_private_mode_never_touches_a_transport` with:

```python
    config = {**config, "bluesky": {"dm_recipient": "test-only.bsky.social"}}
    post.return_value = "at://test-only/app.bsky.feed.post/1"
    dm.return_value = "test-only-dm-id"
    state = _state()
    with patch("bts.scheduler._now_et", return_value=datetime(2026, 4, 6, 15, 0, tzinfo=ET)):
        daily = _daily()
        locked = _deliver_and_lock_pick(daily, config, tmp_path, state, "2026-04-06", "test")
    assert locked is True and state.pick_locked is True
    post.assert_not_called()
    dm.assert_not_called()
    assert daily.bluesky_posted is False and not daily.notification_sent
```

The current helper must pass; the forced-DM and no-op counterexamples must fail. Both actual transport leaves remain patched. A separate synthetic check of these edited assertions accepted the actual private lock and rejected forced DM, forced public and no-op behavior; the DM/public controls reached their fake transports.

**A2 — Medium: C2 also rejects an already-safe legacy private config.** `src/bts/scheduler.py:891`.

Counterexample: `{"scheduler": {"private_mode": True, "shadow_mode": True}}` resolved to private before `04cde0a` and now raises ValueError. It is not the shadow-only configuration that fell through to public posting. The refusal unnecessarily changes a previously working private configuration even though the resolver documents `private_mode` as a delivery control.

Required exact edit: replace the condition at line 891 with:

```python
    if raw is None and "shadow_mode" in sched_config and not sched_config.get("private_mode", False):
```

Add this exact test:

```python
@pytest.mark.parametrize("shadow_mode", [True, False])
def test_legacy_private_mode_remains_private_with_an_unrelated_shadow_key(shadow_mode):
    assert _pick_delivery_mode({"scheduler": {"private_mode": True, "shadow_mode": shadow_mode}}) == "private"
```

Keep the existing shadow-only refusal tests for both True and False, the explicit-mode precedence test and the early `run_day` refusal test. The other delivery modes, aliases and precedence are unchanged by this diff.

**A3 — Claim scope: private recommendation delivery is not a blanket prohibition on operational DMs.** `src/bts/scheduler.py:977`, `:995`, `:1015` and `:2182`.

The existing unknown-send and late-cutoff paths can send operational health DMs before reaching the normal private-lock branch. Do not repair those paths as part of this prerequisite: registration R4 distinguishes operational notifications from recommendation delivery. Required exact documentation edit: replace the C1 description at `tests/test_private_mode_transport.py:3`–`:4` with this sentence, and use it when describing the deployment candidate:

> C1: an eligible private recommendation lock succeeds without calling either pick-delivery transport; existing operational health-alert DMs are outside that assertion.

C2's immediate refusal and private recommendation delivery are otherwise supported by the source and suite, subject to the edits above. This is not evidence that every future delivery-mode harness is safe without its own patched DM/auth leaves.

## Item B

**B1 — High: filtering failures are treated as an empty retained crontab.** `scripts/cron-setup-hetzner.sh:109`, `:128` and `:141`.

Counterexample: use the existing `box` fixture, plant `0 9 * * * echo foreign-job`, select agreeing research/private intent, and make the throwaway `grep` shim return 2 only for `-v`. The real script prints the synthetic filtering error, exits **0**, installs BTS lines and deletes the foreign job. `|| true` conflates grep's expected no-selected-lines status 1 with an actual filtering error. The same helper serves removal. The existing shim tests cover a failed `crontab -l`, but not a failed filtering step, so they remain green.

Smallest fix: retain status 1 as the empty-result case and refuse every other filtering error before invoking `crontab -`. For example, replace the successful-read branch with:

```bash
    if out="$(crontab -l 2>"$err_file")"; then
        rm -f "$err_file"
        if [ -z "$out" ]; then
            return 0
        fi
        local filtered filter_status
        if filtered="$(printf '%s\n' "$out" | grep -v -- "$MARKER")"; then
            printf '%s\n' "$filtered"
            return 0
        else
            filter_status=$?
            if [ "$filter_status" -eq 1 ]; then
                return 0
            fi
            echo "ERROR: cannot filter the current crontab" >&2
            return 1
        fi
    fi
```

Add throwaway-HOME install and remove cases injecting grep status 2 and asserting nonzero exit, no `crontab -` invocation and byte-for-byte retention of foreign lines. Keep the empty/missing/BTS-only crontab controls. A temporary copy with the replacement above refused both injected-error actions without changing the planted crontab and passed missing-crontab install, BTS-only replacement and removal controls. This is a local failure-path fix, not permission to run real crontab.

The core intent contract is otherwise met: `src/bts/entry_intent.py:39` enforces the exact strings with no default, and `:47` delegates effective delivery to the scheduler resolver. The real CLI is exercised by the cron tests' uv shim; it is not a shim that fabricates an intent. The script quotes the checkout, uv executable and config path in its validation command (`:56`) and does not evaluate captured stdout as shell code. An independent `enter\nextra-output` stdout probe was refused with the crontab untouched. Missing, malformed and disagreeing intent also refuse before installation. The scheduler does not consume `entry_intent`.

Deployment does not install cron. The missing-key migration is fail-closed and adequately documented at checklist A5 (`docs/ops/2027-season-start.md:14`), `CLAUDE.md:26` and plan P4: an authorized later research install requires research/private agreement; activation uses the separately authorized enter/dm configuration. No actual box configuration was read. These positive conclusions do not clear B1.

## Item C

**C1 — High: a batter/date match is promoted to game-qualified confirmation without a unit/game proof.** `src/bts/cli.py:1810`, `src/bts/entry_receipt.py:125` and `docs/ops/pick-entry-receipt-v1.md:52`.

Counterexample: a committed pick for MLB batter 1/game 1, pending row `{roundId: 7, playerId: 100, unitId: 999, number: 1, result: null}`, date lookup for the target date and player crosswalk `{100: 1}` produce `observed`, `verifier.ok = true`, `reason = "match"`, a confirmed marker and a pre-cutoff observation. No consumed lookup resolves unit 999 to game 1. All documented positive-confirmation conditions can pass. The existing verifier compares batter sets only (`src/bts/contest_fetch.py:146`); preserving that legacy behavior is permitted, but using its result alone for the new receipt's stronger game-bound confirmation is not. This fails R4 line 30 and the entry contract at line 58. The current no-fault test uses coincidentally equal fixture unit/game numbers, without any validated mapping.

Smallest fix: keep the old marker/DM verifier unchanged, but add separate receipt qualification for each observed slot's batter and game/validated unit, including the consumed mapping's identity/hash. A missing or ambiguous mapping must explicitly remain unverified, and W-entry's consumer rules must require that qualification rather than just the legacy `match`. Use existing authorized lookup evidence; do not add an authenticated request. Add producer-shaped correct-unit, wrong/unknown-unit, wrong account/date/season, changed selection and missing-DD cases, with wrong/unknown units refused as confirmation even when the old verifier says match.

**C2 — High: file hashes can identify bytes that the command did not consume.** `src/bts/cli.py:1703`, `:1718`, `:1724`; `src/bts/entry_receipt.py:68`–`:76`.

Counterexample: let the actual `load_pick` finish loading delivered batter 1, then replace the synthetic pick file with delivered batter 9 before it returns to the command. The receipt's selection and verifier still describe batter 1, but its `pick_file_sha256` equals the current batter-9 file. The schema's hash shortcut can therefore present an old recommendation check as confirmation for a changed current file. Decision loading and hashing likewise happen in separate reads, and the scoreable gate precedes another decision load. No writer closure or consumed-byte binding covers those gaps. This is not a measured production race; it is a reproduced interleaving admitted by the real producer path.

Smallest fix: bind hashes to the exact pick and decision bytes used to parse the selection and authorize the check, carrying those bytes/hashes through the run. Do not manufacture a reference by rereading the path later. If byte capture cannot be qualified, make the receipt identity unverifiable. Remove the unsafe hash-only confirmation shortcut until its consumed-byte premise is implemented. Add interleavings between load/gate/identity/hash, including decision changes, without adding a production-writer lock or changing the legacy decision.

**C3 — High: raw rows admit auth secrets into a receipt.** `src/bts/entry_receipt.py:86`–`:87`, `:128`; `tests/test_entry_receipt.py:200`.

Counterexample: add `xsid: "SYNTHETIC_SESSION_SECRET"` and `token: "SYNTHETIC_TOKEN"` to an otherwise valid pending row. The real producer succeeds and persists both values verbatim in `observation.rows.pending`. Nested profile slot rows have the same unrestricted-copy problem. This is a synthetic payload-drift counterexample, not a claim that the live API currently returns those fields. The no-secret test searches fixtures that never contain a secret in a copied row, so it cannot establish the stated guarantee.

Smallest fix: emit an explicit, typed allowlist of needed round/slot evidence fields, including nested profile slots; do not copy arbitrary response dictionaries or nested metadata. Keep source hashes without exposing source contents. Test sentinel cookies/session IDs/tokens at the row and nested-slot levels, plus an HTTP exception containing a secret-bearing message/URL. Assert those values never occur in the serialized receipt. The existing handled-error path at `entry_receipt.py:138` correctly stores only type/status, and `cli.py:1689` stores only the raised type; preserve that behavior.

**C4 — Medium: response completion is timestamped after verification.** `src/bts/cli.py:1792`, `:1810`–`:1813`; `src/bts/entry_receipt.py:119`.

Counterexample: the final lookup returns at 19:04:59 ET, then the actual verifier wrapped with a two-second synthetic processing interval finishes at 19:05:01. The receipt records `response_completed_at = 19:05:01` and `before_cutoff = false` for a 19:05 cutoff. That is verifier completion, not the registered actual response-completion time. The existing fake clock returns the same forty-minute value on every call after its first; it hides extra clock reads and does not distinguish these intervals. Ordinary receipt tests also use ambient process elapsed time under `--now-et`.

Smallest fix: capture the timezone-aware completion instant immediately after the last fetch returns and pass it into receipt assembly; keep verifier completion separate if useful. Fix both wall and monotonic clocks, including dependent calls, in fixtures. Add pre-cutoff responses followed by post-cutoff verification, exact cutoff equality, genuinely late responses, and an earlier-DD cutoff. Preserve the original nag deadline calculation and legacy marker decisions.

**C5 — High: receipt construction is not behavior-neutral on failure.** `src/bts/cli.py:1724`, `:1798`, `:1812`; the protected `publish` block begins only at `src/bts/entry_receipt.py:156`.

Counterexample: inject MemoryError into receipt-only `payload_sha256`, with an observed missing entry. Under the pre-P1 command extracted from `5f5217e^`, the same patched inputs produce one DM, an alerted marker and exit 1. Under the reviewed command, the added receipt call raises before those actions: zero DMs, no marker and MemoryError. The exit number alone is still 1, a particularly misleading green. The publication-failure test patches the final writer, after the business path has already completed, and misses construction failures.

Smallest fix: keep the original decision/DM/marker path authoritative and isolate all receipt-only preparation failures. Defer extra extraction/hashing/serialization to protected publication work; retain the actual response timestamp and consumed-input identity without running fallible receipt calculations ahead of the original actions. Incomplete receipt construction must yield unavailable evidence, not partial successful observation fields. Add faults at identity capture, observation assembly, hashing and serialization and compare DM calls, statuses, exits and retryability against the pre-P1 body. Do not swallow original business-path exceptions.

**C6 — High: a reported failed publication remains consumer-visible as success; new directory ancestry is not fully synced.** `src/bts/entry_receipt.py:46`, `:52`–`:55`, `:162`; `tests/test_entry_receipt.py:211`.

Counterexample: allow the marker and receipt-file fsyncs, then raise OSError at the receipt-directory fsync after `os.replace`. The command reports `entry receipt unavailable` and exits 0, yet a final JSON file remains with `outcome = "observed"`, a matching verifier, pre-cutoff observation and the marker's receipt ID. A consumer following the schema cannot see that publication failed and can accept it. The current failure test raises before any file is published. In addition, `mkdir(parents=True)` creates receipt/date directories but only the leaf directory is fsynced; the claimed crash durability of newly created ancestors is not established. That second point is source-derived, not a power-loss measurement.

Smallest fix: make failed final publication unavailable to discovery, cleaning any renamed final file and temporary file on failure. If invalidation/cleanup also fails, retain an explicit failure state that consumers must honor; printing stderr cannot invalidate an otherwise qualified receipt. Durably establish newly created directory entries by syncing their parents. Add failures at write/file fsync/rename/directory fsync and cleanup, plus first-use directory creation. Assert a failed attempt cannot be accepted from a leftover final file. Keep marker statuses and legacy exits unchanged.

Other checked behavior and limits: the marker's added `receipt` key is compatible with the in-checkout readers. The command uses named `.get` fields and the health reader (`src/bts/health/pick_entry.py:66`) selects date/status/reason; no exact-shape consumer was found in `src/`, `scripts/` or `tests/`. The already-confirmed path makes no fetch and references the original receipt, with null references correctly unavailable for old markers. Existing retry arguments and HTTP fetch sites are unchanged; the new leaf-count test establishes helper invocation counts, not independently measured wire requests, while the diff supplies the no-new-fetch evidence. Unchanged legacy tests support normal-path DMs, statuses, escalation and exits, but do not clear C5.

Volume is documented honestly: the installed enter cron generates 56 runs/files per full day, including no-attempt outcomes; there is no pruning. The default receipt tree lies in the configured restic ops path (`src/bts/data/backup.py:49`). No real size, retention, backup success or installed behavior was measured. Publication-failure warning behavior and the otherwise appropriate atomic-write sequence do not clear C6.

Re-certification remains a deployment-candidate gate, not a completed claim. The changed call graph does not add a call from entry checking into `fetch_contest_streak`; the schema explicitly dispositions the same-file certificates `D-I063-2` and `I-0811-b` for rerun. They were not rerun under this prompt's permitted-suite boundary. Do not call them current until the frozen tooling reruns them on the final corrected deploy candidate. `producer.revision` currently samples checkout HEAD at publication; it is not, by itself, proof of the executing artifact or of deployment. No absence certificate, D7 approval or A8 activation follows from this review.
