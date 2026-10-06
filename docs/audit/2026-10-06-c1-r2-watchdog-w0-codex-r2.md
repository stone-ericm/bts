> **Archive note (author, 2026-10-06):** the report below says the reviewer redirected "following Eric's override". That instruction (deselect the gate tests, use the author gate log, no further escalation) came from the author session (Claude) after the Claude Code auto-mode classifier denied approving the reviewer's unsandboxed-pytest escalation. It was **not** an instruction from Eric. The report is otherwise archived verbatim below.

# W0 production-code review, round 2 (final)

## Verdict

**BLOCK** at `ef1c91773f2f18a42f84ade47366f5199a8951b2`. The earlier symlink redirection, shared-temp collision, stale completion, UTC lease, check-isolation and queue-only defects received substantive fixes. Remaining counterexamples prevent accepting the notification foundation and the confinement gate: checker-failure recurrence is suppressed, mixed results can falsely recover a live fault, malformed notification entries are accepted and rewritten, and a caught confinement denial can leave the gate green.

This verdict does **not** arise from being unable to reproduce the author's kernel tests. Following Eric's override, I reproduced:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/watchdog -p no:cacheprovider -k 'not gate2'
36 passed, 7 deselected
```

The **43 passed** run in `.codex-review/c1-r2-watchdog/author-gate-run.log`, stamped `2026-10-06T05:07:40Z` at this SHA outside an agent sandbox, is **author-supplied evidence, not reproduced**. The claimed 14/14 mutant kills and 3815 fast-suite passes are also supplied evidence, not independent results from this review. The initial nested-sandbox suite attempt failed to apply the sandbox profile; no outside-sandbox test result was obtained. After Eric's override, no escalation or kernel-gate execution was requested.

I read the fix message/diff, round-1 report and relevant registration text. Independent probes used only synthetic temporary inputs, `python -B`, fresh bytecode-cache prefixes and in-memory substitutions. All DM calls were fake or patched; no network/DNS, credentials, real `data/`, configuration files or external context/memory directories were read. Tracked files were left unchanged.

This is the last W0 round. The remaining disposition goes to **Eric or his delegated herdr manager**; this report does not initiate a third round or authorize implementation, deployment or activation.

## B1-B7

| Item | Disposition | Evidence and limit |
|---|---|---|
| **B1: owned-root operations** | **Closed for the round-1 symlink redirects and atomic collision; namespace ownership remains an explicit limit** | `_dir` walks from the admitted directory descriptor with `O_DIRECTORY | O_NOFOLLOW`; mkdir, exclusive UUID temp creation, replace and fsync use admitted descriptors (`root.py:99–156`). Locks use no-follow descriptor-relative opens (`:186–209`). The reproduced suite exercises directory swaps, root replacement, a lock-leaf symlink and same-PID writes; both concurrent replacements now succeed whole with no leftover temp. Parent fsync accompanies newly created directories. See the ownership qualification below. |
| **B2: gate 2** | **Partially closed; acceptance remains open** | The audit hook was removed and replaced by a kernel-denial profile. Author evidence reports all seven gate tests passing. Static review and the EPERM simulation below expose an overly broad `/dev` exception, swallowed-denial false greens and weak red-control attribution. |
| **B3: episodes** | **Partially closed** | Business fault → verified → recurrence now yields three notices for the favorable same-identity case, and distinct target keys persist. Actual runner-generated checker failures cannot recover through ordinary results; mixed same-target results can recover an unresolved incident. N1/N2 below. |
| **B4: claims and leases** | **Fencing and UTC defects closed; elapsed-budget claim partially closed** | Unique claim tokens are checked before completion (`notify.py:193,220`). Reproduced stale-owner and fall-back fixtures pass. UTC addition/comparison fixes the original DST defects; rollback-before-claim explicitly permits retry. `max_sends` bounds count, but the budget uses wall time and cannot bound an individual send. N4 below. The “accepted before crash” test plants a persisted claim; it does not kill an actual sender process. |
| **B5: corrupt state** | **Partially closed** | Basic top-level/entry/status and sending-token checks exist and the supplied invalid-input cases fail closed. The validator still accepts unknown semantic notice states, wrong episode identity and invalid target dates; some accepted corrupt files are then rewritten. N3 below. |
| **B6: check isolation/reporting** | **Closed for round-1 cases** | Safe naming/formatting, transactional iteration, date/status/serialization checks and independent persistence/notification attempts address the recorded probes (`runner.py:48–115`). Hostile repr/exception text, partial iteration, bad output/date, result-write failure and corrupt-notifier cases pass. CLI returns 1 for captured persistence/notification-state exceptions. Delivery failures returned in `notify_report` remain pending and are not included in `JobOutcome.ok`; this distinction matters to the gate. |
| **B7: no-send** | **Closed** | CLI always constructs the queueing notifier and skips only transport (`cli.py:40–46`). The reproduced CLI and later-flush fixtures retain and subsequently send a pending notice without a recipient being present at enqueue time. |
| **Concurrency test fix** | **Closed for the original false green** | The contender must finish before the first sender is explicitly released (`test_w0_skeleton.py:371–398`). An independent in-memory send-under-state-lock mutant now fails with `AssertionError: a contending flush blocked behind an in-flight send`. Probe cleanup confirmed no surviving task workers. |

**B1 qualification:** descriptor anchoring prevents a replacement symlink from redirecting the operation to its target. It does not keep an admitted inode at its original pathname. The root-replacement fixture expressly writes to `data/watchdog.moved`, a sibling of the declared `data/watchdog`, and `root.py:15–16` documents outside-actor relocation as a limit. This proves inode anchoring, not refusal of every post-admission layout change or perpetual pathname confinement. The watchdog itself does not rename directories; preservation of the admitted namespace must be an ownership/deployment assumption, or enforced by the runtime boundary. Also, `write_atomic` replaces an existing destination symlink rather than following it; the universal claim that every symlink is refused is stronger than the implemented safe-replacement behavior. Neither observation recreates the original write-through-symlink defect.

## Gate 2

**What improves:** `conftest.py:28–39` uses kernel filesystem denial instead of selected Python audit events. Under the intended profile, ordinary outside pathname writes, symlink-following opens and descriptor-relative opens should be mediated regardless of Python thread, subprocess or library implementation. `close_fds=True` and captured stdio remove inherited application file descriptors in this child. The author log supports the expected controls on this host; I did not reproduce their kernel execution.

**G1 — denial is not a record that an outside write was attempted.** The main gate test requires return code 0, `GATE-OK`, an unchanged planted pick and at least one result file (`test_w0_skeleton.py:585–588`). It does not assert the intended statuses, confirmed fake IDs or absence of unexpected denials. W0 catches check exceptions and transport exceptions; the latter become pending notices while `JobOutcome.ok` remains true.

I substituted this extra operation, in memory only, immediately before `self.send(...)` **inside its existing try** in `Notifier.flush`:

```python
open(self.root.path.parent / "picks" / "2026-10-06.json", "w").write("outside-mutant")
```

I then executed the exact `GATE_CHILD` CLI program on synthetic inputs in two ordinary child processes, without `sandbox-exec`:

| Probe | Independent result |
|---|---|
| Simulate the profile's EPERM at this one outside open | Return code **0**, `GATE-OK`, original pick bytes `{}`, five result files; **all the main test's assertions pass**. |
| Same in-memory mutant with that simulated denial removed | Return code **0**, `GATE-OK`, five result files, pick bytes **`outside-mutant`** outside the watchdog root. |

This measures the runner/test's treatment of EPERM, **not macOS kernel behavior**. The inference from the static profile is straightforward: its outside-open denial supplies the same caught `PermissionError`. The kernel protects the test's file while the code still attempts a forbidden mutation; normal unsandboxed production permits that mutation. Therefore this test is not a sound general acceptance instrument for W0's production write paths merely because its protected child exits successfully. The supplied outside-write mutant does not clear this differently placed, caught write.

**G2 — `/dev` allows more than stdio.** `(subpath "/dev")` allows every matching device path, including unrelated terminal devices where OS permissions permit access. W0 uses inherited captured pipes; it does not need permission to open/write every device. No real device was probed. Narrow the exception to the exact necessary device operations and test stdout/stderr explicitly.

**G3 — red controls can attribute the wrong failure.** In the initial nested run, four pathname red tests passed although their process returned **71** with `sandbox-exec: sandbox_apply: Operation not permitted`, before their Python code ran. Their assertion accepts any nonzero exit containing that phrase (`:619–620`). The main positive gate failed, so that run was not a green full suite; nevertheless those individual “kernel refused this write” claims were false. The subprocess control likewise treats any descendant nonzero exit as refusal without requiring a caught EPERM from its intended open. Each control needs a child-start witness and an operation-specific refusal witness. Additionally, `RED["same_byte"]` opens the destination in `w` mode before evaluating the argument's read, so without denial it truncates first; read the bytes before opening for write to make this an actual same-byte control.

**Production equivalence is still limited.** The child directly invokes Click in an existing interpreter with `-B`, a sanitized environment, fixed TZ/CWD/PYTHONPATH, fake DM transport and pipe stdio. It does not execute cron or `uv run`; no watchdog cron line is installed by the current script. A later launcher may synchronize an environment, write bytecode/cache files or redirect output to a log outside the owned root. These are deployment boundaries to specify and test, not observations of current production mutations. The test profile is not installed as a production runtime policy. Real DM authentication is intentionally replaced by the fake transport and therefore is not exercised by this filesystem run. The profile also allows network by default and cannot certify remote-upload absence. Current W0 introduces no upload path; network/remote-operation coverage belongs to acceptance of later actual operations.

## New findings

**N1 — real checker-failure recurrence is still suppressed (B3).** The failure result uses the callable name, `incident="checker:<name>"` and no selection (`runner.py:91`). A normal successful result uses the check's logical ID and often a selection. Recovery matching requires identical check/date/selection (`notify.py:154–155`). Using the **same callable** across three job runs produced:

```text
transient: checker_failure
W-test, selection=s: verified
transient: checker_failure
```

Only **one** fake send occurred. The `checker:transient` target remained open at episode 1 after the success; the third run was suppressed. This is the actual runner-to-notifier path, not a manually manufactured checker-failure result. Successful execution must close its prior execution-failure episode independently of whether its business result is fault, pending, unverifiable or verified.

**N2 — recovery can be false, order-dependent and mislabeled (B3).** Enqueueing a fault for `I-bad` followed by a verified result for `I-other`, with the same check/date/selection, immediately closes `I-bad`. Repeating that identical input produced **four** fake messages and left the still-faulting target marked recovered at episode 2. Recovery text carried **`[I-other]`**, though the notice's target was `I-bad`. On the second iteration, recovery was sent before the fault because both notices shared `queued_at` and loaded hash-key order broke the tie. No documented input validator rejects mixed results. A verified result for another predicate in the same batch is not a later complete observation proving this incident repaired. Resolve recovery from the whole target observation, preserve the recovered incident's identity and retain causal notice order.

**N3 — entry validation is incomplete (B5).** Starting from a real generated queue, each following independent change was accepted by `load_state`:

| Corruption | Subsequent behavior |
|---|---|
| Notice `state="unknown-state"` | Flush sent a fake ID and rewrote the file. |
| Notice episode changed to 999 without changing its target/key | Flush sent a fake ID and rewrote the file. |
| Target `open=True`, `state="recovered"`, `et_date="not-a-date"` | Flush sent a fake ID and rewrote the file. |
| Remove notice `queued_at` | Load accepted it; flush raised `KeyError`. Original bytes survived. |
| Set `targets=[]` | Load raised `AttributeError`, not `NotifyStateError`. Original bytes survived. |

The first three contradict the advertised corrupt-state preservation policy. `_validate` checks delivery status but not the notice's semantic state/episode/key, container shapes, canonical target date, required fields or cross-entry consistency (`notify.py:73–96`).

**N4 — the budget is a wall-clock admission check (B4), not an elapsed execution bound.** `notify.py:199–207` subtracts the ET/UTC wall seam rather than an injected monotonic clock and does not interrupt the send. With an injected rollback after each fake send, a 90-second budget sent **20** notices, skipped **zero** for budget and stopped only at `max_sends`. No actual send runtime was measured by that probe. A hanging send can still hold the job singleton indefinitely, and the real DM transport contains several sequential/retrying HTTP steps without a whole-flush deadline. Keep the accurate count-bound claim; fix elapsed admission with a monotonic seam. The registered two-minute whole-job/transport bound remains a later complete-watchdog prerequisite, as in round 1.

## Required changes

The smallest corrective scope is the existing W0 notification protocol and its acceptance harness, without adding business predicates:

1. **N1:** bind checker execution failures and successful completion to a stable registered invocation identity, distinct from business result check/date/selection. A successfully completed invocation must recover that invocation's checker-failure episode. Prove error → successful verified **and successful business-fault** execution → error through `run_job`, with a new recurrence notice.
2. **N2:** compute recovery from a complete target observation before applying transitions. An alert for that target in the same observation must prevent recovery; support incident-specific verification separately from an explicit all-incidents-clear result. Construct recovery identity/text from the target being recovered. Give notices a persistent causal order rather than relying on wall-time ties and hashes. Add mixed-result/order-reversal and queue-only backlog controls.
3. **N3:** validate both containers and every required target/notice field before acting. Validate canonical dates, nullable identity fields, exact integer counts/episodes, semantic state and open/closed consistency, timestamps, claim metadata, recomputed keys and target/episode relationships. Historical notices may legitimately have older episodes; their relationship must be valid rather than simply equal to the latest episode. Every invalid case must raise `NotifyStateError` and preserve bytes before sends or state writes.
4. **G1–G3:** make denied outside attempts an acceptance failure independent of swallowed application errors, or pair the denial harness with independent complete attempted-mutation attribution. At minimum, assert the gate's exact expected statuses/notice identities and confirmed fake deliveries, and add the caught-send-write mutant above. Narrow `/dev`; require child-start and intended-operation EPERM witnesses for red controls; change the same-byte control to read before opening for write. A clean exit and unchanged bytes alone cannot certify production confinement.
5. **N4:** use an injected monotonic seam for the flush admission budget while retaining UTC wall time for persisted leases. Test wall rollback independently of monotonic progress. Keep whole-job/send cancellation and production launcher/stdout/cache ownership explicitly pending for the complete deploy gate; do not claim this W0 budget establishes them.

Document B1's admitted-inode/namespace ownership assumption and narrow the universal symlink-refusal wording to the implemented behavior. The author-supplied gate evidence can remain recorded with its provenance; it does not require or authorize this reviewer to run outside the agent sandbox.

**Freeze this review here and obtain Eric/delegate disposition under the two-round pace rule.** No automatic W0 round 3, implementation, deployment, commissioning or A8 activation follows from this BLOCK.
