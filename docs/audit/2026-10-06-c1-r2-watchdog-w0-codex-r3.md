# W0 code review, round 3 — final delegated round

## Verdict

**BLOCK** at `c429c41406cc0162ed6acc76fcbcb7a0276c3baf`. The fixes address the original single-checker recurrence, mixed-observation recovery, malformed-entry examples, wall-clock budget and same-process swallowed-denial control. The changed notification protocol still permits false checker recovery, reversed lifecycle delivery and silent loss of a queued alert. These independently reproduced failures prevent signing the foundation. The gate also retains an unproved claim about descendant violations.

Under `C1-r2-w0-review-r3` in register §C (`docs/audit/2026-09-22-exposure-register.md:96`), this verdict **defers the watchdog this cycle, with no fourth round**. The required changes below are a record for a later authorized cycle, not permission to repair, continue building, deploy or activate it. Whole-watchdog SIGN and D7 remain prerequisites for deployment.

Independently reproduced permitted suite:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/watchdog -p no:cacheprovider -k 'not gate2'
50 passed, 9 deselected in 0.61s
```

The **59 passed** run in `.codex-review/c1-r2-watchdog/author-gate-run-r3.log`, stamped `2026-10-06T05:30:14Z` at the pinned SHA outside any agent sandbox, is **author-supplied evidence, not reproduced**. That includes all nine gate2 tests. The commit's 8/8 mutant kills and 3831 fast-suite passes are also supplied evidence, not independently rerun here. No kernel gate was executed and no escalation was requested. Inability to nest `sandbox-exec` is not the reason for this BLOCK.

I read the complete round-3 prompt first, the fix message/diff, the round-2 report and the applicable register row. Additional probes used synthetic temporary inputs, `python -B`, a fresh bytecode-cache prefix and fake transports. No memory/external-context search, real data/configuration/credential read, network/DNS, SSH, GitHub call or DM was performed. No tracked file was changed.

## N1-N4, G1-G3

| Item | Disposition | Evidence and remaining limit |
|---|---|---|
| **N1: execution recovery/recurrence** | **Partially closed** | The runner passes successful execution names separately from business results; the same-callable error → business fault → error test passes (`runner.py:80–119`, `test_w0_skeleton.py:695–713`). But the purported registered identity is only `__name__`, or the job-local fallback `check#i`. Another callable with that name can close a still-failing invocation, including across jobs. R3-1. |
| **N2: complete observation, recovery identity, order** | **Observation and text closed; delivery order partially closed** | Grouping by check/date/selection, the alerting guard and incident-specific/all-clear distinction fix the recorded mixed-result cases (`notify.py:201–215`). Recovery text uses the recovered target (`:238–240`). Persistent `seq` fixes hash/timestamp ties, and the queue-only backlog test passes. But sorting only the currently due notices does not preserve lifecycle delivery order after a failed or live predecessor. R3-2. |
| **N3: validation and preservation** | **Partially closed** | The reproduced semantic-corruption fixtures now reject unknown states, impossible/re-keyed episodes, contradictory targets, missing fields, malformed containers, boolean attempts, wrong keys and misplaced claim fields (`test_w0_skeleton.py:772` onwards). Keys, types and forward references receive substantive validation. Reverse target-to-notice completeness is absent: an open target without its queued notice passes and permanently suppresses the same fault. R3-3. |
| **N4: elapsed admission budget** | **Closed for the requested seam** | `Notifier` injects monotonic time and checks elapsed monotonic duration before each send (`notify.py:188–192,281–286`). The rollback fixture passes: monotonic progress exhausts the 90-second admission budget despite wall-clock rollback. UTC remains the persisted lease basis. The documentation accurately excludes interruption of an in-flight send and the whole-job deadline; those remain complete-watchdog/deploy prerequisites. |
| **G1: denied attempts must fail acceptance** | **Same-process control closed on supplied evidence; descendant claim partially closed** | The kill profile and exact state/episode/delivery/attempt summary materially improve the gate (`conftest.py:35–49`, `test_w0_skeleton.py:551–633`). Author evidence reports the caught-write mutant killed with -9. The harness observes only its top child's exit; it does not observe a fatal refusal in an ignored descendant. A synthetic killed-descendant model satisfies every current positive-gate assertion. R3-4, with kernel inference explicitly separated from measurement. |
| **G2: device exception** | **Closed statically** | The broad `/dev` subpath is replaced by literals for `/dev/null` and `/dev/dtracehelper` (`conftest.py:30–41`). The latter's startup necessity is author-supplied empirical evidence, not a reproduced device experiment. No other device path is admitted by this exception. |
| **G3: attributable red controls** | **Closed for the recorded defects** | Plain-EPERM controls now require return code 0 plus child-start and operation-refusal witnesses; the subprocess body checks both grandchild witnesses. The same-byte control reads before opening for write. A separate kill-profile control requires start, -9/137 and no survival witness (`test_w0_skeleton.py:636–690`). Static inspection and supplied passing results support closure; I did not execute the kernel controls. The descendant acceptance gap is distinguished under G1. |

**B1 documentation is closed.** `root.py:15–24` now distinguishes refused directory/lock/read symlinks from atomic destination-symlink replacement, and explicitly states admitted-inode anchoring plus namespace ownership as a deployment assumption. The original descriptor/race controls pass in the permitted suite. This is not a claim that the admitted inode is pinned to its pathname against an outside actor's rename.

## New findings

**R3-1 — a different checker can falsely recover a checker that is still broken.** The new `executed_ok` mechanism identifies an invocation by the callable's Python name (`runner.py:50–57,102–107`), despite `_run_check` calling it a registered name. `JOBS` is a mapping of jobs to plain callable lists, with no globally unique checker registration or duplicate-name rejection (`cli.py:19`). Recovery then matches only that name and `checker:<name>`, without job identity (`notify.py:216–219`). Equal names are ordinary for wrappers, lambdas, callable objects' fallbacks and functions in different modules.

Independent probe: two different callables both named `check`; one always raises, the other successfully returns a verified business result. Running both in one job twice produced:

```text
run 1 results: checker_failure, verified; outcome.ok=True
run 2 results: checker_failure, verified; outcome.ok=True
notices: checker_failure ep1, RECOVERED ep1, checker_failure ep2, RECOVERED ep2
final checker:check target: open=False, state=recovered, episode=2
```

The broken callable never succeeded. A second probe ran the failing callable in `broken-job`, then the different successful callable in `other-job`: its checker-failure target likewise became recovered with no open target remaining. These are actual `run_job` → `Notifier.enqueue` results, using queue-only mode. They defeat N1's execution identity and manufacture recovery/recurrence churn while the checker remains unavailable.

**R3-2 — recovery overtakes an unresolved fault notice.** `flush` sorts due entries by `seq`, claims all of them, and continues sending after a failure (`notify.py:271–294`). It excludes a live sending predecessor from the due set rather than treating that predecessor as a causal dependency. Sequence numbers therefore order attempts within a selected batch, not successful lifecycle delivery.

Two independently measured schedules, each for one target:

| Synthetic schedule | Confirmed fake-delivery order and persisted state |
|---|---|
| Queue fault then recovery; first fault send raises before any acceptance; recovery send succeeds; flush again | First flush: fault pending, recovery sent. Second flush: fault sent. **Accepted order: recovery, fault.** |
| Pause the fault's fake sender before returning an ID; enqueue recovery; a second notifier flushes while the fault's lease is live; release the first sender | Both notices eventually sent. **Accepted order: recovery, fault.** No expired lease, clock rollback or crash is involved. |

The second probe used events to establish the interleaving and completed both threads successfully. This is not the documented ambiguity of an accepted send followed by a crash. The first schedule is a definite send failure; the second permits overtaking during a valid claim. An operator can receive RECOVERED followed by an obsolete fault for the same episode. The new backlog test covers only successful sequential sends.

**R3-3 — deleting a notice is accepted, rewritten and silences the alert.** Start with a real queue-only fault generated by `enqueue`: one open target, one notice and `seq=1`. Remove just the `notices` entry, retaining the otherwise valid target and sequence counter. This state cannot result from the current queue protocol, which never prunes notices.

```text
load_state: accepted
enqueue identical fault: new_notices=0
final state: seq=1, notices={}, one open target
original corrupt bytes: rewritten
```

`_validate` checks each notice's target, but never requires a target's current episode/state to have its corresponding notice and never checks the complete retained sequence (`notify.py:102–150`). `_alert` sees an already-open target in the same state and does not call `_queue` (`:223–235`). Repeating that live fault remains silent. Validation must detect this impossible state before either suppression, sends or a state rewrite; recomputing existing keys alone cannot do that.

**R3-4 — killing an offending descendant is not observed as a gate failure.** The profile applies a SIGKILL action to a denied write, while `run_confined` captures only the top interpreter's completion (`conftest.py:40,48–49`). The positive gate tests its zero exit, marker, summary, planted bytes and five result files. There is no descendant-violation receipt or supervisor propagating a descendant's failure. The kill controls exercise the top child; the subprocess control uses plain EPERM.

I used the exact `GATE_CHILD` program and all five assertions of the positive gate in two ordinary temporary child processes. An in-memory extra wrapper around `Notifier.flush` launched a descendant with `subprocess.run(..., check=False)` and continued into the original flush:

| Descendant operation | Independent parent/harness result |
|---|---|
| Self-SIGKILL, modeling the offending descendant dying | All five descendant returns were **-9**. Top return **0**, `GATE-OK`, exact expected notice/delivery summary, original pick `{}`, five result files. **Every positive-gate assertion passed.** |
| Ordinary write to the same synthetic pick outside `data/watchdog` | Descendant returns **0**; top return **0**, same marker/summary/five results; pick became **`outside-mutant`**. |

**Measured:** the harness accepts an ignored killed descendant while all expected business behavior completes; the paired ordinary descendant actually writes outside the owned root. **Inferred from the profile/process boundary, not reproduced on macOS:** a descendant denied its write can be the killed process while its parent continues. The current gate supplies no evidence that its SIGKILL action terminates the top interpreter or whole process tree in that case. This is an open acceptance boundary, not a claimed kernel observation. The three notification counterexamples above establish BLOCK independently of this inference.

Probe command:

```text
PYTHONPATH=src:. PYTHONPYCACHEPREFIX=/private/tmp/c1-w0-r3-pycache-20261006-a \
  .venv/bin/python -B /private/tmp/c1-w0-r3-probes-20261006.py
ALL-SYNTHETIC-PROBES-PASSED
```

The probe's temporary synthetic trees were removed on exit. The script is review scratch, not production code or an outside-sandbox gate run.

## Required changes

These are requirements for a later owner-authorized cycle. They are not a verbatim SIGN WITH EDITS: the remaining identity and delivery changes require implementation and meaningful verification.

1. **R3-1 / N1:** provide an explicit, stable invocation identity with collision rules at registration. Bind checker-failure opening and successful-execution recovery to that same identity, including the job scope where registrations are job-local. A callable's display name alone is insufficient. Test distinct same-name callables in one job and across jobs: success of the other callable must not close the failing one's episode; genuine success and subsequent recurrence must still work.
2. **R3-2 / N2:** enforce predecessor completion for notices of the same target across failures and concurrent flushes. A later recovery or recurrence must not overtake an earlier unsent/live-claimed notice. Keep independent targets able to progress and preserve token fencing, short locks and count/monotonic budgets. Test both definite predecessor failure and the live-claim interleaving above, checking confirmed delivery order rather than just sorted state entries.
3. **R3-3 / N3:** validate reverse lifecycle completeness and the retained sequence invariant before acting. Every current target episode/state must have the notice implied by the protocol; deleted notice/sequence holes must fail closed under the current no-pruning format. If pruning is introduced later, it needs an explicit validated retention representation. The deleted-notice probe must raise `NotifyStateError`, preserve the exact bytes and attempt no fake send or state rewrite.
4. **R3-4 / G1:** make a denied attempt in any admitted descendant fail acceptance regardless of whether the parent ignores that process's return. Either enforce and test a process scope that excludes descendants, or obtain complete violation attribution/propagation for the permitted process tree. Add a parent-continues descendant-write mutant to the kill-profile gate and require the overall gate to fail; retain the plain-EPERM attribution controls. Any kernel execution needed to verify this belongs to a separately authorized author run, not this reviewer's sandbox.

**Freeze this final review here. The watchdog is deferred this cycle under the recorded delegation; there is no automatic fourth round, repair or deployment.**
