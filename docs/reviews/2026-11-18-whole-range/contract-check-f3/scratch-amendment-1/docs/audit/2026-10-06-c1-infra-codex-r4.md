## Verdict

**SIGN — N1, N3 and N4 are closed.** Reviewed detached `e60779c016d74e3d4b69a28908f512c05ced37a2`, commit `3d6ae3fae233f6fe6a533e077f5b4f21740a7446` and its message, against the round-three report. The retained/archive recovery and collision counterexamples now take the required failure direction. No change in `3d6ae3f` was found to defeat Eric's five requirements.

The exact permitted suite ran outside the sandbox, as specified by the prompt:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/scripts/test_c1_cycle.py tests/scripts/test_c1_r3_acquire.py -p no:cacheprovider
114 passed
```

Independent read-only Python probes exercised actual launcher functions with synthetic inputs. Filesystem reads/writes, ledger persistence, journal/activity queries, unit submission and register access were replaced in memory; no operational data or register contents were read. These checks establish Python recovery/admission decisions, not new Linux CPU, kill, concurrency or crash measurements. The previously disclosed scheduling, payload-confinement and witness limits remain applicable.

This SIGN clears the infrastructure closure review requested in the r4 prompt. Separate reviews and owner/production gates remain applicable. The count build, `admission.py` and register were not reviewed. No network, SSH, gh, memory/outside-checkout search, tracked-file edit, live unit operation, acquisition, commit, push or deployment was performed.

## Closure of N1, N3, N4

**N1 — Closed.** `_lifecycle` at `scripts/audit/c1/launch.py:205–215` collects authoritative PENDING and TERMINAL inputs from both the root and legacy `jobs/` directory. Differing bytes for duplicate PENDING or TERMINAL records refuse explicitly. `reconcile` at `:218–262` replays every retained invocation regardless of RECONCILED, finds its terminal in either location, and retains the recovery inputs. It writes any required pause, commits the idempotent ledger row, then creates/refreshes a deterministic RECONCILED. Existing RECONCILED is a derived result rather than proof that accounting survives in the current ledger.

Repository tests at `tests/scripts/test_c1_cycle.py:576–597` verify rollback recovery to 51 hours, checkpoint refusal and archive recovery. The interrupted-ledger-write test at `:477–494` remains green; the ambiguity test at `:611–617` checks its explicit refusal.

Independent probes re-ran and extended the r3 counterexamples:

| Synthetic input, journal empty | Actual result |
|---|---|
| 49-hour ledger, root PENDING/TERMINAL for a valid two-hour exit, RECONCILED already present | Accounting restored to **51 hours**; run returned **2**, with no submission. |
| Same receipt pair, both archived in `jobs/` | **51 hours**, checkpoint refusal, no submission. |
| Archived PENDING, root TERMINAL | **51 hours**, checkpoint refusal, no submission. |
| Root PENDING, archived TERMINAL | **51 hours**, checkpoint refusal, no submission. |
| Repeat status after each of the above | Remained **51 hours**, with no duplicate accounting or further RECONCILED rewrite. |
| Interrupt after the ledger commit but before RECONCILED creation | Retry recovered the existing two-hour accounting as **one row**, without double counting. |
| Different root/archive PENDING or TERMINAL bytes for one unit | Explicit ambiguity refusal; no submission. |

The missing/rolled-back accounting row, both archive layouts and partial archive states raised in r3 are repaired. The injected interruption is a process-level model; no real power-loss experiment was performed.

**N3 — Closed.** Replayed archived ordinary failures now receive a root RECONCILED at `launch.py:256–260`. Existing `receipt_failures` at `:273–281` therefore consumes their nonzero exit codes, and `_locked` unions them with journal failures before admission (`:480`). The archived rc=5 regression at `test_c1_cycle.py:600–608` passes, including acknowledgment of exactly the prior unit. Current-workflow ordinary-failure checks at `:497–507` also remain green.

The independent synthetic archived rc=5 probe, with no journal, produced:

- Same-name launch without acknowledgment: **return 2**, no submission.
- Two status calls: the exact failed unit remained in `failed_units`.
- Another same-name retry: **return 2**.
- Different job name: **return 0**.
- Same job with acknowledgment of that exact failed unit: **return 0**.

The simulated intervening successful job had a completed terminal receipt before the later acknowledgment check. Ordinary failures retain their same-name gate without becoming global C1 pauses.

**N4 — Closed.** `unit_taken` at `launch.py:265–270` checks PENDING, TERMINAL, RECONCILED, OVERRUN and RESUME in both current and archived lifecycle locations, plus unit identities in ledger rows. The allocation loop at `:475–485` redraws occupied identities and refuses after five unsuccessful draws, before the PENDING write or submission (`:489–493`). This runs inside the common launcher lock (`:445–446`). Non-reuse now follows from the occupied-identity check over retained history, rather than merely assuming two random suffixes cannot coincide. The receipt-validation behavior remains intact; all nine invalid-receipt cases pass.

The forced-redraw and persistent-collision regressions at `test_c1_cycle.py:635–654` pass. The repository redraw fixture plants stored receipt evidence; the independent probe additionally exercised prior journal accounting and an existing released pause:

- Prior unit `c1-r3-fit-20261005T120000Z-aaaaaaaa`, a **10-second** journal ledger row, retained lifecycle evidence and a hash-bound synthetic Eric release.
- Frozen clock/name; first UUID repeats `aaaaaaaa`, second supplies `bbbbbbbb`.
- Actual launcher used `c1-r3-fit-20261005T120000Z-bbbbbbbb` after **two draws**. Prior evidence stayed byte-identical.
- The modeled new job wrote a **3,540-second overrun** terminal. Status counted **3,550 seconds total**, including the prior 10 seconds, and reported a new unreleased pause for the fresh identity. The old release did not clear it.
- A different C1 job then returned **2**, without another submission.
- A permanently repeated suffix refused after exactly **five draws**, with **zero submissions**.
- Direct checks of `unit_taken` rejected occupation by each of the ten root/archive lifecycle paths and by a ledger row alone.

This closes the r3 overwrite, historical journal suppression and inherited-release counterexample within the stated launcher/retained-evidence model. No random collision or real concurrent launcher was claimed to have been observed.

## New findings (3d6ae3f only)

**None that qualify under this round's scope.** The new archive/replay path preserves pause-before-accounting-before-RECONCILED ordering and fails closed on conflicting authoritative records. The collision loop remains under the existing lock and precedes submission; an overrun under the redrawn identity receives its own accounting and global pause.

Guard enforcement and downloader implementation are unchanged in the permitted range. The full permitted suite preserves the existing child-process, launch-lock, unfinished-request refusal and single cached-schedule-read checks. N2 and N5–N8 were not reopened. Passing substituted-state tests do not enlarge the earlier Linux witness claims.

## Required changes

**None for this closure SIGN.** N1, N3 and N4 need no further edit or review to satisfy the remainders raised in r3. Keep the existing conditional guard bound, retained recovery evidence, exact release binding and separate owner gates. This report makes no broader production or operational acceptance claim.
