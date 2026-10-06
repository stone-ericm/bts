# C2 (a) unit 1: W0 repair design (R3-1 to R3-4)

**What this is:** the design note for C2 item (a), review unit 1 (C2 proposal §5 (a)). It is reviewed together with the W0 code. W0's last review, `docs/audit/2026-10-06-c1-r2-watchdog-w0-codex-r3.md`, ended BLOCK with four required changes; this note fixes how each is closed. Production code: reviewed until SIGN, at most 2 rounds in C2, then D7 with the whole watchdog.
**Code:** branch `c2-watchdog`, starting from `c429c41` (restored in `99a26c9`). **Registration:** `docs/sota_audit/2026-10-04-prereg-c1-watchdog.md` (FROZEN); nothing here changes it.

## R3-1: a registered, job-scoped checker identity
**The defect:** a checker failure, and the recovery that closes it, are keyed by the callable's `__name__`, with no job in the key. A different callable with the same name, in the same job or in another job, closes a still-broken checker's episode.

**The design:**
- **Registration is explicit.** A job is a sequence of `(check_id, callable)` pairs, not bare callables. `register(job, checks)` and `run_job` both validate the sequence before any check runs:
  - the job name and every `check_id` match `[a-z0-9][a-z0-9_-]*`;
  - every `check_id` is unique within the job;
  - every callable is callable;
  - a job is registered once.

  Anything else raises `RegistrationError`. That is a configuration failure, and the CLI exits non-zero (W-self sees it). It is never a per-check result.
- **The invocation identity is `<job>/<check_id>`.** It is the checker-failure result's `check`, and its incident is `checker:<job>/<check_id>`. A successful execution closes only open targets with exactly that check and incident, so the key carries the job scope.
- **The `checker:` namespace is reserved.** A business result whose incident starts with `checker:` is invalid (it becomes a checker failure). Business recovery (the verified/all-clear path) never touches a target whose incident starts with `checker:`. So only the runner's execution record can open or close a checker episode.
- **What stays the same:** success closes the checker episode whatever the business result; a later failure opens episode 2.

## R3-2: a target's notices are delivered in order
**The defect:** `flush` sorted the due notices by `seq` but sent them all independently. So a later notice of a target (a recovery) could be accepted before an earlier one (its fault) in two cases:
- the earlier send failed in the same batch;
- the earlier notice was live-claimed by another flusher.

**The design: head-of-line chains per target.**
- **The order:** a target's notices are ordered by `seq`. Its *unsent* notices form its queue.
- **Claiming:** a flush claims, for each target, a prefix of that queue. It walks the queue from the head and includes each notice while it is due (pending, or `sending` with an expired or rolled-back lease). It stops at the first notice that is not due.
  - So when the head is live-claimed by another flusher, nothing of that target is claimed. That closes the live-claim interleaving.
  - **Chains are claimed whole.** A chain is never cut. So all of a target's `sending` notices belong to one claim and share one lease: they are all due or all live, and a later claimant replaces all of a stale claimant's tokens or none.
  - **Which targets (review r1 B1):** a flush claims the chains of at most `max_sends` targets, least-tried head first, then oldest head. `max_sends` bounds actual send attempts, not claimed notices. So a target whose head fails uses one attempt and independent targets use the rest, and a head that keeps failing yields to fresh alerts. (The first version stopped claiming once `max_sends` *notices* were claimed, so one blocked long chain starved every other target: review r1 B1.)
  - **Why whole (found in the author's false-refusal pass, `808ae12`):** a cut re-claim of a stalled flusher's expired chain (say [fault, recovery] under `max_sends = 1`) would take only the fault. A failed send then leaves the fault pending ahead of a recovery still on the stale claim. That is a state the validator below rejects, so the failing flush could not record its outcome.
- **Sending:** within a flush, a target's chain is sent strictly in order. At the first failure, missing message id, exhausted budget, or the `max_sends` send bound, every later notice of that target in the batch is released **unattempted** (back to `pending`, with an explanatory `last_error`, no attempt counted). Other targets continue.
- **So a notice is accepted only after its target's earlier notices were accepted,** by the same flusher, within this flush or an earlier one.
- **Duplicates stay possible** after a crash or an expired lease. Exactly-once delivery is still not claimed. Reordering is closed.
- **Kept unchanged:** token fencing, the short lock (no network under it), the monotonic budget and UTC leases. `max_sends` still bounds the sends of one flush (now in the send loop).
- **New invariants, validated on load (R3-3; review r1 B2, B3):**
  - by `seq`, a target's notices are `sent`, then `sending`, then `pending`;
  - a target's `sending` notices share one claim time and one lease;
  - an attempt count above zero exactly when `last_attempt_at` is recorded; a `sent` notice has an attempt, its recipient and message id, and no error; an unsent notice has no recipient and no message id.

  Claims take a whole due prefix in one transaction; the claimant's completion records a sent prefix (attempt, time and recipient together) and releases the rest; a stale claimant's completion matches no token; `enqueue` appends fresh pending notices. So every legitimate transition preserves these, and a state that violates one is impossible under the protocol and refuses.

## R3-3: lifecycle completeness and the sequence are validated before acting
**The defect:** validation checked each notice's target, never the reverse. A deleted notice left an open target with no pending notice; it was accepted, rewritten and the fault was silenced for good.

**The design.** The protocol never prunes notices, so the following hold. `_validate` checks them all on every load, before any suppression, send or rewrite. Failure raises `NotifyStateError`, and the bytes are untouched.
1. **The sequence:** the notice `seq` values are exactly `1 … state.seq`, with no hole and no duplicate.
2. **Episodes:** for each target, its notices' episodes are exactly `1 … target.episode`.
3. **Each episode's lifecycle:**
   - its first notice (by `seq`) is an alerting state;
   - at most one `recovered` notice, and if present it is the episode's last;
   - every earlier episode has a `recovered` notice;
   - the current episode has a `recovered` notice exactly when the target is closed.
4. **An open target** has a notice for its current state in its current episode.
5. **Order across episodes:** every notice of episode *e* has a smaller `seq` than every notice of episode *e+1*.
6. **Delivery order:** by `seq`, a target's statuses run `sent`, `sending`, `pending` (R3-2).

If pruning is ever added, it needs an explicit, validated retention record; this note doesn't add one.

## R3-4: no descendant process exists inside the confinement gate
**The defect:** the gate's sandbox profile kills the *process* that attempts a denied write. A descendant killed that way is invisible when its parent ignores its exit, so the gate stayed green.

**The design (a process scope that excludes descendants):**
- **The gate profile also denies process creation, with SIGKILL:** `(deny process-fork (with send-signal SIGKILL))`. Every attempt to create a process kills the top interpreter itself, before a child exists, so the gate fails on its own exit status. No admitted descendant can exist whose fate goes unobserved.
- **Measured on this Mac** (`docs/audit/2026-10-06-c2-w0-evidence/fork_probe.{py,out}`; macOS 27.2, Python 3.12.13):
  - **Kill profile:** every process-creation path tried gave return code −9 with no child started: `subprocess.run`, `os.fork`, `os.posix_spawn`, `os.system`, `multiprocessing` (spawn and fork) and shell `Popen`.
  - **Imports:** importing `bts.cli`, `bts.watchdog.cli` and `bts.dm` needs no process.
  - **EPERM profile:** every path gave an attributable `PermissionError`, except `os.system`, which swallowed it (returned 32512) and continued. That is why the gate uses the kill form, not EPERM.
- **Tests:**
  - the reviewer's two parent-continues mutants, wrapped around `Notifier.flush` (a self-SIGKILLing descendant, and a descendant writing outside the root), must make the positive gate fail;
  - an EPERM-profile control witnesses the fork refusal itself;
  - the existing plain-EPERM controls stay, including the subprocess-inheritance control, under the EPERM profile, which still allows fork.
- **For later units** (stated now; built and reviewed there):
  - the watchdog will need read-only process observations (W-mode's `crontab -l`; any W-restart counters);
  - they go through one process seam, which gates 1 and 2 substitute, as registration §4.1 requires ("patch … process observations");
  - inside the confinement gate, any other process creation still fails the gate;
  - the real seam's argv, timeout and read-only use get their own direct tests;
  - the seam's real child (a fixed system binary run with `-l`) is outside the kernel gate, and that unit's review will state that boundary explicitly.
- **What this does not claim:** the gate runs on macOS only (the box is Linux), as before. It confines the watchdog's own process; it is not evidence about Linux behaviour.

## Tests (red first)
| Change | New test (fails before the fix) |
|---|---|
| R3-1 | two different callables both named `check` in one job: one raises, one succeeds. The broken one's episode stays open across runs, with no recovery notice. The same across two jobs. Genuine success then recurrence still give episode 1 → recovered → episode 2. Duplicate or invalid ids, an invalid job name and an empty job refuse at registration and in `run_job`, before any check runs or anything is written. A business result with a `checker:` incident becomes a checker failure. A business all-clear never closes a checker target. A result with an empty or non-string incident or selection is a checker failure and silences no other check's alert (found while reading `_valid`) |
| R3-2 | (a) the fault send raises and the recovery is in the same batch: the confirmed delivery order across flushes is fault, then recovery, and the recovery is released unattempted (attempts 0). (b) A fault's sender is paused mid-send with a live lease; the recovery is enqueued; a second notifier flushes and sends nothing for that target; the sender is released; a third flush sends the recovery. Confirmed order: fault, recovery. (c) Independent targets still progress when another target is blocked. (d) An uncertain send (no message id) holds its successor. (e) A stalled flusher's expired chain re-claimed under `max_sends = 1` is taken whole and stays valid when the send fails. (f) `max_sends` bounds the sends of one long chain |
| R3-3 | the reviewer's deleted-notice probe (seq kept) refuses through `enqueue` and through `flush`, with the bytes untouched and no fake send. Also: a seq hole; a deleted notice with seq decremented; a missing earlier-episode recovery; an open target escalated after its own recovery; an episode opening with a recovery; a closed target without a recovery; a reopened target without a new episode; an open target without its current-state notice; episodes out of seq order; a `sending` notice behind a pending predecessor. Each through `enqueue` and through `flush`. Positive controls: the generated one-fault, recovered and two-episode states load |
| Review r1 | **B1:** the reviewer's starvation cases (a 21-notice blocked chain at the default bound, a 3-notice chain at `max_sends = 2`) over three flushes: B is accepted on the first, A's head is the only other attempt, and every flush stays within the bound; a fresh alert is tried before a repeatedly failing one. **B2:** seven impossible delivery-metadata records (the reviewer's zero-attempt "sent" among them) refuse through `enqueue` and `flush`, untouched and unsent. **B3:** the split-claim corruption refuses before any send; legitimate stale chains with a later alert, under `max_sends = 1`, and after a clock rollback reclaim whole and stay valid. **B4:** the stale-sender test asserts its worker finished cleanly; a late success never overwrites a newer confirmation |
| R3-4 | the two parent-continues descendant mutants under the kill profile: the top child dies (−9), no `GATE-OK`, and the outside file is unchanged. The fork-refusal EPERM witness |
