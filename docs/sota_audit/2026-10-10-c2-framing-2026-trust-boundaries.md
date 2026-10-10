# The 2026 framing test: trust boundaries of the whole chain (a code-phase design note, round 5)

**Status:** a design note for the code of the 2026 out-of-sample test of catcher-grouped framing (the frozen design
`docs/sota_audit/2026-10-09-prereg-c2-framing-2026-test.md` at `9b344d9`, which does not change). Written by the lead
(bts-lead3) on 2026-10-10 under the herdr manager's ruling of 2026-10-10: after five review rounds (c1 twice, f1 twice,
f2 once) each found a new unenforced trust path at the run boundary, round 5 starts with this note, not with code.
**Revision 2 (2026-10-10):** revision 1 (commit `abd2477`) was reviewed by f2 as an anchored design review (report
`…-code-codex-f2r2.md`, BLOCK, not counted): boundaries 5 and 11 accepted; the reservation of boundaries 8–9 was still a
file the caller creates; a separate C1 `status` call did not serialize the combined decision with the launch; boundary
7 had no state machine and an unobservable, double-counting CPU measure; boundary 0 could not tell a plan certificate
from a code certificate; boundaries 3 and 14 had no enforced order or failure rule; the map omitted the nonblocking
items. Revision 2 answers each, in place and marked "(r2)". Code follows only after a reviewer signs this revision; a new
fresh reviewer on the whole range is the merge gate.

**Scope statement (r2, the manager's rule of 2026-10-10):** a sound reservation (boundaries 8 and 9) requires
enforcement inside C1's own launch lock — a change to `scripts/audit/c1/launch.py` (a combined mode: `--extra-charges`,
`--require-ledger`, the overhead reserve, and a C1-written `RESERVATION_<unit>.json`) and to its tests in
`tests/scripts/test_c1_cycle.py` — not only to `f26.py`. C1 is shared infrastructure; that change is a separate reviewed
change with its own gate, brought to the manager before any code. Everything else in this note changes `f26.py`, its
tests and the evidence only. Nothing under `src/` changes.

**The C1 launcher change is in scope on the manager's conditions (2026-10-10), which bind this plan:**
(a) **additive and default-off** — `launch.py run` behaves byte-for-byte as today without the new flags; the combined mode
exists only when both `--extra-charges <path>` and `--require-ledger` are given; `status` is unchanged;
(b) the existing C1 tests in `tests/scripts/test_c1_cycle.py` pass unchanged, and new tests cover the combined mode, the
`RESERVATION_<unit>.json` receipt, the fail-closed ledger checks and the lock ordering (decision, receipt, PENDING and
`systemd-run` inside one `_locked` call);
(c) **observed from code on 2026-10-10 at commit `caf38c9`** (`grep` over `src/`, the top-level `scripts/*.py`,
`scripts/cron-setup-hetzner.sh` and `pyproject.toml` for `scripts.audit.c1`, `scripts/audit/c1` and `audit.c1`): nothing
in production imports or invokes `scripts/audit/c1`; its only importers are the research code under `scripts/audit/`
(`c1_r3`, `c1_r4a`, `c1_r4b`, `c2_framing`) and their tests. The box's crontab and user units were not read from here;
the repository's cron installer carries no such line. Should any production path be found to invoke it, the change goes
back to the manager before code;
(d) **deploying the changed launcher to the box is its own step with a written procedure:** pull by commit into
`~/projects/bts-c1`, checksum-verify `scripts/audit/c1/launch.py` and `guard.py` against the admitted commit, record it
as a C2 index row, never inside 00:45–03:10 box clock, and nothing launches in combined mode until the box's launcher is
verified at that commit;
(e) the C1 change rides in the same review package as `f26.py`'s changes, so the merge-gate reviewer (f3) sees both.

**What this note is.** The chain, in order, from the exposure row to the aggregate. For each boundary: **what binds
it** (bytes by sha256, a register row, a file identity created exclusively, a cgroup unit, a reviewed constant), **what
happens when the bound thing is missing or failed** (always a refusal, never a default), **which findings** of the five
reviews map onto it, and **what round 5 implements** there. "The caller" means the process or operator invoking a step;
a boundary that trusts the caller is the defect every review found. Findings are cited as c1-N, c2 R2-N, f1 FN, f1r2 BN
and f2 BN.

**The sources of trust, and nothing else:** PINS (the admission's `input_pins`, every file read through
`read_pinned`); CONST (the reviewed code's constants at the admitted commit; production's `LGB_PARAMS`,
`BLEND_CONFIGS`, feature settings); GIT (the exposure commit's content and chronology, `head_admitted`'s closure
check, and (r2) the first-appearance commit of each register row); REG (the register at HEAD: (r2) the gates refuse
when the checked-out register differs from HEAD's, so the text they read is a committed text; Eric's rows by their
source token, the lead's structured rows by grammar); C1 (records the launcher or the guard writes under C1's own
lock and rules: PENDING, TERMINAL, RECONCILED and (r2) RESERVATION); EXCLUSIVE (a file created once, by hard link of a
fsynced temporary, never replaced: it fixes identity, not content); PINNED→DERIVED and RETAINED→RECOMPUTED (pure
functions of pinned inputs or retained bytes). **A trusted producer, not a trusted filename (r2):** a record counts as
evidence only when the process that writes it is one the operator does not control at the time (C1 under its lock, the
guard inside the unit, git) or when its bytes are fixed by a register row; a file the wrapper itself creates is a
declaration, however exclusive or random its name.

## The chain

### 0. The frozen design and the reviewed code
- **Binds:** the design's bytes (sha256 `a0d2919d…`); the executable closure (`scripts/`, `src/bts`, `pyproject.toml`,
  `uv.lock`, the design) at the reviewed commit, by `git diff` against it; a plain SIGN whose single
  `Reviewed-commit` line is that commit, from the archived report's bytes at the exposure commit (GIT). **(r2) The
  eligible certificate is a code review of the whole range, not a design review of this note:** the report must carry,
  in its verdict section, exactly one line `Review-kind: code, whole range <base>..<commit>` with the reviewed commit,
  and `admission._review_signs` requires it; a report without it, or with another kind, is not a certificate. The
  fresh reviewer's brief asks for that line; this note's reviews do not carry it and cannot admit code.
- **Missing or failed:** the admission gate refuses every entry point.
- **Findings:** f2r2 (a plan SIGN could admit unchanged code); the process gates (the kickoff, rows
  `C2-framing-2026-design-frozen`, `-code-reviews`).

### 1. X-37 and the source inventory
- **Binds:** the X-37 row published in the exposure commit, absent at its parent, unchanged at HEAD, in the PREDECLARED
  grammar naming the scope, the review report and sha, the reviewed commit (GIT); the inventory file
  `docs/audit/c2-framing-2026-source-inventory.json` published in that same commit, cited by sha256 in the row, with
  exactly `pa_dir`, `raw_root`, `screen_inputs` (absolute), the registered `selection` and the code's `extraction`
  field list (CONST).
- **Missing or failed:** no row, no file, a wrong sha, a prior publication, another selection or extraction: every gate
  refuses; no 2026 file is opened or hashed.
- **Findings:** c1-5 (the inventory was not bound), c2 R2-1 (the pre-read gate did not check it).

### 2. The preparation read (`prepare`)
- **Binds:** the preparation row `C2-framing-2026-prep-read` (DECLARED, recorded after X-37; GIT chronology, REG); the
  three directories equal to the inventory's, resolved (GIT); the one namespace `OUT_ROOT/inputs`, created exclusively
  (EXCLUSIVE); the 2017–2025 pins equal to the screen's accepted pins (PINS); the game set from the pinned 2025 and
  2026 PA ids, exact integers; the raw feeds read once each by that set; the table's full schema; `PREPARED.json`
  created exclusively with the exposure commit, the cited inventory sha256, the three resolved directories, the
  namespace, the pins, the counts and its CPU.
- **Missing or failed:** every prerequisite (the gate, the row, the inventory, the directories, the account's phase)
  refuses **before any protected read** (a 2026 file or a raw feed); malformed content found during an authorized read
  refuses at that point, records `STOPPED`, and is charged (r2: the two timings distinguished); an existing namespace
  refuses (`FileExistsError`); a failed preparation leaves the namespace for the operator to remove under a recorded
  decision, with its `failed` row in the account.
- **Findings:** c1-5 (direct `prepare` bypass), c1-7 (malformed records repaired silently), c2 R2-1, f1 F4 and f1r2 B1
  (caller-supplied directories), f2r2 (timing wording).

### 3. The prepared row
- **Binds:** register row `C2-framing-2026-prepared` in the PREPARED grammar with sha256 of `PREPARED.json`'s bytes
  (REG, GIT). This is the acquisition record: it fixes which preparation is trusted; it is immutable because the gates
  read it at HEAD (boundary 14). **(r2) The order is enforced through git first appearance:** `row_introduced_at(repo,
  row)` = the first commit touching the register that contains the row (`git log --diff-filter=AM -S`); the gates
  require X-37's commit to be an ancestor of the prep-read row's, the prep-read row's an ancestor of the prepared
  row's, and the prepared row's an ancestor of the inputs row's; equal commits refuse. The `PREPARED.json` the row
  names must have been written after the prep-read row's commit (its `exposure_commit` and the row chronology; the
  file carries no commit of its own, so this is the row order, stated as such).
- **Missing or failed:** `expect`, `run`, `validate_run` and `aggregate` refuse.
- **Findings:** f1r2 B1 (nothing bound the genuine preparation), f2r2 (the order was not enforced).

### 4. The expectation (`expect`)
- **Binds:** the fixed output namespace (no output-directory argument; the one argument, the PA directory, must equal
  the inventory's, r2 wording); `PREPARED.json` there with bytes equal to the row's sha256, carrying the admission's
  exposure commit, the cited inventory sha256, the inventory's three directories, the namespace and a finite CPU; its
  pins exactly the prepared inputs with the screen's historical pins; every input read through its pin; the output and
  `EXPECTED.json` created exclusively; the account's phase (boundary 7): the preparation's `completed` row with this
  `PREPARED.json`'s sha256 must exist. **(r2) The expectation itself is attested by the inputs row:** `EXPECTED.json` is
  the writer's own output and proves nothing; what fixes the expectation is its pin inside the admitted pins, bound by
  the inputs row (boundary 5), and the aggregate's recomputation of every value from the pinned inputs (boundary 13).
  Before the inputs row is recorded, `expect` may be rerun only after the operator removes the namespace's outputs
  under a recorded decision (the `failed`/`abandoned` row), so an expectation replaced before pinning is a recorded
  event, not a silent one; after the inputs row, the pin fixes it.
- **Missing or failed:** every prerequisite refuses before any protected input read (the prerequisite records
  themselves — the register, `PREPARED.json`, the account — are read first, as they must be); a failed step is
  charged; an existing output refuses.
- **Findings:** c2 R2-2 (the expectation as a trust boundary), f1 F4, f1r2 B1, f2r2 (EXPECTED is unsigned; the
  expectation's authority is its pin).

### 5. The inputs row, the admission record and the chain check  — **round 5 (f2 B1)**
- **Binds today:** `admission_2026.json`'s `input_pins` (exact shape); the inputs row `C2-framing-2026-inputs` with
  the pins' digest, after X-37 (REG, GIT); the historical pins equal to the screen's (PINS).
- **The defect (f2 B1):** the chain check parsed only `EXPECTED.json`, an unregistered file, and never compared the
  registered preparation's pins with the admitted pins; a coherently replaced table, pin, expectation and EXPECTED with
  the genuine `PREPARED.json` and row intact was accepted.
- **Round 5 binds:** `preparation_chain_problem` reads `PREPARED.json`, requires its bytes to equal the prepared row's
  sha256, **and requires its `pins` to equal the admitted pins minus the expectation, exactly**; it re-applies the
  record's content checks at consumption (exposure commit == the admission's; inventory sha256 == the cited one; the
  three directories == the inventory's; `out_dir` == the namespace); then `EXPECTED.json` must carry exactly the admitted
  pins, `prepared_sha256` equal to that record's sha256 and `sha256` equal to the admitted expectation pin.
  `validate_run` applies the same chain check (today only `run` and `aggregate` did). **(r2) The actual arguments are
  checked, not only strings inside the record:** `run` and `aggregate` require their `inputs_dir` argument to resolve to
  the fixed namespace and their `data_dir` to the inventory's `pa_dir` (the historical parquets are read from there
  through their pins; a copy of already-pinned historical bytes elsewhere is not a protected read, but the gate
  refuses it anyway so that the one namespace rule has no exception); the pins fix the bytes, the namespace fixes the
  place, and a hash check after an open cannot undo an out-of-inventory read, which is why the place is checked first.
- **Missing or failed:** `run`, `validate_run`, `seed_allowed` (through earlier seeds) and `aggregate` refuse.
- **Test:** the genuine `PREPARED.json` and row kept; the table, its pin, a recomputed expectation, its pin and
  `EXPECTED.json` all changed coherently; `run`, `validate_run`, `seed_allowed` and `aggregate` refuse.

### 6. The allowance
- **Binds:** register row `C2-framing-2026-allowance` in the RULED (Eric) grammar, source token `Eric`, finite positive
  numbers, the stop below the budget, and `cap == CAP_H` (REG, CONST; the cap commit reviewed on its own); the
  manifest's `allowance` equal to the row at validation; the first unit's CPU within the stop.
- **Missing or failed:** `seed_allowed`, `run`, `launch` and `validate_run` refuse; a changed row invalidates earlier
  runs deliberately.
- **Findings:** f1 F3 (the manifest's allowance and the stop at acceptance).

### 7. The accounting record  — **round 5 (f2 B2, f2 B4; r2: the state machine and the CPU model)**
- **Binds today:** C1's `compute_ledger.tsv` through C1's reader (C1) plus `OUT_ROOT/off_launcher_cpu.jsonl`
  (append-only; the first row the lead's seed row; every step a row).
- **The defects:** a missing `compute_ledger.tsv` reads as zero (f2 B2); the seed row accepts any total and source (f2
  B2); a shortened account passes (f2 B2); a failing `execute()` or launcher child is not charged, and `freeze()` before
  the durable writes excludes bookkeeping and startup CPU (f2 B4); revision 1 had no state machine, counted children
  twice and promised a "complete CPU" no process can observe of itself (f2r2).
- **The authoritative ledger (r2):** `ledger_total_hours` refuses a missing ledger, a ledger without C1's header or
  columns, or an invalid value (C1's reader refuses the last already); nothing in the test's code ever creates or
  rewrites the ledger; the combined decision happens inside C1's launcher (boundary 8), which validates the ledger
  before its own sweep so a missing or headerless ledger refuses rather than being recreated.
- **The account's rows (r2):** every row carries `step`, `state`, `cpu_s`, `cpu_source`, `recorded_utc`, and the
  identities that bind it to its subject: `prepare` rows carry the exposure commit and, when completed, the sha256 of
  `PREPARED.json`; `expect` rows the `PREPARED.json` sha256 and, when completed, the expectation's sha256; `launch`
  rows the seed, the admission's sha256, the pins digest, and, when admitted, the C1 unit name from the launcher's
  reservation receipt; `launcher-process` rows the unit and the launcher's exit code; `aggregate` rows the ten units;
  `settlement` rows (written by the lead under a recorded decision) the attempt they settle, the CPU charged for it
  and the source of that figure. States: `begun` (written at entry, before any protected read), then exactly one of
  `completed`, `refused` (a prerequisite failed before protected reads), `failed` (an exception after them),
  `unresolved` (the process could not write its own end row: the next step finds a `begun` without an end). The seed
  row is the ruled prior: `PRIOR_CPU_S = 80.01`, `PRIOR_SOURCE = "C2 index rows of 2026-10-08 and 2026-10-09"`
  (constants; the manager's ruling (b)); `seed_record` writes only those; any other first row refuses.
- **Phases (r2):** each step requires its predecessors in the account, bound by digests, before any protected read:
  `prepare` requires the seed row and a valid ledger and no `begun` prepare without an end row; `expect` requires the
  `completed` prepare row whose sha256 is this `PREPARED.json`'s; the first `launch` requires the `completed` expect row
  whose sha256 is the admitted expectation pin; a later `launch` requires every earlier seed's selected run to have an
  `admitted` launch row naming its unit and a `launcher-process` row for that unit, and every other attempt of any
  earlier seed to be `refused`, `failed` or `settled`; the `aggregate` requires all of that for the ten seeds. An
  `unresolved` attempt (a `begun` without an end row, or a `launcher-process` row without the unit's RECONCILED
  record) pauses every further step until a `settlement` row for it exists: that row is the manager's hand record
  brought into the account (its CPU figure is the manager's, labelled so), under Eric's acknowledgement row when the
  attempt was a C1 invocation (boundary 12). "Exactly one admitted launch per seed" means one selected run; every other
  attempt stays in the account, charged and judged.
- **The CPU model (r2):** one additive owner per CPU-second. `cpu_source` names it: `self` (RUSAGE_SELF of the step's
  process, from process start for CLI steps, so imports and startup are included, from entry for library calls);
  `children` (the RUSAGE_CHILDREN delta around a waited child, used for the `launcher-process` row only — the launcher
  and its `systemd-run`; the guarded unit's CPU is C1's and is never charged here); `manual` (a settlement row's
  figure). The wrapper's own row never uses `screen.cpu_seconds`, which sums self and children. The end row's `cpu_s`
  is sampled at the end of the step's work; the CPU of the final append, shutdown and interpreter exit is not
  observable by the step itself, so **each step's end row adds a fixed `tail_reserve_s`** (a constant, 2 s, declared
  in the code) and the account states it as a reserve, not a measurement; `cpu_s_at_write` in `PREPARED.json` and
  `EXPECTED.json` is a labelled snapshot with no claim of equality to the account. A process killed before its end
  row leaves `begun` without an end: `unresolved`, settled as above. `finally` covers every Python exception; it does
  not cover SIGKILL, which the `unresolved` state covers.
- **The reserve for what has not run yet (r2):** before a launch is admitted, the combined decision (boundary 8) adds
  `OVERHEAD_RESERVE_H` (a constant, 0.5 CPU-h) for the launcher processes, settlements and the final aggregate still to
  come, so the cap is never reached exactly by seeds alone; the aggregate refuses to open outcomes when the effective
  total (including its own `begun` row's reserve) exceeds the cap or when any attempt is `unresolved`.
- **Missing or failed:** a missing or malformed ledger or account, a wrong seed row, a missing predecessor or an
  `unresolved` attempt refuses the step before any protected read (the manager's ruling (c)). **Scope, stated:** the
  account's integrity rests on its append-only process rule and the manager's hand record; the gate's claim is that the
  required rows exist, are linked to the actual records by digest and unit, and sum without double counting; it does
  not authenticate a manual figure or prove the file was never truncated.
- **Findings:** f1 Part 2 (§7 charging), f1r2 B2, f2 B2, f2 B4, f2r2 (state machine, CPU model, reserves).

### 8. The launch reservation  — **round 5 (f2 B3; r2: issued by C1 under its lock)**
- **Binds today:** `launch` charges its own CPU, then requires ledger total + record + the full budget ≤ cap, then runs
  C1 with the budget; the decision is made once, here.
- **The defects:** no durable evidence that the once-only decision happened for a given invocation (f2 B3); revision
  1's nonce file was the wrapper's own creation, which any caller can manufacture, and a separate C1 `status` call is a
  refresh whose lock ends before the launch (f2r2).
- **Round 5 binds (r2): the combined decision is C1's.** `scripts/audit/c1/launch.py run` gains a combined mode,
  `--extra-charges <path> --require-ledger`, used by the reviewed wrapper and reviewed with it. Inside `_locked`, under
  C1's launch lock and after its sweep and reconciliation: (a) with `--require-ledger` the ledger must already exist
  with C1's header before the sweep (a missing or headerless ledger refuses; the sweep never creates it in this mode);
  (b) the extra-charges file is read with the same fail-closed structural rule as the account (every line a JSON object
  with a finite nonnegative `cpu_s`; a missing or malformed file refuses) and summed; (c) `plan_launch` decides
  `ledger total + extra total + OVERHEAD_RESERVE_H + declared ≤ CAP_H` (the ledger-only gate stays as well); (d) on
  admission the launcher writes **`RESERVATION_<unit>.json`** beside `PENDING_<unit>.json`, with the same durable writer,
  carrying the unit, the declared budget and limit, the ledger total, the extra total, the extra file's byte length and
  sha256 at decision, the cap, the overhead reserve and the decision time; then PENDING, then `systemd-run`. The
  receipt has a trusted producer (C1 under its lock), a prior unit binding (the unit C1 chose), and the decision is
  serialized with the launch and with every other C1 launch; a later C1 job cannot slip between the decision and the
  start. **The wrapper's part shrinks to what it can attest:** it validates the account's phase (boundary 7), writes
  its `begun` row, checks the seed order and the window, and invokes C1 in combined mode with the account's path; it
  makes no reservation of its own. A refused C1 launch leaves no RESERVATION (C1 writes it only on admission) and the
  wrapper's row ends `refused`; a C1 failure after the receipt (a `systemd-run` error keeps PENDING, as today) ends the
  wrapper's row `failed` and the attempt is `unresolved` until settled (boundary 7).
- **Missing or failed:** no receipt, a receipt for another unit, a receipt whose account prefix no longer hashes, or a
  decision over the cap: the payload refuses (boundary 9) and acceptance refuses (boundary 11).
- **Findings:** f1 F3 (the budget), f1r2 B2 (the double decision), f2 B3, f2r2 (the manufactured reservation; the lock).

### 9. The C1 unit  — **round 5 (f2 B3; r2: the receipt is checked, nothing is consumed)**
- **Binds today:** the kernel's cgroup record (`<unit>.service/payload`) with a unit name of this seed's; C1's
  `PENDING` naming that unit with Eric's budget in full and the wrapper's wall time; the guard's `TERMINAL` under
  `launch.terminal_problems`; `RECONCILED` equal to the exact record `reconcile` writes (C1).
- **Round 5 binds (r2):** `run` reads `RESERVATION_<unit>.json` for its own cgroup unit (the same place and producer as
  PENDING) and requires: the unit equals its own; the budget equals Eric's; the cap equals `CAP_H`; `ledger + extra +
  reserve + budget ≤ cap` as recorded; and the account's first `extra_bytes` bytes still hash to the receipt's
  `extra_sha256` (append-only: later rows do not disturb the prefix). There is no nonce and nothing to consume: the
  unit name is the binding, chosen by C1 before the start, and a receipt cannot be reused because a unit name is never
  reused (C1 redraws a taken name). `validate_run` requires the same receipt for the manifest's unit under C1's rules
  and that the combined decision it records fits the cap; `seed_allowed` and the aggregate inherit it.
- **Missing or failed:** `run` refuses before the claim; at acceptance a run whose unit has no receipt, or a receipt
  that does not fit, is refused.
- **Findings:** c1-6 (receipts), f1 F3 (C1's rules), f1r2 B3 (other invocations), f2 B3, f2r2.

### 10. The run
- **Binds, before the claim (r2: the two stages distinguished):** the admission gate (0, 1, 5), foreign imports, the
  pins, the trusted evidence, the allowance (6), the guarded unit and its reservation receipt (9), seed order and other
  invocations (12), the chain and the actual directories (5), the deterministic flags and settings (CONST), the account's
  phase (7), then, inside the claim lock, the existing-claim check and the quiet window, then the claim.
- **Binds, after the claim, during computation:** strict prediction, the self-check, the featureless stop, the
  first-unit stop; each failure writes `STOPPED.json` durably with its reason, which acceptance refuses, and the unit's
  TERMINAL/RECONCILED records the exit; the attempt is then judged under boundary 12 (Eric's acknowledgement) before any
  other launch of that seed.
- **Missing or failed:** before the claim, refuses with no run directory; after it, a durable stopped state.
- **Findings:** c1-1 (strict prediction), c1-4 (the 2019 start), the hook's placement (all reviews: closed), f2r2 (the
  stages).

### 11. Acceptance (`validate_run`)  — **round 5 (f2's coercion finding)**
- **Binds:** every declared field against its trusted source, listed in `field-inventory.md` (revision 3 in round 5):
  the manifest against CONST, GIT, PINS, REG; the profiles against the pinned calendar and labels, with ranks following
  the retained probabilities; the catcher evidence against the pinned table, projection and expectation, **with exact
  integer dtypes and values for `game_pk`, `team_id`, `catcher_id` and `n_pa` (no `int()` coercion)**; scorecards,
  diffs and summaries recomputed from retained bytes; the launcher records under C1's rules; the consumed reservation
  (9); the chain (5); other invocations reported with Eric's acknowledgement.
- **Missing or failed:** `RunInvalid`.
- **Findings:** c1-2, c1-3, c1-6, f1 F1, F2, F3, f1r2 B3, f2 B3, f2's coercion finding.

### 12. Seed order (`seed_allowed`) and the pre-launch claim check
- **Binds:** the allowance (6); every earlier seed exactly one run that passes 11; no other C1 invocation of any seed
  up to this one without Eric's row `C2-framing-2026-invocation-<unit>` (REG); the current unit excluded in `run`;
  **(r2) the pre-launch check that the current seed has no run directory at all** (today in `launch`; it answers c1-6's
  repeated launch and stays); an acknowledged other invocation is a recorded history, never a release of any other
  gate, and its account rows stay (boundary 7).
- **Missing or failed:** refuses the launch and the run.
- **Findings:** c1-6 (repeat launch), c2 R2-2, f1 F3, f1r2 B3, f2r2.

### 13. The aggregate
- **Binds:** exactly the ten registered seeds, each passing 11 under the admitted identity and pins; the chain (5);
  the account (7) complete for all ten seeds and every attempt settled; agreement on everything that defines the run;
  the resumed rows and every catcher value recomputed from the pinned inputs; §6's dispositions; every invocation
  reported; its own CPU charged; **(r2) the cap: the effective total, including its own `begun` row and the tail
  reserve, must not exceed the cap, or it refuses to compute any disposition and reports** (an exhausted allowance
  stops and reports; §7).
- **Missing or failed:** `RunInvalid`; its own failure is charged; an `unresolved` attempt or an exceeded cap refuses.
- **Findings:** c1-3, c2 R2-2, f1r2 B2, f2 B2, f2r2 (the final cap rule).

### 14. The register as a source
- **Binds (r2):** the register **at HEAD** of the admitted checkout: the admission gate refuses when the register path
  is modified or untracked against HEAD (`git status --porcelain -- <register>` non-empty), so the text every gate reads
  is a committed text, identified by HEAD; the chronology of rows by their first-appearance commits (boundary 3) and
  against the exposure commit; the rows' grammars; Eric's rows by their source token. **What the code does not and
  cannot establish:** that Eric typed the words (the manager verifies the prompt log; the source token is the process's
  attestation) or that HEAD is on origin (the process rule: main is pushed only with a full suite at that commit).
- **Missing or failed:** a required row missing, malformed, present at the exposure commit when it must be later, or
  changed after its publication refuses; a dirty register refuses; the gates never read a row from the working tree
  that HEAD does not have.
- **Findings:** f1r2's correction (the gates read the checked-out text), f2r2 (no failure rule; the file was not
  frozen for an invocation).

### 15. Disclosed limits that no boundary closes
- The model outputs and the choice of the ten retained batter-games need the day's predictions, which are not
  retained; exact probability ties are reported, not ordered. (all reviews)
- The lookup's and table's derivation from the raw feeds is the admitted preparation's work; the manifest pins the
  feeds, acceptance does not re-derive them. Boundary 5 now guarantees that the final table is the recorded
  preparation's, which is what f2 B1 found missing.
- Deleted lifecycle records cannot be discovered by the scan of boundary 12.
- A manual accounting file's history cannot be authenticated by code (boundary 7's scope); a settlement row's figure
  is the manager's, labelled so.
- (r2) A correct recipe and blend declaration is not independent proof of which fit produced the probabilities; the
  frozen starter proxy, the official-date and resumed-event availability and the serving-availability assumptions of
  the design's §10 are assumptions this test records, not facts it proves.
- (r2) The unobservable tail of each step's CPU is covered by a declared reserve, not measured.

## The findings, mapped
| Review | Finding | Boundary |
|---|---|---|
| c1 | 1 strict prediction incomplete | 10 |
| c1 | 2 profile identities and labels self-certified | 11 |
| c1 | 3 coverage self-certified | 11, 13 |
| c1 | 4 history start | 10 |
| c1 | 5 acquisition unbound | 1, 2 |
| c1 | 6 completion and costs unbound; repeat launch | 9, 11 |
| c1 | 7 malformed records repaired | 2 |
| c2 | R2-1 inventory unchecked pre-read | 1 |
| c2 | R2-2 coverage self-certified at the next-seed gate | 4, 12, 13 |
| f1 | F1 rank order | 11 |
| f1 | F2 the recipe and blends | 11 |
| f1 | F3 the stop, the allowance, C1's rules | 6, 9, 11 |
| f1 | F4 expect's inputs and one-time boundary | 2, 4 |
| f1r2 | B1 caller-selected preparation | 2, 3, 4, 5 |
| f1r2 | B2 the accounting namespace, missing record, failures, double decision | 7, 8 |
| f1r2 | B3 other invocations | 11, 12, 13 |
| f2 | B1 the chain never compared the prepared pins | 5 |
| f2 | B2 missing ledger as zero; the seed row unenforced; no completeness | 7 |
| f2 | B3 no reservation bound to the unit; ledger read outside C1's lock | 8, 9, 11 |
| f2 | B4 failed launcher execution and bookkeeping CPU uncharged | 7 |
| f2 | catcher-evidence integer coercion | 11 |
| f2 | M106/M107 selectors select nothing; inventory and ledger wording | the evidence, not a boundary: fixed in round 5 |
| f2r2 | the reservation was the caller's file; status is not a transaction | 8, 9 |
| f2r2 | no accounting state machine; double-counted and unobservable CPU; no future reserve | 7, 13 |
| f2r2 | a plan SIGN could admit code | 0 |
| f2r2 | no failure rule for the register; row order unenforced | 14, 3 |
| f2r2 | actual arguments unchecked; timing wording; the stages of run | 5, 2, 4, 10 |
| c1 | nonblocking 8 (hook placement and default parity) | 10: closed; the base-vs-current parity is each reviewer's own probe |
| c1 | nonblocking 9 (projection, as-of, disposition arithmetic) | 2, 10, 13: closed |
| c1 | nonblocking 10 (reporting: identified ids, resumed totals, disagreement flag, streak statement, off-launcher costs) | 11, 13 and 7: the machine fields exist; the results note (design §6, §8) must still state coverage denominators, the disagreement prominently and streak movement first |
| c1 | nonblocking 11 (stubbed integration; the test names) | the evidence: the run tests stub admission, features and the walk-forward and say so; only an admitted real run exercises them |
| c2 | nonblocking 3–6 (strict paths, labels, history, receipts: closed cases) | 10, 11, 9: closed |
| c2 | nonblocking 7 (ledger provenance, preparation CPU) | 7 |
| c2 | nonblocking 8–10 (mutation kills' meaning, the exact threshold, the bytecode account) | the evidence: message-only and masked kills are labelled; the threshold has its test and mutant; the bytecode cause is "plausible, not measured" |
| f1 | the ledger error (off-launcher CPU), the early stop's basis, weak test names | 7; the evidence (corrected) |
| f1r2, f2 | inventory inaccuracies, the setup-TypeError RED, six repairs, M87's attribution, the round-4 commit wording | the evidence: revision 3 of the inventory, the ledgers, a NOTE row |

## The evidence: what revision 3 of the inventory must list (r2)
Every C1 TSV column (`invocation`, `unit`, `stopped_at`, `cpu_seconds`) with C1's reader as its check and whether an
accepted unit's row is present; `OVERRUN_` and `RESUME_` records and their C1 enforcement; every raw-manifest field and
every starter-table field as its own row; every aggregate output field, marking recomputations as such; the
`RESERVATION_` receipt's fields; the account's rows field by field with `step`, `state` and `cpu_source` as BOUND; the
TERMINAL failure variants as not carried into `other_units`; `data_dir` as "supplied and checked against the inventory".
Status cells distinguish CHECKED (an independent source), BOUND (a shape or range that refuses), REPORTING (no decision
reads it) and RECOMPUTED (derived here from validated inputs). The count is by script, with every row classified.

## What round 5 implements, test-first, in this order (r2)
Each item's RED runs on the current code before its fix, against the existing boundary, with controls that carry the
legitimate history the new rules require (the ruled seed row, a C1 ledger with its header, the completed prepare and
expect rows), so that a new rule never masks an older test and no RED rests on a missing attribute at setup.
1. Boundary 0: the `Review-kind` line in `admission._review_signs` and in the fresh reviewer's brief. Test: a plan
   SIGN naming the commit is not a certificate; the code SIGN is.
2. Boundary 14: the register path must be unmodified against HEAD at every gate. Test: an edited working-tree register
   refuses; HEAD's row is read.
3. Boundary 3: the row order by first appearance. Test: a prepared row introduced in the same or an earlier commit than
   the prep-read row refuses (synthetic git history).
4. Boundary 5: the prepared-pins comparison, the content re-checks, the actual-argument checks, and the chain in
   `validate_run`. Test: f2's coherent replacement; a run with another `inputs_dir`.
5. Boundary 7: the row schema with `state`, `cpu_source` and identities; the phases; the ruled seed row; the ledger's
   existence and header; `self`-only CPU for the wrapper; the tail reserve; `unresolved` and settlement rows; whole-
   process CPU for CLI steps. Tests: f2's missing ledger, zero prior, shortened account; the valid first sequence
   (seed → prepare → expect → launch) as a control; a `begun` without an end row pauses; a settlement row releases.
6. Boundaries 8 and 9: C1's combined mode (`--extra-charges`, `--require-ledger`, the overhead reserve,
   `RESERVATION_<unit>.json` under the lock) with its own tests in `tests/scripts/test_c1_cycle.py` (the plan's
   arithmetic at the boundary; a missing or headerless ledger refuses in that mode; the receipt's fields); the
   wrapper's invocation of it; `run`'s receipt check against its own unit and the account prefix; `validate_run`'s
   receipt check. Tests: a payload whose unit has no receipt; a receipt for another unit; a receipt whose prefix no
   longer hashes; the f2r2 interleaving (an extra charge between decision and start is impossible under the lock,
   shown by the decision and the PENDING being written in one `_locked` call).
7. Boundary 11: exact integer fields; nullable catcher ids only where the trusted catcher is missing. Test: f2's +0.25
   probe; a valid missing-catcher row accepted.
8. Boundary 13: the final cap rule and the `unresolved` refusal.
9. The evidence: the selectors (M106, M107); inventory revision 3 as listed above; the round-3 ledger's M87
   sentence; the NOTE rows (the round-4 commit wording; the f2 hashes).
10. Mutants M115+ for every new guard, anchors and non-empty selectors verified before any run; no mutation run until
    the manager lifts the disk pause.
