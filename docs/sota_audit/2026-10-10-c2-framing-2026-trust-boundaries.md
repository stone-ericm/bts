# The 2026 framing test: trust boundaries of the whole chain (a code-phase design note, round 5)

**Status:** a design note for the code of the 2026 out-of-sample test of catcher-grouped framing (the frozen design
`docs/sota_audit/2026-10-09-prereg-c2-framing-2026-test.md` at `9b344d9`, which does not change). Written by the lead
(bts-lead3) on 2026-10-10 under the herdr manager's ruling of 2026-10-10: after five review rounds (c1 twice, f1 twice,
f2 once) each found a new unenforced trust path at the run boundary, round 5 starts with this note, not with code. It
is reviewed by f2 as an anchored design review (its second and last round); code follows only after f2 signs it; a new
fresh reviewer (f3) on the whole range is the merge gate.

**What this note is.** The chain, in order, from the exposure row to the aggregate. For each boundary: **what binds
it** (bytes by sha256, a register row, a file identity created exclusively, a cgroup unit, a reviewed constant), **what
happens when the bound thing is missing or failed** (always a refusal, never a default), **which findings** of the five
reviews map onto it, and **what round 5 implements** there. "The caller" means the process or operator invoking a step;
a boundary that trusts the caller is the defect every review found. Findings are cited as c1-N, c2 R2-N, f1 FN, f1r2 BN
and f2 BN.

**The sources of trust, and nothing else:** PINS (the admission's `input_pins`, every file read through
`read_pinned`); CONST (the reviewed code's constants at the admitted commit; production's `LGB_PARAMS`,
`BLEND_CONFIGS`, feature settings); GIT (the exposure commit's content and chronology, `head_admitted`'s closure
check); REG (the checked-out register of the admitted checkout: Eric's rows and the lead's structured rows; the
process fixes its commit, under the rule that main is pushed only with a full suite at that commit); C1 (the
launcher's own records under C1's own rules); EXCLUSIVE (a file created once, by hard link of a fsynced temporary,
never replaced); PINNED→DERIVED and RETAINED→RECOMPUTED (pure functions of pinned inputs or retained bytes).

## The chain

### 0. The frozen design and the reviewed code
- **Binds:** the design's bytes (sha256 `a0d2919d…`); the executable closure (`scripts/`, `src/bts`, `pyproject.toml`,
  `uv.lock`, the design) at the reviewed commit, by `git diff` against it; a plain SIGN whose single
  `Reviewed-commit` line is that commit, from the archived report's bytes at the exposure commit (GIT).
- **Missing or failed:** the admission gate refuses every entry point.
- **Findings:** none directly; the process gates (the kickoff, rows `C2-framing-2026-design-frozen`, `-code-reviews`).

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
- **Missing or failed:** refuses before any 2026 read; an existing namespace refuses (`FileExistsError`); a failed
  preparation is charged (§7) and leaves the namespace for the operator to remove by hand under a recorded decision.
- **Findings:** c1-5 (direct `prepare` bypass), c1-7 (malformed records repaired silently), c2 R2-1, f1 F4 and f1r2 B1
  (caller-supplied directories).

### 3. The prepared row
- **Binds:** register row `C2-framing-2026-prepared` in the PREPARED grammar with sha256 of `PREPARED.json`'s bytes,
  recorded after X-37 (REG, GIT chronology). This is the acquisition record: it fixes which preparation is trusted; it
  becomes immutable when the process commits it.
- **Missing or failed:** `expect`, `run` and `aggregate` refuse.
- **Findings:** f1r2 B1 (nothing bound the genuine preparation).

### 4. The expectation (`expect`)
- **Binds:** the fixed namespace only (no directory argument; the PA directory equals the inventory's); `PREPARED.json`
  there with bytes equal to the row's sha256, carrying the admission's exposure commit, the cited inventory sha256,
  the inventory's three directories, the namespace and a finite CPU; its pins exactly the prepared inputs with the
  screen's historical pins; every input read through its pin; the output and `EXPECTED.json` created exclusively.
- **Missing or failed:** refuses before any read; a failed step is charged; an existing output refuses (one
  expectation per preparation).
- **Findings:** c2 R2-2 (the expectation as a trust boundary), f1 F4, f1r2 B1.

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
  `validate_run` applies the same chain check (today only `run` and `aggregate` did).
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

### 7. The accounting record  — **round 5 (f2 B2, f2 B4)**
- **Binds today:** C1's `compute_ledger.tsv` through C1's reader (C1) plus `OUT_ROOT/off_launcher_cpu.jsonl`
  (append-only; the first row the lead's seed row; every step a row).
- **The defects:** a missing `compute_ledger.tsv` reads as zero (f2 B2); the seed row accepts any total and source (f2
  B2); a shortened account passes (f2 B2); a failing `execute()` or launcher child is not charged, and `freeze()` before
  the durable writes excludes bookkeeping and startup CPU (f2 B4).
- **Round 5 binds:** (a) `ledger_total_hours` refuses a missing ledger, or one without C1's header and columns; (b)
  the seed row must carry the ruled prior, constants in the code: `PRIOR_CPU_S = 80.01` and `PRIOR_SOURCE = "C2 index
  rows of 2026-10-08 and 2026-10-09"` (the manager's ruling (b)); any other first row refuses; `seed_record` writes
  only those; (c) per-step completeness before a launch and at the aggregate: exactly one successful `prepare` row and
  one successful `expect` row, and for every earlier seed exactly one admitted `launch` row (not refused) with its
  `launcher-process` row; (d) every step appends from a `finally`, with the complete CPU measured at exit (the file's
  `cpu_s` becomes `cpu_s_at_write`, a labelled snapshot) and `failed: true` on any exception; `launch` wraps
  `execute()` and the child charge in the same `finally`, charging the child's rusage and marking failure; (e) CLI
  invocations charge the whole process's CPU (startup and imports included), library steps charge from entry.
- **Missing or failed:** a missing or malformed ledger or record, a wrong seed row, or an incomplete account refuses
  every step before any work (the manager's ruling (c)). **Scope, stated:** no code can prove a manual file was never
  truncated; the record's integrity rests on its append-only process rule and the manager's hand record, and the
  gate's claim is completeness of the required rows, not authenticity of their numbers.
- **Findings:** f1 Part 2 (§7 charging), f1r2 B2, f2 B2, f2 B4.

### 8. The launch reservation  — **round 5 (f2 B3)**
- **Binds today:** `launch` charges its own CPU, then requires ledger total + record + the full budget ≤ cap, then runs
  C1 with the budget; the decision is made once, here.
- **The defect (f2 B3):** no durable evidence that the once-only decision happened for a given invocation; a payload
  launched by any C1 command passes `run`; the ledger is read before C1's sweep/reconcile, outside C1's lock.
- **Round 5 binds:** (a) `launch` first runs C1's `status` (sweep and reconcile under C1's launch lock) and only then
  reads the ledger; (b) it writes a reservation record `OUT_ROOT/reservations/RESERVATION_<seed>_<nonce>.json`
  exclusively before running C1, carrying the seed, Eric's budget and cap, the ledger total, the off-launcher total,
  the effective total, the decision time and a random nonce; (c) it passes the nonce to the payload as the environment
  variable `BTS_F26_RESERVATION` on the C1 command line; (d) a refused launch writes no reservation.
- **Missing or failed:** no reservation record, or one already consumed: the payload refuses (boundary 9).
- **Findings:** f1 F3 (the budget), f1r2 B2 (the double decision), f2 B3.

### 9. The C1 unit  — **round 5 (f2 B3)**
- **Binds today:** the kernel's cgroup record (`<unit>.service/payload`) with a unit name of this seed's; C1's
  `PENDING` naming that unit with Eric's budget in full and the wrapper's wall time; the guard's `TERMINAL` under
  `launch.terminal_problems`; `RECONCILED` equal to the exact record `reconcile` writes (C1).
- **Round 5 binds:** `run` reads `BTS_F26_RESERVATION`, requires the reservation record for this seed with Eric's
  budget and cap, and consumes it by creating `CONSUMED_<nonce>.json` exclusively with its unit name and claim; a
  reservation can be consumed once, by one unit. `validate_run` requires, for the manifest's unit, a reservation whose
  consumption names that unit, whose budget and cap are Eric's, and whose effective total plus budget was within the
  cap when decided.
- **Missing or failed:** no nonce, no record, a consumed record, another seed's record: `run` refuses before the claim;
  at acceptance a run without its consumed reservation is refused.
- **Findings:** c1-6 (receipts), f1 F3 (C1's rules), f1r2 B3 (other invocations), f2 B3.

### 10. The run
- **Binds:** everything above, in this order: the admission gate (0, 1, 5), foreign imports, the pins, the trusted
  evidence, the allowance (6), the guarded unit (9), the reservation (9), seed order and other invocations (12), the
  chain (5), the deterministic flags and settings (CONST), the claim namespace and lock, the quiet window; strict
  prediction, the self-check, the featureless stop, the first-unit stop.
- **Missing or failed:** refuses before the claim; a stop writes `STOPPED.json`, which acceptance refuses.
- **Findings:** c1-1 (strict prediction), c1-4 (the 2019 start), the hook's placement (all reviews: closed).

### 11. Acceptance (`validate_run`)  — **round 5 (f2's coercion finding)**
- **Binds:** every declared field against its trusted source, listed in `field-inventory.md` (revision 3 in round 5):
  the manifest against CONST, GIT, PINS, REG; the profiles against the pinned calendar and labels, with ranks following
  the retained probabilities; the catcher evidence against the pinned table, projection and expectation, **with exact
  integer dtypes and values for `game_pk`, `team_id`, `catcher_id` and `n_pa` (no `int()` coercion)**; scorecards,
  diffs and summaries recomputed from retained bytes; the launcher records under C1's rules; the consumed reservation
  (9); the chain (5); other invocations reported with Eric's acknowledgement.
- **Missing or failed:** `RunInvalid`.
- **Findings:** c1-2, c1-3, c1-6, f1 F1, F2, F3, f1r2 B3, f2 B3, f2's coercion finding.

### 12. Seed order (`seed_allowed`)
- **Binds:** the allowance (6); every earlier seed exactly one run that passes 11; no other C1 invocation of any seed
  up to this one without Eric's row `C2-framing-2026-invocation-<unit>` (REG); the current unit excluded in `run`.
- **Missing or failed:** refuses the launch and the run.
- **Findings:** c2 R2-2, f1 F3, f1r2 B3.

### 13. The aggregate
- **Binds:** exactly the ten registered seeds, each passing 11 under the admitted identity and pins; the chain (5);
  the account (7) complete; agreement on everything that defines the run; the resumed rows and every catcher value
  recomputed from the pinned inputs; §6's dispositions; every invocation reported; its own CPU charged.
- **Missing or failed:** `RunInvalid`; its own failure is charged.
- **Findings:** c1-3, c2 R2-2, f1r2 B2, f2 B2.

### 14. The register as a source
- **Binds:** the checked-out register of the admitted checkout; the chronology against X-37 by `git show` at the
  exposure commit; the rows' grammars; Eric's rows by their source token. Its commit is the process's: main is pushed
  only with a full suite at that commit; rows are the lead's words or Eric's verbatim.
- **Findings:** f1r2's correction (the gates read the checked-out text, not `git show HEAD:`).

### 15. Disclosed limits that no boundary closes
- The model outputs and the choice of the ten retained batter-games need the day's predictions, which are not
  retained; exact probability ties are reported, not ordered. (all reviews)
- The lookup's and table's derivation from the raw feeds is the admitted preparation's work; the manifest pins the
  feeds, acceptance does not re-derive them. Boundary 5 now guarantees that the final table is the recorded
  preparation's, which is what f2 B1 found missing.
- Deleted lifecycle records cannot be discovered by the scan of boundary 12.
- A manual accounting file's history cannot be authenticated by code (boundary 7's scope).

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

## What round 5 implements, test-first, in this order
1. Boundary 5: the prepared-pins comparison and content re-checks in `preparation_chain_problem`; the check in
   `validate_run`. Test: f2's coherent replacement.
2. Boundary 7: the ledger's existence and header; the ruled seed row; per-step completeness; `finally` charging with
   complete CPU and failure marking, `launch` included; whole-process CPU for CLI steps. Tests: f2's missing ledger,
   zero prior, failed `execute`, a shortened account.
3. Boundaries 8 and 9: the reservation record, its nonce in the payload's environment, its exclusive consumption by
   the unit, its requirement at acceptance; C1 `status` before the ledger read. Tests: a payload without a nonce, with
   a consumed nonce, with another seed's; acceptance of a run whose unit consumed no reservation.
4. Boundary 11: exact integer dtypes and values in the catcher evidence. Test: f2's +0.25 probe.
5. The evidence: the two selectors; the inventory revision 3 (the lookup/table row, the EXPECTED row, the off-launcher
   step/source as BOUND, the TERMINAL variants not carried, "data_dir is checked" wording; the 89 + 1 count explained);
   the round-3 ledger's M87 sentence; a NOTE that the round-4 commit message's "tests first" contradicts its ledger,
   which is right.
6. Mutants M115+ for every new guard; anchors verified before any run; no mutation run until the manager lifts the
   disk pause.

The RED for each item is run on the current code before the fix (selected tests only until the manager's Vic3 end
signal), so this round's RED is behavioral, not the swap method round 4 had to use.
