## Verdict
**BLOCK.**
Reviewed-commit: 70dccc40e7dc10016f93d55a273bc9998eeed1d2

The ten-seed decision rule and retained-run bindings hold under the reviewed evidence model. The launch contract does not: the job can finish validation inside the forbidden window and still claim a seed; the public `run` route does not require C1 supervision; and the stage-two admission can execute seed 1 when its accepted claim directory is missing. These are synthetic reproductions against the unchanged tip, with the limits stated below. No real seed was launched.

HEAD was detached at the requested commit. The tracked tree was clean before testing and after the mutant runner restored its files. This verdict authorizes no run, repair, deployment or other operational action. A revised implementation needs the second review round and a plain SIGN, followed by X-36 and its admission record.

## Findings

### B1 — P1, verified synthetic: the job checks the window before completing its validation

Evidence: `scripts/audit/c2_framing/screen.py:506` checks the clock, then lines 507–510 import the model parameters, check their deterministic flags and validate feature settings. Lines 513–518 subsequently acquire the claim lock, check existing claims and create the durable run claim. The statement at lines 467–470 that the check follows validation immediately before the claim is therefore too strong.

In an owned scratch tree, I constructed the suite's synthetic accepted stage one and stage-two admission. I left the seed-order checks, accepted-byte checks, claim writer, profiles, labels and scorer in operation. The clock returned 00:44 at the window check. A wrapper around the real settings validator advanced the controlled clock to 00:45 after that validator returned. The unchanged `run` admitted seed 4 (2048), wrote its claim at the controlled 00:45, executed all six synthetic walk-forward units and returned 0. There was exactly one clock observation, at 00:44.

This establishes a validation-to-claim gap, not a measured delay or collision on the box. A cold LightGBM import or scheduling delay can also occur in this interval. Moving the clock check past seed-order validation fixed the earlier gap, but left these later validations before the claim unprotected. S39 and the current transition test cover only a transition during `stage_two_allowed`.

Deciding regression: advance the controlled clock during settings validation and during the existing-claim check; require refusal before a run directory or claim is created. Preserve the 00:44/03:10 acceptance controls and stage one's validation behavior.

### B2 — P1, verified synthetic and source: public `run` does not require the C1 launcher or guard

Evidence: `screen.py:874` exposes both `run` and `launch`; lines 889–890 dispatch `run` directly to the computation. Neither `run` at lines 490–577 nor its admission/seed checks verifies membership in a guarded C1 unit or consults C1 active-job state. Its admission lock is per seed and ends after the claim. `stage_two_allowed` at lines 440–448 checks earlier output completion, not the unit lifecycle or another C1 job. The shared code-admission check also contains no C1 process-membership check.

In the scratch witness, after seed 4 had complete valid synthetic output, I provided C1's active-job observer with a synthetic active C1 unit. Calling seed 5's `run` directly returned 0 and executed all six synthetic units. The active-job observer was called zero times. This is evidence that the public route ignores that state; it is not a claim that a real systemd unit was started or observed here.

The supported C1 launch route is sound on this point: `scripts/audit/c1/launch.py:455` refuses an active unit before accounting, and the locked launch path checks again through `plan_launch`. The two new stage-two-name tests at `tests/scripts/test_c1_cycle.py:160` passed. They test that launcher route, not direct `run`.

The direct-route weakness already existed in stage one. It matters to this review's explicit all-routes question: the stage-two job now enforces its own window precisely to cover entry outside the wrapper, but does not establish the prerequisite that makes serialization, the 50 checkpoint and the CPU guard apply. The addendum's instruction to use C1 is an operational assumption, not enforcement on this entry point.

Deciding regression: an unguarded stage-two `run` must refuse before claiming or reading inputs, including when another C1 job is active. A correctly guarded self-unit must still pass; merely refusing whenever any C1 unit is active would reject the intended job itself.

### B3 — P2, verified synthetic: X-36 can execute a stage-one seed if its accepted claim is absent

Evidence: `screen.py:496` accepts all ten seeds. `seed_allowed` at lines 350–355 applies the accepted-stage-one requirements only to seeds 4–10, and unconditionally allows seed 1 with its old 45 CPU-hour budget. Both `launch` and `run` use this function. The claim check protects a present claim; it does not scope executable seeds to the admission stage.

In a separate owned scratch witness, I constructed the synthetic accepted stage one and switched to the stage-two identity. I moved seed 1's entire accepted claim directory to a retained sibling outside the claim namespace; nothing was deleted. Under that same stage-two admission, `launch` accepted seed 1 and issued one recording-only C1 command, returning 0. Direct `run` then created a new seed-2273360 run with the stage-two identity, executed six synthetic units and returned 0. The old accepted bytes remained retained elsewhere.

Missing accepted state should refuse stage two, not reopen a stage-one execution path. This violates the requested “a stage-one seed is never re-run under this code” condition without requiring any owner invalidation. It also selects the old budget and bypasses the stage-two window rule. The ten-seed aggregate's accepted hash list would subsequently refuse this replacement: this finding concerns unauthorized execution, not an accepted wrong disposition.

Deciding regression: under the stage-two admission, both execution routes must reject each of the first three seeds, even with a missing directory or missing claim. Previously accepted stage-one results must remain validatable under their original identity.

### 4. Mechanical extension, window arithmetic and cap: checked

The frozen stage-one §5 and its implementation already use a strict majority (`len(per_seed) // 2 + 1`). Extending that implemented rule gives six of ten. A literal “2 of 3” fraction alone could admit other extrapolations, but the frozen code removes that ambiguity here. The quantities, equal per-seed/per-season weighting, +0.003 practical threshold, t ≥ 1.5, positive means in both seasons, per-seed evaluator and negative/inconclusive branches are unchanged. The ten-seed aggregate pools all three old seeds with all seven new ones, for both variants. I found no outcome-dependent numerical or selection choice in this extension. The disclosure that it was written after opening stage one remains necessary; ten algorithm seeds are not ten independent season samples.

The seed constants equal canonical-n10 positions 0–9, and stage-two order is positions 3–9. The window is correctly half-open: 00:45 is refused; 03:09 is refused; 03:10 is permitted. `ny_clock` uses America/New_York explicitly, and its system-clock comparison passed. Using the supplied uncontended wall times, a launch just before 00:45 would nominally finish around 02:38–02:39, leaving about 21 minutes before 03:00. That is an expectation based on the reported measurements, not a guaranteed finishing deadline. I did not verify the production schedule or wall times on the box.

The supplied budget arithmetic checks: 45.27 + 6 × 13.28 + 20 = 144.95 for seed 10's launch check; adding one 5.0 collision and C2's remaining 14.96 gives 164.91, below 165. These are projections, not a guarantee that all seeds finish within their budgets.

Commit `f780a8d` changes only the ledger cap constant/docstring and the relative C1 test cases. The gate logic, 50 checkpoint, launcher plan and guard implementation are unchanged. CAP_H is 165, matching Eric's actual cap row. Both real-register pin tests passed, including the exact stage-two cap/budget parser. Raising the cap does not itself create the checkpoint acknowledgement.

### 5. Accepted-run bindings and unchanged computation: checked, with bounded provenance

I independently hashed the stage-one admission bytes at `22d31f23a0c2683a7dfe530ea3380861c090667f`, the r10 report at exposure commit `105561842e2bdf2580d3ecc867e1ee5e57df04f2`, and the current committed hash list. They match all five identity constants and the pinned listing digest. The listing contains 45 unique paths, 15 for each of the three named accepted runs. Real `head_admitted` checks accepted both full historical run commits, including `11e0cfd73acb01be77826dd7d05a587a8e636852`.

`stage_one_runs` requires the accepted directory names and exact listed bytes, then validates under STAGE_ONE_IDENTITY with the stage-two gate's trusted pins. The ten-seed aggregate requires the supplied stage-one paths to resolve to those accepted paths, and validates every new seed under the new admission. Wrong identities, heads, pins, incomplete profiles or foreign claim namespaces are refused by the inherited validator. Both stages must agree on the listed manifest fields. Most agreement fields are already fixed by individual validation; the full LightGBM parameter dictionary needs the explicit cross-run check, and the differing-learning-rate case passed its refusal test.

Under this retained-evidence trust model, I found no route for a differently declared code/input run or substituted stage-one bytes to reach a ten-seed disposition. That does not authenticate the box's actual files or attest an arbitrary producer that fabricates coherent retained evidence. The box hash acquisition and matching Mac copies remain the supplied provenance account; prohibited data was not read in this review.

The post-claim computation in `run` is text-identical to the base. `load_inputs`, framing, relabeling, blend rewriting, settings checks, seed summaries, `validate_run`, `head_admitted` and the three-seed `aggregate` are unchanged. The default disposition count remains three. X-36, the new scope and the addendum closure belong to the new admission; historical validation still uses the old identity. This preserves what ran in stage one, but does not cure B3's future execution admission.

### 6. Positive witness, register fix and test/mutant limits

The suite's ten-run witness passed: ordered seeds under two identities produced the pooled ten-seed aggregate. The inconclusive-stage-one prerequisite, missing/duplicate/stopped earlier seeds, accepted-byte damage, wrong identities, wrong paths and cross-stage parameter disagreement tests all passed. No legitimate-path false refusal was measured in these synthetic checks. The actual future X-36 admission and box guard were not qualified here.

The test-only register fix replaces real rows only for the explicitly named synthetic ids. This restores the intended first-row lookup input, rather than weakening the production parser. The two cap-pin tests use no such fixture and read the real register. Their successful results confirm that the fix does not conceal a real cap/ruling mismatch. I did not independently rerun the old 50-failure main snapshot.

All 283 mutant ids are unique; every current source anchor occurs exactly once. The last committed ledger records 280 RED and G8/H12/N10 SURVIVED, matching the README. I independently replayed seven entries, including S9's isolated second-run case and S37/S39/S40: all seven were RED. S9 failed both the renamed-run and second-run cases, including the extra directory that sorts after the accepted run. Its original masking explanation is supported by the changed fixture and source behavior.

The recorded equivalents remain bounded and sound: trusted pins and their bound digest make G8 redundant; exact five-field trusted identities make H12 redundant within stage one's aggregate; nonempty season-specific profiles with daily rank one make N10 redundant. R133 was RED in my replay, but its named test pins a refusal message while another branch still refuses uncommitted outside code. It is not an independent containment defence.

The current tests cannot fail on the three admission gaps above: their transition clock advances only in seed-order validation, active-job tests enter C1 planning, and the old-seed rerun test retains its original claim. Consequently the RED ledger does not establish the full launch contract. Native large-seed repeatability also remains the author's measurement, not something the stubbed run tests prove.

### 7. Runner vocabulary and large seeds: checked

Only the vocabulary lists change in the runner. The revision-10 driver/audit hook, collection, configuration/environment isolation, boundary checks, classification and change scan are unchanged. The real framing test module and all three package initializers passed the syntax gate. I found no new escape route within the stated committed-code/trusted-installation threat model.

There are **18**, not 17, added attribute names, plus the stated three parameter names and one module. An independent AST set difference establishes that count. My gadget audit visited 164 objects and produced exactly the same seven callable/module flags as the committed revision-10 audit. The README's 17/163 counts should be corrected or tied explicitly to the earlier audit it describes; the flags are unchanged.

All seven new seeds fit unsigned 32-bit range and remain distinct under signed 32-bit wraparound. In a recording-only constructor check, the normal classifier helper received every full Python integer from BTS_LGBM_RANDOM_STATE unchanged; no model was fitted. JSON round trips and repeated NumPy default_rng/RandomState controls succeeded for all seven. The run's NumPy/scorer code does not consume the training seed as a signed 32-bit array index or scorer seed: the scorer keeps its fixed default seed 42. The manifest records the original integer and environment string, and the validator compares them exactly. I found no additional large-seed-sensitive narrowing in this run. LightGBM's native wraparound and repeatability, and the box's installed 4.6.0 identity, remain the supplied measurements/account.

## Required changes

1. **Close B1 before a claim.** Complete parameter/settings validation and the locked existing-claim check before the final stage-two window check, placing that check immediately before run-directory creation. Add regressions that cross 00:45 during each later validation. Retain the wrapper check and the job check, the exact bounds and positive controls.
2. **Close B2 at the job entry.** Require stage-two computation to enter through the expected guarded C1 unit; an unguarded public `run` must refuse before claiming or consuming inputs. Establish the actual guarded/self-unit context, rather than treating a caller-supplied flag as systemd membership. Keep C1's existing serialization and prove that the intended self-unit is admitted while a direct/unguarded call is refused. This also binds the route to the existing checkpoint and CPU supervision.
3. **Close B3 by admission stage.** X-36 must admit execution only for seeds 4–10 on both routes. Reject seeds 1–3 even when their accepted claim is missing. Preserve historical stage-one validation, its computation and the default three-seed rule. Add missing-directory and missing-claim refusal cases under a stage-two identity.

Nonblocking documentation corrections: update the vocabulary/audit counts with their provenance and replace the addendum's stale “still to be ruled/given” authority wording with the already recorded rulings, while distinguishing the acknowledgement row from the not-yet-created acknowledgement file. No tracked implementation or documentation edits were made by this reviewer.

## What was run

All test, synthetic-probe and mutant processes finished before this report was written. Commands ran from this checkout with offline uv, Python bytecode suppression, America/New_York, and TMPDIR/MPLCONFIGDIR under `/private/tmp/c2-framing-s1-AuxgvV`.

- **Permitted pytest selection:** the full framing directory plus the two named C1 files, with the cache provider disabled. Exit 1; **383 passed, 4 failed in 580.12 seconds**. All 281 framing and 25 admission tests passed; C1 cycle had 77 passes and four failures. Python 3.12.13, pytest 9.0.2, pluggy 1.6.0.
- **The four failures:** payload-leaf/terminal-receipt, overrun kill, nested-descendant kill and unconfirmed-empty-payload guard tests. Each reported `guard preexec failed: PermissionError`. They match the prompt's disclosed sandbox failures, and their guard implementation is unchanged. I did not rerun unchanged main, so the earlier identical-main result remains the author's account. No escalation was attempted.
- **Owned synthetic admission probes:** exit 0. Measured the B1 claim inside the controlled window, six-unit direct execution without any active-job observation, and stage-one launch/run under the stage-two identity with the accepted directory retained outside the namespace. Admissions, inputs, features and walk-forwards were synthetic/stubbed; retained claims, output writing, labels and scoring used the real code, with the suite's 200-trial scoring setting.
- **Git/source checks:** verified detached HEAD, initial/final tracked cleanliness, historical identities/hashes/closure acceptance, accepted listing cardinality, canonical seed order, cap/ruling arithmetic, unchanged computation and unchanged runner boundary. An initial historical-head diagnostic used a short id and correctly refused it; rerunning with the resolved full id returned no problems.
- **Recording-only seed and vocabulary checks:** corrected harness exit 0; all seven constructor/NumPy/JSON controls passed, every ledger anchor matched once, syntax-gate checks were empty, and the 283 recorded outcomes reconciled. An initial harness call used the wrong scope-check signature and failed; it was corrected and rerun.
- **Gadget audit:** exit 0, 164 visited objects, unchanged seven flags.
- **Selected official mutant replay:** R133, S9, S36, S37, S39, S40 and S41; exit 0, all seven RED, `NOT RED: none`. Both mutated files were restored; the tracked diff was empty afterwards. The full 283-entry ledger was inspected, not independently rerun.

No prohibited data, runtime configuration, credentials, network, SSH, box access, real walk-forward or real-data run was used. No accidental outside-checkout source/evidence access was identified. No checkout commit, push, live operation or escalation occurred. The permitted tests created disposable Git histories in owned scratch. The only retained review artifact is this report; all owned scratch is deleted at closeout.
