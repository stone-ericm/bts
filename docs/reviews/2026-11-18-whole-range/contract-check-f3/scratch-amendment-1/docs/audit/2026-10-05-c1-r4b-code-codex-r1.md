# C1 rank 4b code and rank-3 acquisition — round 1

## Verdict

**BLOCK.** Reviewed current main `1b4889b1d18176d918353e63f47aa2da50c0f48c`, including the four interim fixes. Registration SHA-256: `4d03d77a699db7601a539b7dd26a25a233746284ad561b41561ac97d404d00c6`.

The weakest remaining claim is that this runner produces the registered, provenance-bound trade for the concrete final policy. Its parity reader can consume inputs outside the verified manifest, the final object omits its fitted rates, required replay evidence is discarded, and validation/projection gate flags are constants. The numerical engines passing tiny-world checks does not close those boundaries. The acquirer also cannot establish receipt-backed resume or guarantee the cycle pause on every stop-persistence failure.

The smallest repair keeps the existing candidate, classifier, objectives, thresholds and grids. Close the input/validation boundaries, retain the declared evidence and final-fit contract, finish the acquisition receipt/stop protocol, and repair the failing test. No additional candidate or outcome-driven tuning is needed. This is the complete round-one issue list; round two should verify these dispositions.

Evidence is local code/document inspection, the authorized existing tests, and in-memory synthetic probes. No real `data/` profile, policy artifact or acquisition output was read; no historical replay, real July parity, network, ssh or gh operation was performed. No tracked file was edited. The box acquisition's actual state and loaded code were not inspected or changed.

The first reviewed head, `429f95b`, passed all 90 requested tests. At current main, the same requested scope produced **96 passed, 1 failed**. Evidence: `.codex-review/c1-r4b-code/tests-r1b.xml`. The failure is `tests/scripts/c1_r4b/test_run_e2e.py:77`, addressed in F11. These are synthetic checks, not historical measurements.

## Findings

### Disposition of the four interim blockers

| Interim finding | Current verification |
|---|---|
| Coverage inspected only the first seed | **Resolved.** `run.coverage` visits every registered `(season, seed)` and reports date → affected seeds. A missing date in seed 2 with complete seed 1 now makes the disposition inconclusive. The new coverage format is correct; the old e2e assertion is stale. |
| Unsupported projection returned zero | **Partly resolved.** Missing phase types and frequency sums below one now raise `Unavailable`; `project_grid` retains unavailable cells and differences. Positive-weight non-finite rates still produce a numeric zero, and the caller still sets `projections_ok=True`; F4. |
| Receipt failure after a 429 lost the marker | **Resolved for that tested sequence.** The feed handler now writes/fsyncs the stop file before attempting the completion receipt. The new regression test passes. Stop-file failure, schedule requests and the launcher boundary remain separate gaps; F9. |
| Calibration/leg-stress removal survived the projection tests | **Resolved for the demonstrated mutant.** The new hand-computed category test kills the in-memory mutant that ignores calibration and analytic leg stress. The new path checks use independently calculated transforms for their expected result, rather than the same mutated helper. |

### F1 — P1: the verified input set is not the consumed input set, and the execution recipe is not fully pinned

**Locations:** `scripts/audit/c1_r4b/run.py:61–93,246–305,313`; `scripts/audit/dd_p_policy_value_sensitivity.py:628–641`; registration §§3, 7.

`verify_profiles` hashes file bytes, but the runner subsequently reopens the directory through July's reader and later reopens each parquet for corrected evaluation. The verified byte buffers are not what either reader parses. Changing a profile after verification therefore breaks the promised consumed-byte identity without another check.

More directly, `_profiles` restricts seasons to 2021–2025, while `ddp.main` calls `load_paired_pooled`, whose recursive glob consumes **every** `backtest_*.parquet` under the root. A synthetic probe supplied one 2021 and one 2026 path and recorded reads of both. A surplus 2026 file is not in `prof_sha`, yet its labels are decoded during parity. This violates the frozen input set and D3 RESERVE even if the subsequent anchor check rejects the run. This is a verified reader-path counterexample, not a claim that the real root contains such a file.

The manifest also omits the required profile-generation command/commit and estimated-PA recipe/schema, replay/solver source hashes, and a binding to the reviewed executable sources. `HEAD` plus X-31 ancestry does not reject a dirty or subsequently unreviewed implementation. Schedules are only hashed as supplied; their hashes are not compared to the frozen outcome-free calendar evidence. A self-consistent shortened schedule could remove an uncovered final date and be accepted as complete. The five published schedule hashes are in the calendar-coverage document; this review did not read the schedule files.

**Required repair:** freeze an explicit authorized inventory and parse its verified bytes in both parity and corrected evaluation; forbid surplus/unregistered outcome reads. Bind the calendar version, producer recipe/schema, lock and executing sources to reviewed evidence before decoding outcomes. Reject missing or mismatched provenance. A newly authorized backfill/calendar version needs its own recorded identity, not a fresh hash treated as sufficient provenance.

### F2 — P1: acquisition game deduplication is an incorrect basis for the opportunity calendar

**Locations:** `scripts/audit/c1_r3/acquire.py:55–70`; `scripts/audit/c1_r4b/data.py:57–64`; registration §4.

`schedule_games` claims to date a suspended game by its last played listing, but preferentially uses `officialDate`, then collapses all listings for the game into one date. In a synthetic schedule with a suspended listing on June 1 and a Final listing on June 2, both carrying official date June 1, it returns June 1. `calendar_from_schedule` consequently produces a one-day calendar ending June 1 and loses June 2. Existing fixtures make every `officialDate` equal to its listing date, hiding this case.

Changing the preference to the listing date alone is insufficient: selecting only the last listing would then lose the original played day when that was its only game. A unique feed inventory and the set of calendar dates with an evidenced opportunity are different objects. `NOT_PLAYED` also treats every other status as played without validating a supported played-status contract.

**Required repair:** construct the opportunity-date census directly from validated schedule listings under the declared contest-date convention, separately from deduplicating game IDs for downloads. Retain and test original and completion listings whose official/listing dates differ. Do not assert that either convention resolves the real 2023-10-02 question; its owner row remains OPEN.

### F3 — P2: required identity validation accepts invalid game and batter IDs

**Locations:** `scripts/audit/c1_r4b/data.py:74–90,118–120`; `scripts/audit/c1_r3/acquire.py:111–116`; registration §3.

Integer dtype alone does not establish a valid identity. A synthetic profile with batter IDs `[0,-7]` and game IDs `[0,-2]` passes validation and produces an eligible different-game partner at rank 2. Its labels can then drive a double success. This contradicts pairing through a **valid** different game identifier. The acquirer similarly accepts JSON `gamePk:101.0` for integer 101 and `gamePk:true` for integer 1 because equality is coercive.

**Required repair:** require positive integral game/batter identities, reject boolean/fractional/coerced identities at decoding boundaries, and test invalid IDs on both the primary and partner. Preserve the existing duplicate/null/rank/label checks. Do not silently demote malformed partner rows into missingness.

### F4 — P1: projection support and required-validation status still do not close the acceptance boundary

**Locations:** `scripts/audit/c1_r4b/project.py:44–53,84–99,109–111`; `scripts/audit/c1_r4b/run.py:218–227,394–395`; registration §§5–6.

The new check validates only existence and the sum of phase frequencies. It does not require finite nonnegative frequencies or supported coherent rates. For a positive-weight type with `p_hit=p_both=NaN`, the ordinary `max(0.0, NaN)` transform treats both as zero. The current synthetic projection returned `{p_reach:0.0, e_best:0.0, mass:1.0}` rather than unavailable. A NaN frequency can also bypass the sum comparison. This is a remaining unsupported-rate counterexample; the new missing-phase test does not cover it.

`project_grid` discards the returned mass and checks neither finite probabilities nor action-table validity. `main` then supplies `validation_ok=True` and `projections_ok=True` unconditionally. It never binds an independent validation result to the sources it actually runs, or derives projection status from its cells. The disposition helper's tests for `validation=False`/`projections=False` cannot protect this caller. The new unavailable-cell handling is useful, but it does not establish gate 6.

The supplied valid-data estimator normally produces finite empirical rates; the NaN probe is a boundary check, not a claim of observed real-profile corruption. Likewise, I have not established that an empty held-out phase can coexist with complete real coverage. The constant gate flags are nevertheless not evidence of the required checks.

**Required repair:** validate projection environments independently of the fitting-bin positive-weight rule: zero-weight cells need no rate; positive-weight cells require finite coherent rates and supported bin mappings. Check finite output and conserved mass, retain unsupported cells as unavailable, and derive gate 6 from actual bound validation/table evidence. Never interpret missing support as quantified zero cost. Include end-to-end tests that cannot pass with constant validation/projection flags.

### F5 — P1: the concrete final object lacks the registered full rate/recipe manifest

**Locations:** `scripts/audit/c1_r4b/run.py:290–295,332,354–367,400–402`; registration §§3, 8–9.

The final fit does correctly use all five seasons and `D_final=max(calendar.horizon)`, and the artifact/action bytes are hashed. But its NPZ contains policies, cuts, horizon, phase length and the generic run-manifest hash only. Neither that manifest nor `fit_stats` retains the actual phase/type frequencies and primary/joint rates. `fit_stats` records counts by bin/partner, not the hit counts or rates. The common-refinement environments and their mappings are also not saved. The same omission affects fold projection provenance.

This is not the promised reviewable object with its **full rates/classifier/calendar/recipe manifest**. It cannot support the declared independent equality re-solve or explain the exact model environment used for the final trade without decoding the historical corpus again. The final projections are separate in the output, which is correct, but their transition environment is not materialized with them.

**Required repair:** save and hash the fitted early/late availability-conditioned environments, refined evaluation environments/mappings, objective/saver/tie/phase/target recipe and calendar identities, together with the action/classifier hashes. Bind final projections and any later packaging to exactly those objects. Retain the correct all-season fit and maximum-horizon rule; do not choose a new rate version or calendar after results.

### F6 — P1: required trade-table evidence is discarded or never computed

**Locations:** `scripts/audit/c1_r4b/run.py:160–182,207,260–261,336–345,381–402`; `scripts/audit/c1_r4b/replay.py:125–127,154–157`; registration §§4–5, 7–8.

The missing outputs are substantive, not a request for cosmetic tables:

- `_arm_metrics` discards every action count returned by replay. Results therefore omit play/single/double/skip counts and legal demotions for all arms/Δ. Replay also excludes no-opportunity/unknown dates from its skip counter, so reporting must identify that denominator and separately retain forced no-play calendar counts.
- Masks are generated but achieved marginal and primary-hit-conditional haircuts are never retained. No primary/partner availability census, partner-rank distribution, excluded-row counts or availability census on arm-skipped dates is reported by season and seed. Availability is inspected before arm decisions, as required; its evidence is simply not delivered.
- `reach57` is stored as an averaged trajectory proportion, not the registered observed reach-57 counts/denominators. A1−A0 adaptation is not explicitly reported. Per-season max results are limited to A0/A1/A2 at Δ=0; comparator and stressed per-season results/ranges are absent.
- Projection policies include only A0/A1/A2, omitting the comparator rows from the registered all-arm trade. Paired jackpot differences are in probability units only; percentage-point units are absent. Fitting-rate and held-out-rate equal-season summaries and scenario ranges, and final-object equal-calendar summaries, are absent.
- Zero-frequency availability/refinement cells are omitted rather than retained explicitly at zero weight. This is numerically equivalent for a supported model, but deviates from the registered census/support representation. The output does not carry the required model-projection/exposure qualifications and headline trade wording; those must accompany any eventual displayed tables.

**Required repair:** preserve the per-seed/per-replicate metrics, counts, mask reductions and denominators needed by the registration, then produce all declared summaries and explicitly qualified projection tables. Report unavailable cells without a favorable-scenario selection. A downstream memo cannot recover the discarded action/coverage/thinning evidence from this `results.json` alone.

### F7 — P1: registered pre-result stops and execution settings are not enforced as one preflight

**Locations:** `scripts/audit/c1_r4b/run.py:61–70,272–274,299–360`; registration §§3–4, 7; calendar-coverage document; owner row C1-4b-2023-10-02.

`--reps` permits arbitrary values, and they are recorded and used as if registered. The real runner can therefore execute a non-200-replicate screen; writing its count into a manifest does not authorize changing the frozen precision rule. Synthetic smoke tests may use a separate explicit test harness, but the outcome-bearing CLI must enforce 200.

The runner checks each fold's fitting bins only immediately before evaluating that fold. A later fold or final-fit degeneracy is discovered after earlier corrected outcomes have already been evaluated. This falls short of the declared empty/degenerate-fitting-bin stop **before result inspection**. No bound independent-evaluator preflight is executed or verified before parity/corrected results; F4 covers the constant status.

The currently unset X-31 guard refuses a run, which is correct. Publishing X-31 is not itself the OPEN coverage decision. The calendar finding explicitly requires an owner choice before an outcome read, and the latest owner row says backfill is being decided before any 4b result. The run guard contains no check for that disposition. Preserve the frozen unknown-coverage → inconclusive rule unless Eric records a replacement; do not silently remove the date, infer a waiver from X-31, or treat this review as choosing a backfill. The separate prospective 2027 contract **has** been waived for 4b by C1-4b-gate; it must not be reintroduced as a blocker.

The stochastic MC ambiguity comparisons are conservative and implement the sampled-zero-SE boundary rule. There is no explicit proved-mask-invariance path; a zero sampled variance alone must not be used to invent such proof. No retuning or automatic result rerun is authorized by these repairs.

**Required repair:** preflight the frozen settings, owner/calendar/provenance disposition, all fold/final fitting environments and bound evaluator checks before corrected result inspection. Make the single authorized run and any invalidation/reviewed repair explicit in the execution receipt. Retain direct Δ=0 comparisons and permit a mask-invariant exception only with actual structural evidence.

### F8 — P1: acquisition receipts do not cover individual requests, and resume ignores their byte identities

**Locations:** `scripts/audit/c1_r3/acquire.py:104–134,149,155–161,183–190,214–229`; `tests/scripts/test_c1_r3_acquire.py:98–103`; owner's rank-3 acquisition ruling.

One intent/completion pair surrounds `_get`, which can make three HTTP attempts. Thus the existing 503 test makes four HTTP requests over two games but records only two intent/completion pairs. Retries have no individual identities/timestamps/status receipts. Schedule requests have no intent/completion receipts at all. This contradicts the acquirer's per-attempt receipt claim and leaves the request interval incomplete.

Resume checks only decompressibility, JSON and matching `gamePk`. It does not verify either hash against a completion receipt. A synthetic changed/unreceipted feed with the same game ID returned `_verified=True`. A crash after `os.replace` but before the completion append produces the same orphan: the next run silently skips it without establishing its response receipt. Feed bytes and their parent directory are not fsynced before the allegedly durable completion is appended either. Receipt fsync alone does not establish durability of the stored response it names.

**Required repair:** receipt each actual HTTP attempt, including schedules and retries, with its result and retained byte identity where available. Before skipping a stored file, verify its receipt-bound hashes; explicitly reconcile or refuse orphan/conflicting bytes. Persist successful response bytes before publishing their durable completion. A syntactically valid body with a game ID is not proof that it is the receipted authoritative metadata; rank-3 fitting still waits for X-34 and its separate metadata checks.

### F9 — P1: the whole-cycle 403/429 pause still fails if stop persistence fails, and schedule stops use the old path

**Locations:** `scripts/audit/c1_r3/acquire.py:96–101,162–172,210–225`; `scripts/audit/c1/launch.py:40–45,89–91,157–160`.

The repaired feed sequence survives a **later receipt** failure. It does not survive failure of `_write_durable` itself: a synthetic 429 plus an injected stop-file write failure leaves only the prior intent and raises `OSError`; there is no stop file or rate-limited completion. The launcher considers stop filenames only, so it can permit the next C1 job. This verified error path still violates “any 403/429 pauses all of C1; no rerun without Eric.”

The schedule handler still uses ordinary `write_text` rather than the new helper. `_write_durable` fsyncs the file but not its new directory entry; durable creation/replacement is not fully established. The launcher scan covers the default C1 output tree and static capture path, but a supported custom acquisition `--out` outside those roots places its marker outside the scan. These are code/configuration gaps, not evidence that a rate limit or persistence failure happened on the box.

**Required repair:** use one durable stop protocol for schedule and feed requests, bind its shared pause location to the launcher independently of output overrides, and fail closed on an unresolved request/failed stop-persistence interval. Retain a recovery witness the launcher can actually inspect when the marker cannot be written; absence of a marker must not prove a cleared rate-limit state. Test schedule 403/429, marker failure, receipt failure, resume and alternate output roots together. Do not automatically clear a proven rate-limit stop or rerun it without the recorded owner decision.

### F10 — P2: the declared 4 CPU-hour/3 wall-hour run limits are not enforced by the launch plan

**Locations:** `scripts/audit/c1/launch.py:77–112`; registration §10.

A pure synthetic `plan_launch(cpu_hours=4, max_hours=3, rows=[])` returns `LimitCPU=360000` (the entire remaining 100-hour cycle budget), and no wall-time limit property. `cpu_hours` is used for the admission check, and `max_hours` for the seasonal sleep-window check, but neither enforces the registered per-job stop during this off-season run. This is a verified argv gap; no runtime/resource measurement on real data is claimed.

**Required repair:** enforce the declared job CPU and wall limits in addition to the cycle cap, with a stop receipt that prevents an automatic retry after overrun. Verify the launch configuration without launching a job. The existing direct restic-backed output root is a disclosed departure from build-plan T8's validation-directory-plus-copy layout; it is acceptable storage if the plan/index is made consistent and the evidence is retained.

### F11 — P2: the current requested suite is red, and some integration assertions are vacuous

**Locations:** `tests/scripts/c1_r4b/test_run_e2e.py:61–77`; `tests/scripts/c1_r4b/test_project.py:91–101,114–156`.

The full current run fails because line 77 expects empty lists, while coverage now correctly returns `{season:{}}`. Change that expectation to the new date → seeds contract and verify the remaining assertions after it. Do not restore the first-seed behavior to satisfy the stale test.

The e2e disposition assertion accepts any of the three possible outputs and therefore cannot catch a wrong disposition. Its parity and X-31 paths are stubbed. The original h=0 “iid equality” compares the same evaluator with `Stress()` and `Stress(h=0.0)`—identical arguments—so that equality alone can only pass and is not an independent iid reference. The m-dependent scalar path oracle and the new independent stress transforms do provide separate useful checks, but none certifies the runner's provenance/status/output boundaries.

**Required repair:** assert an independently determined synthetic disposition and actual caller gate consequences, add bounded synthetic parity-input/failed-parity checks, protect a heterogeneous-calendar final horizon, and compare h=0 to an independent expectation/path calculation. Add regression checks for the remaining findings; run the full requested suite again after the code changes.

### Correctness checks and section coverage

These are bounded checks, not a signature for the full run:

- **§§1–2:** the implemented candidate is own capped E[best], not a field/prize objective. A1 retains matched reach routing and E[best] continuation; A2 uses E[best] throughout. The solver integrates availability **inside each quality bin before choosing one raw action**, rather than optimizing separately after observing the partner. Tiny recursive expectimax/value checks and existing play/stop/saver/crossing tests passed. The reach slice matches production `solve_mdp` in the tested all-partner synthetic environment. No numerical solver deviation was found in these checked cases.
- **A0/§4:** I compared `replay.A0` to the actual production `mdp_objective`, `effective_days`, `lookup_action` and `lookup_tail_action` helpers across **11,136 healthy synthetic cases**, including boundary equality, both routes, horizon caps, terminal states, saver states and trusted best. There were **zero mismatches**. The relevant A0/replay code is unchanged by `1b4889b`; this check is reused from the interrupted review. Legal demotion occurs in replay's shared execution clamp. This does not verify real pinned artifact contents or entry/delivery behavior.
- **§3:** upper-bin cutpoint equality, same-stratum primary/joint rates, four-season fitting and all-five-season final fitting are implemented. Horizon/phase definitions use calendar days. F1–F3/F5 cover the remaining provenance, calendar and identity contract defects. Observed primary absence is not expressible in these profile rows; the code conservatively labels a missing opportunity unknown rather than inventing an absence witness. Report that limitation and its census under F6.
- **§4 stress:** policies are held fixed across Δ, only eligible partner hits are thinned, primary outcomes remain unchanged, streams are bound to `(season, seed)`, and uniforms are nested/shared between arms. F6–F7 cover the missing achieved-haircut/fix-ladder evidence and unguarded replica count. **The cumulative fix ladder itself is absent:** add calendar → legal demotion → m plumbing → A0 tail results for the fixed reference arms; an inert m step is not empirical m validation.
- **§5 projection:** on supported coherent synthetic inputs, the forward evaluator retains best across resets, saver consumption and exogenous primary-hit run state across skipped days, resets r on no-opportunity/no-primary days, applies legal +1 fallback and caps/absorbs at the target. Its stress transform order matches the registration; the new independent transform tests pass. F4–F6 cover support, binding and output gaps. The solver oracle does not call the production transition implementation; the original projection oracle did share `categories`, now supplemented by the independent transform cases. A0/common-refinement tail bin 0 is correct for the specifically pinned one-bin tail, not a generic multibin tail implementation.
- **§§5–6 aggregation:** seeds are averaged within seasons, then seasons equally; replicate-set paired contrasts use that same weighting; `mc_se` is `sd(T_r,ddof=1)/sqrt(R)`. Reach-20 uses the registered two-percentage-point limit against both A0 and A1 at Δ=0 and 0.10. The numerical positive/negative/ambiguity rules match the stated precedence when given valid honest flags. F4/F7 concern whether the caller establishes those flags and prerequisites.
- **§§7–8:** later-seed coverage is repaired, but the required counts, uncertainty/support qualifications and complete historical/projection/final-object summaries are still incomplete; F6. No historical positivity, jackpot-cost estimate or power claim is made by this review.
- **§§9–10 and owner rulings:** final approval remains bound to a concrete object, independent acceptance and D7; integration/loader/deployment are later work. Eric's 4b prospective-contract amendment is controlling. The coverage decision is still OPEN, and X-31 is still unset. F5/F7/F10 are the remaining decision-object/execution prerequisites. No code signature here approves an outcome run, rerun, backfill, commit, push or activation.

## Required changes

1. **F1:** consume only verified frozen bytes in parity and corrected replay; complete and check the producer/source/schema/calendar provenance. Test a surplus 2026 profile and changed-between-verification-and-decode bytes without reading either.
2. **F2–F3:** separate schedule opportunity dates from the unique feed inventory; validate supported date/status and exact positive identities. Test differing official/listing dates, multiple played listings, and malformed primary/partner IDs.
3. **F4:** finish positive-weight projection support validation and mass/finite checks; make the caller's validation/table status evidential. Preserve unavailability and its precedence/qualified reporting rather than quantified zeros.
4. **F5:** save/hash the full fitted and refined environments and recipe, and bind the final-object projections and future package to those unchanged actions/rates/classifier/calendars.
5. **F6:** deliver every registered census, action/demotion/reach count, achieved haircut, season/stress/adaptation result, cumulative fix ladder, all-arm projection and separate equal-season/final-calendar scenario summary with units and qualifications.
6. **F7:** enforce 200 outcome-bearing replicas; preflight all registered checks before corrected results; preserve the OPEN owner coverage decision and the single-run/invalidation rule. Do not add a prospective-contract requirement that Eric waived.
7. **F8–F9:** implement per-request receipts and receipt-backed durable resume, plus a shared fail-closed schedule/feed/launcher stop protocol including persistence errors and output overrides. These edits and any affected acquisition evidence require review; this report does not authorize live recovery or restarting the already running acquisition.
8. **F10:** enforce the declared per-job CPU/wall caps and document the direct restic output location consistently.
9. **F11:** correct the coverage assertion, strengthen the actual boundary/caller tests, and obtain a clean full requested suite at the corrected pinned sources. Keep the demonstrated calibration/leg-stress mutant killed.

No numerical threshold, policy family, fitting population or stress-grid change is requested. Where provenance or an owner prerequisite cannot be established, stop/defer rather than manufacture a pin or select a favorable run.
