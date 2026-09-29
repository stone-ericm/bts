# W1.5 Incident Register — Phase 1 Plan (repo-only) — rev 2

**Design:** `docs/superpowers/specs/2026-09-29-incident-register-design.md` v3.1. Codex design r3 signed it with edits, and all seven edits are applied (`04267b6`).

**Goal:** Publish the register's history-derived part, with fixture certificates, without reading any box data:
- Route H candidates
- records
- historical replay and current defence for fixed Tier-A incidents
- strict expected-failure fixtures for unfixed defects with fixed contracts
- a Phase 1 memo

**Phase 2** is Route R: the box extractors, the V1–V11 invariants, and the §5.6 and §6.5 gates. It gets its own plan. It needs register row X-20 and Eric's go-ahead before any box read.

**Branch:** `w15-incident-register-phase1` at `3f6fd63`. The build worktree is under the session scratchpad, and the fast suite passes there: **2457 passed, 16 xfailed**.

## Deviation from design §12.2 (ruling)
Phase 1's code was built test-first in a scratch worktree *before* this plan was written. Codex therefore reviews this plan and the branch's code **together** (plan + code r1), not plan-then-code.
- **Why:** the Phase 1 artifacts are test fixtures and evidence tooling, not production code. Writing the fixtures was the fastest way to establish that each contract is executable.
- **Cost if wrong:** a plan-level defect surfaces in code review instead of plan review. Nothing is merged or deployed before that review.

## Global constraints
- **No production code changes.** New files only:
  - `tests/test_incident_register_2026.py`
  - `tests/test_incident_register_2026_meta.py`
  - `scripts/audit/incident_register/`
  - `tests/scripts/incident_register/`
  - evidence under `docs/audit/2026-09-29-incident-register-evidence/`
- **No `data/` reads and no box access in Phase 1.** Repo documents are read for incident evidence only; quoted outcome statements cite their existing exposure rows (design §5.5).
- **Isolation:** every evidence run uses an isolated worktree with its own venv. Every run records:
  - the imported `bts` path;
  - collected node ids, phases and exit codes;
  - sha256 of the pinned test files;
  - the exact `src/` tree id.
- **A mutant never touches `tests/`.** The harness refuses such a patch.

## Task 1 — Expected-failure fixtures (DONE on the branch)
`tests/test_incident_register_2026.py` gives each incident its own exception class. The class is raised only by `_oracle`, and only for the declared bad value.

| id | Contract (design §10) | Nodes | Status at `3f6fd63` |
|---|---|---|---|
| L01 | BTS Pass: no official AB and no SF, did not play, or suspended with no hit before suspension (clause C) is `void`. Double Down: Hit+Pass = +1 | `grade_pick_in_feed` ×5 strict xfail + 5 controls; `check-results` single ×5 strict xfail + 2 controls; `check-results` hit+pass ×3 strict xfail; absent-player control | 13 XFAIL, 8 pass |
| L02 | reconcile preserves today's applied hit and saver consumption | 2 strict xfail + preview control | 2 XFAIL, 1 pass |
| E77 | 7/16: a moved-up singleton slate's enterable pick is delivered before the true cutoff | 1 strict xfail (`run_day`, component level) + oracle control | 1 XFAIL, 1 pass |

**Validation performed** (design §9.7):
- **`--runxfail`:** all 15 L01/L02 nodes fail in the call phase with their dedicated class and the declared value. E77 fails the same way: `SingletonSlateUndelivered: … never delivered (true cutoff 18:05); DMs sent: [('19:00', 'BTS health CRITICAL alert(s): - [missed_')]`. The only DM that day is the missed-pick alert, 50 min after the true first pitch; it is recorded as containment.
- **Positive direction:** a throwaway fix sketch flips all 15 L01/L02 nodes to XPASS(strict), and the 8 controls stay green. The sketch covers:
  - no AB and no SF → `void`;
  - clause C → `void`;
  - replay including today's terminal file.
- **`test_check_hit_suspension.py::test_grade_resumed_hit_does_not_count`** asserts `miss` for pre-suspension AB-without-hit. It is kept unchanged and recorded as **conflicting coverage**, since the pinned clause C requires `void`.

**Candidates not yet built:**
- **L03** postponed cached-fallback delivery;
- **L04** scoring crash/restart double application.

Per design §10.3, each must first fail unmarked. If a Phase 1 attempt cannot build a faithful scenario, the candidate is recorded as `deferred` with the reason; neither is plan-named.

## Task 2 — Meta-tests (DONE)
`tests/test_incident_register_2026_meta.py` uses pytester to pin the following:

| Case | Reported outcome |
|---|---|
| Intended oracle | call-phase XFAIL (accepted) |
| Fixed behaviour | `XPASS(strict)` → failed |
| A different mismatch | failed (never absorbed) |
| Unrelated call assertion | failed |
| Unrelated setup exception | ERROR |
| Dedicated exception raised in **setup** | pytest converts it to XFAIL (confirmed in 9.0.2), and `accepted_as_reproduction` rejects it (non-call phase) |

It also pins the summary counts, `xfailed=2, failed=3, errors=1`, and `--runxfail` showing the dedicated exception in the call phase.

## Task 3 — Witness module (DONE)
`scripts/audit/incident_register/witness.py`:
- **`hook(tag)`:** appends the caller stack `[qualname, file, line, frame_id]` and the pytest node to `$W15_WITNESS_PATH`. It is a no-op when the variable is unset.
- **`connected(...)`:** certifies, inside one killing node's call phase, that **exactly one** entry invocation (the `entry` hook sits at the top of the entry function, so a reused frame id cannot fake a link) contains both the mutated-branch record and either:
  - the boundary record (wrong or extra event), or
  - its completion record (missing event — the bounded absence witness).

**Tests:** 8, including r2's disconnected direct-call counterexample and a double-invocation case, both rejected.

## Task 4 — Evidence harness (DONE)
`scripts/audit/incident_register/evidence.py`:
- **`use_src`:** replaces `src/` wholesale and verifies it equals the ref, with no extra files. Mutation-checked: without it, the throwaway repo's new-API node reads `passes_at_parent`, which is the pilot's false green.
- **`parse_junit` / `classify`:** per-node, per-phase results — `symptom_candidate`, `new_api`, `setup_error`, `collection_error`, `passes_at_parent`, `absent_at_parent`, `green_not_passing`.
- **`harness_changes`:** the F^→F change audit outside `src/`. An unaudited change to conftest, test data, lock or config labels the run `semantic_regression_replay_unaudited`.
- **`current_defence`:**
  1. green;
  2. mutant plus witness, with test sha256 checked before and after;
  3. restored green.

**Tests:** 6, against a throwaway git repo.

## Task 5 — Historical replay evidence (TO DO)
**Scope:** every fixed Tier-A candidate with a fix commit (list below). Runs go through `historical_replay` with `python = ["uv", "run", "python"]`, after `uv sync --extra model`. Each node is then classified by hand:

| Label | Meaning |
|---|---|
| `symptom` | An observable-contract assertion: delivery happened or not and when, grade value, persisted state, alert sent |
| `diagnostic_only` | A reason string, call count or helper return |
| `new_api` | The test uses an API the fix introduced |
| `unrelated` | Anything else |

Only `symptom` nodes whose run label is `semantic_regression_replay` (harness audit clean, or reviewed) count as historical replay. Otherwise the record reads `historical_replay: unavailable (<reason>)`.

The pilot (30 fixes, `docs/audit/2026-09-29-incident-register-evidence/` once copied) already shows new-API-only red for:
- 3a6e48b, a364b11, 8bceda1, 4f0257a, 41b2bb1, 0abf503;
- all of ce6676d's red except `test_post_cutoff_rescoring_does_not_flip_a_settled_hit` (the one genuine symptom node);
- 2ff2db9 (return shape);
- 736ea8f (import inside the test).

## Task 6 — Current defence mutants (TO DO)
For each causal link of each fixed Tier-A incident:
1. A contract sheet (design §9.1): contract; entry invocation and mode; trigger; symptom kind; allowed mocks.
2. The smallest semantic mutant, as a patch under `…-evidence/mutants/<id>-<link>.patch`. It includes `from bts._w15_witness import hook` with `hook("entry")` at the top of the declared entry function, `hook("branch")` at the mutated decision, and either `hook("boundary")` at the production call site of the checked boundary or `hook("done")` at the entry's return.
3. `current_defence` with the killing nodes.
4. `connected(...)` over the witness records.
5. A verdict. The certificate is `production_path` when only external boundaries are mocked, otherwise `component`. A survivor carries a §9.5 class.

**Priority order:**
1. **Plan-named:**
   - 8/11 — `404358d`: retry classification; DM category advice.
   - 8/13 — `1b50b78` Warmup lock; `224ddce` E3 gate and commit durability.
   - 8/30:
     - `ac0ce8d` cutoff guard;
     - `67338cd` / `3697512` planner (missing-event symptom at 13:20);
     - `c0c0a97` cached-feed sleep (latency symptom);
     - `314154d` health read of the late DM.
   - 9/03 — `0abf503` regime switch.
2. **Other Tier-A with fixes:**
   - 7/12: `9551818`, `ec242da`, `230f65c`;
   - C-03 `ce6676d`;
   - 4/15 `1d61908`;
   - GH #144 set;
   - 6/11 and 7/08 entry checks: `4f13eb3` → `a6ec548`, `8bceda1`;
   - 7/06 `af6329f`, `540b1ab`;
   - 6/17 `2b4ff1d`;
   - 5/05 `7b701c9`;
   - 4/04 `7638af7`;
   - 4/12 `6fd61f9`, `dd25664`;
   - 6/07 `58c9adc`, 6/10 `8cd7207`;
   - 8/09 `4f0257a`, `41b2bb1`;
   - 4/22–4/23 `684c160`, `ddee3db`, 5/09 `864d3aa`, 6/09 `736ea8f`;
   - 4/30 `ee4190f`;
   - 5/21 `43ddf0c`;
   - C-01 `44df03f`.

Subagents may run batches in their own worktrees. I review every patch, and re-run a sample of at least 20 % myself.

## Task 7 — Route H finalization (TO DO)
**Inputs:** the classification of all 1,172 commits since 3/29:
- 806 code/config commits, first-pass subagent classification;
- 366 docs-only commits, separate sweep;
- my review of the 372 runtime-closure negatives: every subject plus cue scan; full body for each of the 101 cue-bearing ones.

**Outputs:**
- Every positive (52 `incident_fix`, 83 `incident_fix?`, 40 `latent_fix`, 82 records) is verified against its commit/doc text and folded into episodes.
- Deployment intervals come from the deploy runs (192, 14 failed) and git ancestry. The deploy branch's history is authoritative from 4/21; before that, main was deployed.
- A seeded 10 % QC of the remaining feature-only negatives.
- The result is `…-evidence/route_h_candidates.json`: each candidate with its disposition, classes, evidence items (§3 kinds and strength) and occurrences, bounded where unknown.

## Task 8 — Records and Phase 1 memo (TO DO)
- `docs/audit/2026-09-29-incident-register.md` and `.json`, per design §7 and §11, marked **Phase 1: history + fixtures; Route R pending (Phase 2)**. The Phase 1 memo sections:
  - coverage inventory (repo sources);
  - records by disposition;
  - near misses;
  - exclusions;
  - `tier_pending`;
  - fixture certificates;
  - rulings.
- The watchdog field of each record feeds W4 rank 2.

## Task 9 — Reviews and merge (TO DO)
1. Codex plan + code r1 (this document and the branch), repo only.
2. After Tasks 5–8, a Codex result review of the memo and evidence (repo evidence access).
3. Fast-forward merge to main, then update the wrap index W1.5 row, the corrections index (C-03 cross-reference) and memory.

Nothing is deployed. The two new test files add 16 XFAIL nodes and 27 passing nodes to the suite.

## Rulings recorded so far (each with its cost if wrong)
1. **Build-before-plan for Phase 1** (above). *Cost if wrong:* a plan-level defect is caught in code review.
2. **E77 is certified at `component` level.** The fixture mocks `fetch_schedule` (the stale morning schedule is the trigger) and status and boxscore lookups, while `run_day`, `run_single_check` and the classifier run for real. *Cost if wrong:* the certificate overstates path fidelity if a mocked function hides the real mechanism. It is labelled accordingly.
3. **Missed-pick alert DMs are containment, not delivery**, in E77's oracle. *Cost if wrong:* none; the pick file is the delivery record.
4. **Exactly-one-entry-invocation rule** in `connected`. *Cost if wrong:* a legitimate fixture that calls the entry twice cannot be certified at `production_path` and must be restructured.

## Rev 2 — how this answers Codex phase-1 r1
| r1 | Fix (branch `39072e3`) | Regression test |
|---|---|---|
| #1 E77 accepted a no-op scheduler | E77 now proves, with ordinary assertions BEFORE the oracle, one check at 18:10, a `game_started_or_final` classification after the true first pitch, the clock past the 18:05 cutoff, and only alert-identified DMs; the oracle compares the whole delivery outcome (pick file flags + identified pick DM + `delivered_at`). The idle helper that read the real wall clock is a declared mock. The 17:30 control is labelled oracle-only; a **positive execution control** (correct 18:10 schedule, canned confirmed selection) runs the same machinery and DMs at 17:10 | measured: a `return None` at the top of `run_day` now fails E77 and its positive control with ordinary assertions (not XFAIL) |
| #2 harness-only mutants; partial restore | `current_defence_v2`: mutations limited to `allowed_paths`; a **witness-only** stage (hooks, no mutation) must leave every node outcome unchanged; `reset_worktree` (reset --hard + clean -fdx except .venv, verified clean) in `finally`; tracked-manifest hash before/after; restored outcomes must equal the first baseline | `test_conftest_only_mutation_is_refused`, `test_edit_to_a_production_file_outside_the_allowlist_is_refused`, `test_hooks_that_change_behaviour_are_rejected`, `test_stray_files_present_before_a_run_do_not_survive_it` |
| #3 `connected` = co-membership | `witness.certify`: exactly one entry record, entry frame in the declared production FILE, ordered entry → branch → boundary (event) or entry → branch → done with **no** boundary under the invocation (absence), plus `positive_control` from the witness-only run; per-run id, pid/thread; synchronous-only (async/process paths `unavailable`) | `test_boundary_before_branch_is_rejected`, `test_absence_with_an_event_is_rejected`, `test_same_name_entry_from_another_file_is_rejected`, `test_absence_certificate_with_positive_control`, `test_absence_without_positive_control_is_rejected` |
| #4 import/JUnit identity | `evidence_plugin` (loaded as a standalone module, never via the worktree's `scripts` package) records collected node ids, every phase's report with the exception class and traceback frames, and the `bts` modules actually imported (path + sha256) from inside the pytest process; `imports_under` rejects foreign modules | `test_import_identity_is_observed_inside_pytest` |
| #5 acceptance incomplete / not wired | `acceptance.accept_expected_failure`: collected once, setup and teardown passed, call-phase XFAIL, exact exception class, raised inside the oracle function of the test file, strict marker with `raises=[class]`, not imperative; used by the runner | `test_codex_counterexamples_are_rejected` (wrong location, imperative, teardown failure, setup, non-strict) |
| #6 audit labels | `historical_replay_v2` + `replay_label`: every outside-`src/` change — including inside the selected test file — needs a recorded audit decision (neutral / irrelevant / adapter) for the clean label; renames keep both paths | `test_v2_label_requires_a_decision_for_every_change`, `test_v2_rename_keeps_both_paths`, `test_v2_replay_classifies_nodes_and_checks_imports` |
| #7 L01/L02 gaps | L01: bench DNP with explicit zero stats; suspended cases graded with an after-resumption clock; whole-state oracles `(slot_results, day result, streak, saver)`; `official_ab_no_hit` and `pre_suspension_hit` added to the CLI path; Pass+Pass double; saver-streak Pass; feed-fetched path proofs. L02: two reconciles, one oracle over both states; a genuinely undelivered preview control. The CLI fixture is described as real CLI grading (not the cron wrapper / flock / wait loop) | acceptance accepts all 18 XFAIL nodes; a throwaway fix sketch flips all 17 L01/L02 nodes to `XPASS(strict)` with every control green |
| #8 procedure | Tasks 5–8 amended below | — |

### Task 5/6 amendment — acceptance objects
Every certificate is an `acceptance.json` from `current_defence_v2` / `historical_replay_v2` plus a **contract sheet** (contract, production entry and mode, trigger, symptom kind, allowed mocks, certificate-level justification) and a **reviewer decision** (accept / reject + reason) that I record after reading the patch and the killing call's frames. A certificate exists only when the runner verdict is `accepted` AND the reviewer decision is `accept`. The v1 runs made overnight (45 mutant runs under the legacy harness, `$SCR/evidence/defence/`) are **preliminary**: their mutation designs and killing-node choices are reused, but every certificate is re-run under v2 (specs split into `witness_edits` + `mutation_edits`, with `allowed_paths`, `entry_file`, `branch_file`, `kind`).

### Task 7 amendment — negative review and deployment intervals
- **Every runtime-closure negative (372 commits) gets a diff-level review**: a first pass records, per commit, the functions/paths its diff changes and an explicit exclusion reason (feature / pre-ship fix within the same change / refactor / research / docs) from the DIFF, not the message; I adjudicate every commit whose diff touches a delivery, grading, state, alerting, liveness, deploy or backup code path, and a seeded 10 % of the rest.
- **Deployment intervals** come only from retained deploy-run logs (SHA installed, canary outcome, rollback) or a qualified operator report; git ancestry alone gives a *candidate* interval, labelled `unknown` for the certificate. Runs whose logs expired (e.g. 5/09, 6/12) stay `unknown`; the 4/21–4/28 window is flagged (runs "succeeded" while installing nothing, a5af424).

### Task 8 amendment — record schema
`docs/audit/2026-09-29-incident-register.json` is validated against a JSON schema implementing design §7 (occurrences with bounds, deployed/mitigated/verified as separate fix states, residuals, evidence kind + strength, explicit `unavailable` certificates) before the memo is published; a record failing validation is not published.
