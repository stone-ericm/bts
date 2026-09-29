# W1.5 Incident Register — Phase 1 Plan (repo-only) — rev 4

**Design:** `docs/superpowers/specs/2026-09-29-incident-register-design.md` **v3.2** (v3.1 = Codex design r3 SIGN WITH EDITS, all seven applied, `04267b6`; v3.2 = §9.3 step (3) and its witness requirements replaced verbatim by Codex phase-1 r3's amendment).
**History:** rev 1 `04edfac` → Codex phase-1 r1 BLOCK → rev 2 `d8563a7` → Codex phase-1 r2 BLOCK (`docs/audit/2026-09-29-incident-register-codex-phase1-r2.md`: six tooling blockers, E77 success branch, plan consolidation) → rev 3 `9907cb9` → Codex phase-1 r3 BLOCK (`docs/audit/2026-09-29-incident-register-codex-phase1-r3.md`: absence completeness, observer side effects, replay/pair closure, deploy bounds, record binding; r2 #1, #7, #9 RESOLVED) → **rev 4 (this document): tasks rewritten in place; the last two sections map each r3 finding, and each finding of my own review before r4, to its fix and regression test.**

**Goal:** publish the register's history-derived part with fixture certificates, without reading box data:
- Route H candidates and records (design §2, §3, §6.1, §7);
- historical replay and current defence for fixed Tier-A incidents (design §9.2–§9.5);
- strict expected-failure fixtures for unfixed defects with fixed contracts (design §9.7, §10);
- a Phase 1 memo (design §11).

**Phase 2** (Route R: box extractors, V1–V11, the §5.6 and §6.5 gates) gets its own plan and needs register row X-20 plus Eric's go-ahead before any box read. Box-held deployment evidence (`deploy_history.txt`, B1) is Phase 2.

**Branch:** `w15-incident-register-phase1`. The commits:
- `321f401`: the v4 tooling and the L03/L04 fixtures;
- `0166e13`: the E77 verdict change (Task 1). No tooling file differs from `321f401`;
- `27b260d`: tests that close the gaps the first whole-suite sweep found, plus the sweep script. It touches no tooling module or fixture;
- `74b04e8`: the Route H drafts, the sweep output and the pair acceptance;
- earlier: v3 `c5cf6bc`, `638a982`.

The build worktree is under the session scratchpad; evidence runs use separate OWNED worktrees (Task 3).

Measured:
- review suite (`tests/scripts/incident_register` with `uv run --with jsonschema==4.23.0`, at `27b260d`): **175 passed**;
- fast suite (at `27b260d`): **2594 passed, 1 skipped, 22 xfailed**;
- expected-failure pair (`run_expected_failures` on an owned worktree, at `0166e13`; the tooling modules and fixtures are byte-identical at `27b260d`): **accepted, 22/22**; 20 connected to production, 2 `exception_shape` (E77, L03);
- strict mutation sweep:
  - at `321f401`: 67 of 74 killed (`…-evidence/tooling/mutation_sweep-321f401.txt`). The seven gaps are closed in `27b260d` (last section);
  - at `27b260d`: **75 of 75 killed**.

## Rulings (each with its cost if wrong)
1. **Build before plan for Phase 1.** The fixtures and tooling are test code, not production code; Codex reviews plan + code together. *Cost if wrong:* a plan defect surfaces in code review; nothing merges or deploys first.
2. **The causal witness is a trusted external `sys.monitoring` observer**, as design v3.2 §9.3 now permits (Codex r3 supplied the amendment; it did not approve the r3 implementation, see Task 3). *Cost if wrong:* a certificate built on an incomplete recorder overstates absence; every coverage limit is therefore reported as `unavailable`, never as absence.
3. **E77 stays `component` level** (mocks listed in the fixture). *Cost if wrong:* the certificate overstates path fidelity if a mock hides the mechanism; labelled accordingly.
4. **Missed-pick alert DMs are containment, not delivery** (E77 oracle). *Cost if wrong:* none; the pick file + identified pick DM are the delivery record.
5. **Deployment intervals are bounded by log observation points** (Task 4); ancestry alone is a labelled `candidate`. *Cost if wrong:* records before 7/02 carry `unknown`/`candidate` deploy states until Phase 2 reads `deploy_history.txt`.
6. **Contemporaneity (design §3) for continuing conditions** (ruling 2026-09-29, asked by the drafting readers): a report is contemporaneous for an occurrence that cites it when written after the occurrence began and within 48 h of the time the occurrence was last observable — the event itself for a one-shot occurrence; for an occurrence marked `continuing` (a stall, a frozen value) the EARLIEST of its own mitigation / verified recovery and the installs of the fixes of its own links (`links`, default every link), or any time while none of those is known. *Cost if wrong:* a report written during a long-running condition establishes it as observed; a report written long after a one-shot event does not. **Fix steps (consolidation, 2026-09-29):** a report may instead describe the fix step that cites it — a mitigation, an install dated by `operator_report`, or a verification ("verified live on the box"); it is then judged against that step (written after the step could have happened, within 48 h of its latest time) and establishes that step only, never an occurrence. Each role is judged on its own: a report cited by both an occurrence and a step must meet both rules. A report cited by neither is rejected. *Cost if wrong:* a late write-up could date a verification it did not witness; the 48 h window bounds that.
7. **Expected-failure connection kinds**: `return` (the declared production function's last return equals the oracle's actual), `reads` (the values the fixture read back through production loaders, outside any live entry invocation and after the entry ran), `derived` (a label computed by the fixture — E77 — reported as `exception_shape` with the recorded fixture review, never as machine-connected). *Cost if wrong:* E77's reproduction rests on the fixture review, stated as such.

## Global constraints
- **No production code changes.** New files only under `tests/test_incident_register_2026*.py`, `scripts/audit/incident_register/`, `tests/scripts/incident_register/`, `docs/audit/2026-09-29-incident-register*`.
- **No `data/` reads and no box access.** Repo documents are incident evidence only; quoted outcome statements cite their exposure row (X-01 for operational commit text).
- **Every evidence run** happens in an owned worktree (Task 3) with its own `.venv`, under the trusted observer; every stage passes the session gate; `acceptance.json` is written on every path.
- **Mutations** only touch existing tracked `src/bts/**.py` files (hard rule); tests, conftest, config, lock files, scripts and the observer are frozen whatever a spec allows.
- **A certificate** = runner verdict `accepted` AND my recorded reviewer decision `accept` (after reading the patch, the killing call's frames and the certificate's linked events).

## Task 1 — Expected-failure fixtures (DONE)
`tests/test_incident_register_2026.py`: **42 nodes = 20 pass, 22 strict XFAIL** (L01 ×15, L02 ×2, E77 ×1, L04 ×3, L03 ×1). Each incident has its own exception class (not an `AssertionError` subclass) raised only by `_oracle` for the declared bad value; other mismatches are ordinary failures.

| id | Contract (design §10) | XFAIL nodes | Controls |
|---|---|---|---|
| L01 | BTS Pass (§6 A/B/C; clause C → `void`); DD Hit+Pass = +1, Pass+Pass preserved | `grade_pick_in_feed` ×5; real `bts check-results` single ×5, hit+pass ×3, pass+pass, saver-streak Pass | SF-only, official AB no hit, resumed-only, pre-suspension hit (direct + CLI); absent player pending |
| L02 | reconcile preserves today's applied hit and saver consumption | two 23:00 reconciles ×2 | unplayed undelivered preview |
| E77 | a moved-up singleton slate's enterable pick is delivered before the true cutoff | 1 (component) | positive execution control; verdict unit tests (fixed direction passes, bad direction raises, late delivery / no check / other mechanism fail ordinarily) |
| L04 | one result affects the streak and the saver at most once across a crash and restart between the streak write and the terminal pick save (§10.3) | 3: real `bts check-results` hit (5 → 7) and saver miss (12 → 0), and a polling death followed by the 01:00 scorer | no-fault control (a rerun is a no-op) |
| L03 | no delivery names a game evidenced postponed before the send (§10.3) | 1 (component, E77 harness): the 18:10 projected pick, postponed at 18:20, the 18:35 refresh fails, the cached pick is DM'd | positive execution control (the cached pick of a playable game is delivered) |

**E77 (Codex r2 #7):** the schedule mock is truthful at every instant (19:10 before the declared 12:00 move, 18:10 after); the cascade returns a canned confirmed selection; `_e77_verdict` proves the day ran past the cutoff, applies the stale-plan / check-at-first-pitch / containment-only assertions **only on the bad outcome**, and lets a verified pre-cutoff delivery reach the required branch **by any path** (so any repair XPASSes). Changed 2026-09-29 (rev 4, for r4 review): the required branch no longer requires a lineup check before the cutoff, and the up-front 'a check ran' assertion moved into the bad-path mechanism proof. The reason is the L04 lesson: a woken-fallback repair would have failed ordinarily. The harness now also mocks the fallback refresh's import-time `bts.scheduler.run_and_pick`. The change is pinned by `test_e77_verdict_any_pre_cutoff_delivery_path_passes`. Measured: a `return None` at the top of `run_day` fails the marked node and the positive control ordinarily (no XFAIL).

**Registry + acceptance (rev 4):** `docs/audit/2026-09-29-incident-register-evidence/expected_failures.json` (schema v2) binds each node to its exception (`module.qualname`), oracle (`file::_oracle`), production entry, bad/required values (repr for the message, JSON for the structured record) and a CONNECTION: `return` for the five `grade_pick_in_feed` nodes, `reads` for the ten check-results nodes, the two L02 nodes and the three L04 nodes (the fixture's read-back through `load_pick` / `load_streak` / `load_saver_available`, outside any live entry invocation, after the entry ran), `derived` for E77 and L03 (each with its recorded fixture review). The fixture's `_oracle` appends a structured record to `$W15_ORACLE_OUT` in evidence runs. `acceptance.run_pair` runs the marked and `--runxfail` executions over ONE frozen closure (every tracked file's working bytes, the untracked list and the environment's content fingerprint re-checked after each run) and requires, per node: marked XFAIL in the call phase, strict, `raises` = the registered class, not imperative; `--runxfail` failure with exactly that class raised in the registered oracle, the message carrying both values; the production entry observed in the call phase; the oracle record's actual = declared bad = registry; and the connection (`connected`, or `exception_shape` for `derived`). `run_expected_failures` writes a rejected artifact on every failure path, with the resolved ref, registry sha256 and one overall verdict; a session-level reason rejects the whole pair. **Positive direction:** a throwaway fix sketch flips all 17 L01/L02 nodes to XPASS(strict) with every control green (measured on the unchanged L01/L02 fixtures by Codex r2); E77's fixed direction is pinned by `test_e77_verdict_fixed_direction_passes`.

`tests/test_check_hit_suspension.py::test_grade_resumed_hit_does_not_count` (asserts `miss` for a pre-suspension AB without a hit) is kept unchanged and recorded as **conflicting coverage** (clause C requires `void`).

**L03 and L04 (§10.3 characterization candidates) are built (2026-09-29).** Each failed unmarked first, with its dedicated exception and the declared values. A simulated repair in a scratch worktree turned each node XPASS(strict): the pick saved before the streak, in check_results and in run_result_polling; a status check before the fallback delivery. The fault in L04 is the terminal pick save raising a `BaseException` before it writes (as nothing catches SIGKILL/OOM/restart); only the fault point is asserted, never the streak at the death, so any repair shape can reach the required branch. L03 patches both bindings of the cascade (`bts.orchestrator.run_and_pick` and the fallback refresh's import-time `bts.scheduler.run_and_pick`); its connection is `derived` (exception_shape + the recorded fixture review).

## Task 2 — Meta-tests (DONE)
`tests/test_incident_register_2026_meta.py` pins how pytest 9.0.2 reports each shape (intended oracle → XFAIL; fixed → XPASS(strict) failed; other mismatch / unrelated assertion → failed; setup error → ERROR; a dedicated exception raised in setup IS converted to XFAIL by pytest — which is why acceptance requires the call phase). It documents pytest; the acceptance rule is `acceptance.accept`.

## Task 3 — Evidence tooling v4 (DONE on the branch; Codex phase-1 r2 + r3 blockers)
| Module | Role | Answers |
|---|---|---|
| `owned.py` | `create` / `assert_owned` (exact canonical root, linked not primary, detached, matching ownership record in the worktree's private git dir) BEFORE `reset` / `swap_src` / `apply_edits` / `destroy`; hard mutation rule (normalized relative path, existing tracked symlink-free `src/bts/**.py`); edits validated before any write; working-bytes manifest + untracked list; **content** fingerprint of every installed file in the worktree's `.venv` and of every directory a `.pth` adds from outside the worktree | r2 #1, #2; r3 #4 |
| `observer.py` | pytest plugin (loaded by path under a random name). Session identity + the production src digest at session start and end. During the killing nodes' call phase, `sys.monitoring`: entry starts and exits (return or exception), the mutated line, and every call of a declared boundary recorded on the CALLEE side (a boundary mock's `__call__`, a boundary function's code), so C-invoked and threaded calls count; a binding replaced by an unseen callable, or a callable with no Python code, is a `boundary_gap`; threads started in the phase and alive at its end are counted. Values are serialized WITHOUT running application code (exact primitives; a plain instance's raw `__dict__` read through the standard C descriptor; else unavailable) | r2 #5, #6; r3 #1, #2, #7 |
| `runner.py` | trusted bootstrap; `gate` for every stage: observer identity, own venv, rootdir, pytest not from the worktree, finished session with exit status = return code, no collection/observer errors, exact inventory, one setup/call/teardown per node with the per-mode rules (`green`, `mutant` — also the historical red stage, `expected_failure`), return code explained by the nodes, imported `bts` modules under `<wt>/src`, and the src digest at session start and end equal to the tree the runner prepared | r2 #3, #6, #9; r3 #3, #4 |
| `certify.py` | `event` / `return`: branch and event linked by a COMMON LIVE entry invocation (frame on both stacks, no exit and no newer entry record for that frame in between — recursion and frame-id reuse handled). `absence`: the invocation that executed the branch exited inside the interval (completion), no outstanding thread, no boundary coverage gap, zero qualifying calls from ANY caller, none unidentified, and a baseline positive control. Every certificate states its coverage | r2 #5; r3 #1, #7 |
| `defence.py` | GREEN → GREEN-unobserved → MUTANT → MUTANT-unobserved → RESTORE; per-node states must match between observed and unobserved runs (observer-off control); manifest + venv checked after every subprocess; kills located at the innermost frame inside the worktree (its venv excluded), so mock and helper assertions resolve to the test line; assertion anchors by unique text or frozen line + text | r2 #1–#3, #5; r3 #2 |
| `replay.py` | complete fix set; drift (frozen bytes, untracked, environment) checked right after each subprocess, BEFORE the src swap or reset; the red stage uses the full phase gate (only call failures allowed); `acceptance.json` with retained audit entries on every path | r2 #8; r3 #3 |
| `acceptance.py`, `run_expected_failures.py` | Task 1 pair acceptance with closure freeze, oracle records and connections; failure-path artifact | r2 #3, #4; r3 #4, #8 |
| `deploy_runs.py` | Task 4 | r3 #5 |
| `records.py` + `record_schema.json` | Task 8 | r3 #6 |

Tests: `tests/scripts/incident_register/` carries one regression test per Codex r1/r2/r3 counterexample (r3: the `map` send, the post-window worker, the side-effecting `__repr__`, a boundary rebound to a helper whose start event was disabled, a C-implemented boundary, the in-session source change, the observer-aware node, the harness file rewritten during green, the red-stage teardown error, the literal oracle value, the helper changed inside the pair, the driver failure artifact, the four invalid records, publication binding, continuing vs one-shot contemporaneity). Synthetic venvs now expose only a small pytest site (the outer site-packages is never on their path). **Mutation sweep (strict):**
- A clean baseline runs first.
- A mutant counts as KILLED only when a node fails with an `AssertionError` and none errors. Collection and setup errors are `ERRORED`, not kills.
- No bytecode is written during the sweep, and caches from before it are purged.
- It covers 75 mutants, one per check, including those added in self-review:
  - D11/P7: the unexpected-exception reason;
  - V7/V8: a continuing occurrence's end;
  - R14: no bytecode;
  - R15: src at session start;
  - W4: `.pth` trees hashed.
- Every mutant runs the WHOLE suite (no `-x`), so "no node errored" covers every node and every killing node is listed. pytest.raises' `DID NOT RAISE` counts as an assertion-shaped kill.
- One guard is a known equivalent mutant and is not listed: the `completed and` condition on the defence/replay verdict. It is redundant while every exception path records a reason.
- Results are in `…-evidence/tooling/mutation_sweep.txt`.

## Task 4 — Deployment intervals (DONE; rev 4 wording per Codex r3 #5, verbatim)
The retained logs provide SHA observations and deployment-transition bounds. Matching successive pre/post SHAs establishes endpoint agreement; continuity between observations is an explicit assumption, not proof that no out-of-band change occurred. For each fix, retain the latest supported observation without the fix and earliest supported observation containing it, with unknown endpoints where necessary. Keep checkout, service restart and canary health observations separate. Do not assign a single exact installation timestamp from a post-deploy log line.

Facts: `deploy_runs.py` reads each run's log through fixed output templates only (the echoed script source never matches; no other text is stored). 192 runs: 164 logs expired (HTTP 410), 28 retained (6/07 onward). `observations` lists every pre-deploy / deployed / rolled-back SHA as a time point; `installed_timeline` labels the spans between them `transition` (inside a run), `assumed_continuous` (endpoints agree), `drift` (endpoints disagree) or `unknown` (an expired log in between); `first_live(fix)` returns `(not_live_before, live_by]` from observation points — e.g. the 8/13 fixes: pre-deploy line 19:04:33Z, deployed line 19:04:35Z. Limits, checked on the data rather than coded around:
- `live_by` is the earliest observed containing install, so a deploy later rolled back would count. None occurred: the 28 retained logs show 0 failed canaries and 0 rollbacks.
- Observation points follow run creation order. That matches time order for all 28 retained runs (checked 2026-09-29).

## Task 5 — Historical replay (TO DO, after the tooling review)
**Scope from the consolidated drafts (9/29):**
- 30 fixed Tier-A observed incidents with 62 fixed links. The four plan-named ones are I-082–I-085.
- For 48 of the links, the fix commits touch a test file.
- For 14, the fixes are scripts, units or workflows with no test in the fix. Each of those gets a characterization or config fixture where one is faithful, or `unavailable` with the reason.
Scope: every fixed Tier-A candidate with a fix commit. One spec per fix set (label, `fix_set`, tests, symptom nodes with assertion anchors, audit entries with reasons, deployed ref + basis). The pilot's lessons stand (new-API-only red: 3a6e48b, a364b11, 8bceda1, 4f0257a, 41b2bb1, 0abf503; ce6676d's only symptom node is `test_post_cutoff_rescoring_does_not_flip_a_settled_hit`; 2ff2db9 return shape; 736ea8f import inside the test). A record reads `historical_replay: unavailable (<reason>)` unless the runner verdict is `accepted` and my reviewer decision is `accept`.

## Task 6 — Current defence (TO DO, after the tooling review)
**Prepared, not run:** the nine plan-named specs are in `…-evidence/current_defence/specs/`, statically checked (every anchor resolves uniquely; the paths pass the hard rule). The 30-incident worklist is in `…-evidence/current_defence/worklist.md`.
Per causal link: a spec that IS the contract sheet (contract, production entry + mode, trigger, symptom kind, boundary bindings and identity classification, allowed mocks, certificate level + justification), the smallest semantic mutant restoring the pre-fix decision, the branch anchor in the mutated file, killing nodes with assertion anchors. Run `current_defence`; I read the patch, the killing frames and the linked events and record `accept`/`reject` with a reason. Survivors carry a §9.5 class. Priority: plan-named (8/11 `404358d` ×2; 8/13 `1b50b78`, `224ddce`; 8/30 `ac0ce8d`, `67338cd`/`3697512`, `c0c0a97`, `314154d`; 9/03 `0abf503`), then the other fixed Tier-A links (7/12 `9551818` `ec242da` `230f65c`; C-03 `ce6676d`; 4/15 `1d61908`; GH #144; 6/11–7/08 entry checks; 7/06; 6/17; 5/05; 4/04; 4/12; 6/07; 6/10; 8/09; 4/22–4/23, 5/09, 6/09 restart class; 4/30; 5/21; C-01). The 45 preliminary v1 designs are reused as inputs, never as results. Batches run sequentially (no CPU contention); I re-run ≥ 20 % myself.

## Task 7 — Route H finalization (IN PROGRESS)
- **Commits since 3/29:** 1,172 classified (806 code/config, 366 docs-only).
- **Negatives:** all 372 runtime-closure negatives diff-reviewed (two readers × 186; per commit: changed functions/paths + exclusion reason from the DIFF; `…-evidence/negatives/N{1,2}_review.tsv`). I adjudicated all 169 sensitive-path rows and all 28 `possible_fix_of_live_behaviour` rows: no new incident among the sensitive-path features/pre-ship fixes; new candidates E103–E111 (below). A seeded 10 % QC sample of the remaining non-sensitive negatives confirmed 20 of 20 exclusions: 20 of 196 drawn with `random.Random(20260929)`, recorded in `…-evidence/negatives/qc_sample.md`.
- **Positives:** every `incident_fix` / `incident_fix?` / `latent_fix` commit is verified against its commit text and folded into episodes. The per-commit classification is in `…-evidence/route_h/commit_classification.tsv`, and the episodes in `…-evidence/route_h/episodes.md`.
- New candidates from the negative review: E103 `2be445e` local grading lacked the saver rule 3/29–4/01 · E104 `5207a09` worker tiers ran unpulled code · E105 display defects (`b430e45`, `d3e9337`, `c30138b`, `2ee1763`) · E106 `947fce8` O(n²) dead code slowed worker predictions (with E16) · E107 `681eb8c` Healthchecks ping URL committed to the public repo · E108 `18efce1` heartbeat watchdog wrote RUNNING unconditionally · E109 `30452eb` deploy race (in-flight run could ship a later untested commit) · E110 `e5ef7ca` entry check treated `present_unverified` as confirmed and masked a missing DD slot (7/04–7/09; with E72) · E111 `810d7e0`/`815cf50` MDP saver input from an unsound proxy (6/10–6/18).
- Per Codex r3 answer 7:
  - `947fce8`'s second mechanism is its own record, I-113. The Fly shadow host, restored from R2, lacked `mdp_policy.npz` and silently fell back to the heuristic. Since the design's scope covers restore dependencies on any host (§2), I-113 is disposed as an observed incident of the restore path. Its residual reaches production: `sync_to_r2` still omits `mdp_tail_policy.npz`.
  - E107 keeps the exposed-ping-URL candidate without copying the URL anywhere (checked: no UUID or `hc-ping` URL appears in the draft).
- Record drafting: the lead wrote the plan-named records (I-075, I-077, I-082–I-087, I-112, I-201, I-202). Four readers drafted the rest from per-episode evidence packs under a drafting guide (`…-evidence/route_h/`). Each record was validated in draft mode and then reviewed by the lead.
- **Consolidation (lead, 2026-09-29): 119 drafts → 115 records, 0 draft-mode errors.**
  - Merged: I-044 into I-095; I-096 into I-047.
  - Moved to `drafts/exclusions.json`, with the episode notes E70, E81, EX2 and EX3:
    - I-026 and I-076: model quality;
    - I-208: not a deployed path;
    - I-103 and I-104: retired infrastructure without watchdog or restore relevance.
  - Split:
    - I-113 from I-106;
    - I-114 (link 18, the NaN pick-time `platoon_hr`) and I-115 (link 2, the daily CRITICAL DMs on the biased path) from I-059, because their firings are reported.
  - Re-disposed under ruling 6: I-063 is continuing from 6/11; I-064 and I-074 become observed incidents, since their first-hand reports were written while the condition persisted.
  - All 92 log-basis install bounds were recomputed with `first_live`. The (t, t] values the readers took from the old tool are now `(pre-deploy line, deployed line]`, and the pre-7/02 bound starts at the last observed install (6/07 00:11Z, not the expired-log run at 17:55Z).
  - Counts:
    - observed 61 (A 30, B 27, pending 4);
    - latent 24;
    - near miss 4;
    - unresolved 24;
    - pre-ship 2.
  - Thirty fixed Tier-A observed incidents need replay and defence entries (Tasks 5–6).

## Task 8 — Records and Phase 1 memo (schema DONE; 115 records drafted and consolidated, draft-mode valid; fixture fields, publication validation and memo TO DO)
`record_schema.json` implements design §7 (dispositions, classes, tiers, axes, contract, numbered mechanism links with code refs + ref basis, per-link fix states implemented/deployed/mitigated/verified as separate fields, fixtures with explicit `unavailable` reasons and reviewer decisions, residuals, watchdog, evidence kind + strength + locator + exposure row, bounded occurrences and latencies). `records.validate` adds the cross-field rules (observed incidents need a primary machine observation or a contemporaneous operator report written ≤ 48 h after the onset; unresolved candidates name their missing evidence; `counted` false exactly for pre-ship exclusions; fixed Tier-A incidents need a replay entry and a defence entry per fixed link; plan-named records need a fixture or a deferral; unfixed §10 contracts need an expected failure; B → A when a residual reaches production; tier_pending needs a reason; ordered bounds; numeric latencies need bounded endpoints). It runs under `uv run --with jsonschema==4.23.0` (jsonschema is deliberately not added to the lock the box syncs). A record failing validation is not published. The memo sections follow design §11; each record's watchdog field feeds W4 rank 2.
- **Publication gate (rev 4, Codex r3 #6):** `records.validate(..., evidence_root=<repo>)` binds every certified replay/defence entry to its acceptance artifact (file present, sha256 equal, runner verdict `accepted`, nodes / patch / fix set equal) and requires the reviewer's `accept`; a deployed state needs its SHA, time and evidence per basis; a numeric latency needs both endpoint times bounded and may not be tighter than those bounds allow; a contemporaneous report must be cited by the occurrence or fix step it describes and meet ruling 6. Draft mode skips only the fixture-completeness rules.

## Task 9 — Reviews and merge
1. Codex phase-1 r3: this plan + the branch (tooling only; repo only).
2. After Tasks 5–8: Codex result review of the memo and evidence.
3. Fast-forward merge to main; wrap index W1.5 row, corrections C-03 cross-reference, memory.

Nothing is deployed.

## How rev 3 answers Codex phase-1 r2
| r2 | Answer | Regression tests |
|---|---|---|
| #1 destructive ops not confined | `owned`: ownership record + exact-root/linked/detached checks before any reset/clean/swap/edit/removal; edit paths validated (normalized, contained, no symlink component, tracked `src/bts/**.py`) before the first write; evidence output refused inside the worktree | `test_owned.py` (primary checkout, subdirectory, symlinked root, unowned, attached, traversal, absolute, symlinked component, tracked symlink into tests, validate-before-write), `test_defence.py::test_traversal_edit_is_refused_before_writing`, `::test_primary_checkout_is_never_used`, `test_runner.py::test_output_inside_the_worktree_is_refused` |
| #2 allowlist authorizes oracle mutants | hard rule independent of the spec; working-bytes manifest of every other tracked file + untracked list + venv fingerprint after every subprocess; no source instrumentation at all (external observer) | `test_owned.py::test_mutation_paths_outside_the_hard_rule_are_refused`, `test_defence.py::test_expected_value_mutation_is_refused_even_when_allowed`, `::test_a_test_that_writes_a_tracked_file_is_rejected` |
| #3 run-wide errors ignored | one `gate` for every stage (session identity, finish + exit status, collection/observer errors, exact inventory, all phases, return code, imports); killing node must be a clean baseline pass and a member of the kills; no XFAIL allowed in defence runs; rejected acceptance objects written | `test_runner.py` (collection error, teardown error, imperative XFAIL, inventory, unfinished session, observer error, unexplained return code), `test_defence.py::test_declared_killing_node_that_survives_is_rejected`, `::test_collection_error_in_the_baseline_is_rejected`, `::test_malformed_spec_still_writes_a_rejected_acceptance` |
| #4 oracle frame suffix / no production call | exact realpath + function; exception `module.qualname`; registry-bound bad/required values in the message; production entry invoked in the observed call phase; marked/`--runxfail` pair with identical bytes | `test_expected_failure.py` (foreign oracle module, same-named exception elsewhere, literal oracle input, message without values, imperative, non-strict, teardown, changed bytes) |
| #5 absence not bounded; async | recorder of every production→boundary call over the complete observed call phase; zero qualifying calls anywhere + no unidentified calls + baseline positive control; killing failure at the declared assertion; async detected and rejected | `test_defence.py::test_absence_with_a_real_send_elsewhere_is_rejected`, `::test_absence_without_a_positive_control_is_rejected`, `::test_killing_failure_at_an_undeclared_assertion_is_rejected`, `test_runner.py::test_async_path_is_unavailable`, `test_certify.py` |
| #6 observer shadowing / own venv | observer loaded by absolute path under a random module name and passed as a module object; its own file + hash verified; prefix + executable = the worktree's venv; pytest file not from the worktree; `bts` modules under `<wt>/src` with unchanged bytes | `test_runner.py` (identity tampering ×3, planted `pytest.py`, foreign `bts` package, pytest-from-worktree record) |
| #7 E77 success branch impossible | `_e77_verdict` applies the bad-path assertions only on the bad outcome; truthful schedule; canned cascade | `test_e77_verdict_*` (fixed direction passes; bad raises; late / no-check / other-mechanism fail ordinarily); no-op probe fails ordinarily |
| #8 replay acceptance object | `replay.historical_replay` writes `acceptance.json` with verdict, retained audit entries + reasons, symptom checks at declared assertions, session gates, closure + deployed ref | `test_replay.py` |
| #9 plan + nested roots | this rewrite; synthetic repos carry their own `pytest.ini`; the runner passes `--rootdir` and `-c`; the gate checks rootdir | `test_runner.py::test_clean_run_passes_the_gate_and_records_identity` |

## How rev 4 answers Codex phase-1 r3
| r3 | Answer | Regression tests |
|---|---|---|
| #1 absence incomplete (C callback, post-window worker) | callee-side boundary recorder (mock `__call__` / function code: C-invoked and threaded calls seen); entry exits recorded and absence requires the branch's invocation to complete; threads alive at the end of the call phase make absence unavailable; C-implemented or rebound-unseen boundaries are coverage gaps | `test_runner.py::test_a_send_made_from_c_is_recorded`, `::test_a_worker_alive_at_the_end_of_the_call_phase_is_counted`, `::test_a_c_implemented_boundary_is_a_coverage_gap`; `test_defence.py::test_absence_is_not_certified_over_a_send_made_from_c`, `::test_absence_is_not_certified_while_a_worker_is_outstanding`; `test_certify.py` (completion, outstanding, gap, any-caller) |
| #2 observer ran application code | `_safe` serialization only (no repr/str/properties); observer-off conformance stages in defence | `test_runner.py::test_observing_a_return_runs_no_application_code`; `test_defence.py::test_observer_dependent_behaviour_is_rejected_by_the_unobserved_control` |
| #3 replay harness drift + red teardown error | drift checked after each subprocess before any swap/reset; red stage = full phase gate | `test_replay.py::test_a_harness_file_changed_by_the_green_run_is_rejected`, `::test_an_unrelated_red_teardown_error_is_rejected` |
| #4 pair closure + disconnected literal + environment | `run_pair` freezes tracked bytes + untracked + environment content across the pair; oracle records + connections (`return` / `reads` / `derived`); venv content fingerprint; src digest at session start/end | `test_expected_failure.py::test_a_literal_oracle_value_disconnected_from_production_is_rejected`, `::test_a_helper_changed_inside_the_pair_is_rejected`, `::test_derived_connection_needs_a_recorded_review`; `test_runner.py::test_a_source_change_during_the_session_is_rejected` |
| #5 deploy bounds | Task 4 rewritten verbatim; observation points; `(not_live_before, live_by]` from log lines | `test_deploy_runs.py::test_first_live_is_bounded_by_the_runs_own_log_lines`, `::test_timeline_separates_observations_transitions_and_assumptions` |
| #6 record validation | publication binding, per-basis deployed fields, latency ordering/endpoint consistency, contemporaneity (ruling 6) | `test_records.py::test_codex_r3_invalid_records_are_rejected`, `::test_publication_binds_certified_entries_to_their_acceptance`, `::test_latency_tighter_than_its_bounds_is_rejected`, `::test_a_report_written_during_a_continuing_condition_is_contemporaneous`, `::test_an_uncited_operator_report_is_rejected`, `::test_a_report_may_describe_the_fix_step_that_cites_it`, `::test_an_operator_report_can_date_an_install` |
| #7 rebinding / recursion | `restart_events` when a new boundary callee is registered; common-live-invocation linking | `test_runner.py::test_a_boundary_rebound_to_a_disabled_helper_is_still_recorded`; `test_certify.py::test_recursion_links_through_the_common_outer_invocation`, `::test_exit_then_frame_reuse_splits_invocations` |
| #8 driver failure path | rejected artifact first, `finally` restore, ref + registry hash + one verdict | `test_expected_failure.py::test_driver_writes_a_rejected_artifact_on_failure` |
| sweep | strict classification (assertion kills only; collection/setup errors are ERRORED) | `…-evidence/tooling/mutation_sweep.{py,txt}` |
| ruling 2 | design v3.2 §9.3 = Codex's text verbatim | — |

## Found in my own review before r4 (each pinned by a test and a sweep mutant)
| Finding | Answer | Regression tests |
|---|---|---|
| `current_defence` / `historical_replay` computed the verdict in `finally` as "accepted when no reason was recorded": an exception type outside the handled tuple, raised before the first reason, left `acceptance.json` saying `accepted` | the verdict also needs the stages to have run to the end; any other exception is recorded as `aborted` and re-raised (D11, P7) | `test_defence.py::test_an_unexpected_exception_never_writes_an_accepted_artifact`, `test_replay.py::test_an_unexpected_exception_never_writes_an_accepted_artifact` |
| a continuing occurrence ended at the LATEST fix install anywhere in the record, so a report written long after a mitigation still counted | earliest end event; occurrences may name their `links` (V7, V8) | `test_records.py::test_a_continuing_condition_ends_at_its_earliest_end_event`, `::test_an_occurrence_ends_with_the_fixes_of_its_own_links` |
| stale bytecode: a same-size edit within the same second as the last compile runs the old `.pyc` (hit in a scratch probe) | evidence runs already set `PYTHONDONTWRITEBYTECODE=1` after a clean; now pinned; the sweep writes none either (R14) | `test_runner.py::test_a_same_size_edit_with_the_same_mtime_is_what_runs` |
| the "src differs at session start" gate check had no test | pinned (R15) | `test_runner.py::test_a_source_change_before_the_session_starts_is_rejected` |
| the first whole-suite sweep (`321f401`) survived O2, D9, D11, P7 and W3, and counted D8 and W2 as FAILED-OTHER | tests added (`27b260d`); D11/P7 retargeted to the reason the test pins (the completion guard is an equivalent mutant); the classifier counts `DID NOT RAISE`. Each new test fails with its mutant applied | `test_runner.py::test_a_boundary_outside_production_stays_enabled_after_its_first_call`, `test_defence.py::test_observer_dependence_visible_only_under_the_mutant_is_rejected`, `test_owned.py::test_venv_fingerprint_hashes_file_contents_not_just_names`, `::test_venv_fingerprint_hashes_pth_trees_outside_the_worktree` |

