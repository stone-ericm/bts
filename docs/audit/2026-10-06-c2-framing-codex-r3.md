## Verdict
**BLOCK.**
Reviewed-commit: da87dad2a36f07270282c15e74c82dcd15d49694

The original r2 counterexamples are corrected, and the permitted suite passes **100/100**. The metadata-HEAD lifecycle now aggregates, the empty completion and `Ericsson` source refuse, opaque claims and consistently wrong manifests refuse, coherent diff/summary changes with unchanged scorecards refuse, and a mixed failure/setup-error mutant run is INCONCLUSIVE with runner exit 1.

The new semantic validator nevertheless has three material false-green paths: decision-bearing secondary metrics can be changed without changing the profiles; a seed lacking 2025 profile evidence qualifies as complete for releasing seeds 2–3; and a code-changing HEAD can qualify under a supplied admitted identity without verifying that it is a permitted metadata descendant. The runner also still certifies RED without proving that every named test executed.

This is Eric's authorized **third and final round**. The item stops; no fourth round is authorized. This verdict supplies no run admission, launch, acquisition, repair, promotion or production approval.

**Scope:** The complete r3 prompt was read first. HEAD already matched the requested commit and the tracked tree was clean. No outside context corpus, memory registry, other project, real evaluation data, configuration, credentials, box or network was consulted. There was no escalation or tracked edit this round. All independent reproduction artifacts were confined to the owned `/private/tmp/c2-framing-r3-074c_eg8` prefix, which is deleted at completion.

## Findings

### R3-1 — Profile reconciliation omits the secondary scorecard values that can change the disposition

**New validator:** `scripts/audit/c2_framing/screen.py:531–547`. **Declared decision weight:** registration §5; unchanged `src/bts/experiment/runner.py:173–201`.

`validate_run` recomputes only per-season P@1 from the retained profiles. It treats the scorecards' `p_57_exact` and `streak_metrics.mean_max_streak` as authoritative, then recomputes diffs from those unchecked card values. The neutral fallback uses these values to decide a seed's pass, so reconciling only P@1 does not reconcile the decision evidence.

**Measured on artifacts produced by three complete synthetic `run` calls:** B initially had P@1 deltas `(0.0,0.0)` for every seed, zero seed passes, and disposition **negative**. I changed only B's scorecard `p_57_exact` to baseline +0.001 for the first two seeds, then regenerated their diffs and stored summaries using the unchanged production helpers. The baseline/B mean-streak deltas were already zero. All 18 retained profile files stayed byte-identical; their SHA-256 map was checked before and after.

The actual aggregate accepted the modified evidence, counted two seed passes, and changed B to **inconclusive**. The seed-1 validator also accepted with the supplied admitted identity and pins. Direct full-scorecard recomputation from that seed's unchanged B profiles gave:

```text
stored p_57_exact:     0.0010000000000000074
recomputed p_57_exact: 7.476707121345214e-18
profile bytes changed: no
B: negative / 0 passes -> inconclusive / 2 passes
```

This is the new validator's consumption boundary, not a claim that the ordinary producer emitted the altered card. The altered exact probability is demonstrably inconsistent with its retained profiles. Under §5, only an inconclusive variant could justify requesting stage two, so this false green changes the owner's decision category even with all primary deltas fixed at zero. It does not itself authorize stage two.

The original r2 diff-and-summary-only exploit now correctly refuses. Moving the coherent edit one join upstream, into a decision-bearing scorecard field, still succeeds. All decision-bearing metrics must be reconciled with their retained input evidence, including the declared deterministic scoring/simulation settings.

### R3-2 — Six unit names and six filenames can release a seed with no 2025 profile evidence

**New validator/release:** `screen.py:318–339,528–537`. **Registration:** §4 seed order and §5 complete-run validation.

The manifest must list both test seasons and the results must list six `(variant, season)` pairs. But profile files are concatenated before checking their contents. The validator neither checks that each file carries its named season nor requires the reconciled P@1 keys to be exactly `2024` and `2025`.

**Measured:** Starting from the complete synthetic artifacts, I replaced each `profiles_<variant>_2025.parquet` with that variant's retained 2024 profile. Its rows still said season 2024 and had 2024 dates. I reconciled the scorecards/results P@1 dictionaries to the surviving `{"2024": ...}` and regenerated the diffs/summaries. The manifest still declared `[2024,2025]`, and the results still listed all six expected unit names.

The actual seed-order helper returned:

```text
seed_allowed(seed 2, valid Eric release naming this seed 1):
(True, "released", 30.0)
```

The aggregate also accepted all three runs, returning `disposition="incomplete", n_seeds=3` for A and B. That final disposition prevents a positive season conclusion, but it does not repair the earlier **completion false green**: the same validator has already qualified seed 1 to release later computation. `launch` and child `run` both use this helper. Their current admission gate validates current code/inputs; it does not supply the absent completed 2025 evidence.

The empty `{}` result file, wrong release name and different supplied admitted identity now refuse. Those are useful improvements. Semantic completion still requires checking the retained units' actual season evidence, rather than counting matching metadata and filenames. If a named season has no scoreable rows, it must be explicitly incomplete for this gate; it cannot silently qualify as the required complete seed.

### R3-3 — HEAD equality was removed without validating the permitted metadata-descendant relation

**New code:** `screen.py:495–523,567–572`. **Relevant unchanged admission rule:** `scripts/audit/c1/admission.py:146–195`.

The equal-HEAD false refusal is fixed. However, the replacement checks only that each HEAD is a 40-hex string, equals its claim/result code, and accompanies matching declared identity fields. It never checks that the run HEAD contains the exposure/reviewed commits or that its executable closure matches the reviewed commit. Aggregate supplies no trusted expected identity or pins to `validate_run`; its “accepted identity” check accepts any nonempty strings in the five fields.

**Measured with the supplied admitted identity and pins:** I coherently changed seed 1's claim, manifest and results HEAD to the existing r2 commit `02516cfedc812f047295b6f2ab212fd72a944507`, rebound its claim hash, and left the declared admitted identity/pins intact. The actual validator accepted, the release helper returned `(True,"released",30.0)`, and the aggregate accepted mixed HEADs including this commit. A read-only closure diff against the reviewed revision showed changes in `screen.py` and the registration: this is not a metadata-only difference.

I repeated validation and release in a separate owned copy using a producer-format run name `02516cf-20261007T014552Z` and a properly shaped **synthetic** expected identity whose `reviewed_commit` was this r3 target, with full commit/hash fields. Both accepted. `git merge-base --is-ancestor <r3 target> <r2 commit>` returned false. Thus the counterexample does not depend on a run-name prefix inconsistency or merely on the author's symbolic test identity strings. It is not a real accepted review or exposure, and no real admission record was created.

Separately, replacing all three manifests' five identity values with the literal string `bogus` still produced an aggregate. That check establishes field presence and agreement, not an accepted review, published exposure or authentic admission binding.

An ordinary `run` still invokes the shared admission gate before producing evidence; I found no regression in that producer gate. The new consumer and seed-1 completion witness cannot infer a past run's executable identity from those unchecked declarations. R2 required allowing **verified metadata descendants**, not arbitrary matching identity labels beside arbitrary code hashes. Retain each actual HEAD, validate its admitted ancestry/closure, and anchor identity/pins to trusted admission evidence. Do not restore the original whole-HEAD equality refusal.

### R3-4 — Mixed errors now refuse, but fail-fast still makes an unexecuted named test RED

**Revised runner:** `docs/audit/2026-10-06-c2-framing-evidence/mutant_runner.py:12–25,35–53`; classifier test `test_screen.py:749–758`.

The original mixed assertion/setup-error reproduction is now **INCONCLUSIVE(exit 1, errors)**. Exercising `main` with the real captured pytest result printed the FAILED and ERROR witnesses, printed `NOT RED: X1`, returned 1, and restored the scratch target. R2's concrete mixed-error defect is fixed.

The broader requirement that all named tests execute remains unmet. The classifier reads FAILED/ERROR lines and the final summary; it has no expected collection/execution inventory. The runner inherits `PYTEST_ADDOPTS` without constraining fail-fast or selection.

**Measured through the actual runner `main`:** A scratch mutant changed `target.value` from 1 to 2. Its spec named two test nodes. The first asserted the original value; the second wrote a body-entry canary before its assertion. With inherited `PYTEST_ADDOPTS=-x`, the actual pytest subprocess stopped after the first failure. The second body never ran. The reviewed runner nevertheless printed:

```text
X1 RED 1 failed in 0.01s
NOT RED: none
runner return: 0
second named test body executed: false
target restored: true
```

The runner's pytest argv and environment were used; only its initial `uv run python` prefix was replaced with the checkout's venv Python to keep the scratch reproduction independent of package resolution. This is not evidence that the author's recorded ledger used fail-fast. It demonstrates that the new runner still cannot support its unconditional “Every named test runs” claim. The new classifier test also intentionally calls a summary with deselected tests RED, without binding those deselections to the intended selection.

Require an expected selected-test inventory and complete outcomes, or explicitly constrain and validate selection/early-stop options together with structured execution evidence. A clean failure is a kill of its failing test; it does not by itself establish execution of the other named tests.

### Disposition of the four r2 required changes

| R2 requirement | Repeated reproduction on r3 | Disposition |
| --- | --- | --- |
| R2-1: metadata-HEAD lifecycle and executable identity | Three complete synthetic runs at HEADs `a…a, c…c, c…c` now aggregate as A positive/B negative under one identity and pin set. The shared metadata-descendant test passes. | Original false refusal fixed; verification of the allowed HEAD relation remains open in R3-3. |
| R2-2: semantic seed-1 completion, exact source, bound release and positive finite budget | `{}` completion refuses for missing claim; `Ericsson` refuses; a genuine source plus wrong run name or changed supplied identity refuses; zero and a decimal overflowing to infinity refuse. Valid release gives 30; the recording launch executor receives `--cpu-hours 30`, seed-2 name and deterministic environment. | Original reproductions fixed; completion still admits absent 2025 evidence (R3-2) and a wrong executable HEAD (R3-3). |
| R2-3: validate claims, expected manifests, artifacts and joins | Rebound opaque claim refuses as invalid JSON; three consistently wrong `actual_pa`/feature-settings/determinism manifests refuse at their specific checks; missing identity/pins/unit/diff/profile and alternate namespace cases refuse in the suite. The coherent +0.02 B diff/summary edit with unchanged cards now refuses at the card/diff join. | Substantial repair; decision-bearing card/profile join and admitted identity/HEAD anchor remain open (R3-1/R3-3). |
| R2-4: clean, complete mutant failures | Actual mixed pytest failure/setup error is INCONCLUSIVE, runner return 1, target restored. Clean failures, skips, no-tests and non-1 exits have classifier tests. | Mixed-error defect fixed; complete named-test execution is still unproved and bypassed by fail-fast (R3-4). |

### No regression of the r2 closed items

I compared ASTs against r2 for 19 relevant functions: framing/self-check/coverage, resumed counts, frozen input helpers, settings, labels, bases/blends, summaries/dispositions, first-unit stop, pinned loading, launch command and admission gate. All were unchanged. Their tests passed, and the following independent checks also ran:

- **Closed inputs:** Real `compute_all_features` on 48 synthetic PA rows, 24 historical games, a live cache with an extra game, and a 2026 raw canary. Freeze kept exactly the 24 pinned-game entries. After changing cache/raw canaries and installing the closed inputs, any `Path.read_text` would fail; computation succeeded, made zero table reads, and gave exactly equal values for all 16 FEATURE_COLS. Both production/frozen frames had 28 non-null bullpen values. Park drag was all NaN and pitcher self-check was identical on 48 rows.
- **Labels/voids:** Original miss plus resumed-only hit remained label 0; a resumed-only batter had no label and was dropped. Counts were `changed=1, void_dropped=1`, ranks `[1,2]`, hits `[0,1]`. The declared retrospective drop/re-rank estimand remains unchanged; it is not a replay of replacement contest picks.
- **Official-date limit:** The future-resumption history counterexample still changes the June 6 catcher value `0.0 -> 0.1`. Registration §8 continues to disclose it. This is the explicitly bounded convention accepted as a wording resolution in r2, not a new unconditional leak-free claim or a new blocker here.
- **Effective settings and model constructor:** `(20,7)` passes, `(0,10)` refuses. The actual training helper, with a recording LightGBM constructor and no real fit, used each of the three registered seeds and both deterministic/row-wise flags. A synthetic three-tuple blend retained its extra column and parameter dictionary identity.
- **Namespace/cardinality:** The suite rejects `--out-root`, same-seed repeat claims, one/two/four directories, duplicates, seed 42, and copies outside the canonical namespace. The private `_test_out_root` seam remains confined to synthetic use here.
- **Budgets:** `gate(60.48,45,True)=over_cap`; hypothetical owner-released 30-hour budgets give `gate(30.48,30,False)=ok`, `gate(60.48,30,False)=checkpoint`, and `gate(60.48,30,True)=ok`. The wrapper uses the released budget. No actual cost, checkpoint acknowledgement or owner release was measured or issued.

The author's box facts remain unverified assertions: historical lookup provenance depends on the stated raw-directory inventory, and production settings depend on the stated environment defaults. This review proves synthetic behavior/source restrictions, not actual box coverage, hash inventory, real-input feature parity or run readiness. No new feature/formula/label/setting/budget regression was found.

### Test and mutant strength

The permitted suite is **75 framing tests + 25 shared admission tests = 100 passed in 65.87s**. The lifecycle and coherent-evidence damage cases materially improve the earlier suite. Their fixture still stubs admission, PA loading, full feature computation and walk-forward; its scorecards use 200 Monte Carlo trials. It does not create a real committed release or prove actual run admission. The shared suite separately exercises real disposable-repository admission behavior. My feature and constructor probes independently exercised those respective production functions on synthetic inputs.

I verified all **67 mutant specifications**, each with exactly one current source match, and every latest revision-3 output status: **66 RED, G8 SURVIVED, none missing**. The first r3 run records 58 RED, five INCONCLUSIVE and four SURVIVED; the resume supplies eight new RED outcomes and retains G8 as equivalent. The logged runner exit remains 1 for G8; the README's “66 of 66 attributable RED” is an attribution statement, not a clean overall runner exit.

G8's equivalence argument is sound under the current checks: each ten-pin dict must match its digest, and the aggregate still compares those digests, so removing the extra pin-dict comparison cannot admit different pins absent a SHA-256 collision. No production `screen.py` line or runner line changed after `fe5eb85`; subsequent changes are tests, specifications and evidence. I did not run the tracked-source mutator.

The ledger establishes these listed assertion kills, not the missing boundaries:

- H8 distinguishes a changed declared identity; it never checks a wrong executable HEAD under the same supplied identity.
- H9 removes the unit-name/order check; it does not require a file's actual season to match its unit or require both seasons in the reconciled metrics.
- H10 checks P@1 reconciliation, H11 checks card-to-diff reconciliation; neither validates the secondary decision metrics against profiles.
- H5 checks missing identity fields; neither it nor H12 requires those fields to identify an accepted review/exposure/admission.
- H13 kills mixed errors/skips, but does not prove every intended node ran or reject fail-fast truncation.

These are consumption/lifecycle tests missing from the new rules, not a rejection of the already corrected formula tests or an assertion that the recorded RED failures were unexecuted.

## Required changes

No tracked repair was made. These describe the unresolved requirements; they do not authorize reopening, another review round or a run. Eric's ruling stops the item after this non-SIGN result.

1. **Reconcile every decision-bearing metric.** Bind exact P(57), mean maximum streak and their scoring/simulation settings to the retained profiles, and validate those values before deriving seed passes. Add the unchanged-profile, secondary-card-edit counterexample: it must refuse rather than turn negative into inconclusive. Preserve the corrected card/diff and primary P@1 joins.
2. **Require both completed test seasons.** Validate each profile against its named unit/season and its declared date/season convention; require exactly the registered test-season metric keys and adequate completed evidence before seed 1 can release later seeds. An absent/empty/unreadable/wrong-season unit must be incomplete for the release gate. Add the duplicated-2024-as-2025 case at `validate_run`, `launch` and child `run`.
3. **Validate the admitted code identity rather than declarations.** Anchor the accepted review/exposure/admission/pins to trusted evidence; check each retained run HEAD's admitted ancestry and executable closure. Reject arbitrary identity strings and a code-changing/non-descendant HEAD even when claim, manifest and results agree. Keep actual per-run HEADs and the now-correct metadata-release lifecycle; do not reinstate whole-HEAD equality.
4. **Bind mutant RED to complete intended execution.** Retain structured selected-node/outcome evidence and require all intended tests to complete. Errors, skips, interruptions, selection changes or fail-fast must not leave an intended node unexecuted; legitimate exclusion of nodes outside that intended set is separate. Constrain inherited selection/early-stop options as necessary. Add the two-node `PYTEST_ADDOPTS=-x` runner test; it must be INCONCLUSIVE/nonzero when the second body does not run. Keep the corrected mixed-error refusal and verified target restoration.

## What was run

- Read `.codex-review/c2-framing/prompt-r3.md` first, then the in-checkout r2 report, specified change diff and `fe5eb85` mapping, full revised registration/driver/tests/runner, ledger README/spec/output, and relevant unchanged feature, scoring, training, admission and launcher/ledger sources. No outside context corpus was used.
- Confirmed `git rev-parse HEAD` equals the requested revision and a clean tracked tree. `git diff --check 02516cf da87dad` passed. No checkout, escalation, review-driven commit, push or source mutation was performed.
- Ran the prescribed command with scratch `MPLCONFIGDIR`: `UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run python -B -m pytest tests/scripts/c2_framing tests/scripts/test_c1_admission.py -p no:cacheprovider` — **100 passed in 65.87s**.
- Ran owned `probes.py` and `followups.py` with the checkout's venv Python, `-B`, `PYTHONPATH=.`, `BTS_LGBM_DETERMINISTIC=1`, `TZ=America/New_York`, and scratch `MPLCONFIGDIR`. Three real calls of the research `run` used fake admission/loading/features/walk-forwards and real claim/manifest/output/label/scorecard/diff/result code; scorecards used 200 trials. Additional full-scorecard recomputation of the unchanged synthetic B profiles used the default scorer. No walk-forward or real-data evaluation occurred.
- Exercised actual `compute_all_features` on wholly synthetic PA/cache/raw inputs and the actual training helper with a recording, no-fit constructor. The first feature-comparison harness attempt lacked the synthetic `weather_temp` column and stopped with a KeyError; after adding it, the full comparison and subsequent probes passed. That harness error is not attributed to the reviewed code.
- Captured actual scratch pytest mixed failure/setup-error and fail-fast outputs; exercised the actual revised runner `main` with the captured mixed result and with a real two-node fail-fast subprocess on a scratch mutation target. Targets were restored. No tracked-source mutator was executed.
- Verified the 19 unchanged function ASTs, all 67 current anchors/latest ledger statuses, and unchanged production/runner code after `fe5eb85`. Reviewed SHA-256 values: driver `a17d03e209cc3d8c3ca2e6c32591b7669fc34ae3f78da7b4923b5cb1ff4ac7e7`; registration `0e086fa52e02ad62415e4b9dfea055fe46fbffb02e07d1427878f8f444ba4a21`; runner `277d38b9b71a832190d0c6e3af4ef88dc0bf3691a2623ad29136cdbb26b290de`.
- The permitted canonical seed JSON was the only real `data/` input read through the prescribed suite. No real raw/PA/cache/external-table data, `.env`, credentials, production configuration, box, network, SSH, `gh`, live provider or operational state was read. No approval or external action was requested.
- Owned scratch is deleted at completion. Required report structure, unique reviewed-commit line, final hash/receipt binding, final HEAD and clean tracked tree are verified before the completion marker. The report is not edited after its final hash.
