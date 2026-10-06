## Verdict
**BLOCK.**
Reviewed-commit: 02516cfedc812f047295b6f2ab212fd72a944507

**Scope disclosure, before findings:** I read the complete round-two prompt before other work and did not search outside-checkout memories, registries, shared corpora, or other project content this round. The prescribed checkout initially failed because this worktree's Git index is under `/Users/eric/projects/bts/.git/worktrees/bts-c2-framing-review/`, outside the writable sandbox. The runtime required an approval retry, which succeeded. That escalation is an exception to the prompt's no-escalation constraint; it was limited to the requested checkout. No review-driven tracked edit, commit, push, or operational action followed. Dependency caches for my own reproductions were directed inside the owned scratch prefix.

HEAD is the requested revision and the tracked tree was clean after checkout. The permitted suite passes **80/80**, and most round-one counterexamples now refuse or are explicitly bounded. However, the revision introduces a false refusal when the later-seed ruling is committed, and its new completion/release and aggregate validators accept inadequate or contradictory evidence. The mutant runner also still accepts a failed test mixed with an unexecuted test's setup error.

This is **round 2 of 2, the last**. The item returns to the owner. This report supplies no run admission, and does not authorize further repair, another review round, acquisition, launch, or production action.

## Findings

### R2-1 — Whole-HEAD equality rejects an admitted metadata-only release between seeds

**New code:** `scripts/audit/c2_framing/screen.py:359,390,426,467–468,481–483`; revision-2 registration §4 and §5, lines 78–82,126–133. **Relevant unchanged gate:** `scripts/audit/c1/admission.py:146–195`.

The registered sequence runs seed 1, reports its cost, then obtains and records Eric's release of seeds 2–3. Publishing that register ruling in a new commit changes HEAD without changing the executable closure, reviewed commit, accepted review/exposure identity, input pins, or experiment settings. Shared admission deliberately permits such metadata-only descendants: it returns the current HEAD while checking executable equality against `reviewed_commit`.

The new aggregate nevertheless requires the **entire HEAD** to be identical across seeds. Seed 1's claim/manifest/results carry the pre-release HEAD; seeds 2–3 carry the later metadata HEAD. Their individual runs can pass admission, but the aggregate refuses with `runs disagree on head`.

**Measured synthetic sequence:** I called the real `run` three times with the author's synthetic input/feature/walk-forward seams and retained real claims, manifests, scorecards, diffs, and results. Admission returned HEAD `a…a` for seed 1, then HEAD `b…b` after the synthetic release publication for seeds 2–3, with the same accepted identity and ten input pins. All three runs returned `0`; aggregate raised the above error. This modeled a metadata HEAD transition; it did not create commits in the review checkout or run real data. The permitted shared-gate test separately passes metadata-only descendants, and source inspection establishes the same behavior for the framing closure, which excludes the register.

The registration's equal-HEAD requirement matches the implementation, but conflicts with its intended admitted metadata workflow. It can be worked around by keeping the later ruling only in an uncommitted working-tree register; that weaker workflow is not specified and should not be the implicit requirement for completing stage one. Do not rewrite seed 1's retained evidence to manufacture equal HEADs. Compare the frozen executable/review/input identity, while retaining and validating each run's actual metadata HEAD.

### R2-2 — The new seed-order check accepts an empty completion file and a non-owner source prefix

**New code:** `screen.py:301–335,363–366,502–511`; tests `test_screen.py:282–312`.

`completed_run` means only “one directory with a file named `results.json`, and no `STOPPED.json`.” It does not parse that file or require a claim, manifest, correct seed, accepted identity, input pins, six completed units, or retained scoring evidence. The fixture `_complete_seed1` expressly creates only `{}` as its completion witness; the test treats that as sufficient.

The source check is also a prefix check: `cells[3].strip().startswith("Eric")`. It accepts `Ericsson, manager; no owner ruling` as Eric. This is weaker than the unchanged shared admission helper's exact first-token check, `cell.split()[:1] == ["Eric"]`.

**Measured independently:**

```text
seed_2273360/old-run/results.json contains only {}
No CLAIM.json, manifest, profiles, units or scorecards exist there.
```

With the correctly spelled positive ruling cell, the actual `seed_allowed` returned `(True, "released", 30.0)` for seed 2. Replacing only the source cell with `Ericsson, manager; no owner ruling` still returned that tuple; `release_budget` returned `30.0`. The actual `launch` wrapper then submitted the seed-2 C1 command to a recording executor and returned `0`. Admission was stubbed to isolate these new checks; the recording executor launched nothing.

Thus both checks must be corrected: the empty completion bypass works even with a genuine release source; the source-prefix bypass is independently insufficient owner attribution. The absence of a release correctly refuses, but that does not prove the present release/completion witness is valid. The wrapper and child `run` call the same weak helper, so the child check does not repair it.

Completion should be bound to the admitted seed-1 run and its retained evidence, not the existence of a filename. The release must use the exact owner token and a finite positive budget, and its applicability should be bound to that seed-1/admission identity. Current zero/infinite budgets eventually refuse in the C1 launcher; that downstream budget refusal does not repair these witness defects.

### R2-3 — Aggregate checks agreement without establishing a valid completed experiment

**New code:** `screen.py:445–488`; registration §5, lines 126–133.

The count, distinct-path, seed-set, stop, claim-byte-hash, and summary-versus-diff checks are useful. But the validator:

- never parses the claim to bind its run name and code to the manifest;
- requires only matching manifest values across runs, not the registered values or complete accepted identities/pin inventory;
- does not validate the manifest schema, self-check, test seasons, or completion of the six registered units;
- does not read the retained scorecards or profile artifacts, or verify the diffs against the scorecards;
- accepts a correctly named `seed_<seed>` parent anywhere, without identifying a canonical claim namespace or an authenticated portable copy of it.

**Measured false green:** Three distinct scratch directories with the correct seed IDs passed aggregate and produced **positive for both variants**, although their claims contained the opaque bytes `not even JSON`; their manifests all stated `basis="actual_pa"`, `deterministic=false`, and `ROOKIE_GATE_K=0`; accepted identity and input pins were absent; and no units, scorecards, or profiles existed. Hashing the opaque claim bytes and making the malformed manifests agree was sufficient.

**Measured contradiction using retained scorecards:** On artifacts derived from the three complete synthetic runs, B initially aggregated as **negative** with both retained scorecard deltas `0.0`. I changed only each `diff_B.json` and the matching stored B summary to P@1 deltas `(+0.02,+0.02)`. Claims and manifest bindings were made internally coherent for this separate probe; the retained scorecards, profiles, and `results.json.p_at_1_by_season` were unchanged. Aggregate returned **positive**. Its own retained per-season P@1 still equaled baseline, and recomputing the diff from the retained scorecards still gave `(0.0,0.0)`.

These are synthetic consumption-boundary probes, not claims that an untouched ordinary CLI run would emit `actual_pa` or wrong settings: the revised producer correctly refuses those effective settings. The defect is that the new consumer certifies a stage-one disposition despite evidence that could not establish such a run, or evidence that directly contradicts its claimed deltas. Recomputing a summary from an unchecked copied diff is only one join in the chain.

### R2-4 — The revised mutant runner still calls a mixed failure/setup-error run RED

**New code:** `docs/audit/2026-10-06-c2-framing-evidence/mutant_runner.py:21–34`.

The new rule is `returncode == 1 and at least one FAILED line`. Pytest exit 1 can contain both a failed test and a setup error that prevented another selected test's body from running. The runner neither rejects errors nor proves that every intended test executed. It reports only FAILED lines, omitting the setup-error evidence.

**Measured:** A scratch pytest file contained one intentional assertion failure and one test whose fixture raised before the test body. The actual pytest result was:

```text
exit 1
FAILED .../test_mixed.py::test_failure
ERROR  .../test_mixed.py::test_required
1 failed, 1 error
```

I supplied that real captured subprocess result to an unmodified scratch copy of the revised runner, using a scratch mutation target. It printed `X1 RED 1 failed, 1 error in 0.01s`, then `NOT RED: none`, and exited `0`. Its scratch target was restored correctly.

The revision fixes the former “any nonzero exit is RED” rule and makes survivors/invalid anchors fail the runner. Item 8 is nevertheless **partly met**, not closed. Reject error/interruption/incomplete-test outcomes even when another test fails, and retain structured collection/execution/failure evidence for the intended test set. This finding does not reclassify the existing ledger's recorded assertion failures as setup failures; it demonstrates a remaining false-green path in the runner itself.

### Round-one disposition and reproduction matrix

All numbers below come from synthetic probes or the permitted suite, not the box or evaluation data.

| Round-one item | What the revision and repeated probe do | Disposition |
| --- | --- | --- |
| B1 / required change 1: unpinned raw/cache/table reads | Ten-pin inventory; frozen lookup installed before real feature computation; park drag bypassed. A full synthetic computation made **zero** raw/cache/table reads after installation, despite a changed live cache and a 2026 raw canary. | Closed for the run path, subject to the explicitly supplied box/cache facts. |
| B2 / change 2: one/two/four seeds, duplicates, seed 42 | Actual aggregate now raises `AggregateError` for every one of these cases. Direct disposition is incomplete for counts other than three. | Original count/seed exploit closed; completion/evidence validation remains open in R2-3, and metadata HEAD handling is wrong in R2-1. |
| B3 / change 3: arbitrary output roots and later seed without release | `--out-root` now causes argparse exit **2**. A second same-seed call in the canonical namespace refuses its claim. Seed 2 without the release raises `SystemExit` before input reads. The keyword-only `_test_out_root` is explicitly a synthetic test seam. | CLI namespace repair met; owner/completion release validation remains open in R2-2. |
| B4 / change 4: resumed-only hit | Original miss plus resumed-only hit now has label **0**. A batter with only resumed PAs has no original label, is dropped, and ranks are renumbered. Probe counts: `changed=1`, `void_dropped=1`. | Label repair met, with the retrospective/void interpretation below. |
| B4 / change 4: future resumption assigned an old official date | The June 6 feature still changes **0.0 → 0.1** when the September event is assigned June 1. The manifest helper records one resumed row in 2024. §8 now expressly discloses this convention and abandons unconditional leak freedom. | The error mechanism persists, but the bounded retrospective claim is now honest. This was a permitted resolution in r1 change 4; I do not reopen it as an uncorrected unconditional claim. |
| B5 / change 5: P(57)-only disposition flip | Same fixed P@1 witness still changes **inconclusive → positive**, with `m=0.0076666667`, `t=2.0909090909`, when seed 2's exact P(57) delta changes from 0 to +0.001. §5 now correctly states this indirect decision weight. Zero-mean/zero-spread t is documented as 0. Seed origin is disclosed. | Closed; the flip is now the declared rule. |
| B6 / change 6: import-time settings and constructor seed/flags | Effective `(20,7)` passes; `(0,10)` refuses. Actual training helper, with a recording constructor and no real fitting, passed each registered seed and both true deterministic/row-wise flags to LightGBM. Effective settings are recorded. | Closed, conditional on the supplied production-default assertion. |
| B7 / change 7: fixed 45-hour third budget | Old `gate(60.48,45,True)` remains `over_cap`; a hypothetical **owner-released 30-hour** budget gives `gate(30.48,30,False)=ok`, `gate(60.48,30,False)=checkpoint`, and `gate(60.48,30,True)=ok`. Wrapper uses the released number. | Closed as a budget procedure, not a cost measurement or an owner release. |
| Required change 8: accountable mutants | Invalid anchors/survivors/non-1 errors now fail the runner, but the mixed-error probe is RED and exit 0. | Partly met; R2-4 remains. |

### Frozen lookup, labels, and bounded claims

**Freeze versus production:** Given the author's unverified box assertion that `data/raw` holds only 2026 and non-season folders, historical pinned games get their production lookup entries from the cache. `freeze_lookup` preserves those entries unchanged and drops non-pinned game IDs. I repeated the production/frozen comparison with 24 synthetic historical game IDs, an extra cached game, and an uncached 2026 raw feed. Real `compute_all_features` produced equal values for **all 16 FEATURE_COLS** before and after freezing; both frames had 28 non-null bullpen values. The frozen lookup contained exactly the 24 selected game IDs, park drag was all NaN, and pitcher parity passed on the 48 rows. Changing the live cache/raw canaries afterward did not affect the closed computation.

This establishes the implementation behavior for that synthetic case and supports the source-level restriction rule. It does not verify the actual box's directory inventory, cache contents/coverage, production environment, nine PA hashes, or real-input feature parity. Those are supplied author facts or future admission/run evidence. Missing historical cache entries remain missing; the freeze does not invent coverage. The manifest's `lookup_games` is an entry count, not a measured attribution-accuracy rate.

**Void rule:** The new code implements the written rule exactly, including dropping a void rank-1 row and promoting original rank 2 to rank 1. A separate synthetic probe showed that promotion yielding a surviving rank-1 hit. This is an offline ranking metric after original-portion eligibility filtering, not a replay of a committed BTS choice: the production grader's void does not retrospectively enter the next-ranked batter. §4 explicitly states the drop/re-rank operation, so there is no hidden code/spec divergence. Results must keep that estimand and its per-variant missing-day/void effects visible; “labels follow BTS scoring” should not be enlarged into a claim of executable replacement picks or contest-policy value. No real prevalence or impact was measured here.

**Other limits:** The postgame catcher proxy, possible non-starter selection, missing pre-2019 catcher history, consumed seasons, outcome-ranked seed origin, lack of multiplicity adjustment, absence of a pregame serving input, and unmeasured compute estimate are now plainly bounded. The production copied formula and blend rewrites were not changed. A synthetic third-tuple blend config retained its extra column and original parameter dictionary. I found no new grouping-key, deterministic-flag, dynamic-seed, or probability-mode defect in this change.

### Test and ledger strength

The permitted suite is **55 framing tests + 25 shared admission tests = 80 passed in 40.14s**. New tests exercise meaningful input, label, cardinality, source, and setting checks, and the resume ledger is candid about the five masked survivors.

I verified **54 mutation specs, each with exactly one current source match, and 54 latest recorded RED outcomes** in the revision-2 section. F24 is retired; G1–G29 are present. The first revision-2 run has 49 RED and five survivors; the final resume kills those five with specific-check tests. `git diff 307e7bc 02516cf` changes only the README, ledger output, and tests; the driver and runner did not change between that ledger run and this reviewed commit. I did not execute the tracked-source mutation runner.

The evidence proves those listed kills, not the missing trust-boundary cases:

- G9 enforces the equal-HEAD refusal, but does not model an admitted metadata-only owner release.
- G12 removes the completion check entirely; the positive witness is `{}`, so it cannot prove semantic completion.
- G13 removes the source check entirely; its test compares `Eric` with `Manager`, leaving the accepted `Ericsson` prefix untested.
- G4 hashes claim bytes but does not validate claim contents; G6 changes a summary alone but does not coherently replace both diff and summary while leaving scorecards contradictory.
- No new test requires complete registered manifest values or all six unit/profile/scorecard artifacts at aggregation.
- No runner test combines an intended failure with a setup error in another selected test.

The end-to-end fixture still stubs admission, input loading, feature computation, and walk-forward; its Monte Carlo uses 200 trials for speed. I used the same bounded seams for output/claim probes, plus separate real feature-computation and constructor-boundary probes. None was a real walk-forward or measured season result.

## Required changes

No tracked repair was made. These are requirements to resolve if the owner elects to reopen the item; they are not authorization for a third round.

1. **Separate executable identity from metadata HEAD.** Preserve each actual run HEAD and claim code, but validate equality of the admitted reviewed closure, accepted review/exposure identity, ten pinned inputs, and effective settings. Allow only verified metadata descendants under the unchanged shared admission rule. Add a seed-1 → committed owner release → seeds-2/3 synthetic lifecycle test that successfully aggregates without altering earlier evidence. Adjust the registration's equal-HEAD condition accordingly.
2. **Validate the owner/completion witness.** Require the exact `Eric` source token, finite positive released budget, and a release bound to the applicable admitted seed-1 completion. Reuse a semantic single-run validator for that completion: correct seed and claim identity, valid current admitted inputs/code/settings, no stop, six expected completed units, and retained scoring evidence. Refuse empty, wrong-seed, stale, unreadable, incomplete, or invalidated completion artifacts. Test these at both `launch` and child `run`, including the `Ericsson` source counterexample.
3. **Complete aggregate validation.** Validate expected manifest values and complete identities, parse and bind claim semantics, establish the claim namespace/portable provenance, and check all six units and required retained artifacts. Recompute diffs from the retained baseline/variant scorecards and reconcile them with the results' per-season values before deriving summaries. Refuse opaque claims, jointly missing identity/pins, consistently wrong basis/settings, and the negative-to-positive B counterexample where cards/profiles still show zero delta. Do not weaken the currently correct count/duplicate/seed/stop checks while fixing the HEAD refusal.
4. **Reject mixed-error mutant runs.** Require complete collection/execution evidence for the selected tests, zero collection/setup/execution errors or interruptions, and the intended failure witnesses before RED. A failed assertion elsewhere must not mask an unexecuted required test. Preserve target restoration and nonzero overall exit for survivors, invalid anchors, or inconclusive outcomes. Add the mixed pytest failure/setup-error runner test and the omitted lifecycle/witness mutants above.

A plain SIGN was not achieved within the two-round limit. Any additional review needed to admit a repaired commit requires the owner's recorded ruling.

## What was run

- Read `.codex-review/c2-framing/prompt-r2.md` first; used the in-checkout r1 report and its B1–B7/eight-change contract. Read the revised registration, driver, tests, mutant runner/spec/output, change diff and `307e7bc` mapping, plus relevant unchanged admission/launcher/feature/build/backtest/scorecard sources. No outside context corpus was consulted this round.
- Initial `git checkout --detach 02516cfedc812f047295b6f2ab212fd72a944507` failed on the out-of-sandbox worktree index lock. The runtime-required approved retry completed. `git rev-parse HEAD` returned the required commit; the tracked tree was clean. `git diff --check dc58a32 02516cf` passed. No review repair or commit was made in this checkout.
- Ran the permitted command with `MPLCONFIGDIR` exported to the owned scratch directory: `UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run python -B -m pytest tests/scripts/c2_framing tests/scripts/test_c1_admission.py -p no:cacheprovider` — **80 passed in 40.14s**.
- Executed owned scratch `probes.py` and `followups.py` with the checkout's venv Python, `-B`, `PYTHONPATH=.`, `BTS_LGBM_DETERMINISTIC=1`, and scratch `MPLCONFIGDIR`. Results and exact counterexample numbers are retained in this report. Three complete `run` calls used fake walk-forwards, stubbed admission/PA loading/features, real claim/output/scoring code, and 200 Monte Carlo trials. The actual full feature-computation canary used entirely synthetic PA/cache/raw inputs. The constructor probe fitted nothing. No `blend_walk_forward` or real-data run occurred.
- Ran one intentionally failing scratch pytest file to obtain the mixed failure/setup-error result, then exercised the unmodified runner in a scratch repository layout with its subprocess result supplied by a recording stub. Only scratch targets were mutated; source in the checkout was unchanged.
- Verified all 54 mutation anchors and the latest output statuses without running the author's tracked-source mutator. Reviewed driver SHA-256: `23721e9b6e497c6b738a97de6a3eb23a03689118ee4a0d651e2f2bd22a566ee2`; revised registration SHA-256: `a8d2287711c9f094ca07907362445cb5ce136fdeaff9f6bc0379e568eec92e45`; runner SHA-256: `051d8001b4806e892c35398344f4fe1136d63278215894883cbfa6df44ca34d3`.
- Real data reads were limited to the permitted canonical seed JSON through the prescribed tests. No real raw/PA/cache/external-table data, `.env`, credentials, production configuration, box, network, SSH, `gh`, live provider, or operational state was read. The author-supplied box facts remain unverified assertions here.
- Owned scratch `/private/tmp/c2-framing-r2-wpobqiwt` is deleted at completion. Report structure, reviewed-commit uniqueness, final report hash, receipt, final HEAD and unchanged tracked tree are verified before the completion marker. The report is not modified after that hash is finalized.
