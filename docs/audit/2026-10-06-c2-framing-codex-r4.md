## Verdict
**BLOCK.**
Reviewed-commit: 04439f9eb44d527a52c815d552d12a4e6fb5583d

**Scope disclosure first:** The permitted pytest run, and the initial scratch runner attempts, attempted a denied `stat` of `/private/tmp/codex-daemon-501` while collecting tests outside the runner's pinned repository root. No contents from that directory were read. I disclosed this during the review, made no escalation request, and confined the subsequent runner reproduction to an owned repository layout. No outside context corpus or memory registry was consulted this round. The complete r4 prompt was read before any other file; the requested HEAD and clean tracked tree were confirmed.

All four concrete r3 counterexamples now refuse or execute correctly. Two new false greens remain in the revised checks: nullable season evidence changes B from **negative to positive** without changing hits or probabilities, and captured test output substitutes for an outcome of a test whose body never executes. H12's unconditional equivalence claim also exceeds the validator's accepted domain. The permitted suite finished with **121 passed, 1 failed**, not 122 passes.

This is Eric's authorized round 4 and the last under the current ruling. The item stops. A fifth review requires a new ruling from Eric; this BLOCK admits no seed or run.

## Findings

### R4-1 — The new season check accepts nullable missing values and excludes misses from the decision metric

**Changed check:** `scripts/audit/c2_framing/screen.py:569–586`, especially line 575. **Consumed metric:** unchanged `src/bts/validate/scorecard.py:43–47,240–254`.

The new check promises every profile row belongs to its named season:

```python
own = len(part) > 0 and bool((part["season"] == s).all()) and bool((years == s).all())
```

For pandas nullable `Float64`, comparison with `pd.NA` returns a nullable Boolean, and `.all()` skips missing values. A missing season therefore passes. The scorer groups by `season` with its default missing-key exclusion: those rows disappear from per-season P@1, while pooled precision still includes them. Exact scorecard recomputation repeats that omission; the new card/profile join does not catch it.

**Measured at the actual consumption and release boundaries:** Starting from three complete synthetic producer runs, I changed one rank-1 miss in seed 1's B/2025 profile to missing `season`, retaining its 2025 date and every other column. I regenerated the scorecards, diffs and results from the retained profiles using the production helpers. The stored P@1 changed from `0.4` to `0.41025641025641024`. The actual validator accepted, the release helper returned `(True, "released", 30.0)`, and `launch` submitted the seed-2 command to a recording executor. In a separate owned namespace with no seed-2 claim, actual child `run` returned 0 using the fake walk-forward. No real launch or training occurred.

**Stronger decision witness:** On both B profiles of the first two seeds, I made `season` nullable and missing only for rank-1 misses: 22 rows in 2024 and 24 in 2025 per seed. All hit/miss, rank, date, probability and other column values were checked equal before and after. Every date remained in its named year, and every season comparison's `.all()` returned true. Cards, diffs and summaries were regenerated from those damaged profile bytes. Actual aggregation accepted:

| B evidence | Before | After |
| --- | --- | --- |
| Per-season P@1, first two seeds | 2024: 0.45; 2025: 0.40 | 2024: 1.0; 2025: 1.0 |
| Pooled P@1, baseline and B | 0.425 | 0.425 |
| Mean P@1 delta | 2024: 0; 2025: 0 | 2024: 0.3666666666666667; 2025: 0.39999999999999997 |
| Seed-level mean / t | 0 / 0 | 0.3833333333333333 / 2.0 |
| Passing seeds / disposition | 0 / negative | 2 / positive |

This is a semantic-completion false green in the new validator, including its seed-1 release consumer. The ordinary producer writes a non-null integer season; I am not claiming it ordinarily emits this damage. The consumer explicitly claims to reject incomplete retained season evidence and does not. Existing wrong-season and empty-profile tests do not exercise nullable missing membership.

### R4-2 — The new execution inventory accepts captured stdout as a pytest outcome

**Changed runner:** `docs/audit/2026-10-06-c2-framing-evidence/mutant_runner.py:39–54,79–90`.

The runner collects the intended nodes before mutation and correctly clears inherited `-x`. However, `classify` scans the entire pytest stdout for lines beginning `PASSED` or `FAILED`. It does not distinguish pytest's outcome summary from a failing test's captured application output. Exact node-name matching does not authenticate the source of that line.

**Measured through actual runner `main` and actual pytest subprocesses:** The scratch spec selected exactly two nodes and mutated `value = 1` to `value = 2`. The first test printed the exact node id of the second before failing its assertion:

```python
def test_first(request):
    if val() != 1:
        print("PASSED " + request.node.nodeid.rsplit("::", 1)[0] + "::test_second")
    assert val() == 1

@pytest.fixture
def setup_gate():
    if val() != 1:
        pytest.exit("owned before-body stop", returncode=1)

def test_second(setup_gate):
    # The actual probe writes a canary here before its assertion.
    assert val() == 1
```

Under the mutant, the second fixture stopped pytest before that test's body. Its body-entry canary was absent. The actual output contained:

```text
----------------------------- Captured stdout call -----------------------------
PASSED captured_outcome_spoof/test_boundary.py::test_second
=========================== short test summary info ============================
FAILED captured_outcome_spoof/test_boundary.py::test_first - assert 2 == 1
1 failed in 0.11s
!!!!!!!!!!!!!!!! _pytest.outcomes.Exit: owned before-body stop !!!!!!!!!!!!!!!!!
```

The runner returned **0**, classified **RED**, printed `NOT RED: none`, and restored its scratch target byte-for-byte. The captured line filled the missing-node inventory; the exit footer did not match `UNCLEAN`, and exit 1 plus a real failure completed the false certification.

For this isolated reproduction I copied the runner's unchanged bytes into an owned `docs/audit/framing/` layout, so its root directory contained the scratch tests. I replaced only the subprocess prefix `uv run python` with the checkout's venv Python, retaining the pytest arguments, environment and reviewed classification/main logic. This avoided the separate collection failure below. It is independent evidence of the boundary defect, not evidence that the author's recorded ledger used fabricated output.

The r3 inherited-`-x` reproduction in that same layout now correctly executes both named tests: both assertion outcomes appear, the second body canary exists, the runner returns 0/RED, and the target is restored. The original counterexample is fixed; complete execution is still forgeable through the new text inventory.

### R4-3 — H12 is equivalent only under an identity-shape restriction the validator does not enforce

**Claim:** evidence README line 58. **Changed validation:** `screen.py:548–553`. **Removed comparison in H12:** aggregate line 624.

The README says each validated run must equal the trusted identity, so validated runs cannot disagree on identity. In fact, validation compares only the projection onto the five `IDENTITY_KEYS`; aggregate compares the entire identity dictionary. Extra keys are accepted by the validator.

**Measured on an owned H12 copy of `screen.py`:** Adding `identity["unrecognized_extra"] = "owned probe"` to only the third synthetic manifest left all five admitted fields intact. Direct `validate_run` returned true. Original aggregate refused `runs disagree on identity`; the H12 mutant aggregate returned successfully. The manifest was restored after the probe, and no tracked source was mutated.

Thus H12 is not equivalent over all artifacts accepted by the current validator. It is equivalent over the ordinary producer's five-field identity domain, if that narrower domain is stated. This does not demonstrate spoofing of any known admission field: the trusted five-field comparison and the `bogus` refusal work. It does invalidate the current unconditional equivalence argument and leaves an accepted-input shape untested. G8's equivalence remains sound: each run's pin dictionary already equals the same trusted dictionary, and its digest is checked.

### Disposition of R3-1 to R3-4

| Required change | Repeated r3 reproduction on r4 | Assessment |
| --- | --- | --- |
| R3-1: bind every decision-bearing scorecard metric and scoring settings to profiles | B's exact P(57) edited to baseline +0.001 in the first two cards, with coherent diffs/results and all 18 profile SHA-256 values unchanged: both validation and aggregation refuse `scorecard does not match a recomputation from its retained profiles`. | Met for the reviewed card/profile join. The producer reads retained parquet back before scoring; full replay excludes only timestamp. |
| R3-2: actual evidence of both completed seasons at validation, release, launch and child | 2024 profiles standing in for 2025 now refuse `not complete 2025 evidence` at direct validation; release returns false; launch and child run raise `SystemExit` before a later-seed claim. | Original counterexample fixed; requirement remains incomplete because nullable season evidence passes (R4-1). |
| R3-3: trusted identity/pins and admitted ancestry/closure without equal HEADs | Coherent seed-1 HEAD `02516cfedc812f047295b6f2ab212fd72a944507`, with rebound claim hash and unchanged supplied identity, now refuses at validation, release, launch and child: exposure is not its ancestor. Literal `bogus` identities across all three manifests refuse `admitted identity`. | Met for known identity fields and executable HEAD binding. Metadata descendants remain accepted; H12's broader shape claim needs qualification (R4-3). |
| R3-4: all intended nodes execute under inherited fail-fast | In the owned layout, `PYTEST_ADDOPTS=-x` is removed; both selected tests execute and fail, second body canary exists, runner returns 0/RED with restored target. | Original counterexample fixed; complete-execution requirement remains open because captured stdout supplies a missing outcome (R4-2). |

### False-refusal checks and the previously closed items

I produced a full three-seed synthetic stage one at the **registered defaults, `mc_trials=10000, season_length=180`**, using the actual `run` orchestration, retained-parquet readback, full scorer, validator and actual git ancestry/closure helper. The HEADs were existing commits `[068513d…, 04439f9…, 04439f9…]`. Both same-process aggregation and a fresh Python process accepted the runs as A positive/B negative. A separate default-scoring seed-1 run also validated in a fresh process. I found no local false refusal from deterministic scorecard replay or the metadata-HEAD lifecycle.

These are synthetic admission/pin identities: the expected reviewed/exposure commit was `068513df687710820dc074ee81d9f80bc1000bb6`, with shaped synthetic report/admission hashes. Admission and PA/feature loading were stubbed for these orchestration probes, and walk-forward was a fake callback. Actual git checks were restored rather than stubbed. No real accepted SIGN, exposure, committed owner release or admission record was created. Separate actual git probes reject a nonexistent `f…f` HEAD and a descendant whose executable closure differs from the supplied reviewed commit. The shared admission suite passed its disposable-repository cases.

AST comparison against r3 found all **22** previously closed functions unchanged, including feature/self-check, lookup/input closure, labels, settings, variants, summaries/dispositions, first-unit stop, admission, release, seed order and launch. Independent synthetic regression probes repeated the relevant runtime boundaries:

- **Closed inputs:** Actual `compute_all_features` on 48 PA rows across 24 games, with an owned live lookup/cache and 2026 raw canary. Freeze retained exactly 24 games. After changing the live canaries and installing closed inputs, a `Path.read_text` trap recorded zero reads and the park-table spy recorded zero calls. All 16 registered features matched production computation exactly; both frames had 28 non-null bullpen values, park drag was all NaN, and self-check was identical on 48 rows.
- **Labels:** An original miss plus resumed hit remained a miss; a resumed-only batter was dropped. Counts remained `changed=1, void_dropped=1`, ranks `[1,2]`, hits `[0,1]`. The previously disclosed official-date/future-resumption availability limit and retrospective drop/re-rank convention remain unchanged; this is no new unconditional availability claim.
- **Settings and constructor:** Effective `(20,7)` passed and `(0,10)` refused. The real training helper with a recording classifier and no actual fit used all three registered seeds and both deterministic/row-wise flags.
- **Namespace and release:** The permitted suite's cardinality, foreign-seed, canonical-namespace, repeat-claim, missing-admission and release-binding cases passed. Independent release parsing accepted the exact Eric/30-hour row and refused `Ericsson`, zero and decimal overflow to infinity. The nullable witness's recording executor received the registered seed-2 name, deterministic environment and released 30-hour budget.
- **Budget accounting:** `gate(60.48,45,True)=over_cap`; hypothetical 30-hour releases gave `gate(30.48,30,False)=ok`, `gate(60.48,30,False)=checkpoint`, `gate(60.48,30,True)=ok`. No owner release or actual compute expenditure was issued by these probes.

The author's box inventory and environment facts remain stated, unverified facts. Local synthetic replay does not prove exact replay on the box, real-input coverage or actual admission. No new regression in the closed feature/input/label/settings/budget rules was found.

### Suite and mutant evidence strength

The permitted suite collected **97 framing + 25 shared admission tests** and ended **121 passed, 1 failed in 151.66s**. Failure: `test_the_runner_clears_inherited_fail_fast_and_runs_every_named_test`, because runner `main` returned 1/`INVALID: the named tests do not collect`. Its nested collection exited 4 with the denied external-directory `stat` described first. Pinning `--rootdir` to the checkout while targeting tests under external scratch made pytest traverse outside the owned test directory. I did not escalate or claim the author's recorded suite failed identically. The subsequent owned-layout reproduction establishes the inherited-`-x` behavior separately; it does not turn this prescribed-suite result into a pass.

All **88** current mutant specifications have exactly one current source anchor, select node ids, and contain no options. Latest recorded outcomes are **86 RED, G8 and H12 SURVIVED, none missing**. The first r4 sweep had 84 RED/4 SURVIVED; the H5/R5 resume supplied two new RED results. The resume's `exit 0` covers those two selected entries, not a fresh full 88-entry run. I did not execute the tracked-source mutator.

N1–N16 and R1–R5 improve coverage of full scorecards, scoring declarations, actual seasons, trusted fields/HEADs and missing exact nodes. They do not exercise nullable missing seasons, an interrupted test with a captured outcome line, or identity extra keys. The classifier tests use constructed stdout strings, which do not establish provenance of real stdout sections. The ordinary orchestration fixtures use 200 Monte Carlo trials and stub input computation/admission/walk-forward; the additional default-scoring and real-feature probes above address those respective local limitations without claiming real model evaluation. G8 can remain equivalent; H12 needs a narrower documented domain or a distinguishing test/schema rule.

## Required changes

No tracked repair was made. These are unresolved requirements, not authority to reopen the item, repair it or conduct a fifth round.

1. **Reject missing membership before accepting a completed season.** Require every retained row to have a non-missing season and date/year that equals the named unit, with comparison reductions that cannot skip nullable missing values. Add the `Float64`/`pd.NA` witness at validation, seed-1 release, launch, child run and aggregation. The selective-miss omission must refuse before it can change a disposition; preserve the full-card replay and ordinary metadata lifecycle.
2. **Bind execution evidence to pytest reports rather than captured application stdout.** Distinguish selected nodes and completed outcomes using trusted structured pytest/session evidence, and reject interrupted or incomplete sessions. A captured `PASSED <exact intended node>` line cannot count as execution. Add the actual two-node fixture-exit/captured-output witness, alongside the repaired inherited-`-x` test. Establish the runner test's collection in the permitted owned scratch layout without traversing unrelated directories.
3. **Correct H12's equivalence claim and identity-shape contract.** Either enforce the stated exact identity shape and test extra keys, or state the restricted producer-only domain and account explicitly for the distinguishing accepted artifact. Do not treat H12 as unconditionally equivalent over the current validator's accepted manifests. Preserve trusted known-field identity/pin checks and G8's valid equivalence.

## What was run

- Read the complete r4 prompt first, the checkout's r3 report, the seven-file r3-to-r4 diff, revised registration, changed tests, admission/scorer sources and recorded ledger. No outside context corpus or memory lookup.
- Confirmed detached HEAD `04439f9eb44d527a52c815d552d12a4e6fb5583d`, clean tracked status and `git diff --check da87dad2a36f07270282c15e74c82dcd15d49694 04439f9eb44d527a52c815d552d12a4e6fb5583d`.
- Permitted command: `UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run python -B -m pytest tests/scripts/c2_framing tests/scripts/test_c1_admission.py -p no:cacheprovider`. Added `TMPDIR` and `MPLCONFIGDIR` pointing under the owned scratch prefix. Result: **121 passed, 1 failed, 151.66s**; collection diagnosis and external stat attempt disclosed above.
- Owned scripts `probes.py`, `followups.py`, `equivalence.py`, `regression.py`, `defaults.py` and its fresh-process replay used the checkout's `.venv/bin/python -B` with `PYTHONPATH=.`. They exercised all four r3 counterexamples, both new false greens, H12, default-scoring three-seed replay, real git checks, real synthetic feature computation, labels, recording constructors, release parsing and budget accounting. The initial nullable `Int64` probe stopped on JSON serialization of a NumPy integer key; the completed nullable counterexamples use `Float64`, as reported. No real walk-forward or model fit ran.
- Verified 22 unchanged function ASTs, all 88 current anchors/test lists, latest ledger statuses, and that production screen/runner bytes did not change after the fix commit; subsequent commits concern tests/specifications/evidence.
- Reviewed SHA-256: `screen.py` = `c77ab889366f4d986ccaf51a8f81cf2d08c9ffdad2cacb1a2c82519ee0624386`; registration = `e73cdd65451b435b84f63003eb43f9d563acf0b74bd936724f7cdb93da0276aa`; mutant runner = `f33c1b927404530c62edfc39ba0fb1aeb05541b2e07c06cace67541d32752e55`.
- All independent artifacts were confined to `/private/tmp/c2-framing-r4-t34452m5`, deleted at completion. Only this report and `receipt-r4.json` are written in the checkout. No real evaluation-data reads, configuration/credential reads, box/network/SSH/gh, escalation, tracked edits, commit or push. Final report structure/hash, receipt binding, scratch deletion, HEAD and clean tracked status are verified before the completion marker; the report is not edited after its final hash.
