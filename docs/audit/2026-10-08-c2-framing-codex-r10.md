## Verdict
**SIGN.**
Reviewed-commit: a3f5e3e89d8b2615e50d8c4aed5c08cbfe02a56f

R9-1, R9-2 and R9-3 are closed under Eric's revision-10 committed-code ruling. I found no new blocking false green or false refusal in the change. HEAD matched the requested detached commit and the tracked tree was clean before and after the checks. `screen.py` is byte-identical to revision 9.

This signs the reviewed procedure for seed 1 only, subject to the existing exposure, admission, owner and compute gates. No real seed was launched. Seeds 2–3 still require the measured seed-1 report and Eric's exact release; no experiment result or production change is approved here. The certificate assumes reviewed committed code and a trusted interpreter installation, as the new ruling expressly requires.

## Findings

### 1. R9-1: closed; collection is bounded in this sandbox

Verified against the unchanged revision-10 runner. The permitted suite passed **271 tests, zero failures, in 228.47 seconds**. The 27 sandbox failures recorded in round 9 did not recur.

In my own `owned_ancestor/scratch` repository under `/private/tmp/c2-framing-r10-j2iWwYAv`, an initializer above the root would write an ancestor canary if imported. The gate and boundary both returned no problems. The runner collected exactly two intended nodes, completed their failing calls and classified the mutant RED, returning 0; the ancestor canary remained absent and the target was restored. The supplied `r9_witness_replay.py` independently produced the same revision-10 outcome. The common argument builder keeps an empty configuration, no conftests and the root collection cutoff for both collection and execution (`scripts/audit/c2_framing/mutant_runner.py`, `_args`, `collect`, `_drive`).

Scope disclosure: the supplied replay's **old revision-9** runner attempted ancestor metadata traversal at `/private/tmp/codex-daemon-501`. The sandbox denied it. A captured old-collection diagnostic confirmed return code 4 and that permission error; no source from that path was read. In this sandbox, all three unchanged old-runner replay cases were INVALID, with no canaries, rather than reproducing the author's old-runner RED outcomes. My comparative old-runner controls below therefore add only the root cutoff and are explicitly conditional. The unchanged revision-10 controls need no adjustment.

### 2. R9-2: closed; an imported fixture cannot come from local shadow bytes

Verified with my own imported-fixture control in `owned_imported_fixture/scratch`. Its test uses the allowed numpy spelling and a fixture supplied by that import. The syntax gate admitted it, but the boundary reported that `numpy.py` was uncommitted and shadowed the installed numpy module. `main` returned 1 and REFUSED before collection; the fixture canary remained absent and the target remained original. With only the collection cutoff added to the old revision-9 runner, the same fixture executed its canary and the ordinary failing test was certified RED. That is a conditional comparison, not an unchanged-r9 measurement.

The supplied replay also refused the local numpy control before execution. Deliberately bypassing preflight for its low-level run produced INCONCLUSIVE with a retained boundary refusal and no canary. The committed-shadow variant in the permitted suite passed its refusal test as well. Committing a look-alike is insufficient to admit it.

The new preflight walks startup import entries inside the root and the named tests' package roots, recursively through importable directories (`mutant_runner.py:686`). It requires the root to be a committed work-tree top, reads regular-file blob identities from HEAD, and compares current bytes using Git's blob header and the appropriate SHA-1/SHA-256 algorithm. Namespace directories are covered; only top-level installed-name collisions are shadowing. Nested names and non-importable files are permitted. The tests exercise package-root selection, startup entries, changed bytes, symlink blobs replaced by files, inherited Git variables and SHA-256 repositories. I found no admitted import route around the boundary within the reviewed vocabulary and trust assumptions.

### 3. R9-3: closed; linked source is refused, with the ordinary-file control retained

Verified in my own `owned_linked_rewrite/scratch` repository. A pre-existing committed `numpy.py` link named physical source outside the root, which a child could write before the import. The gate returned no problems, but preflight refused the symlink and installed-name collision; no source canary ran. Bypassing preflight solely to test the audit layer returned pytest status 1 and **INCONCLUSIVE: code outside the boundary**, retained the outside-root refusal, and still wrote no canary. The target was restored.

With only the collection cutoff added, the old revision-9 runner certified this linked case RED after its source canary executed. My direct-file comparison instead became INCONCLUSIVE because the change scan detected `numpy.py`. Revision 10 refuses that direct shadow before collection and also refuses it in the forced runtime control. Separately, the admitted `owned_ordinary_file_change/scratch` control wrote a non-code marker during its failing call: revision 10 returned 1 and INCONCLUSIVE for changed files, identifying both the root directory and marker. Thus the existing ordinary-file change defence remains effective. The supplied linked-source replay and the suite's in-root linked-to-committed-source control passed their revision-10 refusals.

### 4. The audit and driver-based collection enforce the stated boundary

Verified by the permitted tests, the forced controls above and selected guard reversions. The driver installs the audit hook before importing pytest (`mutant_runner.py:231`). File-source compilation records a Git blob identity under its filename; execution checks that identity against HEAD or the explicitly authorized mutation. Physical paths outside the root and installed trees, symlink components and installed-name shadowing are refused. Bytecode deserialization is refused even when its advertised source filename names a committed test. Extension imports and native-library loads must resolve inside installed trees.

Each refusal is retained before ImportError is raised. The caught-refusal controls therefore remain INCONCLUSIVE, and caught collection refusals yield no intended selection. Collection obtains its nodes from driver evidence, not printed pytest output (`mutant_runner.py:810`). Passing the mutation's blob identity keeps legitimate imported-target mutants executable. Installed trees come from the scrubbed child interpreter's standard-library, site-package and outside-root startup entries; a tree containing the root cannot turn arbitrary root files into trusted installed code.

This is an audit boundary within the stated model, not a sandbox for arbitrary hostile installed or committed helpers. The hook exempts synthetic angle-bracket filenames and relies on compilation events; the gate excludes direct dynamic execution/import/deserialization routes. Its compiled-byte map is keyed by filename, not by code-object identity. I found no in-scope, admitted route that exploits those facts. Fabricating filenames or executable objects through newly malicious committed helpers would discard the explicit reviewed-code assumption, and is not evidence against this ruling.

### 5. No measured false refusal or regression of the closed experiment rules

The real framing suite and all three package initializers passed both gate and boundary with empty problem lists. All **220 distinct selections named by the 242-entry ledger** appeared among the 271 passing baseline nodes. Through the unchanged audited driver, 75 runner-test functions selected **150 nodes**, passed all of them, and classified SURVIVED. The evidence retained 450 phase reports, 150 call observations, a completed zero-failure finish, no boundary refusals, no foreign plugins, no late registrations, no hook changes, no marked nodes and no outermost failures. This includes the earlier abort, rewritten-report, expected-failure and self-unregistering-wrapper controls. The vocabulary audit revisited 158 objects and produced the same seven reported callable/module entries.

My synthetic three-seed stage one used seeds 2273360, 260991262 and 1746737973, with genuine Git closure checks from registration commit `04568ac93d0cdbcf64a43a9274734dba56622261` to the reviewed HEAD. Both same-process and fresh-process aggregation succeeded with registered scoring **10,000 trials / 180 days**, A positive and B negative. These are synthetic dispositions, not experiment findings. Separate damage probes used 200 trials / 180 days.

The damage probes refused nullable seasons hiding 22 and 24 rank-one misses per year, forged secondary scorecards with unchanged retained profiles, duplicate-year evidence, extra/bogus identity fields, foreign/nonexistent heads, stale executable closure, bad probability strings and scorer exceptions. Release, launch, child and aggregate consumers refused the damaged seed-1 evidence; the fresh child created no seed-2 root.

On 48 synthetic PA rows over 24 games, the production 16-feature baseline and closed-input baseline were identical, with zero trapped ancillary or park-table reads, 28 non-null bullpen values on each side and absent park-drag values. Original-portion labeling changed one hit, dropped one void batter and reranked the two retained batters. Correct feature settings passed and both altered settings refused. Recording-only model constructors retained all three registered seeds, deterministic mode and row-wise forcing; no model was fitted. Exact release-source and finite-positive-budget grammar, checkpoint acknowledgement and hard-cap cases retained their expected decisions. The permitted suite also covered namespace, stop and input-closure rules.

### 6. Mutant evidence is adequate, with its limits stated

The committed revision-10 ledger contains 242 unique entries, including 54 additions R98–R150 and R152. Every source anchor occurs exactly once; no selection contains an option. Its recorded revision-10 section reports **239 RED, G8/H12/N10 SURVIVED, zero INCONCLUSIVE**, ending with process exit 1 because those equivalents survive. I checked the spec, recorded outcomes and new guards; I did not independently rerun the entire ledger or mutate this checkout.

In isolated scratch copies of the runner, I independently applied R8, R99, R110, R112, R134, R137, R138, R140, R143, R145, R148 and R133. All twelve runs returned pytest status 1 and **every named test failed** at its relevant assertion. R8's re-anchored driver removal lost evidence; missing preflight, commit-byte, symlink, bytecode, collection-refusal and mutation-override guards changed the behavior those tests require. Removing compile-byte recording caused a false refusal of an ordinary committed test, so the positive acceptance control protects that rule too. Other layers sometimes still refused a reverted guard; these tests establish the individual preflight/audit contract, not twelve unique routes to a certified false green.

R133 is correctly disclosed as message-only. My separate run with that branch removed still returned INCONCLUSIVE, retained a **not committed** refusal and wrote no canary. Its named test fails because the reason changes; it does not demonstrate removal of a distinct containment defence. Historical message-level entries such as N17 remain disclosed too. The 239 RED count must not be read as 239 independent behavioral defences.

For G8, H12 and N10, source inspection supports the recorded equivalence over accepted runs: trusted pins and bound digests already determine equality, identities already have exactly the trusted fields, and accepted daily ranks already contain rank one in both seasons. My original/mutated aggregate copies agreed on ordinary evidence and refused, respectively, coherently wrong pins, extra identity and missing-rank-one evidence. Their survivors are not an unresolved in-scope false green.

## Required changes

None. The three round-9 required changes are verified closed under the current ruling. Existing seed-1 exposure, admission, budget and owner conditions remain prerequisites; seeds 2–3 require their separate release.

## What was run

All test and runner processes finished before either review artifact was written. Runs used the checkout root, `UV_CACHE_DIR=/tmp/uv-cache`, the requested timezone, Python bytecode suppression, and TMPDIR/MPLCONFIGDIR under the owned prefix `/private/tmp/c2-framing-r10-j2iWwYAv`.

- The exact permitted pytest selection and cache-provider suppression: exit 0, 271 passed in 228.47 seconds, Python 3.12.13 / pytest 9.0.2 / pluggy 1.6.0.
- The supplied `r9_witness_replay.py` and `gadget_audit.py`: exit 0. Old r9 replay cases were INVALID in this sandbox; all current r10 canaries remained absent and the expected RED, REFUSED and forced INCONCLUSIVE outcomes were measured.
- Scratch `owned_boundary_probes.py`: exit 0; independent ancestor, imported-fixture, linked/direct rewrite, ordinary-file-change, complete-pass and real-suite gate/boundary controls. Old-runner comparisons added only the collection cutoff.
- Scratch `normal_driver.py`: exit 0; 150 audited nodes passed, classified SURVIVED, with complete and clean evidence.
- Scratch `profile_probes.py` and `regression.py`: exit 0; default-scoring three-seed/fresh replay, damaged-evidence refusals, equivalent controls, closed features, labels, settings, recording-only constructor parameters, release grammar, budgets and ledger consistency.
- Scratch `guard_mutations.py`: exit 0 as a review harness; its twelve isolated mutant pytest runs each exited 1 and failed every named selection. A separate R133 audit diagnostic also retained its refusal without a canary.

The receipt retains measured outcomes and hashes without reproduction source. Only owned disposable scratch Git repositories were committed; no tracked checkout file was edited, committed or pushed. No prohibited data/configuration/credential source, network, box, real walk-forward or real model fit was used, and no escalation was requested. Owned scratch is deleted as part of closeout; the report and receipt are the retained deliverables.
