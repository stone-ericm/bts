# Historical replay specs: drafting guide (W1.5 Phase 1, Task 5)

A historical replay is a **semantic regression replay** (design §9.2; `scripts/audit/incident_register/replay.py`). The complete fix set's tests, at the last fix commit F, run against the `src/` of the parent P of the FIRST fix commit, inside an owned worktree. It is accepted only if:
- **GREEN** (F's `src`): every selected node passes.
- **RED** (P's `src`, F's tests): every declared symptom node fails in its call phase with the builtin `AssertionError`, and its innermost repo frame is the declared assertion line. A `TypeError`, `ImportError`, `AttributeError`, `NameError` or `ModuleNotFoundError` never counts ("new API only").
- **The audit is complete**: every change outside `src/` between P and F has an audit entry.

It replays `src/` only. A fix to a script, cron line, systemd unit or workflow cannot be replayed this way.

## One spec per fix set
Each fixed link of an incident has fix commits: `fix[].implemented.commits` in `route_h/drafts/E*.json`, and the same list in `current_defence/worklist.md`. Links of one incident that share exactly the same fix commits share one spec, which lists all of those links. The label is `R-I<NNN>-<k>`, where k numbers the incident's fix sets in link order.

## Spec fields (all required except `env` and `notes`)
- `label`, `incident` (`I-NNN`), `links` (the link numbers covered).
- `fix_set`: the fix commits, oldest first, as short SHAs. Every one must be an ancestor of the last.
- `tests`: pytest arguments selecting the nodes to run. Prefer the specific node ids the fix added or changed, not whole files: every selected node must pass at F, and selecting less keeps unrelated timing-sensitive tests out. End with `"-q"`.
- `symptom_nodes`: the tests that encode the fixed contract and must fail at P by assertion. Each is `{"node": ..., "assertion": {"path": ..., "text": ...}}`.
  - `text` must occur exactly once in that file at F (`git show F:path`). If it occurs more than once, add `"line": N`, the 1-based line at F; that line must contain `text`.
  - The assertion must be the contract assertion, the one that fails because of the defect, and never a setup line.
  - Paths are never under `src/`.
- `audit`: one entry for EVERY path in `git diff --name-status -M P F -- . ':(exclude)src'`. For a rename, add an entry for both the old and the new path. Each entry is `{"decision": "neutral"|"irrelevant"|"adapter", "reason": "..."}`:
  - **neutral**: a test or harness change that carries or supports the contract without changing how the old `src` is exercised;
  - **irrelevant**: docs, plans, data manifests, or files the selected tests never load;
  - **adapter**: a harness change needed only so F's tests can run against P's `src`. It must also name its `"evidence"`. Avoid adapters while drafting; if one seems necessary, report it and leave the spec unfinished.
- `deployed_ref`: `{"sha": ..., "basis": ...}` from the link's deploy state in the draft or worklist (`"unknown"` / `"unknown"` when unknown; for a `candidate_ancestry` line, the sha and basis `candidate_ancestry`).
- `env`: `{"TZ": "America/New_York"}`, unless the tests need otherwise.
- `notes`: why each symptom node fails at P (the defect's mechanism in one or two sentences), and the static checks you did.

## Static checks to do for every symptom node (no test runs)
1. **It can only fail by its assertion at P.** Everything the test imports, patches (`@patch("bts.x.y")`, `monkeypatch.setattr`) or calls must exist at P with a compatible signature (`git show P:src/bts/...`). If it calls a function the fix ADDED, or passes a keyword the fix added, it will fail with a non-assertion error. Then it is not a symptom (the plan's "new-API-only red": 3a6e48b, a364b11, 8bceda1, 4f0257a, 41b2bb1, 0abf503).
2. **The assertion's failure follows from the defect.** Trace the code path at P.
3. **The anchor is unique at F**, or carries a line number.

Known lessons:
- `ce6676d`'s only symptom node is `test_post_cutoff_rescoring_does_not_flip_a_settled_hit`.
- `2ff2db9` changed a return shape: a test asserting the new shape fails at P by assertion only if the shape is compared, not unpacked.
- `736ea8f` imports inside the test: an import of a new name fails with ImportError, not an assertion.

## When there is no replay
Write a manifest entry `{"incident", "links", "spec": null, "unavailable": "<reason>"}` instead of a spec when:
- the fix touches no test, or only scripts, units, cron or workflows (`"the fix is outside src/; a semantic replay covers src only (design §9.2)"`);
- no test in the fix can fail by assertion at P (`"new API only: <test> fails at P with <error> because <name> was added by the fix"`);
- the fix set's tests cannot run against P without an adapter (say what the adapter would be).

## Worked example
`specs/R-I009-1.json` (I-009, fix `7638af7`).

## Constraints for drafters
- **Read-only:**
  - `git show`, `git diff`, `git log` and `git merge-base` in `/Users/eric/projects/bts-w15` only;
  - never `git checkout`, `reset`, `switch` or `stash`, and nothing that changes a worktree;
  - read the drafts and worklist there.
- **No test runs:** no pytest, no `uv run`. The machine is running the strict mutation sweep, and replays run only after it.
- **No access:** no `data/` reads, network, ssh or `gh`.
- **Where to write:** only your assigned staging directory: the spec files, plus `manifest.json` listing every link you were given (a spec label, or `null` with an unavailable reason).
