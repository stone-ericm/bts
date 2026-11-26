# Tasks 5 and 6: the certificate runs at the frozen pin f453283 (2026-10-03)

- **Specs and manifests:**
  - `../historical_replay/specs/` (28 R-specs) with `../historical_replay/manifest.json`: every fixed link, given a spec label or an unavailable reason.
  - `../current_defence/specs/` (23 D-specs plus the 9 plan-named prepared specs) with `../current_defence/manifest.json`.
  - The drafting guides are `SPEC_GUIDE.md` in each directory.
- **Results:** `../historical_replay/results-f453283/` and `../current_defence/results-f453283/`.
  - Each holds one directory per spec: the acceptance object, the events and stdout/stderr of every stage, and for a defence, the mutant patch.
  - Each also holds `summary.json`, which binds the pin, the observer source sha256, and each spec's sha256, verdict, acceptance sha256 and duration. The `uv sync` logs are kept as well.
- **D-I071-1, revision 1:** rejected, and kept as `../current_defence/results-f453283/D-I071-1.r1-rejected/` with its spec in `../current_defence/specs-superseded/`. The summary lists it under `superseded`.
- **Scripts:**
  - `run_specs.py` is the driver. It imports the frozen tooling from a checkout at the pin and builds one owned worktree per replay spec, plus one at the pin for all the defence specs.
  - `review_run.py` prints a run for review.
  - `replay.log` and `defence.log` are the drivers' logs.
- **`REVIEW-RUNS.md`:** the reviewer's decision on every run, with the witness contrast check. A record certifies only when the runner accepted it AND the decision is accept.
- **`drafting/`:** the lead reviews of the drafted specs (Task 5 batches b1-b5, Task 6 batches d1-d5) and their consolidated manifests.

The runs were made under `/Users/eric/projects/bts-w15-evidence/runs/`, and the absolute paths in the records point there.
