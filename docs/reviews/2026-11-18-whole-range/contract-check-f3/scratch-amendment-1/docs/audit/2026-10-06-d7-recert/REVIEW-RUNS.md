# D7 re-certification at the deploy candidate `f882411` (2026-10-06)

**Why:** `docs/ops/reconcile-receipt-v1.md` § Re-certification. The rank-2 producer set (P1, P2, P4, C1/C2) touched files that hold seven certified current-defence mutants. Each function body is unchanged, and each mutation still applies at an offset. The seven must be re-run with the frozen runner at the deploy candidate before they are claimed current. Eric's D7 of 2026-10-06, relayed by the herdr manager, names this gate.

**How:**
- **The driver:** `run_at_candidate.py`. It is `task56/run_specs.py`'s defence branch with one difference: the owned worktree is created at the candidate, and every spec's `baseline` is the candidate.
- **The tooling:** imported unchanged from the clean frozen checkout `~/projects/bts-w15-evidence/tool` at `f453283`. The observer source sha256 equals the original run's (`fc1b38da7bbe…`).
- **The specs:** the unchanged files in `../2026-09-29-incident-register-evidence/current_defence/specs/`. Each raw spec sha256 equals the one in the `results-f453283` summary.
- **Outputs:** `results-f882411/` (`summary.json` binds the tool pin, the candidate, the observer hash and each spec).
- **When:** run under `~/projects/bts-w15-evidence/runs/d7-f882411/`, concurrently with a fast-suite run and a Codex review's test run. No run failed.

**The decision rule (as in `task56/REVIEW-RUNS.md`):** a certificate needs the runner's `accepted` AND an `accept` here. I compared each run with its certified `f453283` run:
- the same killing tests;
- the same mutation body (the patch lines without hunk headers and paths);
- the same `ok` per certificate;
- the same `linked` witness events, after normalizing paths, line numbers, ids and addresses.

Only line offsets differ.

| Run | Runner | Decision | Reason |
|---|---|---|---|
| D-I043-1 | accepted | **accept** | identical to its f453283 certificate (2 tests: primary and double-down postponed regenerate); mutation body identical; branch line 708 → 742 |
| D-I043-4 | accepted | **accept** | identical (3 parametrized void-status tests); same deleted block as D-I043-1; branch line 708 → 742 |
| I-0813-a | accepted | **accept** | identical (2 tests: statusCode PW without detailed text; undelivered pick in Warmup not locked); branch line 719 → 753 |
| D-I043-2 | accepted | **accept** | identical (2 tests: schedule void with a live-feed Preview; void primary, score double once); branch line 793 → 827 |
| D-I043-3 | accepted | **accept** | identical (1 test: the polling cap does not overwrite a resolved result); branch line 2337 → 2343 (`scheduler.py`) |
| D-I063-2 | accepted | **accept** | identical (1 test: a genuinely stale profile still writes the snapshot); branch line 1948 → 2023 (`cli.py`) |
| I-0811-b | accepted | **accept** | identical (1 test: a transient auth failure says outage, not cookies); branch line 1891 → 1966 (`cli.py`) |

**Result:** all seven certificates are current at `f882411`.
