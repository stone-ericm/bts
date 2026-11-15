# Verdict

**SIGN** at `60cece20`, reviewing only `acea53f8..HEAD`. B2 is closed. No unintended production behavior change was demonstrated. No tracked file was changed during this review.

# B2

**Closed**, established with real files on Python 3.12.16 (integer limit 4,300; recursion limit 1,000), not injected decoder exceptions. `probe.py` independently exercises all four changed handlers: `src/bts/contest_state.py:143,234,280,322`.

- 18 model cases: 4,301-digit scalar, object containing that integer, and 200,000 nested arrays; manual/automatic/no observation; both `require_contest_state` values. Usable observations equal the healthy-file baseline in every decision field except diagnostics: both become `None`, one warning is logged, and an error description is supplied. Without an observation, optional reads raise model-corrupt `ContestStateError`; required reads raise missing-contest `ContestStateError`. Files remain byte-identical.
- 12 pick cases: the same three bodies, manual/automatic observations, both flags. Both scans independently skip the bad file, retain another settled pick, and distinguish healthy hit/miss outcomes (`lagged`/`stale`). The miss-negative check cannot short-circuit before the corrupt file. Both pre-fix scans independently raise the decoder exception.
- 36 contest cases: both filenames, all three bodies, both flags, alone and with relevant competing observations. Read failures become filename-bearing `ContestStateError` with the actual decoder exception as cause. Active manual overrides still win before corrupt automatic parsing; expired/legacy overrides do not rescue it; corrupt manual still blocks healthy automatic state. Direct `_parse_state_file` reads were checked too.

Evidence: `probe.log`, `probe-results.json`; all 66 cases passed. The old module was loaded solely from a scratch package.

# Blocking findings

None in the scoped delta.

# Notes

**N1 — Broad-catch audit: no unintended swallowing found.** The contest parser and both scan `try` bodies contain only `json.loads(path.read_text(...))`. The model body's additional operations are `Path.exists`, the object check, the existing deliberate `TypeError`, and builtin dictionary `.get` calls. Default JSON decoding returns builtin dictionaries; none of our additional operations introduces a `ValueError`/`RecursionError` for file contents. Schema/date/value validation remains outside these widened handlers. In 2,260 before/after comparisons of ordinary, malformed and schema-invalid inputs (including nonfinite numbers and wrong field types), full results and error messages were identical. AST bodies and comparison counts are preserved in `probe-results.json`.

**N2 — Remaining read/decode boundaries, all reproduced and unchanged from the parent where compared.** These are notes about existing behavior, not regressions introduced by this fix:

| Concrete input | Observed outcome on this Python | Contract relation |
|---|---|---|
| Raw NUL in a JSON string; UTF-8 BOM before `{}` | `JSONDecodeError`; pick skipped, model isolated, contest domain error | Corrupt-JSON behavior in rows 1, 2, 5–7 |
| Escaped `\u0000` inside an otherwise valid object | Decodes normally; defaults/valid observation preserved | No corruption; row 8 defaults remain unchanged |
| Directory or chmod-000 file at each of the four filenames | `IsADirectoryError`/`PermissionError` caught; skip/isolate/domain error | OSError behavior of rows 1, 5–7; existing contest error handling preserved |
| `1e999999`, `-1e999999`, `1e-999999`, 100,000-digit exponent/fraction, 4,301-digit coefficient followed by `.0` | Decode to `inf`, `-inf`, `0.0`, or `1.0`; **no OverflowError reproduced**. Nonobject handling remains intact | Rows 1, 5–7 cover nonobjects; contest retains object/type validation |
| 8 MiB JSON string; valid contest object containing that string | String decodes then takes nonobject handling; object remains accepted | Rows 1, 5–7 cover the scalar; no contract size ceiling for the valid object |
| Sparse file of 1 PiB, beginning with `"` (4 KiB physically allocated, removed after probe) | Real `read_text` **MemoryError escapes** at every read site on both versions, including beside a usable observation | Resource exhaustion is not an explicitly enumerated exception policy; an unbounded reading of the corrupt-file promise would still have this pre-existing gap |
| FIFO with no writer at each filename | Both versions block; each probe child was individually killed/reaped after one second. Fed FIFOs read normally; integer corruption takes the new handler | Rows specify read-error outcomes, with no timeout/nonregular-file policy |
| Self-referential symlink at each filename | Direct read raises ELOOP; pick scans skip it. `exists()` returns False, so model/contest prechecks treat it as missing on both versions | Pick behavior satisfies row 1. Existing model precheck conflicts with rows 5/6's corrupt-state intent: diagnostics become `0/True/None`, optional fallback invents 0. Required missing-state precedence still matches row 7 |

For the symlink model case, usable manual and automatic observations and both flags were additionally run: observations remain usable but get healthy-default diagnostics, identically before/after. This is outside the delta's behavioral changes (`contest_state.py:206,215,316`). Memory exhaustion and FIFO blocking likewise predate the change; this fix does not establish universal recovery from arbitrary filesystem objects/resources.

**N3 — Recursion recovery is safe for the tested decoder failure.** The exact `DEEP` bytes imported from the new test raise `RecursionError` directly, not `JSONDecodeError` or the nonobject check. After each of 100 corrupt-then-healthy cycles in one process, the next real observation read and a normal JSON decode succeed. The default decoder object and recursion limit remain unchanged. All read sites and both flags also passed the independent deep-body matrix. This does not claim recovery from arbitrary caller stack exhaustion.

# False greens

All **12 new parametrized nodes fail** against `acea53f8`'s loader in `scratch-acea53f8/bts`; all pass HEAD. Eleven fail with the intended plain integer-conversion `ValueError`; the deep-body node fails with decoder `RecursionError`. The old package's imported path is printed in `pytest-old-priority.log`. No skips or falsely green nodes occurred.

**Test NOTE:** `test_deeply_nested_body_is_isolated_everywhere` (`tests/test_contest_state_decoder_limits.py:93`) does not exercise the automatic contest filename, and its miss assertion can short-circuit on the healthy miss depending on directory order. Its first pre-fix failure also prevents later assertions from running. This limits that individual test's coverage, not the demonstrated behavior: the independent matrix checks automatic reads and miss-negative scans. Explicit cause assertions and splitting the combined test would make the regression evidence clearer.

# What I ran

- All Python commands used `TZ=America/New_York PYTHONDONTWRITEBYTECODE=1`; no network, account, box, or full fast suite.
- Every pytest invocation used `~/bts-runs/yield_run.sh .codex-review/cs1-confirm/<name>.log ...`; cache disabled, basetemps inside this review directory. Initial sandbox runs could not set niceness; the required low-priority runs were repeated with approved escalation.
- HEAD: `.venv/bin/python -m pytest` on `test_contest_state.py`, `test_contest_state_best_trust.py`, `test_contest_state_corrupt_precedence.py`, `test_contest_state_defensive_reads.py`, `test_contest_state_cs1_review.py`, `test_contest_state_decoder_limits.py` under `tests/`: **112 passed**, `pytest-head-priority.log`.
- Parent: identical new test file with `PYTHONPATH=.codex-review/cs1-confirm/scratch-acea53f8`: **12 failed**, `pytest-old-priority.log`, final wrapper exit 1. Repeated after the wrapper's first studio pause left an anomalous exit status; complete pytest summaries are retained.
- Direct scripts: `probe.py` (66 corruption/precedence cases, 2,260 comparisons, 100 recovery cycles, filesystem/encoding/numeric/string probes); `edge_probe.py` (four real MemoryError sites, four no-writer FIFO sites, fed FIFOs); `supplemental_probe.py` (exact test payload and six symlink/model cases). Corresponding `*-results.json` and logs are in this directory. Resource-limit feasibility attempts are in `resource-experiment.log`; MemoryError evidence uses real sparse-file reads instead.
- `git diff acea53f8 HEAD`, complete changed source/test inspection, and final `git diff --exit-code` / `git diff --cached --exit-code`: tracked tree unchanged.
