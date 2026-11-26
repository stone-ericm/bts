# Current-defence specs: drafting guide (W1.5 Phase 1, Task 6)

A current-defence certificate (design §9.3; `scripts/audit/incident_register/defence.py`, `certify.py`) shows that TODAY's tests defend a fixed link. The certificate needs:
- a small semantic mutant that restores the pre-fix decision in the CURRENT code;
- a run of the current tests under a trusted in-process observer. Some declared killing node must fail at its declared assertion with the builtin `AssertionError`.
- an observed positive witness:
  - **event**: a call of a declared boundary with the declared bad category;
  - **return**: a return of the declared function with the declared bad category or value.

  The witness must be linked to the mutated line through a live invocation of the declared production entry. A missing event (absence) is NOT certifiable in Phase 1 (plan ruling 10). Such a link is `unavailable` without a run.

The tooling is frozen at **`f453283`**, and specs run against that pin. `baseline` is the placeholder `"BASELINE"`.

## Worked examples
`specs/I-0830-a.json` (event kind) and `specs/I-0813-a.json` (return kind), with the README's phase-1 table. Read them before drafting.

## Spec fields
- `label` (`D-I<NNN>-<link>`), `link` (`"I-NNN link K"`).
- `tests`: pytest args for the current test file(s) that hold the contract, ending in `"-q"`.
- `allowed_paths`: the `src/bts/*.py` files the mutant may edit. The hard rule: only existing tracked `src/bts/**.py`.
- `mutation_edits`: a list of `[path, old, new]`.
  - `old` must occur exactly once in that file at `f453283` (check with `git show f453283:<path> | grep -cF`). For multi-line text, check it by reading the file.
  - The mutant restores the PRE-FIX decision with the smallest change: usually it disables the guard the fix added, or reverts its condition. It must not add new behaviour. Name the fix commit in a comment.
- `branch`: `{"path", "text"}` for a line INSIDE one of the replacements, which executes on the defective path. The convention is a marker line, `pass  # W1.5 mutant: pre-<fix sha> — <what the mutant restores>`, placed in `new` where the old guard was. `text` must be unique in the mutated file. Since `f453283` the runner refuses an anchor outside the replaced lines.
- `entry`: `{"path", "qualname"}`, the production function whose live invocation links the mutated line and the witness. The killing test calls it directly, or calls into it.
- **For an event:**
  - `boundaries`: `[{"name", "binding": "module:attr", "value": ["args[i]", "kw:name"], "classify": [[category, regex], ...]}]`;
  - `symptom`: `{"kind": "event", "boundary": name, "category": bad category}`.

  The binding is the attribute the code looks up at the call (for a mock, the name the test patches). `value` accessors pick the argument to classify. The first matching regex wins, and the regex runs on the argument's serialized text.
- **For a return:**
  - `returns`: `[{"path", "qualname", "classify": [[category, regex on the serialized return]]}]`;
  - `symptom`: `{"kind": "return", "path", "qualname", "category"}`, or a `"value"` (a JSON value, compared type-exactly).
- `killing`: `[{"node", "assertion": {"path", "text", "line"?}}]`: current tests that pass at the pin and fail under the mutant at that assertion. The assertion must be the contract assertion. Give `line` when `text` is not unique in the file.
- `contract`: `{"text", "entry_mode", "trigger", "symptom_kind"}`.
- `level`: `"production_path"` (only external I/O is replaced) or `"component"` (internal collaborators are mocked too), plus `level_justification`.
- `allowed_mocks`: everything the killing tests patch on the path.
- `baseline`: `"BASELINE"`; `env`: `{"TZ": "America/New_York"}`.

## Static checks to do (no test runs)
1. Every `old` text is unique at `f453283`. Every `new` text compiles in context (indentation).
2. The branch marker is inside a `new` text, is unique in the file after the edit, and lies on the path the killing test drives.
3. Trace the killing test at the pin, then under the mutant. It passes at the pin. Under the mutant it fails at the declared assertion with `AssertionError`, not with a TypeError, or with an earlier assertion that is not the contract's.
4. The witness happens after the branch line, inside the same live entry invocation: the bad boundary call or the bad return.
5. **Closure (plan ruling 13):** note any killing-path decision that depends on elapsed real time, object lifetimes, interpreter instrumentation or object addresses. Fixed fixture clocks, prewritten files and ordinary identity are fine inputs. If there is such a dependence, say so; the link may need `unavailable`.

## No spec (write a manifest entry with a reason instead)
- **absence:** the defect's observable effect is a MISSING event, such as no alert sent or no pick delivered. Reason: `"absence not certifiable by the Phase 1 recorder"`. Name the regression test that pins it, if one exists.
- **outside src:** the fixed decision is in a workflow, cron line, script or unit: `"the defended decision is outside src/"`.
- **no current defence:** no current test exercises the fixed decision; give the evidence (the grep you ran). This is a coverage finding and is worth knowing.
- **the fix was superseded or removed:** say by what.

## Constraints for drafters
- Read-only git in `/Users/eric/projects/bts-w15`: `show`, `diff`, `log`, `grep` at `f453283` or at the fix commits. Never checkout, reset or stash.
- No test runs: no pytest, no `uv run`. The strict mutation sweep is using the machine, and defence runs come after it.
- No `data/` reads, network, ssh or `gh`.
- Write only to your assigned staging directory: specs, plus `manifest.json` with one entry per link (spec label, or `null` with the reason), each with `confidence` and `open_questions`.
