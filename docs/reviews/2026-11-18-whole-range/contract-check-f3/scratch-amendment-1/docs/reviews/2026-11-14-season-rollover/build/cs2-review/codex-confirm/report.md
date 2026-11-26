# Verdict

**BLOCK.** Reviewed only `a26958dc..ee9fb870`. The reported unpadded-date B1 is closed at all four sites, but replacing `fromisoformat` with `strptime` introduces a different production regression. No tracked files changed.

# B1

**Closed for the reported input.** `confirm_probe.py` ran the real shadow one-shot `main` (argument parser, config, state, strategy, provenance and artifact persistence) and real `run_result_polling` with remote/model/watchdog/posting boundaries stubbed. Dates were `2027-03-24` and `2027-3-24`; the matching-season fixture held streak 3.

| Entry / date | d8eb50d0 | a26958dc | ee9fb870 |
|---|---|---|---|
| Shadow / padded | artifact, completed | artifact, completed | artifact, completed |
| Shadow / unpadded | artifact, completed | no artifact, failed | artifact, completed |
| All void / padded | final, void saved | final, void saved | final, void saved |
| All void / unpadded | final, void saved | ValueError, unresolved | final, void saved |
| Final peer / padded | final | final | final |
| Final peer / unpadded | final | ValueError | final |
| Early peer / padded | final, reply streak 3 | final, reply streak 3 | final, reply streak 3 |
| Early peer / unpadded | final, reply streak 3 | ValueError, no reply | final, reply streak 3 |

Shadow exits 0 even when its analytics status is failed; artifact and persisted status were checked, not just exit codes. Unpadded day/both and the strptime-specific `2027-03- 4` also regain base behavior at all four sites.

# Blocking findings

**B2 — Newly rejects compact and week ISO dates in polling.** Input: `run_result_polling(111, "20270324", picks_dir)` with a delivered unresolved pick at `20270324.json`, streak tagged 2027, final game and all-void slot results. Both base and a26958dc return `final`, persist `result="void"`, and preserve streak bytes. HEAD raises `ValueError` at `src/bts/scheduler.py:36`, reached from line 2618, before saving the void: the pick remains unresolved. `"2027-W12-3"` reproduces identically. These are valid inputs to the replaced `date.fromisoformat` read, so this regression is new in the reviewed commit.

The same two inputs now raise at final-peer line 2613 and early-peer line 2569; the peer's terminal hit remains saved, but the caller fails and the early reply is lost. Evidence: `{base,r2,head}-probe.json`, cases `compact`/`week`, paths `void`/`final_peer`/`early_peer` (six regressions).

**Smallest fix:** keep `date_type.fromisoformat(date).year` as the first parser and fall back to `datetime.strptime(date, "%Y-%m-%d").year` on `ValueError`. Applied only to `sources/fixed/src/bts/scheduler.py`: `fixed-probe.json` confirms all six cases return final again, voids persist, early replies occur, and the unpadded fix remains intact. Leave `update_streak`'s contractually strict validation alone.

# Notes

- **Acceptance differs by entry point.** `date-matrix.md` and the raw JSON cover 22 inputs per path. strptime accepts padded/unpadded month/day and the space-prefixed day above; rejects surrounding whitespace, time suffixes, slash/dot/underscore separators, compact/week dates, invalid days, junk/empty strings, the tested Unicode month, date objects and None. Base shadow likewise rejects those tested strings/objects; run_once rejects date objects/None upstream. No base-rejected shadow string gained an artifact or scoring side effect; shadow failure-status writes already occurred at base.
- **Existing polling compatibility losses remain:** all three base display branches accepted every tested filename-shaped input, including whitespace/time/other separators, invalid strings, date objects and None. HEAD still rejects these (ValueError for strings, TypeError for objects), as a26958dc already did; compact/week rejection is the new B2 above. These pre-existing losses are NOTES in this narrow review, not additional blockers. No tested polling input was newly admitted relative to base. None of the display probes changed streak bytes.
- **Other season reads confirmed for unpadded input:** CLI void scoring raises ValueError at its pre-existing age parser on all three revisions, before resolution/season reads; strategy selection succeeds on all three and current fallback passes season 2027; update_streak succeeds at base but rejects before reads/writes at a26958dc/HEAD, deliberately required by row 3. None retains its documented dateless-update meaning. These modules were not edited here.
- **Clock pin is sound:** both CLI-void cases pass with the default production clock set to 2028; removing only the pin in a scratch test yields two real age-guard failures. With the pin retained, mutating only the CLI void read to dateless yields two failures displaying 13 instead of 0. The pin preserves the test's season assertion. The adopted review header now matches the corruption amendment; no other defect found in the diff.

# False greens

- Both changed test files pass: **122 passed**. The two parametrised scheduler tests against an isolated HEAD package containing only a26958dc's scheduler give **4 failed, 4 passed**: exactly the four unpadded cases fail. Thus all four new regression cases are genuinely red before the fix; the shadow test's selection stub is supplemented by the real one-shot artifact probe.
- The tightened season-type message removes the prior unsupported-keyword false green: current assertions against base give **3 failed**, each because `unexpected keyword argument 'season'` does not match the required message. Removing only `match=` in a scratch test gives **3 passed**. Current production passes all three. However, the scheduler test dates omit compact/week controls, so the 122-green result misses B2; add those controls with the fix (test omission is a NOTE).

# What I ran

- All commands used `TZ=America/New_York PYTHONDONTWRITEBYTECODE=1` and this checkout's `.venv/bin/python`; no network, box, contest account, whole fast suite, or pattern kill. Every pytest invocation ran through `~/bts-runs/yield_run.sh`; the sandbox rejected nice initially, so successful wrapper launches used approved escalation.
- `confirm_probe.py {base,r2,head,fixed}`: **154 events per source**, 616 total, plus direct strptime outcomes. `compare_probe.py` assertions all passed (`comparison.log`); complete entry-point outcomes, reader calls, replies and file changes are in the JSON and `date-matrix.md`. Base/r2/fixed packages and the scheduler-only swap are isolated under `sources/`.
- Pytest logs: `head-tests-escalated.log` (122 pass); `r2-scheduler-tests.log` (4 expected fail/4 pass); `clock2028-tests.log` (2 pass); `base-guard-tests.log` (3 expected fail); `base-old-guard-final.log` (3 pass); `clock2028-old-pin-verified.log` (2 expected age-guard fail); `clock2028-dateless-void-verified.log` (2 expected wrong-display fail).
- Earlier harness attempts had non-string state inspection, missing scratch temp parents/import paths, and a copied mutation namespace bypassing the pinned clock; retained diagnostic logs are excluded from the conclusions. Verified runs above corrected those issues. `git diff --check a26958dc HEAD` passed; final tracked status is clean.
