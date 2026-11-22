# Reviews of the R52 comparison tool (`scripts/audit/bullpen_r52_compare.py`), tier C: the one tool that decides a result

| Round | Report | Marker (checked) | Verdict |
|---|---|---|---|
| 1 | `report-r1.md` | `R52REV-6a640abd` | BLOCK (fixed grid, exact floors, completion bound to files, cell validation, publication, tests) |
| 2 | `report-r2.md` | `R52REV2-807bb7d1` | BLOCK (replaced inputs, JSON over evidence, invalid types exiting 1) |
| Follow-up (R57, fresh reviewer) | `report-followup.md` | `R52FU-821af121` | BLOCK (in-window substitutions, malformed log records, impossible stamp orders, symlinked root and JSON paths) |

Ran: each reviewer ran the tool's test file and synthetic probes at nice 19 (scripts kept here); none read, listed or computed anything from the real outputs. Did not run: the real comparison, the backtests, the wrapper, the full suite.
Outcome: Eric accepted the follow-up's remaining findings as known limits on 2026-11-22 ("1"); the tool is run once on the untouched outputs (FINDINGS, same date).
