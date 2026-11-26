# Season-boundary sequences

Each branch stays on ONE directory. probe.py has eight branches (four opening results × CLI/polling), then three real crash paths and an overlapping two-thread scorer race. Each event in head-probe.jsonl and base-probe.jsonl records exact streak bytes in hex, the readable bytes, scorer output, health result, and real row-6..10 entry points on disk clones. Disk fixtures remain in runs/. Reader clones preserve the original bytes. Only feeds, prediction production, clock/sleep and external posting are stubbed; loaders/scorers/replay/scoring_lock/selection/diagnostics and persistence are real.

Tuple below: streak/saver/season/applied_dates. `-` means absent season.

| Event | Base | HEAD |
| --- | --- | --- |
| seq-hit-cli-last2026 | 13/False/-/['2026-12-31'] | 13/False/2026/['2026-12-31'] |
| seq-hit-cli-winter | 13/False/-/['2026-12-31'] | 13/False/2026/['2026-12-31'] |
| seq-hit-cli-opening | 14/False/-/['2026-12-31', '2027-03-24'] | 1/True/2027/['2027-03-24'] |
| seq-hit-cli-peer-retry | 14/False/-/['2026-12-31', '2027-03-24'] | 1/True/2027/['2027-03-24'] |
| seq-hit-cli-0200 | 1/True/-/['2026-12-31', '2027-03-24'] | 1/True/2027/['2027-03-24'] |
| seq-hit-cli-day2 | 2/True/-/['2026-12-31', '2027-03-24', '2027-03-25'] | 2/True/2027/['2027-03-24', '2027-03-25'] |
| seq-hit-poll-last2026 | 13/False/-/['2026-12-31'] | 13/False/2026/['2026-12-31'] |
| seq-hit-poll-winter | 13/False/-/['2026-12-31'] | 13/False/2026/['2026-12-31'] |
| seq-hit-poll-opening | 14/False/-/['2026-12-31', '2027-03-24'] | 1/True/2027/['2027-03-24'] |
| seq-hit-poll-peer-retry | 14/False/-/['2026-12-31', '2027-03-24'] | 1/True/2027/['2027-03-24'] |
| seq-hit-poll-0200 | 1/True/-/['2026-12-31', '2027-03-24'] | 1/True/2027/['2027-03-24'] |
| seq-hit-poll-day2 | 2/True/-/['2026-12-31', '2027-03-24', '2027-03-25'] | 2/True/2027/['2027-03-24', '2027-03-25'] |
| seq-miss-cli-last2026 | 13/False/-/['2026-12-31'] | 13/False/2026/['2026-12-31'] |
| seq-miss-cli-winter | 13/False/-/['2026-12-31'] | 13/False/2026/['2026-12-31'] |
| seq-miss-cli-opening | 0/False/-/['2026-12-31', '2027-03-24'] | 0/True/2027/['2027-03-24'] |
| seq-miss-cli-peer-retry | 0/False/-/['2026-12-31', '2027-03-24'] | 0/True/2027/['2027-03-24'] |
| seq-miss-cli-0200 | 0/True/-/['2026-12-31', '2027-03-24'] | 0/True/2027/['2027-03-24'] |
| seq-miss-cli-day2 | 1/True/-/['2026-12-31', '2027-03-24', '2027-03-25'] | 1/True/2027/['2027-03-24', '2027-03-25'] |
| seq-miss-poll-last2026 | 13/False/-/['2026-12-31'] | 13/False/2026/['2026-12-31'] |
| seq-miss-poll-winter | 13/False/-/['2026-12-31'] | 13/False/2026/['2026-12-31'] |
| seq-miss-poll-opening | 0/False/-/['2026-12-31', '2027-03-24'] | 0/True/2027/['2027-03-24'] |
| seq-miss-poll-peer-retry | 0/False/-/['2026-12-31', '2027-03-24'] | 0/True/2027/['2027-03-24'] |
| seq-miss-poll-0200 | 0/True/-/['2026-12-31', '2027-03-24'] | 0/True/2027/['2027-03-24'] |
| seq-miss-poll-day2 | 1/True/-/['2026-12-31', '2027-03-24', '2027-03-25'] | 1/True/2027/['2027-03-24', '2027-03-25'] |
| seq-void-cli-last2026 | 13/False/-/['2026-12-31'] | 13/False/2026/['2026-12-31'] |
| seq-void-cli-winter | 13/False/-/['2026-12-31'] | 13/False/2026/['2026-12-31'] |
| seq-void-cli-opening | 13/False/-/['2026-12-31'] | 13/False/2026/['2026-12-31'] |
| seq-void-cli-peer-retry | 13/False/-/['2026-12-31'] | 13/False/2026/['2026-12-31'] |
| seq-void-cli-0200 | 0/True/-/['2026-12-31'] | 0/True/2027/[] |
| seq-void-cli-day2 | 1/True/-/['2026-12-31', '2027-03-25'] | 1/True/2027/['2027-03-25'] |
| seq-void-poll-last2026 | 13/False/-/['2026-12-31'] | 13/False/2026/['2026-12-31'] |
| seq-void-poll-winter | 13/False/-/['2026-12-31'] | 13/False/2026/['2026-12-31'] |
| seq-void-poll-opening | 13/False/-/['2026-12-31'] | 13/False/2026/['2026-12-31'] |
| seq-void-poll-peer-retry | 13/False/-/['2026-12-31'] | 13/False/2026/['2026-12-31'] |
| seq-void-poll-0200 | 0/True/-/['2026-12-31'] | 0/True/2027/[] |
| seq-void-poll-day2 | 1/True/-/['2026-12-31', '2027-03-25'] | 1/True/2027/['2027-03-25'] |
| seq-double-cli-last2026 | 13/False/-/['2026-12-31'] | 13/False/2026/['2026-12-31'] |
| seq-double-cli-winter | 13/False/-/['2026-12-31'] | 13/False/2026/['2026-12-31'] |
| seq-double-cli-opening | 15/False/-/['2026-12-31', '2027-03-24'] | 2/True/2027/['2027-03-24'] |
| seq-double-cli-peer-retry | 15/False/-/['2026-12-31', '2027-03-24'] | 2/True/2027/['2027-03-24'] |
| seq-double-cli-0200 | 2/True/-/['2026-12-31', '2027-03-24'] | 2/True/2027/['2027-03-24'] |
| seq-double-cli-day2 | 3/True/-/['2026-12-31', '2027-03-24', '2027-03-25'] | 3/True/2027/['2027-03-24', '2027-03-25'] |
| seq-double-poll-last2026 | 13/False/-/['2026-12-31'] | 13/False/2026/['2026-12-31'] |
| seq-double-poll-winter | 13/False/-/['2026-12-31'] | 13/False/2026/['2026-12-31'] |
| seq-double-poll-opening | 15/False/-/['2026-12-31', '2027-03-24'] | 2/True/2027/['2027-03-24'] |
| seq-double-poll-peer-retry | 15/False/-/['2026-12-31', '2027-03-24'] | 2/True/2027/['2027-03-24'] |
| seq-double-poll-0200 | 2/True/-/['2026-12-31', '2027-03-24'] | 2/True/2027/['2027-03-24'] |
| seq-double-poll-day2 | 3/True/-/['2026-12-31', '2027-03-24', '2027-03-25'] | 3/True/2027/['2027-03-24', '2027-03-25'] |
| crash-cli-gap | 14/False/-/['2026-09-11', '2026-09-12', '2027-03-24'] | 1/True/2027/['2027-03-24'] |
| crash-cli-reconcile | 14/False/-/['2026-09-11', '2026-09-12', '2027-03-24'] | 1/True/2027/['2027-03-24'] |
| crash-cli-restart | 14/False/-/['2026-09-11', '2026-09-12', '2027-03-24'] | 1/True/2027/['2027-03-24'] |
| crash-poll-gap | 14/False/-/['2026-09-11', '2026-09-12', '2027-03-24'] | 1/True/2027/['2027-03-24'] |
| crash-poll-reconcile | 14/False/-/['2026-09-11', '2026-09-12', '2027-03-24'] | 1/True/2027/['2027-03-24'] |
| crash-poll-restart | 14/False/-/['2026-09-11', '2026-09-12', '2027-03-24'] | 1/True/2027/['2027-03-24'] |
| crash-early-gap | 14/False/-/['2026-09-11', '2026-09-12', '2027-03-24'] | 1/True/2027/['2027-03-24'] |
| crash-early-reconcile | 14/False/-/['2026-09-11', '2026-09-12', '2027-03-24'] | 1/True/2027/['2027-03-24'] |
| crash-early-restart | 14/False/-/['2026-09-11', '2026-09-12', '2027-03-24'] | 1/True/2027/['2027-03-24'] |
| race | 14/False/-/['2026-09-11', '2026-09-12', '2027-03-24'] | 1/True/2027/['2027-03-24'] |

No unintended loss, double count or old-year carry-over in these sequences. Opening void leaves the old bytes intact while all seasonal readers show fresh state; the 02:00 replay then stamps 2027. Opening hit/miss/double stamp immediately. Matching-season misses from 10 through 15 consume the saver; 9 and 16 reset. The dateless-between-dated sequence is 6,7,8 on both commits, with tag retained only on HEAD as required. Both tagless and tagged pp1 B1 marker guards preserve bytes. A stale-date override can still deliberately score out of chronological order; this is outside the requested chronology, explicitly warned by the existing CLI.
