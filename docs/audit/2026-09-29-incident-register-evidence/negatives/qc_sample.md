# Seeded 10% QC of the non-sensitive runtime-closure negatives (plan rev 3 Task 7)

Population: 196 negatives with sensitive_path = no and a non-fix exclusion reason (N1 + N2 reviews). Sample: 20, random.Random(20260929).sample. Each checked by the lead against its stat, subject/body and, where the subject could hide a live fix, its diff.

| hash | date | reader's reason | lead check |
|---|---|---|---|
| 039641d | 2026-04-10 | preship_fix | confirmed: belt-and-suspenders dtype coercion; its companion 151bfe4 documents a dtype crash on the Fly SHADOW host during the 4/10 migration (not the production path; infra_migration) |
| 498c5a8 | 2026-04-24 | feature | confirmed |
| d292614 | 2026-07-07 | preship_fix | confirmed: pre-merge review hardening of the unshipped park_drag shadow feature (deployed with it) |
| ab2ba2e | 2026-04-07 | feature | confirmed |
| 5a3a7ec | 2026-05-30 | research | confirmed |
| 1b28720 | 2026-04-12 | docs_or_config_text | confirmed |
| 7101025 | 2026-03-31 | feature | confirmed |
| d16bae4 | 2026-04-10 | feature | confirmed |
| 08cf91e | 2026-04-03 | preship_fix | confirmed |
| 1c1069e | 2026-05-04 | research | confirmed |
| c78f44b | 2026-04-09 | feature | confirmed |
| e1ebde9 | 2026-04-15 | feature | confirmed: deliberate policy upgrade (24-seed pooled policy v2) on A/B evidence, not a fix |
| e17eebf | 2026-03-29 | research | confirmed |
| 9a89f5a | 2026-04-24 | feature | confirmed |
| 23051b7 | 2026-03-29 | feature | confirmed |
| 44fdc84 | 2026-04-08 | feature | confirmed |
| a98f1ad | 2026-04-24 | feature | confirmed |
| 82c2ae6 | 2026-05-01 | research | confirmed |
| ddfb163 | 2026-05-03 | research | confirmed |
| 975fdc4 | 2026-04-10 | feature | confirmed |

Result: 20/20 exclusions confirmed; no candidate added.
