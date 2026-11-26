# Verdict

**SIGN** `d67a25e7`.

## Blocking findings

None. The round 1 false off-day is fixed. The offline probes found no other realistic malformed reply that passed as a clean off-day.

## Notes

Round 1's findings stand outside this confirmation. Upstream plausibility is based on repository fixtures and offline audit records. Those records do not establish whether MLB has ever emitted duplicate keys or malformed encodings. [Round 1](round1-report.md) · [MLB schedule summary](../../docs/audit/2026-11-14-cold-start-rehearsal/schedule-2027-03-20_27.txt) · [July incident audit](../../docs/audit/2026-07-12-eve-of-break-restart-loop.md)

## 1. Non-JSON constants and callers

- `NaN`, `Infinity` and `-Infinity` each raised `ValueError: Invalid MLB schedule JSON: non-JSON constant <token>` as a root value, in a root field and nested in a game that regular-season filtering would otherwise discard. Base and round 1 both returned `[]` for the root-field and filtered-game cases.
- For each token in `totalGames` with `dates: []`, `bts schedule` exited 1 after one request, before the off-day message or idle call. `bts day-outcome` exited 1 with `schedule_unverifiable` and one captured alert DM. Both `{"dates":[]}` and `{"dates":[],"totalGames":0}` idled normally and returned `no_games` with exit 0 and no DM.
- Quoted spellings parsed normally. A hydrated-shape slate returned the same regular-season game on all three revisions, with its strings, integers, floats, exponents, negatives and 30-digit integer intact. [Probe output](probe-confirmed.log)

## 2. Other response bodies

| Body | Base `9e0ec28a` | Head `d67a25e7` | Upstream assessment |
|---|---|---|---|
| Duplicate `dates` or `games` | Last value wins. Empty last returns `[]`; populated `dates` last returns `[{}]`. | Same | Constructed ambiguity. No occurrence found in available MLB records. |
| UTF-8 BOM before an empty slate | `[]` | `[]` | Plausible encoding quirk. The decoded body is a valid off-day. |
| UTF-16 or UTF-32 empty slate, with and without BOM | `[]` | `[]` | Constructed encoding variants that decode to a valid off-day. |
| Windows-1252 byte in a string | `UnicodeDecodeError` | `ValueError` with the schedule JSON prefix | Plausible encoding fault. Both reject it. |
| Raw UTF-8 surrogate bytes in ignored metadata | `[]` | `[]` | Constructed malformed encoding accepted by Python's byte decoder. No evidence of it in MLB responses. |
| Trailing object or text after the schedule | `JSONDecodeError` (`Extra data`) | `ValueError` with the schedule JSON prefix | Plausible corrupted reply. Both reject it. Ordinary trailing whitespace is accepted by both. |
| HTML error page containing `{"dates":[]}` | `JSONDecodeError` | `ValueError` with the schedule JSON prefix | Plausible proxy or upstream error page. Both reject the body. |
| Empty `dates` with a positive count and an `error` field | `[]` | `[]` | Constructed contradiction in ignored metadata. No evidence MLB emits this combination. |
| Empty `dates` with `totalGames: 1e9999` | `[]` | `[]` | Legal JSON number syntax with a constructed count outside realistic range. |

## 3. Final fallback

The real `run_day` reached the post-loop final fallback at 18:35 ET with a fresh selection and a 19:05 cutoff. Changing only the dated best streak after selection made the real delivery guard refuse. The saved pick had no attempt or confirmation, the scheduler stayed unlocked, and no decision was recorded. One deduplicated `contest_state` DM and the later `missed_pick` DM were captured. With the final observation matching, the same path sent one pick DM, marked the attempt and confirmation, locked and recorded `delivered`. [Run trace](probe-confirmed.log)

The missing final-fallback regression remains a test gap. Round 1's two CLI catch-up mutation gaps remain recorded for follow-up. Both locked catch-up paths already lacked a fresh decision to exempt.

## 4. Scope

`git diff 9cb84536 HEAD --stat` lists only `src/bts/scheduler.py` and `tests/test_wr_a_schedule.py` (19 insertions, one deletion). The base-to-head scheduler diff leaves A3's `run_single_check` return and overrun reuse logic, and A4's loop wake logic, byte-identical. No tracked files changed during review.

## What I ran

- `probe.py` replayed identical bytes through `fetch_schedule` from archived `9e0ec28a` and `9cb84536` and the current checkout, then drove both Click commands and the full-day final fallback. The first invocation stopped at the head import-path assertion because `PYTHONPATH` was relative. The unchanged probe completed with an absolute path. [Complete run](probe-confirmed.log)
- `~/bts-runs/yield_run.sh .codex-review/wr-a-r2/pytest-schedule.log env TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/test_wr_a_schedule.py -q --tb=short` passed all 24 tests. Also ran the commit and scope diffs plus `git diff --check 9cb84536 HEAD`. All probes used local fixtures and captured transports. [Pytest output](pytest-schedule.log)
