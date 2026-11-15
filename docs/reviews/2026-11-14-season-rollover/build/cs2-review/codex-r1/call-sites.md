# Complete call-site census at 3b4196be

Generated search evidence: `calls-rg.txt`; searched both `src` and `scripts` with `rg`, then read each implementation. No calls or direct model-streak JSON access in `scripts`. Definitions/imports/comments are distinguishable in the saved search.

Opening-state example throughout: season 2026 (or no season), streak 13, saver false, only 2026 applied dates; contest date 2027-03-24. No inference from `updated` or pick files.

| Call (src/bts/) | Argument / opening behavior |
| --- | --- |
| scheduler.py:1899 load_streak | season=date_type.fromisoformat(date).year; shadow receives 0. |
| scheduler.py:2561 load_streak | scored date year; peer-scored early-hit reply receives 0 for a still-old file. |
| scheduler.py:2563 update_streak | date=date; internally derives year, all three reads and write use it. |
| scheduler.py:2605 load_streak | scored date year; final peer branch receives 0. |
| scheduler.py:2609 update_streak | date=date; internally scopes state. |
| scheduler.py:2610 load_streak | scored date year; all-void display receives 0, no streak write. |
| cli.py:2437 update_streak | date=date; internally scopes state. |
| cli.py:2438 load_streak | scored date year; all-void display receives 0, no streak write. |
| cli.py:2543,2546 load_streak (two calls) | season=now_et().year, using receipt.clock; both summary branches scoped to ET runtime year. |
| strategy.py:573 load_saver_available | selection date year when explicit saver is None; true on old/tagless files. |
| picks.py:593 load_applied_dates | season=season; [] for explicit 2027 versus old markers for None. |
| picks.py:653 load_streak | season derived from update date, or None when dateless; 0 or legacy 13 respectively. |
| picks.py:654 load_saver_available | same; true or legacy false respectively. |
| picks.py:655 load_applied_dates | same, only called for a dated update; [] on mismatch. |
| picks.py:660,665,668 save_streak (three calls) | season=season; dated update stamps 2027, dateless preserves the existing tag and saver/markers. |
| picks.py:707 load_applied_dates | deliberately NO season. On opening with only 2026 markers, today is absent. If a tagless file already contains today's full date, replay refuses an unresolved today, preserving crash idempotency. |
| picks.py:1372 save_streak | season=replay_day.year, fresh ET reading under lock; resets old markers, explicitly saves replayed saver. None replay saves nothing. |
| web.py:1089 load_streak | dashboard's different, local function; no season parameter, returns raw 13 only if decision-state loading failed. Error subtitle remains visible. Normal model-only dashboard state uses scoped diagnostics and is 0. |

Direct file accesses (including branches hidden behind these calls):

| File and lines | Operation / opening result |
| --- | --- |
| picks.py:519 | private seasonal JSON read; strict integer match, otherwise {}; catches OSError/ValueError and swallows corrupt bytes. |
| picks.py:533-536,545-548,557-566 | legacy JSON reads without season; return 13/false/old dates; seasonal branches call private reader. Applied-date loader retains its broad legacy exception tolerance. |
| picks.py:578-606 | seasonal existing-state read through helper; dateless existing-object/tag reads at 584/589; atomic JSON write at 606. Default None is reachable from tests/API, no direct dateless production caller. |
| contest_state.py:315-326 | direct diagnostics read, now_year from caller; stale/tagless object returns (0,true,None); corruption stays a warning/error. |
| web.py:115-118 | local direct fallback read returns 13, unchanged. Corrupt contest-account JSON with an otherwise valid stale model file can reach it. Display only; not a pick decision. |
| health/streak_validation.py:29-64 | no season check; stale but structurally valid 13/false gets no alert. No new-key breakage; cannot establish freshness. |
| state/export.py:49-69 | reads raw values; exports 13/false and historical picks, does not export season. Unchanged. |
| state/verify.py:65-82 | reads live and regenerated JSON, compares only streak/saver; identical values with a dropped/wrong tag report clean. Unchanged. |
| state/regenerate.py:387-395 | writes raw numeric/saver fields without a season; opening fallback reads that tagless file as 0/true. Recovering CURRENT-season state also loses its season tag, so fallback/next dated scorer reset until successful reconcile. Material recovery follow-up, outside this diff. |
| data/backup.py:327-334 | restore verification finds streak.json recursively and checks JSON parsing; ignores season. No rewrite; ordinary backup preserves original bytes. Reports no streak-specific issue for the tagged fixture. |

All six row-12 modules remain byte-identical to base. The tag does not break parsing/validation/export. The regeneration/verification semantic gap and stale error-path dashboard fallback matter operationally, and are NOTES because their code is deliberately unchanged.

Production date types: cli.py:2194-2208 forwards Click's --date string or _today_et() string to run_day; scheduler.py:3273 forwards that string to polling. check-results has a required Click --date string (cli.py:2213). Shadow's one-shot argparse --date also defaults to today_et() and passes a string (scripts/shadow_predict_once.py:27,71). No production update_streak caller passes datetime.date or a time-bearing string. Malformed supplied CLI strings generally fail upstream age/date handling; the date-object TypeError in the new regex check is an API robustness note.
