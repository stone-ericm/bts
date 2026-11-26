# Contract (draft 1, NOT approved to build): F5 producer, a park-drag refresh over an empty current-season store

Status: written 2026-11-18 by the fresh session after three BLOCKs (`STOP-AUDIT-F4F8.md`). It is for a contract check, then a decision by the lead and Eric between **building this** and **leaving the producer as on `main`** (checklist row B7: reseed after the first 2027 game). Nothing from branch `sb-audit-f4f8` is reused. Tier B. The park-drag table feeds the shadow model only; no production pick reads it. The freshness health check (`src/bts/health/park_drag_freshness.py`) is NOT touched (ruling R39).

## The defect on `main`
`park_drag_producer.refresh` takes the season from the newest date in `producer/pitches_current.csv.gz`. A store with a header and no rows (a reseed made before the season's first game) makes that `int(NaN)`: the run fails (`ok=false`, a `WARN`) every day, and it can never fetch its way out, because the failure comes before the fetch. So the reseed has to be done by hand after the first game.

## What changed from the blocked attempt
The blocked commit let an empty refresh SUCCEED (`ok=true`, zero rows), and the third review showed that this cleared a real failure warning beside a stale table. Here **an empty store never succeeds.** The only new behaviour is that an empty store may fetch, and if the fetch returns pitches the run continues as a normal refresh. Every state that fails on `main` and ends with the store still empty fails here too.

## The rule
"Empty" means the store file parses and has zero rows. A store with rows behaves byte-for-byte as on `main` (rule 0). For an empty store, in this order, every failure being a `ProducerError` (`ok=false`, its message in `error`, and no file written or changed except `producer_status.json`):
1. **Before any network call:** if `park_drag_export.csv` or `park_drag_manifest.json` exists (anything at either path), the manifest must be a readable JSON object whose `max_source_game_date` parses with `date.fromisoformat` to a year BELOW `today.year`. Otherwise fail: the table already holds this season (or cannot be read), so an empty store means lost data and needs a reseed, not a refetch.
2. The season is `today.year`; the window is March 1 to yesterday. An empty window (today on or before March 1) fails: nothing to fetch yet.
3. One fetch of that window, through the existing `fetch_savant_ff` with all its guards (the 403/429 stop, the row-cap guard, the HTML and column checks). Zero rows fails: "no pitches yet for the season; existing table kept".
4. Rows: the store becomes exactly the fetched rows, is written, and the run continues through the unchanged path (meta, game level, export, manifest, `ok=true`).

## The table: states that must still fail or still alarm
| # | State (store empty unless said) | Required outcome |
|---|---|---|
| P1 | **July 15, 2027: static store header-only, Savant returns a header-only reply, an export and a manifest with source through July 14** (the third review's finding) | fails at step 1 with NO network call; `ok=false`; export, manifest and stores byte-unchanged; the health check gives `WARN` "last run FAILED" |
| P2 | Only the export exists; only the manifest exists; the manifest is not an object, has no date, a date that does not parse, or is unreadable; a directory at either path | fails at step 1, no network call |
| P3 | Last season's table present (source through 2026-09-27), any day from March 2, Savant has no pitches yet | fails at step 3; table unchanged; `WARN` daily (and the freshness `CRITICAL` that R39 already expects) |
| P4 | A clean seed: no export, no manifest, nothing fetched (whatever the static store holds) | fails at step 3. NOT a success: the reviewer allowed this one case to succeed empty; this contract does not, so no empty state succeeds |
| P5 | January, February or March 1 | fails at step 2, no network call |
| P6 | The fetch is refused (403 or 429), returns HTML, lacks columns, or is at the row cap (a whole season requested in mid-year) | fails exactly as that failure does for a populated store; `rate_limited` set as today; the store stays empty |
| P7 | The fetch returns pitches but the game-level step yields no game (too few qualified pitches) | the store is written; the rest is what `main` does for a populated store with such a day (the export is rebuilt from the static store; the freshness check alarms on its old source date) |
| P8 | A store WITH rows from last season, March onward | "store season 2026 is behind 2027", as on `main` (rule 0) |
| P9 | A store with rows but no parseable date; a missing input file; a store that does not parse | as on `main` |
| P10 | Any populated store, any day: fetched or not, new games or none | summary, store, meta, export and manifest byte-identical to `main` for the same inputs and clock |

Must SUCCEED (the reason for the change): **S1** last season's table present, empty store, the morning after the first game, Savant returns that game: the store holds exactly those rows, meta is fetched for the new games, the export and manifest are rebuilt with the new source date, `ok=true`, and the next day's run is an ordinary populated refresh. **S2** the same from a clean seed.

## Limits stated now
1. Between the reseed and the first fetched game the run fails every day with a `WARN`: as on `main`, with a clearer message. This contract removes no alarm.
2. Savant's index lags a night game: the first success can come a day late.
3. A reply that is short without reaching the row cap is trusted, as for every daily window on `main`.
4. After S1 the store holds only what one request returned. A reseed made late (weeks into the season) can hit the row cap and then needs a populated seed, as today.

## Tests an author must deliver
One test per row P1 to P10 and S1, S2 through the real `refresh` with an injected `http_get` that fails the test if called where the row says "no network call"; P1 to P5 RED against a build that returns `ok=true` for any still-empty store; P10 as a byte comparison against `main` on the existing populated fixtures; the real `park_drag_freshness.check` run on the P1 and P3 results.

## The alternative: leave as on `main`
Keep checklist row B7 (reseed by hand after the first 2027 game, with a populated store). Cost if wrong: the shadow model's park-drag feature is empty for the days until someone reseeds; no effect on any pick.
