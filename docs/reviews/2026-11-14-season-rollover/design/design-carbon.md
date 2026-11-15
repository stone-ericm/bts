# Season-Boundary Streak & Saver Design (`design-carbon`)

**Disclosures.**
- **Read outside the task's allowed set:** None. Only files inside `/Users/eric/projects/bts-design-sr` were read.
- **Runtime-injected context:** Global instructions `~/.gemini/AGENTS.md` (`RULE[user_global]`), built-in Antigravity CLI skill headers, and one mid-session context-compaction summary of my own earlier reads/probes in this conversation.
- **Verification probes run:** Executed `.venv/bin/python` scripts in `/Users/eric/.gemini/antigravity-cli/brain/e52d46ee-70b3-4f1e-b11b-047345d2a3b6/scratch/{probe_design.py,verify_details.py}` against `main` (`src/bts/*`, `tests/*`, `data/models/mdp_policy.npz`, `data/models/mdp_tail_policy.npz`). Tracked files untouched (`git diff` empty).

---

## 1. State Space

Let `target_year` be the ET contest year under decision: `date.fromisoformat(target_date).year` when a caller supplies `target_date` (`scheduler`, `orchestrator`, `run_and_pick`, `cli.run`, `cli.preview`, `health/runner`), else `now.astimezone(ET).year`. Every decision day is a tuple across six observable axes:

1. **Calendar & clock (`C`):** `target_date` vs MLB's `target_year` opener (`2027-03-24`, 1 game; `2027-03-25`, full slate); `now` (UTC vs ET around Dec 31/Jan 1); host NTP skew.
2. **Manual override (`M`, `account_state/contest_streak.manual.json`):** `M0` absent; `M1` unexpired (`override_expires_at > now`) with `source_date.year == target_year`; `M2` unexpired with `source_date is None` or `source_date.year != target_year`; `M3a` explicitly expired (`override_expires_at <= now`); `M3b` legacy (`override_expires_at is None`) with `source_date.year == target_year`; `M4` legacy (`override_expires_at is None`) from `< target_year`; `M5` malformed JSON / invalid UTF-8 / non-object / bool or negative streak.
3. **Auto fetch file (`A`, `account_state/contest_streak.json`) & last attempt (`F`, `health_state/contest_streak_fetch_status.json`):**
   - `F` records `last_success_at`, `last_error`, `last_error_at`, `last_error_category` (on `main`, `cli.py:1602` overwrites `F` on error without preserving `last_success_at`).
   - `A0` absent; `A1` `current_settled` (`>=1` settled prediction maps to a `target_year` round in `rounds.json`); `A2` `empty_reset` (`target_year` in `rounds.json`, 0 settled predictions, `active=0, best=0`); `A3a` `counter_only_unsettled` (`target_year` in `rounds.json`, 0 settled predictions, `active>0` or `best>0`) chained from a prior `proven_season == target_year` file with `today_et >= earliest_round_date`; `A3b` `counter_only_unsettled` without prior `target_year` anchor (or `today_et < earliest_round_date`); `A4a` `unmapped_settled` (`>0` settled predictions, 0 match `rounds.json`) with prior `last_proven_source_date` in `target_year`; `A4b` `unmapped_settled` without prior `target_year` settled proof; `A5` `prior_settled_only` (`source_date.year < target_year`, or `rounds.json` lists only `< target_year`); `A6` legacy `v1` file (`provenance` absent: subcases `source_date` in `target_year`, `source_date` in `< target_year`, `source_date=None` with `recorded_at` in `< target_year`, or bare fixture with both `None`); `A7` future `source_date > today_et + 1d`; `A8` malformed.
4. **Saver flag (`V`, `account_state/saver_state.json`):** `V0` `uninitialized` (missing, corrupt, or `season != target_year`); `V1` `not_earned` (`season == target_year`); `V2` `active` (`season == target_year`); `V3` `used` (`season == target_year`).
5. **Local replay & pick files (`L`, `streak.json` + root `YYYY-MM-DD.json`):** `L0` no `target_year` settled picks (`2026-*.json` may exist); `L1` `>=1` `target_year` settled picks (subcases: `gap=0`, `gap=1`, `gap>=2`, unconfirmed miss); `L2` corrupt / invalid-UTF-8 (`0xff`) / non-dict (`[]`) root `.json` file; `L3` `streak.json` still carrying `2026` streak before `02:00` ET `reconcile_results`.
6. **Caller mode (`R`):** `require_contest_state=True` (live pick selection) vs `False` (`preview`, research, fallback).

| ID | State summary (observables) | Reachable? & why |
|---|---|---|
| **S1a** | `M1` unexpired `target_year` override (`streak=0, best=0`), any `A` (incl. `A4b`/`A5`/`A8`), `V1` | **Yes** — normal Opening Night/Day 2 when operator follows checklist B5. |
| **S1b** | `M1` unexpired `streak=0` override left active after Day 1 hit while `A1`/`A3a` already shows `streak>=1` | **Yes** — operator sets multi-day TTL on Opening Night and Judge hits. |
| **S2** | `M0`/`M4`, `A2` (`empty_reset`: `0/0`, `predictions=[]`, `2027` in `rounds.json`), `V1`, `L0` or `L1` | **Yes** — MLB resets profile + `rounds.json` before Opening Night (or user skips opening days). |
| **S3** | `M0`, `A3a` (`counter_only_unsettled`: `active>=1`, `predictions=[]`, chained from `S2`/`S6`), `V1`, `L1` | **Yes** — Day 2 (`2027-03-25`): `predictions[]` lags `activeStreak` by 1 settled round (`contest_fetch.py:215`). |
| **S4** | `M0` (override forgotten/expired), `A4b` (`unmapped_settled`: 2026 `predictions[]` vs 2027 `rounds.json`, `active=14`), Day 1 or Day 2 (`L0` or `L1`) | **Yes** — profile not reset by Day 2 while `rounds.json` is 2027 (Audit F2/F3; Carbon R2 B2). |
| **S5** | `M0`, `A5` (`prior_settled_only`: first 2027 fetch sees `rounds.json` still in 2026, `source_date=2026-09-26` or `None`), Day 1 or Day 2 | **Yes** — `rounds.json` rolls over later than pre-season fetch or lags through Day 2 (Carbon R2 B2). |
| **S6** | `M0`, `A1` (`current_settled`: `source_date=2027-05-31`, `active=13`), `V2`, `L0` (restored box mid-season, no local pick files) | **Yes** — Hetzner rebuild from R2 `ops` state without local `2027-*.json` history. |
| **S7** | `M0`, `A4a` (mid-season `unmapped_settled` on restored box `S6` after `rounds.json` ID drift; or `F` error on `rounds={}` leaving `S6` untouched), `V2`, `L0` | **Yes** — Codex R2 B1: mid-season restore + `rounds.json` empty or unmapped. |
| **S8** | `M0`, `A4b` or `A0` (bare mid-season wipe with **neither** `2027` `contest_streak.json` **nor** local picks, plus broken `rounds.json`), `V0` | **Yes (rare)** — catastrophic loss of both local disk and R2 `ops` backup coinciding with MLB `rounds.json` failure. |
| **S9** | Any valid `M`/`A`, `L2` (root `2026-07-04.json` has invalid UTF-8 `0xff` or JSON `[]`) | **Yes** — truncated write/disk bitflip; crashes `main` (`contest_state.py:209,252`). |
| **S10** | Box restored on `2027-03-24` after `07:40` ET (`L3`: `streak.json` has `streak=14` from 2026); Day 1 settles at `22:15` ET via `update_streak` | **Yes** — `update_streak` (`picks.py:608`) adds `+1` to 2026's `14 -> 15` until `02:00` ET `reconcile_results`. |
| **S11** | Clock skew / TZ split at year boundary (`2026-12-31 23:30 ET` = `2027-01-01 04:30 UTC`, +/- NTP drift, or 2027 test running `target_date="2026-04-01"`) | **Yes** — UTC/ET split happens every Dec 31 after 19:00 ET; historical `target_date` runs in tests/backfills. |
| **S12** | Valid `M1`/`A1` in `2027`, but `V0` (`saver_state.json` still `season=2026` or missing) at `streak < 10` or `10 <= streak <= 15` | **Yes** — operator runs `set-contest-streak --streak 0` without `--best-streak 0` or skips `saver-state --init`. |
| **S13** | `M5`, `M2`/`M3a`/`M4` without proven `target_year` `A`, (`M0` + `A8`/`A7`), or (`M0` + `A6` untouched 2026 file) | **Yes** — untouched restored box before first 2027 fetch (`A6`), expired opening override when auto is still unproven (`M3a`+`A4b`/`A0`), or corrupt/future state (`M5`/`A7`/`A8`). |
| **U1** | `A1` (`current_settled` in `2027`) while `rounds_seasons == [2026]`, or auto `best_streak < active_streak` | **Unreachable** — `current_settled` requires a `2027` date in `rounds`, and `validate_fetch` rejects `best < active`. |

---

## 2. Evidence

**Core rule:** MLB account state is **never** inferred from the presence or absence of local `data/picks/YYYY-MM-DD.json` files (which failed on mid-season restore in Codex R2 B1 and on Day 2 in Carbon R2 B2). Local pick files measure only **how many settled bot picks post-date an already season-proven contest observation**.

| Decision | Evidence & source | Recorded at source or inferred? |
|---|---|---|
| Target contest season (`target_year`) | Caller's `target_date[:4]`, else `now.astimezone(ET).year` | Recorded at call site / system ET clock (never UTC year). |
| Auto file belongs to `target_year` | `proven_season == target_year` in `contest_streak.json` (for legacy `v1` without `provenance`: `source_date.year == target_year`, or `source_date is None` and `recorded_at` absent; `recorded_at` ET year `< target_year` is rejected) | **Recorded at fetch source** from `rounds.json` + `profile` + prior `contest_streak.json` of same `(user_id, username)`. Zero local pick reads. |
| Manual override validity & precedence | `M1` (`override_expires_at > now` and `source_date.year == target_year`) wins before parsing `A`. `M2` (wrong-season unexpired) and `M3a` (`override_expires_at <= now`, explicitly expired) **never** resurrect when `A` is missing or unproven (fixes `main` `contest_state.py:195`, which falls back to an expired override when `A` is absent). `M3b` (tagless legacy `override_expires_at is None`, `source_date.year == target_year`) remains fallback only when `A` is absent; `M4` (`< target_year`) is rejected | **Recorded at operator CLI** (`bts set-contest-streak`). |
| Policy `saver_available` (`bool`) | `saver_state.json` (`season == target_year`) + `decision_streak`: `active -> True`; `not_earned -> (decision_streak < 10)`; `used -> False`; `uninitialized -> False` | `saver_state` recorded by guarded transitions (`saver_state.py:113`); `<10` mapping is an exact MDP table invariant (`mdp.py`: `sv=1` means unspent). |
| Auto-earn saver (`maybe_auto_earn_saver`) | Allowed **only** when `proven_season == obs_year`: `uninitialized -> not_earned` if `best_streak < 10`; `not_earned -> active` **only** if `provenance == "current_settled"` and `best_streak >= 10` | Recorded from season-proven `seasonBestStreak`; un-reset `unmapped_settled` (`best=18`) can never touch `saver_state.json`. |
| Tail-policy `best_streak` trust | `classify_best_streak`: `0 <= streak <= best <= 57`, `source_date.year == target_year`, `source_date <= today_et`, valid schema | Recorded at source (`current_settled` or unexpired `M1`). |
| Freshness `status` (`fresh`/`lagged`/`stale`) | `source_date` vs `_resolved_pick_dates(picks_dir, season=target_year)` | Inferred lag label only (`status` does **not** alter `streak` or `allow_double`). |
| `streak.json` (`model_streak`) season | `season` field in `streak.json` (fallback for legacy: max year in `applied_dates`, else ET year of `updated`); if `< target_year`, treat as `(0, True, [])` | Recorded by `save_streak` / `reconcile_results`. |

### Fetch-side file format & signatures fixed by this design

1. **Empty `rounds.json` guard:** In `fetch-contest-streak`, if `_fetch_rounds()` returns `{}` (0 valid rounds), raise `ContestFetchError("rounds.json returned zero rounds")`. Do not touch `contest_streak.json`.
2. **Preserve `last_success_at` on fetch failure (`contest_streak_fetch_status.json`):** `_contest_fetch_alert` preserves `prior.get("last_success_at")` alongside `last_error`, `last_error_at`, `last_error_category` so operators and diagnostics always see both last successful and last attempted fetch.
3. **Extended `account_state/contest_streak.json` format** (retains `schema_version: "bts_contest_streak_auto_v1"` so `classify_best_streak` at `contest_state.py:101` stays compatible without dual-schema churn, adding explicit provenance fields):
```json
{
  "schema_version": "bts_contest_streak_auto_v1",
  "active_streak": 0,
  "best_streak": 0,
  "source": "mlb_bts_profile",
  "source_date": null,
  "recorded_at": "2027-03-24T12:10:00Z",
  "user_id": 595403,
  "username": "stonehengee",
  "saver_available": null,
  "provenance": "empty_reset",
  "proven_season": 2027,
  "last_proven_source_date": null,
  "rounds_seasons": [2027],
  "earliest_round_date": "2027-03-24",
  "settled_predictions_count": 0,
  "mapped_settled_count": 0
}
```
**Field meanings & transition rules (`obs_year = recorded_at.astimezone(ET).year`, `today_et = recorded_at.astimezone(ET).date()`):**
- `rounds_seasons`: sorted distinct years in `rounds.values()`. `earliest_round_date`: min round date in `obs_year` (or `null`).
- `settled_predictions_count`: count of `predictions[]` with `result in RESOLVED`. `mapped_settled_count`: count of those whose `roundId` exists in `rounds`.
- `provenance` (exhaustive 5-way partition):
  - `"current_settled"`: `>=1` settled row maps to a non-future date in `obs_year` (`mapped_date <= today_et + 1d`) -> `proven_season = obs_year`, `last_proven_source_date = source_date`.
  - `"empty_reset"`: `settled_predictions_count == 0`, `obs_year in rounds_seasons`, `active_streak == 0`, `best_streak == 0` -> `proven_season = obs_year`, `last_proven_source_date = prior_lpsd_if_obs_year`.
  - `"counter_only_unsettled"`: `settled_predictions_count == 0`, `obs_year in rounds_seasons`, `(active_streak > 0 or best_streak > 0)` -> `proven_season = obs_year` **iff** same-account prior file had `proven_season == obs_year` **and** `earliest_round_date is not None` **and** `today_et >= earliest_round_date`; else `null`.
  - `"unmapped_settled"`: `settled_predictions_count > 0` and (`mapped_settled_count == 0` or all mapped dates `> today_et + 1d`) -> `proven_season = obs_year` **iff** same-account prior file had `last_proven_source_date` in `obs_year` (carrying `last_proven_source_date` forward); else `null`.
  - `"prior_settled_only"`: `mapped_settled_count > 0` with `source_date.year < obs_year`, or `obs_year not in rounds_seasons` -> `proven_season = source_date.year` (if any) else `max(rounds_seasons)`.
- **Non-clobber rule:** If existing `contest_streak.json` (same account) has `proven_season == obs_year` and a new fetch yields `provenance == "prior_settled_only"` (e.g., CDN serves stale `rounds.json`), raise `ContestFetchError` rather than regressing a proven season.
- **Signatures:**
  - `load_contest_streak_state(picks_dir: Path, *, now: datetime | None = None, target_year: int | None = None) -> ContestStreakState | None`
  - `load_decision_streak_state(picks_dir: Path, *, require_contest_state: bool = False, now: datetime | None = None, target_date: date | str | None = None) -> DecisionStreakState`
  - `health.contest_state.check(picks_dir: Path, *, expected: bool = False, now: datetime | None = None, today: date | None = None) -> list[Alert]`

---

## 3. The Decision Table (Recommended Design — Improved Option A)

Columns show `require_contest_state=True` (live scheduler/orchestrator) and `[False]` where preview/research differs. **No row ends in "refuse forever":** every refusal row recovers immediately via `bts set-contest-streak` (which takes precedence before parsing `contest_streak.json`) or automatically on the next proven fetch.

| State | Streak (`req=True` / `[False]`) | Policy saver | Pick made? | Alert & timing | Operator recovery | If row's contest assumption is wrong |
|---|---|---|---|---|---|---|
| **S1a** (`M1` `s=0,b=0`, `V1`) | `0` / `[0]` | `True` (`not_earned`, `s<10`) | **Yes** | `INFO` in EOD health (override active until expiry) | None needed; expires automatically | If account somehow carried streak across seasons, picks at `0` until override expires/updated. |
| **S1b** (`M1` `s=0` left on; `A1`/`A3a` has `s>=1`) | `0` / `[0]` | `True` (`V1`/`V2`) | **Yes** | **`WARN`** at EOD health (`22:30` ET): active override (`0`) disagrees with proven `target_year` auto (`>=1`) | `rm .../contest_streak.manual.json` (or let <=72h TTL expire) | Action identical for `s=1..5` (`d>=170, sv=1`); if ignored past `s>=6` (Day 5+), over-doubles until expiry. |
| **S2** (`A2` `empty_reset`, `V1`) | `0` / `[0]` | `True` | **Yes** | None (`0` settled `Y` picks); `INFO` if `1` settled `Y` pick; `WARN` if `>=2` settled `Y` picks | None needed | If MLB zeroed profile erroneously mid-season (impossible on `S6` if `predictions` non-empty; if cleared too), picks at `0` until corrected via `set-contest-streak`. |
| **S3** (`A3a` `counter_only` chained, `V1`) | `A.streak` / `[A.streak]` | `True` (`s<10`) | **Yes** | `INFO` at EOD health (`1` settled `Y` pick); `WARN` if `>=2` settled `Y` picks with `predictions=[]` | None needed; if `WARN` persists, inspect fetch / `set-contest-streak` | If MLB flashed `0/0` pre-opener then restored 2026 `active>0` on Opening Day with `predictions=[]`, picks at stale streak -> mitigated by checklist B5 override on Days 1–2. |
| **S4** (`A4b` un-reset 2026 vs 2027 rounds, `M0`) | **Refuse** / `[model: 0]` | n/a / `[V->bool]` | **No** (`ContestStateError`) / `[Yes]` | **Immediate DM** on scheduler check (`scheduler.py:1739`), `CRITICAL` at `22:30` health, DM at `23:00` `day-outcome` (`no_decision`) | `bts set-contest-streak --streak 0 --best-streak 0 --ttl-hours 48` (recovers on next check same day) | If account was genuinely mid-season `2027` on a bare wiped box (`S8`), refusal prevents zeroing streak `13->0`; operator sets real streak. |
| **S5** (`A5` `prior_settled_only`, `M0`) | **Refuse** / `[model: 0]` | n/a / `[V->bool]` | **No** / `[Yes]` | Same as **S4** (names `prior_settled_only`, `rounds_seasons`) | Same 1-command `set-contest-streak` override | If MLB keeps `2026` rows in `predictions[]` with `rounds=[2026,2027]` and `active=0`, refuses once `M1` expires if no `2027` pick settled yet; operator extends `M1` until first `2027` settlement. |
| **S6** (`A1` mid-season restore, `L0`, `V2`) | `13` / `[13]` | `True` (`active`) | **Yes** | None (`source_date` in `2027`, `0` newer local picks -> `fresh`) | None needed | None — `source_date` is proven `2027` from MLB endpoints. |
| **S7** (`A4a` mid-season restore + broken `rounds.json`, `V2`, `L0`) | `A.streak` (`13`, or `0` on miss) | `True` (`active`) | **Yes** (`status="stale"`) | Fetch DM if `rounds={}`; **`CRITICAL` DM** at `22:30` EOD health (`source_date is None`, `unmapped_settled`) | Fix `rounds.json` parser or set `bts set-contest-streak` to restore `fresh` | `activeStreak` still updates live (so a miss `13->0` is respected!), while `last_proven_source_date` in `2027` prevents false reset to `0`. |
| **S8** (bare mid-season wipe + broken `rounds.json`, `M0`, `V0`) | **Refuse** / `[model: 0]` | n/a / `[False]` | **No** / `[Yes]` | Fetch DM + scheduler refusal DM + `22:30` `CRITICAL` + `23:00` `day-outcome` DM | `bts set-contest-streak --streak <S> --best-streak <B>` + `bts saver-state --init <state>` | Refusing avoids both silent `13->0` wipeout and silent `2026` carryover; 2 CLI commands restore exact state. |
| **S9** (`L2` corrupt/non-UTF-8 `.json` pick file) | Normal `M`/`A` streak | Normal `V` | **Yes** | `WARN` logged on unreadable pick file (never crashes reader) | Repair/remove corrupt pick JSON when convenient | Unreadable local pick cannot mask an unconfirmed miss -> treat unreadable post-`source_date` `target_year` pick as `status="stale"` + `WARN`. |
| **S10** (`L3` 2026 `streak.json` at Day-1 `update_streak`) | Live uses `M1`/`A`; `model_streak` resets `0->1` | Live uses `V`; `model_saver=True` | **Yes** | None (`update_streak` sees `streak.json` season `<2027`, resets base to `0, True, []`) | Automatic on first `2027` `update_streak` or `02:00` `reconcile` | None — `streak.json` carries `season: 2027`. |
| **S11** (Dec 31/Jan 1 TZ/NTP skew; `target_date` in 2026) | Anchored to `target_year` (ET) | Anchored to `target_year` | Per `target_year` | Fetch on Jan 1 against `2026` rounds marks `prior_settled_only` (`proven_season=2026`) | Automatic once `2027` rounds publish | Clock skew at Jan 1 cannot mint `proven_season=2027` because `rounds.json` has `[2026]`. |
| **S12** (`V0` `uninitialized` in `2027` at `s<10` or `10..15`) | Normal `M1`/`A1` streak | **`False`** (fail-closed) | **Yes** (conservative saver) | **`WARN`** in `health/contest_state` whenever `expected=True` (any `s`) or `10<=s<=15` | `bts saver-state --init not_earned` (or `active`/`used`), or automatic on next `A1`/`A2` fetch when `best<10` | At `s=5,7,8,9` runs `sv=0` until initialized; at `s=0..4` (`d>30`), `sv=0` and `sv=1` have 0 policy diffs. |
| **S13** (`A6` 2026 file; `M2`/`M3a`/`M4` w/o proven `A`; `A7` future date; `M5`/`A8` corrupt) | **Refuse** (`M3b` legacy `Y` w/ `A0` still works + `WARN`; `M1` overrides `A6`/`A7`/`A8`) | n/a (`M3b`: `V->bool`) | **No** (`M3b`: **Yes**) | Immediate scheduler refusal DM + `22:30` `CRITICAL` + `23:00` `day-outcome` DM | `bts set-contest-streak ...` (renews `M3a`, fixes `M2`/`M5`, bypasses `A6`/`A7`/`A8`) | None — expired, wrong-season, future, or corrupt state never silently drives a live pick. |

---

## 4. What Stays Unknowable Until March 2027

Every item below is **unsure** until observed against the live MLB endpoints during the March 2027 pre-season window (`2027-03-03..03-25`). None causes a silent stale-streak pick or an unrecoverable refusal:

1. **When `user/{id}/profile` resets (`activeStreak`, `seasonBestStreak`, `predictions[]`) — unsure** (settled by logging dry-run/cron `fetch-contest-streak` payloads from `T-21d` through `2027-03-26`):
   - *Resets before Opening Night (`0/0, []`) with `2027` `rounds.json`:* Lands in **S2** (`empty_reset`, `proven_season=2027`), auto-initializes `2027` `saver_state` to `not_earned`, works even without manual override.
   - *Resets on Opening Night or several days late:* Lands in **S4** (`unmapped_settled`, `proven_season=None`). With B5 override (**S1a**), Days 1–2 pick at `streak=0`; if still un-reset when override expires, refuses loudly (**S4**) and operator extends override (`--ttl-hours 48`) until reset or first `2027` settlement.
   - *Clears `predictions=[]` while leaving `activeStreak=14` un-reset (`A3b`):* Because no prior `proven_season=2027` exists (or `today_et < earliest_round_date`), `proven_season` stays `None` -> refuses (**S4**-class) unless B5 override is active, and **never** calls `maybe_auto_earn_saver` with `best=18`.
2. **Order of `rounds.json` rollover vs `profile` rollover, and single-season vs multi-season `rounds.json` — unsure** (settled by inspecting `ROUNDS_URL` JSON in March 2027; 2026 fixture started at `id: 823`, `2026-03-25`):
   - *`rounds.json` lags in `[2026]` while `profile` resets (`0/0, []`):* Lands in **S5** (`prior_settled_only` because `2027 not in rounds_seasons`). B5 override carries Opening Night; `check-pick-entered` will also degrade until `rounds.json` lists `2027`, surfaced by `day-outcome` (`entry_unverifiable`).
   - *`rounds.json` becomes `[2026, 2027]` AND `profile` retains 2026 `predictions[]` while resetting `activeStreak=0, best=0`:* Before any 2027 pick settles, `mapped_current_year_count == 0` -> **S5** (`prior_settled_only`); B5 override carries Opening Days, and the **first** settled 2027 pick sets `mapped_current_year_count >= 1` -> permanently transitions to **S6** (`current_settled`, `proven_season=2027`) even with 2026 rows still in `predictions[]`.
3. **`roundId` numbering across seasons (monotonic `>1007` vs reset to `1`) — unsure** (settled on first 2027 `rounds.json`):
   - *Monotonic:* 2026 IDs (`823..1007`) do not collide with 2027 IDs (`unmapped_settled` until a 2027 pick settles).
   - *Resets to `1` while 2026 `predictions[]` (`823..`) are un-reset!!* **Critical trap:** A full-season 2027 `rounds.json` starting at `1` has ~185 rounds (`1..185`), so `823` still fails to map (`unmapped_settled`). Only if 2026 IDs had been `1..185` AND `rounds.json` reused `1..185` while `profile` was un-reset would an old `roundId` map to a **future** 2027 date (e.g. Sep 2027), which `source_date > today_et + 1d` (**S13**/`A7`) and `classify_best_streak` (`source_date > today_et`) **reject**; we also tighten `current_settled` to require `mapped_date <= today_et + 1d` so a future-mapped colliding `roundId` is classified `"unmapped_settled"`, never `"current_settled"`!
4. **Carried cross-season field or new profile schema (`season` key, renamed streak key) — unsure** (settled on first March 2027 profile response):
   - Missing/non-int `activeStreak` or `seasonBestStreak` raises `ContestFetchError` (`validate_fetch`) and alerts via DM. If `activeStreak` becomes lifetime/carried instead of seasonal, B5 override protects Opening Days while the fetch parser is updated.
5. **Day-2 `predictions[]` lag on Round 1 (`activeStreak=1, predictions=[]`) — unsure** (settled on `2027-03-25` morning fetch):
   - Handled natively by **S3** (`counter_only_unsettled` chained from `empty_reset`) with `INFO` (not `CRITICAL`) at EOD health.

---

## 5. Refuting the Raw Leading Option, and Making It Fail-Safe

**The raw leading option:** *(1) operator override to streak 0 for opening days, (2) fetch-side provenance for everything after, (3) code that only refuses a file that is plainly last season's.*

### Strongest case against the raw form

1. **"Only refuses a file plainly last season's" fails on the exact March partial-rollover state (S4):** If A4 (`fetch-contest-streak` at `T-14d..T-1d`) runs after `rounds.json` switches to `2027` and before MLB resets `profile`, `fetch-contest-streak` writes a **2027-timestamped** `contest_streak.json` with `recorded_at: "2027-03-23T..."`, `source_date: null`, `active_streak: 14`, `best_streak: 18`. That file is **not** plainly from 2026 by timestamp or `source_date`. Unless the reader refuses **unproven** auto observations (`proven_season != target_year`), forgetting or expiring the override picks on 2026's streak `14` (`skip` on Q0 instead of `single`/`double`).
2. **Cost if the operator forgets the override:**
   - Under the raw rule: **silent stale-streak pick** (2nd-worst failure) if a March fetch wrote `source_date: null` (`S4`).
   - Under our improved rule (`proven_season == target_year` required): scheduler refuses each check with an **immediate `ContestStateError` DM** (`scheduler.py:1739`), `22:30` health fires `CRITICAL`, and `23:00` `day-outcome` DMs `no_decision`. Because Opening Night (`2027-03-24`) is an evening game, the operator has from the first `T-180` check DM until `T-5` cutoff to run one command; subsequent checks (or the `T-20` extra attempt) pick on the same day.
3. **Cost if the operator sets it late:**
   - *Before any pick locks (e.g., after a `ContestStateError` refusal DM or an MDP `skip`):* Zero permanent cost; next check re-evaluates and delivers.
   - *After an early thin-slate lock (`T-180`/`T-120` on 1-game Opening Night) under a stale streak:* Once delivered, `select_pick` sees `locked=True` and will not overwrite Day 1; recovery takes effect Day 2. (Under our improved rule, `S4`/`S5`/`S13` **refuse** rather than locking a stale-streak pick, so setting the override before `T-5` still rescues Day 1!)
4. **Cost if the operator leaves `streak=0` past the first hit (measured on `data/models/mdp_policy.npz` and `mdp_tail_policy.npz`):**
   - Opening fortnight `2027-03-24..04-09` has `days_remaining >= 170` (clamped to `d=180` for `03-24..03-30`).
   - With `saver_available=True` (`sv=1`), **`s = 0, 1, 2, 3, 4, 5` choose `double` in all 5 quality bins across all `d >= 170` (`0` cells differ; clamped to `single` on 1-game `2027-03-24`)**. Max pre-pick streak in 2027 is `0` on Wed `03-24`, `1` on Thu `03-25`, `3` on Fri `03-26`, `5` on Sat `03-27`, first reaching `6` or `7` on **Sun `2027-03-28` (Day 5)** (or Day 4 on a multi-game opener: `2->4->6`).
   - **Action divergence begins at `s = 6` (`d=180, sv=1`):** `s=0` picks `double` where true `s=6` Q0 picks `single`, `s=7` Q0 picks `skip`, `s=8` Q0–Q4 picks `single` in **all 5 bins**, and `s=9` Q0 picks `skip`. Even at `s=1..5`, a left-on override mislabels `streak=0` in Bluesky DMs/posts and `decision.json` (and in `mdp_tail_policy.npz` at `d<=28`, `s=0` already differs at `s=1`).
5. **What it does to the saver (two concrete traps on `main`):**
   - *Trap A:* `bts set-contest-streak --streak 0` defaults `--best-streak` to `None` (`cli.py:1370`), making `maybe_auto_earn_saver` a **no-op** (`saver_state.py:175`). `saver_state.json` stays `season=2026` (`uninitialized` -> `False`), flipping `s=5` Q0 `double->single`, `s=7` Q0 `skip->single`, `s=8` Q0–Q3, and failing B8.
   - *Trap B:* The 4x/day `fetch-contest-streak` cron keeps running during the override. On `main` (`cli.py:2104`), an un-reset `profile` (`seasonBestStreak=18`, `source_date=None`) runs `maybe_auto_earn_saver(..., season=2027, best_streak=18)`, promoting `2027` `not_earned -> active` at streak 0 (sticky for the entire season!).

### How Option A becomes fail-safe without "refuse forever"
1. Require `proven_season == target_year` on auto files (with `mapped_date <= today_et + 1d`), while carrying `last_proven_source_date` across mid-season `unmapped_settled` (**S7**).
2. Gate `maybe_auto_earn_saver` strictly on `proven_season == obs_year` (and `not_earned -> active` on `provenance == "current_settled"`).
3. Map `not_earned` + `decision_streak < 10 -> saver_available = True`, keeping `uninitialized -> False`.
4. Add `--season-start` convenience mode (or require `--best-streak 0` in B5) on `set-contest-streak` capping TTL to `<=72h` (expiring before Day 4 `s=6`), and warn in `health/contest_state` whenever an active override disagrees with a proven `target_year` auto file (**S1b**) or `saver_state` is `uninitialized` (**S12**).

---

## 6. Alternative in Kind (Alternative B) and Recommendation

**Alternative B — Contest-Calendar-Bounded Automatic Degrade-to-Zero (no mandatory pre-season operator override):**
Instead of refusing unproven auto state on Opening Days unless a human wrote `contest_streak.manual.json`, Alternative B uses `rounds.json`'s own `target_year` schedule (`past_rounds = count(d in rounds.values() : d.year == target_year and d < today_et)`, recorded in `contest_streak.json`): when no `M1` override exists and `proven_season != target_year`, **if `target_year in rounds_seasons` and `past_rounds <= 2`**, `load_decision_streak_state` automatically degrades to `(streak = 0, saver_available = True, status = "season_start_fallback")` and picks while firing a `WARN` DM; once `past_rounds >= 3` (Day 4+, before `s=6` is reachable on Day 5!) or when `target_year not in rounds_seasons`, it refuses (`ContestStateError`) just like Option A.

### Decision Table for Alternative B (same columns as Section 3)

| State | Streak (`req=True` / `[False]`) | Policy saver | Pick made? | Alert & timing | Operator recovery | If row's contest assumption is wrong |
|---|---|---|---|---|---|---|
| **S1a, S1b, S2, S3, S6, S7, S9, S10, S11, S12** | **Identical to Option A** | **Identical** | **Identical** | **Identical** | **Identical** | **Identical to Section 3** |
| **S4 (Days 1–3, `past_rounds <= 2`)** (`A4b` un-reset vs 2027 `rounds.json`, `M0`) | **`0`** / `[0]` | **`True`** | **Yes** (auto-degraded) | `WARN` DM on pick + `WARN` at `22:30` EOD health (`season_start_fallback`) | None needed on Days 1–3; if un-reset by Day 4, `bts set-contest-streak` | Under current `mdp_policy.npz`, `s=0,sv=1` matches `s=1..5` (`d>=170`), **except**: if policy `.npz` is re-solved to split `s=0..3`, picks wrong action on Days 2–3; DMs/`decision.json` log `streak=0` when real streak is `1..3`. |
| **S4 (Day 4+, `past_rounds >= 3`)** | **Refuse** / `[model]` | n/a / `[V->bool]` | **No** (`ContestStateError`) | Scheduler refusal DM + `22:30` `CRITICAL` + `23:00` `day-outcome` DM | `bts set-contest-streak --streak <S> --best-streak <B>` | Same as Option A **S4**. |
| **S5** (`rounds.json` still `[2026]`, `M0`) | **Refuse** / `[model: 0]` | n/a / `[V->bool]` | **No** | Same refusal DMs as Option A **S5** | `bts set-contest-streak --streak 0 --best-streak 0` | Cannot compute `2027` `past_rounds` without `2027` in `rounds.json` (unless adding a hardcoded `SEASON_START_DATES` table). |
| **S8** (bare mid-season wipe + broken `rounds.json`, `M0`) | **Refuse** (`past_rounds >= 3` or unmapped) | n/a | **No** | Same refusal DMs as Option A **S8** | `bts set-contest-streak` + `bts saver-state --init` | Safe in June (`past_rounds >> 2`); if MLB truncated early rounds from `rounds.json` so `past_rounds <= 2` mid-season, **would falsely zero a mid-season streak!** |
| **S13** (`A6` untouched 2026 file before any 2027 fetch; `M5`/`A7`/`A8`) | **Refuse** (`A6` has no `2027` `rounds_seasons`) | n/a | **No** | Same refusal DMs as Option A **S13** | Run `fetch-contest-streak` or `set-contest-streak` | If cookies also expired on Opening Night (`A6` stays on disk), B still refuses just like A! |

### Recommendation and Cost of Being Wrong

**Recommendation: Improved Option A.**
- **Why Option A wins:** Alternative B only saves the operator one CLI command in the narrow slice where `fetch-contest-streak` **succeeded** against `2027` `rounds.json` while `profile` lagged (`S4`, Days 1–3) — it **still refuses** when cookies are expired (`S13`/`A6`) or `rounds.json` lags (`S5`). Worse, B bakes an assumption of today's `.npz` table (`s=0 == s=1..5`) into state loading, writes false `streak=0` provenance into `decision.json` on Days 2–3, and creates a new failure mode if `rounds.json` ever drops historical rounds mid-season (`past_rounds <= 2` -> zeroing a mid-season streak). Option A never fabricates a streak in code.
- **Cost if Option A is wrong:** Operator skips checklist B5 **and** MLB's profile hasn't reset on Opening Night -> `ContestStateError` DM fires at `T-180`; if the operator ignores every DM through `23:05` UTC first pitch, **1 missed Opening Night pick** (never a stale-streak pick, never permanent refusal).
- **Cost if Alternative B is wrong:** If `mdp_policy.npz` changes at `s=1..3` or `rounds.json` drops completed rounds on a bare restore, **silent wrong-streak live pick** (worse than an alerted pre-game pause when operator has hours before first pitch), plus corrupted `decision.json` streak records on Days 2–3.

---

## 7. RED Tests Implied by the Design

1. `A4b` partial rollover (`2027` `rounds.json`, un-reset `2026` `predictions[]`, `active=14, best=18`) in `fetch-contest-streak` -> writes `provenance="unmapped_settled"`, `proven_season=None`, and leaves `2027` `saver_state` (`not_earned`) untouched (Audit F2).
2. `load_decision_streak_state(require_contest_state=True)` on `S4` (`provenance="unmapped_settled"`, `proven_season=None`) on Day 1 (`2027-03-24`, no `2027` local picks) and Day 2 (`2027-03-25`, 1 settled `2027` pick) -> raises `ContestStateError` on both days (Carbon R2 B2).
3. First 2027 fetch with `rounds.json` listing only `2026` rounds (`S5`, `source_date=2026-09-26` or `predictions=[]`) -> writes `provenance="prior_settled_only"`, `proven_season=2026`; `load_decision_streak_state(require_contest_state=True, target_date="2027-03-24")` raises `ContestStateError` (Carbon R2 B2).
4. Empty `rounds.json` (`{}`) in `fetch-contest-streak` (`S7`) -> raises `ContestFetchError`, exits nonzero, preserves existing `contest_streak.json` (`source_date="2027-05-31"`, `streak=13`), and preserves `last_success_at` in `contest_streak_fetch_status.json` (Codex R2 B1).
5. Mid-season restored box (`S6`/`S7`: prior `contest_streak.json` has `source_date="2027-05-31"`, zero local `2027-*.json` picks) followed by unmapped `rounds.json` fetch (`active=13` or miss `active=0`) -> carries `last_proven_source_date="2027-05-31"`, `proven_season=2027`, returns `streak=13` (or `0`), `status="stale"`, `health` `CRITICAL` (Codex R2 B1).
6. Pre-opener backend flap (`empty_reset` on `2027-03-22` followed by `unmapped_settled` `active=14` on `2027-03-23` with `last_proven_source_date=None`) -> `proven_season` resets to `None` (does not launder 2026 streak).
7. `A3b` cleared `predictions=[]` with un-reset `active=14, best=18` and no prior `proven_season=2027` -> `proven_season=None`, refused by `require_contest_state=True`, `saver_state` untouched.
8. Day-2 1-round `predictions[]` lag (`S3`: `empty_reset` on Day 1 -> `active=1, best=1, predictions=[]` on `2027-03-25` with 1 settled `2027` pick) -> `provenance="counter_only_unsettled"`, `proven_season=2027`, returns `streak=1`, EOD `health/contest_state` emits `INFO`, not `CRITICAL` (Audit F5).
9. Colliding `roundId` reset to `1` mapping a stale `predictions[]` row to a future date `2027-09-15` on `2027-03-24` -> excluded from `current_settled`, `proven_season=None`.
10. `saver_state="not_earned"` (`season=2027`) at `streak=0` and `streak=5` -> `load_decision_streak_state` returns `saver_available=True` (`select_pick` chooses `double` at `s=5, q=0, d=180`), and returns `False` if `streak=12` (Audit F1).
11. `saver_state="uninitialized"` (missing or `season=2026`) at `streak=7` (`test_uses_fresh_contest_streak_for_live_action`), `streak=8` (`best=12`), and `streak=12` -> `saver_available=False` in all three; `health/contest_state(expected=True)` emits `WARN` even at `streak < 10` (`sr-state` regression + Carbon R2 B3).
12. `web.saver_dashboard_context` and `saver_transition_response` on `2027-03-10` with `2026` `contest_streak.json` (`source_date="2026-09-27"`) and `2026` `saver_state.json` -> reads `uninitialized` for `2027` and writes `season=2027`, never `2026` (Audit F3).
13. Override precedence & expiry (`S1b`, `S13`): active `2027` override (`streak=0`) while proven `2027` auto has `active_streak=2` uses `0` and emits `health` `WARN`; once `override_expires_at <= now` (`M3a`) with missing or unproven auto (`A0`/`A4b`), `require_contest_state=True` raises `ContestStateError` instead of falling back to the expired override, while `M3b` (`override_expires_at=None`, `source_date` in `target_year`, `A0`) still works.
14. Corrupt JSON / invalid UTF-8 (`0xff`) / non-dict (`[]`, `7`) in root pick files (`2026-07-04.json`) and `account_state/*.json` -> `_resolved_pick_dates` and `_has_unconfirmed_miss` never crash (unreadable post-`source_date` `target_year` pick marks `status="stale"`), and `_parse_state_file` converts `UnicodeDecodeError` to `ContestStateError` (`health` `CRITICAL`) (Codex R2 B4, Carbon R2 P1).
15. `picks.update_streak([True], picks_dir, date="2027-03-24")` when `streak.json` holds `2026` state (`streak=14`, `applied_dates=["2026-09-27"]`) -> writes `streak=1, season=2027, applied_dates=["2027-03-24"]`, not `15` (`S10`).
16. Year-boundary clock/target-date checks (`S11`): fetch at `2026-12-31T23:30-05:00` (`2027-01-01T04:30Z`) uses `obs_year=2026`; fetch at `2027-01-01T00:05-05:00` with `2026` `rounds.json` sets `proven_season=2026`; `run_and_pick(config, "2026-04-01")` run under a `2027` wall clock evaluates `target_year=2026`.
17. Existing 2026-dated unit tests in `tests/test_contest_state.py` and `tests/health/test_contest_state.py` that call `load_decision_streak_state(tmp_path)` / `check(tmp_path)` with `now=None` -> still pass when system clock is mocked to `2027-03-24` once pinned to explicit `now`/`target_date`.

---

## 8. Change Sets (Smallest First) and Pre-2027 Gate

| Order | Change set & scope | Must be live before `2027-03-24` pick 1? |
|---|---|---|
| **CS1** | **Defensive pick-file & status-file I/O** (`contest_state.py:_parse_state_file,_resolved_pick_dates,_has_unconfirmed_miss` catch `UnicodeDecodeError` + require `isinstance(body, dict)`; `cli.py:_contest_fetch_alert` preserves `last_success_at`). | **Yes** — prevents crash on corrupt/non-UTF-8/non-dict JSON and preserves fetch observability. |
| **CS2** | **Local `streak.json` season tag** (`picks.py:save_streak,load_streak,load_saver_available,update_streak,reconcile_results` reset prior-season `streak.json` when `file_season < target_season`, preserving tagless legacy test fixtures). | **Yes** — closes `22:15–02:00` ET Opening Night `14 -> 15` bug (`S10`) and research-mode carryover. |
| **CS3** | **Saver policy mapping & `target_year` binding** (`saver_state.py` / `contest_state.py` / `web.py`: `not_earned` + `s<10 -> True`, `uninitialized -> False`; replace `season_for(source_date, ...)` in readers/dashboard with `target_year`; `health/contest_state` warns on `expected=True` + `uninitialized` at all streaks). | **Yes** — fixes `s=5,7,8,9` policy actions (Audit F1), dashboard `2026` write bug (Audit F3), and preserves `uninitialized -> False` safety. |
| **CS4** | **Fetch-side provenance & saver auto-earn guard** (`contest_fetch.py`, `cli.py:fetch-contest-streak`: reject empty `rounds={}`; record `provenance`, `proven_season`, `last_proven_source_date`, `rounds_seasons`, `earliest_round_date`, counts; non-clobber rule; gate `maybe_auto_earn_saver` on `proven_season == obs_year`). | **Yes** — core source-provenance layer; stops un-reset `best=18` from poisoning `2027` saver (Audit F2) and preserves mid-season restores (Codex R2 B1). |
| **CS5** | **Decision & health season enforcement + `target_date` plumbing** (`contest_state.py`, `health/contest_state.py`, `health/runner.py`, `orchestrator.py`, `scheduler.py`, `cli.py`, plus pinning `now` in 2026-hardcoded unit tests): enforce `proven_season == target_year` (and manual `source_date.year == target_year`); `INFO` vs `CRITICAL` for `S2`/`S3` EOD health; `WARN` on override-vs-proven-auto disagreement (`S1b`). | **Yes** — prevents picking on stale 2026 streak and eliminates Opening Night false `CRITICAL` DMs (Audit F5). |
| **CS6** | **Operator CLI & runbook ergonomics** (`cli.py:set-contest-streak --season-start` shorthand setting `streak=0, best_streak=0, ttl<=72h` and initializing `not_earned`; update `docs/ops/2027-season-start.md` B5/B8 to pass `--best-streak 0` and verify `provenance` / `saver_state`). | **No** (nice-to-have before activation; checklist edit + existing `--streak 0 --best-streak 0 --ttl-hours 48` + `saver-state --init not_earned` already work once CS1–CS5 land). |
