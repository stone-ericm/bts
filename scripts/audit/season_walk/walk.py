"""Walk every 2027 date through the real decision path (closing test of the whole-range review, part d).

For each calendar date and each synthetic contest state, a fresh temporary picks directory gets
that state through the real CLI under a pinned ET clock (``bts set-contest-streak --date`` for a
dated observation; ``bts season-contract enable`` + ``bts fetch-contest-streak`` with stubbed
network calls + ``bts saver-state --init`` for the automatic path), then
``orchestrator.run_and_pick`` (the scheduler's decision call) chooses from three synthetic
candidates in three different games with the shipped policy artifacts
(``data/models/mdp_policy.npz``, ``data/models/mdp_tail_policy.npz``). No network (sockets
refuse), no real data, no 2026 or 2027 outcomes.

Checks (``check``): no error; the policy's day count equals the count of calendar dates from the
date through the 2027 finale (2027-09-26) inclusive, 0 after it, and its effective count is that
clamped to the table length; every date yields a pick or a NAMED skip (a skip that carries the
policy's structured decision); every decision comes from the MDP (the shipped artifacts loaded);
the saver the policy sees matches the documented rule; the pick records the contest-state kind
it was chosen on; after the finale every action is a skip; on the finale streak 56 plays; a skip's
operator notice (``scheduler.format_skip_dm`` over ``build_skip_summary``, as the scheduler builds
it from the same decision) never says the best candidate is below a bar that candidate clears.
``boundary_checks`` covers the stored boundary, the year boundary and the season-end fallback.

Run from the repository root:
    .venv/bin/python -m scripts.audit.season_walk.walk --out <file.json>
Exit 0 when no problem is found, 1 otherwise. Limits are listed in README.md beside this file.
"""
from __future__ import annotations

import argparse
import datetime as _dtmod
import json
import os
import socket
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

from bts.simulate.tail_policy import LONGEST_STREAK_FROM, LONGEST_STREAK_HORIZON, OBJECTIVE_LONGEST

ET = ZoneInfo("America/New_York")
REAL_DATETIME = _dtmod.datetime
REPO = Path(__file__).resolve().parents[3]

FIRST_GAME_DAY = date(2027, 3, 24)
LAST_GAME_DAY = date(2027, 9, 26)     # CLAUDE.md: the 2027 playing finale
WALK_START = date(2027, 3, 1)
WALK_END = date(2027, 10, 5)
SEASON_LENGTH = 180                   # the shipped table's season length
BASE_SHA_PREFIX = "66d15471"
TAIL_SHA_PREFIX = "dc5d0c99"
ACCOUNT = "walk_account"
UID = 4242


@dataclass(frozen=True)
class State:
    kind: str          # "dated" | "automatic"
    streak: int
    best: int
    saver: str         # "not_earned" | "active" | "used"
    top_p: float

    def label(self) -> str:
        return f"{self.kind}/{self.streak}/{self.best}/{self.saver}/{self.top_p}"


DEFAULT_STATES = tuple(
    [State("dated", s, b, v, p)
     for (s, b, v) in ((0, 0, "not_earned"), (5, 5, "not_earned"), (12, 12, "active"),
                       (12, 12, "used"), (25, 30, "used"), (43, 56, "used"), (50, 50, "used"),
                       (55, 55, "active"), (56, 56, "used"))
     for p in (0.96, 0.72)]
    + [State("automatic", s, b, v, 0.96)
       for (s, b, v) in ((12, 12, "active"), (43, 56, "used"), (56, 56, "used"))]
    + [State("dated", LONGEST_STREAK_FROM, LONGEST_STREAK_FROM, "used", 0.96)]
)


def expected_days_left(day: date) -> int:
    """Independent oracle: calendar dates from ``day`` through the finale, inclusive."""
    return max(0, (LAST_GAME_DAY - day).days + 1)


def in_season(day: date) -> bool:
    return FIRST_GAME_DAY <= day <= LAST_GAME_DAY


def expected_saver_in_hand(state: State) -> bool:
    """CLAUDE.md change set 3: in hand when active, or not_earned with the streak below 10."""
    return state.saver == "active" or (state.saver == "not_earned" and state.streak < 10)


def table_boundary_problems() -> list[str]:
    from bts.strategy import SEASON_END_DATES
    stored = date.fromisoformat(SEASON_END_DATES[2027])
    if stored != LAST_GAME_DAY + timedelta(days=1):
        return [f"SEASON_END_DATES[2027] is {stored}, expected the day after the finale "
                f"{LAST_GAME_DAY + timedelta(days=1)}"]
    return []


# ---------------------------------------------------------------- clock, network, CLI

class _Clock(REAL_DATETIME):
    at: REAL_DATETIME | None = None

    @classmethod
    def now(cls, tz=None):
        value = cls.at
        return value.astimezone(tz) if tz else value.astimezone(ET).replace(tzinfo=None)

    @classmethod
    def utcnow(cls):
        return cls.at.astimezone(_dtmod.timezone.utc).replace(tzinfo=None)


def _refuse_network(*args, **kwargs):
    raise OSError("season walk: network refused")


@contextmanager
def offline(at: REAL_DATETIME):
    """Pin every ``datetime`` the bts package reads to ``at`` (ET-aware) and refuse sockets."""
    import bts.cli, bts.contest_state, bts.contest_fetch, bts.orchestrator, bts.picks  # noqa: F401
    import bts.saver_state, bts.season_contract, bts.strategy  # noqa: F401
    import bts.health.contest_state, bts.health.tail_policy  # noqa: F401
    _Clock.at = at
    with ExitStack() as stack:
        stack.enter_context(patch.object(_dtmod, "datetime", _Clock))
        for name, module in list(sys.modules.items()):
            if (name == "bts" or name.startswith("bts.")) and getattr(module, "datetime", None) is REAL_DATETIME:
                stack.enter_context(patch.object(module, "datetime", _Clock))
        for method in ("connect", "connect_ex", "sendto"):
            stack.enter_context(patch.object(socket.socket, method, _refuse_network))
        stack.enter_context(patch.object(socket, "getaddrinfo", _refuse_network))
        yield


def _set_clock(at: REAL_DATETIME) -> None:
    _Clock.at = at


def _invoke(*args) -> tuple[int, str]:
    from click.testing import CliRunner
    from bts.cli import cli
    result = CliRunner().invoke(cli, [str(a) for a in args])
    text = result.output
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        text += f"\n[{type(result.exception).__name__}: {result.exception}]"
    return result.exit_code, text


@contextmanager
def _fetch_stubs(streak: int, best: int, settled_round_day: date):
    from bts import cli as cli_module
    from bts import contest_fetch
    from bts.leaderboard import auth
    session = SimpleNamespace(user_id=UID, username=ACCOUNT, xsid="walk_xsid")
    profile = {"activeStreak": streak, "seasonBestStreak": best,
               "predictions": [{"roundId": 1, "result": "hit"}]}
    with ExitStack() as stack:
        stack.enter_context(patch.object(auth, "load_session_cookies", lambda: {}))
        stack.enter_context(patch.object(auth, "extract_uid", lambda cookies: UID))
        stack.enter_context(patch.object(auth, "fetch_login_session", lambda **kw: session))
        stack.enter_context(patch.object(contest_fetch, "fetch_profile", lambda *a: profile))
        stack.enter_context(patch.object(cli_module, "_fetch_rounds", lambda: {1: settled_round_day}))
        yield


def install_state(picks: Path, day: date, state: State) -> list[str]:
    """Install ``state`` for ``day`` through the real CLI. Returns setup problems."""
    problems = []
    noon = REAL_DATETIME.combine(day, _dtmod.time(12), ET)
    if state.kind == "dated":
        _set_clock(noon - timedelta(hours=2))
        code, out = _invoke("set-contest-streak", "--picks-dir", picks, "--date", day.isoformat(),
                            "--streak", state.streak, "--best-streak", state.best,
                            "--saver", state.saver)
        if code != 0:
            problems.append(f"set-contest-streak exit {code}: {out.strip()[:300]}")
    elif state.kind == "automatic":
        _set_clock(noon - timedelta(hours=3))
        code, out = _invoke("season-contract", "enable", "--season", day.year, "--picks-dir", picks)
        if code != 0:
            problems.append(f"season-contract enable exit {code}: {out.strip()[:300]}")
        with _fetch_stubs(state.streak, state.best, day - timedelta(days=1)):
            code, out = _invoke("fetch-contest-streak", "--picks-dir", picks,
                                "--expected-username", ACCOUNT)
        if code != 0:
            problems.append(f"fetch-contest-streak exit {code}: {out.strip()[:300]}")
        code, out = _invoke("saver-state", "--init", state.saver, "--force", "--season", day.year,
                            "--picks-dir", picks)
        if code != 0:
            problems.append(f"saver-state --init exit {code}: {out.strip()[:300]}")
    else:
        raise ValueError(state.kind)
    _set_clock(noon)
    return problems


def _predictions(day: date, top_p: float):
    import pandas as pd
    start = REAL_DATETIME.combine(day, _dtmod.time(19, 5), ET).astimezone(_dtmod.timezone.utc)
    rows = []
    for i, p in enumerate((top_p, round(top_p - 0.01, 4), round(top_p - 0.05, 4))):
        rows.append(dict(batter_name=f"Batter {i}", batter_id=100 + i, team=f"T{i}", lineup=1 + i,
                         pitcher_name=f"Pitcher {i}", pitcher_id=900 + i, p_game_hit=p, flags="",
                         game_pk=70000 + i, game_time=start.isoformat().replace("+00:00", "Z"),
                         pitcher_team=f"U{i}"))
    return pd.DataFrame(rows)


def _config(picks: Path) -> dict:
    return {"orchestrator": {"picks_dir": str(picks), "models_dir": str(picks.parent / "models")},
            "tiers": [{"name": "season_walk"}],
            "scheduler": {"pick_delivery": "dm", "entry_intent": "enter",
                          "contest_state_required": True},
            "health_checks": {"contest_state_expected": True}}


def walk_one(day: date, state: State) -> dict:
    """One date, one contest state, through ``orchestrator.run_and_pick``."""
    from bts import orchestrator, picks as picks_module, strategy
    rec = dict(day=day.isoformat(), state=state.label(), in_season=in_season(day),
               expected_days_left=expected_days_left(day), error=None, setup=[], action=None,
               source=None, objective=None, named=False, days_left=None, days_effective=None,
               batter=None, double=None, no_pick_reason=None, saver_available=None,
               chosen_on_kind=None, reason=None, pick_bar=None, skip_dm=None, skip_bar=None,
               best_p=None)
    frame = _predictions(day, state.top_p)
    statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame["game_pk"]}
    with tempfile.TemporaryDirectory(prefix="season-walk-") as tmp:
        picks = Path(tmp) / "picks"
        picks.mkdir()
        try:
            with offline(REAL_DATETIME.combine(day, _dtmod.time(10), ET)), \
                    patch.object(orchestrator, "run_cascade", lambda tiers, d: (frame.copy(), "season_walk")), \
                    patch.object(picks_module, "get_game_statuses_detailed", lambda d: statuses):
                rec["setup"] = install_state(picks, day, state)
                rec["days_left"] = strategy._days_remaining(day.isoformat())
                _, sel, _ = orchestrator.run_and_pick(_config(picks), day.isoformat())
        except Exception as exc:  # recorded, then reported by check()
            rec["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"
            return rec
    mdp = strategy._load_mdp() or {}
    rec["artifacts"] = dict(base=(mdp.get("base_sha256") or "")[:8],
                            tail=(getattr(mdp.get("tail"), "sha256", None) or "")[:8],
                            base_error=mdp.get("base_error"), tail_error=mdp.get("tail_error"))
    if sel is None:
        rec["no_pick_reason"] = "selection was None"
        return rec
    rec["action"], rec["source"], rec["no_pick_reason"] = sel.action, sel.source, sel.no_pick_reason
    rec["saver_available"] = sel.saver_available
    decision = sel.decision
    if decision is not None:
        rec["objective"] = decision.objective
        rec["days_effective"] = decision.days_effective
        rec["reason"], rec["pick_bar"] = decision.reason, decision.pick_bar
        rec["named"] = sel.action == "skip" and decision.source in {"mdp", "heuristic"}
        if sel.action == "skip":
            # The operator's skip notice, built as run_single_check builds it from this decision.
            from bts.scheduler import build_skip_summary, format_skip_dm
            summary = build_skip_summary(frame, decision.streak, pick_bar=decision.pick_bar,
                                         decision=decision)
            if summary is not None:
                rec["skip_dm"] = format_skip_dm(day.isoformat(), summary)
                rec["skip_bar"], rec["best_p"] = summary.get("bar"), summary["best_p"]
    if sel.pick_result is not None and sel.pick_result.daily is not None:
        daily = sel.pick_result.daily
        rec["batter"] = daily.pick.batter_name
        rec["double"] = daily.double_down.batter_name if daily.double_down else None
        rec["chosen_on_kind"] = (daily.chosen_on or {}).get("kind")
        if state.streak >= LONGEST_STREAK_FROM and rec["days_left"] > 0:
            chosen = picks_module._valid_chosen_on(daily.chosen_on)
            rec["chosen_on_valid"] = bool(
                chosen and chosen["required"] and chosen["kind"] == state.kind
                and (state.kind != "dated" or chosen["dated"]["effective_date"] == rec["day"]))
    return rec


def check(records: list[dict]) -> list[str]:
    problems = []
    for r in records:
        where = f"{r['day']} {r['state']}"
        if r.get("error"):
            problems.append(f"{where}: error {r['error']}")
            continue
        if r.get("setup"):
            problems.append(f"{where}: setup problems {r['setup']}")
        expected = r.get("expected_days_left", expected_days_left(date.fromisoformat(r["day"])))
        if r.get("days_left") != expected:
            problems.append(f"{where}: days left {r.get('days_left')}, expected {expected}")
        state = r["state"].split("/")
        longest = int(state[1]) >= LONGEST_STREAK_FROM and expected > 0
        horizon = LONGEST_STREAK_HORIZON if longest else SEASON_LENGTH
        if r.get("days_effective") is not None and r["days_effective"] != min(expected, horizon):
            problems.append(f"{where}: effective days {r['days_effective']}, expected "
                            f"{min(expected, horizon)}")
        picked = r.get("action") in {"single", "double"} and r.get("batter") is not None
        named_skip = r.get("action") == "skip" and r.get("named") is True
        if not (picked or named_skip):
            problems.append(f"{where}: unnamed empty (action={r.get('action')!r}, "
                            f"no_pick_reason={r.get('no_pick_reason')!r})")
        if r.get("source") not in (None, "mdp"):
            problems.append(f"{where}: decision source {r.get('source')!r} (shipped policy not used)")
        art = r.get("artifacts")
        if art and (not art["base"].startswith(BASE_SHA_PREFIX) or not art["tail"].startswith(TAIL_SHA_PREFIX)
                    or art["base_error"] or art["tail_error"]):
            problems.append(f"{where}: policy artifacts not the shipped ones: {art}")
        if longest:
            want_action = "double" if expected <= 3 else "single"
            if r.get("action") != want_action:
                problems.append(f"{where}: longest-streak action {r.get('action')!r}, expected {want_action!r}")
            if r.get("objective") != OBJECTIVE_LONGEST:
                problems.append(f"{where}: longest-streak objective {r.get('objective')!r}")
            if picked and (not r.get("chosen_on_valid") or bool(r.get("double")) != (r["action"] == "double")):
                problems.append(f"{where}: longest-streak choice has invalid chosen_on or selected slots")
        if state[0] in {"dated", "automatic"} and r.get("saver_available") is not None:
            want = expected_saver_in_hand(State(state[0], int(state[1]), int(state[2]), state[3], float(state[4])))
            if r["saver_available"] is not want:
                problems.append(f"{where}: saver in hand {r['saver_available']}, expected {want}")
        if picked and r.get("chosen_on_kind") not in (None, state[0]):
            problems.append(f"{where}: chosen on {r.get('chosen_on_kind')!r}, expected {state[0]!r}")
        dm = r.get("skip_dm") or ""
        if r.get("in_season") and r.get("action") == "skip" and "below the" in dm and r.get("best_p") is not None:
            bar = r.get("skip_bar") if r.get("skip_bar") is not None else 0.80   # "~80%" when no bar
            if r["best_p"] >= bar:
                problems.append(f"{where}: skip message calls the best ({r['best_p']:.1%}) below a bar "
                                f"it clears ({bar:.1%}): {dm}")
        day = date.fromisoformat(r["day"])
        if day > LAST_GAME_DAY and r.get("action") != "skip":
            problems.append(f"{where}: after the finale but action {r.get('action')!r}")
        if day == LAST_GAME_DAY and state[1] == "56" and r.get("action") == "skip":
            problems.append(f"{where}: last game day at streak 56 skipped")
    return problems


def boundary_checks() -> dict:
    """The stored boundary, the year boundary and the season-end fallback."""
    from bts import strategy
    from bts.contest_state import ContestStateError, load_decision_streak_state
    from bts.health import tail_policy
    out: dict = {"problems": list(table_boundary_problems())}
    out["season_end_2028"] = strategy.season_end_date("2028-05-01")
    out["warn_2028"] = [a.message for a in tail_policy.season_end_alerts(today=date(2028, 4, 1))]
    out["warn_2027"] = [a.message for a in tail_policy.season_end_alerts(today=date(2027, 4, 1))]
    out["days_2026_12_31"] = strategy._days_remaining("2026-12-31")
    out["days_2027_01_01"] = strategy._days_remaining("2027-01-01")
    out["days_2028_09_27"] = strategy._days_remaining("2028-09-27")
    in_window = tail_policy.check(base_path=REPO / "data/models/mdp_policy.npz",
                                  tail_path=REPO / "data/models/mdp_tail_policy.npz",
                                  today=date(2027, 9, 10))
    out["tail_policy_alerts_2027_09_10"] = [f"{a.level} {a.message}" for a in in_window]
    if in_window:
        out["problems"].append(f"tail_policy health check alerts on 2027-09-10: {out['tail_policy_alerts_2027_09_10']}")
    if out["season_end_2028"] != "2028-09-28":
        out["problems"].append(f"2028 fallback boundary {out['season_end_2028']}")
    if len(out["warn_2028"]) != 1 or "not configured" not in out["warn_2028"][0]:
        out["problems"].append(f"2028 WARN {out['warn_2028']}")
    if out["warn_2027"]:
        out["problems"].append(f"2027 WARN {out['warn_2027']}")
    if out["days_2026_12_31"] != 0 or out["days_2027_01_01"] != 269:
        out["problems"].append(f"year-boundary day counts {out['days_2026_12_31']}, {out['days_2027_01_01']}")
    # The 2027 automatic switch at the year boundary: accepted at 23:59 ET on 12-31, refused after midnight ET.
    with tempfile.TemporaryDirectory(prefix="season-walk-year-") as tmp:
        picks = Path(tmp) / "picks"
        picks.mkdir()
        evening = REAL_DATETIME(2027, 12, 31, 20, tzinfo=ET)
        with offline(evening):
            state = State("automatic", 12, 20, "used", 0.96)
            out["year_setup"] = install_state(picks, date(2027, 12, 31), state)
            for key, at in (("switch_2027_at_2027_12_31_2359", REAL_DATETIME(2027, 12, 31, 23, 59, tzinfo=ET)),
                            ("switch_2027_at_2028_01_01_0001", REAL_DATETIME(2028, 1, 1, 0, 1, tzinfo=ET))):
                _set_clock(at)
                try:
                    load_decision_streak_state(picks, require_contest_state=True, now=at)
                    out[key] = "accepted"
                except ContestStateError as exc:
                    out[key] = "refused: " + str(exc).splitlines()[0][:200]
    if out["year_setup"]:
        out["problems"].append(f"year-boundary setup {out['year_setup']}")
    if out["switch_2027_at_2027_12_31_2359"] != "accepted":
        out["problems"].append(f"2027 switch at 2027-12-31 23:59 ET: {out['switch_2027_at_2027_12_31_2359']}")
    if not out["switch_2027_at_2028_01_01_0001"].startswith("refused"):
        out["problems"].append("2027 switch still accepted on 2028-01-01")
    return out


def walk(start: date = WALK_START, end: date = WALK_END, states=DEFAULT_STATES, progress=None) -> list[dict]:
    records = []
    day = start
    while day <= end:
        for state in states:
            records.append(walk_one(day, state))
        if progress:
            progress(day)
        day += timedelta(days=1)
    return records


def summarize(records: list[dict]) -> dict:
    by_day: dict = {}
    for r in records:
        slot = by_day.setdefault(r["day"], {"single": 0, "double": 0, "skip": 0, "other": 0,
                                            "days_left": r.get("days_left")})
        slot[r["action"] if r.get("action") in {"single", "double", "skip"} else "other"] += 1
    return by_day


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", type=date.fromisoformat, default=WALK_START)
    parser.add_argument("--end", type=date.fromisoformat, default=WALK_END)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    os.chdir(REPO)   # the policy loader resolves data/models/ relative to the working directory
    records = walk(args.start, args.end,
                   progress=lambda d: print(d.isoformat(), flush=True) if d.day == 1 else None)
    boundary = boundary_checks()
    problems = check(records) + boundary["problems"]
    args.out.write_text(json.dumps({"start": args.start.isoformat(), "end": args.end.isoformat(),
                                    "states": [s.label() for s in DEFAULT_STATES],
                                    "runs": len(records), "problems": problems, "boundary": boundary,
                                    "by_day": summarize(records), "records": records},
                                   indent=1, default=str))
    print(f"runs={len(records)} problems={len(problems)}")
    for p in problems[:50]:
        print("PROBLEM", p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
