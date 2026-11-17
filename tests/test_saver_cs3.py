"""CS3 contract: real flag files, policy artifact, health, dashboard and CLI."""
import json
from datetime import date, datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

from bts.cli import cli
from bts.contest_state import load_decision_streak_state
from bts.health.contest_state import check
from bts.saver_state import SaverState, load_saver_state, maybe_auto_earn_saver

NOW = datetime(2027, 7, 15, 16, tzinfo=timezone.utc)
STATES = ["uninitialized", "not_earned", "active", "used"]


def _flag(picks, state, season=2027):
    folder = picks / "account_state"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "saver_state.json"
    if state != "uninitialized":
        path.write_text(json.dumps({"state": state, "season": season}))
    return path


def _contest(picks, streak=7, source="2026-06-18"):
    folder = picks / "account_state"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "contest_streak.json"
    path.write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1", "active_streak": streak,
        "best_streak": streak, "source": "mlb_bts_profile", "source_date": source,
    }))
    return path


@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("streak", [0, 9, 10, 15, 16, 57])
def test_policy_saver_table(state, streak):
    from bts.saver_state import policy_saver_available
    expected = state == "active" or (state == "not_earned" and streak < 10)
    assert policy_saver_available(SaverState(state, 2027), streak) is expected


def test_decision_does_not_reuse_last_seasons_active_flag(tmp_path):
    _contest(tmp_path)
    _flag(tmp_path, "active", 2026)
    state = load_decision_streak_state(tmp_path, now=NOW)
    assert state.saver_available is False
    assert state.saver_flag_state == "uninitialized"


@pytest.mark.parametrize("flag", STATES)
@pytest.mark.parametrize("streak", [7, 10])
def test_contest_decision_uses_current_flag_and_policy_mapping(tmp_path, flag, streak):
    _contest(tmp_path, streak)
    _flag(tmp_path, flag)
    state = load_decision_streak_state(tmp_path, now=NOW)
    assert state.streak == streak
    assert state.saver_available is (flag == "active" or (flag == "not_earned" and streak < 10))
    assert state.saver_flag_state == flag


@pytest.mark.parametrize("flag", STATES)
@pytest.mark.parametrize("streak", [7, 12])
def test_model_decision_uses_its_returned_streak(tmp_path, flag, streak):
    _flag(tmp_path, flag)
    (tmp_path / "streak.json").write_text(json.dumps({
        "season": 2027, "streak": streak, "saver_available": False,
    }))
    state = load_decision_streak_state(tmp_path, now=NOW)
    assert state.source == "model" and state.streak == streak
    assert state.saver_available is (flag == "active" or (flag == "not_earned" and streak < 10))
    assert state.saver_flag_state == flag


def test_reserved_saver_changes_the_real_policy_action(tmp_path):
    from bts.simulate.mdp import load_policy
    from bts.strategy import DecisionContext, resolve_policy_decision
    _contest(tmp_path, 3)
    _flag(tmp_path, "not_earned")
    state = load_decision_streak_state(tmp_path, now=NOW)
    table, boundaries, length = load_policy("data/models/mdp_policy.npz")
    ctx = DecisionContext(.80, .80, True, "2027-08-29", True, {
        "policy_table": table, "boundaries": boundaries, "season_length": length,
    })
    old = resolve_policy_decision(ctx, state.streak, load_saver_state(tmp_path, season=2027).is_available)
    new = resolve_policy_decision(ctx, state.streak, state.saver_available)
    assert old.source == new.source == "mdp"
    assert old.action == "skip"
    assert new.action == "double"


@pytest.mark.parametrize("flag", ["missing", "stale", "not_earned", "active", "used"])
@pytest.mark.parametrize("streak", [0, 9, 10, 15, 16])
@pytest.mark.parametrize("expected", [False, True])
def test_health_saver_matrix(tmp_path, flag, streak, expected):
    _contest(tmp_path, streak, "2027-06-18")
    if flag == "stale":
        _flag(tmp_path, "active", 2026)
    elif flag != "missing":
        _flag(tmp_path, flag)
    warn = ((flag in {"missing", "stale"} and (expected or 10 <= streak <= 15))
            or (flag == "not_earned" and 10 <= streak <= 15))
    alerts = check(tmp_path, expected=expected, now=NOW)
    assert [(a.level, a.source) for a in alerts] == ([("WARN", "saver_state")] if warn else [])
    if warn:
        message = alerts[0].message
        assert "2027" in message and str(streak) in message
        assert "counting the saver as spent" in message
        command = "bts saver-state --use --season 2027" if flag == "not_earned" else "bts saver-state --init used --season 2027"
        assert command in message


def test_health_reads_current_season_despite_prior_source_date(tmp_path):
    _contest(tmp_path, 12)
    _flag(tmp_path, "active", 2026)
    alerts = check(tmp_path, expected=True, now=NOW)
    assert [(a.level, a.source) for a in alerts] == [("WARN", "saver_state")]
    _flag(tmp_path, "active", 2027)
    assert check(tmp_path, expected=True, now=NOW) == []


@pytest.mark.parametrize("real_check", [False, True])
def test_saver_warn_gets_first_occurrence_attention_but_coverage_does_not(tmp_path, real_check):
    from bts.health.alert import Alert
    from bts.health.attention import apply_warn_attention_policy
    if real_check:
        _contest(tmp_path, 12, "2027-06-18")
        _flag(tmp_path, "not_earned")
        for day in ["2027-06-19", "2027-06-20"]:
            (tmp_path / f"{day}.json").write_text(json.dumps({"result": "hit"}))
        alerts = check(tmp_path, expected=True, now=NOW)
        assert {(a.level, a.source) for a in alerts} == {("WARN", "contest_state"), ("WARN", "saver_state")}
    else:
        alerts = [Alert("WARN", "saver_state", "saver needs reconciliation"),
                  Alert("WARN", "contest_state", "contest coverage lag: 2 settled picks")]
    critical, attention = apply_warn_attention_policy(
        alerts, state_path=tmp_path / "attention.json", today=NOW.date())
    assert critical == []
    assert [a.source for a in attention] == ["saver_state"]
    assert "1st observed day" in attention[0].message


@pytest.mark.parametrize("flag_year,code", [(2026, 409), (2027, 200)])
def test_dashboard_post_targets_current_year_without_consuming_prior_flag(tmp_path, flag_year, code):
    from bts.web import saver_dashboard_context, saver_transition_response
    _contest(tmp_path)
    flag = _flag(tmp_path, "active", flag_year)
    before = flag.read_bytes()
    ctx = saver_dashboard_context(tmp_path, now=NOW)
    assert ctx.state == ("active" if flag_year == 2027 else "uninitialized")
    status, _ = saver_transition_response(tmp_path, expected_prior="active", new_state="used",
                                          same_origin=True, now=NOW)
    assert status == code
    if flag_year == 2026:
        assert flag.read_bytes() == before
    else:
        body = json.loads(flag.read_text())
        assert body["season"] == 2027 and body["state"] == "used"


def _page_clock(monkeypatch):
    import bts.web as web
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)
    monkeypatch.setattr(web, "datetime", Clock)
    monkeypatch.setattr(web, "fetch_bluesky_posts", lambda: [])


@pytest.mark.parametrize("bad", ["missing", "unreadable", "non_utf8", "non_object", "refused", "directory", "malformed"])
def test_dashboard_survives_contest_files_and_post_never_reads_them(tmp_path, monkeypatch, bad):
    import bts.contest_state as cs
    import bts.web as web
    _flag(tmp_path, "active")
    path = _contest(tmp_path)
    if bad == "missing":
        path.unlink()
    elif bad == "unreadable":
        path.chmod(0)
    elif bad == "directory":
        path.unlink()
        path.mkdir()
    else:
        path.write_bytes({"non_utf8": b"\xff", "non_object": b"[]", "malformed": b"{",
                          "refused": b'{"active_streak":58,"best_streak":58}'}[bad])
    try:
        ctx = web.saver_dashboard_context(tmp_path, now=NOW)
        assert ctx.state == "active" and ctx.warning is False and ctx.nudge is False
        monkeypatch.setattr(web, "PICKS_DIR", tmp_path)
        _page_clock(monkeypatch)
        assert "Streak Saver: AVAILABLE" in web.render_page()
        def forbidden(*a, **kw):
            raise AssertionError("POST must not load the contest observation")
        monkeypatch.setattr(cs, "load_contest_streak_state", forbidden)
        status, _ = web.saver_transition_response(tmp_path, expected_prior="active", new_state="used",
                                                  same_origin=True, now=NOW)
        assert status == 200
        status, _ = web.saver_transition_response(tmp_path, expected_prior="active", new_state="used",
                                                  same_origin=True, now=NOW)
        assert status == 409
    finally:
        if bad == "unreadable":
            path.chmod(0o600)


def test_bad_contest_suppresses_even_a_likely_save_nudge(tmp_path):
    from bts.web import saver_dashboard_context
    _flag(tmp_path, "active")
    path = _contest(tmp_path, 12, "2027-06-18")
    snapshot = {"predictions": [
        {"roundId": 1, "result": "hit", "streak": 12},
        {"roundId": 2, "result": "not_hit", "streak": 12},
    ]}
    (path.parent / "contest_ledger.jsonl").write_text("\n".join([json.dumps(snapshot)] * 2))
    assert saver_dashboard_context(tmp_path, now=NOW).nudge is True
    path.write_bytes(b"\xff")
    ctx = saver_dashboard_context(tmp_path, now=NOW)
    assert ctx.state == "active" and ctx.warning is False and ctx.nudge is False


@pytest.mark.parametrize("streak", [7, 10, 12, None])
def test_page_reserved_label_uses_existing_decision_streak(tmp_path, monkeypatch, streak):
    import bts.web as web
    _contest(tmp_path, 7)
    _flag(tmp_path, "not_earned")
    monkeypatch.setattr(web, "PICKS_DIR", tmp_path)
    state = None if streak is None else SimpleNamespace(streak=streak, model_streak=streak, source="model")
    monkeypatch.setattr(web, "load_dashboard_decision_state", lambda: state)
    _page_clock(monkeypatch)
    page = web.render_page()
    assert "Streak Saver: " in page
    assert ("reserved (not yet earned)" in page) is (streak is not None and streak < 10)
    if streak is None or streak >= 10:
        assert "Streak Saver: not yet earned" in page


@pytest.mark.parametrize("now,season", [(datetime(2027, 1, 1, 2, tzinfo=timezone.utc), 2026),
                                        (datetime(2027, 1, 1, 5, tzinfo=timezone.utc), 2027)])
def test_season_is_the_et_year_at_midnight(tmp_path, now, season):
    from bts.web import saver_dashboard_context
    _contest(tmp_path, 7)
    _flag(tmp_path, "active", 2026)
    state = load_decision_streak_state(tmp_path, now=now)
    assert state.saver_available is (season == 2026)
    assert saver_dashboard_context(tmp_path, now=now).state == ("active" if season == 2026 else "uninitialized")


@pytest.mark.parametrize("prior", STATES)
@pytest.mark.parametrize("option", ["--use", "--undo"])
def test_cli_transition_matrix_and_one_audit_per_attempt(tmp_path, prior, option):
    _flag(tmp_path, prior)
    result = CliRunner().invoke(cli, ["saver-state", "--picks-dir", str(tmp_path), "--season", "2027", option])
    success = prior in {"not_earned", "active", "used"} if option == "--use" else prior == "used"
    state = ("used" if option == "--use" else "active") if success else prior
    assert result.exit_code == (0 if success else 1), result.output
    assert load_saver_state(tmp_path, season=2027).state == state
    assert f"saver_state: {state} (season 2027)" in result.output
    assert result.output.splitlines()[-1] == f"saver_state: {state} (season 2027)"
    if prior == "used" and option == "--use":
        assert "already used" in result.output.lower()
    if prior == "uninitialized" and option == "--use":
        assert "bts saver-state --init used --season 2027" in result.output
    audit = (tmp_path / "account_state/saver_transitions.jsonl").read_text().splitlines()
    assert len(audit) == 1
    assert json.loads(audit[0])["source"] == "cli"


def test_between_fetch_use_cannot_be_resurrected_by_auto_earn(tmp_path):
    _flag(tmp_path, "not_earned")
    result = CliRunner().invoke(cli, ["saver-state", "--picks-dir", str(tmp_path), "--season", "2027", "--use"])
    assert result.exit_code == 0, result.output
    assert load_saver_state(tmp_path, season=2027).state == "used"
    maybe_auto_earn_saver(tmp_path, best_streak=10, season=2027)
    assert load_saver_state(tmp_path, season=2027).state == "used"


def _observe(monkeypatch, picks, command, source, best):
    args = [command, "--picks-dir", str(picks)]
    if command == "fetch-contest-streak":
        from tests.test_contest_fetch import _mock_fetch
        predictions = [] if source is None else [{"roundId": 1, "result": "hit"}]
        rounds = {} if source is None else {1: source}
        _mock_fetch(monkeypatch, {"activeStreak": 0, "seasonBestStreak": best, "predictions": predictions}, rounds)
        args += ["--expected-username", "stonehengee"]
    else:
        args += ["--streak", "0", "--best-streak", str(best)]
        if source is not None:
            args += ["--source-date", source.isoformat()]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("best", [0, 12])
def test_fetch_without_settled_source_never_creates_a_flag(tmp_path, monkeypatch, best):
    _observe(monkeypatch, tmp_path, "fetch-contest-streak", None, best)
    assert not (tmp_path / "account_state/saver_state.json").exists()
    assert json.loads((tmp_path / "account_state/contest_streak.json").read_text())["source_date"] is None


@pytest.mark.parametrize("command", ["fetch-contest-streak", "set-contest-streak"])
@pytest.mark.parametrize("best", [0, 12])
@pytest.mark.parametrize("seed_offset", [None, -1, 0])
@pytest.mark.parametrize("source_offset", [-1, 1])
def test_other_year_observation_does_not_touch_flag(tmp_path, monkeypatch, command, best, seed_offset, source_offset):
    year = datetime.now(ZoneInfo("America/New_York")).year
    flag = tmp_path / "account_state/saver_state.json"
    if seed_offset is not None:
        _flag(tmp_path, "not_earned", year + seed_offset)
    before = flag.read_bytes() if flag.exists() else None
    _observe(monkeypatch, tmp_path, command, date(year + source_offset, 6, 18), best)
    assert (flag.read_bytes() if flag.exists() else None) == before


@pytest.mark.parametrize("command", ["fetch-contest-streak", "set-contest-streak"])
def test_current_year_observation_initializes_then_earns(tmp_path, monkeypatch, command):
    year = datetime.now(ZoneInfo("America/New_York")).year
    _observe(monkeypatch, tmp_path, command, date(year, 6, 18), 7)
    assert load_saver_state(tmp_path, season=year).state == "not_earned"
    _observe(monkeypatch, tmp_path, command, date(year, 6, 19), 10)
    assert load_saver_state(tmp_path, season=year).state == "active"
    assert json.loads((tmp_path / "account_state/saver_state.json").read_text())["season"] == year


def test_manual_omitted_source_is_today_not_null(tmp_path, monkeypatch):
    year = datetime.now(ZoneInfo("America/New_York")).year
    _observe(monkeypatch, tmp_path, "set-contest-streak", None, 7)
    assert load_saver_state(tmp_path, season=year).state == "not_earned"
    source = json.loads((tmp_path / "account_state/contest_streak.manual.json").read_text())["source_date"]
    assert date.fromisoformat(source).year == year


# --- added by the lead after review round 1 ---

@pytest.mark.parametrize("value", [None, "7", [], {}, True, -1, 7.0], ids=["null", "str", "list", "object", "bool", "negative", "float"])
@pytest.mark.parametrize("flag", ["uninitialized", "not_earned", "active", "used"])
def test_model_only_branch_refuses_a_model_streak_that_is_not_a_count(tmp_path, value, flag):
    """Codex r1 B1: with no contest observation the decision streak IS the model streak, and the
    policy saver compares it with 10. A readable streak.json whose streak is not a non-negative
    int is corrupt model state: a ContestStateError, never a TypeError and never an invented 0."""
    from bts.contest_state import ContestStateError
    _flag(tmp_path, flag)
    (tmp_path / "streak.json").write_text(json.dumps({"season": 2027, "streak": value}))
    with pytest.raises(ContestStateError, match="model state is corrupt"):
        load_decision_streak_state(tmp_path, now=NOW)
    # A contest observation does not depend on the model file (change set 1): unchanged.
    _contest(tmp_path, 7, "2027-07-14")
    state = load_decision_streak_state(tmp_path, now=NOW)
    assert (state.source, state.streak) == ("contest", 7)


def _commands(message):
    return [part for part in message.split("`")[1::2] if part.startswith("bts saver-state")]


def _paste(picks, command):
    return CliRunner().invoke(cli, [*command.split()[1:], "--picks-dir", str(picks)])


@pytest.mark.parametrize("flag", ["missing", "stale"])
@pytest.mark.parametrize("streak", [0, 12])
def test_uninitialized_warning_gives_a_command_for_each_thing_the_operator_can_find(tmp_path, flag, streak):
    """The lead's finding in review round 1: the message gave only the command that marks the saver
    USED, and it is sent every day at the start of a season, when the right answer is not_earned."""
    _contest(tmp_path, streak, "2027-06-18")
    if flag == "stale":
        _flag(tmp_path, "active", 2026)
    (alert,) = check(tmp_path, expected=True, now=NOW)
    commands = _commands(alert.message)
    assert commands == ["bts saver-state --init not_earned --season 2027",
                        "bts saver-state --init active --season 2027",
                        "bts saver-state --init used --season 2027"]
    for command, state in zip(commands, ["not_earned", "active", "used"]):
        picks = tmp_path / state
        _contest(picks, streak, "2027-06-18")
        if flag == "stale":
            _flag(picks, "active", 2026)
        result = _paste(picks, command)
        assert result.exit_code == 0, result.output
        assert load_saver_state(picks, season=2027).state == state


def test_not_earned_in_zone_warning_gives_both_commands(tmp_path):
    _contest(tmp_path, 12, "2027-06-18")
    _flag(tmp_path, "not_earned")
    (alert,) = check(tmp_path, expected=False, now=NOW)
    commands = _commands(alert.message)
    assert commands == ["bts saver-state --init active --force --season 2027",
                        "bts saver-state --use --season 2027"]
    for command, state in zip(commands, ["active", "used"]):
        picks = tmp_path / state
        _flag(picks, "not_earned")
        result = _paste(picks, command)
        assert result.exit_code == 0, result.output
        assert load_saver_state(picks, season=2027).state == state


def test_use_from_not_earned_says_how_to_take_it_back(tmp_path):
    """Codex r1 note: --undo after a mistaken --use ends at `active`, a saver that was never earned.
    The command that really takes it back is printed when --use succeeds from not_earned."""
    _flag(tmp_path, "not_earned")
    result = CliRunner().invoke(cli, ["saver-state", "--picks-dir", str(tmp_path), "--season", "2027", "--use"])
    assert result.exit_code == 0, result.output
    revert = "bts saver-state --init not_earned --force --season 2027"
    assert revert in result.output
    assert result.output.splitlines()[-1] == "saver_state: used (season 2027)"
    assert _paste(tmp_path, revert).exit_code == 0
    assert load_saver_state(tmp_path, season=2027).state == "not_earned"
    # From active there is nothing to explain: --undo is the exact inverse.
    _flag(tmp_path / "b", "active")
    result = CliRunner().invoke(cli, ["saver-state", "--picks-dir", str(tmp_path / "b"), "--season", "2027", "--use"])
    assert "--init not_earned" not in result.output


def test_not_earned_warning_covers_a_stale_contest_streak_at_season_start(tmp_path):
    """Codex r2 N1: on 2027-03-24 the contest file can still hold September's streak 12. After the
    operator correctly runs --init not_earned the warning must not offer only "earned" or "used":
    MLB shows streak 0 and an unearned saver, and the thing to fix is the contest streak."""
    opening = datetime(2027, 3, 24, 20, tzinfo=timezone.utc)
    _contest(tmp_path, 12, "2026-09-28")
    _flag(tmp_path, "used", 2026)
    (first,) = [a for a in check(tmp_path, expected=True, now=opening) if a.source == "saver_state"]
    assert _paste(tmp_path, _commands(first.message)[0]).exit_code == 0
    assert load_saver_state(tmp_path, season=2027).state == "not_earned"
    (second,) = [a for a in check(tmp_path, expected=True, now=opening) if a.source == "saver_state"]
    message = second.message
    assert "not earned yet" in message and "stale" in message
    assert "leave the flag" in message and "bts set-contest-streak" in message
    # The flag commands are unchanged, and still exactly two.
    assert _commands(message) == ["bts saver-state --init active --force --season 2027",
                                  "bts saver-state --use --season 2027"]
    # Following the third case silences the warning without touching the flag.
    result = CliRunner().invoke(cli, ["set-contest-streak", "--picks-dir", str(tmp_path), "--streak", "0",
                                      "--best-streak", "0", "--source-date", "2027-03-24"])
    assert result.exit_code == 0, result.output
    assert load_saver_state(tmp_path, season=2027).state == "not_earned"
