"""CLI and receipt contract for an unavailable season replay."""

from contextlib import contextmanager
from datetime import datetime
import importlib.util
import io
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
from zoneinfo import ZoneInfo

from click.testing import CliRunner
import pytest

from bts import picks, reconcile_receipt


ET = ZoneInfo("America/New_York")
RUN = datetime(2027, 4, 20, 2, tzinfo=ET)
TODAY = RUN.date().isoformat()
PAST = "2027-04-16"
PENDING = "no final result on the pick"
BASE = "88b69a16"


@pytest.fixture(autouse=True)
def deadline():
    class DeadlineExceeded(BaseException):
        pass

    def expire(*args):
        raise DeadlineExceeded("test exceeded its I/O deadline")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(30)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def invoke(root, monkeypatch, *, clock=lambda: RUN, command=None, lookback=0, receipt_class=None,
           run_id=None):
    from bts.cli import cli

    receipt_class = receipt_class or reconcile_receipt.ReconcileReceipt

    def make_receipt(**kwargs):
        receipt = receipt_class(clock=clock, lookback_days=kwargs["lookback_days"])
        if run_id is not None:
            receipt.run_id = run_id
        return receipt

    with monkeypatch.context() as scoped:
        scoped.setattr(reconcile_receipt, "ReconcileReceipt", make_receipt)
        return CliRunner().invoke(command or cli, [
            "reconcile", "--picks-dir", str(root), "--lookback", str(lookback),
        ])


def only_receipt(root):
    day, = reconcile_receipt.receipts_root(root).iterdir()
    receipt, = day.glob("*.json")
    return json.loads(receipt.read_bytes())


def warning(day, reason):
    return f"Season replay unavailable for {day} ({reason}); streak retained, not recalculated.\n"


@pytest.fixture(scope="session")
def base_modules(tmp_path_factory):
    """Import the three unmodified modules from a local git archive of 88b69a16."""
    root = tmp_path_factory.mktemp("c3v-base")
    repo = Path(__file__).resolve().parents[1]
    files = [f"src/bts/{name}.py" for name in ("picks", "reconcile_receipt", "cli")]
    archive = subprocess.check_output(["git", "-C", str(repo), "archive", BASE, *files], timeout=10)
    modules = []
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle, pytest.MonkeyPatch.context() as scoped:
        for member in files:
            source = bundle.extractfile(member).read()
            file = root / member
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(source)
            spec = importlib.util.spec_from_file_location(f"_wr_c3v_base_{file.stem}", file)
            module = importlib.util.module_from_spec(spec)
            scoped.setitem(sys.modules, spec.name, module)
            spec.loader.exec_module(module)
            modules.append(module)
    return modules


class PinnedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return RUN.astimezone(tz)


def outcome(call):
    try:
        return call()
    except Exception as exc:
        return type(exc), exc.args


def compare_reconcile(root, monkeypatch, base_modules, *, clock=lambda: RUN, lookback=0,
                      result_feed=None):
    old_picks, old_receipt, old_cli = base_modules
    old_root = root.parent.parent / "base" / "picks"
    shutil.copytree(root, old_root, symlinks=True)
    today = clock().astimezone(ET).date()
    with monkeypatch.context() as scoped:
        for module in (old_picks, picks):
            scoped.setattr(module, "datetime", PinnedDatetime)
            scoped.setattr(module, "_git_head_sha", lambda *args: BASE)
            if result_feed is not None:
                scoped.setattr(module, "resolve_daily_slot_results", result_feed)
        args = (today.year, today.isoformat())
        expected_fold = outcome(lambda: old_picks._replay_season_streak(old_root, *args))
        assert outcome(lambda: picks._replay_season_streak(root, *args)) == expected_fold
        with scoped.context() as archived:
            archived.setitem(sys.modules, "bts.picks", old_picks)
            expected = invoke(old_root, archived, clock=clock, command=old_cli.cli,
                              receipt_class=old_receipt.ReconcileReceipt, run_id="a" * 32, lookback=lookback)
        actual = invoke(root, scoped, clock=clock, run_id="a" * 32, lookback=lookback)

    assert actual.exit_code == expected.exit_code, (actual.output, expected.output)
    def exception_signature(result):
        exc = result.exception
        return None if exc is None else (type(exc), exc.args)

    assert exception_signature(actual) == exception_signature(expected)
    assert (root / "streak.json").read_bytes() == (old_root / "streak.json").read_bytes()
    if only_receipt(old_root)["replay"]["state"] == "saved":
        assert actual.stdout.encode() == expected.stdout.encode()
        assert actual.stderr == expected.stderr
        new_path, = reconcile_receipt.receipts_root(root).glob("*/*.json")
        old_path, = reconcile_receipt.receipts_root(old_root).glob("*/*.json")
        assert new_path.read_bytes() == old_path.read_bytes()
    return actual, expected_fold


@pytest.mark.parametrize(("kind", "day", "reason"), [
    ("corrupt", PAST, "pick file unreadable or not valid JSON"),
    ("applied", TODAY, "today's result is applied to streak.json but not yet saved on the pick"),
    ("pending", PAST, PENDING),
    ("lost_hit", PAST, "hit pick could not be loaded"),
])
def test_blocked_replay_names_first_blocker(tmp_path, monkeypatch, kind, day, reason):
    root = tmp_path / "picks"
    root.mkdir()
    target = root / "streak.json"
    before = json.dumps({"streak": 7, "season": RUN.year, "applied_dates": [TODAY]}).encode()
    target.write_bytes(before)
    path = root / f"{day}.json"
    path.write_text({"corrupt": "{", "lost_hit": '{"result":"hit"}'}.get(kind, "{}"))
    if kind == "lost_hit":
        monkeypatch.setattr(picks, "load_pick", lambda *_: None)
    if day < TODAY:
        (root / f"{TODAY}.json").write_text("{later blocker")

    result = invoke(root, monkeypatch)
    assert result.exit_code == 0, result.output
    assert target.read_bytes() == before
    assert result.stderr == warning(day, reason)
    assert result.stdout == (
        "Reconciling last 0 days of picks...\nNo scoring changes detected. Streak retained: 7\n"
    )
    assert only_receipt(root)["replay"] == {"state": "unavailable", "blocked_date": day, "reason": reason}


def test_completed_replay_is_byte_identical(tmp_path, monkeypatch, base_modules):
    from tests.test_wr_c3v_corpus import dump, pick_data, seed_streak

    root = tmp_path / "current" / "picks"
    target = seed_streak(root)
    before = target.read_bytes()
    dump(root / f"{PAST}.json", pick_data(PAST, "hit"))
    result, fold = compare_reconcile(root, monkeypatch, base_modules)
    assert result.exit_code == 0, result.output
    assert fold == (1, True)
    assert only_receipt(root)["replay"] == {"state": "saved", "streak": 1, "saver_available": True}
    assert target.read_bytes() != before


@pytest.mark.parametrize(("suffix", "rendered_suffix"), [("", ""), ("\n", r"\n")], ids=["canonical", "lf"])
def test_blocker_filename_renders_as_one_line(tmp_path, monkeypatch, base_modules, suffix, rendered_suffix):
    from tests.test_wr_c3v_corpus import seed_streak

    root = tmp_path / "current" / "picks"
    target = seed_streak(root, streak=9)
    before = target.read_bytes()
    date = PAST + suffix
    (root / f"{date}.json").write_text("{}")
    result, fold = compare_reconcile(root, monkeypatch, base_modules, lookback=8)
    assert fold is None
    assert result.exit_code == 0, result.output
    assert result.stderr == warning(PAST + rendered_suffix, PENDING)
    assert result.stdout == (
        "Reconciling last 8 days of picks...\nNo scoring changes detected. Streak retained: 9\n"
    )
    assert only_receipt(root)["replay"] == {"state": "unavailable", "blocked_date": date, "reason": PENDING}
    assert target.read_bytes() == before


def test_blocker_escapes_control_and_unicode_in_both_fields(tmp_path, monkeypatch):
    date = PAST + "\r\N{ARABIC-INDIC DIGIT SIX}\x7f"
    reason = "synthetic\nreason\x1b\N{LINE SEPARATOR}"

    def blocked(*args, receipt, **kwargs):
        receipt.replayed(None, blocked_date=date, reason=reason)
        return []

    monkeypatch.setattr(picks, "reconcile_results", blocked)
    root = tmp_path / "picks"
    result = invoke(root, monkeypatch)
    assert result.exit_code == 0, result.output
    assert result.stderr == warning(PAST + r"\r\u0666\x7f", r"synthetic\nreason\x1b\u2028")
    assert only_receipt(root)["replay"] == {"state": "unavailable", "blocked_date": date, "reason": reason}


def test_directory_at_pick_path_has_base_read_failure(tmp_path, monkeypatch, base_modules):
    from tests.test_wr_c3v_corpus import seed_streak

    root = tmp_path / "current" / "picks"
    target = seed_streak(root)
    before = target.read_bytes()
    (root / f"{PAST}.json").mkdir()
    result, fold = compare_reconcile(root, monkeypatch, base_modules)
    assert fold is None
    assert result.exit_code == 0, result.output
    assert result.stderr == warning(PAST, "pick file unreadable or not valid JSON")
    assert target.read_bytes() == before


def test_summary_without_a_replay_receipt_uses_existing_wording(tmp_path, monkeypatch):
    monkeypatch.setattr(picks, "reconcile_results", lambda *args, **kwargs: [])
    root = tmp_path / "picks"
    result = invoke(root, monkeypatch)
    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert result.stdout == "Reconciling last 0 days of picks...\nNo scoring changes detected. Streak: 0\n"
    assert only_receipt(root)["replay"] == {"state": "not_reached"}


def test_interrupt_during_replay_publishes_base_receipt(tmp_path, monkeypatch, base_modules):
    from tests.test_wr_c3v_corpus import seed_streak

    root = tmp_path / "current" / "picks"
    target = seed_streak(root)
    before = target.read_bytes()
    pick_path = root / f"{PAST}.json"
    pick_path.write_text('{"result":"hit"}')
    read_text = Path.read_text

    def interrupt(path, *args, **kwargs):
        if path.name == pick_path.name:
            raise KeyboardInterrupt
        return read_text(path, *args, **kwargs)

    old_picks, old_receipt, old_cli = base_modules
    old_root = tmp_path / "base" / "picks"
    shutil.copytree(root, old_root)
    monkeypatch.setattr(Path, "read_text", interrupt)
    for module in (picks, old_picks):
        monkeypatch.setattr(module, "_git_head_sha", lambda *args: BASE)
    with monkeypatch.context() as archived:
        archived.setitem(sys.modules, "bts.picks", old_picks)
        expected = invoke(old_root, archived, command=old_cli.cli,
                          receipt_class=old_receipt.ReconcileReceipt, run_id="a" * 32)
    actual = invoke(root, monkeypatch, run_id="a" * 32)

    assert actual.exit_code == expected.exit_code == 1
    assert actual.stdout == expected.stdout
    assert actual.stderr == expected.stderr
    assert only_receipt(root) == only_receipt(old_root)
    assert only_receipt(root)["replay"] == {"state": "not_reached"}
    assert only_receipt(root)["error"] == "KeyboardInterrupt"
    assert (old_root / target.name).read_bytes() == target.read_bytes() == before
    assert pick_path.read_bytes() == b'{"result":"hit"}'


@pytest.mark.parametrize(("start", "replay_time", "day", "reason"), [
    ("2027-04-19T23:59", "2027-04-20T00:01", "2027-04-19", PENDING),
    ("2027-12-31T23:59", "2028-01-01T00:01", "2028-01-01",
     "today's result is applied to streak.json but not yet saved on the pick"),
])
def test_replay_names_the_blocker_from_under_the_lock(tmp_path, monkeypatch, start, replay_time, day, reason):
    from tests.test_wr_c3v_corpus import dump, pick_data, seed_streak

    root = tmp_path / "picks"
    now = datetime.fromisoformat(start).replace(tzinfo=ET)
    after = datetime.fromisoformat(replay_time).replace(tzinfo=ET)
    target = seed_streak(root, season=after.year, markers=(day,))
    before = target.read_bytes()
    dump(root / f"{day}.json", pick_data(day))
    if now.year != after.year:
        (root / f"{now.date().isoformat()}.json").write_text("{different season")
    real_lock = picks.scoring_lock

    @contextmanager
    def delayed_lock(directory):
        nonlocal now
        with real_lock(directory):
            now = after
            yield

    monkeypatch.setattr(picks, "scoring_lock", delayed_lock)
    result = invoke(root, monkeypatch, clock=lambda: now)
    assert result.exit_code == 0, result.output
    assert result.stderr == warning(day, reason)
    assert result.stdout.endswith("Streak retained: 21\n")
    assert only_receipt(root)["replay"]["blocked_date"] == day
    assert target.read_bytes() == before


def test_rerun_reports_pending_preview_until_operator_sets_it_aside(tmp_path, monkeypatch):
    from tests.test_wr_c3v_corpus import cases, dump, pick_data, seed_streak

    root = tmp_path / "picks"
    target = seed_streak(root)
    before = target.read_bytes()
    preview = next(case for case in cases() if case.name == "refused_preview")
    path = dump(root / f"{PAST}.json", preview.transform(PAST, root))
    next_day = "2027-04-17"
    next_path = dump(root / f"{next_day}.json", preview.transform(next_day, root))
    dump(root / f"{TODAY}.json", pick_data(TODAY, "hit"))
    for _ in range(2):
        result = invoke(root, monkeypatch)
        assert result.exit_code == 0, result.output
        assert result.stderr == warning(PAST, PENDING)
        assert target.read_bytes() == before
    aside = root / PAST / "set_aside_never_entered.json"
    aside.parent.mkdir()
    path.rename(aside)
    result = invoke(root, monkeypatch)
    assert result.exit_code == 0, result.output
    assert result.stderr == warning(next_day, PENDING)
    assert target.read_bytes() == before

    next_aside = root / next_day / aside.name
    next_aside.parent.mkdir()
    next_path.rename(next_aside)
    result = invoke(root, monkeypatch)
    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert result.stdout.endswith("No scoring changes detected. Streak: 1\n")
    assert json.loads(target.read_bytes())["streak"] == 1
