"""Item 3 probe: cooldown stamp matrix across base, r1, and head on failure & success lanes."""
import datetime as dt
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import httpx
from click.testing import CliRunner

REVIEW = Path(__file__).resolve().parent
REV = sys.argv[1]
assert REV in ("base", "r1", "head", "compare"), REV

if REV == "compare":
    b = json.loads((REVIEW / "cooldown-base.json").read_text())
    r1 = json.loads((REVIEW / "cooldown-r1.json").read_text())
    h = json.loads((REVIEW / "cooldown-head.json").read_text())
    diffs = []
    gt6h_only_head = []
    for key in h:
        hb, hr, hh = b[key], r1[key], h[key]
        if hb != hh or hr != hh:
            diffs.append({
                "case": key,
                "base_fail": (hb["failure"]["exit"], hb["failure"]["exc"], hb["failure"]["dm_sent"], hb["failure"]["status_after"]),
                "r1_fail": (hr["failure"]["exit"], hr["failure"]["exc"], hr["failure"]["dm_sent"], hr["failure"]["status_after"]),
                "head_fail": (hh["failure"]["exit"], hh["failure"]["exc"], hh["failure"]["dm_sent"], hh["failure"]["status_after"]),
                "base_succ": (hb["success"]["exit"], hb["success"]["status_after"]),
                "r1_succ": (hr["success"]["exit"], hr["success"]["status_after"]),
                "head_succ": (hh["success"]["exit"], hh["success"]["status_after"]),
            })
        if (hh["failure"]["dm_sent"] == 0 and hh["after_8h_failure"]["dm_sent"] == 0
                and hb["failure"]["dm_sent"] != 0):
            gt6h_only_head.append(key)
    summary = {"total_cases": len(h), "diff_count": len(diffs), "gt6h_suppressed_only_on_head": gt6h_only_head, "diffs": diffs}
    (REVIEW / "cooldown-compare.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Compared {len(h)} cases; {len(diffs)} cases differ across base/r1/head")
    print("Suppressed >6h on HEAD where base did NOT suppress:", gt6h_only_head)
    for d in diffs:
        print(f"  {d['case']:34} | base_fail={d['base_fail'][:3]} | r1_fail={d['r1_fail'][:3]} | head_fail={d['head_fail'][:3]}")
    raise SystemExit(0)

ROOT = REVIEW.parents[1] if REV == "head" else REVIEW / REV
sys.path.insert(0, str(ROOT / "src"))

import pandas  # noqa: F401
from bts import cli, contest_fetch as cf, contest_state as cs, saver_state as saver, season_contract as sc
from bts.leaderboard import auth

expected_src = (ROOT / "src").resolve()
for mod in (cli, cf, cs, saver, sc):
    assert Path(mod.__file__).resolve().is_relative_to(expected_src), (REV, mod.__file__)

REAL_DT = dt.datetime
ET = ZoneInfo("America/New_York")
NOW = REAL_DT(2027, 7, 15, 12, 0, 0, tzinfo=ET)  # 2027-07-15T16:00:00Z
ID = "a" * 32

STAMPS = [
    ("aware-inside-6h", "2027-07-15T15:00:00+00:00"),
    ("aware-outside-6h", "2027-07-15T09:00:00+00:00"),
    ("aware-exact-6h", "2027-07-15T10:00:00+00:00"),
    ("extreme-9999", "9999-12-31T23:59:59-01:00"),
    ("extreme-0001", "0001-01-01T00:00:00+01:00"),
    ("naive", "2027-07-15T15:00:00"),
    ("z-suffix-inside", "2027-07-15T15:00:00Z"),
    ("z-suffix-outside", "2027-07-15T09:00:00Z"),
    ("no-colon-inside", "2027-07-15T15:00:00+0000"),
    ("no-colon-outside", "2027-07-15T09:00:00+0000"),
    ("date-only", "2027-07-15"),
    ("date-with-Z", "2027-07-15Z"),
    ("garbage", "not-a-timestamp"),
    ("number-nonzero", 42),
    ("number-zero", 0),
    ("z-sep-inside", "2027-07-15Z15:00:00Z"),
    ("z-sep-future-gt6h", "2027-07-15Z23:00:00Z"),
    ("z-sep-far-future", "9999-12-31Z23:59:59-01:00"),
    ("space-sep-future", "2027-07-15 23:00:00+00:00"),
]


@contextmanager
def env_at(now, fail=False):
    class PinnedDT(REAL_DT):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    send_mock = Mock(return_value="dm-ack")
    login_fn = (Mock(side_effect=auth.AuthError("expired"))
                if fail else lambda **_: SimpleNamespace(user_id=1, username="offline", xsid="x"))
    with (
        patch.object(dt, "datetime", PinnedDT),
        patch.object(cs, "datetime", PinnedDT),
        patch.object(saver, "datetime", PinnedDT),
        patch.object(sc, "datetime", PinnedDT),
        patch.object(httpx, "get", Mock(side_effect=AssertionError("network forbidden"))),
        patch.object(auth, "load_session_cookies", lambda: {}),
        patch.object(auth, "extract_uid", lambda _: "u"),
        patch.object(auth, "fetch_login_session", login_fn),
        patch.object(cf, "fetch_profile", lambda *_: {
            "activeStreak": 7, "seasonBestStreak": 9, "predictions": [{"roundId": 1, "result": "hit"}],
        }),
        patch.object(cli, "_fetch_rounds", lambda: {1: dt.date(2027, 7, 14)}),
        patch("bts.dm.send_dm", send_mock),
    ):
        yield send_mock


def clean_status(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return {k: d[k] for k in ("last_alert_at", "last_alert_at_by_category") if k in d}


out = {}
with TemporaryDirectory(dir=REVIEW, prefix=f"cooldown-{REV}-") as tmp:
    root = Path(tmp)
    for layout in ("legacy", "category"):
        for label, stamp in STAMPS:
            key = f"{layout}:{label}"
            prior = ({"last_alert_at": stamp} if layout == "legacy"
                     else {"last_alert_at_by_category": {"actionable": stamp}})

            # Lane A: failure directly, then another failure at NOW + 8h
            p_fail = root / key / "fail" / "picks"
            (p_fail / "account_state").mkdir(parents=True)
            (p_fail / "account_state/season_contract.json").write_text(json.dumps({
                "schema_version": "bts_season_contract_v1", "season": 2027, "status": "automatic",
                "activation_id": ID, "switched_at": "2027-07-15T16:00:00Z", "revoked_at": None, "note": None,
            }))
            st_fail = p_fail.parent / "health_state/contest_streak_fetch_status.json"
            st_fail.parent.mkdir(parents=True)
            st_fail.write_text(json.dumps(prior))
            with env_at(NOW, fail=True) as send1:
                r_fail = CliRunner().invoke(cli.cli, [
                    "fetch-contest-streak", "--picks-dir", str(p_fail), "--dm-recipient", "offline.test",
                ])
            after_fail = clean_status(st_fail)
            with env_at(NOW + dt.timedelta(hours=8), fail=True) as send_8h:
                r_8h = CliRunner().invoke(cli.cli, [
                    "fetch-contest-streak", "--picks-dir", str(p_fail), "--dm-recipient", "offline.test",
                ])

            # Lane B: success first, then failure at NOW
            p_succ = root / key / "succ" / "picks"
            (p_succ / "account_state").mkdir(parents=True)
            (p_succ / "account_state/season_contract.json").write_text(json.dumps({
                "schema_version": "bts_season_contract_v1", "season": 2027, "status": "automatic",
                "activation_id": ID, "switched_at": "2027-07-15T16:00:00Z", "revoked_at": None, "note": None,
            }))
            st_succ = p_succ.parent / "health_state/contest_streak_fetch_status.json"
            st_succ.parent.mkdir(parents=True)
            st_succ.write_text(json.dumps(prior))
            with env_at(NOW, fail=False) as send_s:
                r_succ = CliRunner().invoke(cli.cli, [
                    "fetch-contest-streak", "--picks-dir", str(p_succ), "--dm-recipient", "offline.test",
                ])
            after_succ = clean_status(st_succ)
            with env_at(NOW, fail=True) as send_after_s:
                r_after_s = CliRunner().invoke(cli.cli, [
                    "fetch-contest-streak", "--picks-dir", str(p_succ), "--dm-recipient", "offline.test",
                ])

            out[key] = {
                "stamp": stamp,
                "failure": {
                    "exit": r_fail.exit_code,
                    "exc": type(r_fail.exception).__name__ if (r_fail.exception and not isinstance(r_fail.exception, SystemExit)) else None,
                    "dm_sent": send1.call_count,
                    "status_after": after_fail,
                },
                "after_8h_failure": {
                    "exit": r_8h.exit_code,
                    "exc": type(r_8h.exception).__name__ if (r_8h.exception and not isinstance(r_8h.exception, SystemExit)) else None,
                    "dm_sent": send_8h.call_count,
                },
                "success": {
                    "exit": r_succ.exit_code,
                    "dm_sent": send_s.call_count,
                    "status_after": after_succ,
                },
                "failure_after_success": {
                    "exit": r_after_s.exit_code,
                    "exc": type(r_after_s.exception).__name__ if (r_after_s.exception and not isinstance(r_after_s.exception, SystemExit)) else None,
                    "dm_sent": send_after_s.call_count,
                },
            }

(REVIEW / f"cooldown-{REV}.json").write_text(json.dumps(out, indent=2) + "\n")
print(f"{REV}: ran {len(out)} cases")
