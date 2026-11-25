"""Verify `bts run` CLI at streak 58 and 400 (saves chosen_on and posts Streak: 58 / 400)."""
from __future__ import annotations

import datetime as dt
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

OUT_DIR = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")


def main() -> int:
    from click.testing import CliRunner
    from bts import cli, picks, posting
    from scripts.audit.season_walk import walk

    day = dt.date(2027, 7, 15)
    runner = CliRunner()
    problems = []
    frame = walk._predictions(day, 0.96).head(2).copy()
    frame["game_time"] = "2027-07-15T22:00:00Z"  # 18:00 ET (1h after 17:00 ET)
    statuses = {int(g): {"abstract": "P", "detailed": "Scheduled"} for g in frame["game_pk"]}
    at_17 = walk.REAL_DATETIME(2027, 7, 15, 17, 0, tzinfo=ET)

    with walk.offline(at_17), tempfile.TemporaryDirectory(prefix="h2c-bts-run-") as tmp:
        root = Path(tmp)
        for s_val in (58, 400):
            pdir = root / f"r_{s_val}" / "picks"
            pdir.mkdir(parents=True)
            assert walk.install_state(pdir, day, walk.State("dated", s_val, s_val, "used", 0.96)) == []
            posted = []
            with ExitStack() as st:
                st.enter_context(patch("bts.model.predict.run_pipeline", lambda *a, **kw: frame))
                st.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *a: statuses))
                st.enter_context(patch.object(posting, "_now_et", lambda: at_17))
                st.enter_context(patch.object(posting, "post_to_bluesky", side_effect=lambda t: (posted.append(t) or "at://r/1")))
                res = runner.invoke(cli.cli, ["run", "--date", day.isoformat(), "--picks-dir", str(pdir), "--models-dir", str(root / "m")])
            saved = picks.load_pick(day.isoformat(), pdir)
            if res.exit_code != 0 or len(posted) != 1 or f"Streak: {s_val}" not in posted[0] or saved.chosen_on["streak"] != s_val:
                problems.append(f"s={s_val} code={res.exit_code} posted={posted} out={res.output!r}")

    msg = f"bts run verified at 58 and 400; problems={len(problems)}\n" + "\n".join(problems)
    (OUT_DIR / "bts_run_verify.log").write_text(msg + "\n", encoding="utf-8")
    print(msg)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
