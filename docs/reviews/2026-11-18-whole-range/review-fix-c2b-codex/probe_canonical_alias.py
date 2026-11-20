"""Place an actual acknowledged receipt behind a presentation-path alias."""

from contextlib import redirect_stderr
from io import StringIO
import os
from unittest.mock import Mock, call

from probe_receipt_identity import DAY, NOW, encounter, stage_pick
from probe_receipt_matrix import acknowledge, assert_locked
from probe_support import FAILURE_PREFIX, deadline, fixtures, prepare_schedules, receipt, status_body, world_at

from bts.health import alert


def check(root, patch, opening, route, kind):
    with world_at(root, patch, opening, NOW) as (config, directory, _, _, scoped):
        sender = Mock(return_value="genuine-id")
        scoped.setattr(alert, "send_dm", sender)
        source_store = root / "source" / "picks"
        source_config = config | {"orchestrator": config["orchestrator"] | {"picks_dir": str(source_store)}}
        with redirect_stderr(StringIO()):
            genuine = acknowledge(DAY, route, source_config, source_store)
        sender.assert_called_once_with("offline.test", status_body(DAY))

        target = receipt(directory, DAY)
        source = receipt(source_store, DAY)
        target.parent.parent.mkdir(parents=True)
        if kind == "dedicated directory symlink":
            target.parent.symlink_to(source.parent, target_is_directory=True)
        else:
            target.parent.mkdir()
            if kind == "receipt symlink":
                target.symlink_to(source)
            else:
                os.link(source, target)
        assert target.read_bytes() == genuine

        sender.reset_mock()
        daily = stage_pick(DAY, directory)
        prepare_schedules(daily)
        with redirect_stderr(StringIO()) as stderr:
            encounter(route, DAY, config, directory, daily)
            encounter(route, DAY, config, directory, daily)
        assert_locked(DAY, directory)
        assert sender.call_args_list == [call("offline.test", status_body(DAY))] * 2
        assert sum(line.startswith(FAILURE_PREFIX) for line in stderr.getvalue().splitlines()) == 2
        assert source.read_bytes() == target.read_bytes() == genuine
        if kind == "dedicated directory symlink":
            assert target.parent.is_symlink()
        elif kind == "receipt symlink":
            assert target.is_symlink()
        else:
            assert target.stat().st_nlink == 2
        print(f"{route}: {kind} with genuine same-slate acknowledgement: both encounters paged and preserved the alias")


if __name__ == "__main__":
    with deadline(35), fixtures("canonical-alias") as (root, patch, opening):
        for route in ("direct", "classification"):
            for index, kind in enumerate(("dedicated directory symlink", "receipt symlink", "receipt hardlink")):
                check(root / f"{route}-{index}", patch, opening, route, kind)
