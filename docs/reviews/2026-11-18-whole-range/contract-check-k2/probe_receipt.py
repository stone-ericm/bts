import json
import socket
import tempfile
from pathlib import Path
from unittest.mock import patch

from bts.health import alert
from probe_scheduler import D, STATUS, fixture, game, offline, start


def main(root):
    sent = {"status": "sent", "sent_day": D, "sent_sources": ["WARN_ATTENTION:season_boundary:behind_schedule"]}
    out = {}
    for case in ("schema-less-sent", "alias", "interrupted-receipt", "count-changed"):
        box = root / case
        config = fixture(box)
        status = box / "health_state" / STATUS
        status.parent.mkdir()
        if case == "schema-less-sent":
            status.write_text(json.dumps(sent))
        elif case == "alias":
            general = status.with_name("health_dm_delivery_status.json")
            general.write_text(json.dumps({"status": "sent", "sent_day": D,
                                           "sent_sources": ["CRITICAL:contest_state"]}))
            status.symlink_to(general)
        elif case == "interrupted-receipt":
            real_write = alert._write_regular_file_text

            def interrupt_write(path, text):
                if path == status:
                    path.write_text("{")
                    raise KeyboardInterrupt
                return real_write(path, text)

        if case == "interrupted-receipt":
            with patch.object(alert, "_write_regular_file_text", interrupt_write):
                first = start(config, D, [game(D)])
        else:
            first = start(config, D, [game(D)])
        second = start(config, D, [game(D), game(D, 1)] if case == "count-changed" else [game(D)])
        out[case] = {"first": first, "second": second}
        if case == "alias":
            out[case]["general_status"] = json.loads(general.read_text())
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    with (tempfile.TemporaryDirectory() as tmp, patch.object(socket, "getaddrinfo", offline),
          patch.object(socket.socket, "connect", offline)):
        main(Path(tmp))
