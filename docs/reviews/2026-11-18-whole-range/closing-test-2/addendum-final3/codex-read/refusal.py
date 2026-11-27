import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import bts
from bts import contest_state, picks, scheduler
from bts.health import alert
from tests.test_wr_k1_refusal_dm import AT, BOX, DAY, cases, fixtures, offline, refusal


def main():
    print(f"bts.__file__={bts.__file__}")
    dated = "slate_date" in inspect.signature(scheduler._alert_contest_state_failure).parameters
    maximum = (0, "")
    outcomes = {}
    inventory = [(key, variant) for key, variant in cases if key != "synthetic"]
    inventory.extend([
        ("authority", {"schema_version": "🧢" * 1000}),
        ("authority", {"schema_version": "2027\nunsupported"}),
        ("auto_parse", {"active_streak": "🧢" * 1000}),
        ("dated", ("effective_date", "2027-99-99")),
        ("dated", ("expires_at", f"{DAY}T23:59:59.999999+00:00")),
        ("dated", ("recorded_at", "bad timestamp")),
        ("saved", "dated_to_automatic"),
        ("saved", "automatic_to_dated"),
        ("synthetic", "x" * 400),
    ])
    for index, (key, variant) in enumerate(inventory):
        with TemporaryDirectory() as name, offline(AT):
            root = Path(name) / "picks"
            account = root / "account_state"
            account.mkdir(parents=True)
            receipt = root.parent / "health_state" / "health_dm_delivery_status.json"
            receipt.parent.mkdir()
            if key == "synthetic":
                error = contest_state.ContestStateError(variant)
            else:
                fixtures(key, variant, root, account)
                if key == "saved" and variant.endswith(("_automatic", "_dated")):
                    d = contest_state.load_decision_streak_state(root, require_contest_state=True)
                    a = SimpleNamespace(observation_kind="automatic", dated_identity=None,
                                        streak=12, saver_available=False)
                    chosen, live = (d, a) if variant == "dated_to_automatic" else (a, d)
                    record = picks.chosen_on_record(chosen, required=True)
                    text = picks.saved_pick_refusal(SimpleNamespace(date=DAY, chosen_on=record), live)
                    assert text
                    error = contest_state.ContestStateError(text)
                else:
                    error = refusal(key, variant, root)
            print(f"{index:02d} source={key} diagnostic={str(error).splitlines()[0][:150]}")
            assert str(root) not in str(error)
            if f"{BOX}/" in str(error):
                assert "picks" in str(error)
            config = {"orchestrator": {"picks_dir": str(root)}, "bluesky": {"dm_recipient": "capture.invalid"}}
            for prior in (1, 999, 1000, 10**300):
                receipt.write_text(json.dumps({"status": "failed", "critical_count": prior}))
                attempts = []

                def transport(recipient, body):
                    assert recipient == "capture.invalid"
                    body.encode("utf-8")
                    attempts.append(body)
                    if len(body) > 1000:
                        raise ValueError("chat body exceeds 1,000 code points")
                    return "captured"

                with patch.object(alert, "send_dm", transport):
                    kwargs = {"slate_date": DAY} if dated else {}
                    scheduler._alert_contest_state_failure(config, error, **kwargs)
                assert len(attempts) == 1
                body, = attempts
                status = json.loads(receipt.read_text())["status"]
                outcomes[status] = outcomes.get(status, 0) + 1
                maximum = max(maximum, (len(body), f"{index}:{key}"))
                if dated:
                    assert status == "sent" and len(body) < 1000
                    assert "See the scheduler journal" in body
                    assert "bts season-contract status" in body and "B8a" in body
                    assert "999+ CRITICAL" in body if prior > 999 else f"({prior} CRITICAL)" in body
                    assert f"Reason: {str(error).splitlines()[0][:397]}" in body
    print(f"diagnostics={len(inventory)} attempts={sum(outcomes.values())} outcomes={outcomes} peak={maximum}")


if __name__ == "__main__":
    main()
