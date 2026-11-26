import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from bts.health import alert, attention, contest_state

from probe_support import DAY, HERE, flag, manual


def transport(_, body):
    print("DM length", len(body))
    if len(body) > 1000:
        raise ValueError("simulated chat length rejection")
    return "sent"


with tempfile.TemporaryDirectory(dir=HERE, prefix="overflow-") as directory:
    root = Path(directory)
    picks = root / "picks"
    flag(picks, "active")
    manual(picks)
    mismatch, = contest_state.check(picks, expected=True, now=DAY)
    existing = alert.Alert("WARN", "pick_entry", "Existing operator alert. " * 25)
    captured = Mock(side_effect=transport)
    with patch.object(alert, "send_dm", captured):
        for label, alerts in (("base", [existing]), ("candidate", [existing, mismatch])):
            _, warns = attention.apply_warn_attention_policy(
                alerts, state_path=root / "attention.json", today=DAY.date(),
            )
            status = root / f"{label}.json"
            assert alert.dispatch_dm_for_health_alerts(alerts, "offline.test", warn_attention=warns,
                                                       status_path=status, now_et_date=DAY.date())
            disposition = alert._read_health_dm_delivery_status(status)["status"]
            print(label, disposition)
            assert disposition == ("sent" if label == "base" else "failed")
