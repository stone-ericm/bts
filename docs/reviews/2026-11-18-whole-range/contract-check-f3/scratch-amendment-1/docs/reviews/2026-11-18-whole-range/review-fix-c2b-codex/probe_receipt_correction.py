"""Check that a scratch correction closes the reproduced receipt holes."""

from datetime import date
import hashlib
import json
from unittest.mock import Mock

import probe_receipt_identity as identity
from probe_receipt_identity import DAY, KEY, alert, deadline, fixtures, observe, status_body


def checksum(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def foreign_receipt(root, patch):
    path = root / "foreign-ack.json"
    with patch.context() as scoped:
        sender = Mock(return_value="foreign-id")
        scoped.setattr(alert, "send_dm", sender)
        assert alert.dispatch_dm_for_health_alerts(
            [alert.Alert("CRITICAL", "late_delivery", "LATE DELIVERY REFUSED: fixture")],
            "offline.test", status_path=path, now_et_date=date.fromisoformat(DAY),
            require_message_id=True,
        ) is True
        _, foreign_body = sender.call_args.args
    raw = path.read_text()
    parsed = json.loads(raw)
    assert parsed == identity.SHAPE | {"message_id": "foreign-id", "body_sha256": checksum(foreign_body)}
    assert parsed["sent_sources"] == [KEY]
    assert parsed["body_sha256"] != identity.SHAPE["body_sha256"]
    return raw


def main():
    old_shape = identity.SHAPE.copy()
    identity.SHAPE["body_sha256"] = checksum(status_body(DAY))
    valid = json.dumps(identity.SHAPE)
    extra = json.dumps(identity.SHAPE | {"error": "send_dm returned no usable message id"})
    duplicate = '{"status":"failed",' + valid[1:]
    with deadline(45), fixtures("receipt-correction") as (root, patch, opening):
        cases = {
            "extra field despite correct fingerprint": extra,
            "duplicate member despite correct fingerprint": duplicate,
            "same key and slate but a different acknowledged body": foreign_receipt(root, patch),
            "old five-member receipt": json.dumps(old_shape),
            "real prior C2 acknowledgement": None,
        }
        for route in ("direct", "classification"):
            for index, (label, document) in enumerate(cases.items()):
                got = observe(root / f"{route}-{index}", patch, opening, route, label, document)
                assert got == int(document is not None), (route, label, got)
                print(f"{route}: {label}: {got} C2 DMs across two encounters")


if __name__ == "__main__":
    main()
