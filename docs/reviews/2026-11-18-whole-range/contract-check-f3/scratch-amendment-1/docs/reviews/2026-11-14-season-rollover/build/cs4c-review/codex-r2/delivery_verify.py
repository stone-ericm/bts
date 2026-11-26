"""Compare recorded deliveries after accounting for their scratch directory names."""

import hashlib
import json
import re

from delivery_probe import REVIEW, compare


SCRATCH_PATH = re.compile(re.escape(str(REVIEW)) + r"/delivery-data-[a-z0-9_]+/\d+(?=/)")
HEALTH_STATUS = "health_state/health_dm_delivery_status.json"


def hash_body(body):
    return hashlib.sha256(body.encode()).hexdigest()


def read(revision):
    data = json.loads((REVIEW / f"delivery-{revision}.json").read_text())
    normalized = 0
    for case in data.values():
        existing = case["existing"]
        for call in existing["alerts"]:
            body = call[1]
            canonical = SCRATCH_PATH.sub("{case}", body)
            if body == canonical:
                continue
            raw = bytes.fromhex(existing["files"][HEALTH_STATUS]).decode()
            actual = hash_body(body)
            assert json.loads(raw)["body_sha256"] == actual
            existing["files"][HEALTH_STATUS] = raw.replace(actual, hash_body(canonical), 1).encode().hex()
            call[1] = canonical
            normalized += 1
    print(f"{revision}: verified and normalized {normalized} path-dependent health DM bodies and hashes")
    return data


if __name__ == "__main__":
    compare(read("base"), read("branch"))
