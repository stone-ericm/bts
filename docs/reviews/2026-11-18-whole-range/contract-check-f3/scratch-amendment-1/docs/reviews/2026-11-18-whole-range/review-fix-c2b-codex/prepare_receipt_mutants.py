"""Build independent source copies for the seven requested receipt mutations."""

from pathlib import Path
from shutil import copytree


REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[1]
ALERT = Path("src/bts/health/alert.py")


def change_once(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after)


def main():
    original = (ROOT / ALERT).read_text(encoding="utf-8")
    keys = '            and prior.keys() == {"schema_version", "status", "sent_day", "sent_sources", "message_id", "body_sha256"}\n'
    digest = 'body_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()'
    compare_hash = '            and prior.get("body_sha256") == body_sha256\n'
    start = original.index("        if (\n            prior is not None")
    end = original.index("            return False", start)
    cases = {
        "drop-keys": (keys, ""),
        "accept-superset": (keys, keys.replace(" == ", " >= ")),
        "allow-duplicates": (", object_pairs_hook=_unique_receipt_members", ""),
        "drop-hash-compare": (compare_hash, ""),
        "hash-constant": (digest, 'body_sha256 = hashlib.sha256(b"fixed review fixture").hexdigest()'),
        "hash-message": (digest, 'body_sha256 = hashlib.sha256(body.split("\\n", 1)[1].split("] ", 1)[1].encode("utf-8")).hexdigest()'),
        "compare-day-only": (original[start:end], '        if prior is not None and prior.get("sent_day") == day:\n'),
    }
    for name, (before, after) in cases.items():
        destination = REVIEW / "mutants" / name
        updated = change_once(original, before, after)
        compile(updated, str(destination / ALERT), "exec")
        copytree(ROOT / "src", destination / "src")
        (destination / ALERT).write_text(updated, encoding="utf-8")
        print(f"{name}: {destination / ALERT}")


if __name__ == "__main__":
    main()
