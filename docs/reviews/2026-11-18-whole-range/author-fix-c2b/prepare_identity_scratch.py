"""Prepare isolated C2b verification sources."""

from pathlib import Path
from shutil import copytree
import subprocess


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ALERT = Path("src/bts/health/alert.py")
BASE = "e897c766"
DOCS = Path("/Users/eric/projects/bts/docs/reviews/2026-11-18-whole-range")


def change(source, before, after):
    if source.count(before) != 1:
        raise ValueError(before)
    return source.replace(before, after)


def main():
    current = (ROOT / ALERT).read_text()
    keys = '            and prior.keys() == {"schema_version", "status", "sent_day", "sent_sources", "message_id", "body_sha256"}\n'
    hash_check = '            and prior.get("body_sha256") == body_sha256\n'
    start = current.index("        if (\n            prior is not None")
    finish = current.index("            return False", start)
    variants = {
        "base": subprocess.check_output(["git", "show", f"{BASE}:{ALERT}"], cwd=ROOT, text=True),
        "drop-keys": change(current, keys, ""),
        "allow-superset": change(current, keys, keys.replace(" == ", " >= ")),
        "allow-duplicates": change(current, ", object_pairs_hook=_unique_receipt_members", ""),
        "drop-hash-check": change(current, hash_check, ""),
        "hash-constant": change(current, 'body_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()', 'body_sha256 = hashlib.sha256(b"fixture").hexdigest()'),
        "hash-alert-message": change(
            current,
            'body_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()',
            'body_sha256 = hashlib.sha256(body.split("\\n", 1)[-1].partition("] ")[-1].encode("utf-8")).hexdigest()',
        ),
        "compare-day-only": change(current, current[start:finish], '        if prior and prior.get("sent_day") == day:\n'),
    }
    for name, source in variants.items():
        target = HERE / "scratch" / name
        copytree(ROOT / "src", target / "src", dirs_exist_ok=True)
        (target / ALERT).write_text(source)

    adopted = DOCS / "review-fix-c2-carbon-r2/adopt/test_wr_c2_r2_carbon.py"
    (HERE / "scratch/base/carbon_original.py").write_bytes(adopted.read_bytes())


if __name__ == "__main__":
    main()
