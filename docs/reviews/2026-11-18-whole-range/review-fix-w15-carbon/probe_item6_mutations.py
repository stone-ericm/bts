"""Probe family 6: mutation testing on an isolated scratch copy of src/bts against the W15 test suite (never touching tracked files)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REVIEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = REVIEW_DIR.parents[1]
HEAD_SRC = REPO_ROOT / "src"

TEST_TARGETS = [
    "tests/test_wr_w15_corpus.py",
    "tests/test_wr_w15_incidents.py",
    "tests/test_wr_w15_health.py",
    "tests/test_probable_cache_review.py",
    "tests/test_leads_small_fixes.py",
]


def replace_exact(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text, f"Target snippet not found in {path}:\n{old}"
    assert text.count(old) == 1, f"Target snippet matched multiple times in {path}"
    path.write_text(text.replace(old, new, 1))


MUTATIONS: list[tuple[str, callable]] = [
    (
        "1_accept_bool_as_int_producer",
        lambda root: replace_exact(
            root / "bts/features/probable_pitcher_cache.py",
            "and all(value is None or type(value) is int for value in entry.values()))",
            "and all(value is None or isinstance(value, int) for value in entry.values()))",
        ),
    ),
    (
        "1b_accept_bool_as_int_status_reader",
        lambda root: replace_exact(
            root / "bts/probable_pitcher_status.py",
            "elif (any(type(value) is not int or value < 0 for value in (dropped, aliases, unusable, missing))",
            "elif (any(not isinstance(value, int) or value < 0 for value in (dropped, aliases, unusable, missing))",
        ),
    ),
    (
        "2_skip_scanner_candidate_validation",
        lambda root: replace_exact(
            root / "bts/features/probable_pitcher_cache.py",
            "            if valid(entry):\n                scanned[game] = entry",
            "            if True:\n                scanned[game] = entry",
        ),
    ),
    (
        "3_rewrite_before_preserving",
        lambda root: replace_exact(
            root / "bts/features/probable_pitcher_cache.py",
            "def build(cache, raw):\n"
            "    members, original, unreadable, special = read_cache(cache)\n"
            "    lookup, invalid = load_entries(members)\n"
            "    scanned, rejected, incomplete = scan(Path(raw), lookup)\n"
            "    can_rewrite = True\n"
            "    if unreadable:\n"
            "        outcomes = [[game, entry] for game, entry in sorted(scanned.items())]\n"
            "        counts = [None, None, len(scanned), None, None]\n"
            '        can_rewrite = report(cache, original, special, "unreadable_cache", scanned, outcomes, counts)\n'
            "    elif invalid:\n"
            "        outcomes, counts = drop_outcomes(lookup, invalid, scanned, rejected, incomplete)\n"
            "        dropped, aliases, rescanned, unusable, missing = counts\n"
            '        noun = "entry" if dropped == 1 else "entries"\n'
            '        sample = ", ".join(str(outcome[0]) for outcome in outcomes[:10])\n'
            "        warn(\n"
            '            f"probable-pitcher cache {cache}: dropped {dropped} invalid {noun} "\n'
            '            f"({aliases} recovered from valid cache entries, {rescanned} rescanned from raw feeds, "\n'
            '            f"{unusable} with unusable feeds, {missing} without feeds); first game ids: {sample}"\n'
            "        )\n"
            '        can_rewrite = report(cache, original, special, "invalid_entries", scanned, outcomes, counts)\n\n'
            "    lookup.update(scanned)\n"
            "    if scanned and can_rewrite and not special:\n"
            "        cache.parent.mkdir(parents=True, exist_ok=True)\n"
            '        temp = cache.with_name(f".{cache.name}.{os.getpid()}.tmp")\n'
            "        try:\n"
            "            temp.write_text(json.dumps({str(game): entry for game, entry in lookup.items()}))\n"
            "            os.replace(temp, cache)\n"
            "        finally:\n"
            "            temp.unlink(missing_ok=True)\n"
            "    return lookup",
            "def build(cache, raw):\n"
            "    members, original, unreadable, special = read_cache(cache)\n"
            "    lookup, invalid = load_entries(members)\n"
            "    scanned, rejected, incomplete = scan(Path(raw), lookup)\n"
            "    res = dict(lookup)\n"
            "    res.update(scanned)\n"
            "    if scanned and not special:\n"
            "        cache.parent.mkdir(parents=True, exist_ok=True)\n"
            '        temp = cache.with_name(f".{cache.name}.{os.getpid()}.tmp")\n'
            "        try:\n"
            "            temp.write_text(json.dumps({str(game): entry for game, entry in res.items()}))\n"
            "            os.replace(temp, cache)\n"
            "        finally:\n"
            "            temp.unlink(missing_ok=True)\n"
            "    if unreadable:\n"
            "        outcomes = [[game, entry] for game, entry in sorted(scanned.items())]\n"
            "        counts = [None, None, len(scanned), None, None]\n"
            '        report(cache, original, special, "unreadable_cache", scanned, outcomes, counts)\n'
            "    elif invalid:\n"
            "        outcomes, counts = drop_outcomes(lookup, invalid, scanned, rejected, incomplete)\n"
            "        dropped, aliases, rescanned, unusable, missing = counts\n"
            '        noun = "entry" if dropped == 1 else "entries"\n'
            '        sample = ", ".join(str(outcome[0]) for outcome in outcomes[:10])\n'
            "        warn(\n"
            '            f"probable-pitcher cache {cache}: dropped {dropped} invalid {noun} "\n'
            '            f"({aliases} recovered from valid cache entries, {rescanned} rescanned from raw feeds, "\n'
            '            f"{unusable} with unusable feeds, {missing} without feeds); first game ids: {sample}"\n'
            "        )\n"
            '        report(cache, original, special, "invalid_entries", scanned, outcomes, counts)\n'
            "    return res",
        ),
    ),
    (
        "4_publish_after_cache_replace",
        lambda root: replace_exact(
            root / "bts/features/probable_pitcher_cache.py",
            '    target = cache.parent.parent / "health_state" / SOURCE / f"{day}.{reason}.{identity[:12]}.json"\n'
            "    try:\n"
            "        publish(target, row)\n"
            "    except Exception as error:\n"
            '        warn(f"could not write probable-pitcher cache status at {target} ({type(error).__name__})")\n'
            "        return False\n"
            "    return preserved_ok\n\n\n"
            "def build(cache, raw):\n"
            "    members, original, unreadable, special = read_cache(cache)\n"
            "    lookup, invalid = load_entries(members)\n"
            "    scanned, rejected, incomplete = scan(Path(raw), lookup)\n"
            "    can_rewrite = True\n"
            "    if unreadable:\n"
            "        outcomes = [[game, entry] for game, entry in sorted(scanned.items())]\n"
            "        counts = [None, None, len(scanned), None, None]\n"
            '        can_rewrite = report(cache, original, special, "unreadable_cache", scanned, outcomes, counts)\n'
            "    elif invalid:\n"
            "        outcomes, counts = drop_outcomes(lookup, invalid, scanned, rejected, incomplete)\n"
            "        dropped, aliases, rescanned, unusable, missing = counts\n"
            '        noun = "entry" if dropped == 1 else "entries"\n'
            '        sample = ", ".join(str(outcome[0]) for outcome in outcomes[:10])\n'
            "        warn(\n"
            '            f"probable-pitcher cache {cache}: dropped {dropped} invalid {noun} "\n'
            '            f"({aliases} recovered from valid cache entries, {rescanned} rescanned from raw feeds, "\n'
            '            f"{unusable} with unusable feeds, {missing} without feeds); first game ids: {sample}"\n'
            "        )\n"
            '        can_rewrite = report(cache, original, special, "invalid_entries", scanned, outcomes, counts)\n\n'
            "    lookup.update(scanned)\n"
            "    if scanned and can_rewrite and not special:\n"
            "        cache.parent.mkdir(parents=True, exist_ok=True)\n"
            '        temp = cache.with_name(f".{cache.name}.{os.getpid()}.tmp")\n'
            "        try:\n"
            "            temp.write_text(json.dumps({str(game): entry for game, entry in lookup.items()}))\n"
            "            os.replace(temp, cache)\n"
            "        finally:\n"
            "            temp.unlink(missing_ok=True)\n"
            "    return lookup",
            '    target = cache.parent.parent / "health_state" / SOURCE / f"{day}.{reason}.{identity[:12]}.json"\n'
            "    return preserved_ok, (target, row)\n\n\n"
            "def build(cache, raw):\n"
            "    members, original, unreadable, special = read_cache(cache)\n"
            "    lookup, invalid = load_entries(members)\n"
            "    scanned, rejected, incomplete = scan(Path(raw), lookup)\n"
            "    can_rewrite = True\n"
            "    pending = None\n"
            "    if unreadable:\n"
            "        outcomes = [[game, entry] for game, entry in sorted(scanned.items())]\n"
            "        counts = [None, None, len(scanned), None, None]\n"
            '        can_rewrite, pending = report(cache, original, special, "unreadable_cache", scanned, outcomes, counts)\n'
            "    elif invalid:\n"
            "        outcomes, counts = drop_outcomes(lookup, invalid, scanned, rejected, incomplete)\n"
            "        dropped, aliases, rescanned, unusable, missing = counts\n"
            '        noun = "entry" if dropped == 1 else "entries"\n'
            '        sample = ", ".join(str(outcome[0]) for outcome in outcomes[:10])\n'
            "        warn(\n"
            '            f"probable-pitcher cache {cache}: dropped {dropped} invalid {noun} "\n'
            '            f"({aliases} recovered from valid cache entries, {rescanned} rescanned from raw feeds, "\n'
            '            f"{unusable} with unusable feeds, {missing} without feeds); first game ids: {sample}"\n'
            "        )\n"
            '        can_rewrite, pending = report(cache, original, special, "invalid_entries", scanned, outcomes, counts)\n\n'
            "    lookup.update(scanned)\n"
            "    if scanned and can_rewrite and not special:\n"
            "        cache.parent.mkdir(parents=True, exist_ok=True)\n"
            '        temp = cache.with_name(f".{cache.name}.{os.getpid()}.tmp")\n'
            "        try:\n"
            "            temp.write_text(json.dumps({str(game): entry for game, entry in lookup.items()}))\n"
            "            os.replace(temp, cache)\n"
            "        finally:\n"
            "            temp.unlink(missing_ok=True)\n"
            "    if pending is not None:\n"
            "        try:\n"
            "            publish(*pending)\n"
            "        except Exception as error:\n"
            '            warn(f"could not write probable-pitcher cache status at {pending[0]} ({type(error).__name__})")\n'
            "    return lookup",
        ),
    ),
    (
        "5_reuse_evidence_by_name_only",
        lambda root: replace_exact(
            root / "bts/features/probable_pitcher_cache.py",
            "    for candidate in sorted(folder.iterdir()):\n"
            "        if not EVIDENCE.fullmatch(candidate.name):\n"
            "            continue\n"
            "        try:\n"
            "            with open_regular(candidate) as stream:\n"
            '                candidate_sha = hashlib.file_digest(stream, "sha256").hexdigest()\n'
            "        except OSError:\n"
            "            continue\n"
            "        if candidate_sha == original_sha:\n"
            "            return candidate.name",
            "    for candidate in sorted(folder.iterdir()):\n"
            "        if EVIDENCE.fullmatch(candidate.name) and candidate.name.endswith(f'.{original_sha[:12]}.json'):\n"
            "            return candidate.name",
        ),
    ),
    (
        "6_leave_date_out_of_record_name",
        lambda root: (
            replace_exact(
                root / "bts/features/probable_pitcher_cache.py",
                'target = cache.parent.parent / "health_state" / SOURCE / f"{day}.{reason}.{identity[:12]}.json"',
                'target = cache.parent.parent / "health_state" / SOURCE / f"{reason}.{identity[:12]}.json"',
            ),
            replace_exact(
                root / "bts/probable_pitcher_status.py",
                'or path.name != f"{day}.{reason}.{identity[:12]}.json"):',
                'or path.name != f"{reason}.{identity[:12]}.json"):',
            ),
            replace_exact(
                root / "bts/health/probable_pitcher_cache.py",
                'patterns = (f"{today}.*.json", f"{today - timedelta(days=1)}.*.json")',
                'patterns = ("invalid_entries.*.json", "unreadable_cache.*.json")',
            ),
        ),
    ),
    (
        "7_hash_ten_id_sample_instead_of_full_outcomes",
        lambda root: replace_exact(
            root / "bts/features/probable_pitcher_cache.py",
            "identity = digest(canonical(dict(observation, outcomes=outcomes)))",
            'identity = digest(canonical(dict(observation, outcomes=[o[0] for o in outcomes[:10]] if reason == "invalid_entries" else [])))',
        ),
    ),
    (
        "8_drop_today_minus_1_day",
        lambda root: replace_exact(
            root / "bts/health/probable_pitcher_cache.py",
            'patterns = (f"{today}.*.json", f"{today - timedelta(days=1)}.*.json")',
            'patterns = (f"{today}.*.json",)',
        ),
    ),
    (
        "9_share_one_incident_key_per_reason",
        lambda root: replace_exact(
            root / "bts/health/probable_pitcher_cache.py",
            'incident_key=f"{SOURCE}:{row[\'recorded_date_et\']}:{row[\'identity\']}"',
            'incident_key=f"{SOURCE}:{row[\'reason\']}"',
        ),
    ),
    (
        "10_follow_symlink_when_reading_record",
        lambda root: replace_exact(
            root / "bts/probable_pitcher_status.py",
            "def read_record(path):\n    with open_regular(path) as stream:",
            "def read_record(path):\n    with open_regular(path, follow_symlinks=True) as stream:",
        ),
    ),
    # Extra adversarial mutations beyond the required 10!
    (
        "11_extra_skip_existing_record_value_comparison_in_publish",
        lambda root: replace_exact(
            root / "bts/features/probable_pitcher_cache.py",
            '    existing = read_record(path)\n    if any(existing[key] != value for key, value in row.items() if key != "recorded_at"):\n        raise FileExistsError(path)',
            "    read_record(path)",
        ),
    ),
    (
        "12_extra_inaccessible_raw_root_treated_as_missing_instead_of_unusable",
        lambda root: replace_exact(
            root / "bts/features/probable_pitcher_cache.py",
            "    except OSError:\n        return scanned, rejected, True",
            "    except OSError:\n        return scanned, rejected, False",
        ),
    ),
    (
        "13_extra_skip_producer_64kib_limit_in_publish",
        lambda root: replace_exact(
            root / "bts/features/probable_pitcher_cache.py",
            '    if len(payload) > MAX_BYTES:\n        raise ValueError("status exceeds size limit")',
            "    if False:\n        pass",
        ),
    ),
    (
        "14_extra_use_12hex_prefix_in_incident_key",
        lambda root: replace_exact(
            root / "bts/health/probable_pitcher_cache.py",
            'incident_key=f"{SOURCE}:{row[\'recorded_date_et\']}:{row[\'identity\']}"',
            'incident_key=f"{SOURCE}:{row[\'recorded_date_et\']}:{row[\'identity\'][:12]}"',
        ),
    ),
]


def main() -> None:
    survivors = []
    killed = []
    with tempfile.TemporaryDirectory(prefix="w15_mut_") as tmp:
        base_tmp = Path(tmp)
        for name, apply_fn in MUTATIONS:
            mut_root = base_tmp / name
            shutil.copytree(HEAD_SRC / "bts", mut_root / "bts")
            apply_fn(mut_root)
            # Notice test_wr_w15_incidents.py child_env() prepends REPO_ROOT (".../wr-review-w15-carbon"),
            # which does NOT contain `bts` directly (`bts` is under `src/bts`), EXCEPT editable install `.pth`
            # in site-packages comes AFTER `PYTHONPATH`! Putting `mut_root` FIRST on `PYTHONPATH` ensures
            # both parent pytest and `child_env()` import `mut_root / "bts"`!
            # Let's verify bts.__file__ inside the mutated environment first!
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            env["PYTHONPATH"] = str(mut_root)
            chk = subprocess.run(
                [sys.executable, "-c", "import bts; print(bts.__file__)"],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=10,
            )
            assert Path(chk.stdout.strip()).resolve() == (mut_root / "bts/__init__.py").resolve(), chk.stdout

            res = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-x", "--no-header", "-p", "no:cacheprovider", *TEST_TARGETS],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60,
            )
            if res.returncode == 0:
                survivors.append(name)
                print(f"SURVIVED: {name}")
            else:
                first_fail = next(
                    (ln for ln in res.stdout.splitlines() if ln.startswith("FAILED ") or "AssertionError" in ln),
                    res.stdout.splitlines()[-1] if res.stdout.splitlines() else f"rc={res.returncode}",
                )
                killed.append((name, first_fail))
                print(f"KILLED:   {name} -> {first_fail}")

    print(f"\nSUMMARY: {len(killed)} killed, {len(survivors)} survived: {survivors}")


if __name__ == "__main__":
    main()
