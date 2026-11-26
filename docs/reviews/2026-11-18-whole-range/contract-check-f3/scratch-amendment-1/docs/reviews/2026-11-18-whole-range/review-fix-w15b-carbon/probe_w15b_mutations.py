"""Item 5 Mutation Harness: W15b mutations + Round-1 & Round-2 mutation sets on scratch copies of src/bts."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
REPO_SRC = REPO_ROOT / "src"

TEST_TARGETS = [
    "tests/test_wr_w15_corpus.py",
    "tests/test_wr_w15_incidents.py",
    "tests/test_wr_w15_health.py",
    "tests/test_wr_w15b.py",
    "tests/test_probable_cache_review.py",
    "tests/test_leads_small_fixes.py",
]

MUTANTS: list[tuple[str, str, str, str]] = [
    # --- W15b required mutations ---
    (
        "w15b_01_summary_embeds_status_path",
        "bts/health/probable_pitcher_cache.py",
        'def summarize(today, identities, kinds):\n'
        '    # Fixed words, a count and fixed labels only: no path or other text read from disk, so the\n'
        '    # summary can never be OOM evidence for the attention policy (W15b) and stays under 600 characters.\n'
        '    count = len(identities)\n'
        '    noun = "observation" if count == 1 else "observations"\n'
        '    labels = ", ".join(sorted(kinds))\n'
        '    text = (\n'
        '        f"Probable-pitcher cache: {count} {noun} ({labels}) recorded today or yesterday; "\n'
        '        "details in this health run\'s log and in health_state/probable_pitcher_cache beside the picks directory. "',
        'def summarize(today, identities, kinds, status_dir="data/health_state/probable_pitcher_cache"):\n'
        '    count = len(identities)\n'
        '    noun = "observation" if count == 1 else "observations"\n'
        '    labels = ", ".join(sorted(kinds))\n'
        '    text = (\n'
        '        f"Probable-pitcher cache: {count} {noun} ({labels}) recorded today or yesterday; "\n'
        '        f"details in {status_dir}. "',
    ),
    (
        "w15b_02_skip_display_preserve_warn",
        "bts/features/probable_pitcher_cache.py",
        'warn(f"could not preserve probable-pitcher cache evidence at {display(target)} ({type(error).__name__})")',
        'warn(f"could not preserve probable-pitcher cache evidence at {target} ({type(error).__name__})")',
    ),
    (
        "w15b_03_skip_display_status_warn",
        "bts/features/probable_pitcher_cache.py",
        'warn(f"could not write probable-pitcher cache status at {display(target)} ({type(error).__name__})")',
        'warn(f"could not write probable-pitcher cache status at {target} ({type(error).__name__})")',
    ),
    (
        "w15b_04_skip_display_dropped_warn",
        "bts/features/probable_pitcher_cache.py",
        'f"probable-pitcher cache {display(cache)}: dropped {dropped} invalid {noun} "',
        'f"probable-pitcher cache {cache}: dropped {dropped} invalid {noun} "',
    ),
    (
        "w15b_05_skip_display_log_unreadable",
        "bts/health/probable_pitcher_cache.py",
        'log.warning("Probable-pitcher cache status unreadable at %s. Send this alert %s", display(path), ROUTE)',
        'log.warning("Probable-pitcher cache status unreadable at %s. Send this alert %s", path, ROUTE)',
    ),
    (
        "w15b_06_skip_display_evidence_copy",
        "bts/health/probable_pitcher_cache.py",
        'evidence = f"That run recorded an evidence copy at {display(copy)}."',
        'evidence = f"That run recorded an evidence copy at {copy}."',
    ),
    (
        "w15b_07_display_drops_backticks",
        "bts/probable_pitcher_status.py",
        'return str(value).replace("\\\\", "\\\\\\\\").replace("`", "\\\\x60")',
        'return str(value).replace("\\\\", "\\\\\\\\").replace("`", "")',
    ),
    (
        "w15b_08_display_skips_backslash_doubling",
        "bts/probable_pitcher_status.py",
        'return str(value).replace("\\\\", "\\\\\\\\").replace("`", "\\\\x60")',
        'return str(value).replace("`", "\\\\x60")',
    ),
    (
        "w15b_09_escape_inside_record_cache_path",
        "bts/features/probable_pitcher_cache.py",
        "reason=reason, cache_path=str(cache), original_sha256=original_sha, preserved=preserved,",
        "reason=reason, cache_path=display(cache), original_sha256=original_sha, preserved=preserved,",
    ),
    (
        "w15b_10a_escape_inside_file_incident_key",
        "bts/health/probable_pitcher_cache.py",
        "identities.append(path.name)",
        "identities.append(display(path.name))",
    ),
    (
        "w15b_10b_escape_inside_dir_incident_key",
        "bts/health/probable_pitcher_cache.py",
        "return [summarize(today, [status_dir.name], {STATUS_UNREADABLE})]",
        "return [summarize(today, [display(status_dir.name)], {STATUS_UNREADABLE})]",
    ),
    (
        "w15b_11_escape_unreadable_cache_stderr",
        "bts/features/probable_pitcher_cache.py",
        'warn(f"unreadable probable-pitcher cache {path} ({error_name}); rebuilding it from raw feeds")',
        'warn(f"unreadable probable-pitcher cache {display(path)} ({error_name}); rebuilding it from raw feeds")',
    ),
    (
        "w15b_12_stderr_errors_replace",
        "bts/features/probable_pitcher_cache.py",
        'os.write(fd, f"{line}\\n".encode(stream.encoding or "utf-8", stream.errors or "backslashreplace"))',
        'os.write(fd, f"{line}\\n".encode(stream.encoding or "utf-8", errors="replace"))',
    ),
    # --- Earlier rounds' mutation sets (Round 1 + R5 + Round 2) ---
    (
        "r1_13_producer_accept_bool_as_int",
        "bts/features/probable_pitcher_cache.py",
        "and all(value is None or type(value) is int for value in entry.values()))",
        "and all(value is None or isinstance(value, int) for value in entry.values()))",
    ),
    (
        "r1_14_reader_accept_bool_as_int",
        "bts/probable_pitcher_status.py",
        "elif (any(type(value) is not int or value < 0 for value in (dropped, aliases, unusable, missing))",
        "elif (any(not isinstance(value, int) or value < 0 for value in (dropped, aliases, unusable, missing))",
    ),
    (
        "r1_15_skip_scanner_validation",
        "bts/features/probable_pitcher_cache.py",
        "            if valid(entry):\n                scanned[game] = entry",
        "            if True:\n                scanned[game] = entry",
    ),
    (
        "r1_16_rewrite_even_when_preserve_or_publish_fails",
        "bts/features/probable_pitcher_cache.py",
        "    if scanned and can_rewrite and not special:",
        "    if scanned and not special:",
    ),
    (
        "r1_17_publish_after_cache_replace",
        "bts/features/probable_pitcher_cache.py",
        "        can_rewrite = report(cache, original, special, \"invalid_entries\", scanned, outcomes, counts)\n\n    lookup.update(scanned)\n    if scanned and can_rewrite and not special:\n        cache.parent.mkdir(parents=True, exist_ok=True)\n        temp = cache.with_name(f\".{cache.name}.{os.getpid()}.tmp\")\n        try:\n            temp.write_text(json.dumps({str(game): entry for game, entry in lookup.items()}))\n            os.replace(temp, cache)\n        finally:\n            temp.unlink(missing_ok=True)",
        "        pass\n\n    lookup.update(scanned)\n    if scanned and not special:\n        cache.parent.mkdir(parents=True, exist_ok=True)\n        temp = cache.with_name(f\".{cache.name}.{os.getpid()}.tmp\")\n        try:\n            temp.write_text(json.dumps({str(game): entry for game, entry in lookup.items()}))\n            os.replace(temp, cache)\n        finally:\n            temp.unlink(missing_ok=True)\n    if invalid and not unreadable:\n        report(cache, original, special, \"invalid_entries\", scanned, outcomes, counts)",
    ),
    (
        "r1_18_reuse_evidence_by_name_only",
        "bts/features/probable_pitcher_cache.py",
        "        if not EVIDENCE.fullmatch(candidate.name):\n            continue\n        try:\n            with open_regular(candidate) as stream:\n                candidate_sha = hashlib.file_digest(stream, \"sha256\").hexdigest()\n        except OSError:\n            continue\n        if candidate_sha == original_sha:\n            return candidate.name",
        "        if not EVIDENCE.fullmatch(candidate.name):\n            continue\n        if candidate.name.endswith(f\".{original_sha[:12]}.json\"):\n            return candidate.name",
    ),
    (
        "r1_19_omit_date_from_record_name",
        "bts/features/probable_pitcher_cache.py",
        'target = cache.parent.parent / "health_state" / SOURCE / f"{day}.{reason}.{identity[:12]}.json"',
        'target = cache.parent.parent / "health_state" / SOURCE / f"{reason}.{identity[:12]}.json"',
    ),
    (
        "r1_20_hash_10_id_sample_instead_of_full_outcomes",
        "bts/features/probable_pitcher_cache.py",
        "identity = digest(canonical(dict(observation, outcomes=outcomes)))",
        "identity = digest(canonical(dict(observation, outcomes=outcomes[:10])))",
    ),
    (
        "r1_21_drop_yesterday_from_health_window",
        "bts/health/probable_pitcher_cache.py",
        'patterns = (f"{today}.*.json", f"{today - timedelta(days=1)}.*.json")',
        'patterns = (f"{today}.*.json",)',
    ),
    (
        "r1_22_one_incident_key_per_reason",
        "bts/health/probable_pitcher_cache.py",
        'return Alert("WARN", SOURCE, text, incident_key=f"{SOURCE}:{today}:{digest}")',
        'return Alert("WARN", SOURCE, text, incident_key=f"{SOURCE}:{today}:{sorted(kinds)[0]}")',
    ),
    (
        "r1_23_follow_status_record_symlinks",
        "bts/probable_pitcher_status.py",
        "    if not follow_symlinks:\n        flags |= os.O_NOFOLLOW",
        "    if False:\n        flags |= os.O_NOFOLLOW",
    ),
    (
        "r1_r5_24_skip_existing_record_field_comparison",
        "bts/features/probable_pitcher_cache.py",
        '    if any(existing[key] != value for key, value in row.items() if key != "recorded_at"):\n        raise FileExistsError(path)',
        "    return",
    ),
    (
        "r1_r5_25_raw_root_oserror_returns_incomplete_false",
        "bts/features/probable_pitcher_cache.py",
        "    except OSError:\n        return scanned, rejected, True",
        "    except OSError:\n        return scanned, rejected, False",
    ),
    (
        "r1_r5_26_skip_producer_64kib_check",
        "bts/features/probable_pitcher_cache.py",
        '    if len(payload) > MAX_BYTES:\n        raise ValueError("status exceeds size limit")',
        "    pass",
    ),
    (
        "r2_27_fd_leak_in_open_regular",
        "bts/probable_pitcher_status.py",
        '    fd = os.open(path, flags)\n    try:\n        if not stat.S_ISREG(os.fstat(fd).st_mode):\n            raise NonRegularFile(path)\n        with os.fdopen(fd, "rb", closefd=False) as stream:\n            yield stream\n    finally:\n        os.close(fd)',
        '    with os.fdopen(os.open(path, flags), "rb") as stream:\n        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):\n            raise NonRegularFile(path)\n        yield stream',
    ),
    (
        "r2_28_unbuffered_print_warn_epipe_exit120",
        "bts/features/probable_pitcher_cache.py",
        "        if fd < 0:\n            print(line, file=stream, flush=True)\n        else:",
        "        if True:\n            print(line, file=stream, flush=True)\n        else:",
    ),
]


def main():
    killed = []
    survived = []
    for name, rel_path, target, replacement in MUTANTS:
        with TemporaryDirectory(dir=HERE, prefix=f"mut-{name}-") as tmp:
            scratch_src = Path(tmp) / "src"
            shutil.copytree(REPO_SRC, scratch_src)
            mod_file = scratch_src / rel_path
            original_text = mod_file.read_text(encoding="utf-8")
            assert target in original_text, f"Target not found for {name} in {rel_path}"
            assert original_text.count(target) == 1, f"Target matched multiple times for {name}"
            mod_file.write_text(original_text.replace(target, replacement, 1), encoding="utf-8")

            env = {
                **os.environ,
                "PYTHONPATH": f"{scratch_src}{os.pathsep}{REPO_ROOT}",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            # Verify bts.__file__ resolves to scratch_src
            chk = subprocess.run(
                [sys.executable, "-c", "import bts, sys; from pathlib import Path; assert Path(bts.__file__).resolve() == Path(sys.argv[1]).resolve() / 'bts/__init__.py'", str(scratch_src)],
                env=env,
                timeout=10,
                check=True,
            )
            # Note: test_wr_w15_incidents.py child_env() prepends REPO_ROOT to PYTHONPATH!
            # Wait: REPO_ROOT does NOT have bts/ at top-level (bts is under REPO_ROOT/src/bts),
            # while editable install .pth in .venv points to REPO_ROOT/src AFTER PYTHONPATH!
            # Putting scratch_src on PYTHONPATH keeps scratch_src ahead of .venv's .pth even when REPO_ROOT is prepended!
            chk_child = subprocess.run(
                [sys.executable, "-c", "from tests.test_wr_w15_incidents import child_env; import subprocess, sys; subprocess.run([sys.executable, '-c', 'import bts, sys; from pathlib import Path; assert Path(bts.__file__).resolve() == Path(sys.argv[1]).resolve() / \"bts/__init__.py\"', sys.argv[1]], env=child_env(), check=True)", str(scratch_src)],
                cwd=REPO_ROOT,
                env=env,
                timeout=10,
                check=True,
            )

            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-x", "--no-header", *TEST_TARGETS],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if proc.returncode != 0:
                first_fail = next(
                    (ln.strip() for ln in proc.stdout.splitlines() if ln.startswith("FAILED ")),
                    f"rc={proc.returncode}",
                )
                killed.append((name, first_fail))
                print(f"KILLED   {name}: {first_fail}", flush=True)
            else:
                survived.append(name)
                print(f"SURVIVED {name}!", flush=True)

    print(f"\nSUMMARY: {len(killed)}/{len(MUTANTS)} killed, {len(survived)} survived: {survived}", flush=True)


if __name__ == "__main__":
    main()
