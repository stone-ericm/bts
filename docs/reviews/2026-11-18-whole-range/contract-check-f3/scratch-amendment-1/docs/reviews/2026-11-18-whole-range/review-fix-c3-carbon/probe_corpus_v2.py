"""Runner for probe_corpus_and_fold importing bts.picks before locking socket."""
from __future__ import annotations

import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

import probe_corpus_and_fold as core


REVIEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = REVIEW_DIR.parents[1]


def _worker_main(src_dir: str) -> int:
    sys.path.insert(0, src_dir)
    import bts.picks as picks

    def _deny_net(*args, **kwargs):
        raise AssertionError("Network access is forbidden")

    socket.socket = _deny_net  # type: ignore[assignment]
    socket.create_connection = _deny_net  # type: ignore[assignment]

    class _Alarm(BaseException):
        pass

    def _on_alarm(*_args):
        raise _Alarm()

    signal.signal(signal.SIGALRM, _on_alarm)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        picks_dir = Path(req["picks_dir"])
        season = req["season"]
        today = req["today"]
        signal.alarm(3)
        try:
            res = picks._replay_season_streak(picks_dir, season, today)
        except _Alarm:
            out = {"status": "timeout"}
        except Exception as exc:  # noqa: BLE001
            out = {"status": "raised", "exc": type(exc).__name__, "msg": str(exc)}
        else:
            if res is None:
                out = {"status": "none"}
            else:
                s, sv = res
                assert type(s) is int and type(sv) is bool, f"Unexpected return types: {res!r}"
                out = {"status": "ok", "value": [s, sv]}
        finally:
            signal.alarm(0)
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()
    return 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        return _worker_main(sys.argv[2])

    with TemporaryDirectory(dir=REVIEW_DIR, prefix="_corpus_run_") as scratch_str:
        scratch = Path(scratch_str)
        base_root = scratch / "base_c2039541"
        base_root.mkdir()
        with subprocess.Popen(
            ["git", "archive", "c2039541", "src"],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
        ) as archive:
            assert archive.stdout is not None
            with tarfile.open(fileobj=archive.stdout, mode="r|") as tf:
                tf.extractall(base_root, filter="data")
            assert archive.wait() == 0, "git archive c2039541 failed"

        all_cases = core._build_deterministic_cases() + core._build_seeded_cases()
        assert len(all_cases) >= 300, f"Expected >=300 histories, built {len(all_cases)}"

        base_proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(base_root / "src")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        head_proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(REPO_ROOT / "src")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )

        def _ask(proc: subprocess.Popen, picks_dir: Path, season: int, today: str) -> dict:
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(json.dumps({"picks_dir": str(picks_dir), "season": season, "today": today}) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            assert line, "Worker exited unexpectedly"
            return json.loads(line)

        summary_counts = {"total": len(all_cases), "same": 0, "a": 0, "b": 0, "c": 0, "indep_mismatches": 0}
        divergences: list[dict] = []
        indep_mismatches: list[dict] = []

        try:
            for idx, spec in enumerate(all_cases):
                case_dir = scratch / f"case_{idx:04d}"
                case_dir.mkdir()
                spec.builder(case_dir)

                base_res = _ask(base_proc, case_dir, spec.season, spec.today)
                head_res = _ask(head_proc, case_dir, spec.season, spec.today)
                indep_res = core.independent_fold(case_dir, spec.season, spec.today)

                head_norm = {"status": head_res["status"], **({"value": head_res["value"]} if head_res["status"] == "ok" else {})}
                indep_norm = {"status": indep_res["status"], **({"value": indep_res["value"]} if indep_res["status"] == "ok" else {})}
                if head_norm != indep_norm:
                    summary_counts["indep_mismatches"] += 1
                    indep_mismatches.append({
                        "name": spec.name,
                        "head": head_res,
                        "indep": indep_res,
                    })

                cat = core._classify_actual_diff(spec, base_res, head_res)
                summary_counts[cat] += 1
                if cat != "same":
                    divergences.append({
                        "name": spec.name,
                        "category": cat,
                        "real_writer_possible": spec.real_writer_possible,
                        "base": base_res,
                        "head": head_res,
                        "indep_blocker": indep_res.get("blocker"),
                        "note": spec.note,
                    })
        finally:
            for proc in (base_proc, head_proc):
                if proc.stdin:
                    proc.stdin.close()
                proc.wait(timeout=5)

        out_report = {
            "summary": summary_counts,
            "indep_mismatches": indep_mismatches,
            "cat_c": [d for d in divergences if d["category"] == "c"],
            "all_divergences": divergences,
        }
        out_path = REVIEW_DIR / "corpus_results.json"
        out_path.write_text(json.dumps(out_report, indent=2) + "\n")
        print(json.dumps(summary_counts, indent=2))
        if indep_mismatches:
            print("INDEPENDENT FOLD MISMATCHES:", json.dumps(indep_mismatches, indent=2))
            return 1
        if out_report["cat_c"]:
            print("CATEGORY C DIVERGENCES:", json.dumps(out_report["cat_c"], indent=2))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
