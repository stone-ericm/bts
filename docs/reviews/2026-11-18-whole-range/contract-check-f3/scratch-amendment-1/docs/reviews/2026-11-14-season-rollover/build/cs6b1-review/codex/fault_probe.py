"""Exercise advisory faults, malformed on-disk inputs, signals, and real DM timing."""

from contextlib import redirect_stderr
from datetime import timedelta
from http.client import HTTPResponse
from io import BufferedReader, BytesIO, RawIOBase, StringIO
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Thread
import time
from types import SimpleNamespace
from urllib.error import URLError

from day_probe import MORNING, REVIEW, ROOT, SHARED, STALE, sandbox, snapshot


class ReviewFault(BaseException):
    pass


class Unprintable(RuntimeError):
    def __str__(self):
        raise RuntimeError("formatting fault")


class ReachedFirstPrediction(BaseException):
    pass


def at_prediction(config, root, now, events, stderr, *, check_patch=None):
    from bts import orchestrator, scheduler

    observed = []

    def prediction(*args, **kwargs):
        observed.append(snapshot(root, events, stderr, StringIO()))
        raise ReachedFirstPrediction()

    if check_patch is None:
        orchestrator.run_cascade.side_effect = prediction
    else:
        check_patch.setattr(scheduler, "run_single_check", prediction)
    try:
        scheduler.run_day("2027-07-15", config)
    except ReachedFirstPrediction:
        assert len(observed) == 1
        return observed[0]
    except BaseException as exc:
        raise AssertionError(f"escaped before first prediction: {exc!r}") from exc
    raise AssertionError(f"did not reach a prediction at {now[0]:%H:%M:%S}")


def unchanged(reference, observed):
    def existing(data):
        return {key: value for key, value in data["files"].items() if not key.endswith(MORNING)}

    assert existing(reference) == existing(observed)
    assert reference["events"] == observed["events"], observed["events"]


def reference(root, prepared=None):
    from bts import picks
    from tests.test_morning_state_cs6b1 import START
    from tests.test_pick_chosen_on_cs5a import daily_pick, set_dated

    with sandbox(root) as (_, config, directory, now, _, events), redirect_stderr(StringIO()) as err:
        if prepared == "delivered":
            picks.save_pick(daily_pick(), directory)
        if prepared == "age":
            set_dated(directory, now=START)
        data = at_prediction(config, root, now, events, err)
        assert "Morning state check" not in err.getvalue()
        return data


def exception_cases(scratch, base):
    from bts import contest_state, daily_decision, picks, scheduler
    from bts.health import alert
    from tests.test_morning_state_cs6b1 import DAY, START
    from tests.test_pick_chosen_on_cs5a import daily_pick, set_dated

    sites = {
        "clock": (scheduler, "_now_et"),
        "min": (scheduler, "min"),
        "gate": (scheduler, "_contest_state_required"),
        "commit": (daily_decision, "is_scoreable_commit"),
        "pick": (picks, "load_pick"),
        "delivered": (picks, "pick_was_delivered"),
        "read": (contest_state, "load_decision_streak_state"),
        "age": (contest_state, "automatic_reading_record"),
        "dispatch": (alert, "dispatch_dm_for_health_alerts"),
        "transport": (alert, "send_dm"),
        "stderr_initial": (scheduler, "print"),
        "stderr_failure": (scheduler, "print"),
        "status_writer": (Path, "write_text"),
    }
    errors = [RuntimeError("ordinary"), OSError("IO"), KeyboardInterrupt("interrupt"), SystemExit(23),
              GeneratorExit("generator"), ReviewFault("custom"),
              BaseExceptionGroup("group", [KeyboardInterrupt(), RuntimeError()]), Unprintable()]
    for site, (module, name) in sites.items():
        for index, error in enumerate(errors):
            root = scratch / f"{site}-{index}"
            with sandbox(root) as (patch, config, directory, now, _, events), redirect_stderr(StringIO()) as err:
                if site == "delivered":
                    picks.save_pick(daily_pick(), directory)
                if site == "age":
                    set_dated(directory, now=START)
                injected = []
                attempted = []
                original = getattr(module, name, {"print": print, "min": min}.get(name))
                if site not in {"clock", "min", "gate", "status_writer"}:
                    alert.send_dm.side_effect = lambda *_: attempted.append(now[0])

                def raise_once(*args, **kwargs):
                    inject = now[0] == START and not injected
                    if site == "clock":
                        caller = sys._getframe(1)
                        inject &= (caller.f_code is scheduler.run_day.__code__
                                   and "state" in caller.f_locals and "coalesced" not in caller.f_locals)
                    if site in {"stderr_initial", "stderr_failure"}:
                        prefix = "  MORNING STATE CHECK" if site == "stderr_initial" else "  Morning state check failed"
                        inject &= args[0].startswith(prefix)
                    if site == "status_writer":
                        inject &= args[0].name == MORNING
                    if inject:
                        injected.append(now[0])
                        raise error
                    return original(*args, **kwargs)

                patch.setattr(module, name, raise_once, raising=False)
                if site == "stderr_failure":
                    def force_report(*args, **kwargs):
                        raise RuntimeError("earlier startup failure")

                    patch.setattr(contest_state, "load_decision_streak_state", force_report)
                observed = at_prediction(config, root, now, events, err)
                assert injected == [START], (site, error, err.getvalue())
                unchanged(base[site if site in {"delivered", "age"} else "plain"], observed)
                if site == "stderr_initial":
                    assert attempted == [START], (site, error, err.getvalue())
                if site in {"commit", "pick"} and isinstance(error, OSError):
                    assert attempted == [START]
        print(f"{site}: {len(errors)} injected exceptions; first real prediction reached", flush=True)

    for site in ("first check", "initial state save"):
        root = scratch / site
        with sandbox(root) as (patch, config, _, _, _, _), redirect_stderr(StringIO()):
            error = KeyboardInterrupt(site)

            def interrupt(*args, **kwargs):
                raise error

            patch.setattr(scheduler, "run_single_check" if site == "first check" else "save_state", interrupt)
            try:
                scheduler.run_day(DAY, config)
            except KeyboardInterrupt as received:
                assert received is error
            else:
                raise AssertionError(f"ordinary scheduler interrupt swallowed at {site}")
            print(f"{site}: outside-startup interrupt propagated")


def plan_faults(scratch):
    from bts import scheduler
    from tests.test_morning_state_cs6b1 import DAY, FIRST

    for key, plan, error_type in (
        ("missing time_et", [{"game_pks": [101]}], KeyError),
        ("naive time_et", [{"game_pks": [101], "time_et": FIRST.replace(tzinfo=None)}], TypeError),
    ):
        with sandbox(scratch / key) as (patch, config, _, _, _, _), redirect_stderr(StringIO()) as err:
            patch.setattr(scheduler, "plan_run_times", lambda *a, **kw: plan)
            try:
                scheduler.run_day(DAY, config)
            except error_type as error:
                print(f"{key}: existing {type(error).__name__} {error}, stderr {err.getvalue().splitlines()!r}")
            else:
                raise AssertionError(key)


def print_variants(scratch):
    from bts import scheduler
    from bts.health import alert
    from tests.test_morning_state_cs6b1 import DAY, START
    from tests.test_pick_chosen_on_cs5a import set_dated
    from tests.test_stale_delivery_cs4c import seed_automatic

    for kind in ("accepted", "stale", "due", "empty"):
        faulted = []

        class FaultStream(StringIO):
            def write(self, message):
                if message.startswith(("  Morning state check: ", "  MORNING STATE CHECK", "  Morning state check skipped")):
                    faulted.append(message)
                    raise KeyboardInterrupt("stderr write boundary")
                return super().write(message)

        with sandbox(scratch / kind) as (patch, config, directory, now, _, events), redirect_stderr(FaultStream()) as err:
            if kind == "accepted":
                set_dated(directory, now=START)
            if kind == "stale":
                seed_automatic(directory, (START - timedelta(hours=20)).isoformat(), now=START, streak=30)
            if kind in {"due", "empty"}:
                plan = [] if kind == "empty" else [{"game_pks": [101], "time_et": now[0]}]
                patch.setattr(scheduler, "plan_run_times", lambda *a, **kw: plan)
            try:
                scheduler.run_day(DAY, config)
            except BaseException as error:
                raise AssertionError(f"stderr fault escaped for {kind}: {error!r}") from error
            assert len(faulted) == 1 and "stderr write boundary" in err.getvalue()
            assert alert.send_dm.called == (kind == "stale")
            assert any(event[0] == "run_single_check" for event in events) == (kind != "empty")
            print(f"stream.write at {kind}: contained, send={alert.send_dm.called}", flush=True)


def files(scratch):
    from bts import daily_decision, scheduler
    from bts.health import alert
    from tests.test_morning_state_cs6b1 import DAY, START

    records = {
        "decision bad encoding": ("decision", b"\xff"), "pick bad encoding": ("pick", b"\xff"),
        "decision action array": ("decision", json.dumps({"schema_version": daily_decision.ACCEPTED_SCHEMAS[0],
                                                           "action": [], "date": DAY, "scoreable": True}).encode()),
        "decision deeply nested": ("decision", b"[" * 20000 + b"]" * 20000),
        "pick deeply nested": ("pick", b"[" * 20000 + b"]" * 20000),
    }
    for name, (kind, value) in records.items():
        root = scratch / name
        with sandbox(root) as (patch, config, directory, now, _, events), redirect_stderr(StringIO()) as err:
            path = daily_decision.decision_path(DAY, directory) if kind == "decision" else directory / f"{DAY}.json"
            path.parent.mkdir(parents=True)
            path.write_bytes(value)
            at_prediction(config, root, now, events, err, check_patch=patch if kind == "pick" else None)
            assert alert.send_dm.called == ("encoding" in name)
            print(f"{name}: DM={alert.send_dm.called}, {advisory(err)!r}")

    for mode in ("unwritable", "directory", "invalid json", "valid array", "valid null"):
        root = scratch / mode
        with sandbox(root) as (patch, config, directory, now, _, events), redirect_stderr(StringIO()) as err:
            path = directory.parent / "health_state" / MORNING
            path.parent.mkdir(parents=True)
            original_write = Path.write_text
            if mode == "unwritable":
                def write(self, *args, **kwargs):
                    if self == path:
                        raise PermissionError(13, "permission denied", str(self))
                    return original_write(self, *args, **kwargs)

                patch.setattr(Path, "write_text", write)
            elif mode == "directory":
                path.mkdir()
            else:
                path.write_text({"invalid json": "{", "valid array": "[]", "valid null": "null"}[mode])
            at_prediction(config, root, now, events, err)
            status = json.loads(path.read_text()).get("status") if mode == "invalid json" else None
            assert alert.send_dm.called == (mode in {"unwritable", "directory", "invalid json"})
            print(f"status {mode}: DM={alert.send_dm.called}, status={status}, {advisory(err)!r}")

    for status in (MORNING, STALE):
        shared_status(scratch / status, status)


def shared_status(scratch, status_name):
    from bts import scheduler
    from bts.health import alert
    from tests.test_morning_state_cs6b1 import DAY, START
    from tests.test_stale_delivery_cs4c import seed_automatic

    for outcome in ("sent", "failed", "no recipient"):
        root = scratch / outcome
        with sandbox(root) as (patch, config, directory, now, _, _), redirect_stderr(StringIO()):
            shared = directory.parent / "health_state" / SHARED
            shared.parent.mkdir(parents=True)
            seed = b'{ "identifiable": "shared " }\n'
            shared.write_bytes(seed)
            touched = []
            if status_name == STALE:
                seed_automatic(directory, (START - timedelta(hours=20)).isoformat(), now=START, streak=30)
            with patch.context() as isolated:
                original = Path.open

                def open_path(self, *args, **kwargs):
                    if self == shared:
                        touched.append(args)
                        raise AssertionError("morning attempted to open shared file")
                    return original(self, *args, **kwargs)

                isolated.setattr(Path, "open", open_path)
                if outcome == "failed":
                    alert.send_dm.side_effect = OSError("captured transport failure")
                if outcome == "no recipient":
                    config["bluesky"].clear()
                scheduler._morning_state_check(config, directory, DAY, START, START + timedelta(minutes=10))
            assert not touched and shared.read_bytes() == seed
            path = shared.with_name(status_name)
            if outcome == "no recipient":
                assert not path.exists() and not alert.send_dm.called
            else:
                alert.send_dm.assert_called_once()
                assert json.loads(path.read_text())["status"] == outcome
            print(f"shared status {status_name} {outcome}: zero opens and identical bytes")


def advisory(err):
    return [line for line in err.getvalue().splitlines()
            if line.startswith(("  Morning state check", "  MORNING STATE CHECK"))]


def timing(root, dribble_seconds):
    from bts import dm, scheduler, util
    from bts.health import alert
    from tests.test_morning_state_cs6b1 import DAY, FIRST

    with sandbox(root) as (patch, config, _, now, _, events), redirect_stderr(StringIO()):
        now[0] = FIRST - timedelta(minutes=10)
        starts = now[0]
        requests = []
        backoffs = []
        responses = [b'{"accessJwt":"capture"}', b'{"did":"did:plc:capture"}',
                     b'{"convo":{"id":"capture"}}', b'{"id":"capture"}']

        def advance(seconds):
            now[0] += timedelta(seconds=seconds)

        def backoff(seconds):
            backoffs.append(seconds)
            advance(seconds)

        def captured_urlopen(request, timeout):
            requests.append([request.full_url.rsplit("/", 1)[-1], timeout])
            advance(timeout)
            step = len(requests)
            if step not in {3, 6, 9, 10}:
                raise URLError(TimeoutError("captured connect timeout"))
            body = responses[[3, 6, 9, 10].index(step)]
            if step == 10 and dribble_seconds:
                pulses = dribble_seconds // 14
                body = b" " * pulses + body

                class Wire(RawIOBase):
                    def __init__(self):
                        self.headers = f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n\r\n".encode()
                        self.body = BytesIO(body)
                        self.remaining = pulses

                    def readable(self):
                        return True

                    def readinto(self, target):
                        if self.headers:
                            part, self.headers = self.headers[:len(target)], self.headers[len(target):]
                        elif self.remaining:
                            self.remaining -= 1
                            advance(14)
                            part = self.body.read(1)
                        else:
                            part = self.body.read(len(target))
                        target[:len(part)] = part
                        return len(part)

                response = HTTPResponse(SimpleNamespace(makefile=lambda *a, **kw: BufferedReader(Wire())))
                response.begin()
                return response
            return BytesIO(body)

        patch.setattr(util, "urlopen", captured_urlopen)
        patch.setattr(util, "time", SimpleNamespace(sleep=backoff))
        patch.setattr(dm, "get_dm_password", lambda: "capture")
        # world replaces bts.dm.send_dm; recover the real function already imported by the dispatcher.
        patch.setattr(alert, "send_dm", REAL_SEND_DM)
        scheduler.run_day(DAY, config)
        first = next(e[1] for e in events if e[0] == "run_single_check")
        assert len(requests) == 10 and all(entry[1] == 15 for entry in requests)
        assert backoffs == [5, 10] * 3
        print(f"transport at {starts:%H:%M:%S}: requests={requests!r}; backoffs={backoffs}; "
              f"elapsed={195 + dribble_seconds}s; first target={FIRST:%H:%M:%S}; actual check={first}")


def external_signal(root, head):
    from bts import scheduler
    from bts.health import alert
    from tests.test_morning_state_cs6b1 import DAY, FIRST

    start, sent = Event(), Event()

    def sender():
        if start.wait(2):
            os.kill(os.getpid(), signal.SIGINT)
            sent.set()

    def parked(*args):
        start.set()
        time.sleep(.25)

    thread = Thread(target=sender)
    with sandbox(root) as (patch, config, _, _, _, events), redirect_stderr(StringIO()) as err:
        if head:
            alert.send_dm.side_effect = parked
        else:
            patch.setattr(scheduler, "_watchdog_ping_sleep", parked)
        assert signal.getsignal(signal.SIGINT) == signal.default_int_handler
        thread.start()
        escaped = False
        try:
            scheduler.run_day(DAY, config)
        except KeyboardInterrupt:
            escaped = True
        finally:
            thread.join()
        assert sent.is_set() and escaped is not head
        checks = [event[1] for event in events if event[0] == "run_single_check"]
        assert (FIRST.isoformat() in checks) is head
        print(f"external SIGINT: escaped={escaped}; checks={checks}; journal={advisory(err)}")


if __name__ == "__main__":
    from bts.dm import send_dm as REAL_SEND_DM

    with TemporaryDirectory(prefix="cs6b1-fault-") as scratch:
        scratch = Path(scratch)
        if len(sys.argv) > 1 and sys.argv[1] == "reference":
            print(json.dumps({name: reference(scratch / name, name) for name in ("plain", "delivered", "age")}))
        elif len(sys.argv) > 1:
            if sys.argv[1] == "base":
                plan_faults(scratch)
            external_signal(scratch / "signal", sys.argv[1] == "head")
        else:
            source = REVIEW / "base" / "src"
            base = subprocess.check_output([sys.executable, __file__, "reference"], cwd=ROOT,
                                           env={**os.environ, "PYTHONPATH": f"{source}:{ROOT}"})
            exception_cases(scratch, json.loads(base))
            plan_faults(scratch)
            print_variants(scratch)
            files(scratch)
            timing(scratch / "normal", 0)
            timing(scratch / "slow", 630)
            for identity, source in (("base", REVIEW / "base" / "src"), ("head", ROOT / "src")):
                subprocess.run([sys.executable, __file__, identity], check=True, cwd=ROOT,
                               env={**os.environ, "PYTHONPATH": f"{source}:{ROOT}"})
