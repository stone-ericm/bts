"""C2 step 2a §5 gate: the scenarios, run identically against the deployed baseline (f882411, by `generate.py`) and the
candidate (by `tests/c2_2a/test_golden.py`). Design `docs/superpowers/specs/2026-10-06-c2-2a-serving-record-design.md`
§5.2–§5.5.

Each scenario builds the synthetic world (`world.py`) in a fresh directory, runs the REAL pick path — `run_day` /
`run_single_check` / `run_and_pick` / `run_cascade` / `predict_local` / `run_pipeline` / `predict` / `save_slate` /
`select_pick` / delivery — with only the leaves replaced: the MLB API (a fixed router), training (fixed stand-in
models), the season refresh, clocks (one scenario clock behind every module's `datetime`/`date` and the scheduler's
`time`), the delivery transports (recorded), result polling and the live-forward trigger. It returns an observation of
the compared surface (§5.2): the predictions selection read, each `SelectionResult`, each lock decision and fallback
plan, the transport log, every file the day wrote under `data/picks` (bytes), the cache file's bytes, any exception,
and the slate (whose schema tag, `serving` and posted rows' `projected` are the only intended differences).

Faults that target code only the candidate has (the witness's own names) are applied only where that name exists, so
the baseline runs the plain scenario and equality shows the fault is contained.
"""
from __future__ import annotations

import builtins
import contextlib
import hashlib
import io
import json
import os
import pickle
import shutil
import sys
import time
from datetime import date as _real_date, datetime as _real_datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from tests.c2_2a.golden import fakes, world as W

ET = ZoneInfo("America/New_York")
UTC = timezone.utc
PRED_COLUMNS = ["p_game_hit", "p_game_blend", "p_hit_vs_starter", "p_hit_vs_reliever", "est_pas", "starter_pas",
                "reliever_pas", "flags", "p_game_hit_raw"]
_MISSING = object()


# ---------------------------------------------------------------- patching and clocks

class Patcher:
    def __init__(self):
        self._undo = []

    def set(self, obj, name, value):
        old = getattr(obj, name, _MISSING)
        self._undo.append((obj, name, old))
        setattr(obj, name, value)

    def set_dict_item(self, d: dict, key, value):
        old = d.get(key, _MISSING)
        self._undo.append((_DictSlot(d, key), "value", old))
        d[key] = value

    def set_if_exists(self, obj, name, value) -> bool:
        if not hasattr(obj, name):
            return False
        self.set(obj, name, value)
        return True

    def restore(self):
        for obj, name, old in reversed(self._undo):
            if old is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, old)
        self._undo.clear()


class _DictSlot:
    """Lets Patcher.restore() put a dict item back through setattr/delattr."""
    def __init__(self, d, key):
        object.__setattr__(self, "_d", d)
        object.__setattr__(self, "_key", key)

    def __setattr__(self, name, value):
        self._d[self._key] = value

    def __delattr__(self, name):
        self._d.pop(self._key, None)


class Clock:
    def __init__(self, start: _real_datetime, step_on_read: timedelta | None = None):
        self.now = start
        self.step_on_read = step_on_read

    def read(self) -> _real_datetime:
        now = self.now
        if self.step_on_read is not None:
            self.now = self.now + self.step_on_read
        return now

    def advance(self, seconds: float):
        self.now = self.now + timedelta(seconds=seconds)


def _frozen_types(clock: Clock):
    class FrozenDateTime(_real_datetime):
        @classmethod
        def _of(cls, x):
            return cls(x.year, x.month, x.day, x.hour, x.minute, x.second, x.microsecond, tzinfo=x.tzinfo,
                       fold=x.fold)

        @classmethod
        def now(cls, tz=None):
            c = clock.now
            return cls._of(c.astimezone(ET).replace(tzinfo=None) if tz is None else c.astimezone(tz))

        @classmethod
        def utcnow(cls):
            return cls._of(clock.now.astimezone(UTC).replace(tzinfo=None))

        @classmethod
        def today(cls):
            return cls.now()

    class FrozenDate(_real_date):
        @classmethod
        def today(cls):
            d = clock.now.astimezone(ET).date()
            return cls(d.year, d.month, d.day)

    return FrozenDateTime, FrozenDate


class _TimeProxy:
    """The scheduler's `time`: sleeps advance the scenario clock and monotonic reads it; nothing else is touched."""

    def __init__(self, clock: Clock, crash_after_sleeps: int | None = None):
        self._clock = clock
        self._crash_after = crash_after_sleeps
        self.sleeps = 0

    def sleep(self, seconds):
        self.sleeps += 1
        if self._crash_after is not None and self.sleeps > self._crash_after:
            raise _DaemonKilled(f"killed at sleep {self.sleeps}")
        self._clock.advance(float(seconds))

    def monotonic(self):
        return (self._clock.now - _real_datetime(2026, 1, 1, tzinfo=UTC)).total_seconds()

    def time(self):
        return self._clock.now.timestamp()

    def __getattr__(self, name):
        return getattr(time, name)


class _DaemonKilled(BaseException):
    pass


# ---------------------------------------------------------------- the observation

def _num(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return "nan"
    except (TypeError, ValueError):
        pass
    if isinstance(v, (bool, str)):
        return v
    try:
        return repr(float(v))
    except (TypeError, ValueError):
        return repr(v)


def _predictions(df) -> dict | None:
    if df is None:
        return None
    rows = []
    for _, r in df.iterrows():
        row = {"batter_id": int(r["batter_id"]), "game_pk": int(r["game_pk"])}
        for c in PRED_COLUMNS:
            if c in df.columns:
                row[c] = _num(r[c])
        row["projected_lineup"] = "PROJECTED" in str(r.get("flags", ""))
        rows.append(row)
    return {"columns": sorted(c for c in PRED_COLUMNS if c in df.columns), "rows": rows}


def _jsonable(obj):
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return _num(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        items = [_jsonable(v) for v in obj]
        return sorted(items, key=repr) if isinstance(obj, set) else items
    if isinstance(obj, _real_datetime):
        return obj.isoformat()
    if hasattr(obj, "__dataclass_fields__"):
        return {f: _jsonable(getattr(obj, f)) for f in obj.__dataclass_fields__}
    return repr(obj)


def _selection(sel) -> dict | None:
    if sel is None:
        return None
    out = {k: _jsonable(getattr(sel, k, None)) for k in (
        "action", "source", "primary_candidate", "double_candidate", "no_pick_reason", "streak", "saver_available")}
    pr = getattr(sel, "pick_result", None)
    daily = getattr(pr, "daily", None) if pr is not None else None
    out["locked"] = getattr(pr, "locked", None) if pr is not None else None
    out["daily"] = None if daily is None else {
        "pick": _jsonable(daily.pick), "double_down": _jsonable(daily.double_down),
        "runner_up": _jsonable(daily.runner_up)}
    return out


def _files(root: Path, inputs: dict) -> dict:
    out = {}
    picks = root / "data" / "picks"
    for p in sorted(picks.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if not p.is_file() or "/slates/" in f"/{p.relative_to(picks).as_posix()}":
            continue
        raw = p.read_bytes()
        if inputs.get(rel) == hashlib.sha256(raw).hexdigest():
            continue                                         # an unchanged input
        try:
            out[rel] = raw.decode("utf-8")
        except UnicodeDecodeError:
            out[rel] = "sha256:" + hashlib.sha256(raw).hexdigest()
    return out


def _cache(root: Path) -> str | None:
    p = root / "data" / "models" / f"blend_{W.DATE}.pkl"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def _slate(root: Path) -> dict | None:
    p = root / "data" / "picks" / "slates" / f"{W.DATE}.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    rows = s.get("rows") or []
    return {"tier": s.get("tier"), "date": s.get("date"), "n_rows": s.get("n_rows"), "written_at": s.get("written_at"),
            "envelope_keys": sorted(k for k in s if k != "serving"),
            "rows": [{k: v for k, v in r.items() if k != "projected"} for r in rows],
            "projected": [r.get("projected") for r in rows],
            "schema_version": s.get("schema_version"), "serving": s.get("serving", _MISSING_SERVING)}


_MISSING_SERVING = "<absent>"


# ---------------------------------------------------------------- the harness

_TEMPLATE: dict[str, Path] = {}


def _template(repo: Path, key: str, **world_kw) -> tuple[Path, dict]:
    import tempfile
    if key not in _TEMPLATE:
        root = Path(tempfile.mkdtemp(prefix=f"c2-2a-golden-template-{key}-"))
        W.write_world(root, repo=repo, **world_kw)
        _TEMPLATE[key] = root
    return _TEMPLATE[key], {}


def config(delivery: str = "private") -> dict:
    return {
        "orchestrator": {"picks_dir": "data/picks", "models_dir": "data/models", "heartbeat_path": "hb/.heartbeat"},
        "tiers": [{"name": "local", "type": "local"}],
        "bluesky": {"dm_recipient": "golden-recipient"},
        "scheduler": {"pick_delivery": delivery, "early_lock_gap": 0.03, "lineup_check_offset_min": 60,
                      "cluster_min": 10, "doubleheader_recheck_min": 15, "fallback_deadline_min": 35,
                      "fallback_deadline_min_morning": 25, "results_poll_interval_min": 15,
                      "results_cap_hour_et": 5, "cascade_budget_min": 12, "operator_reserve_min": 10,
                      "missed_pick_alert_min": 10},
        "health_checks": {"enabled": False},
    }


class Harness:
    """One scenario's world, patches and spies."""

    def __init__(self, repo: Path, workdir: Path, *, start_et=(10, 0), streak=0, calibration_history=True,
                 all_posted=False, statuses=None, env=None, step_on_read=None):
        self.repo, self.p, self.obs = repo, Patcher(), {
            "exception": None, "selections": [], "lock_decisions": [], "fallback_plans": [], "transport": [],
            "pick_reads": {}, "resolver_inventory": [], "stderr_failures": []}
        key = f"s{streak}-c{int(calibration_history)}"
        template, _ = _template(repo, key, streak=streak, calibration_history=calibration_history)
        self.root = workdir
        if self.root.exists():
            shutil.rmtree(self.root)
        shutil.copytree(template, self.root)
        self.inputs = {p.relative_to(self.root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((self.root / "data").rglob("*")) if p.is_file()}
        self.obs["world_inputs"] = dict(self.inputs)
        self.clock = Clock(_real_datetime(2026, 6, 30, *start_et, tzinfo=ET), step_on_read)
        self.router = W.Router(statuses=statuses)
        self.all_posted, self.env = all_posted, env or {}
        self._cwd = None
        self._env_undo: list = []

    # -- lifecycle
    def __enter__(self):
        self._cwd = os.getcwd()
        os.chdir(self.root)
        self._patch_standard()
        return self

    def __exit__(self, *exc):
        self.p.restore()
        for key, old in reversed(self._env_undo):
            if old is _MISSING:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
        os.chdir(self._cwd)
        return False

    def _patch_standard(self):
        import bts.contest_state, bts.daily_decision, bts.dm, bts.heartbeat, bts.orchestrator, bts.picks  # noqa: F401
        import bts.posting, bts.saver_state, bts.scheduler, bts.slate, bts.strategy, bts.util  # noqa: F401
        from bts.model import predict as P
        sch = sys.modules["bts.scheduler"]
        p = self.p
        # The MLB API, training, the season refresh.
        if self.all_posted:
            p.set(W, "POSTED_TEAMS", {t[0] for t in W.TEAMS})
        p.set(P, "urlopen", self.router)
        p.set(sys.modules["bts.util"], "urlopen", self.router)
        p.set(P, "_refresh_season_data", lambda *a, **k: None)
        p.set(P, "train_model", fakes.train_model)
        p.set(P, "train_blend", fakes.train_blend)
        # Clocks.
        fdt, fd = _frozen_types(self.clock)
        for name, mod in list(sys.modules.items()):
            if not (name == "bts" or name.startswith("bts.")) or mod is None:
                continue
            if getattr(mod, "datetime", None) is _real_datetime:
                p.set(mod, "datetime", fdt)
            if getattr(mod, "date", None) is _real_date:
                p.set(mod, "date", fd)
        p.set(sch, "_now_et", lambda: self.clock.read().astimezone(ET))
        self.time_proxy = _TimeProxy(self.clock)
        p.set(sch, "time", self.time_proxy)
        # Transports and side processes.
        transport = self.obs["transport"]
        p.set(sys.modules["bts.dm"], "send_dm",
              lambda recipient, text, *a, **k: (transport.append(["dm", recipient, text]), f"dm-{len(transport)}")[1])
        p.set(sys.modules["bts.posting"], "post_to_bluesky",
              lambda text, *a, **k: (transport.append(["post", text]), f"at://golden/{len(transport)}")[1])
        p.set(sch, "run_result_polling", lambda *a, **k: (transport.append(["result_polling"]), "final")[1])
        p.set(sch, "_trigger_live_forward_capture_on_lock", lambda *a, **k: transport.append(["live_forward"]))
        for k, v in self.env.items():
            self._env(k, v)
        # Spies (call the real function; record what it returned).
        strat = sys.modules["bts.strategy"]
        real_select = strat.select_pick

        def select_spy(predictions, *a, **k):
            sel = real_select(predictions, *a, **k)
            self.obs["selections"].append({"predictions": _predictions(predictions), "selection": _selection(sel)})
            return sel
        p.set(strat, "select_pick", select_spy)
        real_lock = sch._lock_decision_from_predictions

        def lock_spy(*a, **k):
            d = real_lock(*a, **k)
            self.obs["lock_decisions"].append(_jsonable(d))
            return d
        p.set(sch, "_lock_decision_from_predictions", lock_spy)
        real_plan = sch.plan_fallback_action

        def plan_spy(*a, **k):
            plan = real_plan(*a, **k)
            self.obs["fallback_plans"].append(_jsonable(plan))
            return plan
        p.set(sch, "plan_fallback_action", plan_spy)
        # The calibration resolver's inventory at each call (the files it will consume, in its order).
        import bts.model.calibrate as Cal
        real_resolve = Cal._resolve_pick_outcomes

        def resolve_spy(picks_dir, *a, **k):
            self.obs["resolver_inventory"].append(sorted(f.name for f in Path(picks_dir).glob("2*.json")))
            return real_resolve(picks_dir, *a, **k)
        p.set(Cal, "_resolve_pick_outcomes", resolve_spy)
        # stderr: the handled-failure lines (caught exceptions' messages) are compared; the candidate's own
        # non-fatal witness notices are excluded.
        failures = self.obs["stderr_failures"]
        real_err = sys.stderr

        class Tee(io.TextIOBase):
            def write(self, text):
                for line in str(text).splitlines():
                    if any(w in line for w in ("failed", "Failed", "REFUSED", "refused", "Error")) \
                            and "Serving witness" not in line:
                        failures.append(line)
                return real_err.write(text)

            def flush(self):
                return real_err.flush()
        p.set(sys, "stderr", Tee())
        # Reads of the history pick files, whatever the method (normal path: one read per file per fit).
        reads = self.obs["pick_reads"]
        for meth in ("read_bytes", "read_text"):
            real_meth = getattr(Path, meth)

            def counted(path, *a, _real=real_meth, **k):
                if _is_history_pick(path):
                    reads[path.name] = reads.get(path.name, 0) + 1
                return _real(path, *a, **k)
            p.set(Path, meth, counted)

    def _env(self, key, value):
        self._env_undo.append((key, os.environ.get(key, _MISSING)))
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    # -- running
    def run(self, fn):
        try:
            fn()
        except _DaemonKilled as e:
            self.obs["exception"] = {"type": "DaemonKilled", "message": str(e)}
        except Exception as e:                          # noqa: BLE001 - the exception is part of the surface
            self.obs["exception"] = {"type": f"{type(e).__module__}.{type(e).__qualname__}", "message": str(e)}

    def observe(self) -> dict:
        self.obs["files"] = _files(self.root, self.inputs)
        self.obs["cache_sha256"] = _cache(self.root)
        self.obs["slate"] = _slate(self.root)
        self.obs["unknown_urls"] = list(self.router.unknown)
        self.obs["api_calls"] = list(self.router.calls)
        text = json.dumps(self.obs, sort_keys=True)
        for root in sorted({str(self.root), os.path.realpath(self.root)}, key=len, reverse=True):
            text = text.replace(root, "<ROOT>")
        return json.loads(text)


# ---------------------------------------------------------------- drivers

def _run_day(h: Harness, delivery="private"):
    from bts.scheduler import run_day
    h.run(lambda: run_day(date=W.DATE, config=config(delivery)))


def _run_and_pick(h: Harness):
    from bts.orchestrator import run_and_pick
    h.run(lambda: run_and_pick(config(), W.DATE, require_detailed_statuses=False))


def _cached_blend_bytes() -> bytes:
    return pickle.dumps({**fakes.train_blend(None), "_model": fakes.train_model(None)})


def _write_cache(h: Harness, raw: bytes):
    path = h.root / "data" / "models" / f"blend_{W.DATE}.pkl"
    path.write_bytes(raw)
    h.inputs[path.relative_to(h.root).as_posix()] = None    # always report the cache's final state


# ---------------------------------------------------------------- injected faults

class _NoAppend(list):
    def append(self, x):
        raise RuntimeError("golden: injected collector failure")


def _read_bytes_fault(h: Harness, predicate, nth=None):
    real = Path.read_bytes
    seen = {"n": 0}

    def rb(self):
        if predicate(self):
            seen["n"] += 1
            if nth is None or seen["n"] == nth:
                raise MemoryError("golden: injected buffer failure")
        return real(self)
    h.p.set(Path, "read_bytes", rb)


def _sha_fault(h: Harness, predicate):
    real = hashlib.sha256

    def sha(*a, **k):
        if a and predicate(a[0]):
            raise MemoryError("golden: injected hash failure")
        return real(*a, **k)
    h.p.set(hashlib, "sha256", sha)


def _is_history_pick(p: Path) -> bool:
    return p.parent.name == "picks" and p.suffix == ".json" and p.name[:4] == "2026" and p.name != f"{W.DATE}.json"


def _fault_witness_names(h: Harness, which: str):
    """Faults inside the witness's own containment (candidate-only names; absent at the baseline)."""
    import bts.model.calibrate as C
    from bts.model import predict as P
    sw = sys.modules.get("bts.serving_witness")
    if which == "hashing_writer_construction":
        class Broken:
            def __init__(self, *a, **k):
                raise MemoryError("golden: injected writer construction failure")
        h.p.set_if_exists(P, "HashingWriter", Broken)
    elif which == "hashing_writer_hash":
        base = getattr(P, "HashingWriter", None)
        if base is not None:
            class BadUpdate(base):
                def __init__(self, f, errors=None):
                    super().__init__(f, errors)

                    class _H:
                        def update(self, b):
                            raise MemoryError("golden: injected hash update failure")
                    self._h = _H()
            h.p.set(P, "HashingWriter", BadUpdate)
    elif which == "hashing_writer_finalisation":
        base = getattr(P, "HashingWriter", None)
        if base is not None:
            class BadFinal(base):
                def __init__(self, f, errors=None):
                    super().__init__(f, errors)
                    real = self._h

                    class _H:
                        def update(self, b):
                            real.update(b)

                        def hexdigest(self):
                            raise MemoryError("golden: injected finalisation failure")
                    self._h = _H()
            h.p.set(P, "HashingWriter", BadFinal)
    elif which == "collector_appends":
        if sw is not None and hasattr(C, "_collect"):
            real = sw.collect
            h.p.set(C, "_collect", lambda items, item, errors, what: real(
                None if items is None else _NoAppend(), item, errors, what))
            h.p.set_if_exists(P, "collect", lambda items, item, errors, what: real(
                None if items is None else _NoAppend(), item, errors, what))
    elif which == "sample_canonicalisation":
        h.p.set_if_exists(C, "_canon_sha256", lambda o: (_ for _ in ()).throw(ValueError("golden: injected canon")))
    elif which == "map_extraction":
        if hasattr(C, "_fitted_map"):
            real = C._fitted_map

            class Unreadable:
                def __getattr__(self, name):
                    raise RuntimeError("golden: injected map extraction failure")
            h.p.set(C, "_fitted_map", lambda cal, errors: real(Unreadable(), errors))
    elif which == "error_recording":
        if sw is not None:
            real = sw.note
            h.p.set(sw, "note", lambda errors, msg: real(None if errors is None else _NoAppend(), msg))
            h.p.set_if_exists(C, "_note", lambda errors, msg: real(None if errors is None else _NoAppend(), msg))
            h.p.set_if_exists(P, "note", lambda errors, msg: real(None if errors is None else _NoAppend(), msg))
    elif which == "build":
        if sw is not None and hasattr(sw, "build"):
            h.p.set(sw, "build", lambda **k: (_ for _ in ()).throw(MemoryError("golden: injected build failure")))
    elif which == "attrs_assignment":
        class NoAttrs:
            @property
            def attrs(self):
                raise RuntimeError("golden: injected attrs failure")
        for mod, name in ((P, "_attach_pipeline_provenance"), (sys.modules["bts.orchestrator"], "_attach_serving_witness"),
                          (sys.modules["bts.orchestrator"], "_take_pipeline_provenance")):
            real = getattr(mod, name, None)
            if real is not None:
                h.p.set(mod, name, (lambda r: lambda frame, *a, **k: r(NoAttrs(), *a, **k))(real))
    else:
        raise ValueError(which)


# ---------------------------------------------------------------- scenarios

def sc_day_private(h):
    _run_day(h, "private")


def sc_day_dm(h):
    _run_day(h, "dm")


def sc_day_public(h):
    _run_day(h, "public")


def sc_day_all_posted(h):
    _run_day(h, "dm")


def sc_day_mdp_skip(h):
    _run_day(h, "dm")


def sc_day_prediction_failure(h):
    from bts.model import predict as P
    h.p.set(P, "predict", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("golden: prediction failure")))
    _run_day(h, "dm")


def sc_day_fallback_cached_pick(h):
    """The day runs as in day_dm until the second fallback refresh (21:35 ET), whose cascade fails: the refresh takes
    its "re-predict failed, using cached pick" branch and the planner acts on the cached pick."""
    from bts.model import predict as P
    real = P.predict
    fail_from = _real_datetime(2026, 6, 30, 21, 30, tzinfo=ET)

    def flaky(*a, **k):
        if h.clock.now >= fail_from:
            raise RuntimeError("golden: fallback-refresh prediction failure")
        return real(*a, **k)
    h.p.set(P, "predict", flaky)
    _run_day(h, "dm")


def sc_day_restart(h):
    """The daemon dies at its second sleep and restarts from the persisted state."""
    from bts.scheduler import run_day
    sch = sys.modules["bts.scheduler"]
    h.p.set(sch, "time", _TimeProxy(h.clock, crash_after_sleeps=1))
    h.run(lambda: run_day(date=W.DATE, config=config("dm")))
    first = dict(h.obs["exception"] or {})
    h.obs["exception"] = None
    h.p.set(sch, "time", _TimeProxy(h.clock))
    h.run(lambda: run_day(date=W.DATE, config=config("dm")))
    h.obs["first_run_exception"] = first


def sc_model_cached(h):
    _write_cache(h, _cached_blend_bytes())
    _run_and_pick(h)


def sc_model_cold(h):
    _run_and_pick(h)


def sc_model_empty_cached_dict(h):
    _write_cache(h, pickle.dumps({}))
    _run_and_pick(h)


def _calibration(h, value="1"):
    h._env("BTS_USE_CALIBRATION", value)


def sc_calibration_on(h):
    _calibration(h)
    _run_and_pick(h)


def sc_calibration_off_explicit(h):
    _calibration(h, "0")
    _run_and_pick(h)


def sc_calibration_no_pa_file(h):
    _calibration(h)
    (h.root / "data" / "processed" / "pa_2026.parquet").unlink()
    _run_and_pick(h)


def sc_calibration_insufficient_support(h):
    _calibration(h)
    picks = h.root / "data" / "picks"
    for f in sorted(picks.glob("2026-0*.json"))[:-12]:
        f.unlink()
    _run_and_pick(h)


def sc_calibration_no_sklearn(h):
    _calibration(h)
    real = builtins.__import__

    def imp(name, *a, **k):
        if name.startswith("sklearn"):
            raise ImportError(f"golden: no {name}")
        return real(name, *a, **k)
    h.p.set(builtins, "__import__", imp)
    _run_and_pick(h)


def sc_calibration_changed_history(h):
    """The changing-historical-probabilities counterexample: same dates and outcomes, different served p."""
    _calibration(h)
    for f in sorted((h.root / "data" / "picks").glob("2026-0*.json")):
        try:
            rec = json.loads(f.read_text())
        except ValueError:
            continue
        if isinstance(rec, dict) and isinstance(rec.get("pick"), dict):
            rec["pick"]["p_game_hit"] = round(1.5 - rec["pick"]["p_game_hit"], 4)
            f.write_text(json.dumps(rec))
    _run_and_pick(h)


def sc_calibration_two_thresholds(h):
    """30 selected samples with two served probabilities collapse to a two-threshold map."""
    _calibration(h)
    picks = h.root / "data" / "picks"
    for i, f in enumerate(sorted(picks.glob("2026-0*.json"))):
        try:
            rec = json.loads(f.read_text())
        except ValueError:
            continue
        if isinstance(rec, dict) and isinstance(rec.get("pick"), dict):
            rec["pick"]["p_game_hit"] = 0.65 if i % 2 else 0.85
            rec["double_down"] = None
            f.write_text(json.dumps(rec))
    _run_and_pick(h)


def sc_calibration_decode_error(h):
    _calibration(h)
    (h.root / "data" / "picks" / "2026-06-29x-bad.json").write_bytes(b"\xff\xfe not utf-8")
    h.inputs["data/picks/2026-06-29x-bad.json"] = hashlib.sha256(b"\xff\xfe not utf-8").hexdigest()
    _run_and_pick(h)


def sc_calibration_fit_failure(h):
    _calibration(h)
    from sklearn.isotonic import IsotonicRegression
    h.p.set(IsotonicRegression, "fit", lambda self, *a, **k: (_ for _ in ()).throw(ValueError("golden: fit")))
    _run_and_pick(h)


def sc_calibration_apply_failure(h):
    _calibration(h)
    import bts.model.calibrate as C
    h.p.set(C, "apply_calibrator_series", lambda s, c: (_ for _ in ()).throw(ValueError("golden: apply")))
    _run_and_pick(h)


def sc_calibration_error_after_assignment(h):
    _calibration(h)

    class Stderr(io.StringIO):
        def write(self, s):
            if "Applied calibration" in s:
                raise OSError("golden: stderr failure after assignment")
            return super().write(s)
    h.p.set(sys, "stderr", Stderr())
    _run_and_pick(h)


# Genuine computation faults.

def sc_genuine_parquet_parse(h):
    (h.root / "data" / "processed" / "pa_2025.parquet").write_bytes(b"PAR1 golden corrupt")
    h.inputs["data/processed/pa_2025.parquet"] = None
    _run_and_pick(h)


def sc_genuine_cache_unpickle(h):
    _write_cache(h, b"\x80\x04 golden corrupt")
    _run_and_pick(h)


def sc_genuine_cache_unpickle_stateful(h):
    _write_cache(h, _cached_blend_bytes())
    calls = []

    def failing(*a, **k):
        calls.append(1)
        raise pickle.UnpicklingError("golden: stateful unpickler")
    h.p.set(pickle, "loads", failing)
    h.p.set(pickle, "load", failing)
    _run_and_pick(h)
    h.obs["unpickle_calls"] = len(calls)


def sc_genuine_save_serialization(h):
    from bts.model import predict as P

    def unpicklable(df, **k):
        blend = fakes.train_blend(df, **k)
        blend["zz_unpicklable"] = (lambda: 0, [])
        return blend
    h.p.set(P, "train_blend", unpicklable)
    (h.root / "data" / "models" / f"blend_{W.DATE}.pkl").unlink(missing_ok=True)
    _run_and_pick(h)


class _FailingFile:
    def __init__(self, f, after):
        self._f, self._left = f, after

    def write(self, b):
        if self._left <= 0:
            raise OSError(28, "No space left on device (golden)")
        self._left -= 1
        return self._f.write(b)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._f.close()
        return False


def sc_genuine_partial_write(h):
    real_open = builtins.open
    target = f"blend_{W.DATE}.pkl"

    def opener(path, mode="r", *a, **k):
        f = real_open(path, mode, *a, **k)
        return _FailingFile(f, after=1) if "w" in mode and str(path).endswith(target) else f
    h.p.set(builtins, "open", opener)
    _run_and_pick(h)


# Injected ancillary faults (one per §3.0 boundary).

def sc_fault_parquet_buffer(h):
    _read_bytes_fault(h, lambda p: p.name.startswith("pa_") and p.suffix == ".parquet")
    _run_and_pick(h)


def sc_fault_parquet_hash(h):
    _sha_fault(h, lambda b: isinstance(b, (bytes, bytearray)) and bytes(b[:4]) == b"PAR1")
    _run_and_pick(h)


def sc_fault_cache_buffer(h):
    _write_cache(h, _cached_blend_bytes())
    _read_bytes_fault(h, lambda p: p.name == f"blend_{W.DATE}.pkl")
    _run_and_pick(h)


def sc_fault_cache_hash(h):
    _write_cache(h, _cached_blend_bytes())
    _sha_fault(h, lambda b: isinstance(b, (bytes, bytearray)) and bytes(b[:1]) == b"\x80")
    _run_and_pick(h)


def sc_fault_hashing_writer_construction(h):
    _fault_witness_names(h, "hashing_writer_construction")
    _run_and_pick(h)


def sc_fault_hashing_writer_hash(h):
    _fault_witness_names(h, "hashing_writer_hash")
    _run_and_pick(h)


def sc_fault_hashing_writer_finalisation(h):
    _fault_witness_names(h, "hashing_writer_finalisation")
    _run_and_pick(h)


def sc_fault_calibration_pa_buffer(h):
    _calibration(h)
    _read_bytes_fault(h, lambda p: p.name == "pa_2026.parquet", nth=2)
    _run_and_pick(h)


def sc_fault_pick_buffer(h):
    _calibration(h)
    _read_bytes_fault(h, _is_history_pick)
    _run_and_pick(h)


def sc_fault_pick_decoder(h):
    _calibration(h)
    import bts.model.calibrate as C

    class IoProxy:
        """calibrate's own `io` (candidate only): the decoder's construction fails; nothing else is touched."""
        def __getattr__(self, name):
            return getattr(io, name)

        @staticmethod
        def TextIOWrapper(*a, **k):
            raise MemoryError("golden: injected decoder preparation failure")
    h.p.set_if_exists(C, "io", IoProxy())
    _run_and_pick(h)


def sc_fault_collector_appends(h):
    _calibration(h)
    _fault_witness_names(h, "collector_appends")
    _run_and_pick(h)


def sc_fault_sample_canonicalisation(h):
    _calibration(h)
    _fault_witness_names(h, "sample_canonicalisation")
    _run_and_pick(h)


def sc_fault_map_extraction(h):
    _calibration(h)
    _fault_witness_names(h, "map_extraction")
    _run_and_pick(h)


def sc_fault_error_recording(h):
    _calibration(h)
    _sha_fault(h, lambda b: isinstance(b, (bytes, bytearray)) and bytes(b[:4]) == b"PAR1")
    _fault_witness_names(h, "error_recording")
    _run_and_pick(h)


def sc_fault_build(h):
    _fault_witness_names(h, "build")
    _run_and_pick(h)


def sc_fault_attrs_assignment(h):
    _fault_witness_names(h, "attrs_assignment")
    _run_and_pick(h)


# Code review r1 (F1-F5): the faults it reproduced, each at its own boundary.

class _BadRepr(MemoryError):
    def __repr__(self):
        raise MemoryError("golden: formatting the witness error")

    def __str__(self):
        raise MemoryError("golden: formatting the witness error")


def sc_fault_attrs_copy_calibration(h):
    """pandas copying attached provenance fails during calibration and slate-row extraction (r1 F1). The baseline
    attaches no such dicts, so the fault cannot fire there."""
    _calibration(h)
    g = pd.DataFrame.__finalize__.__globals__
    real = g["deepcopy"]

    def failing(obj, *a, **k):
        if isinstance(obj, dict) and ("serving_model" in obj or "serving" in obj):
            raise MemoryError("golden: copying attached provenance")
        return real(obj, *a, **k)
    h.p.set_dict_item(g, "deepcopy", failing)
    _run_and_pick(h)


def sc_fault_pick_decoder_oserror(h):
    """An OSError preparing the decoder after a successful read takes the fallback (r1 F2)."""
    _calibration(h)
    import bts.model.calibrate as C

    class IoProxy:
        def __getattr__(self, name):
            return getattr(io, name)

        @staticmethod
        def TextIOWrapper(*a, **k):
            raise OSError("golden: decoder preparation after a successful read")
    h.p.set_if_exists(C, "io", IoProxy())
    _run_and_pick(h)


def sc_fault_omitted_input_lost_error(h):
    """One consumed (out-of-window) pick input fails to collect and every error record is lost (r1 F3)."""
    _calibration(h)
    extra = h.root / "data" / "picks" / "2026-04-02.json"
    if not extra.exists():
        extra.write_text(json.dumps({"date": "2026-04-02", "result": "hit", "pick": {"batter_id": 10101,
                                                                                    "p_game_hit": 0.9}}))
    import bts.model.calibrate as C
    sw = sys.modules.get("bts.serving_witness")
    if sw is not None and hasattr(C, "_collect") and hasattr(sw, "collect"):
        real = sw.collect

        def collect(items, build, errors, what):
            if items is not None and what == "pick input":
                try:
                    rec = build()
                except Exception:
                    rec = None
                if rec and rec.get("file") == "2026-04-02.json":
                    return real(_NoAppend(), build, _NoAppend(), what)
            return real(items, build, errors, what)
        h.p.set(C, "_collect", collect)
        h.p.set(sw, "note", lambda *a, **k: False)
        h.p.set_if_exists(C, "_note", lambda *a, **k: False)
    _run_and_pick(h)


def sc_fault_undescribable_parquet(h):
    _read_bytes_fault_exc(h, lambda p: p.name == "pa_2025.parquet", _BadRepr)
    _run_and_pick(h)


def sc_fault_undescribable_pick(h):
    _calibration(h)
    _read_bytes_fault_exc(h, _is_history_pick, _BadRepr)
    _run_and_pick(h)


def sc_fault_undescribable_cache(h):
    _write_cache(h, _cached_blend_bytes())
    _read_bytes_fault_exc(h, lambda p: p.name == f"blend_{W.DATE}.pkl", _BadRepr)
    _run_and_pick(h)


def sc_fault_parquet_buffer_alloc(h):
    """The held buffer's construction fails after a successful read (r1 F6: distinct from the read itself)."""
    from bts.model import predict as P

    class IoProxy:
        def __getattr__(self, name):
            return getattr(io, name)

        @staticmethod
        def BytesIO(*a, **k):
            raise MemoryError("golden: buffer allocation after a successful read")
    h.p.set_if_exists(P, "io", IoProxy())
    _run_and_pick(h)


class _ShortFile:
    """A real file whose every write writes, and reports, only half its argument (a successful short write)."""
    def __init__(self, f):
        self._f = f

    def write(self, b):
        data = bytes(b)
        return self._f.write(data[: max(1, len(data) // 2)])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._f.close()
        return False


def sc_fault_short_write(h):
    real_open = builtins.open
    target = f"blend_{W.DATE}.pkl"

    def opener(path, mode="r", *a, **k):
        f = real_open(path, mode, *a, **k)
        return _ShortFile(f) if "w" in mode and str(path).endswith(target) else f
    h.p.set(builtins, "open", opener)
    _run_and_pick(h)


def sc_fault_package_query(h):
    sw = sys.modules.get("bts.serving_witness")
    if sw is not None and hasattr(sw, "version"):
        real = sw.version
        h.p.set(sw, "version", lambda name: (_ for _ in ()).throw(LookupError("golden: " + name))
                if name == "pyarrow" else real(name))
    _run_and_pick(h)


def sc_fault_map_hash(h):
    """The map's canonical hash fails while the samples' hash succeeds (r1 F6: independent of sample hashing)."""
    _calibration(h)
    import bts.model.calibrate as C
    if hasattr(C, "_canon_sha256"):
        real = C._canon_sha256
        h.p.set(C, "_canon_sha256", lambda o: (_ for _ in ()).throw(ValueError("golden: map canon"))
                if isinstance(o, dict) and "X_thresholds" in o else real(o))
    _run_and_pick(h)


def sc_fault_calibration_record(h):
    """Building the serving calibration record fails after the assignment (r1 F3: never a stale record)."""
    _calibration(h)
    orch = sys.modules["bts.orchestrator"]
    h.p.set_if_exists(orch, "_calibration_record",
                      lambda *a, **k: (_ for _ in ()).throw(MemoryError("golden: calibration record")))
    _run_and_pick(h)


class _AppendThenRaise:
    """A collector whose append lands and then raises: its length looks complete; only the reported failure says not."""
    def __init__(self, target):
        self.target = target

    def append(self, x):
        self.target.append(x)
        raise RuntimeError("golden: injected append-then-raise")


def _raise_at_line(func, pattern: str, exc: BaseException):
    """A sys.settrace hook raising `exc` once, at the one source line of `func` whose stripped text matches `pattern`
    (r2 R2-1: an allocation fault at a statement, not only at a helper's boundary)."""
    import inspect
    import re
    lines, start = inspect.getsourcelines(func)
    hits = [start + i for i, text in enumerate(lines) if re.fullmatch(pattern, text.strip())]
    if len(hits) != 1:
        raise RuntimeError(f"golden: fault anchor {pattern!r} matched {len(hits)} lines")
    code, line = func.__code__, hits[0]

    def local(frame, event, arg):
        if event == "line" and frame.f_lineno == line:
            raise exc
        return local
    return lambda frame, event, arg: local if frame.f_code is code else None


def sc_fault_provenance_allocation(h):
    """Allocating the provenance-removal helper's parts fails (r2 R2-1). The baseline has no such helper."""
    _calibration(h)
    helper = getattr(sys.modules["bts.orchestrator"], "_take_pipeline_provenance", None)
    if helper is None:
        _run_and_pick(h)
        return
    tracer = _raise_at_line(helper, r"parts(: dict)? = \{\}", MemoryError("golden: provenance dict allocation"))
    from bts.orchestrator import run_and_pick

    def go():
        sys.settrace(tracer)
        try:
            run_and_pick(config(), W.DATE, require_detailed_statuses=False)
        finally:
            sys.settrace(None)
    h.run(go)


def sc_fault_provenance_take(h):
    """Invoking the provenance-removal helper fails, and pandas copying any attached provenance would fail too (r2
    R2-1: the frame must be cleared before calibration, never calibrated with the provenance on it)."""
    _calibration(h)
    h.p.set_if_exists(sys.modules["bts.orchestrator"], "_take_pipeline_provenance",
                      lambda *a, **k: (_ for _ in ()).throw(MemoryError("golden: provenance take")))
    g = pd.DataFrame.__finalize__.__globals__
    real = g["deepcopy"]

    def failing(obj, *a, **k):
        if isinstance(obj, dict) and ("serving_model" in obj or "serving" in obj):
            raise MemoryError("golden: copying attached provenance")
        return real(obj, *a, **k)
    h.p.set_dict_item(g, "deepcopy", failing)
    _run_and_pick(h)


def sc_fault_pa_append_landed(h):
    """Every PA-input append lands and then raises, and its error record is lost (r2 R2-2)."""
    _calibration(h)
    from bts.model import predict as P
    sw = sys.modules.get("bts.serving_witness")
    if sw is not None and hasattr(P, "collect"):
        real = sw.collect

        def collect(items, build, errors, what):
            if items is not None and what == "PA input":
                return real(_AppendThenRaise(items), build, _NoAppend(), what)
            return real(items, build, errors, what)
        h.p.set(P, "collect", collect)
    _run_and_pick(h)


def _read_bytes_fault_exc(h: Harness, predicate, exc_type):
    real = Path.read_bytes

    def rb(self):
        if predicate(self):
            raise exc_type("golden: injected")
        return real(self)
    h.p.set(Path, "read_bytes", rb)


# Delivery at the cutoff (§5.2) and the advancing clock (§5.5).

def _cutoff_pick(h):
    """Select today's pick through the real path, then deliver it directly at a controlled clock."""
    from bts.orchestrator import run_and_pick
    from bts.picks import save_pick, submission_cutoff_et
    _, sel, _ = run_and_pick(config(), W.DATE, require_detailed_statuses=False)
    daily = sel.pick_result.daily
    save_pick(daily, Path("data/picks"))
    return daily, sel, submission_cutoff_et(daily)


def _deliver_at(h, offset: timedelta, step_on_read=None):
    import bts.scheduler as sch

    def go():
        daily, sel, cutoff = _cutoff_pick(h)
        h.clock.now = cutoff + offset
        h.clock.step_on_read = step_on_read
        state = sch.SchedulerState(date=W.DATE, schedule_fetched_at="golden", games=[], confirmed_game_pks=[],
                                   runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None,
                                   next_wakeup=None)
        ok = sch._deliver_and_lock_pick(daily, config("dm"), Path("data/picks"), state, W.DATE, "golden",
                                        selection=sel)
        h.obs["delivered"] = ok
        h.obs["cutoff_et"] = cutoff.isoformat()
    h.run(go)


def sc_cutoff_minus_one_second(h):
    _deliver_at(h, timedelta(seconds=-1))


def sc_cutoff_exact(h):
    _deliver_at(h, timedelta(0))


def sc_cutoff_advancing_clock(h):
    """The observation starts 1 s before the cutoff and every clock read advances 2 s, so the guard's read is past
    the cutoff: delivery must be refused."""
    _deliver_at(h, timedelta(seconds=-1), step_on_read=timedelta(seconds=2))


SCENARIOS = {
    # name: (function, harness options)
    "day_private": (sc_day_private, {}),
    "day_dm": (sc_day_dm, {}),
    "day_public": (sc_day_public, {}),
    "day_all_posted": (sc_day_all_posted, {"all_posted": True}),
    "day_mdp_skip": (sc_day_mdp_skip, {"streak": 9, "all_posted": True}),
    "day_prediction_failure": (sc_day_prediction_failure, {}),
    "day_fallback_cached_pick": (sc_day_fallback_cached_pick, {}),
    "day_restart": (sc_day_restart, {"all_posted": False}),
    "model_cached": (sc_model_cached, {}),
    "model_cold": (sc_model_cold, {}),
    "model_empty_cached_dict": (sc_model_empty_cached_dict, {}),
    "calibration_on": (sc_calibration_on, {}),
    "calibration_off_explicit": (sc_calibration_off_explicit, {}),
    "calibration_no_pa_file": (sc_calibration_no_pa_file, {}),
    "calibration_insufficient_support": (sc_calibration_insufficient_support, {}),
    "calibration_no_sklearn": (sc_calibration_no_sklearn, {}),
    "calibration_changed_history": (sc_calibration_changed_history, {}),
    "calibration_two_thresholds": (sc_calibration_two_thresholds, {}),
    "calibration_decode_error": (sc_calibration_decode_error, {}),
    "calibration_fit_failure": (sc_calibration_fit_failure, {}),
    "calibration_apply_failure": (sc_calibration_apply_failure, {}),
    "calibration_error_after_assignment": (sc_calibration_error_after_assignment, {}),
    "genuine_parquet_parse": (sc_genuine_parquet_parse, {}),
    "genuine_cache_unpickle": (sc_genuine_cache_unpickle, {}),
    "genuine_cache_unpickle_stateful": (sc_genuine_cache_unpickle_stateful, {}),
    "genuine_save_serialization": (sc_genuine_save_serialization, {}),
    "genuine_partial_write": (sc_genuine_partial_write, {}),
    "fault_parquet_buffer": (sc_fault_parquet_buffer, {}),
    "fault_parquet_hash": (sc_fault_parquet_hash, {}),
    "fault_cache_buffer": (sc_fault_cache_buffer, {}),
    "fault_cache_hash": (sc_fault_cache_hash, {}),
    "fault_hashing_writer_construction": (sc_fault_hashing_writer_construction, {}),
    "fault_hashing_writer_hash": (sc_fault_hashing_writer_hash, {}),
    "fault_hashing_writer_finalisation": (sc_fault_hashing_writer_finalisation, {}),
    "fault_calibration_pa_buffer": (sc_fault_calibration_pa_buffer, {}),
    "fault_pick_buffer": (sc_fault_pick_buffer, {}),
    "fault_pick_decoder": (sc_fault_pick_decoder, {}),
    "fault_collector_appends": (sc_fault_collector_appends, {}),
    "fault_sample_canonicalisation": (sc_fault_sample_canonicalisation, {}),
    "fault_map_extraction": (sc_fault_map_extraction, {}),
    "fault_error_recording": (sc_fault_error_recording, {}),
    "fault_build": (sc_fault_build, {}),
    "fault_attrs_assignment": (sc_fault_attrs_assignment, {}),
    "fault_attrs_copy_calibration": (sc_fault_attrs_copy_calibration, {}),
    "fault_pick_decoder_oserror": (sc_fault_pick_decoder_oserror, {}),
    "fault_omitted_input_lost_error": (sc_fault_omitted_input_lost_error, {}),
    "fault_undescribable_parquet": (sc_fault_undescribable_parquet, {}),
    "fault_undescribable_pick": (sc_fault_undescribable_pick, {}),
    "fault_undescribable_cache": (sc_fault_undescribable_cache, {}),
    "fault_parquet_buffer_alloc": (sc_fault_parquet_buffer_alloc, {}),
    "fault_short_write": (sc_fault_short_write, {}),
    "fault_package_query": (sc_fault_package_query, {}),
    "fault_map_hash": (sc_fault_map_hash, {}),
    "fault_calibration_record": (sc_fault_calibration_record, {}),
    "fault_provenance_allocation": (sc_fault_provenance_allocation, {}),
    "fault_provenance_take": (sc_fault_provenance_take, {}),
    "fault_pa_append_landed": (sc_fault_pa_append_landed, {}),
    "cutoff_minus_one_second": (sc_cutoff_minus_one_second, {"start_et": (12, 0)}),
    "cutoff_exact": (sc_cutoff_exact, {"start_et": (12, 0)}),
    "cutoff_advancing_clock": (sc_cutoff_advancing_clock, {"start_et": (12, 0)}),
}


def run(name: str, repo: Path, workdir: Path) -> dict:
    fn, opts = SCENARIOS[name]
    with Harness(repo, workdir, **opts) as h:
        fn(h)
    obs = h.observe()                      # after every patch is restored: the harness reads with the real I/O
    obs["scenario"] = name
    return obs
