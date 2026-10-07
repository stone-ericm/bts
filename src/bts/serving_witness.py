"""Serving provenance for each persisted slate (C2 step 2a; design `docs/superpowers/specs/2026-10-06-c2-2a-serving-record-design.md`).

C1 rank 4a's registration (`docs/sota_audit/2026-10-04-prereg-c1-calibration.md`, X-E1) scores served forecasts only
when each capture shows the recipe, model artifact and inputs that produced it, recorded in the serving process rather
than reconstructed later. The local tier builds this witness from what that process used, and `save_slate` stores it
in the slate envelope (`serving`):
- **recipe:** the sha256 of every `.py` file under `bts/model`, `bts/features` and `bts/data`, plus the source of the
  serving functions outside those packages (`RECIPE_FUNCTIONS`), and one fingerprint over both;
- **env:** the values of the recipe flags (`RECIPE_ENV`, an allowlist), and the names (never the values) of every
  other `BTS_*` variable;
- **packages:** the Python and numeric-stack versions;
- **model:** `{source: cache | trained | trained_unsaved, sha256}` from the branch `run_pipeline` actually took;
- **inputs:** each PA parquet's `{file, bytes, sha256}`, of the bytes that were parsed;
- **calibration:** the serving calibration record (design §3.2);
- **errors:** every provenance failure; any entry makes the witness incomplete provenance.

The amended witness is `bts_serving_witness_v2` (e66b440's v1 was never deployed; the model and calibration parts
changed shape).

**Limits:** the recipe hashes the files on disk when the witness is built, so the loaded code and the files can differ
only between a deploy's checkout and its restart. Env values are read when the witness is built; import-time
constants come from the same process environment. The witness does not retain the live schedule or lineup responses.

The containment helpers are shared by every capture point (design §3.0). Each contains its own failure: a provenance
problem is recorded in a local error list and never replaces, retries or alters the computation. `build` never
raises; a part that fails is null with its error.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

SCHEMA = "bts_serving_witness_v2"
RECIPE_PACKAGES = ("model", "features", "data")
RECIPE_FUNCTIONS = ("bts.orchestrator:predict_local", "bts.picks:is_resume_date_game",
                    "bts.util:is_regular_season_game")
RECIPE_ENV = ("BTS_LGBM_RANDOM_STATE", "BTS_LGBM_DETERMINISTIC", "BTS_USE_CALIBRATION", "BTS_ROOKIE_GATE_K",
              "BTS_PITCHER_HR_30G_MIN_PERIODS", "BTS_REFRESH_ALWAYS", "BTS_PARK_DRAG_TABLE")
PACKAGES = ("lightgbm", "numpy", "pandas", "pyarrow", "scikit-learn")


def _describe(what, exc=None) -> str:
    """A bounded description, built defensively: an exception whose str or repr fails still yields a message."""
    try:
        return (str(what) if exc is None else f"{what}: {type(exc).__name__}: {exc}")[:300]
    except Exception:
        try:
            return f"{what}: {type(exc).__name__} (undescribable)"[:300]
        except Exception:
            return "undescribable provenance error"


def note(errors, what, exc=None) -> bool:
    """Record a provenance error. The message is formatted inside the guard; never raises; returns whether recorded."""
    if errors is None:
        return False
    try:
        errors.append(_describe(what, exc))
        return True
    except Exception:
        return False


def collect(items, build, errors, what: str) -> bool:
    """Append `build()` to a provenance collector. The record is built inside the guard. Never raises; returns False
    when building or appending failed (the caller then treats that part as incomplete, whether or not the error itself
    could be recorded)."""
    if items is None:
        return True
    try:
        items.append(build())
        return True
    except Exception as e:
        try:
            note(errors, what + ": collector append failed", e)
        except Exception:
            pass
        return False


def sha256_or_none(raw, errors, what: str):
    """sha256 hex of the held bytes, or None with a recorded error."""
    try:
        return hashlib.sha256(raw).hexdigest()
    except Exception as e:
        try:
            note(errors, what + ": sha256 failed", e)
        except Exception:
            pass
        return None


def canon_sha256(obj) -> str:
    """sha256 of canonical JSON (sorted keys, no whitespace, UTF-8); a non-finite float raises ValueError."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _function_source(spec: str) -> str:
    mod, name = spec.split(":")
    return inspect.getsource(getattr(importlib.import_module(mod), name))


def recipe(root: Path | None = None) -> dict:
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    files = {}
    for pkg in RECIPE_PACKAGES:
        for p in sorted((root / pkg).rglob("*.py")):
            files[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    files = dict(sorted(files.items()))
    functions = {spec: hashlib.sha256(_function_source(spec).encode()).hexdigest() for spec in RECIPE_FUNCTIONS}
    body = json.dumps({"files": files, "functions": functions}, sort_keys=True, separators=(",", ":"))
    return {"files": files, "functions": functions, "sha256": hashlib.sha256(body.encode()).hexdigest()}


def _packages() -> dict:
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = version(name)
        except Exception:                     # a missing package is recorded as null, never raised
            out[name] = None
    return out


def build(*, model, inputs, calibration, errors=None) -> dict:
    """The serving witness for one local-tier forecast. Never raises; a failed part is null with its error.

    `errors` are the upstream provenance errors (run_pipeline's and predict_local's); they are copied.
    """
    errs: list = []
    try:
        errs.extend(errors or [])
    except Exception as e:
        note(errs, "upstream errors", e)
    parts = {}
    for key, fn in (("recipe", recipe), ("packages", _packages),
                    ("env", lambda: {k: os.environ.get(k) for k in RECIPE_ENV}),
                    ("env_names", lambda: sorted(k for k in os.environ if k.startswith("BTS_")))):
        try:
            parts[key] = fn()
        except Exception as e:
            parts[key] = None
            note(errs, key, e)
    try:
        for name, v in (parts["packages"] or {}).items():
            if v is None:
                note(errs, f"packages: {name} version unavailable")
    except Exception as e:
        note(errs, "packages", e)
    try:
        built_at = datetime.now(timezone.utc).isoformat()
    except Exception as e:
        built_at = None
        note(errs, "built_at", e)
    return {"schema": SCHEMA, "tier_type": "local", "recipe": parts["recipe"], "env": parts["env"],
            "env_names": parts["env_names"], "packages": parts["packages"], "model": model, "inputs": inputs,
            "calibration": calibration, "built_at": built_at, "errors": errs}



# ---------------------------------------------------------------- the run's witness (code review r4)
#
# Eric's row C2-2a-review-r4: completeness is earned, never assumed, and every witness statement on the pick path is a
# guarded hook. The pick path's deployed computation statements are unchanged; each hook is called from a one-statement
# `try: ... except Exception: pass` beside them. Each hook is total: a designed failure inside it (an unreadable file,
# a failed hash or append) is handled there and returns a safe value, so a caller's guard is reached only by an
# unforeseen fault, and a fault inside a hook's own handler stops at that guard. The hooks find the run's witness
# through a context variable that only `predict_local` sets, after its cache load (the one deployed statement there
# whose failure propagates) and resets before it returns; with none open, every hook returns at once with the deployed
# value (the path, the file, nothing), so every other caller of these functions (the CLI, the shadow tier,
# experiments, backtests) runs exactly as deployed and records nothing.

import contextvars as _contextvars
import functools as _functools
import io as _io
import types as _types

_CURRENT = _contextvars.ContextVar("bts_serving_witness", default=None)


def begin(witness):
    """Make `witness` the current local-tier run's witness; returns the token for `end`."""
    return _CURRENT.set(witness)


def end(token) -> None:
    _CURRENT.reset(token)


def current():
    """The current local-tier run's witness while it is open for recording, else None (every other caller, or a
    witness already sealed and attached to its predictions)."""
    w = _CURRENT.get()
    return None if w is None or w.sealed else w


class Ledger:
    """One consumed part's file records (PA parquets, pick files), each earned by confirmation.

    `hold(path, wrap, unreadable)` reads the file once and returns `wrap(raw, sha)`, the object the computation
    statement then consumes in place of the path, recording `{file, bytes, sha256}` for exactly those bytes. It never
    raises. If the read itself fails it returns `unreadable(path)` (default: the path, so the computation reads it as
    deployed); if preparing the held object fails it returns the path; either way it records nothing. If only the
    record fails, the held object is still returned (one read), and that file is left unconfirmed.
    `confirm(used)` runs after that statement. When `used` IS the object `hold` returned for the last record, that
    record is confirmed: it stands for bytes the computation really consumed. Otherwise the computation read the
    path, and the file is recorded as consumed with unknown bytes. A record counts only once confirmed, and the part
    is complete only when every record is confirmed and the records name exactly the files the computation consumed
    (`complete`). A failure can therefore only leave a record unconfirmed or a name missing: no combination of
    failures makes an incomplete part look complete (code review r3 R3-2)."""

    def __init__(self, errors, what: str):
        self.errors, self.what = errors, what
        self.records, self.confirmed = [], []
        self._held = None

    def hold(self, path, wrap, unreadable=None):
        try:
            raw = path.read_bytes()
        except OSError as e:
            note(self.errors, f"{self.what} {_name(path)}: unreadable", e)
            return path if unreadable is None else unreadable(path)
        except Exception as e:
            note(self.errors, f"{self.what} {_name(path)}: capture preparation failed; read from the path, not from "
                              "the hashed bytes", e)
            return path
        try:
            sha = sha256_or_none(raw, self.errors, self.what)
            used = wrap(raw, sha)
        except Exception as e:
            note(self.errors, f"{self.what} {_name(path)}: capture preparation failed; read from the path, not from "
                              "the hashed bytes", e)
            return path
        try:                                   # a lost record is not a lost buffer: the held bytes are still read once
            self.records.append({"file": path.name, "bytes": len(raw), "sha256": sha})
            self._held = (len(self.records), used)
        except Exception as e:
            note(self.errors, f"{self.what} {_name(path)}: record failed", e)
        return used

    def confirm(self, used) -> None:
        """Never raises (a hook is total): a failure here leaves the file unconfirmed, so the part is incomplete."""
        try:
            held, self._held = self._held, None
            if held is not None and held[1] is used and held[0] == len(self.records):
                self.confirmed.append(held[0])
            elif isinstance(used, os.PathLike):         # the computation read the path: consumed, bytes unknown
                self.records.append({"file": used.name, "bytes": None, "sha256": None})
                self.confirmed.append(len(self.records))
            else:                                       # a held object whose record did not stand
                note(self.errors, f"{self.what}: a consumed file's record did not stand")
        except Exception as e:
            note(self.errors, f"{self.what}: confirmation failed", e)

    def complete(self, names) -> bool:
        """Every record confirmed, in order, naming exactly `names` (what the computation consumed)."""
        return (self.confirmed == list(range(1, len(self.records) + 1))
                and [r["file"] for r in self.records] == list(names))


def _name(path) -> str:
    try:
        return str(path.name)
    except Exception:
        return "(unnamed)"


def _pick_text(path):
    def wrap(raw, sha):
        """A pick file's stand-in for `json.loads(f.read_text())`. Its `read_text` is the C-implemented `read` of a text
        reader over exactly the held bytes, with `Path.read_text()`'s semantics (default text encoding, strict errors,
        universal newlines): no witness code runs inside the computation statement, the text is decoded there, once,
        and a decoding error raises there exactly as deployed (one read)."""
        reader = _io.TextIOWrapper(_io.BytesIO(raw), encoding=_io.text_encoding(None))
        return _types.SimpleNamespace(read_text=reader.read, name=path.name, sha256=sha)
    return wrap


def _unreadable_pick(path):
    """An unreadable pick file's stand-in: `read_text` is a C-level callable raising OSError (os.read on an invalid
    descriptor), so the deployed `except (json.JSONDecodeError, OSError): continue` skips the file exactly as it
    would have, with no second read (r1 decision 2)."""
    import os as _os
    return _types.SimpleNamespace(read_text=_functools.partial(_os.read, -1, 0), name=path.name, sha256=None)


def names(directory, pattern: str, count: int | None = None) -> list:
    """The files a computation consumed from `directory`, listed again after it: the earned completeness checks compare
    the ledger's confirmed records with this independent listing (a change in between makes the part incomplete)."""
    found = [p.name for p in sorted(Path(directory).glob(pattern))]
    if count is not None and len(found) != count:
        raise LookupError(f"{pattern}: {len(found)} files now, {count} consumed")
    return found


class _NullLedger:
    """run_pipeline's ledger when no witness is open: the deployed path, and nothing recorded."""

    def hold(self, path, wrap, unreadable=None):
        return path

    def confirm(self, used) -> None:
        return None


_NULL_LEDGER = _NullLedger()


class HashingWriter:
    """A write-only file wrapper for the cache's `pickle.dump` (design §3.1). Each write is forwarded unchanged, once,
    and its result returned; the sha256 of what was written is updated beside it. The digest is earned: it stands only
    when the bytes hashed equal the bytes the file reports written (`f.tell()`), so a failed update or a short write
    withholds it (r1 F5). `write` is called from inside pickle (C) with no caller guard, so its update has a guard
    and that guard a guard of its own; only the forwarded write (`return self._f.write(b)`) is computation."""

    def __init__(self, f, h):
        self._f, self._h, self.hashed = f, h, 0

    def write(self, b):
        try:
            try:
                self._update(b)
            except Exception:
                pass
        except Exception:
            pass
        return self._f.write(b)

    def _update(self, b):
        self._h.update(b)
        self.hashed += memoryview(b).nbytes

    def digest(self) -> str | None:
        return self._h.hexdigest() if self.hashed == self._f.tell() else None


class Serving:
    """One local-tier forecast's witness, recorded by the hooks while the run happens."""

    def __init__(self):
        self.sealed = False
        self.errors: list = []
        self.pipeline_inputs = None                         # the PA ledger, opened once by run_pipeline
        self.pipeline_names = None
        self.model = None                                   # {source, sha256}
        self.cache = None                                   # (sha256, the loader handed to the computation)
        self.cache_used = False
        self.save_digest = None
        self.calibration = Calibration()

    def record(self) -> dict:
        """The witness to persist. Every part is earned; a part whose facts are missing is null with an error."""
        inputs = None
        try:
            led = self.pipeline_inputs
            if led is not None and self.pipeline_names is not None and led.complete(self.pipeline_names):
                inputs = led.records
            else:
                note(self.errors, "PA inputs incomplete; withheld")
        except Exception as e:
            note(self.errors, "PA inputs", e)
        calibration = None
        try:
            calibration = self.calibration.record()
        except Exception as e:
            note(self.errors, "calibration record", e)
        return build(model=self.model, inputs=inputs, calibration=calibration, errors=self.errors)


class Calibration:
    """The serving calibration record's facts (design §3.2), each set by a hook only when it happened."""

    def __init__(self):
        self.errors: list = []
        self.enabled = None
        self.applied = False
        self.failure = None
        self.outcome = None
        self.pa = Ledger(self.errors, "calibration PA input")
        self.pa_name = None
        self.fit = None
        self.picks = Ledger(self.errors, "pick input")
        self.pick_names = None
        self.bindings: list = []
        self.bound: list = []                               # each binding's confirmation, after its append returned

    def record(self) -> dict | None:
        if self.enabled is None:
            note(self.errors, "calibration: whether it was enabled is unknown; record withheld")
            return None
        fit = self.fit if isinstance(self.fit, dict) else {}
        if not self.enabled:
            status = "off"
        elif self.applied:
            status = "applied"
        elif self.failure is not None:
            status = "failed"
        elif self.outcome == "no_pa_file":
            status = "no_pa_file"
        elif fit.get("status") in ("insufficient_support", "no_sklearn"):
            status = fit["status"]
        else:
            status = None
        if self.failure is not None:
            note(self.errors, "calibration (after the probability assignment)" if self.applied else "calibration",
                 self.failure)
        applied = True if self.applied else (False if status is not None else None)
        pa_input = None
        if self.pa_name is not None and self.pa.complete([self.pa_name]):
            pa_input = self.pa.records[0]
        elif self.enabled and self.pa_name is not None:
            note(self.errors, "calibration PA input incomplete; withheld")
        self.errors.extend(fit.get("errors") or [])
        return {"enabled": self.enabled, "applied": applied, "status": status, "pa_input": pa_input,
                "pick_inputs": fit.get("pick_inputs"), "n_fit": fit.get("n_fit"), "samples": fit.get("samples"),
                "samples_sha256": fit.get("samples_sha256"), "map": fit.get("map"),
                "map_sha256": fit.get("map_sha256"), "errors": self.errors}


# ---------------------------------------------------------------- the hooks (each total; callers guard every call)

def run_end(predictions, witness, token) -> None:
    """predict_local's last hook: seal the witness, attach its record as `attrs["serving"]` and stop recording."""
    try:
        witness.sealed = True
        if predictions is not None:
            try:
                predictions.attrs["serving"] = witness.record()
            except Exception as e:
                try:
                    print(f"  [local] Serving witness not attached (non-fatal): {type(e).__name__}", file=_stderr())
                except Exception:
                    pass
    finally:
        end(token)


def _stderr():
    import sys as _sys
    return _sys.stderr


def cache_loader(path, deployed, w):
    """For `cached_blend = load_blend(cache_path)`: a loader that unpickles the cache's held, hashed bytes, so the
    bytes hashed are the bytes unpickled (one read). Returns `deployed` (the deployed load_blend, which reads the
    path) when the bytes cannot be held. Records into predict_local's witness `w`, which is not yet current: the
    load's genuine failure propagates as deployed, so the witness becomes current only after it."""
    if w is None:
        return deployed
    try:
        raw = path.read_bytes()
    except Exception as e:
        note(w.errors, "cache: capture preparation failed; loaded from the path, not from the hashed bytes", e)
        return deployed
    try:
        sha = sha256_or_none(raw, w.errors, "cache")
        loader = _held_unpickler(raw)
        w.cache = (sha, loader)
        return loader
    except Exception as e:
        note(w.errors, "cache: capture preparation failed; loaded from the path, not from the hashed bytes", e)
        return deployed


def _held_unpickler(raw):
    import pickle  # noqa: S403 — loading our own cached models

    def load_blend(path):
        return pickle.loads(raw)  # noqa: S301 — the deployed load_blend's operation, on the held bytes
    return load_blend


def cache_used(used, w) -> None:
    """After `cached_blend = load_blend(cache_path)` completed: the cache's sha256 stands only when the loader the
    computation used IS the held one."""
    if w is None:
        return
    if w.cache is not None and w.cache[1] is used:
        w.cache_used = True
    else:
        note(w.errors, "cache: loaded from the path, not from the hashed bytes")


def pa_open():
    """run_pipeline, before its parquet loop: this run's PA ledger. One per witness; with no open witness, or for a
    second run_pipeline under one witness, a ledger that records nothing."""
    w = current()
    if w is None or w.pipeline_inputs is not None:
        return _NULL_LEDGER
    w.pipeline_inputs = Ledger(w.errors, "PA input")
    return w.pipeline_inputs


def pa_names(directory, count: int, ledger) -> None:
    """run_pipeline, after its parquet loop: the PA files it consumed, listed independently, for this run's ledger."""
    w = current()
    if w is not None and w.pipeline_inputs is ledger:
        try:
            w.pipeline_names = names(directory, "pa_*.parquet", count)
        except Exception as e:
            note(w.errors, "PA inputs: the consumed files could not be listed", e)


def parquet_buffer(raw, sha):
    """A PA parquet for the computation to parse: a buffer over exactly the hashed bytes."""
    return _io.BytesIO(raw)


def model(source: str) -> None:
    """run_pipeline: which branch produced the model, with the sha256 that branch earned."""
    w = current()
    if w is None:
        return
    if source == "cache":
        w.model = {"source": "cache", "sha256": w.cache[0] if w.cache_used and w.cache is not None else None}
    elif source == "trained":
        w.model = {"source": "trained", "sha256": w.save_digest}
    else:
        w.model = {"source": "trained_unsaved", "sha256": None}


def hashing_writer(f):
    """save_blend, inside `with open(path, "wb") as f:`: the object to dump into. The file itself when no witness is
    open, so every other caller writes as deployed."""
    if current() is None:
        return f
    try:
        return HashingWriter(f, hashlib.sha256())
    except Exception:
        return f


def saved(f) -> None:
    """save_blend, after `pickle.dump(blend, f)`: the digest of what was written, when earned (r1 F5)."""
    w = current()
    if w is None:
        return
    try:
        w.save_digest = f.digest() if isinstance(f, HashingWriter) else None
    except Exception as e:
        w.save_digest = None
        note(w.errors, "blend save: sha256 finalization failed", e)
        return
    if w.save_digest is None:
        note(w.errors, "blend save: the bytes hashed are not the bytes the file reports written; digest withheld")


def calibration_enabled(flag) -> None:
    w = current()
    if w is not None:
        w.calibration.enabled = flag


def calibration_pa(path):
    """predict_local, before `pa_df = pd.read_parquet(current_pa)`: the held buffer to parse (or the path)."""
    w = current()
    if w is None:
        return path
    w.calibration.pa_name = path.name
    return w.calibration.pa.hold(path, parquet_buffer)


def calibration_pa_used(used) -> None:
    w = current()
    if w is not None:
        w.calibration.pa.confirm(used)


def calibration_applied() -> None:
    w = current()
    if w is not None:
        w.calibration.applied = True


def calibration_outcome(word: str) -> None:
    w = current()
    if w is not None:
        w.calibration.outcome = word


def calibration_failed(e) -> None:
    w = current()
    if w is not None:
        w.calibration.failure = e


def pick_file(path):
    """_resolve_pick_outcomes, before `data = json.loads(f.read_text())`: the held text's stand-in (or the path)."""
    w = current()
    if w is None:
        return path
    return w.calibration.picks.hold(path, _pick_text(path), _unreadable_pick)


def pick_read(f) -> None:
    """After the pick file was read and parsed."""
    w = current()
    if w is not None:
        w.calibration.picks.confirm(f)


def pick_skipped(f) -> None:
    """In the deployed handler that skips a pick file: it was consumed when its text was read (a JSON error), not when
    the read failed (an OSError)."""
    import sys as _sys
    w = current()
    if w is not None and isinstance(_sys.exc_info()[1], json.JSONDecodeError):
        w.calibration.picks.confirm(f)


def pick_names(directory, read: bool = True) -> None:
    """At the resolver's end: the pick files it consumed, listed independently (none when it read none)."""
    w = current()
    if w is not None:
        try:
            w.calibration.pick_names = names(directory, "2*.json") if read else []
        except Exception as e:
            note(w.calibration.errors, "pick inputs: the consumed files could not be listed", e)


def pick_bind(f, pick_date, slot_key: str, bid, sample) -> None:
    """After `samples.append(...)`: the sample's binding, carrying the exact appended values."""
    w = current()
    if w is None:
        return
    c = w.calibration
    try:
        c.bindings.append({"file": f.name, "file_sha256": getattr(f, "sha256", None), "date": pick_date.isoformat(),
                           "slot": slot_key, "batter_id": bid, "p": sample[0], "y": sample[1]})
        c.bound.append(len(c.bindings))                   # earned: only an append that returned is confirmed
    except Exception as e:
        note(c.errors, "sample binding failed", e)


def fit(status: str, samples=None, cal=None) -> None:
    """fit_calibrator_from_picks, before each return: the fit's witness, from local facts. Pick inputs and sample
    bindings stand only when earned: every pick file confirmed against an independent listing, and every binding's
    append confirmed with the bindings' values exactly the samples'."""
    w = current()
    if w is None:
        return
    c = w.calibration
    rec = {"status": status, "n_fit": None, "pick_inputs": None, "samples": None, "samples_sha256": None,
           "map": None, "map_sha256": None, "errors": []}
    if samples is not None:
        rec["n_fit"] = len(samples)
        if c.pick_names is not None and c.picks.complete(c.pick_names):
            rec["pick_inputs"] = c.picks.records
        else:
            note(rec["errors"], "calibration pick inputs incomplete; withheld")
        if (c.bound == list(range(1, len(c.bindings) + 1))
                and [(b["p"], b["y"]) for b in c.bindings] == [(s[0], s[1]) for s in samples]):
            rec["samples"] = c.bindings
            rec["samples_sha256"] = _canon_or_none(c.bindings, rec["errors"], "calibration samples")
        else:
            note(rec["errors"], "calibration sample bindings incomplete; withheld")
    if cal is not None:
        rec["map"] = _fitted_map(cal, rec["errors"])
        rec["map_sha256"] = None if rec["map"] is None else _canon_or_none(rec["map"], rec["errors"],
                                                                            "calibration map")
    c.fit = rec


def _canon_or_none(obj, errors, what: str):
    try:
        return canon_sha256(obj)
    except Exception as e:
        note(errors, what + ": canonical sha256 failed", e)
        return None


def _fitted_map(cal, errors):
    try:
        return {"X_thresholds": cal.X_thresholds_.tolist(), "y_thresholds": cal.y_thresholds_.tolist(),
                "increasing": bool(cal.increasing_), "out_of_bounds": cal.out_of_bounds,
                "y_min": cal.y_min, "y_max": cal.y_max}
    except Exception as e:
        note(errors, "calibration map: extraction failed", e)
        return None
