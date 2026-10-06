"""C1 rank 4a: the serving contract and each capture's serving witness (X-E1; review r1 R2).

X-E1 requires the freeze to name the old serving recipe, the model-training schedule, the active blend, the
aggregation and fallback definitions, and the configuration (calibration, determinism and seed flags). Each immutable
capture must carry its own model, input and recipe hashes. A missing pin or witness refuses acceptance. An
unregistered recipe change makes the study inconclusive, with no refit, window reset or pooling.

**The serving contract** is frozen before the first 2027 capture (with the calendar, under checklist A3), outside the
code closure, and pinned by `admission.json` `input_pins.serving_contract`. It is outcome-free:

    {"schema": "c1_r4a_serving_contract_v1", "season": 2027, "witness_schema": "bts_serving_witness_v1",
     "tier": {"name": <the production tier's name>, "type": "local"},
     "recipe_sha256": <the deployed serving code's recipe fingerprint>,
     "env": {<every recipe flag>: <its value or null>}, "packages": {<name>: <version>},
     "calibration_enabled": <bool>, "schedule": <text>}

- **What it pins:**
  - the recipe: `bts.serving_witness.recipe`, a fingerprint of the model, features and data packages and of the
    serving functions, so it covers the blend, the aggregation and the per-date model cache;
  - the flags and versions;
  - the tier, which is the fallback definition: only the registered local tier serves.
- **The schedule** (daily retraining, cached per date) is checked through each capture's model file name.

**Each slate's witness** (`bts_slate_v2` envelope `serving`, written by the serving process) is checked:
- **`missing` (refuses acceptance):**
  - no witness, or a different schema;
  - a recipe whose fingerprint does not recompute from its own file and function hashes;
  - a model without a source (`cache` / `trained`) and a 64-hex sha256;
  - no inputs (each a `pa_<season>.parquet` with a size and a sha256);
  - no calibration state, or no env or package record.
- **`changed` (an evidenced unregistered change; the study is inconclusive):** another tier, recipe fingerprint, flag
  value, package version or calibration setting; or a model file other than that date's daily blend
  (`blend_<date>.pkl`).
- **`ok`:** otherwise.

The checks never read an outcome.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date

SCHEMA = "c1_r4a_serving_contract_v1"
WITNESS = "bts_serving_witness_v1"
SEASON = 2027
HEX64 = re.compile(r"[0-9a-f]{64}")
PARQUET = re.compile(r"pa_20[0-9]{2}\.parquet")


class ContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class Contract:
    obj: dict
    sha256: str


def _hex(v) -> bool:
    return isinstance(v, str) and HEX64.fullmatch(v) is not None


def _str_map(v, *, nullable: bool) -> bool:
    return isinstance(v, dict) and all(isinstance(k, str) and (isinstance(x, str) or (nullable and x is None))
                                       for k, x in v.items())


def load(raw: bytes, pin: str) -> Contract:
    sha = hashlib.sha256(raw).hexdigest()
    if sha != pin:
        raise ContractError(f"the serving contract sha256 {sha[:12]} is not its pin {str(pin)[:12]}")
    try:
        obj = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise ContractError(f"unreadable serving contract: {exc}") from None
    ok = (isinstance(obj, dict) and obj.get("schema") == SCHEMA and type(obj.get("season")) is int
          and obj["season"] == SEASON and obj.get("witness_schema") == WITNESS
          and isinstance(obj.get("tier"), dict) and isinstance(obj["tier"].get("name"), str)
          and obj["tier"]["name"] and obj["tier"].get("type") == "local"
          and _hex(obj.get("recipe_sha256")) and _str_map(obj.get("env"), nullable=True) and obj["env"]
          and _str_map(obj.get("packages"), nullable=True) and obj["packages"]
          and type(obj.get("calibration_enabled")) is bool
          and isinstance(obj.get("schedule"), str) and obj["schedule"])
    if not ok:
        raise ContractError(f"not a {SCHEMA} object for season {SEASON}")
    return Contract(obj, sha)


def recipe_fingerprint(files: dict, functions: dict) -> str:
    """= bts.serving_witness.recipe's fingerprint, restated so the reader pins it (a test checks they agree)."""
    body = json.dumps({"files": files, "functions": functions}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def _missing(w) -> list[str]:
    if not isinstance(w, dict) or w.get("schema") != WITNESS:
        return ["no bts_serving_witness_v1 witness"]
    out = []
    rec = w.get("recipe")
    if not (isinstance(rec, dict) and _str_map(rec.get("files"), nullable=False) and rec["files"]
            and all(_hex(v) for v in rec["files"].values()) and _str_map(rec.get("functions"), nullable=False)
            and rec["functions"] and all(_hex(v) for v in rec["functions"].values()) and _hex(rec.get("sha256"))
            and rec["sha256"] == recipe_fingerprint(rec["files"], rec["functions"])):
        out.append("the recipe witness is missing or does not recompute")
    model = w.get("model")
    if not (isinstance(model, dict) and model.get("source") in ("cache", "trained")
            and isinstance(model.get("file"), str) and _hex(model.get("sha256"))):
        out.append("the model witness is missing (source, file, sha256)")
    inputs = w.get("inputs")
    if not (isinstance(inputs, list) and inputs and all(
            isinstance(i, dict) and isinstance(i.get("file"), str) and PARQUET.fullmatch(i["file"])
            and type(i.get("bytes")) is int and i["bytes"] >= 0 and _hex(i.get("sha256")) for i in inputs)):
        out.append("the input witness is missing")
    cal = w.get("calibration")
    if not (isinstance(cal, dict) and type(cal.get("enabled")) is bool and type(cal.get("applied")) is bool):
        out.append("the calibration witness is missing")
    if not _str_map(w.get("env"), nullable=True) or not _str_map(w.get("packages"), nullable=True):
        out.append("the env or package witness is missing")
    return out


def check(envelope: dict, d: date, contract: Contract) -> tuple[str, list[str]]:
    """('ok' | 'missing' | 'changed', reasons) for one slate envelope (the slate object without its rows)."""
    w = envelope.get("serving") if isinstance(envelope, dict) else None
    missing = _missing(w)
    if missing:
        return "missing", missing
    c = contract.obj
    changed = []
    if envelope.get("tier") != c["tier"]["name"] or w.get("tier_type") != c["tier"]["type"]:
        changed.append(f"tier {envelope.get('tier')!r}/{w.get('tier_type')!r} is not the registered one")
    if w["recipe"]["sha256"] != c["recipe_sha256"]:
        changed.append("the recipe fingerprint is not the registered one")
    if w["env"] != c["env"]:
        changed.append("the recipe flags are not the registered ones")
    if w["packages"] != c["packages"]:
        changed.append("the package versions are not the registered ones")
    if w["calibration"]["enabled"] != c["calibration_enabled"]:
        changed.append("the calibration setting is not the registered one")
    if w["model"]["file"] != f"blend_{d.isoformat()}.pkl":
        changed.append(f"the model {w['model']['file']!r} is not the date's daily blend (the registered schedule)")
    return ("changed", changed) if changed else ("ok", [])
