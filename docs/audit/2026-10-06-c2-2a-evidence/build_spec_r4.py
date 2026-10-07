"""C2 step 2a ledger spec revision 4 (Eric's row C2-2a-review-r4)."""
import json, os, re, subprocess, sys
from pathlib import Path
EV = Path("docs/audit/2026-10-06-c2-2a-evidence")
SW, PR, CA, OR, SL = ("src/bts/serving_witness.py", "src/bts/model/predict.py", "src/bts/model/calibrate.py",
                      "src/bts/orchestrator.py", "src/bts/slate.py")
T = "tests/c2_2a/"
core, pw, cw, lw = T + "test_witness_core.py::", T + "test_predict_witness.py::", T + "test_calibrate_witness.py::", \
    T + "test_local_witness.py::"
r1, r2 = T + "test_r1_counterexamples.py::", T + "test_r2_counterexamples.py::"
gold = lambda *names: [T + "test_golden.py", "-k", " or ".join(names)]   # noqa: E731
KEEP = ("W1", "W3", "W4", "W5", "W6", "W7", "W8", "W9", "W10", "W11", "W12", "P12", "B1", "S1", "S2", "D1", "D2", "D3",
        "O14", "O16")
old = json.loads(subprocess.run(["git", "show", "e36e038:" + str(EV / "mutants.json")], capture_output=True, text=True,
                                check=True).stdout)          # the revision-3 spec, as committed
assert len(old) == 91
kept = [m for m in old if m["id"] in KEEP]
retired = [dict(m, retired_r4="the code it mutated was replaced by the r4 class repair" if m["id"] not in ("W2", "W2b")
                else "collect() was removed in r4 (no caller after the class repair)") for m in old if m["id"] not in KEEP]
N = []
def q(i, rule, f, a, b, tests):
    N.append({"id": i, "rule": rule, "file": f, "old": a, "new": b, "tests": tests, "added_r4": True})
# the witness core
q("Q1", "current: a sealed witness records nothing", SW, "    return None if w is None or w.sealed else w", "    return w",
  [core + "test_a_sealed_witness_records_nothing"])
q("Q2", "hold: an unreadable file takes the caller's stand-in (no second read)", SW,
  "            return path if unreadable is None else unreadable(path)", "            return path",
  [core + "test_only_an_unreadable_file_takes_the_unreadable_stand_in",
   cw + "test_a_genuine_read_error_skips_the_file_without_an_added_retry"])
q("Q3", "hold: a capture-preparation failure parses the path", SW,
  '            used = wrap(raw, sha)\n        except Exception as e:\n            note(self.errors, f"{self.what} {_name(path)}: '
  'capture preparation failed; read from the path, not from "\n                              "the hashed bytes", e)\n'
  '            return path', '            used = wrap(raw, sha)\n        except Exception as e:\n            note(self.errors, '
  'f"{self.what} {_name(path)}: capture preparation failed; read from the path, not from "\n                              '
  '"the hashed bytes", e)\n            return None',
  [r1 + "test_a_decoder_preparation_oserror_takes_the_fallback"])
q("Q4", "hold: the held object is bound to its record", SW, "            self._held = (len(self.records), used)",
  "            self._held = None", [pw + "test_parquet_is_parsed_once_from_the_hashed_buffer"])
q("Q5", "hold: a lost record still hands back the held bytes (one read)", SW,
  '            note(self.errors, f"{self.what} {_name(path)}: record failed", e)\n        return used',
  '            note(self.errors, f"{self.what} {_name(path)}: record failed", e)\n        return path',
  [core + "test_a_lost_record_still_returns_the_held_object",
   pw + "test_a_lost_record_still_parses_the_held_buffer_and_withholds_the_inputs"])
q("Q6", "confirm: only the held object itself confirms", SW,
  "            if held is not None and held[1] is used and held[0] == len(self.records):",
  "            if held is not None and held[0] == len(self.records):",
  [core + "test_a_confirmation_stands_only_for_the_held_object"])
q("Q7", "confirm: only for the last record", SW,
  "            if held is not None and held[1] is used and held[0] == len(self.records):",
  "            if held is not None and held[1] is used:", [core + "test_a_confirmation_stands_only_for_the_last_record"])
q("Q8", "confirm: consumed-with-unknown-bytes only for a path the computation read", SW,
  "            elif isinstance(used, os.PathLike):", "            elif True:",
  [core + "test_a_held_pick_whose_record_did_not_stand_is_not_counted"])
q("Q9", "complete: every record confirmed, in order", SW,
  "        return (self.confirmed == list(range(1, len(self.records) + 1))\n                and ",
  "        return (True\n                and ", [cw + "test_a_lost_confirmation_withholds_the_pick_inputs"])
q("Q10", "complete: the records name exactly the files consumed", SW,
  '                and [r["file"] for r in self.records] == list(names))', "                and True)",
  [cw + "test_a_changed_pick_listing_withholds_the_pick_inputs",
   pw + "test_a_changed_parquet_listing_withholds_the_inputs"])
q("Q11", "names: a count mismatch is refused", SW, "    if count is not None and len(found) != count:", "    if False:",
  [core + "test_the_listing_refuses_a_count_mismatch"])
q("Q12", "digest: earned only when bytes hashed equal bytes written", SW,
  "        return self._h.hexdigest() if self.hashed == self._f.tell() else None", "        return self._h.hexdigest()",
  [r1 + "test_a_short_successful_write_withholds_the_digest",
   pw + "test_a_hash_update_failure_keeps_the_bytes_and_withholds_the_digest"])
q("Q13", "digest: each update counts its bytes", SW, "        self.hashed += memoryview(b).nbytes", "        pass",
  [r1 + "test_a_full_write_keeps_the_digest"])
q("Q14", "record: PA inputs only when complete", SW,
  "            if led is not None and self.pipeline_names is not None and led.complete(self.pipeline_names):",
  "            if led is not None:", [pw + "test_a_changed_parquet_listing_withholds_the_inputs"])
q("Q15", "calibration record: withheld when enabled is unknown", SW, "        if self.enabled is None:", "        if False:",
  [core + "test_an_unknown_enabled_flag_withholds_the_calibration_record"])
q("Q16", "calibration record: an unknown status is not 'not applied'", SW,
  "        applied = True if self.applied else (False if status is not None else None)",
  "        applied = True if self.applied else False", [core + "test_an_unknown_status_is_never_published_as_not_applied"])
q("Q17", "calibration record: the PA input only when complete", SW,
  "        if self.pa_name is not None and self.pa.complete([self.pa_name]):",
  "        if self.pa_name is not None and self.pa.records:",
  [r2 + "test_a_pa_input_that_raised_after_appending_is_withheld[3-pa_input]"])
q("Q18", "run_end: seals the witness", SW, "        witness.sealed = True", "        pass",
  [core + "test_run_end_seals_attaches_and_stops_recording"])
q("Q19", "run_end: stops recording (resets the context)", SW, "    finally:\n        end(token)",
  "    finally:\n        pass", [core + "test_run_end_seals_attaches_and_stops_recording"])
q("Q20", "cache: the held bytes' hash is recorded", SW, "        w.cache = (sha, loader)", "        pass",
  [lw + "test_a_cached_blend_is_loaded_from_the_hashed_bytes_once"])
q("Q21", "cache: the hash stands only when the held loader was used", SW,
  "    if w.cache is not None and w.cache[1] is used:", "    if w.cache is not None:",
  [core + "test_the_cache_hash_stands_only_for_the_held_loader"])
q("Q22", "pa_open: one PA ledger per witness", SW, "    if w is None or w.pipeline_inputs is not None:",
  "    if w is None:", [pw + "test_a_second_run_pipeline_under_one_witness_records_nothing"])
q("Q23", "pa_names: only for this run's ledger", SW, "    if w is not None and w.pipeline_inputs is ledger:",
  "    if w is not None:", [core + "test_a_second_pipeline_ledger_cannot_name_the_first_ones_files"])
q("Q24", "model: the cache's hash only when the held load was used", SW,
  '        w.model = {"source": "cache", "sha256": w.cache[0] if w.cache_used and w.cache is not None else None}',
  '        w.model = {"source": "cache", "sha256": w.cache[0] if w.cache is not None else None}',
  [pw + "test_a_cache_hash_stands_only_when_the_held_load_was_used"])
q("Q25", "model: trained with the save digest", SW, '        w.model = {"source": "trained", "sha256": w.save_digest}',
  '        w.model = {"source": "trained", "sha256": None}',
  [pw + "test_an_empty_cached_blend_trains_and_is_witnessed_as_trained"])
q("Q26", "hashing_writer: unwitnessed saves write through the plain file", SW,
  "    if current() is None:\n        return f\n    try:\n        return HashingWriter(f, hashlib.sha256())",
  "    try:\n        return HashingWriter(f, hashlib.sha256())",
  [pw + "test_an_unwitnessed_save_writes_through_the_plain_file"])
q("Q27", "saved: a withheld digest is recorded with its error", SW,
  '        note(w.errors, "blend save: the bytes hashed are not the bytes the file reports written; digest withheld")',
  "        pass", [core + "test_a_withheld_save_digest_is_recorded_as_an_error"])
q("Q28", "calibration_pa: the PA file's name is recorded", SW, "    w.calibration.pa_name = path.name", "    pass",
  [lw + "test_calibration_applied_records_every_part"])
q("Q29", "pick_file: an unreadable pick takes the C-level OSError stand-in", SW,
  "    return w.calibration.picks.hold(path, _pick_text(path), _unreadable_pick)",
  "    return w.calibration.picks.hold(path, _pick_text(path))",
  [cw + "test_a_genuine_read_error_skips_the_file_without_an_added_retry"])
q("Q30", "pick_skipped: only a JSON error means the text was consumed", SW,
  "    if w is not None and isinstance(_sys.exc_info()[1], json.JSONDecodeError):", "    if w is not None:",
  [core + "test_a_path_read_that_failed_is_never_counted_as_consumed"])
q("Q31", "pick_names: a resolver that read nothing consumed no files", SW,
  '            w.calibration.pick_names = names(directory, "2*.json") if read else []',
  '            w.calibration.pick_names = names(directory, "2*.json")',
  [core + "test_a_resolver_that_read_nothing_consumed_no_pick_files"])
q("Q32", "pick_bind: a binding is confirmed after its append", SW,
  "        c.bound.append(len(c.bindings))                   # earned: only an append that returned is confirmed",
  "        pass", [cw + "test_fit_witness_records_n_fit_bindings_and_the_canonical_map"])
q("Q33", "fit: bindings only when every append was confirmed", SW,
  "        if (c.bound == list(range(1, len(c.bindings) + 1))\n                and ", "        if (True\n                and ",
  [r1 + "test_a_binding_that_raised_after_appending_is_still_withheld"])
q("Q34", "fit: bindings only when their values are exactly the samples'", SW,
  '                and [(b["p"], b["y"]) for b in c.bindings] == [(s[0], s[1]) for s in samples]):',
  "                and True):", [cw + "test_mismatched_bindings_withhold_the_samples"])
q("Q35", "fit: pick inputs only when complete", SW,
  "        if c.pick_names is not None and c.picks.complete(c.pick_names):", "        if c.pick_names is not None:",
  [r1 + "test_an_omitted_input_with_a_lost_error_is_never_published_as_complete"])
q("Q36", "pick text: Path.read_text's encoding", SW,
  "        reader = _io.TextIOWrapper(_io.BytesIO(raw), encoding=_io.text_encoding(None))",
  '        reader = _io.TextIOWrapper(_io.BytesIO(raw), encoding="latin-1")',
  [cw + "test_the_decoder_matches_read_text_and_the_production_helper"])
q("Q37", "unreadable pick: its read raises OSError", SW,
  "    return _types.SimpleNamespace(read_text=_functools.partial(_os.read, -1, 0), name=path.name, sha256=None)",
  "    return _types.SimpleNamespace(read_text=str, name=path.name, sha256=None)",
  [core + "test_the_unreadable_pick_stand_in_raises_oserror_from_c"])
q("Q38", "confirm: never raises", SW,
  '        except Exception as e:\n            note(self.errors, f"{self.what}: confirmation failed", e)',
  '        except Exception as e:\n            raise', [core + "test_a_confirmation_never_raises"])
# the hooks beside the deployed statements
q("H1", "run_pipeline: the PA parse uses the held bytes", PR,
  "            parquet = ledger.hold(parquet, _sw.parquet_buffer)     # the held, hashed bytes, or the path itself",
  "            pass", [pw + "test_parquet_is_parsed_once_from_the_hashed_buffer"])
q("H2", "run_pipeline: each PA parse is confirmed", PR,
  "            ledger.confirm(parquet)             # earned: confirms the record only if those bytes were parsed",
  "            pass", [pw + "test_parquet_is_parsed_once_from_the_hashed_buffer"])
q("H3", "run_pipeline: the consumed PA files are listed", PR, "        _sw.pa_names(proc, len(dfs), ledger)", "        pass",
  [pw + "test_parquet_is_parsed_once_from_the_hashed_buffer"])
q("H4", "run_pipeline: the cache branch is witnessed", PR, '            _sw.model("cache")', "            pass",
  [pw + "test_a_cached_blend_is_witnessed_as_cache_with_the_held_bytes_hash"])
q("H5", "run_pipeline: the train branch is witnessed", PR,
  '            _sw.model("trained" if save_blend_path else "trained_unsaved")', "            pass",
  [pw + "test_training_without_a_save_path_is_witnessed_as_unsaved"])
q("H6", "save_blend: the dump goes through the hashing writer", PR, "            f = _sw.hashing_writer(f)", "            pass",
  [pw + "test_save_writes_exactly_the_pickle_bytes_and_records_their_hash"])
q("H7", "save_blend: the digest is recorded", PR, "            _sw.saved(f)", "            pass",
  [pw + "test_save_writes_exactly_the_pickle_bytes_and_records_their_hash"])
q("H8", "run_pipeline: this run's PA ledger is opened", PR,
  "        ledger = _sw.pa_open()                  # serving witness: this run's PA records (records nothing unwitnessed)",
  "        ledger = _sw._NULL_LEDGER", [pw + "test_parquet_is_parsed_once_from_the_hashed_buffer"])
q("H9", "resolver: an empty PA frame consumed no pick files", CA, "            _sw.pick_names(picks_dir, read=False)",
  "            pass", [core + "test_a_resolver_that_read_nothing_consumed_no_pick_files"])
q("H10", "resolver: each pick is read from its held bytes", CA,
  "            f = _sw.pick_file(f)                # the held, hashed text's stand-in, or the path itself",
  "            pass", [cw + "test_normal_path_reads_each_pick_file_once"])
q("H11", "resolver: a pick skipped for its JSON was still consumed", CA, "                _sw.pick_skipped(f)",
  "                pass", [cw + "test_pick_inputs_record_every_held_buffer_in_read_order_including_skipped_files"])
q("H12", "resolver: each read pick is confirmed", CA,
  "            _sw.pick_read(f)                    # earned: confirms the record only if that text was read",
  "            pass", [cw + "test_pick_inputs_record_every_held_buffer_in_read_order_including_skipped_files"])
q("H13", "resolver: each sample is bound", CA, "                _sw.pick_bind(f, pick_date, slot_key, bid, samples[-1])",
  "                pass", [cw + "test_bindings_follow_the_samples_exactly"])
q("H14", "resolver: the consumed pick files are listed", CA, "        _sw.pick_names(picks_dir)\n", "        pass\n",
  [cw + "test_pick_inputs_record_every_held_buffer_in_read_order_including_skipped_files"])
q("H15", "fit: no sklearn is witnessed", CA, '            _sw.fit("no_sklearn")', "            pass",
  [cw + "test_no_sklearn_is_witnessed"])
q("H16", "fit: insufficient support is witnessed", CA, '            _sw.fit("insufficient_support", samples)',
  "            pass", [cw + "test_insufficient_support_is_witnessed_and_returns_none"])
q("H17", "fit: a fitted calibrator is witnessed", CA, '        _sw.fit("fitted", samples, cal)', "        pass",
  [cw + "test_fit_witness_records_n_fit_bindings_and_the_canonical_map"])
q("H18", "predict_local: a witness is made", OR, "        witness, token = _sw.Serving(), None",
  "        witness, token = None, None", [lw + "test_inputs_and_only_the_serving_key_remain_in_attrs"])
q("H19", "predict_local: the cache is loaded from its held bytes", OR,
  "            load_blend = _sw.cache_loader(cache_path, load_blend, witness)    # one read: the held, hashed bytes",
  "            pass", [lw + "test_a_cached_blend_is_loaded_from_the_hashed_bytes_once"])
q("H20", "predict_local: the held cache load is confirmed", OR, "            _sw.cache_used(load_blend, witness)",
  "            pass", [lw + "test_a_cached_blend_is_loaded_from_the_hashed_bytes_once"])
q("H21", "predict_local: the witness is made current after the cache load", OR,
  "        token = _sw.begin(witness)       # current only after the cache, whose genuine failure propagates as deployed",
  "        token = None", [lw + "test_inputs_and_only_the_serving_key_remain_in_attrs"])
q("H22", "predict_local: a prediction failure stops recording", OR,
  "        try:\n            _sw.end(token)\n        except Exception:\n            pass\n        print(",
  "        try:\n            pass\n        except Exception:\n            pass\n        print(",
  [lw + "test_a_prediction_failure_returns_none_as_before"])
q("H23", "predict_local: whether calibration is enabled is recorded", OR,
  '        _sw.calibration_enabled(os.environ.get("BTS_USE_CALIBRATION", "0") == "1")', "        pass",
  [lw + "test_calibration_off"])
q("H24", "predict_local: calibration's PA is parsed from its held bytes", OR,
  "                    current_pa = _sw.calibration_pa(current_pa)    # the held, hashed bytes, or the path itself",
  "                    pass", [lw + "test_calibration_applied_records_every_part"])
q("H25", "predict_local: calibration's PA parse is confirmed", OR, "                    _sw.calibration_pa_used(current_pa)",
  "                    pass", [lw + "test_calibration_applied_records_every_part"])
q("H26", "predict_local: applied is recorded from the assignment", OR, "                        _sw.calibration_applied()",
  "                        pass", [lw + "test_calibration_applied_records_every_part"])
q("H27", "predict_local: no PA file is recorded", OR, '                    _sw.calibration_outcome("no_pa_file")',
  "                    pass", [lw + "test_calibration_without_a_pa_file"])
q("H28", "predict_local: a calibration failure is recorded", OR, "                _sw.calibration_failed(e)",
  "                pass", [lw + "test_a_genuine_calibration_failure_before_assignment_is_failed"])
q("H29", "predict_local: the witness is sealed and attached", OR,
  '        _sw.run_end(predictions, witness, token)        # seal, attach attrs["serving"], stop recording', "        pass",
  [lw + "test_inputs_and_only_the_serving_key_remain_in_attrs"])
q("H30", "save_slate: the witness is taken off the frame", SL,
  "            serving = _take_serving(predictions)    # off the frame before the row extraction copies its attrs",
  "            serving = None", [T + "test_slate_v3.py"])
q("H31", "save_slate: the frame's witness is dropped before the rows even if taking it failed", SL,
  '            predictions.attrs.pop("serving", None)\n        except Exception:\n            pass\n        cols',
  "            pass\n        except Exception:\n            pass\n        cols",
  [core + "test_the_slate_drops_the_witness_from_the_frame_even_if_taking_it_failed"])
q("H32", "save_slate: the envelope carries the witness", SL, '            payload["serving"] = serving', "            pass",
  [T + "test_slate_v3.py"])
# after the final sweep (e36e038): the six lines neither the sweep nor a unit test reached
q("Q39", "cache_loader: with no witness, the deployed loader", SW,
  "    if w is None:\n        return deployed\n    try:\n        raw = path.read_bytes()",
  "    try:\n        raw = path.read_bytes()", [core + "test_the_cache_hooks_without_a_witness_are_the_deployed_load"])
q("Q40", "cache_used: with no witness, nothing recorded", SW,
  "    if w is None:\n        return\n    if w.cache is not None and w.cache[1] is used:",
  "    if w.cache is not None and w.cache[1] is used:",
  [core + "test_the_cache_hooks_without_a_witness_are_the_deployed_load"])
q("H33", "predict_local: a failed stop after a prediction failure is contained", OR,
  "        try:\n            _sw.end(token)\n        except Exception:\n            pass\n        print(",
  "        _sw.end(token)\n        print(",
  [lw + "test_a_failed_stop_after_a_prediction_failure_changes_nothing_deployed"])
q("H34", "_take_serving: a failed warning for an unserializable witness is contained", SL,
  '        try:\n            log.warning("serving witness not serializable (slate still written)")\n        except Exception:\n'
  '            pass', '        log.warning("serving witness not serializable (slate still written)")',
  [T + "test_slate_v3.py::test_a_failed_warning_for_an_unserializable_witness_still_writes_the_slate"])
# Masked layers, as r1's O11 (ledger run at c007271): each mutant removes a layer whose loss the next layer hides, so
# it is run combined with that next layer, whose own removal has its own entry (H31; the call-site guard is reached
# only by an unforeseen fault).
sl = Path(SL).read_text()
def span(a, b):
    i = sl.index(a); j = sl.index(b, i) + len(b)
    assert sl.count(a) == 1 and sl.count(b) == 1
    return sl[i:j]
o14 = span('        serving = predictions.attrs.pop("serving", None)\n', '            predictions.attrs.pop("serving", None)\n')
o14_new = o14.replace('        serving = predictions.attrs.pop("serving", None)\n', '        serving = predictions.attrs.get("serving")\n'
                      ).replace('            predictions.attrs.pop("serving", None)\n', '            pass\n')
for m in kept:
    if m["id"] == "O14":
        m.update(rule="F1: save_slate takes the witness off the frame before building rows (combined with the backstop "
                      "drop, which masks it alone; r4)", old=o14, new=o14_new)
inner = '        try:\n            log.warning("serving witness not serializable (slate still written)")\n        except Exception:\n            pass'
guard = ('        try:\n            serving = _take_serving(predictions)    # off the frame before the row extraction copies '
         'its attrs\n        except Exception:\n            serving = None')
h34 = span(inner, guard)
h34_new = h34.replace(inner, '        log.warning("serving witness not serializable (slate still written)")').replace(
    guard, '        serving = _take_serving(predictions)    # off the frame before the row extraction copies its attrs')
for m in N:
    if m["id"] == "H34":
        m.update(rule="_take_serving: a failed warning for an unserializable witness is contained (combined with "
                      "save_slate's call-site guard, which masks it alone)", old=h34, new=h34_new)
ms = kept + N
ids = [m["id"] for m in ms]
assert len(ids) == len(set(ids))
bad = [(m["id"], Path(m["file"]).read_text().count(m["old"])) for m in ms if Path(m["file"]).read_text().count(m["old"]) != 1]
env = {**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache", "TZ": "America/New_York", "OMP_NUM_THREADS": "1"}
nocoll = []
for m in ms:
    r = subprocess.run(["uv", "run", "python", "-B", "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                        *m["tests"]], capture_output=True, text=True, env=env)
    got = re.search(r"(\d+) tests? collected", r.stdout)
    want = len(m["tests"]) if all("::" in t for t in m["tests"]) else 1
    if r.returncode != 0 or got is None or (int(got.group(1)) != want if want > 1 or "::" in m["tests"][0]
                                            else int(got.group(1)) < 1):
        nocoll.append((m["id"], got and got.group(1)))
print(len(ms), "entries:", len(kept), "kept,", len(N), "new;", len(retired), "retired | bad anchors:", bad,
      "| not collecting:", nocoll)
if not bad and not nocoll and "--write" in sys.argv:
    (EV / "mutants.json").write_text(json.dumps(ms, indent=1) + "\n")
    (EV / "mutants_retired_r4.json").write_text(json.dumps(retired, indent=1) + "\n")
    print("written")
