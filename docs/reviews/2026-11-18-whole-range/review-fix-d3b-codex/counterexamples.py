"""Counterexamples for positive evidence in the stop and HTTP proofs."""
from ctypes import CDLL, POINTER, byref, c_char_p, c_int, c_uint, c_void_p
from itertools import product
import json
import sys

from review_box import DEPLOY, RECOVER, box, group, inject_shim, invoke, posts, record

MANAGER = inject_shim(r'''
if command == "systemctl" and args[:2] == ["show", "bts-scheduler.service"]:
    old_stop = (root / "legacy-stopped").exists()
    group = "/legacy-group" if scenario == "not-found-with-group" else "/legacy-pre"
    if scenario == "legacy-fragment-present" and old_stop:
        group = "/legacy-post"
    fields = dict(Id=args[1], LoadState="not-found", ActiveState="inactive", MainPID="0", ControlGroup=group)
    if scenario == "legacy-fragment-present" and not old_stop:
        fields.update(LoadState="loaded", ActiveState="active", MainPID="5042")
    with (root / "observations").open("a") as log:
        print(json.dumps(dict(args=args, fields=fields)), file=log)
    for flag in args:
        if flag in fields:
            print(f"{flag}={fields[flag]}")
    raise SystemExit(0)
if command == "systemctl" and args == ["is-enabled", "bts-scheduler.service"]:
    print("disabled")
    raise SystemExit(1)
''')


def absent():
    for recovery, mode in product((False, True), ("not-found-with-group", "legacy-fragment-present")):
        with box(record() if recovery else None, shim=MANAGER) as path:
            if mode == "legacy-fragment-present":
                fragment = path / "etc/systemd/system/bts-scheduler.service"
                fragment.parent.mkdir(parents=True)
                fragment.write_text("[Service]\nType=oneshot\nExecStart=/bin/false\n")
                group(path, "/legacy-pre", 0)
                group(path, "/legacy-post", 0)
            else:
                group(path, "/legacy-group", 1)
            state, _ = invoke(path, f"absence proof {mode} {recovery=}", RECOVER if recovery else DEPLOY,
                              mode, fault="" if recovery else "reset:17")
            print((path / "observations").read_text(), end="", flush=True)
            assert "Scheduler stopped: verified." in state["out"]
            assert state["rc"] == 1 and " verified verified verified" in state["attempt"]
            if mode == "not-found-with-group":
                assert not any("legacy" in entry for entry in state["groups"])
                assert not any(c[1:2] == ["stop"] for c in state["writes"])
            else:
                assert "/legacy-pre" in state["groups"] and "/legacy-post" in state["groups"]


def curl_path(url, preserve=False):
    # libcurl URL parsing only. No command execution, sockets, DNS or external URL calls.
    lib = CDLL("/usr/lib/libcurl.4.dylib")
    lib.curl_url.restype = c_void_p
    lib.curl_url_cleanup.argtypes = [c_void_p]
    lib.curl_free.argtypes = [c_void_p]
    lib.curl_url_set.argtypes = [c_void_p, c_int, c_char_p, c_uint]
    lib.curl_url_get.argtypes = [c_void_p, c_int, POINTER(c_char_p), c_uint]
    url_handle = lib.curl_url()
    value = c_char_p()
    try:
        assert lib.curl_url_set(url_handle, 0, url.encode(), (1 << 4) if preserve else 0) == 0
        assert lib.curl_url_get(url_handle, 7, byref(value), 0) == 0
        return value.value.decode()
    finally:
        if value:
            lib.curl_free(value)
        lib.curl_url_cleanup(url_handle)


def dot_path():
    for recovery in (False, True):
        with box(record() if recovery else None, url="https://checks.invalid/configured/../different-check") as path:
            state, calls = invoke(path, f"normalized POST path {recovery=}", RECOVER if recovery else DEPLOY,
                                  fault="" if recovery else "reset:17")
            (argv,) = posts(calls)
            actual = curl_path(argv[-1], "--path-as-is" in argv)
            literal = curl_path(argv[-1], True)
            print(json.dumps(dict(literal=literal, actual=actual, argv=argv), ensure_ascii=False), flush=True)
            assert (literal, actual) == ("/configured/../different-check/fail", "/different-check/fail")
            assert "Failure ping received HTTP 2xx: verified." in state["out"]
            assert state["attempt"].endswith(" verified verified verified\n")


if __name__ == "__main__":
    globals()[sys.argv[1]]()
