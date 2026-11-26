"""Apply the chat service's grapheme cap at the real sendMessage boundary."""

import ctypes
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import bts
from report_harness import Chat, _box, _write_decision, observed, report


def grapheme_counter():
    """Use the OS Unicode character-break iterator, with code units in host byte order."""
    icu = ctypes.CDLL("/usr/lib/libicucore.A.dylib")
    icu.ubrk_open.argtypes = [ctypes.c_int32, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int32,
                              ctypes.POINTER(ctypes.c_int32)]
    icu.ubrk_open.restype = ctypes.c_void_p
    icu.ubrk_next.argtypes = [ctypes.c_void_p]
    icu.ubrk_next.restype = ctypes.c_int32
    icu.ubrk_close.argtypes = [ctypes.c_void_p]
    icu.ubrk_close.restype = None

    def count(text):
        data = ctypes.create_string_buffer(text.encode("utf-16-le" if sys.byteorder == "little" else "utf-16-be"))
        error = ctypes.c_int32(0)
        iterator = icu.ubrk_open(0, b"en_US", data, (len(data) - 1) // 2, ctypes.byref(error))
        assert iterator and error.value <= 0, error.value
        try:
            clusters = 0
            while icu.ubrk_next(iterator) != -1:
                clusters += 1
            return clusters
        finally:
            icu.ubrk_close(iterator)

    return count


graphemes = grapheme_counter()
assert graphemes("e\u0301👩🏽‍💻🇺🇸\r\n") == 4
candidate = sys.argv[1] == "candidate"
print(f"{sys.argv[1]} bts.__file__ = {bts.__file__}")
CASES = {
    "ordinary accented pick": ("name", "José Ramírez", True, False),
    "1000 code points": ("name", "a" * 395, True, False),
    "1001 code points": ("name", "a" * 396, True, True),
    "833 base code points": ("name", "a" * 700, True, True),
    "graphemes fit, combined code points do not": ("name", "e\u0301" * 300, True, True),
    "1000 base graphemes, over 1000 base code points": ("name", "👩🏽‍💻" * 867, True, True),
    "1001 base graphemes": ("name", "a" * 868, False, True),
    "long detail without pick": ("source", "offline-restored-" * 41, True, True),
    "clean with long pick": ("clean", "a" * 800, True, True),
}

with TemporaryDirectory() as tmp:
    for index, (label, (kind, value, base_accepted, compact)) in enumerate(CASES.items()):
        picks, cfg = _box(Path(tmp) / str(index))
        if kind == "source":
            _write_decision(picks, action="skip", source=value, scoreable=False)
            title = "SKIP NOT DELIBERATE"
        else:
            _write_decision(picks, primary={"batter_name": value, "batter_id": 11, "game_pk": 22})
            if kind == "clean":
                observed(picks)
                title = "MODEL STREAK FILE NEEDS REVIEW"
            else:
                observed(picks, reason="no_pick", confirmed=False)
                title = "NOT ENTERED"

        good, good_chat = report(picks, cfg, 1, chat=Chat(count=graphemes))
        payload = json.loads(good.stdout)
        assert good.exit_code == int(kind != "clean")
        assert len(good_chat.accepted) == int(kind != "clean" and base_accepted)
        (picks / "streak.json").write_bytes(b"[]")
        bad, bad_chat = report(picks, cfg, 1, chat=Chat(count=graphemes))
        if not candidate:
            assert (good.exit_code, good.stdout, good.stderr, good_chat.attempted) == (
                bad.exit_code, bad.stdout, bad.stderr, bad_chat.attempted)
        else:
            expected = payload | {"ok": False, "streak_problem": "not a JSON object"}
            assert bad.exit_code == 1 and json.loads(bad.stdout) == expected and bad.stderr == ""
            assert len(bad_chat.attempted) == len(bad_chat.accepted) == 1
            body = bad_chat.accepted[0]
            prefix = f"BTS day check 2027-04-02: {title}. "
            assert body.startswith(prefix), body
            assert (len(body) < 300) == compact, (label, len(body))
            if compact:
                assert body.startswith(prefix + "Local model streak file needs review (not a JSON object).")
                assert "day-outcome JSON in cron.log" in body
            else:
                assert body.startswith(good_chat.attempted[0] + " Also: ")
                assert len(body) == len(good_chat.attempted[0]) + 472
            if label == "graphemes fit, combined code points do not":
                assert len(good_chat.attempted[0]) + 472 > 1000
                assert graphemes(good_chat.attempted[0]) + 472 <= 1000

        def sizes(chat):
            return [(len(body), graphemes(body)) for body in chat.attempted]

        print(f"{label}: good {sizes(good_chat)} accepted={len(good_chat.accepted)}; "
              f"broken {sizes(bad_chat)} accepted={len(bad_chat.accepted)}")
