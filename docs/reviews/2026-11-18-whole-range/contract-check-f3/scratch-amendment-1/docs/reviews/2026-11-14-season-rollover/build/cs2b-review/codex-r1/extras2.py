"""Correct the extra fixtures to carry real delivery evidence and an open correction window.

The original evidence is retained to make the initial non-exercised race visible.
"""
from pathlib import Path
source=(Path(__file__).resolve().parent/'extras.py').read_text()
source=source.replace("notification_sent=True)","notification_sent=True,bluesky_posted=True)")
source=source.replace("f'extras-{MODE}.jsonl'","f'extras2-{MODE}.jsonl'")
source=source.replace("f'extras-fixtures-{MODE}'","f'extras2-fixtures-{MODE}'")
source=source.replace("d=directory();plant(d,'2026-12-30','hit')",
    "d=directory();plant(d,'2027-03-24' if correction else '2026-12-30','hit')")
source=source.replace("clock=dt.datetime(2027,1,1,1,tzinfo=dt.timezone.utc)",
    "clock=dt.datetime(2027,3,25,2,tzinfo=ET) if correction else dt.datetime(2027,1,1,1,tzinfo=dt.timezone.utc)")
exec(compile(source,__file__,'exec'))
