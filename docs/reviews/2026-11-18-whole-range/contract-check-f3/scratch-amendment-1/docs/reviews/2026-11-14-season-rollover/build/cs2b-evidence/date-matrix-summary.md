# Recorded date matrix

Base `d8eb50d0` versus the CS2B branch, 18 values × 7 real entry paths.
Raw records include return/exception, output, observed reads, pick result, and exact streak bytes (hex).
All clocks fixed to 2027-03-24 23:00 ET; external services mocked.

`a`: season-scoped read/write; `b`: unknown dated write rejected; `D`: required dateless tag preservation (outside the requested a/b classes).

| Value | Shadow | Void | Final peer | Early peer | CLI | Strategy | Update |
|---|---|---|---|---|---|---|---|
| padded: `"2027-03-24"` | a | a | a | a | a | a | a |
| unpadded-month: `"2027-3-24"` | a | a | a | a | = | a | a |
| unpadded-day: `"2027-03-4"` | a | a | a | a | = | a | a |
| unpadded-both: `"2027-3-4"` | a | a | a | a | = | a | a |
| compact: `"20270324"` | a | a | a | a | a | a | a |
| week: `"2027-W12-3"` | a | a | a | a | a | a | a |
| leading-space: `" 2027-03-24"` | = | = | = | = | = | = | b |
| trailing-space: `"2027-03-24 "` | = | = | = | = | = | = | b |
| time: `"2027-03-24T10:00"` | = | = | = | = | = | = | b |
| slash: `"2027/03/24"` | = | = | = | = | = | = | b |
| dot: `"2027.03.24"` | = | = | = | = | = | = | b |
| impossible: `"2027-02-30"` | = | = | = | = | = | = | b |
| junk: `"junk"` | = | = | = | = | = | = | b |
| empty: `""` | = | = | = | = | = | = | b |
| date-object: `date(2027, 3, 24)` | = | = | = | = | = | = | b |
| none: `None` | = | = | = | = | = | = | D |
| bytes: `b"2027-03-24"` | = | = | = | = | = | = | b |
| integer: `20270324 (int)` | = | = | = | = | = | = | b |

Counts: {'a': 39, '=': 75, 'b': 11, 'D': 1}.

For a: scheduler streak 13 → 0, CLI void display 13 → 0 where its age guard permits entry; strategy saver False → True (later base date exceptions remain identical); scorer 14 → 1, saver False → True, season 2027, original marker retained.
For b: base returns 14 except date/bytes objects raise TypeError; branch raises ValueError before reads and preserves all bytes.
For D: date=None is dateless. Both return 14, but the base removes the tag and the branch preserves season 2026, explicitly required by rows B/D. No row-L implementation can remove this difference while preserving those rows.

## Every differing pair

| Value | Path | Class | Differing recorded fields |
|---|---|---|---|
| padded | shadow | a | outcome, reads |
| padded | void | a | output, reads |
| padded | final_peer | a | output, reads |
| padded | midgame_peer | a | outcome, reads |
| padded | cli | a | outcome, reads |
| padded | strategy | a | outcome, reads |
| padded | update | a | outcome, reads, after, after_hex |
| unpadded-month | shadow | a | outcome, reads |
| unpadded-month | void | a | output, reads |
| unpadded-month | final_peer | a | output, reads |
| unpadded-month | midgame_peer | a | outcome, reads |
| unpadded-month | strategy | a | outcome, reads |
| unpadded-month | update | a | outcome, reads, after, after_hex |
| unpadded-day | shadow | a | outcome, reads |
| unpadded-day | void | a | output, reads |
| unpadded-day | final_peer | a | output, reads |
| unpadded-day | midgame_peer | a | outcome, reads |
| unpadded-day | strategy | a | outcome, reads |
| unpadded-day | update | a | outcome, reads, after, after_hex |
| unpadded-both | shadow | a | outcome, reads |
| unpadded-both | void | a | output, reads |
| unpadded-both | final_peer | a | output, reads |
| unpadded-both | midgame_peer | a | outcome, reads |
| unpadded-both | strategy | a | outcome, reads |
| unpadded-both | update | a | outcome, reads, after, after_hex |
| compact | shadow | a | outcome, reads |
| compact | void | a | output, reads |
| compact | final_peer | a | output, reads |
| compact | midgame_peer | a | outcome, reads |
| compact | cli | a | outcome, reads |
| compact | strategy | a | reads |
| compact | update | a | outcome, reads, after, after_hex |
| week | shadow | a | outcome, reads |
| week | void | a | output, reads |
| week | final_peer | a | output, reads |
| week | midgame_peer | a | outcome, reads |
| week | cli | a | outcome, reads |
| week | strategy | a | reads |
| week | update | a | outcome, reads, after, after_hex |
| leading-space | update | b | outcome, reads, after, after_hex |
| trailing-space | update | b | outcome, reads, after, after_hex |
| time | update | b | outcome, reads, after, after_hex |
| slash | update | b | outcome, reads, after, after_hex |
| dot | update | b | outcome, reads, after, after_hex |
| impossible | update | b | outcome, reads, after, after_hex |
| junk | update | b | outcome, reads, after, after_hex |
| empty | update | b | outcome, reads, after, after_hex |
| date-object | update | b | outcome, reads |
| none | update | D | after, after_hex |
| bytes | update | b | outcome, reads |
| integer | update | b | outcome, reads, after, after_hex |
