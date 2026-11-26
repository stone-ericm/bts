"""Full independent Cartesian comparison. Run with the venv Python, no network."""
import collections
import gzip
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from review_support import *


def switches():
    out = {"absent": None, "automatic": contract(), "revoked": contract(status="revoked", revoked_at="2027-07-14T00:00:00Z"),
           "other_season": contract(season=2026)}
    bad = {2: ("schema_version", None), 3: ("season", True), 4: ("status", []),
           5: ("activation_id", "A" * 32), 6: ("switched_at", None),
           7: ("revoked_at", "naive"), 8: ("note", [])}
    for n in range(1, 9):
        out[f"bad{n}"] = [] if n == 1 else contract(**dict([bad[n]]))
    for a, b in itertools.combinations(range(1, 9), 2):
        out[f"bad{a}+{b}"] = [contract(**dict([bad[b]]))] if a == 1 else contract(**dict([bad[a], bad[b]]))
    return out


def autos():
    out = {"absent": None, "v1": auto(schema_version="bts_contest_streak_auto_v1"), "v2": auto(),
           "other_id": auto(activation_id="f" * 32), "other_season": auto(season=2026),
           "refused_active": auto(active_streak=-1), "refused_best": auto(best_streak=58),
           "refused_saver": auto(saver_available="yes"), "utf8": b"\xff", "nonobject": [],
           "other_account": auto(user_id=999), "malformed": b"{", "missing_season": {k:v for k,v in auto().items() if k != "season"},
           "missing_id": {k:v for k,v in auto().items() if k != "activation_id"}}
    for field, vals in {"season": [None, True, "2027", [], {}, 2027.5],
                        "activation_id": [None, True, [], {}, "A" * 32, "a" * 31, "a" * 33, "g" * 32]}.items():
        for i,v in enumerate(vals):
            out[f"{field}_{i}"] = auto(**{field:v})
    return out


def responses():
    import httpx
    good = {"activeStreak":7, "seasonBestStreak":9, "predictions":[{"roundId":1,"result":"hit"}]}
    rounds = {1: REAL_DATE(2027,7,14)}
    out = {"valid":(good,rounds), "no_settled":(good | {"predictions":[{"roundId":1,"result":None}]}, rounds),
           "missing":({k:v for k,v in good.items() if k != "predictions"},rounds),
           "null":(good|{"predictions":None},rounds), "nonlist":(good|{"predictions":{}},rounds),
           "nonobject":(good|{"predictions":[None]},rounds), "empty_predictions":(good|{"predictions":[]},rounds),
           "empty_rounds":(good,{}), "profile_null":(None,rounds),
           "best57":(good|{"seasonBestStreak":57},rounds), "best58":(good|{"seasonBestStreak":58},rounds),
           "active_above_best":(good|{"activeStreak":10},rounds), "bool_active":(good|{"activeStreak":True},rounds),
           "bool_best":(good|{"seasonBestStreak":True},rounds),
           "bad_round_id":(good|{"predictions":[{"roundId":"bad","result":"hit"}]},rounds)}
    for code in [503,429]:
        error = httpx.HTTPStatusError(str(code),request=httpx.Request("GET","https://offline.invalid/rounds"), response=httpx.Response(code))
        out[f"rounds{code}"] = (good,error)
        out[f"null_predictions_rounds{code}"] = (good|{"predictions":None},error)
        out[f"null_profile_rounds{code}"] = (None,error)
    return out


def worker(label, source):
    h = Harness(source)
    case = ROOT / ("matrix-case-" + label)
    output = ROOT / ("matrix-" + label + ".jsonl.gz")
    problems = []
    count = 0
    counters = collections.Counter()
    with gzip.open(output,"wt",compresslevel=1) as stream:
        for (sn,sv),(an,av),(rn,rv),dry in itertools.product(switches().items(),autos().items(),responses().items(),[False,True]):
            if case.exists(): shutil.rmtree(case)
            case.mkdir()
            picks = case / "picks"
            if sv is not None: put(picks,"season_contract.json",sv)
            if av is not None: put(picks,"contest_streak.json",av)
            # Seed a saver to prove ledger-only fetches never even read it.
            put(picks,"saver_state.json",{"season":2027,"state":"not_earned"})
            h.profile,h.rounds = rv
            r = h.fetch(case,picks,dry)
            r["case"] = [sn,an,rn,dry]
            fs = decode_files(r["after"])
            r["status"] = json.loads(fs["health_state/contest_streak_fetch_status.json"]) if "health_state/contest_streak_fetch_status.json" in fs else None
            # Atomic-writer temporary names are random: preserve exact traces and
            # compare logical events rather than those names.
            stream.write(json.dumps(r,separators=(",",":")) + "\n")
            counters[str(r["exit"])] += 1
            if label == "branch":
                try:
                    identity_fail = an == "other_account"
                    raw_fail = an in {"utf8","nonobject"}
                    response_fail = rn not in {"valid","no_settled","empty_predictions","best57"}
                    corrupt = sn.startswith("bad")
                    expected = 1 if raw_fail else 2 if identity_fail or response_fail or corrupt else 0
                    assert r["exit"] == expected, (r["exit"],expected,r["exception"])
                    old = decode_files(r["before"])
                    key = "picks/account_state/contest_streak.json"
                    changed = old.get(key) != fs.get(key)
                    installs = not (raw_fail or identity_fail or response_fail or corrupt or dry) and sn == "automatic"
                    assert changed == installs, "observation write"
                    switch_reads = [e for e in r["trace"] if e[0] == "open" and "season_contract.json" in e[1]]
                    assert len(switch_reads) == (0 if raw_fail or identity_fail or response_fail or sn == "absent" else 1), "snapshot reads"
                    saver_io = [e for e in r["trace"] if any("saver_" in str(s) for s in e[1:])]
                    if sn != "automatic" or dry or expected:
                        assert not saver_io, "unexpected saver I/O"
                    ledger_key = "picks/account_state/contest_ledger.jsonl"
                    if expected or dry:
                        assert ledger_key not in fs, "unexpected ledger"
                    else:
                        row = json.loads(fs[ledger_key])
                        assert row["season"] == 2027 and row["switched"] is (sn == "automatic")
                        assert row["activation_id"] == (IDENT if sn == "automatic" else None)
                        assert sum(row["settled_by_year"].values()) + row["settled_unmapped"] + row["unsettled"] == len(row["predictions"])
                    if raw_fail: assert r["status"] is None and not r["dms"]
                    elif expected:
                        assert r["status"]["last_error"] and len(r["dms"]) == 1
                        if not identity_fail and not response_fail and corrupt:
                            assert "check " + sn[3:].split("+")[0] in r["output"]
                    else: assert not r["dms"]
                except AssertionError as exc:
                    problems.append({"case":r["case"],"problem":str(exc)})
            count += 1
            if count % 5000 == 0: print(label,count,flush=True)
    write_json(ROOT / ("matrix-"+label+"-summary.json"),dict(count=count,exits=counters,problems=problems))
    print(label,"done",count,"problems",len(problems),flush=True)


def main():
    if len(sys.argv)>1:
        worker(sys.argv[1],Path(sys.argv[2]))
        return
    base = ROOT / "base"
    base.mkdir(exist_ok=True)
    archived = subprocess.run(["git","archive","8421e3cd","src"],cwd=REPO,stdout=subprocess.PIPE,check=True).stdout
    import io
    with tarfile.open(fileobj=io.BytesIO(archived)) as tf: tf.extractall(base,filter="data")
    for label,source in [("base",base/"src"),("branch",REPO/"src")]:
        subprocess.run([sys.executable,__file__,label,str(source)],check=True)
    differences = collections.Counter()
    unexplained=[]
    with gzip.open(ROOT/"matrix-base.jsonl.gz","rt") as a,gzip.open(ROOT/"matrix-branch.jsonl.gz","rt") as b:
        for al,bl in zip(a,b,strict=True):
            av,bv=json.loads(al),json.loads(bl)
            assert av["case"]==bv["case"]
            sn,an,rn,dry=bv["case"]
            # Row 5 preserves the prior nonobject/UTF8 crash, and row 4 preserves identity priority.
            if an in {"utf8","nonobject","other_account"}:
                group="preserved_prior_failure"
                if any(av[k]!=bv[k] for k in ["exit","output","dms","after"]): unexplained.append(bv["case"])
            elif rn not in {"valid","no_settled","empty_predictions","best57"}:
                group="row3_validation_or_preserved_failure"
            elif sn.startswith("bad"):
                group="row4_corrupt_switch"
            else: group="rows4_6_persistence_or_dry"
            differences[group]+=1
    write_json(ROOT/"matrix-comparison.json",dict(groups=differences,unexplained=unexplained,
              switches=len(switches()),automatic_files=len(autos()),responses=len(responses()),dry_settings=2))


if __name__=="__main__": main()
