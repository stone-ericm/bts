"""Author's check for fresh Part 1 F6: are the prepared summary's spec_sha256 values the SHA-256 of the shipped,
unmodified spec bytes at c59ee58? (prepared_cost.py hashed `path.read_bytes()` before substituting the baseline in
memory.) Run from the bts-w15 worktree or any checkout holding c59ee58 and 98945f4."""
import hashlib, json, subprocess
summ = json.loads(subprocess.run(["git", "show", "98945f4:docs/audit/2026-09-29-incident-register-evidence/current_defence/prepared-c59ee58.json"],
                                 capture_output=True, check=True).stdout)
for r in sorted(summ["results"], key=lambda r: r["label"]):
    raw = subprocess.run(["git", "show", f"c59ee58:docs/audit/2026-09-29-incident-register-evidence/current_defence/specs/{r['label']}.json"],
                         capture_output=True, check=True).stdout
    print(r["label"], r["spec_sha256"][:16], "match" if hashlib.sha256(raw).hexdigest() == r["spec_sha256"] else "MISMATCH")
