"""Compare old vs new MDP policies."""
from bts.simulate.mdp import load_policy, lookup_action
from datetime import datetime

new_policy, new_bounds, new_len = load_policy("data/models/mdp_policy.npz")
old_policy, old_bounds, old_len = load_policy("data/models/mdp_policy_old_aug_sept.npz")

days_remaining = (datetime(2026, 9, 28) - datetime(2026, 4, 2)).days
print(f"Days remaining: {days_remaining}, Streak: 4, Saver: True, Top pick: 0.715")

for label, pol, bnd, sl in [("OLD", old_policy, old_bounds, old_len),
                              ("NEW", new_policy, new_bounds, new_len)]:
    a = lookup_action(pol, bnd, 4, days_remaining, True, 0.715, sl)
    print(f"  {label}: {a}")

print("\n=== Where they diverge (p=0.82, saver=True) ===")
for dr_label, dr in [("Aug 1 (~58d)", 58), ("Sep 1 (~27d)", 27)]:
    for s in [0, 5, 10, 20, 30, 40]:
        old_a = lookup_action(old_policy, old_bounds, s, dr, True, 0.82, old_len)
        new_a = lookup_action(new_policy, new_bounds, s, dr, True, 0.82, new_len)
        if old_a != new_a:
            print(f"  {dr_label}, streak={s}: {old_a} -> {new_a}")

diffs = 0
total = 0
for s in range(57):
    for d in range(1, 181):
        for sv in [0, 1]:
            for q in range(len(new_bounds) + 1):
                total += 1
                oq = min(q, old_policy.shape[3] - 1)
                nq = min(q, new_policy.shape[3] - 1)
                if old_policy[s, min(d, old_len), sv, oq] != new_policy[s, min(d, new_len), sv, nq]:
                    diffs += 1
print(f"\nTotal differences: {diffs}/{total} ({100 * diffs / total:.1f}%)")
