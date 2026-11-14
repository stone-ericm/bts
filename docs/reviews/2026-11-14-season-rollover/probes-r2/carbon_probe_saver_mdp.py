from bts.strategy import _load_mdp, _mdp_action_from, effective_pick_bar

mdp = _load_mdp()
print("boundaries:", mdp["boundaries"])
for s in range(0, 17):
    diffs = []
    for p in [x / 1000 for x in range(600, 900)]:
        a_true = _mdp_action_from(mdp, p, s, "2027-03-24", True)
        a_false = _mdp_action_from(mdp, p, s, "2027-03-24", False)
        if a_true != a_false:
            diffs.append((p, a_true, a_false))
    bt = effective_pick_bar(s, "2027-03-24", True)
    bf = effective_pick_bar(s, "2027-03-24", False)
    if diffs or bt != bf:
        print(f"streak={s}: n_diffs={len(diffs)}, sample={diffs[0]}..{diffs[-1]}, bar_T={bt}, bar_F={bf}")
