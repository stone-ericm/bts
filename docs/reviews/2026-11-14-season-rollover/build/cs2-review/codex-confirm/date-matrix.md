# Date acceptance and side-effect evidence

Sources: base=d8eb50d0; r2=a26958dc; head=ee9fb870. Each cell is base / r2 / head.
Shadow strings invoke the real one-shot main; date objects/None invoke run_once and fail upstream.
Polling uses real reads, scoring locks, terminal persistence and replies (posting transport stubbed).
Only prediction, watchdog, remote statuses/results and MDP artifact availability are stubbed; socket connections are denied.

| Input repr | strptime result | Shadow | All void | Final peer | Early peer |
|---|---|---|---|---|---|
| '2027-03-24' | 2027 | artifact / artifact / artifact | final / final / final | final / final / final | final / final / final |
| '2027-3-24' | 2027 | artifact / no artifact / artifact | final / ValueError / final | final / ValueError / final | final / ValueError / final |
| '2027-03-4' | 2027 | artifact / no artifact / artifact | final / ValueError / final | final / ValueError / final | final / ValueError / final |
| '2027-3-4' | 2027 | artifact / no artifact / artifact | final / ValueError / final | final / ValueError / final | final / ValueError / final |
| ' 2027-03-24' | ValueError | no artifact / no artifact / no artifact | final / ValueError / ValueError | final / ValueError / ValueError | final / ValueError / ValueError |
| '2027-03-24 ' | ValueError | no artifact / no artifact / no artifact | final / ValueError / ValueError | final / ValueError / ValueError | final / ValueError / ValueError |
| ' 2027-03-24 ' | ValueError | no artifact / no artifact / no artifact | final / ValueError / ValueError | final / ValueError / ValueError | final / ValueError / ValueError |
| '2027-03-24\n' | ValueError | no artifact / no artifact / no artifact | final / ValueError / ValueError | final / ValueError / ValueError | final / ValueError / ValueError |
| '2027-03-24T00:00:00' | ValueError | no artifact / no artifact / no artifact | final / ValueError / ValueError | final / ValueError / ValueError | final / ValueError / ValueError |
| '2027-03-24 00:00:00' | ValueError | no artifact / no artifact / no artifact | final / ValueError / ValueError | final / ValueError / ValueError | final / ValueError / ValueError |
| '2027/03/24' | ValueError | no artifact / no artifact / no artifact | final / ValueError / ValueError | final / ValueError / ValueError | final / ValueError / ValueError |
| '2027.03.24' | ValueError | no artifact / no artifact / no artifact | final / ValueError / ValueError | final / ValueError / ValueError | final / ValueError / ValueError |
| '2027_03_24' | ValueError | no artifact / no artifact / no artifact | final / ValueError / ValueError | final / ValueError / ValueError | final / ValueError / ValueError |
| '20270324' | ValueError | no artifact / no artifact / no artifact | final / final / ValueError | final / final / ValueError | final / final / ValueError |
| '2027-W12-3' | ValueError | no artifact / no artifact / no artifact | final / final / ValueError | final / final / ValueError | final / final / ValueError |
| '2027-02-30' | ValueError | no artifact / no artifact / no artifact | final / ValueError / ValueError | final / ValueError / ValueError | final / ValueError / ValueError |
| 'nonsense' | ValueError | no artifact / no artifact / no artifact | final / ValueError / ValueError | final / ValueError / ValueError | final / ValueError / ValueError |
| '' | ValueError | no artifact / no artifact / no artifact | final / ValueError / ValueError | final / ValueError / ValueError | final / ValueError / ValueError |
| '2027-03- 4' | 2027 | artifact / no artifact / artifact | final / ValueError / final | final / ValueError / final | final / ValueError / final |
| '2027-٣-24' | ValueError | no artifact / no artifact / no artifact | final / ValueError / ValueError | final / ValueError / ValueError | final / ValueError / ValueError |
| datetime.date(2027, 3, 24) | TypeError | TypeError / TypeError / TypeError | final / TypeError / TypeError | final / TypeError / TypeError | final / TypeError / TypeError |
| None | TypeError | TypeError / TypeError / TypeError | final / TypeError / TypeError | final / TypeError / TypeError | final / TypeError / TypeError |

All 616 entry-point events are retained in {base,r2,head,fixed}-probe.json.
The fixed scratch package tries date.fromisoformat first and strptime as a ValueError fallback.
No base-rejected shadow input gained an artifact at head. Display probes never mutate streak.json.
CLI unpadded dates fail its pre-existing age parser before resolution or streak reads at all three revisions.
Unpadded strategy succeeds at all three revisions. Unpadded update_streak updates at base, but rejects before reads/writes at r2/head, as required.
None remains the documented dateless update_streak operation; it updates at all revisions.
Initial probe attempts failed while inspecting non-string scheduler state, before JSON export; attempt2 logs are the complete runs.
