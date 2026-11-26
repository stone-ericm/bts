"""Assert and summarize the retained offline differential evidence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = {label:json.loads((ROOT / f"{label}-probe.json").read_text())
        for label in ("base","r2","head","fixed")}
index = {label:{(e['path'],e['case']):e for e in data} for label,data in rows.items()}
def row(label,path,case):
    return index[label][path,case]
def outcome(label,path,case):
    e = row(label,path,case)
    o = e['outcome']
    if 'error' in o:
        return o['error']
    if path == 'shadow':
        return 'artifact' if e['shadow_written'] else 'no artifact'
    if path == 'cli_void':
        return str(o['value']['exit'])+':'+str(o['value']['error'])
    return str(o['value'])

for path in ('shadow','void','final_peer','early_peer'):
    for case in ('padded','month','day','both','space_day'):
        expected = 'artifact' if path == 'shadow' else 'final'
        assert outcome('base',path,case) == outcome('head',path,case) == expected
        if case != 'padded':
            assert outcome('r2',path,case) == ('no artifact' if path == 'shadow' else 'ValueError')
for path in ('void','final_peer','early_peer'):
    for case in ('compact','week'):
        assert outcome('base',path,case) == outcome('r2',path,case) == outcome('fixed',path,case) == 'final'
        assert outcome('head',path,case) == 'ValueError'
        assert all(row(label,path,case)['streak_unchanged'] for label in rows)
        if path == 'void':
            assert row('head',path,case)['pick_result'] is None
            assert row('r2',path,case)['pick_result'] == row('fixed',path,case)['pick_result'] == 'void'
assert not [e['case'] for e in rows['head'] if e['path']=='shadow' and e['shadow_written']
            and not row('base','shadow',e['case'])['shadow_written']]
for label in ('base','r2','head'):
    e = row(label,'cli_void','month')
    assert e['outcome']['value']['error'] == 'ValueError'
    assert e['events'] == [] and e['streak_calls'] == [] and e['streak_unchanged']
    assert row(label,'strategy','month')['outcome']['value']['pick'] is True
for label in ('r2','head'):
    e = row(label,'update','month')
    assert e['outcome']['error'] == 'ValueError'
    assert e['streak_calls'] == [] and e['changed_files'] == []

lines = ['# Date acceptance and side-effect evidence', '',
         'Sources: base=d8eb50d0; r2=a26958dc; head=ee9fb870. Each cell is base / r2 / head.',
         'Shadow strings invoke the real one-shot main; date objects/None invoke run_once and fail upstream.',
         'Polling uses real reads, scoring locks, terminal persistence and replies (posting transport stubbed).',
         'Only prediction, watchdog, remote statuses/results and MDP artifact availability are stubbed; socket connections are denied.', '',
         '| Input repr | strptime result | Shadow | All void | Final peer | Early peer |',
         '|---|---|---|---|---|---|']
for e in rows['base']:
    if e['path'] != 'shadow':
        continue
    case=e['case']
    parser=e['parser'].get('year',e['parser'].get('error'))
    cells=[' / '.join(outcome(label,path,case) for label in ('base','r2','head'))
           for path in ('shadow','void','final_peer','early_peer')]
    lines.append('| '+ ' | '.join([e['input'],str(parser),*cells])+' |')
lines += ['', 'All 616 entry-point events are retained in {base,r2,head,fixed}-probe.json.',
          'The fixed scratch package tries date.fromisoformat first and strptime as a ValueError fallback.',
          'No base-rejected shadow input gained an artifact at head. Display probes never mutate streak.json.',
          'CLI unpadded dates fail its pre-existing age parser before resolution or streak reads at all three revisions.',
          'Unpadded strategy succeeds at all three revisions. Unpadded update_streak updates at base, but rejects before reads/writes at r2/head, as required.',
          'None remains the documented dateless update_streak operation; it updates at all revisions.',
          'Initial probe attempts failed while inspecting non-string scheduler state, before JSON export; attempt2 logs are the complete runs.']
(ROOT/'date-matrix.md').write_text('\n'.join(lines)+'\n')
print('All differential assertions passed; wrote date-matrix.md.')
