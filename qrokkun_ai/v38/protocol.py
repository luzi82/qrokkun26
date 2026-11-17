"""Formal closure checks and frozen control pinning."""
import json
from pathlib import Path
from qrokkun_ai.v24.core import file_hash
from qrokkun_ai.v35.protocol import FINAL_SEEDS, paired
from qrokkun_ai.v37.fingerprint import source_digest

def read(out, name):
    return json.loads((Path(out)/name).read_text())

def check_stop(stop):
    if stop():
        raise InterruptedError('owner stop')

def validate_record(row):
    """Death wins on the cap frame; only surviving to the cap is censored."""
    frames, death, censored = row['frames'], row['death'], row['censored']
    if (type(frames) is not int or not 0 < frames <= 7200
        or row['cap_frames'] != 7200 or type(death) is not bool
        or type(censored) is not bool or death == censored
        or (censored and frames != 7200)):
        raise ValueError('invalid episode frames/death/censored/cap')

def closure(out, stop=lambda: False):
    out = Path(out)
    m,c,t = [read(out,n) for n in ('manifest.json','closed.json','status.json')]
    s=m['semantic']
    version = '37' if s['role']=='control' else '38'
    if (m['arm'] != 'Pv'+version+'m' or s['schema'] != ('pv37_manifest_v2' if version=='37' else 'pv38_manifest_v1')
        or s['nonformal'] or not s['formal_authorized'] or not c['formal']
        or t['status']!='complete' or t.get('errors') or t.get('owner_stop')
        or c['completed_updates']!=1000 or c['latest_validated_update']!=1000
        or t['completed_updates']!=1000 or t['target_updates']!=1000):
        raise ValueError('complete authorized formal closure required')
    for k,v in dict(updates=1000,episodes=20,cap=7200,seed=17,hidden=472,validation=16).items():
        if s[k]!=v: raise ValueError('formal config mismatch: '+k)
    schedule=list(range(0,1001,10))
    if s['checkpoint_boundaries']!=schedule or s['validation_boundaries']!=schedule or c['saved_checkpoints']!=schedule:
        raise ValueError('101 checkpoint/validation schedule required')
    if c['manifest_sha']!=file_hash(out/'manifest.json') or s['source_tree_sha']!=source_digest() or c['source_tree_sha']!=s['source_tree_sha']:
        raise ValueError('source/manifest mismatch')
    names={f'checkpoint-{u:06d}.pt' for u in schedule}
    if set(c['checkpoint_hashes'])!=names: raise ValueError('checkpoint pin coverage')
    for name,sha in c['checkpoint_hashes'].items():
        check_stop(stop)
        if file_hash(out/name)!=sha: raise ValueError('checkpoint hash mismatch')
    totals=[]
    for u in schedule:
        check_stop(stop)
        e=read(out,f'evaluation-{u:06d}.json')
        rows=e['episodes']
        if e['update']!=u or [r['seed'] for r in rows]!=list(range(600000,600016)) or e['total_frames']!=sum(r['frames'] for r in rows):
            raise ValueError('validation records incomplete')
        for row in rows:
            validate_record(row)
        totals.append((u,e['total_frames']))
    if max(totals,key=lambda x:x[1])[0]!=c['best_update']: raise ValueError('earliest best mismatch')
    for name,u in [('initial',0),('best',c['best_update']),('terminal',1000),('latest',1000)]:
        if read(out,name+'.json')!={'update':u,'checkpoint':f'checkpoint-{u:06d}.pt'}: raise ValueError('pointer mismatch')
    check_stop(stop)
    return m,c

def control_pin(out, stop=lambda: False):
    m,c=closure(out,stop)
    if m['semantic']['role']!='control' or m['arm']!='Pv37m': raise ValueError('Pv37 control required')
    import torch
    from qrokkun_ai.v21.agents.player_v21 import state_dict_sha256
    from qrokkun_ai.v35.initialization import random_player
    from qrokkun_ai.v35.model import PlayerV35
    states={}
    for u in sorted({0,c['best_update'],1000}):
        check_stop(stop)
        payload=torch.load(Path(out)/f'checkpoint-{u:06d}.pt',map_location='cpu',weights_only=False)
        net=PlayerV35.from_metadata(payload['architecture'])
        net.load_state_dict(payload['model'],strict=True)
        states[u]=state_dict_sha256(net.state_dict())
        if payload['arm']!='Pv37m' or payload['schema']!='pv37_checkpoint_v2' or payload['update']!=u or payload['source_tree_sha']!=m['semantic']['source_tree_sha']:
            raise ValueError('loaded control checkpoint identity mismatch')
    if states[0]!=m['semantic']['initial_player_state_sha'] or states[0]!=state_dict_sha256(random_player().state_dict()):
        raise ValueError('untrained initialization required')
    f=read(out,'final.json'); status=read(out,'final-status.json')
    if f['status']!='complete' or f['spawner_offset']!=24000000 or status['status']!='complete' or status['errors'] or status['owner_stop']: raise ValueError('completed control final required')
    for label,u in [('best',c['best_update']),('terminal',1000)]:
        check_stop(stop)
        r=f['results'][label]
        if read(out,'final-'+label+'.json') != r:
            raise ValueError('control saved per-seed result mismatch')
        for row in r['episodes']:
            validate_record(row)
        if r['player_mode']!='argmax' or r['spawner_mode']!='stochastic':
            raise ValueError('control final mode mismatch')
        if r['state_sha']!=states[u] or r['update']!=u or r['file_sha']!=c['checkpoint_hashes'][f'checkpoint-{u:06d}.pt'] or [x['seed'] for x in r['episodes']]!=FINAL_SEEDS or any(x['state_sha']!=states[u] for x in r['episodes']):
            raise ValueError('control final identity mismatch')
    check_stop(stop)
    names = {'manifest.json','closed.json','status.json','final.json','final-status.json',
             'final-best.json','final-terminal.json','updates.jsonl',
             'initial.json','best.json','latest.json','terminal.json'}
    names.update(c['checkpoint_hashes'])
    names.update(f'evaluation-{u:06d}.json' for u in m['semantic']['validation_boundaries'])
    pin = {}
    for name in sorted(names):
        check_stop(stop)
        pin[name] = file_hash(Path(out)/name)
    return pin
