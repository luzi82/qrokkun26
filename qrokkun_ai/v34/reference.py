"""Read only, pinned completed control reference. Never executes benchmark episodes."""
import json
from pathlib import Path
import torch
from qrokkun_ai.v24.core import file_hash
from qrokkun_ai.v21.agents.player_v21 import state_dict_sha256
from qrokkun_ai.v26.run import HYPER, S_STATE
from qrokkun_ai.v25.core import S_SHA, VALIDATION
from qrokkun_ai.v31 import core
from qrokkun_ai.v33.protocol import closure_valid
from .protocol import FINAL_SEEDS, schedule

PIN_SHA = 'a1c67f116b31ef334b0cf06027b10792fc089f76c0606c969dad2aa7f031480b'
_pin_path = Path(__file__).with_name('control-reference.json')
if file_hash(_pin_path) != PIN_SHA:
    raise ValueError('pinned reference contract changed')
PIN = json.loads(_pin_path.read_text())


def load_reference(directory, stop=lambda: False):
    directory = Path(directory)
    hashes = dict(PIN['file_hashes'])
    def read(name):
        if stop():
            raise InterruptedError("owner stop reading control reference")
        path=directory/name
        sha=file_hash(path)
        if name in hashes and hashes[name] != sha:
            raise ValueError('control reference hash mismatch: '+name)
        hashes[name]=sha
        return json.loads(path.read_text())
    for name, expected_sha in PIN['file_hashes'].items():
        if stop():
            raise InterruptedError('owner stop')
        if file_hash(directory/name) != expected_sha:
            raise ValueError('control reference hash mismatch: '+name)
    m,c,status=read('manifest.json'),read('closed.json'),read('status.json')
    s=m['semantic']
    if not closure_valid(m,c,status) or c['best_update'] != 910 or s['source_commit'] != PIN['training_source_commit']:
        raise ValueError('control closure identity mismatch')
    expected=dict(episodes=20,cap=7200,seed=17,hidden=472,validation=16,
        hyperparameters=HYPER,P_offset=core.P_OFFSET,S_train_offset=core.S_TRAIN_OFFSET,
        S_validation_offset=core.S_VAL_OFFSET,permutation_seed=27000017,
        spawner_state_sha=S_STATE,player_mode_training='stochastic',player_mode_evaluation='argmax',
        spawner_mode='stochastic',validation_seeds=list(VALIDATION),final_seeds=FINAL_SEEDS,
        checkpoint_boundaries=schedule(),validation_boundaries=schedule(),checkpoint_cadence=10,validation_cadence=10)
    if any(s.get(k)!=v for k,v in expected.items()) or s['inputs']['spawner'] != S_SHA:
        raise ValueError('control matched protocol mismatch')
    learner=s['runtime']['learner']
    if (s['runtime']['rollout'] != dict(device='cpu',threads=1,batch=1) or
        learner.get('ppo_backend')!='cuda_hoisted_single_stream' or not learner.get('hoisted_features') or
        not learner.get('deferred_host_sync') or learner.get('cuda_matmul_allow_tf32') or learner.get('cudnn_allow_tf32') or
        m.get('workers')!=10 or m.get('pin_cores')!='on'):
        raise ValueError('control runtime mismatch')
    if c['manifest_sha'] != hashes['manifest.json'] or c['saved_checkpoints'] != schedule():
        raise ValueError('control checkpoint closure mismatch')
    curve=[]
    for u in schedule():
        e=read(f'evaluation-{u:06d}.json')
        rows=e['episodes']
        if e['update']!=u or [r['seed'] for r in rows]!=list(VALIDATION) or e['total_frames']!=sum(r['frames'] for r in rows):
            raise ValueError('control validation mismatch')
        curve.append(dict(update=u,total_frames=e['total_frames'],wall_s=e['wall_s']))
    if max(curve,key=lambda r:r['total_frames'])['update']!=910:
        raise ValueError('control best selection mismatch')
    final=read('final.json'); fs=read('final-status.json')
    if final['status']!='complete' or fs['status']!='complete' or fs.get('errors') or fs.get('owner_stop'):
        raise ValueError('control final incomplete')
    for name,u in [('best',910),('terminal',1000)]:
        result=read('final-'+name+'.json')
        if final['results'][name]!=result or result['update']!=u or [r['seed'] for r in result['episodes']]!=FINAL_SEEDS:
            raise ValueError('control final records mismatch')
        if result['player_mode']!='argmax' or result['spawner_mode']!='stochastic':
            raise ValueError('control final modes mismatch')
        for r in result['episodes']:
            if type(r['frames']) is not int or not 0 < r['frames'] <= 7200 or r['cap_frames']!=7200 or r['death']==r['censored']:
                raise ValueError('control incomplete episode')
        if name!='original':
            path=directory/f'checkpoint-{u:06d}.pt'
            sha=file_hash(path); hashes[path.name]=sha
            payload=torch.load(path,map_location='cpu',weights_only=False)
            if (sha!=c['checkpoint_hashes'][path.name] or sha!=result['file_sha'] or
                payload['update']!=u or payload['schema']!='pv33_checkpoint_v1' or
                state_dict_sha256(payload['model'])!=result['state_sha']):
                raise ValueError('control selected state identity mismatch')
    u0=torch.load(directory/'checkpoint-000000.pt',map_location='cpu',weights_only=False)
    if (u0['schema'] != 'pv33_checkpoint_v1' or u0['update'] != 0 or
        state_dict_sha256(u0['model']) != '49a7c9426a1054fc000caf359fc1a0a3c90db49adfe6a53c3275350a9b5a7942' or
        u0['actor']['state'] or u0['critic']['state']):
        raise ValueError('control u0/fresh Adam mismatch')
    updates=directory/'updates.jsonl' ; hashes[updates.name]=file_hash(updates)
    records=[json.loads(line) for line in updates.read_text().splitlines()]
    if [r['update'] for r in records]!=list(range(1,1001)):
        raise ValueError('control updates incomplete')
    return dict(identity=dict(directory=str(directory),file_hashes=hashes,pinned=PIN,pin_sha=PIN_SHA),curve=curve,
        results=final['results'],timings=c['timings'])
