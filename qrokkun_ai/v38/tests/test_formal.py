from qrokkun_ai.v37.run import parser, validate

def test_formal_control_default_role():
    a = parser().parse_args(['train','--out-dir','unused','--source-commit','a'*40])
    assert a.role == 'control'
    assert validate(a)[1] == list(range(0,1001,10))

import copy
import json
import pytest
from qrokkun_ai.v38 import protocol, final, supervise

@pytest.fixture
def documents(monkeypatch):
    schedule=list(range(0,1001,10))
    s=dict(role='control',schema='pv37_manifest_v2',nonformal=False,formal_authorized=True,updates=1000,episodes=20,cap=7200,seed=17,hidden=472,validation=16,checkpoint_boundaries=schedule,validation_boundaries=schedule,source_tree_sha='sha')
    c=dict(formal=True,completed_updates=1000,latest_validated_update=1000,saved_checkpoints=schedule,manifest_sha='sha',source_tree_sha='sha',best_update=0,checkpoint_hashes={f'checkpoint-{u:06d}.pt':'sha' for u in schedule})
    docs={'manifest.json':dict(arm='Pv37m',semantic=s),'closed.json':c,'status.json':dict(status='complete',completed_updates=1000,target_updates=1000)}
    for label,u in [('initial',0),('best',0),('terminal',1000),('latest',1000)]:
        docs[label+'.json']=dict(update=u,checkpoint=f'checkpoint-{u:06d}.pt')
    for u in schedule:
        docs[f'evaluation-{u:06d}.json']=dict(update=u,total_frames=16,episodes=[dict(seed=x,frames=1,cap_frames=7200,death=True,censored=False) for x in range(600000,600016)])
    monkeypatch.setattr(protocol,'read',lambda out,name:copy.deepcopy(docs[name]))
    monkeypatch.setattr(protocol,'file_hash',lambda p:'sha')
    monkeypatch.setattr(protocol,'source_digest',lambda:'sha')
    return docs

def test_complete_closure(documents):
    assert protocol.closure('memory')[1]['best_update']==0

def test_requires_terminal(documents):
    documents['closed.json']['completed_updates']=2
    with pytest.raises(ValueError): protocol.closure('memory')

def test_hash_coverage(documents):
    documents['closed.json']['checkpoint_hashes'].pop('checkpoint-000010.pt')
    with pytest.raises(ValueError,match='coverage'): protocol.closure('memory')

def test_validation_coverage(documents):
    documents['evaluation-000010.json']['episodes'].pop()
    with pytest.raises(ValueError,match='validation'): protocol.closure('memory')

def test_earliest_best(documents):
    documents['closed.json']['best_update']=10
    with pytest.raises(ValueError,match='earliest'): protocol.closure('memory')

def test_stop_during_scan(documents):
    with pytest.raises(InterruptedError): protocol.closure('memory',lambda:True)

def test_wrong_arm(documents):
    documents['manifest.json']['arm']='Pv38m'
    with pytest.raises(ValueError): protocol.closure('memory')

def test_source_mismatch(documents):
    documents['manifest.json']['semantic']['source_tree_sha']='other'
    with pytest.raises(ValueError,match='source'): protocol.closure('memory')

def test_formal_synthetic_rejected():
    a=parser().parse_args(['train','--out-dir','unused','--source-commit','a'*40,'--synthetic'])
    with pytest.raises(ValueError,match='synthetic'): validate(a)

def test_search_entry_rejects_control():
    from qrokkun_ai.v38.run import main
    with pytest.raises(ValueError,match='search role'):
        main(['train','--out-dir','unused','--source-commit','a'*40,'--role','control'])

def test_paired_correct_direction():
    rows=[dict(seed=s,frames=60) for s in protocol.FINAL_SEEDS]
    other=[dict(seed=s,frames=120) for s in protocol.FINAL_SEEDS]
    assert protocol.paired(other,rows)['mean_seconds_difference']==1

def test_preflight_scoped_digest(monkeypatch):
    from qrokkun_ai.v31 import core
    from qrokkun_ai.v37.fingerprint import source_digest
    monkeypatch.setattr(final,'_preflight',lambda *a:core.source_digest())
    assert final.preflight(None)==source_digest()

def test_claim_competitor_no_clobber(tmp_path,monkeypatch):
    from types import SimpleNamespace
    (tmp_path/'final-claimed.json').write_text('{"owner":1}')
    (tmp_path/'final-status.json').write_text('{"status":"complete"}')
    monkeypatch.setattr(final,'preflight',lambda *args:({},{}))
    with pytest.raises(FileExistsError):
        final._execute(SimpleNamespace(out_dir=str(tmp_path),source_commit='a'*40))
    assert json.loads((tmp_path/'final-claimed.json').read_text())==dict(owner=1)
    assert json.loads((tmp_path/'final-status.json').read_text())==dict(status='complete')

def test_supervisor_order_and_failure(tmp_path,monkeypatch):
    calls=[]
    class Child:
        pid=123
        def __init__(self,args,**kw):
            calls.append(args[-1]); self.returncode=1 if args[-1]=='search-train' else 0
        def poll(self):return self.returncode
        def wait(self,**kw):return self.returncode
    monkeypatch.setattr(supervise.subprocess,'Popen',Child)
    monkeypatch.setattr(supervise,'preflight',lambda *a:None)
    a=parser().parse_args(['train','--out-dir',str(tmp_path/'pair'),'--source-commit','a'*40])
    assert supervise.execute(a,child_argv=lambda s:['fake',s])==1
    assert calls==['control-train','control-final','search-train']
    status=json.loads((tmp_path/'pair/supervisor.json').read_text())
    assert status['errors'] and len(status['child_exits'])==3


def test_final_cap_death_and_survival():
    for death in (True, False):
        row=dict(frames=7200,cap_frames=7200,death=death,censored=not death)
        protocol.validate_record(row)
        assert row['death'] is death and row['censored'] is not death
    protocol.validate_record(dict(frames=1,cap_frames=7200,death=True,censored=False))


def test_final_inconsistent_records():
    for changes in (dict(death=True,censored=True), dict(death=False,censored=False),
                    dict(frames=7199), dict(frames=0), dict(frames=7201)):
        row=dict(frames=7200,cap_frames=7200,death=False,censored=True)
        row.update(changes)
        with pytest.raises(ValueError):
            protocol.validate_record(row)


def test_control_pin_consumed_set(documents,monkeypatch):
    import torch
    from qrokkun_ai.v21.agents import player_v21
    from qrokkun_ai.v35 import initialization, model
    class Net:
        def load_state_dict(self,*args,**kwargs): pass
        def state_dict(self): return {}
    monkeypatch.setattr(model.PlayerV35,'from_metadata',lambda _:Net())
    monkeypatch.setattr(initialization,'random_player',lambda:Net())
    monkeypatch.setattr(player_v21,'state_dict_sha256',lambda _:'state')
    monkeypatch.setattr(torch,'load',lambda path,**kwargs:dict(architecture={},model={},arm='Pv37m',schema='pv37_checkpoint_v2',update=int(path.stem.split('-')[1]),source_tree_sha='sha'))
    documents['manifest.json']['semantic']['initial_player_state_sha']='state'
    results={}
    for label,u in [('best',0),('terminal',1000)]:
        r=dict(state_sha='state',update=u,file_sha='sha',player_mode='argmax',spawner_mode='stochastic',episodes=[dict(seed=s,frames=7200,cap_frames=7200,death=True,censored=False,state_sha='state') for s in protocol.FINAL_SEEDS])
        results[label]=r
        documents['final-'+label+'.json']=r
    documents['final.json']=dict(status='complete',results=results,spawner_offset=24000000)
    documents['final-status.json']=dict(status='complete',errors=[],owner_stop=[])
    schedule=documents['manifest.json']['semantic']['checkpoint_boundaries']
    expected={'manifest.json','closed.json','status.json','final.json','final-status.json','final-best.json','final-terminal.json','updates.jsonl'}
    expected.update(label+'.json' for label in ('initial','best','latest','terminal'))
    expected.update(f'{kind}-{u:06d}.{suffix}' for u in schedule for kind,suffix in [('checkpoint','pt'),('evaluation','json')])
    assert set(protocol.control_pin('memory'))==expected
