import json
import sys
import pytest
import torch
from qrokkun_ai.v22 import run
from test_core import fixture_player

@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_deadlines(value,tmp_path,monkeypatch):
    with pytest.raises(ValueError): run.Guard(value)
    monkeypatch.setattr(sys,'argv',['sv22','train','--out-dir',str(tmp_path),'--train-deadline-unix='+str(value)])
    with pytest.raises(ValueError): run.main()

@pytest.mark.parametrize('corrupt',[float('nan'),float('inf'),run.HARD+100])
def test_hard_independent(corrupt,monkeypatch):
    g=run.Guard(run.HARD); g.deadline=corrupt
    monkeypatch.setattr(run.time,'time',lambda:run.HARD)
    with pytest.raises(run.Stopped): g.check()

def setup_run(tmp_path,monkeypatch,extra=()):
    out=tmp_path/'run'
    monkeypatch.setattr(run,'load_player',lambda *a:(fixture_player(),{}))
    def invoke(command,*flags):
        monkeypatch.setattr(sys,'argv',['sv22',command,'--out-dir',str(out),'--updates','1',*extra,*flags]); run.main()
    return out,invoke

def records(*args,**kw):
    return [dict(seed=s,seconds=2.,frames=120,death=False,censored=True) for s in args[2]],[( [0.]*325,[0.]*5,0.,-.1)]

def test_formal_counts(tmp_path,monkeypatch):
    out,invoke=setup_run(tmp_path,monkeypatch)
    with pytest.raises(ValueError): invoke('train','--eval-seeds','1')
    assert not out.exists()

def test_one_pair_never_evidence():
    a=[dict(seed=1,seconds=1.)]; b=[dict(seed=1,seconds=2.)]
    assert run.compare(a,b)['interpretation']=='inconclusive'

@pytest.mark.usefixtures("synthetic_deadlines")
def test_primary_survives_diagnostic(tmp_path,monkeypatch):
    out,invoke=setup_run(tmp_path,monkeypatch,('--quick','--eval-seeds','1','--final-seeds','1'))
    monkeypatch.setattr(run,'episodes',records)
    invoke('train')
    def final(*args,**kw):
        if args[6]=='deterministic':
            result=json.loads((out/'final.json').read_text())
            assert result['conclusion']=='inconclusive'
            raise run.Stopped('diagnostic')
        return records(*args,**kw)
    monkeypatch.setattr(run,'episodes',final)
    invoke('final')
    assert (out/'final.json').exists()
    assert (out/'final-diagnostic-incomplete.json').exists()
    assert not (out/'final-incomplete.json').exists()
    with pytest.raises(FileExistsError): invoke('final')

@pytest.mark.usefixtures("synthetic_deadlines")
def test_live_validate_and_progress(tmp_path,monkeypatch):
    out,invoke=setup_run(tmp_path,monkeypatch,('--quick',))
    states=[]
    def capture(*args,**kw):
        states.append(json.loads((out/'status.json').read_text())['status'])
        return records(*args,**kw)
    monkeypatch.setattr(run,'episodes',capture)
    invoke('train')
    progress=[json.loads(s) for s in (out/'progress.jsonl').read_text().splitlines()]
    assert len(progress)==1 and progress[0]['update']==1
    assert {'wallclock','elapsed','loss','rollout','validation'}<=progress[0].keys()
    assert {'baseline','rollout','validation'}<=set(states)
    (out/'training-closed.json').unlink()
    invoke('validate')
    assert json.loads((out/'validate-status.json').read_text())['update']==1
    with pytest.raises(FileNotFoundError): invoke('final')

@pytest.mark.usefixtures("synthetic_deadlines")
def test_partial_ppo_not_completed(tmp_path,monkeypatch):
    out,invoke=setup_run(tmp_path,monkeypatch,('--quick',))
    monkeypatch.setattr(run,'episodes',records)
    monkeypatch.setattr(run,'ppo',lambda *a: (_ for _ in ()).throw(run.Stopped('mid-ppo')))
    invoke('train')
    assert torch.load(out/'latest.pt',weights_only=False)['update']==0
    assert not (out/'progress.jsonl').exists() or not (out/'progress.jsonl').read_text()

@pytest.mark.usefixtures("synthetic_deadlines")
def test_final_hash_lock_before_evaluation(tmp_path,monkeypatch):
    out,invoke=setup_run(tmp_path,monkeypatch,('--quick',))
    monkeypatch.setattr(run,'episodes',records)
    invoke('train')
    with (out/'best.pt').open('ab') as f: f.write(b'changed')
    monkeypatch.setattr(run,'episodes',lambda *a,**kw: pytest.fail('evaluation before hash lock'))
    with pytest.raises(ValueError,match='locked best'): invoke('final')
    assert not (out/'final-claimed.json').exists()

@pytest.mark.usefixtures("synthetic_deadlines")
def test_nonformal_manifest_blocks_negative_ci_claim(tmp_path,monkeypatch):
    out,invoke=setup_run(tmp_path,monkeypatch,('--nonformal','--eval-seeds','2','--final-seeds','2'))
    monkeypatch.setattr(run,'episodes',records)
    invoke('train')
    count=0
    def negative(*args,**kw):
        nonlocal count
        count+=1
        r,d=records(*args,**kw)
        for row in r: row['seconds']=1. if count==3 else 2.
        return r,d
    monkeypatch.setattr(run,'episodes',negative)
    invoke('final')
    result=json.loads((out/'final.json').read_text())
    assert result['comparisons']['initial']['ci95'][1]<0
    assert result['conclusion']=='inconclusive'
    assert all(c['interpretation']=='inconclusive' for c in result['comparisons'].values())
