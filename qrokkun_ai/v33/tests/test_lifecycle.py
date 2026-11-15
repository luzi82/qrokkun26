"""Small in-process wiring fixtures; no real optimizer training or subprocess pool."""
import json
import shutil
from types import SimpleNamespace
import pytest
from qrokkun_ai.v33 import run

@pytest.mark.parametrize('fail',[False,True,'cleanup'])
def test_train_wiring_and_cleanup(tmp_path,monkeypatch,fail):
    calls=[]
    class Pool:
        pinned_cpus=[]
        last_pure_rollout_s=0.
        def __init__(self,**kw):
            self.kw=kw
            self.stop_event=SimpleNamespace(set=lambda:None)
            calls.append('pool')
        def sync_weights(self,*a,**kw): calls.append('sync')
        def collect(self,seeds,cap,**kw):
            calls.append(('validation' if kw['validation'] else 'training',list(seeds)))
            return [SimpleNamespace(frames=2,elapsed=2/60,summary=lambda seed=s:dict(seed=seed,frames=2,death=False,censored=True)) for s in seeds]
        def shutdown_and_stats(self): calls.append('shutdown'); return []
        def close(self):
            calls.append('close')
            if fail == 'cleanup': raise OSError('observed cleanup failure')
    monkeypatch.setattr(run,'EpisodePool',Pool)
    monkeypatch.setattr(run.core,'assemble_batch',lambda *a:None)
    def train(*a,**kw):
        calls.append('ppo')
        if fail: raise RuntimeError('observed learner failure')
        return {}
    monkeypatch.setattr(run.core,'train',train)
    def disk(path):
        calls.append('disk'); return SimpleNamespace(free=10**12)
    monkeypatch.setattr(shutil,'disk_usage',disk)
    argv=['train','--out-dir',str(tmp_path/'run'),'--source-commit','a'*40,'--nonformal','--synthetic','--hidden','4','--cap','2','--updates','1','--episodes','1','--validation','1','--workers','1','--device','cpu','--pin-cores','off']
    if fail:
        with pytest.raises(RuntimeError,match='learner failure'): run.main(argv)
        assert 'close' in calls
    else:
        assert run.main(argv)==0
        assert 'shutdown' in calls
        c=json.loads((tmp_path/'run/closed.json').read_text())
        assert c['best_update']==0 and c['latest_validated_update']==1
        assert len(c['saved_checkpoints'])==2
        assert calls.count('disk')==4 # startup, post-pool guard, u0, u1; never per frame
    assert ('validation',[600000]) in calls and ('training',[100000]) in calls
    status=json.loads((tmp_path/'run/status.json').read_text())
    assert status['status']==('failed' if fail else 'complete')
    if fail: assert 'observed learner failure' in str(status['errors'])
    if fail == 'cleanup':
        assert 'observed cleanup failure' in str(status['errors'])
        assert not (tmp_path/'run/closed.json').exists()

def test_final_two_arms_only(tmp_path,monkeypatch):
    import torch
    from qrokkun_ai.v33 import final
    from qrokkun_ai.v33.initialization import random_player
    from qrokkun_ai.v24.core import file_hash
    out=tmp_path/'run'; out.mkdir()
    sf=tmp_path/'s'; sf.write_text('fixture')
    p=random_player(4)
    for u in (0,1000):
        torch.save(dict(schema='pv33_checkpoint_v1',update=u,model=p.state_dict()),out/run.ckpt_name(u))
    for u in range(0,1001,10):
        (out/run.eval_name(u)).write_text('{}')
    rows=[dict(seed=s,frames=60) for s in range(800000,800064)]
    warm=dict(identity={'fixture':True},results={k:dict(episodes=rows) for k in ('original','best','terminal')},curve=[],timings={})
    m={'semantic':{'inputs':dict(spawner=file_hash(sf)),'warm_reference':warm['identity']}}
    monkeypatch.setattr(final,'preflight',lambda *a:(m,dict(best_update=0,timings={})))
    monkeypatch.setattr(final,'load_reference',lambda *a:warm)
    monkeypatch.setattr(final,'locked_spawner',lambda *a:None)
    windows=[]
    class Pool:
        def __init__(self,*a): self.stop_event=SimpleNamespace(set=lambda:None)
        def sync_weights(self,*a,**kw): pass
        def collect(self,seeds,*a,**kw):
            windows.extend(seeds)
            return [SimpleNamespace(summary=lambda seed=s:dict(seed=seed,frames=120,death=True,censored=False)) for s in seeds]
        def shutdown_and_stats(self): return []
        def close(self): pass
    monkeypatch.setattr(final,'EpisodePool',Pool)
    a=run.parser().parse_args(['final','--out-dir',str(out),'--source-commit','a'*40,'--spawner-checkpoint',str(sf)])
    assert final.execute(a)==0
    assert windows==list(range(800000,800064))*2
    result=json.loads((out/'final.json').read_text())
    assert set(result['results'])=={'best','terminal'}
    assert result['primary']['ci95']==[1,1]
    with pytest.raises(FileExistsError): final.execute(a)
