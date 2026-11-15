"""Small in-process wiring fixtures; no real optimizer training or subprocess pool."""
import json
import shutil
from types import SimpleNamespace
import pytest
from qrokkun_ai.v32 import run

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

def test_final_weight_sync_shared_window(tmp_path,monkeypatch):
    import torch
    from qrokkun_ai.v32 import final
    from qrokkun_ai.v21.agents.player_v21 import PlayerV21
    from qrokkun_ai.v24.core import file_hash
    out=tmp_path/'run'; out.mkdir()
    pfile=tmp_path/'p'; sfile=tmp_path/'s'; pfile.write_text('p'); sfile.write_text('s')
    p=PlayerV21(hidden=4)
    for u in (0,1000):
        torch.save(dict(schema='pv32_checkpoint_v1',update=u,model=p.state_dict()),out/run.ckpt_name(u))
    m={'semantic':{'inputs':dict(player=file_hash(pfile),spawner=file_hash(sfile))}}
    monkeypatch.setattr(final,'preflight',lambda *a:(m,dict(best_update=0)))
    monkeypatch.setattr(final,'inputs',lambda *a:(p,None,{}))
    syncs=[]; windows=[]; shutdown=[]
    class Pool:
        def __init__(self,*a): self.stop_event=SimpleNamespace(set=lambda:None)
        def sync_weights(self,sd,policy_version,**kw): syncs.append(policy_version)
        def collect(self,seeds,*a,**kw):
            windows.extend(seeds)
            return [SimpleNamespace(summary=lambda seed=s:dict(seed=seed,frames=60,death=True,censored=False)) for s in seeds]
        def shutdown_and_stats(self): shutdown.append(True)
        def close(self): shutdown.append(True)
    monkeypatch.setattr(final,'EpisodePool',Pool)
    a=run.parser().parse_args(['final','--out-dir',str(out),'--source-commit','a'*40,'--player-checkpoint',str(pfile),'--spawner-checkpoint',str(sfile)])
    assert final.execute(a)==0
    assert syncs==[0,1,2] and shutdown==[True]
    assert windows==list(range(800000,800064))*3
    result=json.loads((out/'final.json').read_text())
    assert result['primary']['ci95']==[0,0] and result['secondary']['descriptive_only']
    assert (out/'final-claimed.json').exists()
