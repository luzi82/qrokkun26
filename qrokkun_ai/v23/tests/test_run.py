import json
import torch
import pytest
from qrokkun_ai.v23 import core as c

def test_real_update_and_contract(tmp_path):
    from qrokkun_ai.v23 import run as r
    torch.set_num_threads(1)
    a=r.parser().parse_args(['train','--out-dir',str(tmp_path)])
    assert a.updates==1000 and a.deadline_unix is None and a.episodes_per_update==16
    r.validate_args(a)
    a.updates=999
    with pytest.raises(ValueError): r.validate_args(a)
    p,_=c.load_player(c.PLAYER_PATH)
    before={k:v.clone() for k,v in p.state_dict().items()}
    net=c.Spawner(); opt=torch.optim.Adam(net.parameters(),lr=3e-4)
    records,data,stats=r.episodes(net,p,[2000000,2000001],120,'cpu',r.Guard(),train_rng=torch.Generator().manual_seed(17))
    metrics=r.ppo(net,opt,data,r.Guard(),2)
    assert metrics['parameter_delta']>0 and metrics['grad_norm']>0
    assert all(torch.equal(v,before[k]) for k,v in p.state_dict().items())
    assert all(v.grad is None and not v.requires_grad for v in p.parameters())
    assert stats['decisions']>=2 and sum(stats['birth_histogram'])==stats['decisions']
    assert all(x['cap_frames']==120 for x in records)
    assert data[0][3]<0
    with pytest.raises(ValueError): c.load_spawner(c.COMPARATOR_PATH)
    r.atomic(tmp_path/'test.pt',dict(model=net.state_dict(),**net.architecture_metadata()))
    loaded,_=c.load_spawner(tmp_path/'test.pt')
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in net.state_dict().items())

def test_lifecycle_and_final_lock(tmp_path,monkeypatch):
    from qrokkun_ai.v23 import run as r
    # Unit fixture explicitly substitutes evaluation, never writes experiment results.
    a=r.parser().parse_args(['train','--out-dir',str(tmp_path/'run'),'--source-commit','0123456789abcdef0123456789abcdef01234567','--quick','--updates','1','--episodes-per-update','2','--eval-seeds','1'])
    r.main(a)
    out=tmp_path/'run'
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['source_commit']==a.source_commit==manifest['config']['source_commit']
    ck=torch.load(out/'latest.pt',weights_only=False)
    assert ck['completed_updates']==1 and ck['attempted_updates']==1 and not ck['partial_update']
    assert ck['optimizer']['state'] and ck['architecture']==c.ARCHITECTURE
    status=json.loads((out/'status.json').read_text())
    assert status['status']=='updates_complete' and status['last_error'] is None
    assert not (out/'final-claimed.json').exists()
    a.command='final'
    with pytest.raises(ValueError,match='1000'): r.main(a)

def test_exact_1000_loop_synthetic_unit(tmp_path,monkeypatch):
    """Synthetic lifecycle fixture: no policy trials or experiment evidence."""
    from qrokkun_ai.v23 import run as r
    out=tmp_path/'synthetic-unit'
    a=r.parser().parse_args(['train','--out-dir',str(out)])
    attempts=[]; saved={}
    def fake_episodes(net,p,seeds,cap,device,guard,mode='stochastic',**kw):
        seeds=list(seeds)
        if kw.get('train_rng') is not None: attempts.append(seeds)
        records=[dict(seed=s,seconds=1.,frames=60,death=True,censored=False,synthetic_unit_fixture=True) for s in seeds]
        if kw.get('on_complete'):
            for record in records: kw['on_complete'](record)
        return records,[],{}
    def fake_atomic(path,obj):
        saved[path.name]={k:obj[k] for k in ('completed_updates','attempted_updates','partial_update')}
        path.write_text(json.dumps(saved[path.name]))
    monkeypatch.setattr(r,'episodes',fake_episodes)
    monkeypatch.setattr(r,'ppo',lambda *args:{'synthetic_unit_fixture':True})
    monkeypatch.setattr(r,'atomic',fake_atomic)
    r.main(a)
    assert len(attempts)==1000 and attempts[0][0]==2000000 and attempts[-1][-1]==2015999
    assert saved['latest.pt']['completed_updates']==1000
    assert saved['update-1000.pt']['completed_updates']==1000
    assert saved['best.pt']['completed_updates']==1
    assert not (out/'final-claimed.json').exists()
    assert len(list(out.glob('validation-*-completion.json')))==104 # 101 validations + 3 controls
    assert json.loads((out/'training-closed.json').read_text())['completed_updates']==1000

def test_final_primary_locked_and_diag_interrupted_mock_only(tmp_path,monkeypatch):
    """Final is explicitly mocked: no heldout environment is constructed."""
    from qrokkun_ai.v23 import run as r
    out=tmp_path/'mock-final'; out.mkdir()
    a=r.parser().parse_args(['final','--out-dir',str(out),'--quick','--eval-seeds','1','--final-seeds','1'])
    net=c.Spawner()
    for name in ['best.pt','initial.pt','update-1000.pt']:
        r.atomic(out/name,dict(**net.architecture_metadata(),model=net.state_dict(),completed_updates=1000))
    r.json_write(out/'manifest.json',dict(config={**vars(a),'command':'train'}))
    r.json_write(out/'training-closed.json',dict(status='updates_complete',completed_updates=1000,best_sha=c.file_hash(out/'best.pt'),u1000_sha=c.file_hash(out/'update-1000.pt')))
    calls=[]
    def mock_final(net,p,seeds,cap,device,guard,mode='stochastic',**kw):
        calls.append(mode)
        if mode=='deterministic':
            assert (out/'final.json').exists()
            raise r.Stopped('mock diagnostic signal')
        records=[dict(seed=s,frames=60,seconds=1.,death=True,censored=False,synthetic_mock_final=True) for s in seeds]
        for row in records: kw['on_complete'](row)
        return records,[],{}
    monkeypatch.setattr(r,'episodes',mock_final)
    with pytest.raises(r.Stopped): r.main(a)
    result=json.loads((out/'final.json').read_text())
    assert result['formal'] is False and result['conclusion']=='inconclusive'
    assert set(result['comparisons'])=={'scripted','initial','sv22'}
    assert json.loads((out/'final-status.json').read_text())['status']=='primary_complete_diagnostic_incomplete'
    with pytest.raises(FileExistsError): r.main(a)

def test_interrupted_update_preserves_complete_latest(tmp_path,monkeypatch):
    import signal
    from qrokkun_ai.v23 import run as r
    a=r.parser().parse_args(['train','--out-dir',str(tmp_path/'interrupt'),'--quick','--updates','1','--episodes-per-update','2','--eval-seeds','1'])
    original=r.ppo
    def interrupt(net,opt,data,guard,epochs):
        original(net,opt,data,guard,epochs)
        guard.signal(signal.SIGTERM); guard.check()
    monkeypatch.setattr(r,'ppo',interrupt)
    with pytest.raises(r.Stopped): r.main(a)
    latest=torch.load(tmp_path/'interrupt/latest.pt',weights_only=False)
    partial=torch.load(tmp_path/'interrupt/interrupted.pt',weights_only=False)
    assert latest['completed_updates']==latest['attempted_updates']==0
    assert partial['completed_updates']==0 and partial['attempted_updates']==1 and partial['partial_update']
    assert any(not torch.equal(v,partial['model'][k]) for k,v in latest['model'].items())
    assert not (tmp_path/'interrupt/training-closed.json').exists()

def test_completed_progress_survives_validation_stop(tmp_path,monkeypatch):
    from qrokkun_ai.v23 import run as r
    a=r.parser().parse_args(['train','--out-dir',str(tmp_path/'val-stop'),'--quick','--updates','1','--episodes-per-update','2','--eval-seeds','1'])
    original=r.episodes; trained=False
    def interrupt(*args,**kwargs):
        nonlocal trained
        if kwargs.get('train_rng') is not None:
            trained=True
        elif trained: raise r.Stopped('validation unit signal')
        return original(*args,**kwargs)
    monkeypatch.setattr(r,'episodes',interrupt)
    with pytest.raises(r.Stopped): r.main(a)
    row=json.loads((tmp_path/'val-stop/progress.jsonl').read_text().splitlines()[-1])
    assert row['completed_updates']==1 and 'validation' not in row['metrics']
