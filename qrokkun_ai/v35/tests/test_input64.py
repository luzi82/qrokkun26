import pytest
import torch
from qrokkun_ai.v35.model import PlayerV35
from qrokkun_ai.v35.initialization import random_player, old_random_player, MAPPING
from qrokkun_ai.v35 import run

def obs():
    return torch.zeros(2,6),torch.ones(2,64,4)*.1,torch.zeros(2,64,dtype=torch.bool)

def test_from_metadata():
    p=random_player(8)
    assert PlayerV35.from_metadata(p.architecture_metadata()).in_dim==344

def test_shape():
    p=random_player(8); assert p.features(*obs()).shape==(2,344)

def test_metadata():
    m=random_player(8).architecture_metadata()
    assert (m['policy_input_top_k'],m['teacher_top_k'],m['clearance_top_k'],m['in_dim'])==(64,12,12,344)

def test_old_metadata_rejected():
    with pytest.raises(ValueError): PlayerV35.from_metadata(old_random_player(8).architecture_metadata())

def test_old_weights_rejected():
    with pytest.raises(RuntimeError): random_player(8).load_state_dict(old_random_player(8).state_dict())

def test_rng():
    before=torch.get_rng_state().clone(); random_player(8); assert torch.equal(before,torch.get_rng_state())

def test_mapping():
    old=old_random_player(8).state_dict(); new=random_player(8).state_dict()
    for k,v in old.items():
        if k in ('policy_body.0.weight','value_body.0.weight'):
            for a,b,c,d in MAPPING: assert torch.equal(v[:,a:b],new[k][:,c:d])
            assert new[k][:,54:262].count_nonzero()==new[k][:,274:326].count_nonzero()==0
        else: assert torch.equal(v,new[k])

def test_initial_outputs():
    o=obs(); old=old_random_player(8); new=random_player(8)
    a,b=old(*o); c,d=new(*o)
    torch.testing.assert_close(a.logits,c.logits,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(b,d,atol=1e-6,rtol=1e-6)
    assert torch.equal(a.logits.argmax(-1),c.logits.argmax(-1))

def test_extra_policy_value_and_clearance():
    p=random_player(8); x,b,m=obs(); c=b.clone(); c[:,12,0]+=1
    with torch.no_grad():
        p.policy_body[0].weight[:,54]=1; p.value_body[0].weight[:,54]=1
    f=p.features(x,b,m); g=p.features(x,c,m)
    assert torch.equal(f[:,-18:],g[:,-18:])
    assert not torch.equal(p(x,b,m)[0].logits,p(x,c,m)[0].logits)
    assert not torch.equal(p(x,b,m)[1],p(x,c,m)[1])
    assert p.top_k==p.teacher_top_k==12

def test_mask():
    p=random_player(8); x,b,m=obs(); m[:,12:]=True
    c=b.clone(); c[:,12:]=99
    assert torch.equal(p.features(x,b,m),p.features(x,c,m))
    assert p.features(x,b,m)[:,274:326].count_nonzero()==0

def test_gradient_extra():
    p=random_player(8); pi,v=p(*obs()); (pi.logits[:,0].sum()+v.sum()).backward()
    for body in (p.policy_body,p.value_body): assert body[0].weight.grad[:,54:262].count_nonzero()>0

def test_fresh_adam():
    from qrokkun_ai.v31.core import optimizers
    assert all(not o.state for o in optimizers(random_player(8)))

def test_cli_formal():
    a=run.parser().parse_args(['train','--out-dir','/tmp/unused','--source-commit','a'*40]); assert len(run.validate(a)[1])==101
    a.hidden=8
    with pytest.raises(ValueError): run.validate(a)

def test_cli_reduced():
    a=run.parser().parse_args(['train','--out-dir','/tmp/unused','--source-commit','a'*40,'--nonformal','--updates','1','--episodes','3','--workers','2','--validation','2']); run.validate(a)

def test_hooks():
    from qrokkun_ai.v31 import core
    from qrokkun_ai.v35.adapter import runtime
    old=core.PlayerV21
    with pytest.raises(RuntimeError):
        with runtime():
            assert core.PlayerV21 is PlayerV35
            raise RuntimeError()
    assert core.PlayerV21 is old

def test_teacher_and_legacy_counterfactual():
    from qrokkun_ai.v31.core import compute_kinematic_expert_targets
    x,b,m=obs(); c=b.clone(); c[:,12,0]+=1
    old=old_random_player(8)
    assert torch.equal(old.features(x,b,m),old.features(x,c,m))
    for a,z in zip(old(x,b,m),old(x,c,m)):
        assert torch.equal(a.logits if hasattr(a,'logits') else a,z.logits if hasattr(z,'logits') else z)
    p=random_player(8)
    a=compute_kinematic_expert_targets(x,b,m,horizon=p.lookahead_horizon,top_k=p.teacher_top_k)
    z=compute_kinematic_expert_targets(x,c,m,horizon=p.lookahead_horizon,top_k=p.teacher_top_k)
    assert all(torch.equal(v,w) for v,w in zip(a,z))

def test_all_scheduled_control_evaluations_pinned():
    from qrokkun_ai.v35.reference import PIN, schedule
    assert {f'evaluation-{u:06d}.json' for u in schedule()} <= PIN['file_hashes'].keys()

def test_saved_control_and_u0():
    from qrokkun_ai.v35.reference import load_reference, PIN
    from qrokkun_ai.v21.agents.player_v21 import state_dict_sha256
    from qrokkun_ai.v35.initialization import OLD_SHA
    from pathlib import Path
    directory=Path('/mnt/minion/88_minion_rw/projects/qrokkun26/runs/20261116-1219-pv34m-threat-filter-u1000')
    reference=load_reference(directory)
    assert len(reference['curve'])==101 and reference['results']['best']['update']==210
    saved=torch.load(directory/'checkpoint-000000.pt',map_location='cpu',weights_only=False)
    old=old_random_player()
    assert state_dict_sha256(old.state_dict())==OLD_SHA
    assert all(torch.equal(v,saved['model'][k]) for k,v in old.state_dict().items())
    assert len(PIN['file_hashes'])>=110

def test_spawn_training_roundtrip(tmp_path):
    import json
    from qrokkun_ai.v31 import core
    old=core.PlayerV21
    out=tmp_path/'unit-training'
    assert run.train_main(['train','--out-dir',str(out),'--source-commit','a'*40,
        '--nonformal','--synthetic','--device','cpu','--updates','1','--episodes','1',
        '--cap','2','--validation','1','--hidden','8','--workers','1','--pin-cores','off'])==0
    assert core.PlayerV21 is old
    ck=torch.load(out/'checkpoint-000001.pt',map_location='cpu',weights_only=False)
    net=PlayerV35.from_metadata(ck['architecture']); net.load_state_dict(ck['model'])
    assert ck['model']['policy_body.0.weight'].shape==(8,344)
    assert json.loads((out/'status.json').read_text())['status']=='complete'
    assert not (out/'final.json').exists()

def test_cleanup_failure_retained():
    from qrokkun_ai.v32.tests.test_pool import test_pool_unlink_failure_is_not_success
    from qrokkun_ai.v35.adapter import EpisodePool
    from qrokkun_ai.v32.pool import EpisodePool as LifecyclePool
    assert EpisodePool.close is LifecyclePool.close
    test_pool_unlink_failure_is_not_success()

def test_public_preflight_digest_context(monkeypatch):
    from qrokkun_ai.v35 import final
    from qrokkun_ai.v35.fingerprint import source_digest
    from qrokkun_ai.v31 import core
    old=core.source_digest
    def probe(*args):
        assert core.source_digest is source_digest
        return 'checked'
    monkeypatch.setattr(final,'_preflight',probe)
    assert final.preflight(None)=='checked'
    assert core.source_digest is old

def test_final_saved_control_only(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from qrokkun_ai.v35 import final
    from qrokkun_ai.v24.core import file_hash
    out=tmp_path/'final-fixture'; out.mkdir()
    sf=tmp_path/'synthetic-s'; sf.write_text('fixture')
    p=random_player(8)
    for u in (210,1000):
        torch.save(dict(schema='pv35_checkpoint_v1',update=u,architecture=p.architecture_metadata(),model=p.state_dict()),out/run.ckpt_name(u))
    rows=[dict(seed=s,frames=60,death=True,censored=False) for s in final.FINAL_SEEDS]
    warm=dict(identity={'fixture':True},curve=[],timings={},results={n:dict(episodes=rows) for n in ('best','terminal')})
    m=dict(semantic=dict(control_reference=warm['identity'],inputs=dict(spawner=file_hash(sf))))
    monkeypatch.setattr(final,'preflight',lambda *a:(m,dict(best_update=210,timings={})))
    monkeypatch.setattr(final,'random_player',lambda:p)
    monkeypatch.setattr(final,'locked_spawner',lambda *a:None)
    monkeypatch.setattr(final,'load_reference',lambda *a:warm)
    for u in range(0,1001,10): run.json_write(out/run.eval_name(u),dict(update=u,episodes=[]))
    windows=[]; syncs=[]; closed=[]
    class Pool:
        def __init__(self,*a): pass
        def sync_weights(self,sd,policy_version,**kw):
            assert sd['policy_body.0.weight'].shape==(8,344)
            syncs.append(policy_version)
        def collect(self,seeds,*a,**kw):
            windows.extend(seeds)
            return [SimpleNamespace(summary=lambda seed=s:dict(seed=seed,frames=60,death=True,censored=False)) for s in seeds]
        def shutdown_and_stats(self): closed.append(True)
        def close(self): closed.append(True)
    monkeypatch.setattr(final,'EpisodePool',Pool)
    a=run.parser().parse_args(['final','--out-dir',str(out),'--source-commit','a'*40,'--spawner-checkpoint',str(sf)])
    assert final.execute(a)==0
    assert windows==final.FINAL_SEEDS*2 and syncs==[0,1] and closed==[True]
    result=json.loads((out/'final.json').read_text())
    assert result['primary']['ci95']==[0,0] and result['secondary']['descriptive_only']
    with pytest.raises(FileExistsError): final.execute(a)

def test_expanded_checkpoint_space_guard():
    from qrokkun_ai.v35.adapter import estimate_bytes, runtime
    from qrokkun_ai.v31 import core
    old=core.estimate_bytes
    assert estimate_bytes(472,101)>old(472,101)
    with runtime(): assert core.estimate_bytes is estimate_bytes
    assert core.estimate_bytes is old
