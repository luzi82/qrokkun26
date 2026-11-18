import numpy as np
import pytest
from qrokkun_ai.v39.world import LearnedEnv, View, snapshot, restore
from qrokkun_ai.v39.adapter import encode_obs, runtime
from qrokkun_ai.v24.core import observation

def make(offset=40):
    seen=[]
    def callback(obs):
        seen.append(float(obs[4])); return (0.,30.,0.)
    e=LearnedEnv(callback, seed=17, difficulty_offset=offset); e.reset()
    return e,seen

def test_real_s_callback_clock():
    e,seen=make(); assert seen == [pytest.approx(1/3)]
    assert e.frame == 0 and e.elapsed == 0

def test_real_p_encoder_clock():
    e,_=make(); assert encode_obs(e)[0][4] == pytest.approx(1/3)
    assert e.elapsed == 0

def test_views_readonly():
    e,_=make()
    with pytest.raises(AttributeError): View(e,'S').frame=1

def test_s_unclipped():
    e,_=make(); e.frame=7200
    assert observation(View(e,'S'))[4] == pytest.approx(4/3)

def test_p_clipped():
    e,_=make(); e.elapsed=100
    assert encode_obs(e)[0][4] == 1

def test_normal_callback_clock():
    e,seen=make(0); assert seen == [0.]

def test_forks_restore_full_state():
    e,_=make()
    for _ in range(11): e.step(0)
    state=snapshot(e)
    a=restore(state,e.callback); b=restore(state,e.callback)
    for _ in range(60):
        assert a.step(0) == b.step(0)
    sa,sb=snapshot(a),snapshot(b)
    for key in ('rng','s_rng','field_rng'):
        assert sa.pop(key).getstate() == sb.pop(key).getstate()
    assert sa == sb
    assert e.frame == 11 and a.difficulty_offset == 40

def test_runtime_aliases_restored():
    from qrokkun_ai.v37 import teaching, episode
    old=(teaching.restore,episode.encode_obs_fast)
    with runtime():
        assert teaching.restore is restore
        assert episode.encode_obs_fast is encode_obs
    assert old == (teaching.restore,episode.encode_obs_fast)

def test_sidefield_prior_second():
    e,_=make()
    for _ in range(60): e.step(0)
    assert e.field_history == [(60,30.)]

def test_first_bullet_delay():
    e,_=make(); b=e.bullets[0]; x=b.x
    assert not b.moved
    e.step(0)
    assert b.x == x and b.moved

def test_source_closure():
    from qrokkun_ai.v39.fingerprint import source_files, source_digest
    names={str(p) for p in source_files()}
    assert any(p.endswith('/qrokkun_env/gv39.py') for p in names)
    assert any(p.endswith('/v37/teaching.py') for p in names)
    assert all('/tests/' not in p for p in names)
    assert len(source_digest()) == 64

def test_u0_and_fresh_adams():
    import torch
    from qrokkun_ai.v35.initialization import random_player
    from qrokkun_ai.v21.agents.player_v21 import state_dict_sha256
    from qrokkun_ai.v31.core import optimizers
    with torch.random.fork_rng(devices=[]): p=random_player()
    assert state_dict_sha256(p.state_dict()) == 'fd355475c38bdfcd065c13bd5d1b92ddb33327f3533f08baedb92b788fc05fb0'
    assert all(not o.state for o in optimizers(p))

def test_spawn_pool_hard_and_normal_and_tiny_ppo(tmp_path):
    import torch
    from qrokkun_ai.v39.adapter import EpisodePool
    from qrokkun_ai.v35.initialization import random_player
    from qrokkun_ai.v24.core import Spawner
    from qrokkun_ai.v31 import core
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        p=random_player(hidden=8); s=Spawner().eval()
    with runtime():
        pool=EpisodePool(p_template=p,s_cpu=s,max_slots=1,max_cap=8,workers=1,pin_cores='off',root=tmp_path/'cache',search=False)
        try:
            pool.sync_weights(p.state_dict(),policy_version=0)
            hard=pool.collect([17],8,validation=False)
            normal=pool.collect([17],8,validation=True)
            assert hard[0].player[0,4] == pytest.approx(1/3)
            assert normal[0].player[0,4] == 0
            oa,oc=core.optimizers(p)
            batch=core.assemble_batch(hard,p,'cpu')
            metrics=core.train(p,oa,oc,batch,generator=torch.Generator().manual_seed(17))
            assert metrics and oa.state and oc.state
            stats=pool.shutdown_and_stats()
            assert len(stats)==1 and (tmp_path/'workers.json').is_file()
        finally: pool.close()

def test_real_capture_sampler_fullworld_hooks():
    import torch
    from qrokkun_ai.v24.core import Spawner
    from qrokkun_ai.v37 import teaching, episode
    with torch.random.fork_rng(devices=[]): model=Spawner().eval()
    sampler=teaching.Sampler(model,torch.Generator().manual_seed(42).get_state())
    with runtime(search=True):
        env=episode.CaptureEnv(sampler,difficulty_offset=40,seed=17)
        env.reset()
        capture=teaching.Capture(model,sampler.generator); env.capture=capture
        for _ in range(11): env.step(0)
        anchor=capture.anchor(10)
        before=teaching.snapshot(anchor)
        rng=anchor.callback.generator.get_state().clone()
        left=teaching.fork(anchor); right=teaching.fork(anchor)
        assert left.callback.generator is not right.callback.generator
        for _ in range(60):
            assert left.step(0)==right.step(0)
            np.testing.assert_array_equal(teaching.encode_obs(left)[0],teaching.encode_obs(right)[0])
        assert left.frame==right.frame and 10<left.frame<=70
        assert left.elapsed==right.elapsed and left.difficulty_elapsed==right.difficulty_elapsed
        assert teaching.snapshot(left)['bullets']==teaching.snapshot(right)['bullets']
        assert torch.equal(left.callback.generator.get_state(),right.callback.generator.get_state())
        assert torch.equal(anchor.callback.generator.get_state(),rng)
        assert teaching.snapshot(anchor)['bullets']==before['bullets'] and anchor.frame==10

        from qrokkun_env.gv36 import Gv36Env
        mover=Gv36Env(seed=0); mover.restore_player_state(anchor.get_player_state())
        positions=[[mover.px,mover.py]]
        for _ in range(60):
            mover._integrate_player(*mover._parse_action(0)); positions.append([mover.px,mover.py])
        candidate=dict(actions=[0]*60,positions=positions)
        first=teaching.validate(anchor,[candidate]); second=teaching.validate(anchor,[candidate])
        assert first[1]==second[1]==1 and len(first[0])==len(second[0])
        assert torch.equal(anchor.callback.generator.get_state(),rng) and anchor.frame==10

@pytest.mark.parametrize('control_changed',[False,True])
def test_complete_orchestration_inprocess(tmp_path,monkeypatch,control_changed):
    import json,time
    from types import SimpleNamespace
    import torch
    from qrokkun_ai.v39 import run
    from qrokkun_ai.v35.initialization import random_player
    from qrokkun_ai.v24.core import Spawner
    from qrokkun_ai.v21.agents.player_v21 import state_dict_sha256
    with torch.random.fork_rng(devices=[]): p=random_player(hidden=8); s=Spawner().eval()
    initial=state_dict_sha256(p.state_dict())
    monkeypatch.setattr(run,'random_player',lambda:p)
    monkeypatch.setattr(run,'locked_spawner',lambda path:s)
    monkeypatch.setattr(run,'archive_source',lambda commit,path:path.write_bytes(b'archive'))
    control_checks=[]
    def consume(*args):
        assert not args[2]()
        control_checks.append(True)
        if control_changed and len(control_checks)==2: raise ValueError('frozen control changed')
        return {}
    monkeypatch.setattr(run,'consume_control',consume)
    monkeypatch.setattr(run,'sync_cuda',lambda device:None)
    monkeypatch.setattr(run.core,'configure_runtime',lambda device:None)
    monkeypatch.setattr(run.core,'select_learner_cpu',lambda *a,**k:None)
    from qrokkun_ai.v39 import protocol
    original_read=protocol.read
    def read(root,name):
        if name=='contract.json': return {'initial_state_sha':initial}
        if name=='evaluation-000000.json' and Path(root)==tmp_path:
            return dict(episodes=[dict(seed=i,frames=1,death=True,censored=False) for i in range(600000,600016)])
        return original_read(root,name)
    monkeypatch.setattr(protocol,'read',read)
    monkeypatch.setattr(run.core,'assemble_batch',lambda trajs,*a:trajs)
    monkeypatch.setattr(torch,'save',lambda payload,path:Path(path).write_bytes(str(payload['update']).encode()))
    from pathlib import Path
    calls=[]
    class Pool:
        def __init__(self,**kwargs): self.stop_event=SimpleNamespace(set=lambda:None); self.version=None; self.workers=kwargs['workers']
        def sync_weights(self,state,policy_version,**kw): self.version=policy_version; calls.append(('sync',policy_version))
        def collect(self,seeds,cap,validation,**kw):
            assert self.version is not None
            calls.append(('collect',self.version,validation,list(seeds)))
            rows=[]
            for seed in seeds:
                summary=dict(seed=seed,frames=1,death=True,censored=False,bootstrap=0.,elapsed=1/60)
                rows.append(SimpleNamespace(frames=1,summary=lambda summary=summary:summary,demos=[],teaching={}))
            return rows
        def shutdown_and_stats(self): return [{'pid':100+i,'cpu_s':0.01} for i in range(self.workers)]
        def close(self): pass
    monkeypatch.setattr(run,'EpisodePool',Pool)
    def train(p,oa,oc,batch,**kwargs):
        assert kwargs['stop_check']() is False
        for opt in (oa,oc):
            param=opt.param_groups[0]['params'][0]
            opt.state[param]['step']=opt.state[param].get('step',0)+run.core.v21.PPO_EPOCHS
        return dict(n_samples=len(batch))
    monkeypatch.setattr(run.core,'train',train)
    monkeypatch.setattr(run,'auxiliary',lambda *args:dict(stepped=False))
    spawner=tmp_path/'s.pt'; spawner.write_bytes(b'tiny'); pins=tmp_path/'pins.json'; pins.write_text('{}')
    a=SimpleNamespace(out_dir=str(tmp_path/'out'),scratch=str(tmp_path),stop_path=str(tmp_path/'STOP'),
        source_commit='a'*40,control_dir=str(tmp_path),control_pins=str(pins),spawner_checkpoint=str(spawner),
        device='cuda',smoke=False,deadline_epoch=time.time()+5000)
    # The full formal schedule is cheap here: no rollout, weights, or GPU execution.
    original_generator=torch.Generator
    monkeypatch.setattr(torch,'Generator',lambda **kwargs:original_generator())
    monkeypatch.setattr(type(p),'to',lambda self,*a,**kw:self)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    if control_changed:
        with pytest.raises(ValueError,match='frozen control changed'): run.execute(a)
        assert len(control_checks)==2
        assert not (Path(a.out_dir)/'closed.json').exists()
        assert json.loads((Path(a.out_dir)/'status.json').read_text())['status']=='failed'
        return
    assert run.execute(a)==0
    assert len(control_checks)==2
    assert calls[0]==('sync',0) and calls[1]==('collect',0,True,list(range(600000,600016)))
    vals=[x for x in calls if x[0]=='collect' and x[2]]
    assert [x[1] for x in vals]==list(range(0,1001,10))
    trains=[x for x in calls if x[0]=='collect' and not x[2]]
    assert len(trains)==1000 and all(len(x[3])==40 for x in trains)
    assert trains[-1][3][-1]==139999
    out=Path(a.out_dir)
    assert json.loads((out/'best.json').read_text())['update']==0
    updates=[json.loads(line) for line in (out/'updates.jsonl').read_text().splitlines()]
    assert len(updates)==1000 and updates[-1]['ordinary_actor_cumulative']==1000*run.core.v21.PPO_EPOCHS
    assert updates[-1]['auxiliary_actor_cumulative']==0
    assert all(row['frames']==row['metrics']['n_samples']==len(row['episodes'])==40 for row in updates)
    assert len(list(out.glob('checkpoint-*.pt')))==101
    from qrokkun_ai.v39.final import closure
    manifest,closed=closure(out)
    assert closed['completed_updates']==1000 and closed['best_update']==0
    (out/'workers.json').write_text('[{"cpu_s":999}]')
    with pytest.raises(ValueError,match='worker stats changed'): closure(out)

@pytest.mark.parametrize('control_changed',[False,True])
def test_complete_final_roles_persistence_and_claim(tmp_path,monkeypatch,control_changed):
    import json,time
    from types import SimpleNamespace
    import torch
    from qrokkun_ai.v39 import final
    from qrokkun_ai.v35.initialization import random_player
    from qrokkun_ai.v24.core import Spawner,file_hash
    from qrokkun_ai.v39.protocol import FINAL_SEEDS
    from qrokkun_ai.v39.fingerprint import source_digest
    with torch.random.fork_rng(devices=[]): net=random_player(hidden=8); spawner=Spawner().eval()
    out=tmp_path/'out'; out.mkdir(); old=tmp_path/'control'; old.mkdir()
    s=tmp_path/'s.pt'; s.write_bytes(b's')
    pins=tmp_path/'pins.json'; pins.write_text('{}')
    for u in [10,1000]: (out/f'checkpoint-{u:06d}.pt').write_bytes(str(u).encode())
    hashes={f'checkpoint-{u:06d}.pt':file_hash(out/f'checkpoint-{u:06d}.pt') for u in [10,1000]}
    m=dict(semantic=dict(source_commit='a'*40,control_pin={},inputs={'spawner':file_hash(s)}))
    c=dict(best_update=10,checkpoint_hashes=hashes)
    monkeypatch.setattr(final,'closure',lambda *args:(m,c))
    monkeypatch.setattr(final,'consume_control',lambda *args:{})
    monkeypatch.setattr(final,'random_player',lambda:net)
    monkeypatch.setattr(final,'locked_spawner',lambda path:spawner)
    from qrokkun_ai.v31 import core
    monkeypatch.setattr(core,'configure_runtime',lambda *args:None)
    loads=[]
    def load(path,**kw):
        assert str(path).startswith(str(tmp_path)) and 'pv39-final-' in str(path)
        u=int(path.read_text()); loads.append(u)
        return dict(schema='pv39_checkpoint_v1',update=u,source_tree_sha=source_digest(),architecture=net.architecture_metadata(),model=net.state_dict())
    monkeypatch.setattr(torch,'load',load)
    class Pool:
        def __init__(self,**kwargs): assert kwargs['search'] is False; self.stop_event=SimpleNamespace(set=lambda:None)
        def sync_weights(self,*args,**kw): pass
        def collect(self,seeds,cap,validation,**kw):
            assert cap==7200 and validation
            return [SimpleNamespace(summary=lambda seed=seed:dict(seed=seed,frames=7200,death=True,censored=False,elapsed=120.)) for seed in seeds]
        def shutdown_and_stats(self):
            if control_changed: (old/'final.json').write_text('{"results":{}}')
            return [{'cpu_s':0.01}]
        def close(self): pass
    monkeypatch.setattr(final,'EpisodePool',Pool)
    oldrows=[dict(seed=seed,frames=7199,death=True,censored=False,cap_frames=7200) for seed in FINAL_SEEDS]
    (old/'final.json').write_text(json.dumps(dict(results={k:dict(episodes=oldrows) for k in ['best','terminal']})))
    m['semantic']['control_pin']={'final.json':file_hash(old/'final.json')}
    control_checks=[]
    def consume(*args):
        assert not args[2]()
        control_checks.append(True)
        if file_hash(old/'final.json')!=m['semantic']['control_pin']['final.json']:
            raise ValueError('frozen control changed')
        return dict(m['semantic']['control_pin'])
    monkeypatch.setattr(final,'consume_control',consume)
    a=SimpleNamespace(out_dir=str(out),scratch=str(tmp_path),stop_path=str(tmp_path/'STOP'),device='cuda',smoke=False,
        deadline_epoch=time.time()+5000,control_dir=str(old),control_pins=str(pins),source_commit='a'*40,spawner_checkpoint=str(s))
    if control_changed:
        with runtime(),pytest.raises(ValueError,match='frozen control changed'): final.execute(a)
        assert len(control_checks)==2
        assert not (out/'final.json').exists()
        status=json.loads((out/'final-status.json').read_text())
        assert status['status']=='failed' and status['errors']
        return
    with runtime(): assert final.execute(a)==0
    assert len(control_checks)==2
    result=json.loads((out/'final.json').read_text())
    assert loads==[10,1000] and len(result['results']['best']['episodes'])==64
    assert result['primary']['mean_seconds_difference']==pytest.approx(1/60)
    status=(out/'final-status.json').read_bytes()
    with runtime(),pytest.raises(FileExistsError): final.execute(a)
    assert (out/'final-status.json').read_bytes()==status
