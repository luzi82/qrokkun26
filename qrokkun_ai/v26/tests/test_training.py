import copy
import torch
from qrokkun_ai.v21.agents.player_v21 import PlayerV21
from qrokkun_ai.v21.train import train_player_v21 as v21
from qrokkun_ai.v24.core import Spawner
from qrokkun_ai.v26.training import collect, train
from qrokkun_ai.v26.trace import Writer, Reader


def execute(root, workers):
    torch.manual_seed(17)
    p = PlayerV21(hidden=8)
    s = Spawner().requires_grad_(False).eval()
    w = Writer(root,reserve=0)
    ro = collect(p,s,'cpu',[100000,100001],80,workers,w,1)
    batch = v21.rollouts_to_batch_v21(ro,.9995,.999,torch.device('cpu'))
    oa = torch.optim.Adam(p.actor_parameters(),lr=3e-4,eps=1e-8)
    oc = torch.optim.Adam(p.critic_parameters(),lr=1e-3,eps=1e-8)
    train(p,oa,oc,batch,w,1,torch.Generator().manual_seed(27000017))
    w.close()
    return p, list(Reader(root).events())


def test_real_cpu_barrier_and_ppo(tmp_path):
    torch.set_num_threads(1)
    p, a = execute(tmp_path/'c',1)
    for workers in (2,4):
        q,b = execute(tmp_path/str(workers),workers)
        assert [{k:v for k,v in e.items() if k!='audit'} for e in a] == [{k:v for k,v in e.items() if k!='audit'} for e in b]
        assert all(torch.equal(v,q.state_dict()[k]) for k,v in p.state_dict().items())
    kinds = {e['kind'] for e in a}
    assert {'P','G','reset','episode_end','batch','permutation','minibatch','actor_preclip','actor_postclip','actor_prestep','actor_poststep','critic_preclip','critic_postclip','critic_prestep','critic_poststep'} <= kinds
    assert any(e['kind']=='G' and any(x.get('type')=='spawn' and x['bullet']['origin']=='field' for x in e['data']['events']) for e in a)


def test_traced_untraced_and_original_ppo_parity(tmp_path):
    from qrokkun_ai.v26.kernel import private
    torch.set_num_threads(1)
    torch.manual_seed(17)
    p=PlayerV21(hidden=8)
    q=copy.deepcopy(p)
    s=Spawner().requires_grad_(False).eval()
    before=torch.get_rng_state().clone()
    w=Writer(tmp_path/'parity',reserve=0)
    a=collect(p,s,'cpu',[100000,100001],80,4,w,1)
    b=collect(q,s,'cpu',[100000,100001],80,1,None,1)
    assert torch.equal(before,torch.get_rng_state())
    assert [(r.actions,r.rewards,r.bootstrap_value) for r in a]==[(r.actions,r.rewards,r.bootstrap_value) for r in b]
    batch=v21.rollouts_to_batch_v21(a,.9995,.999,torch.device('cpu'))
    def opts(net):
        return (torch.optim.Adam(net.actor_parameters(),lr=3e-4,eps=1e-8),torch.optim.Adam(net.critic_parameters(),lr=1e-3,eps=1e-8))
    oa,oc=opts(p);qa,qc=opts(q)
    g=torch.Generator().manual_seed(27000017)
    other=torch.Generator().manual_seed(27000017)
    class Proxy:
        def __getattr__(self,name):
            if name=='randperm':return lambda n,device:torch.randperm(n,device=device,generator=other)
            return getattr(torch,name)
    original=private(v21.train_step_v21,{'torch':Proxy()})
    expected=original(q,qa,qc,batch)
    actual=train(p,oa,oc,batch,w,1,g)
    assert actual==expected
    assert all(torch.equal(v,q.state_dict()[k]) for k,v in p.state_dict().items())
    assert torch.equal(g.get_state(),other.get_state())
    w.close()


def test_deliberately_reordered_worker_completion(tmp_path,monkeypatch):
    import time
    from qrokkun_ai.v26.kernel import Env
    original=Env.traced_step
    def delayed(self,action):
        if self.seed%2==0:time.sleep(.001)
        return original(self,action)
    monkeypatch.setattr(Env,'traced_step',delayed)
    _,a=execute(tmp_path/'serial',1)
    _,b=execute(tmp_path/'thread',4)
    assert [(e['kind'],e['key'],e['data']) for e in a]==[(e['kind'],e['key'],e['data']) for e in b]


def test_explicit_teacher_soft_targets(tmp_path):
    torch.set_num_threads(1)
    p=PlayerV21(hidden=8);s=Spawner().requires_grad_(False).eval()
    w=Writer(tmp_path/'teacher',reserve=0)
    collect(p,s,'cpu',[100000],1,1,w,1)
    w.close();r=Reader(tmp_path/'teacher')
    data=next(e['data'] for e in r.events() if e['kind']=='P')
    assert 'teacher_probs' in data
    logits=torch.frombuffer(bytearray(r.raw(data['teacher_logits'])),dtype=torch.float32).reshape(1,9)
    probs=torch.frombuffer(bytearray(r.raw(data['teacher_probs'])),dtype=torch.float32).reshape(1,9)
    assert torch.equal(probs,torch.softmax(logits,-1))
