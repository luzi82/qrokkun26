import math
import numpy as np
import pytest
import torch
from qrokkun_ai.v23 import core as c

def test_birth():
    x=torch.tensor([[0.,0.],[1e-25,1e-25],[-3.,4.]],dtype=torch.float64,requires_grad=True)
    a=c.birth_angle(x)
    assert a[0]==0
    assert a[1].item()==pytest.approx(math.pi/4)
    assert a[2].item()==pytest.approx(math.atan2(4,-3))
    a.sum().backward(); assert torch.isfinite(x.grad).all()

def test_wrapped():
    grid=torch.linspace(-math.pi,math.pi,40001,dtype=torch.float64)
    for sigma in (.01,.1,1.,3.,math.pi):
        lp=c.wrapped_log_prob(grid,torch.tensor(.72),torch.tensor(sigma))
        assert torch.trapezoid(lp.exp(),grid).item()==pytest.approx(1,abs=1e-10)
        edge=c.wrapped_log_prob(torch.tensor([-math.pi,math.pi],dtype=torch.float64),torch.tensor(.72),torch.tensor(sigma))
        assert edge[0].item()==pytest.approx(edge[1].item(),abs=1e-9)
    g=torch.Generator().manual_seed(17)
    a=c.wrapped_sample(torch.tensor(0.),torch.tensor(3.),g,(10000,))
    assert (a>=-math.pi).all() and (a<math.pi).all()
    assert abs(a.cos().mean().item()-math.exp(-4.5))<.025

def test_speed():
    g=torch.Generator().manual_seed(17)
    for mu,s in [(30.,.5),(115.,50.),(200.,170.)]:
        m=torch.tensor(mu,dtype=torch.float64,requires_grad=True)
        sigma=torch.tensor(s,dtype=torch.float64,requires_grad=True)
        a=c.speed_sample(m.expand(40000),sigma.expand(40000),g)
        assert (a>=30).all() and (a<=200).all()
        grid=torch.linspace(30,200,50001,dtype=torch.float64)
        p=c.speed_log_prob(grid,m,sigma).exp()
        assert torch.trapezoid(p,grid).item()==pytest.approx(1,abs=1e-5)
        expected=torch.trapezoid(p*grid,grid).item()
        assert a.mean().item()==pytest.approx(expected,abs=.8)
        c.speed_log_prob(a.detach(),m,sigma).mean().backward()
        assert torch.isfinite(m.grad) and torch.isfinite(sigma.grad)

def test_state_scales():
    torch.manual_seed(17); net=c.Spawner()
    x=torch.stack([torch.zeros(325),torch.ones(325)])
    m,s=net.parameters_at(x)
    assert net.actor(x).shape==(2,8) and s.shape==(2,3)
    assert ((s[0]-s[1]).abs()>1e-8).all()
    assert s[0].detach().numpy()==pytest.approx([3.,50.,.1],rel=.02)
    s.sum().backward()
    assert (net.actor[-1].weight.grad[5:].abs().sum(1)>0).all()
    assert not any('log_std' in n for n,_ in net.named_parameters())

def test_conditional():
    torch.manual_seed(17); net=c.Spawner(); x=torch.zeros(32,325); x[:,0:2]=.5
    g=torch.Generator().manual_seed(17)
    a,old=net.sample(x,g)
    new=net.log_prob(x,a)
    assert a.shape==(32,3)
    assert torch.equal(old,new)
    assert torch.equal((new-old).exp(),torch.ones_like(new))
    m,s=net.parameters_at(x)
    assert c.fire_mean(x,m,a[:,0]).std()>0.1
    # Zero offset retains fire exploration at spawn.
    assert c.wrap(a[:,2]-c.fire_mean(x,m,a[:,0])).std()>.03
    (-new.mean()+net.value(x).square().mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())

def test_environment():
    from qrokkun_env.gv20 import spawn_frames,ray_position
    e=c.LearnedEnv(lambda obs:np.array([0.,115.,0.]))
    e.reset(17)
    assert c.observation(e).shape==(325,)
    b=e.bullets[0]; pos=(b.x,b.y)
    assert pos==ray_position(0.) and b.radius==2.6 and not b.moved
    e.step(0); assert (b.x,b.y)==pos
    for _ in range(119): e.step(0)
    assert [f for f,_ in e.s_history]==list(spawn_frames(121))
    for f,s in e.field_history:
        expected=np.mean([v for t,v in e.s_history if f-60<=t<f])
        assert s==expected
    assert e.field_history
    _,ret=c.timed_returns([0,30,60],90)
    assert ret.tolist()==pytest.approx([-90/7200,-60/7200,-30/7200])

def test_numeric_edges_and_geometry():
    from qrokkun_env.gv20 import ray_position,aimed_velocity
    from qrokkun_env import constants as C
    x=torch.zeros(1,325); x[0,:2]=torch.tensor([160/C.FIELD_W,126/C.FIELD_H])
    m=torch.tensor([[1.,0.,0.,450.,-350.]],dtype=torch.float64,requires_grad=True)
    for b in [0.,math.pi/2,math.pi,-math.pi/2,.7]:
        mean=c.fire_mean(x,m,torch.tensor([b],dtype=torch.float64))
        vx,vy=aimed_velocity(ray_position(b),(float(x[0,0])*C.FIELD_W,float(x[0,1])*C.FIELD_H),(450.,-350.),115)
        assert mean.item()==pytest.approx(math.atan2(vy,vx),abs=1e-12)
    mean.sum().backward(); assert torch.isfinite(m.grad).all()
    for sig in [.01,math.pi]:
        mu=torch.tensor(.3,dtype=torch.float64,requires_grad=True); s=torch.tensor(sig,dtype=torch.float64,requires_grad=True)
        c.wrapped_log_prob(torch.tensor(-math.pi),mu,s).backward()
        assert torch.isfinite(mu.grad) and torch.isfinite(s.grad)
    a=torch.tensor([[-math.pi,30.,math.pi],[math.pi-1e-8,200.,-math.pi]],dtype=torch.float64)
    net=c.Spawner(); obs=x.expand(2,-1)
    old=net.log_prob(obs,a).detach(); new=net.log_prob(obs,a)
    assert torch.equal((new-old).exp(),torch.ones(2,dtype=torch.float64))
    new.sum().backward(); assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())

def test_inverse_cdf_open_interval(monkeypatch):
    # A PRNG can return exact zero: inverse CDF must still yield a finite speed.
    monkeypatch.setattr(torch,'rand',lambda shape,**kw:torch.zeros(shape,dtype=torch.float64))
    a=c.speed_sample(torch.tensor([115.]),torch.tensor([.5]),torch.Generator())
    assert torch.isfinite(a).all() and (a>=30).all() and (a<=200).all()

def test_float32_subnormal_direction_gradient():
    x=torch.tensor([[1e-40,-1e-40],[0.,0.]],requires_grad=True)
    a=c.birth_angle(x); assert a[0].item()==pytest.approx(-math.pi/4)
    a.sum().backward(); assert torch.isfinite(x.grad).all()
