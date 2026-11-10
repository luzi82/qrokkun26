"""Sv23 physical conditional policy. All distribution arithmetic is float64."""
import math
import hashlib
import numpy as np
import torch
from torch import nn

OBS_DIM=325
ARCHITECTURE='sv23_conditional_physical_v1'

class _StableBearing(torch.autograd.Function):
    @staticmethod
    def forward(ctx,vector):
        v=vector.double(); scale=v.abs().amax(-1,keepdim=True)
        z=v/torch.where(scale==0,torch.ones_like(scale),scale)
        zero=(scale==0).squeeze(-1)
        x=torch.where(zero,torch.ones_like(z[...,0]),z[...,0])
        y=torch.where(zero,torch.zeros_like(z[...,1]),z[...,1])
        angle=torch.atan2(y,x)
        radius=scale.squeeze(-1)*torch.linalg.vector_norm(z,dim=-1)
        ctx.save_for_backward(angle,radius,zero)
        ctx.input_dtype=vector.dtype
        return angle
    @staticmethod
    def backward(ctx,grad):
        angle,radius,zero=ctx.saved_tensors
        # The forward direction is exact for every nonzero vector. Only the
        # singular derivative below 1e-12 is bounded, without an east sector.
        inv=torch.where(zero,torch.zeros_like(radius),1/radius.clamp_min(1e-12))
        derivative=torch.stack([-angle.sin()*inv,angle.cos()*inv],-1)
        return (grad[...,None]*derivative).to(ctx.input_dtype)

def birth_angle(vector): return _StableBearing.apply(vector)

def wrap(x): return torch.remainder(x+math.pi,2*math.pi)-math.pi

def wrapped_sample(mean,sigma,generator,shape=None):
    return wrap(mean.double()+sigma.double()*torch.randn(shape or mean.shape,dtype=torch.float64,generator=generator))

def wrapped_log_prob(value,mean,sigma):
    d=wrap(value.double()-mean.double())
    s=sigma.double()
    # |d|<=pi, sigma<=pi. Omitted images start at 11*pi (>11 sigma).
    images=torch.arange(-5,6,dtype=torch.float64,device=d.device)*2*math.pi
    return torch.logsumexp(-.5*((d[...,None]+images)/s[...,None])**2-s[...,None].log()-.5*math.log(2*math.pi),-1)

def speed_cdfs(mean,sigma):
    mean,sigma=mean.double(),sigma.double()
    return torch.special.ndtr((30-mean)/sigma),torch.special.ndtr((200-mean)/sigma)

def speed_sample(mean,sigma,generator):
    lo,hi=speed_cdfs(mean,sigma)
    # Open 52-bit uniform avoids ndtri(0/1), including after affine rounding.
    # This discretizes the PRNG, never clamps a physical velocity.
    u=torch.randint(1,2**52,mean.shape,generator=generator,dtype=torch.int64).double()/2**52
    return mean.double()+sigma.double()*torch.special.ndtri(lo+(hi-lo)*u)

def speed_log_prob(value,mean,sigma):
    lo,hi=speed_cdfs(mean,sigma)
    s=sigma.double()
    lp=-.5*((value.double()-mean.double())/s)**2-s.log()-.5*math.log(2*math.pi)-(hi-lo).log()
    return torch.where((value>=30)&(value<=200),lp,torch.full_like(lp,-torch.inf))

class Spawner(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(325,128),nn.Tanh(),nn.Linear(128,8))
        self.critic=nn.Sequential(nn.Linear(325,128),nn.Tanh(),nn.Linear(128,1))
        nn.init.zeros_(self.actor[-1].weight)
        with torch.no_grad():
            nn.init.normal_(self.actor[-1].weight[5:],std=.001)
            self.actor[-1].bias[:5].copy_(torch.tensor([1.,0.,0.,0.,0.]))
            for i,(v,lo,hi) in enumerate(zip([3.,50.,.1],[.01,.5,.01],[math.pi,170.,math.pi])):
                self.actor[-1].bias[5+i]=math.log((v-lo)/(hi-v))
    def parameters_at(self,x):
        h=self.actor(x).double()
        lo=h.new_tensor([.01,.5,.01]); hi=h.new_tensor([math.pi,170.,math.pi])
        return h[...,:5],lo+(hi-lo)*h[...,5:].sigmoid()
    def value(self,x): return self.critic(x).squeeze(-1)
    def architecture_metadata(self):
        return dict(architecture=ARCHITECTURE,architecture_version=1,obs_dim=325,hidden=128,actor_components=8,physical_dofs=3)
    def sample(self,x,generator,deterministic=False):
        m,s=self.parameters_at(x)
        b=birth_angle(m[...,:2])
        b=wrap(b) if deterministic else wrapped_sample(b,s[...,0],generator)
        speed=115+85*m[...,2].tanh()
        if not deterministic: speed=speed_sample(speed,s[...,1],generator)
        fire=fire_mean(x,m,b)
        fire=wrap(fire) if deterministic else wrapped_sample(fire,s[...,2],generator)
        a=torch.stack([b,speed,fire],-1).detach()
        return a,self.log_prob(x,a)
    def log_prob(self,x,a):
        m,s=self.parameters_at(x)
        return (wrapped_log_prob(a[...,0],birth_angle(m[...,:2]),s[...,0])+
                speed_log_prob(a[...,1],115+85*m[...,2].tanh(),s[...,1])+
                wrapped_log_prob(a[...,2],fire_mean(x,m,a[...,0]),s[...,2]))

def fire_mean(obs,means,birth):
    # Exactly production ray rectangle: center (160,126), half extents (160,114).
    cx,sy=birth.double().cos(),birth.double().sin()
    distance=torch.minimum(160/cx.abs(),114/sy.abs())
    spawn=torch.stack([160+distance*cx,126+distance*sy],-1)
    target=obs[...,:2].double()*obs.new_tensor([C.FIELD_W,C.FIELD_H]).double()+means[...,3:5]
    delta=target-spawn
    # Production aimed_velocity's <1e-12 zero-direction fallback.
    zero=torch.linalg.vector_norm(delta,dim=-1)<1e-12
    delta=torch.where(zero[...,None],delta.new_tensor([1.,0.]),delta)
    return birth_angle(delta)

from qrokkun_env import constants as C
from qrokkun_env.gv20 import Gv20Env, Bullet, ray_position
from qrokkun_ai.v21.agents.player_v21 import load_v21_checkpoint
PLAYER_SHA='b8466bb13972de62003e16b0b570567a016d0dab4dd5667e66a3666974cfd5f8'
STATE_SHA='250f3c3f1053b18129f992e97cb9ae15139fc2e5fa962b5491db5dddb2eb8f91'
PLAYER_PATH='artifacts/sv22-media-inputs/pv21-u400.pt'
COMPARATOR_PATH='artifacts/sv22-media-inputs/sv22-u880.pt'
COMPARATOR_SHA='c19e8504946b4567ea273022921490b20fb8b49260676e0f5b61354e3102e7d7'

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def load_player(path,expected=PLAYER_SHA,device='cpu'):
    if expected!=PLAYER_SHA or file_hash(path)!=PLAYER_SHA: raise ValueError('player identity mismatch')
    p,m=load_v21_checkpoint(path,device)
    if m['extra']['update']!=400 or m['state_dict_sha256']!=STATE_SHA: raise ValueError('player update/state mismatch')
    return p.requires_grad_(False).eval(),m

def observation(e):
    o=e.observe(); rows=np.asarray(o['bullets'],dtype=np.float32); mask=~np.asarray(o['pad_mask'])
    rows[~mask]=0
    return np.concatenate(([e.px/C.FIELD_W,e.py/C.FIELD_H,e.pvx/200,e.pvy/200,e.frame/7200],rows.ravel(),mask)).astype(np.float32)

class LearnedEnv(Gv20Env):
    def __init__(self,callback,**kw):
        self.callback=callback
        super().__init__(**kw)
    def _spawn_one(self):
        b,s,f=np.asarray(self.callback(observation(self)),dtype=np.float64)
        if not all(math.isfinite(v) for v in (b,s,f)) or not 30<=s<=200: raise ValueError('invalid physical action')
        pos=ray_position(b)
        self.bullets.append(Bullet(*pos,s*math.cos(f),s*math.sin(f)))
        self.s_history.append((self.frame,float(s)))

def timed_returns(frames,end):
    f=np.asarray(frames)
    return -np.diff(np.append(f,end))/7200.,-(end-f)/7200.

def load_spawner(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('architecture')!=ARCHITECTURE or ck.get('architecture_version')!=1: raise ValueError('not Sv23')
    net=Spawner(); net.load_state_dict(ck['model'],strict=True)
    return net.eval(),ck
