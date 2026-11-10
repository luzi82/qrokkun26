"""Sv22 public-state spawner and finite-horizon credit, independent of P internals."""
import hashlib
import math
import numpy as np
import torch
from torch import nn
from qrokkun_env import constants as C
from qrokkun_env.gv20 import Gv20Env, ray_position, learned_speed, learned_aim_delta
from qrokkun_ai.v21.agents.player_v21 import load_v21_checkpoint

PLAYER_SHA='b8466bb13972de62003e16b0b570567a016d0dab4dd5667e66a3666974cfd5f8'
STATE_SHA='250f3c3f1053b18129f992e97cb9ae15139fc2e5fa962b5491db5dddb2eb8f91'
PLAYER_PATH='/mnt/minion/88_minion_rw/projects/qrokkun26/runs/20261109-1803-v21-exploratory/best_player_v21.pt'
OBS_DIM=325

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def load_player(path, expected=PLAYER_SHA, device='cpu'):
    if expected != PLAYER_SHA or file_hash(path)!=expected:
        raise ValueError('owner player file identity mismatch')
    p,m=load_v21_checkpoint(path,device)
    if m['extra']['update']!=400 or m['state_dict_sha256']!=STATE_SHA:
        raise ValueError('owner update/state mismatch')
    p.requires_grad_(False)
    return p.eval(),m

def observation(e):
    o=e.observe()
    rows=np.asarray(o['bullets'],dtype=np.float32)
    mask=~np.asarray(o['pad_mask'])
    rows[~mask]=0
    return np.concatenate(([e.px/C.FIELD_W,e.py/C.FIELD_H,
        e.pvx/200,e.pvy/200,e.frame/7200],rows.ravel(),mask)).astype(np.float32)

def mapping(raw):
    a=np.asarray(raw,dtype=np.float64)
    if a.shape!=(5,) or not np.isfinite(a).all(): raise ValueError('five finite raw actions required')
    return (ray_position(math.pi*(math.tanh(a[0])+1)),
            learned_aim_delta(float(a[2]),(float(a[3]),float(a[4]))),
            learned_speed(math.tanh(float(a[1]))))

class LearnedEnv(Gv20Env):
    def __init__(self,callback,**kw):
        self.callback=callback
        super().__init__(**kw)
    def _spawn_one(self):
        pos,delta,speed=mapping(self.callback(observation(self)))
        self._add_bullet(pos,delta,speed)
        self.s_history.append((self.frame,speed))

def timed_returns(frames,end):
    f=np.asarray(frames)
    return -np.diff(np.append(f,end))/7200., -(end-f)/7200.

class Spawner(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(OBS_DIM,128),nn.Tanh(),nn.Linear(128,5))
        self.critic=nn.Sequential(nn.Linear(OBS_DIM,128),nn.Tanh(),nn.Linear(128,1))
        nn.init.zeros_(self.actor[-1].weight)
        with torch.no_grad(): self.actor[-1].bias.copy_(torch.tensor([0.,0.,-3.,1.,0.]))
        self.log_std=nn.Parameter(torch.tensor([math.log(1.5),math.log(.6),-.7,-.7,-.7]))
    def distribution(self,x):
        return torch.distributions.Normal(self.actor(x),self.log_std.clamp(-5,2).exp())
    def value(self,x): return self.critic(x).squeeze(-1)
