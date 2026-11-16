"""Bounded replay checkpoints; planner never sees future random streams."""
from collections import deque
import time
import numpy as np
import torch
from .world import LearnedEnv, snapshot, restore
from .planner import search
from qrokkun_ai.v31.core import sample_spawner_action
from qrokkun_ai.v35.adapter import encode_obs

class Sampler:
    def __init__(self, model, state):
        self.model = model
        self.generator = torch.Generator().set_state(state.clone())
    def __call__(self, obs):
        return sample_spawner_action(self.model, obs, self.generator, False)

class CaptureEnv(LearnedEnv):
    def __init__(self, callback, **kw):
        self.capture = None
        super().__init__(callback, **kw)
    def step(self, action):
        if self.capture is not None: self.capture.before(self, action)
        return super().step(action)

class Capture:
    def __init__(self, model, generator):
        self.model, self.generator = model, generator
        self.checkpoints = deque(maxlen=2)
        self.actions = deque(maxlen=128)
    def before(self, env, action):
        if env.frame % 64 == 0:
            self.checkpoints.append((env.frame, snapshot(env), self.generator.get_state().clone()))
        self.actions.append((env.frame, int(action)))
    def anchor(self, frame):
        f, state, rng = max((c for c in self.checkpoints if c[0] <= frame), key=lambda x:x[0])
        branch = restore(state, Sampler(self.model, rng))
        for at, action in self.actions:
            if f <= at < frame: branch.step(action)
        assert branch.frame == frame and not branch.dead
        return branch

def fork(env):
    return restore(snapshot(env), Sampler(env.callback.model, env.callback.generator.get_state()))

def validate(anchor, candidates):
    # Freeze the complete candidate list before any full-world replay.
    import copy
    candidates = copy.deepcopy(candidates)
    validated = 0
    for candidate in candidates:
        if len(candidate['actions']) != 60 or len(candidate['positions']) != 61:
            raise ValueError('candidate must contain exactly 60 actions and 61 positions')
        # Independent no-spawn real integrator check, before full-world branch.
        from qrokkun_env.gv36 import Gv36Env
        mover = Gv36Env(seed=0)
        mover.restore_player_state(anchor.get_player_state())
        for action, position in zip(candidate['actions'], candidate['positions'][1:]):
            mover._integrate_player(*mover._parse_action(action))
            if [mover.px, mover.py] != position: raise ValueError('planner position mismatch')
        branch = fork(anchor)
        rows = []
        validated += 1
        for action in candidate['actions']:
            obs = encode_obs(branch)
            rows.append((*obs, action))
            if branch.step(action)[2]: break
        else:
            if len(rows) != 60: raise ValueError('path must survive all 60 steps')
            return rows, validated
    return [], validated

def finish(env):
    metrics = dict(deathcount=int(env.dead), searched=0, skippedearly=0,
                   censored=int(not env.dead), candidateN=0, modelcompletecaseN=0,
                   validatedpathN=0, deathwithsuccessN=0, demoframes=0,
                   search_s=0., replay_s=0., validate_s=0.)
    if not env.dead: return [], metrics
    if env.frame < 12:
        metrics['skippedearly'] = 1
        return [], metrics
    started = time.perf_counter()
    anchor = env.capture.anchor(env.frame-12)
    metrics['replay_s'] = time.perf_counter()-started
    bullets = [[b.x,b.y,b.vx,b.vy,b.radius,b.moved] for b in anchor.bullets]
    result = search(anchor.get_player_state(), bullets)
    metrics.update({k:result[k] for k in ('candidateN','modelcompletecaseN','search_s')})
    metrics['searched'] = 1
    started = time.perf_counter()
    rows, n = validate(anchor, result['candidates'])
    metrics.update(validatedpathN=n, deathwithsuccessN=int(bool(rows)), demoframes=len(rows),
                   validate_s=time.perf_counter()-started)
    return rows, metrics

def auxiliary(net, optimizer, rows, update, weight=.1):
    started = time.perf_counter()
    result = dict(auxsteps=0, auxloss=0., auxnorm=0., samples=0, aux_s=0.)
    if not rows or weight == 0: return result
    gen = torch.Generator().manual_seed(37000017+update)
    indices = torch.randperm(len(rows), generator=gen)[:512].tolist() if len(rows)>512 else range(len(rows))
    chosen = [rows[i] for i in indices]
    device = next(net.parameters()).device
    pl, bu, pad, actions = [torch.as_tensor(np.stack(x), device=device) for x in zip(*chosen)]
    optimizer.zero_grad(set_to_none=True)
    dist = net.forward_policy(pl, bu, pad)
    loss = -dist.log_prob(actions.long()).mean()*weight
    if not torch.isfinite(loss): raise FloatingPointError('aux loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(net.actor_parameters(), 1.)
    if not torch.isfinite(norm): raise FloatingPointError('aux gradient')
    optimizer.step()
    result.update(auxsteps=1, auxloss=float(loss.detach()), auxnorm=float(norm), samples=len(chosen), aux_s=time.perf_counter()-started)
    return result
