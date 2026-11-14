"""Pinned CPU episode workers; central P sampling and frame barrier."""
import hashlib
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from qrokkun_ai.v24.core import LearnedEnv, Spawner
from qrokkun_ai.v21.agents.player_v21 import state_dict_sha256
from qrokkun_ai.v21.obs_v21 import encode_obs_v21, compute_kinematic_expert_targets
from qrokkun_ai.v21.train import train_player_v21 as v21


def cpu_budget():
    limits={'logical':os.cpu_count() or 1}
    if hasattr(os,'sched_getaffinity'): limits['affinity']=len(os.sched_getaffinity(0))
    # Resolve this process's cgroup, including nested quota constraints.
    groups=Path('/proc/self/cgroup').read_text().splitlines() if Path('/proc/self/cgroup').exists() else []
    for line in groups:
        _,controllers,relative=line.split(':',2)
        base=Path('/sys/fs/cgroup')
        if controllers and 'cpu' not in controllers.split(','): continue
        if controllers: base=base/'cpu'
        current=base/relative.lstrip('/')
        while current==base or base in current.parents:
            try:
                if (current/'cpu.max').exists():
                    quota,period=(current/'cpu.max').read_text().split()
                    if quota!='max': limits[str(current/'cpu.max')]=max(1,math.floor(int(quota)/int(period)))
                elif (current/'cpu.cfs_quota_us').exists():
                    quota=int((current/'cpu.cfs_quota_us').read_text())
                    if quota>0: limits[str(current)]=max(1,quota//int((current/'cpu.cfs_period_us').read_text()))
            except (OSError,ValueError): pass
            if current==base: break
            current=current.parent
    return min(limits.values()),limits


class Env(LearnedEnv):
    def __init__(self,s,seed,validation):
        self.sg=torch.Generator().manual_seed(seed+(24000000 if validation else 25000000))
        def callback(obs):
            with torch.no_grad(): action,_=s.sample(torch.from_numpy(obs).unsqueeze(0),self.sg)
            return action[0].numpy()
        super().__init__(callback,seed=seed)
        self.reset(seed=seed)


def step(e,action):
    _,base,done,info=e.step(int(action))
    return encode_obs_v21(e),v21.shaped_reward_v21(e,float(base),bool(done)),bool(done),float(info.get('elapsed',e.elapsed))


def worker(conn,state,assignments,validation):
    torch.set_num_threads(1)
    started=time.process_time(); started_wall=time.perf_counter(); tasks=0
    try:
        s=Spawner(); s.load_state_dict(state); s.requires_grad_(False).eval()
        frozen=state_dict_sha256(s.state_dict())
        envs={i:Env(s,seed,validation) for i,seed in assignments}
        conn.send(('ok',{i:encode_obs_v21(e) for i,e in envs.items()}))
        with torch.no_grad():
            while True:
                commands=conn.recv()
                if commands is None:
                    if state_dict_sha256(s.state_dict())!=frozen: raise RuntimeError('worker S mutated')
                    conn.send(('ok',dict(pid=os.getpid(),tasks=tasks,cpu_s=time.process_time()-started,wall_s=time.perf_counter()-started_wall,episodes=assignments,frozen_S_verified=True,cuda_initialized=torch.cuda.is_initialized())))
                    break
                result={}
                for i,a in commands:
                    result[i]=step(envs[i],a); tasks+=1
                conn.send(('ok',result))
    except BaseException:
        try: conn.send(('error',traceback.format_exc()))
        except (BrokenPipeError,EOFError): pass
    finally: conn.close()


def receive(conn,process,check):
    while not conn.poll(.05):
        check()
        if not process.is_alive(): raise RuntimeError('worker exited '+str(process.exitcode))
    kind,data=conn.recv()
    if kind!='ok': raise RuntimeError(data)
    return data


@torch.no_grad()
def collect(p,s,device,seeds,cap,parallel,validation=False,check=lambda:None):
    mode=p.training; p.eval(); children=[]; stats=[]; envs=[]
    count=min(len(seeds),cpu_budget()[0]) if parallel else 0
    ros=[v21.RolloutV21(seed=seed) for seed in seeds]
    generators=[torch.Generator(device=device).manual_seed(seed+26000000) for seed in seeds]
    obs={}; assignments=[]
    try:
        if parallel:
            ctx=mp.get_context('spawn')
            for k in range(count):
                assignment=[(i,seed) for i,seed in enumerate(seeds) if i%count==k]
                parent,child=ctx.Pipe(); proc=ctx.Process(target=worker,args=(child,s.state_dict(),assignment,validation))
                try: proc.start()
                except BaseException: parent.close(); child.close(); raise
                child.close(); children.append((parent,proc)); assignments.append({i for i,_ in assignment})
            for conn,proc in children: obs.update(receive(conn,proc,check))
        else:
            envs=[Env(s,seed,validation) for seed in seeds]
            obs={i:encode_obs_v21(e) for i,e in enumerate(envs)}
        active=list(range(len(seeds)))
        for frame in range(cap):
            check()
            if not active: break
            current=[obs[i] for i in active]
            pt,bt,mt=[torch.tensor(np.stack([o[j] for o in current]),device=device) for j in range(3)]
            dist,values=p(pt,bt,mt)
            expert,logits=compute_kinematic_expert_targets(pt,bt,mt,horizon=p.lookahead_horizon,top_k=p.top_k)
            acts=dist.logits.argmax(-1) if validation else torch.cat([torch.multinomial(dist.probs[j],1,generator=generators[i]) for j,i in enumerate(active)])
            lp=dist.log_prob(acts); actions=acts.cpu().tolist(); results={}
            if parallel:
                for (conn,_),assigned in zip(children,assignments): conn.send([(i,a) for i,a in zip(active,actions) if i in assigned])
                for conn,proc in children: results.update(receive(conn,proc,check))
            else: results={i:step(envs[i],a) for i,a in zip(active,actions)}
            remaining=[]
            for j,(i,a) in enumerate(zip(active,actions)):
                obs[i],reward,done,elapsed=results[i]; r=ros[i]
                r.player.append(current[j][0]); r.bullets.append(current[j][1]); r.pad.append(current[j][2])
                r.actions.append(a); r.log_probs.append(float(lp[j].item())); r.values.append(float(values[j].item()))
                r.expert_actions.append(int(expert[j].item())); r.expert_logits.append(logits[j].cpu().numpy())
                r.rewards.append(reward); r.dones.append(done); r.elapsed=elapsed; r.terminated=done
                if not done: remaining.append(i)
            active=remaining
        if active:
            tensors=[torch.tensor(np.stack([obs[i][j] for i in active]),device=device) for j in range(3)]
            values=p.forward_value(*tensors)
            for i,v in zip(active,values.cpu().tolist()):
                ros[i].truncated=ros[i].censored=True; ros[i].bootstrap_value=float(v)
        for conn,proc in children: conn.send(None)
        for conn,proc in children: stats.append(receive(conn,proc,check))
        return ros,stats
    finally:
        for conn,proc in children: conn.close()
        for _,proc in children:
            proc.join(timeout=1)
            if proc.is_alive(): proc.terminate(); proc.join(timeout=2)
            if proc.is_alive(): proc.kill(); proc.join()
            proc.close()
        p.train(mode)


def optimizers(p):
    a=torch.optim.Adam(p.actor_parameters(),lr=3e-4,eps=1e-8)
    c=torch.optim.Adam(p.critic_parameters(),lr=1e-3,eps=1e-8)
    assert not a.state and not c.state
    return a,c


def finite(state):
    for k,v in state.items():
        if not torch.isfinite(v).all(): raise FloatingPointError(k)


def weight_diff(a,b):
    return [k for k in sorted(a.keys()|b.keys()) if k not in a or k not in b or a[k].shape!=b[k].shape or a[k].dtype!=b[k].dtype or not torch.equal(a[k].detach().cpu().contiguous().view(torch.uint8),b[k].detach().cpu().contiguous().view(torch.uint8))]


def source_digest():
    root=Path(__file__).resolve().parents[2]; h=hashlib.sha256()
    for path in sorted([*root.glob('qrokkun_ai/**/*.py'),*root.glob('qrokkun_env/**/*.py')]):
        if 'tests' in path.parts or '__pycache__' in path.parts: continue
        h.update(str(path.relative_to(root)).encode()); h.update(path.read_bytes())
    return h.hexdigest()


def estimate(hidden,updates):
    params=2*((84+1)*hidden+(hidden+1)*hidden)+(hidden+1)*10
    return params*4*8*(updates+2)+64*1024**2


import torch.nn as nn
import torch.nn.functional as F
from qrokkun_ai.v21.agents.player_v21 import PlayerV21
def train(
    net: PlayerV21,
    opt_actor: torch.optim.Optimizer,
    opt_critic: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    generator,
    check=lambda:None,
    dagger_weight: float = 1.0,
    ppo_weight: float = 0.3,
    entropy_coef: float = 0.005,
    minibatch: int = v21.PPO_MINIBATCH,
) -> dict[str, float]:
    net.train()
    player, bullets, pad = batch["player"], batch["bullets"], batch["pad"]
    actions, old_lp = batch["actions"], batch["old_log_probs"]
    old_values, returns = batch["values"], batch["returns"]
    raw_adv = batch["advantages"]
    advantages = (raw_adv - raw_adv.mean()) / (raw_adv.std(unbiased=False) + 1e-8)
    exp_acts = batch["expert_actions"]
    exp_logits = batch["expert_logits"]

    actor_params = net.actor_parameters()
    critic_params = net.critic_parameters()

    n_samples = int(player.shape[0])
    pl_sum, vl_sum, dl_sum, kl_sum, agr_sum, seen = 0.0, 0.0, 0.0, 0.0, 0.0, 0

    for _ in range(v21.PPO_EPOCHS):
        check()
        perm = torch.randperm(n_samples, device=player.device, generator=generator)
        for start in range(0, n_samples, minibatch):
            mb = perm[start : start + minibatch]
            dist, v_pred = net(player[mb], bullets[mb], pad[mb])
            log_prob = dist.log_prob(actions[mb])
            ratio = torch.exp(log_prob - old_lp[mb])
            surr1 = ratio * advantages[mb]
            surr2 = torch.clamp(ratio, 1.0 - v21.PPO_CLIP, 1.0 + v21.PPO_CLIP) * advantages[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy = dist.entropy().mean()

            hard_ce = F.cross_entropy(dist.logits, exp_acts[mb])
            t_probs = F.softmax(exp_logits[mb], dim=-1)
            s_logp = F.log_softmax(dist.logits, dim=-1)
            soft_ce = -(t_probs * s_logp).sum(-1).mean()
            dagger_loss = 0.65 * hard_ce + 0.35 * soft_ce

            actor_loss = ppo_weight * policy_loss + dagger_weight * dagger_loss - entropy_coef * entropy
            opt_actor.zero_grad()
            actor_loss.backward()
            finite({n:v.grad for n,v in net.named_parameters() if v.grad is not None})
            nn.utils.clip_grad_norm_(actor_params, v21.PPO_MAX_GRAD_NORM)
            opt_actor.step()

            value_loss = F.mse_loss(v_pred, returns[mb])
            opt_critic.zero_grad()
            value_loss.backward()
            finite({n:v.grad for n,v in net.named_parameters() if v.grad is not None})
            nn.utils.clip_grad_norm_(critic_params, v21.PPO_MAX_GRAD_NORM)
            opt_critic.step()

            with torch.no_grad():
                approx_kl = float((old_lp[mb] - log_prob).mean().item())
                agr = float((dist.logits.argmax(dim=-1) == exp_acts[mb]).float().sum().item())
            bs = int(mb.shape[0])
            pl_sum += float(policy_loss.item()) * bs
            vl_sum += float(value_loss.item()) * bs
            dl_sum += float(dagger_loss.item()) * bs
            kl_sum += approx_kl * bs
            agr_sum += agr
            seen += bs

    with torch.no_grad():
        ret_var = torch.var(returns)
        ev = float(1.0 - torch.var(returns - old_values) / (ret_var + 1e-8))

    denom = max(seen, 1)
    return {
        "policy_loss": pl_sum / denom,
        "value_loss": vl_sum / denom,
        "dagger_loss": dl_sum / denom,
        "approx_kl": kl_sum / denom,
        "expert_agreement": agr_sum / denom,
        "explained_variance": ev,
        "n_samples": n_samples,
    }

