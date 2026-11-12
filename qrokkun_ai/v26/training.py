"""Same central active batch, barrier, canonical merge; v21 numerical PPO below."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext, contextmanager
from dataclasses import asdict
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qrokkun_ai.v21.train import train_player_v21 as v21
from qrokkun_ai.v21.obs_v21 import encode_obs_v21, compute_kinematic_expert_targets
from .kernel import Env, global_rng, sample_player


@contextmanager
def capture_forward(net, *, critic_only=False):
    """Observe existing module calls, without retaining or replacing autograd tensors."""
    captured={}
    body=net.value_body if critic_only else net.policy_body
    hooks=[body.register_forward_pre_hook(lambda mod,args:captured.update(features=args[0].detach().clone()))]
    if not critic_only:
        hooks.append(net.policy_head.register_forward_hook(lambda mod,args,out:captured.update(raw_logits=out.detach().clone())))
    try:yield captured
    finally:
        for hook in hooks:hook.remove()


def snapshot(p,oa,oc,generator):
    return dict(model=p.state_dict(),actor=oa.state_dict(),critic=oc.state_dict(),
                gradients={n:v.grad for n,v in p.named_parameters()},global_rng=global_rng(),
                training_rng=generator.get_state())


@torch.no_grad()
def collect(p,s,device,seeds,cap,workers,writer,update,validation=False,check=lambda:None):
    mode = p.training
    p.eval()
    phase = 'validation' if validation else 'train'
    envs = [Env(s,seed,validation,device) for seed in seeds]
    ros = [v21.RolloutV21(seed=seed) for seed in seeds]
    def emit(kind,key,data,audit=None):
        if writer:
            writer.emit(kind,key,data,audit)
    for i,e in enumerate(envs):
        emit('reset',(update,phase,i,0),dict(seed=seeds[i],state=e.state(),initial_generators=e.initial_generators,events=e.drain()))
    active = list(range(len(envs)))
    pool = ThreadPoolExecutor(max_workers=workers) if workers>1 else nullcontext(None)
    try:
        with pool as executor:
            for frame in range(cap):
                check()
                if not active:
                    break
                obs = [encode_obs_v21(envs[i]) for i in active]
                pt,bt,mt = [torch.tensor(np.stack([o[j] for o in obs]),device=device) for j in range(3)]
                rng_pre = global_rng()
                captured = []
                raw_logits = []
                raw_hook = p.policy_head.register_forward_hook(lambda mod,args,out:raw_logits.append(out.detach().clone()))
                hook = p.policy_body.register_forward_pre_hook(lambda mod,args:captured.append(args[0].detach().clone()))
                try:
                    dist, values = p(pt,bt,mt)
                finally:
                    hook.remove()
                    raw_hook.remove()
                expert, logits = compute_kinematic_expert_targets(pt,bt,mt,horizon=p.lookahead_horizon,top_k=p.top_k)
                if validation:
                    acts = dist.logits.argmax(-1)
                    draws = []
                else:
                    acts, draws = sample_player(dist.probs,[envs[i].pg for i in active],[frame]*len(active))
                lp = dist.log_prob(acts)
                emit('P',(update,phase,frame),dict(active=active,player=pt,bullets=bt,pad=mt,features=captured[0],
                     raw_logits=raw_logits[0],logits=dist.logits,probs=dist.probs,values=values,teacher_actions=expert,teacher_logits=logits,teacher_probs=F.softmax(logits,dim=-1),
                     actions=acts,logp=lp,draws=draws,global_pre=rng_pre,global_post=global_rng()))
                # Submit every step before awaiting any. No model forward, refill or optimizer here.
                actions = acts.cpu().tolist()
                if executor:
                    futures = [executor.submit(envs[i].traced_step,a) for i,a in zip(active,actions)]
                    results = [f.result() for f in futures]
                else:
                    results = [envs[i].traced_step(a) for i,a in zip(active,actions)]
                remaining = []
                for j,(i,a,(result,data,audit)) in enumerate(zip(active,actions,results)):
                    _,base,done,info = result
                    e,r = envs[i],ros[i]
                    reward = v21.shaped_reward_v21(e,float(base),bool(done))
                    data['reward'] = reward
                    emit('G',(update,phase,i,frame+1),data,audit)
                    r.player.append(obs[j][0]); r.bullets.append(obs[j][1]); r.pad.append(obs[j][2])
                    r.actions.append(a); r.log_probs.append(float(lp[j].item())); r.values.append(float(values[j].item()))
                    r.expert_actions.append(int(expert[j].item())); r.expert_logits.append(logits[j].cpu().numpy())
                    r.rewards.append(reward); r.dones.append(bool(done)); r.elapsed = float(info.get('elapsed',e.elapsed))
                    r.terminated = bool(done)
                    if not done:
                        remaining.append(i)
                active = remaining
        if active:
            obs = [encode_obs_v21(envs[i]) for i in active]
            tensors = [torch.tensor(np.stack([o[j] for o in obs]),device=device) for j in range(3)]
            with capture_forward(p,critic_only=True) as captured:
                values = p.forward_value(*tensors)
            emit('bootstrap',(update,phase),dict(active=active,player=tensors[0],bullets=tensors[1],pad=tensors[2],features=captured["features"],values=values))
            for i,v in zip(active,values.cpu().tolist()):
                ros[i].truncated = ros[i].censored = True
                ros[i].bootstrap_value = float(v)
        for i,(e,r) in enumerate(zip(envs,ros)):
            emit('episode_end',(update,phase,i),dict(seed=r.seed,frames=len(r.actions),terminated=r.terminated,
                 truncated=r.truncated,censored=r.censored,bootstrap=r.bootstrap_value,state=e.state(),
                 rng_draw_counts={n:getattr(e,n).counter for n in ('rng','s_rng','field_rng')},
                 S_draw_count=e.counter[0],P_draw_count=0 if validation else len(r.actions),
                 unused=[n for n in ('rng','s_rng','field_rng') if getattr(e,n).counter==0]))
        return ros
    finally:
        p.train(mode)


def train(p,oa,oc,batch,writer,update,generator,check=lambda:None):
    writer.emit('batch',(update,),dict(batch=batch,global_rng=global_rng(),training_rng=generator.get_state()))
    optstep = [0]
    epoch_id = [0]
    def permutation(n, device):
        check()
        pre = generator.get_state().clone()
        out = torch.randperm(n,device=device,generator=generator)
        writer.emit('permutation',(update,epoch_id[0]),dict(source='PPO.randperm',counter=epoch_id[0],pre=pre,
                    post=generator.get_state(),output=out,global_rng=global_rng()))
        epoch_id[0] += 1
        return out
    def record(kind, **data):
        check()
        for name,v in p.named_parameters():
            if not torch.isfinite(v).all() or (v.grad is not None and not torch.isfinite(v.grad).all()):
                raise FloatingPointError('nonfinite parameter/gradient '+name)
        writer.emit(kind,(update,optstep[0]),dict(**data,**snapshot(p,oa,oc,generator)))
        if kind == 'critic_poststep':
            optstep[0] += 1
    return _ppo(p,oa,oc,batch,permutation=permutation,record=record)


# v21 train_step_v21 numerical body copied verbatim; only RNG/trace hooks added.
def _ppo(
    net: PlayerV21,
    opt_actor: torch.optim.Optimizer,
    opt_critic: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    permutation, record,
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
        perm = permutation(n_samples, device=player.device)
        for start in range(0, n_samples, minibatch):
            mb = perm[start : start + minibatch]
            record("minibatch", indices=mb, normalized_advantages=advantages[mb])
            with capture_forward(net) as captured:
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
            record("actor_preclip", loss=actor_loss, losses=dict(policy=policy_loss,entropy=entropy,hard_ce=hard_ce,soft_ce=soft_ce,dagger=dagger_loss), teacher_probs=t_probs, logits=dist.logits, values=v_pred, **captured, probs=dist.probs)
            nn.utils.clip_grad_norm_(actor_params, v21.PPO_MAX_GRAD_NORM)
            record("actor_postclip", loss=actor_loss, losses=dict(policy=policy_loss,entropy=entropy,hard_ce=hard_ce,soft_ce=soft_ce,dagger=dagger_loss), teacher_probs=t_probs, logits=dist.logits, values=v_pred, **captured, probs=dist.probs)
            record("actor_prestep", loss=actor_loss, losses=dict(policy=policy_loss,entropy=entropy,hard_ce=hard_ce,soft_ce=soft_ce,dagger=dagger_loss), teacher_probs=t_probs, logits=dist.logits, values=v_pred, **captured, probs=dist.probs)
            opt_actor.step()
            record("actor_poststep", loss=actor_loss, losses=dict(policy=policy_loss,entropy=entropy,hard_ce=hard_ce,soft_ce=soft_ce,dagger=dagger_loss), teacher_probs=t_probs, logits=dist.logits, values=v_pred, **captured, probs=dist.probs)

            value_loss = F.mse_loss(v_pred, returns[mb])
            opt_critic.zero_grad()
            value_loss.backward()
            record("critic_preclip", loss=value_loss)
            nn.utils.clip_grad_norm_(critic_params, v21.PPO_MAX_GRAD_NORM)
            record("critic_postclip", loss=value_loss)
            record("critic_prestep", loss=value_loss)
            opt_critic.step()
            record("critic_poststep", loss=value_loss)

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
