"""Reuse v21 numerical routines; inject only the frozen opponent environment."""
import types
import torch
from qrokkun_ai.v21.train import train_player_v21 as v21
from qrokkun_ai.v21.agents.player_v21 import state_dict_sha256
from qrokkun_ai.v24.core import LearnedEnv
from qrokkun_ai.v24 import run as v24


def check_frozen(s, digest):
    if s.training or any(p.requires_grad or p.grad is not None for p in s.parameters()):
        raise ValueError('opponent no longer frozen')
    if any(p.device.type != 'cpu' for p in s.parameters()) or state_dict_sha256(s.state_dict()) != digest:
        raise ValueError('opponent state/device changed')


def optimizers(p, guard):
    opts = (torch.optim.Adam(p.actor_parameters(), lr=v21.ACTOR_LR, eps=1e-8),
            torch.optim.Adam(p.critic_parameters(), lr=v21.CRITIC_LR, eps=1e-8))
    for opt in opts:
        opt.register_step_pre_hook(lambda *_: guard.check())
    return opts


def environment(s, seed, guard):
    rng = torch.Generator(device='cpu').manual_seed(seed + 25000000)

    def callback(obs):
        guard.check()
        with torch.no_grad():
            action, _ = s.sample(torch.from_numpy(obs).unsqueeze(0), rng)
        return action[0].numpy()

    class GuardedEnv(LearnedEnv):
        def step(self, action):
            guard.check()
            return super().step(action)

    return GuardedEnv(callback, seed=seed)


def collect(p, s, device, seeds, cap, guard):
    # A private globals dictionary avoids mutating the imported trainer or global env.
    namespace = dict(v21.collect_rollouts_v21.__wrapped__.__globals__)
    namespace['Gv20Env'] = lambda seed: environment(s, seed, guard)
    original = v21.collect_rollouts_v21.__wrapped__
    collector = types.FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    collector.__kwdefaults__ = original.__kwdefaults__
    with torch.no_grad():
        return collector(p, device, seeds, max_steps=cap)


def evaluate(p, s, seeds, cap, device, guard, on_complete=None):
    # Exact v24 active ordering, player_actions, physical mapping and per-seed +24M RNG.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    was_training = p.training
    try:
        with torch.random.fork_rng(devices=devices):
            p.eval()
            records, _, _ = v24.episodes(s, p, list(seeds), cap, device, guard, on_complete=on_complete)
            guard.check()
            return v24.summary(records)
    finally:
        p.train(was_training)
