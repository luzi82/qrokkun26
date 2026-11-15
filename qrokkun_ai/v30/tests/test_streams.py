import pytest
import torch
from qrokkun_ai.v30 import core
from qrokkun_ai.v21.agents.player_v21 import PlayerV21
from qrokkun_ai.v24.core import Spawner


def exercise(device, monkeypatch):
    torch.manual_seed(17)
    p = PlayerV21(hidden=8)
    s = Spawner().requires_grad_(False)
    eps, _ = core.collect_serial(p, s, [100000], 3)
    p.to(device)
    batch = core.assemble_batch(eps, p, device)
    a, c = core.optimizers(p)
    assert not a.state and not c.state
    before = {k: v.clone() for k, v in p.state_dict().items()}
    categorical = torch.distributions.Categorical
    def distribution(*args, **kwargs):
        assert kwargs.get('validate_args') is False
        return categorical(*args, **kwargs)
    monkeypatch.setattr(torch.distributions, 'Categorical', distribution)
    original = core.finite
    checks = []
    def guard(state):
        # Both backwards must precede the first host finite guard; each guard
        # reads only its own partition, before either optimizer updates.
        assert all(x.grad is not None for x in p.parameters())
        assert len(state) == len(p.actor_parameters())
        checks.append(tuple(state))
        original(state)
    monkeypatch.setattr(core, 'finite', guard)
    metrics = core.train(p, a, c, batch, generator=torch.Generator(device=device).manual_seed(17), minibatch=3)
    assert checks and a.state and c.state
    assert core.weight_diff(before, p.state_dict())
    assert metrics['ppo_backend'] == ('cuda_actor_critic_streams' if device == 'cuda' else 'cpu_serial')
    assert all(torch.isfinite(x).all() for x in p.parameters())
    return metrics


def test_partitioned_updates_and_join_before_return_cpu(monkeypatch):
    exercise('cpu', monkeypatch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA concurrency NOT verified on CPU')
def test_cuda_callable_stream_path(monkeypatch):
    core.configure_runtime('cuda')
    metrics = exercise('cuda', monkeypatch)
    assert metrics['cuda_streams_enabled'] is True
    assert torch.cuda.current_stream().query()
