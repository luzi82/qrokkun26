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
    def forbidden_pool(*args, **kwargs):
        raise AssertionError("CPU must remain serial")
    monkeypatch.setattr(core, "ThreadPoolExecutor", forbidden_pool)
    metrics = exercise('cpu', monkeypatch)
    assert metrics["host_dispatch_workers"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA concurrency NOT verified on CPU')
def test_cuda_callable_stream_path(monkeypatch):
    core.configure_runtime('cuda')
    metrics = exercise('cuda', monkeypatch)
    assert metrics['cuda_streams_enabled'] is True
    assert torch.cuda.current_stream().query()


def test_paired_dispatch_submits_both_before_waiting():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    barrier = Barrier(2)
    def work(value):
        barrier.wait(timeout=3)
        return value
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert core._dispatch_pair(executor, lambda: work("actor"), lambda: work("critic")) == ("actor", "critic")


def test_paired_dispatch_drains_after_base_exception():
    events = []
    class Future:
        def __init__(self, lane):
            self.lane = lane
        def done(self):
            return True
        def result(self):
            events.append(("finished", self.lane))
            if self.lane == 0:
                raise KeyboardInterrupt("actor")
            return 1
    class Executor:
        def submit(self, fn):
            lane = len(events)
            events.append(("submitted", lane))
            return Future(lane)
    with pytest.raises(KeyboardInterrupt, match="actor"):
        core._dispatch_pair(Executor(), lambda: None, lambda: None)
    assert events == [("submitted", 0), ("submitted", 1), ("finished", 0), ("finished", 1)]


def test_paired_dispatch_drains_if_second_submission_fails():
    events = []
    class Future:
        def done(self):
            return True
        def result(self):
            events.append("finished")
    class Executor:
        def submit(self, fn):
            if events:
                raise RuntimeError("submit")
            events.append("submitted")
            return Future()
    with pytest.raises(RuntimeError, match="submit"):
        core._dispatch_pair(Executor(), lambda: None, lambda: None)
    assert events == ["submitted", "finished"]


def test_join_streams_syncs_rest_and_reraises_first():
    seen = []

    class Stream:
        def __init__(self, label, error):
            self.label = label
            self.error = error

        def synchronize(self):
            seen.append(self.label)
            raise self.error

    first, second = RuntimeError("first"), RuntimeError("second")
    with pytest.raises(RuntimeError, match="^first$") as caught:
        core._join_streams([Stream("0", first), Stream("1", second)])
    assert caught.value is first
    assert seen == ["0", "1"]

    cuda_err = RuntimeError("cuda_fail")
    with pytest.raises(RuntimeError, match="^cuda_fail$") as caught_cuda:
        core._join_streams([Stream("0", KeyboardInterrupt()), Stream("1", cuda_err)])
    assert caught_cuda.value is cuda_err


def test_train_joins_both_streams_and_keeps_primary_failure(monkeypatch):
    net = PlayerV21(hidden=4)
    n = 2
    zeros = torch.zeros(n)
    batch = {
        "player": torch.zeros(n, 6),
        "bullets": torch.zeros(n, 12, 4),
        "pad": torch.zeros(n, 12, dtype=torch.bool),
        "actions": torch.zeros(n, dtype=torch.int64),
        "old_log_probs": zeros,
        "values": zeros,
        "returns": zeros,
        "advantages": torch.zeros(n),
        "expert_actions": torch.zeros(n, dtype=torch.int64),
        "expert_logits": torch.zeros(n, 9),
    }
    actor, critic = core.optimizers(net)
    generator = torch.Generator().manual_seed(0)

    # Exact post-backward join failure: stream 0 raises on synchronize(), stream 1 still syncs,
    # neither optimizer updates, and the first synchronize() error object is preserved across finally.
    sync_seen = []
    first_sync_err = RuntimeError("actor_stream_sync_1")

    class StickyActorStream:
        def __init__(self):
            self.calls = 0

        def synchronize(self):
            self.calls += 1
            sync_seen.append(("0", self.calls))
            raise first_sync_err if self.calls == 1 else RuntimeError("actor_stream_sync_sticky")

    class HealthyCriticStream:
        def synchronize(self):
            sync_seen.append(("1", 1))

    monkeypatch.setattr(core, "_make_streams", lambda _device: [StickyActorStream(), HealthyCriticStream()])
    with pytest.raises(RuntimeError) as caught_sync:
        core.train(net, actor, critic, batch, generator=generator, minibatch=n)
    assert caught_sync.value is first_sync_err
    assert ("0", 1) in sync_seen and ("1", 1) in sync_seen
    assert not actor.state and not critic.state

    def run(primary):
        seen = []

        class Stream:
            def __init__(self, label):
                self.label = label

            def synchronize(self):
                seen.append(self.label)
                raise RuntimeError("cuda")

        monkeypatch.setattr(core, "_make_streams", lambda _device: [Stream("0"), Stream("1")])

        def fail(_feat):
            raise primary

        monkeypatch.setattr(net.policy_body, "forward", fail)
        try:
            core.train(net, actor, critic, batch, generator=generator, minibatch=n)
        except BaseException as exc:
            return exc, seen
        raise AssertionError("train swallowed the failure")

    step = RuntimeError("step")
    got, seen = run(step)
    assert got is step and seen == ["0", "1"]
    for primary in (InterruptedError("owner stop"), KeyboardInterrupt()):
        got, seen = run(primary)
        assert type(got) is RuntimeError and str(got) == "cuda" and seen == ["0", "1"]
