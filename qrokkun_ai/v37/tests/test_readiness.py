"""Focused JSON pairing and cooperative post-PPO cancellation regressions."""
import copy
import json
import pickle
import signal
from types import SimpleNamespace

import pytest
import torch

from qrokkun_ai.v35.initialization import initialization_metadata, random_player
from qrokkun_ai.v37 import protocol


def test_control_initialization_json_roundtrip(tmp_path):
    # All identity fields remain present; use the real initializer's tuple mapping.
    keys = ('environment initial_player_state_sha spawner_state_sha inputs source_commit '
            'source_tree_sha updates episodes cap seed validation validation_seeds '
            'hidden hyperparameters runtime architecture fresh_adam player_mode_training '
            'player_mode_evaluation spawner_mode validation_boundaries').split()
    semantic = dict.fromkeys(keys, 'paired')
    semantic['initialization'] = initialization_metadata('initial-player-sha', 4)
    for name, data in {
        'manifest': dict(arm='Pv37-control', semantic=semantic),
        'status': dict(status='complete'),
        'closed': dict(completed_updates=2),
    }.items():
        (tmp_path / (name + '.json')).write_text(json.dumps(data))
    protocol.verify_control(tmp_path, semantic)


def test_control_rejects_changed_initialization(tmp_path):
    semantic = dict.fromkeys(('environment initial_player_state_sha spawner_state_sha inputs '
        'source_commit source_tree_sha updates episodes cap seed validation validation_seeds '
        'hidden hyperparameters runtime architecture fresh_adam player_mode_training '
        'player_mode_evaluation spawner_mode validation_boundaries').split(), 'paired')
    semantic['initialization'] = initialization_metadata('initial-player-sha', 4)
    control = copy.deepcopy(semantic)
    control['initialization']['mapping'][0] = (0, 53, 0, 53)
    for name, data in {
        'manifest': dict(arm='Pv37-control', semantic=control),
        'status': dict(status='complete'),
        'closed': dict(completed_updates=2),
    }.items():
        (tmp_path / (name + '.json')).write_text(json.dumps(data))
    with pytest.raises(ValueError, match='paired control mismatch: initialization'):
        protocol.verify_control(tmp_path, semantic)


def test_stop_at_ppo_return_skips_auxiliary_and_cleans_up(tmp_path, monkeypatch):
    from qrokkun_ai.v37 import run, teaching
    from qrokkun_ai.v24.core import Spawner
    from qrokkun_ai.v37.tests.test_teaching import demo_rows

    net = random_player(4)
    actor, critic = run.core.optimizers(net)
    before_optimizer = pickle.dumps(actor.state_dict())
    before_weights = copy.deepcopy(net.state_dict())
    calls = []
    ep = SimpleNamespace(frames=1, elapsed=0., demos=demo_rows(), teaching={})
    ep.summary = lambda: {}

    class Pool:
        pinned_cpus = []
        last_pure_rollout_s = 0.
        def __init__(self, **kwargs):
            self.stop_event = SimpleNamespace(set=lambda: calls.append('stop'))
        def sync_weights(self, *args, **kwargs): pass
        def collect(self, *args, **kwargs): return [ep]
        def close(self): calls.append('close')

    def trainer(*args, **kwargs):
        calls.append('ppo')
        signal.raise_signal(signal.SIGTERM)
        assert kwargs['stop_check']()
        return {}

    original_auxiliary = teaching.auxiliary
    def auxiliary(*args, **kwargs):
        calls.append('auxiliary')
        return original_auxiliary(*args, **kwargs)

    monkeypatch.setattr(protocol, 'archive_source', lambda commit, path: path.write_bytes(b'controlled-source-fixture'))
    monkeypatch.setattr(protocol, 'verify_control', lambda *args: None)
    monkeypatch.setattr(run, 'random_player', lambda *args: net)
    monkeypatch.setattr(run, 'PlayerV21', lambda **kwargs: random_player(4))
    monkeypatch.setattr(run, 'locked_spawner', lambda *args: Spawner().eval().requires_grad_(False))
    monkeypatch.setattr(run, 'file_hash', lambda *args: 'fixture')
    monkeypatch.setattr(run, 'EpisodePool', Pool)
    monkeypatch.setattr(run.core, 'optimizers', lambda *args: (actor, critic))
    monkeypatch.setattr(run.core, 'source_digest', lambda: 'fixture')
    monkeypatch.setattr(run.core, 'select_learner_cpu', lambda *args, **kwargs: None)
    monkeypatch.setattr(run.core, 'assemble_batch', lambda *args: None)
    monkeypatch.setattr(run.core, 'train', trainer)
    monkeypatch.setattr(teaching, 'auxiliary', auxiliary)
    monkeypatch.setattr(torch, 'save', lambda payload, path: path.write_bytes(b'fixture'))
    out = tmp_path / 'stopped'
    with pytest.raises(InterruptedError, match='owner stop'):
        run._train_main(['train', '--role', 'search', '--control-dir', 'fixture',
            '--out-dir', str(out), '--source-commit', 'a' * 40, '--nonformal',
            '--device', 'cpu', '--updates', '2', '--episodes', '3', '--workers', '2',
            '--validation', '2', '--pin-cores', 'off'])
    status = json.loads((out / 'status.json').read_text())
    assert status['owner_stop'] and status['status'] == 'stopped'
    assert status['phase'] == 'cleanup' and status['completed_updates'] == 0
    assert any('owner stop' in error for error in status['errors'])
    assert not (out / 'closed.json').exists()
    assert calls == ['ppo', 'stop', 'close']
    assert pickle.dumps(actor.state_dict()) == before_optimizer
    assert all(torch.equal(before_weights[k], v) for k, v in net.state_dict().items())


def test_control_rejects_runtime_mismatch_in_memory(monkeypatch):
    from pathlib import Path
    from qrokkun_ai.v31 import core

    keys = ('environment initial_player_state_sha spawner_state_sha inputs source_commit '
            'source_tree_sha updates episodes cap seed validation validation_seeds '
            'hidden hyperparameters initialization architecture fresh_adam player_mode_training '
            'player_mode_evaluation spawner_mode validation_boundaries').split()
    search = dict.fromkeys(keys, 'paired')
    search['runtime'] = core.configure_runtime('cpu')
    control = copy.deepcopy(search)
    documents = {'manifest.json': dict(arm='Pv37-control', semantic=control),
                 'status.json': dict(status='complete'),
                 'closed.json': dict(completed_updates=2)}
    monkeypatch.setattr(Path, 'read_text', lambda path: json.dumps(documents[path.name]))
    protocol.verify_control(Path('in-memory'), search)
    search['runtime']['learner']['device'] = 'cuda'
    with pytest.raises(ValueError, match='paired control mismatch: runtime'):
        protocol.verify_control(Path('in-memory'), search)


def test_close_attempts_all_cleanup_and_preserves_errors(monkeypatch):
    from qrokkun_ai.v37 import adapter

    calls = []
    def fail_close(self):
        calls.append('super.close')
        raise RuntimeError('v32 close full failure message')

    class Sidecar:
        def __init__(self, name): self.name = name
        def unlink(self):
            calls.append(self.name)
            raise OSError('unlink full failure message: ' + self.name)

    monkeypatch.setattr(adapter.LifecyclePool, 'close', fail_close)
    monkeypatch.setattr(adapter.Path, 'glob', lambda *args: [Sidecar('one.npz'), Sidecar('two.npz')])
    pool = object.__new__(adapter.EpisodePool)
    pool.children = []
    pool.root = 'in-memory'
    with pytest.raises(BaseExceptionGroup) as caught:
        pool.close()
    assert calls == ['super.close', 'one.npz', 'two.npz']
    # The runner persists cleanup failures using repr, without a full traceback.
    recorded = 'cleanup: ' + repr(caught.value)
    for message in ('v32 close full failure message', 'unlink full failure message: one.npz',
                    'unlink full failure message: two.npz'):
        assert message in recorded
