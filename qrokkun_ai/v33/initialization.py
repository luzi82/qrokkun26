"""Only learner initialization differs from the warm control."""
import torch
from qrokkun_ai.v21.agents.player_v21 import PlayerV21
from qrokkun_ai.v24.core import load_spawner, file_hash, PLAYER_SHA, STATE_SHA, ARCHITECTURE
from qrokkun_ai.v25.core import S_SHA
from qrokkun_ai.v26.run import S_STATE
from qrokkun_ai.v21.agents.player_v21 import state_dict_sha256


def random_player(hidden=472):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(17)
        return PlayerV21(hidden=hidden)


def initialization_metadata(sha):
    return dict(method='PlayerV21 default nn.Linear all actor/critic weights and biases',
                seed=17, device='cpu', rng_isolated=True, u0_raw_state_sha=sha)


def locked_spawner(path):
    if file_hash(path) != S_SHA:
        raise ValueError('locked Sv24 file hash mismatch')
    s, m = load_spawner(path)
    if (m.get('completed_updates') != 980 or m.get('attempted_updates') != 980 or
        m.get('partial_update') is not False or m.get('player_sha') != PLAYER_SHA or
        m.get('player_state_sha') != STATE_SHA or m.get('architecture') != ARCHITECTURE or
        state_dict_sha256(s.state_dict()) != S_STATE):
        raise ValueError('Sv24 metadata/state/original lineage mismatch')
    return s.requires_grad_(False).eval()
