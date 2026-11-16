"""64 raw bullet inputs; legacy top_k is exclusively the teacher argument."""
import torch
from qrokkun_ai.v21.agents.player_v21 import PlayerV21

ARCHITECTURE = 'player_v35_input64_clearance12_decoupled'

class PlayerV35(PlayerV21):
    architecture = ARCHITECTURE
    architecture_version = 1
    policy_input_top_k = 64
    teacher_top_k = 12
    clearance_top_k = 12

    def __init__(self, hidden=472):
        # Isolate all construction, including the replaced first layers.
        with torch.random.fork_rng(devices=[]):
            super().__init__(top_k=12, hidden=hidden)
            self.in_dim = 344
            self.policy_body[0] = torch.nn.Linear(344, hidden)
            self.value_body[0] = torch.nn.Linear(344, hidden)

    def features(self, player, bullets, pad):
        legacy = super().features(player, bullets, pad)
        live = (~pad[:, :64]).to(player.dtype)
        return torch.cat((player, (bullets[:, :64] * live.unsqueeze(-1)).flatten(1),
                          live, legacy[:, -18:]), dim=-1)

    def architecture_metadata(self):
        return {**super().architecture_metadata(),
                'policy_input_top_k':64, 'teacher_top_k':12, 'clearance_top_k':12,
                'top_k_role':'legacy teacher argument only',
                'actor_input_dim':344, 'critic_input_dim':344}

    @classmethod
    def from_metadata(cls, metadata):
        if not isinstance(metadata, dict) or type(metadata.get('hidden')) is not int:
            raise ValueError('v35 architecture metadata required')
        net = cls(hidden=metadata['hidden'])
        if metadata != net.architecture_metadata():
            raise ValueError('v35 architecture mismatch')
        return net
