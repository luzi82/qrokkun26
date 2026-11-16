"""Paired 84->344 input embedding; fresh Adam, never a trained P warm start."""
import torch
from qrokkun_ai.v33.initialization import random_player as old_random_player, locked_spawner
from qrokkun_ai.v21.agents.player_v21 import state_dict_sha256
from .model import PlayerV35

MAPPING = [(0,54,0,54),(54,66,262,274),(66,84,326,344)]
OLD_SHA = '49a7c9426a1054fc000caf359fc1a0a3c90db49adfe6a53c3275350a9b5a7942'

def random_player(hidden=472):
    old = old_random_player(hidden)
    new = PlayerV35(hidden)
    sd = old.state_dict()
    for key in ('policy_body.0.weight','value_body.0.weight'):
        expanded = torch.zeros(hidden,344,dtype=sd[key].dtype)
        for a,b,c,d in MAPPING:
            expanded[:,c:d] = sd[key][:,a:b]
        sd[key] = expanded
    new.load_state_dict(sd, strict=True)
    return new

def initialization_metadata(sha, hidden=472):
    return dict(schema='pv35_paired_embedding_v1', seed=17, device='cpu',rng_isolated=True,
                old_state_sha=state_dict_sha256(old_random_player(hidden).state_dict()),
                new_state_sha=sha, mapping=MAPPING, zero_columns=[[54,262],[274,326]],
                method='Pv34 u0 default fan_in84 embedded into fan_in344; all later parameters exact',
                caveat='expanded GEMM may round differently; no exact logits or trajectory claim',
                fresh_adam=True)
