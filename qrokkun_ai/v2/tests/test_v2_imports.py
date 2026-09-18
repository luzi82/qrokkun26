import torch

from qrokkun_env.env import ACTIONS
from qrokkun_ai.v2.agents.player_v2 import OBS_DIM_V2, PlayerV2
from qrokkun_ai.v2.train import both_v2


def test_player_v2_is_defined_by_v2_not_an_import_alias() -> None:
    assert PlayerV2.__module__ == "qrokkun_ai.v2.agents.player_v2"


def test_player_v2_preserves_checkpoint_tensor_contract() -> None:
    model = PlayerV2(hidden=32)
    distribution, value = model(torch.zeros(OBS_DIM_V2))
    assert OBS_DIM_V2 == 45
    assert distribution.logits.shape == (len(ACTIONS),)
    assert value.shape == ()


def test_v2_both_trainer_imports() -> None:
    assert both_v2.PlayerAC is PlayerV2
