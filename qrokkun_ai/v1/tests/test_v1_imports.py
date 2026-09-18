import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    (
        "qrokkun_ai.v1.train.player_v1",
        "qrokkun_ai.v1.train.player_v1_long",
        "qrokkun_ai.v1.train.spawner_v1",
        "qrokkun_ai.v1.train.both_v1",
    ),
)
def test_v1_historical_trainers_import(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None
