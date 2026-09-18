"""Observation vectorizer."""

from qrokkun_env.env import Qrokkun26Env
from qrokkun_ai.obs import OBS_DIM, vectorize


def test_obs_dim():
    env = Qrokkun26Env(seed=0)
    env.reset(seed=0)
    assert len(vectorize(env)) == OBS_DIM
