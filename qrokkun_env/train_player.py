#!/usr/bin/env python3
"""Back-compat shim → `qrokkun_env.train.player_v0`."""

from qrokkun_env.train.player_v0 import *  # noqa: F403
from qrokkun_env.train.player_v0 import main

if __name__ == "__main__":
    main()
