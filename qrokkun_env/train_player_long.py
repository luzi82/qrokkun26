#!/usr/bin/env python3
"""Back-compat shim → `qrokkun_env.train.player_v1_long`."""

from qrokkun_env.train.player_v1_long import *  # noqa: F403
from qrokkun_env.train.player_v1_long import main

if __name__ == "__main__":
    main()
