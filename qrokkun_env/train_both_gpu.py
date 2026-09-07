#!/usr/bin/env python3
"""Back-compat shim → `qrokkun_env.train.both_v1`."""

from qrokkun_env.train.both_v1 import *  # noqa: F403
from qrokkun_env.train.both_v1 import main

if __name__ == "__main__":
    main()
