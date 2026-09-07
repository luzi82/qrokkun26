#!/usr/bin/env python3
"""Back-compat shim → `qrokkun_env.train.both_v2`."""

from qrokkun_env.train.both_v2 import *  # noqa: F403
from qrokkun_env.train.both_v2 import main

if __name__ == "__main__":
    main()
