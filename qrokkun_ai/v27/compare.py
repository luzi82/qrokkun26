"""Offline bounded checkpoint comparison; no replay."""
import argparse
from pathlib import Path
from .supervise import compare
from qrokkun_ai.v26.cli import boundary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('directory')
    return compare(Path(p.parse_args(argv).directory))


def cli(argv=None): return boundary(main,argv)
if __name__=='__main__': raise SystemExit(cli())
