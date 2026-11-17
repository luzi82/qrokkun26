"""Public Pv38 search entry point."""
from qrokkun_ai.v37 import run

def main(argv=None):
    a = run.parser().parse_args(argv)
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    if '--role' in args and a.role != 'search':
        raise ValueError('Pv38 requires search role')
    if '--role' not in args:
        args += ['--role','search']
    a = run.parser().parse_args(args)
    if a.command == 'final':
        from .final import execute
        return execute(a)
    return run.train_main(args)

if __name__ == '__main__':
    raise SystemExit(run.boundary(main))
