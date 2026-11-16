"""Exclusive direct-child supervisor. No retry, resume, or final replay."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from . import run
from .final import preflight


def execute(a, child_argv=None, poll=.1):
    run.validate(a)
    out = Path(a.out_dir)
    launch = out.with_name(out.name+'.launcher')
    launch.mkdir(parents=True,exist_ok=False)
    child = None
    cleanup_incomplete = False
    stops, errors, exits = [], [], []
    handlers = {}
    state = 'starting'
    def stop(sig,*_):
        stops.append(sig)
    def reap():
        nonlocal cleanup_incomplete
        if child is not None and not cleanup_incomplete and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired as exc:
                errors.append('TERM deadline: '+repr(exc))
                child.kill()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    cleanup_incomplete = True
                    exits.append(dict(stage=stage,pid=child.pid,code=child.returncode,cleanup_incomplete=True))
                    raise
    try:
        for sig in (signal.SIGINT,signal.SIGTERM):
            handlers[sig] = signal.signal(sig,stop)
        if out.exists():
            raise ValueError('occupied run directory')
        for stage in ('train','final'):
            if stops:
                raise InterruptedError('owner stop')
            if stage == 'final':
                if a.nonformal:
                    state = 'nonformal_complete'
                    break
                preflight(a,lambda:bool(stops))
            args = [sys.executable,'-m','qrokkun_ai.v35.run',stage]
            for k,v in vars(a).items():
                if k in ('command','arm','resume') or v is None:
                    continue
                if isinstance(v,bool):
                    if v:
                        args.append('--'+k.replace('_','-'))
                else:
                    args.extend(['--'+k.replace('_','-'),str(v)])
            if child_argv:
                args = child_argv(stage)
            with (launch/(stage+'.stdout')).open('ab',buffering=0) as log:
                if stops:
                    raise InterruptedError('owner stop before spawn')
                child = subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
                while child.poll() is None and not stops:
                    time.sleep(poll)
                if stops:
                    reap()
                code = child.wait()
                exits.append(dict(stage=stage,pid=child.pid,code=code))
                if code:
                    errors.append(stage+' child exit '+str(code))
                if stops or code:
                    raise RuntimeError('child stopped or failed')
        else:
            state = 'complete'
    except BaseException as exc:
        errors.append(repr(exc))
        state = 'stopped' if stops else 'failed'
    finally:
        try:
            reap()
        except BaseException as exc:
            errors.append('cleanup: '+repr(exc))
            state = 'failed'
        for sig,h in handlers.items():
            signal.signal(sig,h)
        run.json_write(launch/'supervisor.json',dict(status=state,errors=errors,owner_stop=stops,child_exits=exits))
    return 0 if state in ('complete','nonformal_complete') else 1


def main(argv=None):
    a = run.parser().parse_args(['train',*(sys.argv[1:] if argv is None else argv)])
    return execute(a)

if __name__ == '__main__':
    raise SystemExit(main())
