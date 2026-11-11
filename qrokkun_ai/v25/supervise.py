"""Exclusive direct-child train -> final for Pv25."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from . import run


def execute(a, child_argv=None, poll=0.2):
    run.validate_args(a)
    out = Path(a.out_dir).resolve()
    launch = out.with_name(out.name + '.launcher')
    try:
        launch.mkdir(parents=True)
    except FileExistsError:
        raise ValueError('launcher already claimed; resume unsupported')
    child = None
    requested = []
    handlers = {}
    stage = 'initializing'
    code = None
    state = {}

    def write(reason=None):
        state.update(
            pid=os.getpid(),
            child_pid=child.pid if child and child.poll() is None else None,
            stage=stage,
            exit_code=code,
            reason=reason,
            time=time.time(),
            out_dir=str(out),
        )
        run.json_write(launch / 'supervisor.json', state)

    def handle(sig, *_):
        requested.append(sig)

    def expired():
        return a.deadline_unix is not None and time.time() >= a.deadline_unix

    def stop():
        if child is None or child.poll() is not None:
            return
        child.send_signal(signal.SIGTERM)
        end = time.monotonic() + 10
        while child.poll() is None and time.monotonic() < end:
            time.sleep(min(poll, 0.2))
        if child.poll() is None:
            child.kill()
        child.wait()

    try:
        write()
        if out.exists() and any(out.iterdir()):
            raise ValueError('occupied output; resume unsupported')
        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, handle)
        for stage in ('train', 'final'):
            if requested or expired():
                break
            if stage == 'final':
                if a.nonformal:
                    closed = json.loads((out / 'training-closed.json').read_text())
                    if closed['status'] != 'nonformal_complete':
                        raise ValueError('nonformal training did not complete')
                    stage, code = 'nonformal_complete', 0
                    write()
                    return state
                run.preflight_final(a)
            args = [sys.executable, '-m', 'qrokkun_ai.v25.run', stage]
            for key, value in vars(a).items():
                if key == 'command' or value is None:
                    continue
                flag = '--' + key.replace('_', '-')
                if isinstance(value, bool):
                    if value:
                        args.append(flag)
                else:
                    args.extend([flag, str(value)])
            if child_argv:
                args = child_argv(stage)
            with (launch / (stage + '.stdout')).open('ab', buffering=0) as log:
                if requested or expired():
                    stage = 'stopped'
                    write('owner signal' if requested else 'explicit deadline')
                    return state
                child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=False)
                write()
                while child.poll() is None and not requested and not expired():
                    time.sleep(min(poll, 0.2))
                if requested or expired():
                    stop()
                    code = child.returncode
                    stage = 'stopped'
                    write('owner signal' if requested else 'explicit deadline')
                    return state
                code = child.wait()
                os.fsync(log.fileno())
            write()
            if code != 0:
                stage = 'failed'
                write('child failed')
                return state
        if requested or expired():
            stage = 'stopped'
            write('owner signal or explicit deadline')
        else:
            stage = 'complete'
            write()
        return state
    except BaseException as e:
        stop()
        stage = 'failed'
        write(str(e))
        raise
    finally:
        stop()
        for sig, h in handlers.items():
            signal.signal(sig, h)


def main():
    p = run.parser()
    p.set_defaults(device='cuda')
    state = execute(p.parse_args(['train', *sys.argv[1:]]))
    if state['stage'] not in ('complete', 'nonformal_complete'):
        raise SystemExit(state.get('exit_code') or 1)


if __name__ == '__main__':
    main()
