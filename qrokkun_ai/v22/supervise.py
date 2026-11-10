"""Direct-child Sv22 supervisor; launch from a managed remote parent session."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from . import run


def execute(a, child_argv=None, poll=.2):
    """child_argv is a unit-test seam, never exposed by the CLI."""
    if not all(math.isfinite(v) for v in (a.deadline_unix,a.train_deadline_unix)):
        raise ValueError('finite deadlines required')
    if not a.train_deadline_unix < a.deadline_unix <= run.HARD or a.train_deadline_unix>run.TRAIN_HARD:
        raise ValueError('training deadline must precede hard deadline, no later than owner cutoff')
    cutoff=a.deadline_unix-10
    if time.time()>=cutoff: raise ValueError('launch cutoff passed')
    out=Path(a.out_dir).resolve(); launch=out.with_name(out.name+'.launcher')
    resume=getattr(a,'resume_from_run',None)
    if resume:
        from .resume import validate_output
        validate_output(resume,out)
    # Exclusive launcher directory is the one-time launch claim, even after failure.
    try: launch.mkdir(parents=True)
    except FileExistsError:
        state=launch/'supervisor.json'
        if state.exists(): return json.loads(state.read_text())
        raise ValueError('launcher already claimed; no resume')
    child=None; requested=[]; stage='initializing'; code=None
    state={}; old={}
    stop_mono=time.monotonic()+max(0,cutoff-time.time())
    def write(reason=None):
        state.update(pid=os.getpid(),child_pid=child.pid if child and child.poll() is None else None,
                     stage=stage,exit_code=code,time=time.time(),reason=reason,
                     out_dir=str(out),train_stdout=str(launch/'train.stdout'),final_stdout=str(launch/'final.stdout'))
        run.json_write(launch/'supervisor.json',state)
    def handle(sig,*_): requested.append(sig)
    def expired(): return time.time()>=cutoff or time.monotonic()>=stop_mono
    def stop():
        if child is None or child.poll() is not None: return
        child.send_signal(requested[0] if requested else signal.SIGTERM)
        end=time.monotonic()+5
        while child.poll() is None and time.monotonic()<end: time.sleep(min(poll,1))
        if child.poll() is None: child.kill()
        child.wait()
    try:
        write()
        if out.exists() and any(out.iterdir()): raise ValueError('occupied output directory; no resume')
        for sig in (signal.SIGTERM,signal.SIGINT):
            old[sig]=signal.signal(sig,handle)
        for stage in ('train','final'):
            if requested or expired(): break
            if stage=='final':
                closed=json.loads((out/'training-closed.json').read_text())
                if run.file_hash(out/'best.pt')!=closed['best_sha']: raise ValueError('best hash mismatch')
                if (out/'final-claimed.json').exists(): raise ValueError('final already claimed')
            args=[sys.executable,'-m','qrokkun_ai.v22.run',stage]
            for key,value in vars(a).items():
                if key=='command' or value is None or (key=='resume_from_run' and stage!='train'): continue
                flag='--'+key.replace('_','-')
                if isinstance(value,bool):
                    if value: args.append(flag)
                else: args.extend([flag,str(value)])
            if child_argv is not None: args=child_argv(stage)
            with (launch/(stage+'.stdout')).open('ab',buffering=0) as log:
                child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=False)
                write()
                while child.poll() is None and not requested and not expired(): time.sleep(min(poll,1))
                if requested or expired():
                    stop(); code=child.returncode; stage='stopped'; write('signal-'+str(requested[0]) if requested else 'hard cutoff'); return state
                code=child.wait(); os.fsync(log.fileno())
            write()
            if code!=0: stage='failed'; write('child failed'); return state
        if requested or expired(): stage='stopped'; write('signal or hard cutoff')
        else: stage='complete'; write()
        return state
    except BaseException as e:
        stop(); stage='failed'; write(str(e)); raise
    finally:
        stop()
        for sig,handler in old.items(): signal.signal(sig,handler)


def main():
    p=run.parser()
    # Supervisor owns train -> final; callers cannot supply an alternate executable.
    p.set_defaults(device='cuda')
    a=p.parse_args(['train',*sys.argv[1:]])
    state=execute(a)
    if state['stage']!='complete': raise SystemExit(1)


if __name__=='__main__': main()
