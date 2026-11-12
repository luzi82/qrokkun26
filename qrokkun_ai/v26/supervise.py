"""Exclusive c -> t -> compare supervisor; direct child reaped on owner stop."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from .run import parser, validate, estimate
from .trace import json_write


def supervise(root, commands, after_child=lambda label,stop:None):
    root=Path(root)
    stopped=[False]; child=[None]; handlers={}; codes={}
    def stop(*args):
        stopped[0]=True
        if child[0] is not None and child[0].poll() is None:
            child[0].terminate()
    result=dict(status='running',exitcodes=codes,counters={})
    try:
        for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,stop)
        for label in ('Pv26c','Pv26t','compare'):
            if stopped[0]: break
            json_write(root/'pair-status.json',dict(result,phase=label))
            with (root/(label+'.log')).open('x') as log:
                if stopped[0]:break
                # Atomic launch commit: block stop delivery, check queued requests,
                # create and publish the handle, then restore delivery. The child
                # restores the original mask in a stdlib-only exec trampoline.
                mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM})
                try:
                    if stopped[0] or signal.sigpending() & {signal.SIGINT,signal.SIGTERM}:
                        stopped[0]=True
                        break
                    command=[sys.executable,'-m','qrokkun_ai.v26.child_exec',','.join(str(int(x)) for x in mask-{signal.SIGINT,signal.SIGTERM}),*commands[label]]
                    child[0]=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK,mask)
                while child[0].poll() is None:
                    try:child[0].wait(timeout=.2)
                    except subprocess.TimeoutExpired:
                        if stopped[0]:
                            try:child[0].wait(timeout=5)
                            except subprocess.TimeoutExpired:child[0].kill();child[0].wait()
                codes[label]=child[0].returncode
                if codes[label] in (130,-signal.SIGINT,-signal.SIGTERM):stopped[0]=True
                child[0]=None
            after_child(label,stop)
            closure=root/label/'closed.json'
            if closure.exists():
                c=json.loads(closure.read_text())
                result['counters'][label]={k:c[k] for k in ('completed_updates','trace_bytes','wall_s')}
            if stopped[0] or codes[label] not in ((0,2) if label=='compare' else (0,)):break
        if stopped[0]:result['status']='stopped'
        elif codes.get('compare')==0:result['status']='complete_equal'
        elif codes.get('compare')==2:result['status']='complete_different'
        else:result['status']='runtime_or_incomplete'
        json_write(root/'paired-closure.json',result)
        return result
    finally:
        if child[0] is not None and child[0].poll() is None:
            child[0].terminate()
            try:child[0].wait(timeout=5)
            except subprocess.TimeoutExpired:child[0].kill();child[0].wait()
        for sig,h in handlers.items():signal.signal(sig,h)


def main(argv=None):
    p=parser()
    p.description=__doc__
    p._option_string_actions['--arm'].required=False
    a=p.parse_args(argv)
    if a.arm is not None:raise ValueError('pair owns arm labels')
    a.arm='Pv26t'; workers=validate(a)
    if a.estimate:
        storage=estimate(a.hidden,a.episodes,a.cap,a.updates,a.validation)
        print(json.dumps(dict(per_arm=storage,pair_bytes=2*storage['estimated_bytes']),indent=2));return 0
    out,trace=Path(a.out_dir).resolve(),Path(a.trace_dir).resolve()
    if out.exists() or trace.exists():raise FileExistsError('exclusive pair requires fresh paths')
    trace.parent.mkdir(parents=True,exist_ok=True)
    storage=estimate(a.hidden,a.episodes,a.cap,a.updates,a.validation)
    if shutil.disk_usage(trace.parent).free < 2*storage['estimated_bytes']:
        raise OSError('insufficient space for BOTH complete arms')
    out.mkdir(parents=True,exist_ok=False);trace.mkdir(parents=True,exist_ok=False)
    with (out/'pair-claim.json').open('x') as f:
        json.dump(dict(pid=os.getpid(),source=a.source_commit,arms=['Pv26c','Pv26t']),f);f.flush();os.fsync(f.fileno())
    commands={}
    for label,w in [('Pv26c',1),('Pv26t',workers)]:
        args=['--arm',label,'--workers',str(w),'--out-dir',str(out/label),'--trace-dir',str(trace/label)]
        for k in ('source_commit','player_checkpoint','spawner_checkpoint','device','updates','episodes','cap','seed','validation','hidden'):
            args+=['--'+k.replace('_','-'),str(getattr(a,k))]
        for k in ('nonformal','synthetic','deterministic'):
            if getattr(a,k):args+=['--'+k]
        commands[label]=[sys.executable,'-m','qrokkun_ai.v26.run',*args]
    commands['compare']=[sys.executable,'-m','qrokkun_ai.v26.compare',str(out/'Pv26c'),str(out/'Pv26t'),'--out-report',str(out/'comparison.json'),'--expected-commit',a.source_commit]
    result=supervise(out,commands)
    print(json.dumps(result,indent=2))
    return {'complete_equal':0,'complete_different':2,'stopped':130}.get(result['status'],3)


def cli(argv=None):
    from .cli import boundary
    return boundary(main,argv)


if __name__=='__main__':raise SystemExit(cli())
