"""Episode-local samplers and environment instrumentation; no process-global patches."""
import random
import threading
import time
import types
import numpy as np
import torch
from qrokkun_ai.v24 import core as s24
from qrokkun_env import constants as C


def global_rng():
    return dict(python=random.getstate(), numpy_legacy=np.random.get_state(),
                numpy_generators={}, torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def sample_player(probs, generators, counters=None):
    outputs, records = [], []
    for i, g in enumerate(generators):
        pre = g.get_state().clone()
        a = torch.multinomial(probs[i], 1, generator=g)
        outputs.append(a)
        records.append(dict(source='P.multinomial',counter=counters[i] if counters else 0,
                            pre=pre,post=g.get_state().clone(),output=a.clone()))
    return torch.cat(outputs), records


class PythonRNG:
    def __init__(self, rng, source, records):
        self.rng, self.source, self.records = rng, source, records
        self.counter = 0

    def getstate(self):
        return self.rng.getstate()

    def __getattr__(self, name):
        attr = getattr(self.rng, name)
        if not callable(attr) or name in ('setstate','seed'):
            return attr
        def call(*args, **kwargs):
            pre = self.rng.getstate()
            result = attr(*args, **kwargs)
            self.records.append(dict(type='rng',source=self.source+'.'+name,counter=self.counter,
                                     pre=pre,post=self.rng.getstate(),output=result,args=args,kwargs=kwargs))
            self.counter += 1
            return result
        return call


def private(fn, changes):
    ns = dict(fn.__globals__)
    ns.update(changes)
    f = types.FunctionType(fn.__code__,ns,fn.__name__,fn.__defaults__,fn.__closure__)
    f.__kwdefaults__ = fn.__kwdefaults__
    return f


class TorchDraws:
    def __init__(self, records, counter):
        self.records, self.counter = records, counter

    def __getattr__(self, name):
        original = getattr(torch,name)
        if name not in ('randn','randint'):
            return original
        def call(*args, **kwargs):
            g = kwargs['generator']
            pre = g.get_state().clone()
            out = original(*args, **kwargs)
            self.records.append(dict(type='rng',source='S.'+name,counter=self.counter[0],
                                     pre=pre,post=g.get_state().clone(),output=out.clone()))
            self.counter[0] += 1
            return out
        return call


def sample_spawner(s, obs, generator, records, counter=None):
    proxy = TorchDraws(records, counter if counter is not None else [0])
    wrapped = private(s24.wrapped_sample, {'torch':proxy})
    speed = private(s24.speed_sample, {'torch':proxy})
    sample = private(s24.Spawner.sample, {'wrapped_sample':wrapped,'speed_sample':speed})
    with torch.no_grad():
        means, std = s.parameters_at(obs)
        action, logp = sample(s,obs,generator)
    records.append(dict(type='S',observation=obs.clone(),means=means.clone(),std=std.clone(),
                        action=action.clone(),logp=logp.clone()))
    return action, logp


class Env(s24.LearnedEnv):
    def __init__(self, s, seed, validation=False, player_device="cpu"):
        self.validation = validation
        self.records = []
        self.counter = [0]
        self.ids = {}
        self.birth = 0
        self.sg = torch.Generator().manual_seed(seed+(24000000 if validation else 25000000))
        self.pg = torch.Generator(device=player_device).manual_seed(seed+26000000)
        self.initial_generators = dict(P=self.pg.get_state().clone(),S=self.sg.get_state().clone())
        def callback(obs):
            with torch.no_grad():
                action, _ = sample_spawner(s,torch.from_numpy(obs).unsqueeze(0),self.sg,self.records,self.counter)
            return action[0].numpy()
        super().__init__(callback,seed=seed)
        self.reset(seed=seed)

    def reset(self, seed=None):
        episode_seed = int(self.seed if seed is None else seed)
        self.records = []
        self.counter[0] = 0
        self.ids.clear()
        self.birth = 0
        self.pg.manual_seed(episode_seed+26000000)
        self.sg.manual_seed(episode_seed+(24000000 if self.validation else 25000000))
        self.initial_generators = dict(P=self.pg.get_state().clone(),S=self.sg.get_state().clone())
        return super().reset(seed=seed)

    def _spawn_step(self):
        if self.frame == 0 and not isinstance(self.field_rng,PythonRNG):
            for name in ('rng','s_rng','field_rng'):
                rng = getattr(self,name)
                self.records.append(dict(type='rng_init',source=name,state=rng.getstate()))
                setattr(self,name,PythonRNG(rng,name,self.records))
        return super()._spawn_step()

    def tag(self, origin):
        bullet = self.bullets[-1]
        self.ids[id(bullet)] = (self.birth, origin)
        self.birth += 1
        self.records.append(dict(type='spawn',bullet=self.bullet(bullet)))

    def _spawn_one(self):
        super()._spawn_one()
        self.tag('S')

    def _add_bullet(self, *args):
        super()._add_bullet(*args)
        self.tag('field')

    def bullet(self,b):
        ordinal, origin = self.ids[id(b)]
        return dict(id=ordinal,origin=origin,x=b.x,y=b.y,vx=b.vx,vy=b.vy,radius=b.radius,moved=b.moved)

    def _integrate_bullets(self):
        previous = list(self.bullets)
        super()._integrate_bullets()
        live = {id(b) for b in self.bullets}
        for b in previous:
            if id(b) not in live:
                self.records.append(dict(type='removal',reason='offscreen',bullet=self.bullet(b)))
                del self.ids[id(b)]

    def _check_hit(self):
        hit = super()._check_hit()
        if hit:
            ids = [self.ids[id(b)][0] for b in self.bullets
                   if (b.x-self.px)**2+(b.y-self.py)**2 <= (C.PLAYER_RADIUS+b.radius)**2]
            self.records.append(dict(type='collision',ids=ids))
        return hit

    def state(self):
        clock = self._events.gi_frame.f_locals
        return dict(player=[self.px,self.py,self.pvx,self.pvy],elapsed=self.elapsed,frame=self.frame,
                    dt=self.dt,spawn_acc=self.spawn_acc,dead=self.dead,bullets=np.asarray([(self.ids[id(b)][0],0 if self.ids[id(b)][1]=='S' else 1,b.x,b.y,b.vx,b.vy,b.radius,b.moved) for b in self.bullets], dtype=[('id','<i8'),('origin','u1'),('x','<f8'),('y','<f8'),('vx','<f8'),('vy','<f8'),('radius','<f8'),('moved','?')]),
                    next_event=self._next_event,event_clock={k:clock[k] for k in ('elapsed','acc','frame','first','interval')},
                    s_history=np.asarray(self.s_history,dtype=[('frame','<i8'),('speed','<f8')]),field_history=np.asarray(self.field_history,dtype=[('frame','<i8'),('speed','<f8')]),
                    python_rng={n:getattr(self,n).getstate() for n in ('rng','s_rng','field_rng')},
                    P=self.pg.get_state().clone(),S=self.sg.get_state().clone(),birth=self.birth)

    def drain(self):
        events = self.records
        self.records = []
        for name in ('rng','s_rng','field_rng'):
            getattr(self,name).records = self.records
        return events

    def traced_step(self, action):
        start = time.perf_counter()
        before = self.state()
        with torch.no_grad():  # grad mode is thread local
            result = self.step(int(action))
        data = dict(before=before,after=self.state(),events=self.drain(),action=int(action),
                    observation=result[0],base_reward=result[1],done=result[2],info=result[3])
        return result, data, dict(thread_id=threading.get_ident(),wall_s=time.perf_counter()-start)
