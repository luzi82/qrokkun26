"""Synthetic only: never opens an actual run or heldout window."""
import pytest
import unittest
from unittest.mock import patch
from qrokkun_ai.v22 import run

class ExtensionTests(unittest.TestCase):
    def test_deadlines(self):
        self.assertEqual(run.HARD,1794322800)
        a=run.parser().parse_args(['train','--out-dir','synthetic'])
        self.assertEqual(a.train_deadline_unix,1794320100)
        with patch.object(run.time,'time',return_value=1794308401): run.Guard(run.HARD).check()
        with patch.object(run.time,'time',return_value=1794322800):
            with self.assertRaises(run.Stopped): run.Guard(run.HARD).check()
    def test_resume_parameter(self):
        a=run.parser().parse_args(['train','--out-dir','synthetic','--resume-from-run','frozen'])
        self.assertEqual(a.resume_from_run,'frozen')

import copy
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from qrokkun_ai.v22.core import Spawner, PLAYER_SHA, STATE_SHA, file_hash
from qrokkun_ai.v22.resume import FILES, read_snapshot, restore

@pytest.mark.usefixtures("synthetic_deadlines")
class SyntheticResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.a=run.parser().parse_args(['train','--out-dir','synthetic','--resume-from-run',str(self.root)])
        self.old=vars(self.a).copy(); self.old.pop('resume_from_run')
        self.old.update(deadline_unix=1794308400,train_deadline_unix=1794305700)
        self.manifest=dict(config=self.old,cap_frames=7200,player_state=STATE_SHA,protocol=1,formal=True)
        torch.manual_seed(17); np.random.seed(17)
        self.net=Spawner(); self.initial=copy.deepcopy(self.net)
        self.opt=torch.optim.Adam(self.net.parameters(),lr=self.a.lr)
        self.rng=torch.Generator().manual_seed(17)
        self.cp(0,'initial.pt'); self.step(); self.cp(1,'latest.pt'); self.cp(1,'best.pt')
        (self.root/'manifest.json').write_text(json.dumps(self.manifest))
        controls={m:{'per_seed':[{'seed':s} for s in range(400000,400016)]} for m in ('scripted','stochastic')}
        (self.root/'validation-controls.json').write_text(json.dumps(controls))
        (self.root/'progress.jsonl').write_text(json.dumps({'update':1})+'\n')
        self.rehash()
    def tearDown(self): self.tmp.cleanup()
    def step(self):
        x=torch.randn(8,325,generator=self.rng)
        # All three restored RNG streams affect the synthetic Adam update.
        loss=self.net.value(x).square().mean()+torch.rand(())*self.net.value(x).mean()+float(np.random.rand())*self.net.value(x).mean()
        loss=loss+self.net.distribution(x).mean.square().mean()+self.net.distribution(x).stddev.square().mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
    def cp(self,n,name):
        torch.save(dict(model=self.net.state_dict(),optimizer=self.opt.state_dict(),update=n,attempted_update=n,partial_update=False,train_rng=self.rng.get_state(),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),manifest=self.manifest,player_sha=PLAYER_SHA,player_state=STATE_SHA,metrics={'validation':{'mean_seconds':12.,'per_seed':[{'seed':i} for i in range(400000,400016)]}}),self.root/name)
    def rehash(self):
        (self.root/'provenance.json').write_text(json.dumps(dict(run_path='/synthetic/original',source_commit='c4e322b',heldout_access=False,frozen=True,core_sha256=file_hash(Path(run.__file__).with_name('core.py')),hashes={n:file_hash(self.root/n) for n in FILES})))
    def snapshot(self): return read_snapshot(self.root,vars(self.a),7200)
    def mutate(self,**changes):
        p=self.root/'latest.pt'; cp=torch.load(p,weights_only=False); cp.update(changes); torch.save(cp,p); self.rehash()
    def test_exact_parameters_optimizer_and_rng(self):
        snapshot=self.snapshot()
        self.step(); expected=copy.deepcopy(self.net.state_dict()); optimizer=copy.deepcopy(self.opt.state_dict())
        expected_rng=(torch.get_rng_state().clone(),self.rng.get_state().clone(),np.random.get_state())
        self.net=Spawner(); self.initial=Spawner(); self.opt=torch.optim.Adam(self.net.parameters(),lr=self.a.lr); self.rng=torch.Generator()
        restore(snapshot,self.net,self.initial,self.opt,self.rng); self.step()
        for k,v in expected.items(): self.assertTrue(torch.equal(v,self.net.state_dict()[k]),k)
        for k,state in optimizer['state'].items():
            for name,v in state.items(): self.assertTrue(torch.equal(v,self.opt.state_dict()['state'][k][name]))
        self.assertTrue(torch.equal(expected_rng[0],torch.get_rng_state()))
        self.assertTrue(torch.equal(expected_rng[1],self.rng.get_state()))
        self.assertTrue(np.array_equal(expected_rng[2][1],np.random.get_state()[1]))
    def test_partial_rejected(self):
        self.mutate(partial_update=True)
        with self.assertRaises(ValueError): self.snapshot()
    def test_attempt_rejected(self):
        self.mutate(attempted_update=2)
        with self.assertRaises(ValueError): self.snapshot()
    def test_config_rejected(self):
        self.a.entropy=.2
        with self.assertRaises(ValueError): self.snapshot()
    def test_wrong_player_rejected(self):
        self.mutate(player_sha='wrong')
        with self.assertRaises(ValueError): self.snapshot()
    def test_claim_rejected(self):
        (self.root/'final-claimed.json').write_text('{}')
        with self.assertRaises(ValueError): self.snapshot()
    def test_actual_heldout_rejected(self):
        (self.root/'progress.jsonl').write_text('{"seed":500000}\n'); self.rehash()
        with self.assertRaises(ValueError): self.snapshot()
    def test_optimizer_rejected(self):
        snapshot=self.snapshot(); snapshot['checkpoints']['latest.pt']['optimizer']['param_groups'][0]['betas']=(.1,.2)
        with self.assertRaises(ValueError): restore(snapshot,self.net,self.initial,self.opt,self.rng)
    def test_architecture_rejected(self):
        snapshot=self.snapshot(); snapshot['checkpoints']['latest.pt']['model'].pop(next(iter(self.net.state_dict())))
        with self.assertRaises(RuntimeError): restore(snapshot,self.net,self.initial,self.opt,self.rng)

    def test_continuation_main_keeps_best_and_seed_index(self):
        self.a.updates=2
        # The same total safety maximum must already be in the parent manifest.
        self.old['updates']=2
        self.manifest['config']=self.old
        for name in ('initial.pt','latest.pt','best.pt'):
            cp=torch.load(self.root/name,weights_only=False); cp['manifest']=self.manifest; torch.save(cp,self.root/name)
        (self.root/'manifest.json').write_text(json.dumps(self.manifest)); self.rehash()
        out=self.root.with_name(self.root.name+'-output'); self.addCleanup(__import__('shutil').rmtree,out,True); seen=[]
        def episodes(*args,**kwargs):
            seen.extend(args[2]); return ([dict(seed=s,seconds=1.,death=False,censored=True) for s in args[2]],['synthetic'])
        argv=['synthetic','train','--out-dir',str(out),'--resume-from-run',str(self.root),'--updates','2']
        with patch('sys.argv',argv), patch.object(run,'load_player',return_value=(torch.nn.Linear(1,1).requires_grad_(False),None)), patch.object(run,'episodes',side_effect=episodes), patch.object(run,'ppo',return_value={'synthetic_loss':0.}), patch.object(run.signal,'signal'):
            run.main()
        self.assertEqual(seen,list(range(1000016,1000032)))
        for name in ('initial.pt','best.pt','validation-controls.json'):
            self.assertEqual((out/name).read_bytes(),(self.root/name).read_bytes())
        rows=[json.loads(x) for x in (out/'progress.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),1); self.assertEqual(rows[0]['update'],2)
        self.assertEqual(rows[0]['inherited_updates'],1); self.assertEqual(rows[0]['locally_completed_updates'],1)
    def test_missing_adam_state_rejected(self):
        snapshot=self.snapshot(); snapshot['checkpoints']['latest.pt']['optimizer']['state']={}
        with self.assertRaises(ValueError): restore(snapshot,self.net,self.initial,self.opt,self.rng)
    def test_wrong_mapping_source_rejected(self):
        p=self.root/'provenance.json'; obj=json.loads(p.read_text()); obj['core_sha256']='wrong'; p.write_text(json.dumps(obj))
        with self.assertRaises(ValueError): self.snapshot()

    def test_pending_validation_rejected(self):
        for n in (1,10):
            self.mutate(update=n,attempted_update=n,metrics={'rollout':{}})
            with self.assertRaisesRegex(ValueError,'validation'): self.snapshot()
    def test_incomplete_validation_rejected(self):
        self.mutate(metrics={'validation':{'mean_seconds':12.,'per_seed':[{'seed':400000}]}})
        with self.assertRaisesRegex(ValueError,'validation'): self.snapshot()
    def test_nonboundary_without_validation_accepted(self):
        self.mutate(update=2,attempted_update=2,metrics={'rollout':{}})
        self.assertEqual(self.snapshot()['checkpoints']['latest.pt']['update'],2)
    def test_provenance_path_rejected(self):
        for value in ('relative/run','/synthetic/../original'):
            p=self.root/'provenance.json'; obj=json.loads(p.read_text()); obj['run_path']=value; p.write_text(json.dumps(obj))
            with self.assertRaisesRegex(ValueError,'run_path'): self.snapshot()
    def test_preflight_readonly_and_rng(self):
        from qrokkun_ai.v22.resume import preflight
        before={p.name:p.read_bytes() for p in self.root.iterdir()}
        tr=torch.get_rng_state().clone(); nr=np.random.get_state()
        result=preflight(self.root,vars(self.a))
        self.assertEqual(result['update'],1); self.assertEqual(result['best'],12.)
        self.assertTrue(torch.equal(tr,torch.get_rng_state()))
        self.assertTrue(np.array_equal(nr[1],np.random.get_state()[1]))
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.root.iterdir()})
        self.mutate(torch_rng=torch.tensor([1],dtype=torch.uint8))
        with self.assertRaises(Exception): preflight(self.root,vars(self.a))
        self.assertTrue(torch.equal(tr,torch.get_rng_state()))
    def test_protected_outputs_before_creates(self):
        from qrokkun_ai.v22 import supervise
        from qrokkun_ai.v22.resume import validate_output
        for root in (self.root,Path('/synthetic/original')):
            for out in (root/'child',):
                # Explicit launcher-root case uses provenance pointing at sibling launcher.
                if out==root: continue
                with self.assertRaises(ValueError): validate_output(self.root,out)
        for module in (run,supervise):
            self.a.out_dir=str(self.root/'forbidden')
            with patch.object(Path,'mkdir',side_effect=AssertionError('created before rejection')):
                with self.assertRaises(ValueError):
                    if module is run:
                        with patch('sys.argv',['test','train','--out-dir',self.a.out_dir,'--resume-from-run',str(self.root)]): run.main()
                    else: supervise.execute(self.a)

    def test_preflight_checks_all_restorations(self):
        from qrokkun_ai.v22.resume import preflight
        for name in ('initial.pt','best.pt'):
            saved=(self.root/name).read_bytes()
            cp=torch.load(self.root/name,weights_only=False); cp['model']={}
            torch.save(cp,self.root/name); self.rehash()
            with self.assertRaises(RuntimeError): preflight(self.root,vars(self.a))
            (self.root/name).write_bytes(saved); self.rehash()
        cp=torch.load(self.root/'latest.pt',weights_only=False); cp['optimizer']['state']={}
        torch.save(cp,self.root/'latest.pt'); self.rehash()
        with self.assertRaises(ValueError): preflight(self.root,vars(self.a))
    def test_launcher_root_and_symlink_rejected(self):
        from qrokkun_ai.v22.resume import validate_output
        p=self.root/'provenance.json'; obj=json.loads(p.read_text())
        obj['run_path']=str(self.root/'destination.launcher'); p.write_text(json.dumps(obj))
        with self.assertRaises(ValueError): validate_output(self.root,self.root/'destination')
        sibling=self.root.with_name(self.root.name+'-link')
        sibling.symlink_to(self.root,target_is_directory=True)
        self.addCleanup(sibling.unlink)
        with self.assertRaises(ValueError): validate_output(self.root,sibling/'child')

    def test_original_launcher_collision_entrypoints(self):
        from qrokkun_ai.v22 import supervise
        out=self.root.with_name(self.root.name+'-new')
        p=self.root/'provenance.json'; obj=json.loads(p.read_text())
        obj['run_path']=str(out.with_name(out.name+'.launcher')); p.write_text(json.dumps(obj))
        self.a.out_dir=str(out)
        with patch.object(Path,'mkdir',side_effect=AssertionError('created before rejection')):
            with self.assertRaises(ValueError): supervise.execute(self.a)
            with patch('sys.argv',['test','train','--out-dir',str(out),'--resume-from-run',str(self.root)]):
                with self.assertRaises(ValueError): run.main()

class TrainCapTests(unittest.TestCase):
    def test_cap_both_entrypoints(self):
        from qrokkun_ai.v22 import supervise
        argv=['train','--out-dir','synthetic','--train-deadline-unix','1794320101']
        with patch('sys.argv',['test']+argv), patch.object(Path,'mkdir',side_effect=AssertionError('must reject before creates')):
            with self.assertRaises(ValueError): run.main()
        with self.assertRaises(ValueError): supervise.execute(run.parser().parse_args(argv))
