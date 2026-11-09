import unittest
import numpy as np
from skillforge.key_learning import DEFAULT,validate_recipe,parse_recipe,nested_indices,fit_key,reward,best_trial
from skillforge.heat_curve_data import build_splits


class KeyLearningTests(unittest.TestCase):
    def test_recipe_bounds_duplicates_and_unknown_fields(self):
        import json
        self.assertEqual(parse_recipe(json.dumps(DEFAULT)),DEFAULT)
        for bad in [dict(DEFAULT,ridge=-1),dict(DEFAULT,anchor='anything'),dict(DEFAULT,negative_weight=float('nan')),dict(DEFAULT,positive_target=True),dict(DEFAULT,lr=.3)]:
            with self.assertRaises(ValueError):validate_recipe(bad)
        with self.assertRaises(ValueError):parse_recipe('{"ridge":0,"ridge":1}')

    def test_data_disjointness_and_nested_sizes(self):
        splits=build_splits();self.assertEqual({k:len(v) for k,v in splits.items()},dict(train=1024,development=128,final=256))
        for a,b in [('train','development'),('train','final'),('development','final')]:
            for field in ['id','question','setting','template_id']:
                self.assertFalse({r[field] for r in splits[a]} & {r[field] for r in splits[b]})
        labels=[r['category'] for r in splits['train']];plans=nested_indices(labels,17,[16,64,256,1024])
        for small,large in [(16,64),(64,256),(256,1024)]:self.assertTrue(set(plans[small])<=set(plans[large]))
        for n,indices in plans.items():
            self.assertEqual(len(set(indices)),n)
            for c in set(labels):self.assertEqual(sum(labels[i]==c for i in indices),n//4)

    def test_synthetic_fit_reduces_leakage_deterministically(self):
        import torch
        torch.set_num_threads(2)
        y=np.array([True]*8+[False]*24);q=np.zeros((32,3,64));q[:,:,0]=np.where(y[:,None],8.,-8.)
        initial=np.zeros(64);z=np.zeros((32,3));correct=np.ones(32,dtype=bool)
        k,trace=fit_key(q,z,y,correct,initial,10.,DEFAULT,steps=200)
        kk,_=fit_key(q,z,y,correct,initial,10.,DEFAULT,steps=200)
        self.assertTrue(np.array_equal(k,kk));self.assertLess(np.linalg.norm(k),10)
        mass=1/(1+np.exp(-(q@k-z)))
        self.assertGreater(mass[y].mean(),.7);self.assertLess(mass[~y].mean(),.3)
        self.assertEqual(initial.sum(),0);self.assertEqual(trace[-1]['step'],200)
        def stop():raise RuntimeError('stop')
        with self.assertRaises(RuntimeError):fit_key(q,z,y,correct,initial,10.,DEFAULT,steps=200,check=stop)

    def test_reward_and_selection_do_not_use_final_data(self):
        b=dict(metrics=dict(positive_correct=2));m=dict(positive_correct=6,positive_n=8,initially_correct_negative_retained=8,initially_correct_negative_n=10,negative_mean_mass=.2)
        self.assertAlmostEqual(reward(dict(metrics=m),b)['value'],0.)
        trials=[dict(step=1,status='accepted',recipe=dict(DEFAULT),reward=dict(value=1)),dict(step=0,status='accepted',recipe=dict(DEFAULT,ridge=.02),reward=dict(value=1)),dict(step=2,status='rejected')]
        self.assertEqual(best_trial(trials)['step'],0);self.assertEqual(best_trial([])['recipe'],DEFAULT)

    def test_runner_freezes_every_key_before_final_access(self):
        import json,hashlib,sys,tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
        import run_key_learning as runner
        from skillforge.selective_key import metrics
        from test_attention_probe import bank
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'run';out.mkdir();(out/'approved_manifest.json').write_text('fixture')
            (out/'source_receipt.json').write_text(json.dumps(dict(status='verified',manifest_sha256=hashlib.sha256(b'fixture').hexdigest())))
            parent=bank();parent['alpha'][2,0]=.75;np.savez_compressed(root/'parent.npz',**parent)
            splits=build_splits()
            for name,rows in splits.items():(root/(name+'.json')).write_text(json.dumps(rows))
            (root/'guide.md').write_text('fixture');(root/'local.json').write_text('{"codex_cli":"unused"}')
            config=dict(resources={},model_sha256='fixture',parent_bank=str(root/'parent.npz'),guide=str(root/'guide.md'),local_paths=str(root/'local.json'),
                        train=str(root/'train.json'),development=str(root/'development.json'),final=str(root/'final.json'),catalog='unused',instructions='unused',wall_guard_seconds=999,optimizer_steps=1)
            def record(row):
                g=row['choices']['label'].index(row['answerKey']);return dict(id=row['id'],gold=g,prediction=0,correct=g==0,virtual_mass=[])
            def features(study,rows):
                if rows[0]['split']=='final':self.assertTrue((out/'frozen_curve.json').exists())
                return np.zeros((len(rows),3,64)),np.zeros((len(rows),3)),[record(r) for r in rows]
            def evaluate(study,rows,b,mode,baseline,check):
                rec=[record(r) for r in rows];return dict(records=rec,metrics=metrics(rows,rec,baseline))
            calls=[];fits=[]
            def collect(cli,catalog,instructions,folder,text,seconds,heartbeat):
                folder.mkdir();calls.append(text);heartbeat()
                return ('invalid' if len(calls)==1 else json.dumps(DEFAULT)),dict(thread_id=str(len(calls)))
            def fit(*args):fits.append(len(args[0]));return args[4].copy(),[]
            real=Path.read_text
            def read(path,*a,**kw):
                if path==root/'final.json':
                    self.assertEqual(len(json.loads(real(out/'frozen_curve.json'))),20)
                    self.assertTrue((out/'selected_recipes.json').exists())
                return real(path,*a,**kw)
            with patch.object(runner,'features',features),patch.object(runner,'evaluate',evaluate),patch.object(runner,'fit_key',fit),patch.object(runner,'collect',collect),patch.object(runner,'model_hash',lambda m:'fixture'),patch.object(Path,'read_text',read):
                runner.run(lambda:SimpleNamespace(model=None),out,config)
            result=json.loads((out/'results.json').read_text());self.assertEqual(result['status'],'complete')
            self.assertEqual(len(calls),4);self.assertEqual(len(fits),27);self.assertEqual(len(result['curve']),20)
            self.assertEqual(result['selected_recipes']['astra']['step'],1)
            self.assertNotIn('mountain shelter',' '.join(calls))
