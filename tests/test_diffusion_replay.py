import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import json
import hashlib
import numpy as np
from skillforge.diffusion_replay import Denoiser, PosteriorSampler, time_embedding, SHAPES, contact_sheet_pgm


class DiffusionTests(unittest.TestCase):
    def test_embedding_order_and_units(self):
        np.testing.assert_array_equal(time_embedding(0), [0]*8+[1]*8)
        for t in (1, 713, 999):
            f = np.array([1 / 10000**(k/8) for k in range(8)])
            np.testing.assert_allclose(time_embedding(t), np.r_[np.sin(t*f),np.cos(t*f)], atol=3e-8)
        for t in (-1, 1000, .5):
            with self.assertRaises(ValueError): time_embedding(t)

    def test_layout_and_relu(self):
        a = {k: np.zeros(s,np.float32) for k,s in SHAPES.items()}
        a['W1'][0,0]=1; a['W1'][152,1]=1
        a['W2'][0,0]=2; a['W2'][1,1]=3
        a['W3'][0,4]=4; a['W3'][1,5]=5
        model=Denoiser(a); x=np.zeros((2,144),np.float32); x[:,0]=[-2,2]
        y=model.predict(x,0)
        np.testing.assert_array_equal(y[:,4],[0,16])
        np.testing.assert_array_equal(y[:,5],[15,15])
        self.assertEqual(np.count_nonzero(y),3)
        a['W1'][0,0]=99
        np.testing.assert_array_equal(model.predict(x,0),y)

    def test_posterior_against_scalar_formula(self):
        s=PosteriorSampler(); x=np.full((1,144),.3,np.float32); pred=np.full_like(x,2); z=np.full_like(x,.7)
        for t in (1, 17, 999):
            b=s.beta[t]; a=1-b; abar=s.abar[t]; prev=s.abar[t-1]
            expected=b*np.sqrt(prev)/(1-abar)+np.sqrt(a)*(1-prev)/(1-abar)*.3+np.sqrt(b*(1-prev)/(1-abar))*.7
            np.testing.assert_allclose(s.step(x,pred,t,z),expected,rtol=2e-6)
        self.assertTrue(np.all(s.variance>=0))
        self.assertTrue(np.all(s.variance<=s.beta))

    def test_final_step_is_clipped_x0(self):
        s=PosteriorSampler(); x=np.full((2,144),99,np.float32); p=np.full_like(x,-2)
        np.testing.assert_array_equal(s.step(x,p,0),-np.ones_like(x))
        with self.assertRaises(ValueError): s.step(x,p,0,np.zeros_like(x))
        with self.assertRaises(ValueError): s.step(x,p,1)
        with self.assertRaises(ValueError): s.step(x,p,-1)

    def test_sampler_stream_and_timing_with_synthetic_oracle(self):
        class Oracle:
            def __init__(self): self.times=[]
            def predict(self,x,t): self.times.append(t); return np.full_like(x,.25)
        a=Oracle(); s=PosteriorSampler(); x,tr=s.sample(a,2,7)
        np.testing.assert_array_equal(x,np.full((2,144),.25,np.float32))
        self.assertEqual(a.times,list(range(999,-1,-1)))
        _,again=s.sample(Oracle(),2,7)
        for k in tr: np.testing.assert_array_equal(tr[k],again[k])
        rng=np.random.Generator(np.random.PCG64(7))
        np.testing.assert_array_equal(tr['x_t_999'],rng.standard_normal((2,144)).astype(np.float32)[:1])
        with self.assertRaises(RuntimeError):
            s.sample(Oracle(),1,7,lambda t: (_ for _ in ()).throw(RuntimeError('stop')))

    def test_bad_artifact_rejected(self):
        a={k:np.zeros(s,np.float32) for k,s in SHAPES.items()}
        for change in ('dtype','shape','finite','extra'):
            b=a.copy()
            if change=='dtype': b['W1']=b['W1'].astype(np.float64)
            if change=='shape': b['W1']=b['W1'].T
            if change=='finite': b['W1']=np.full(SHAPES['W1'],np.nan,np.float32)
            if change=='extra': b['extra']=np.zeros(1,np.float32)
            with self.assertRaises(ValueError): Denoiser(b)

    def test_preview_pixel_orientation(self):
        x=np.full((64,144),-1,np.float32); x[1,0]=1; x[8,12]=1
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'sheet.pgm'; contact_sheet_pgm(x,path)
            data=path.read_bytes(); self.assertTrue(data.startswith(b'P5\n96 96\n255\n'))
            p=np.frombuffer(data.split(b'\n',3)[3],np.uint8).reshape(96,96)
            self.assertEqual(p[0,12],255); self.assertEqual(p[13,0],255)
            self.assertEqual(np.count_nonzero(p),2)

    def test_runner_archival_and_controlled_stop_with_fake_sampler(self):
        from run_diffusion_replay import run, ControlledStop
        a={k:np.zeros(s,np.float32) for k,s in SHAPES.items()}
        class FakeSampler:
            def __init__(self, stop=False): self.calls=0; self.stop=stop
            def sample(self,model,count,seed,callback):
                if self.stop and self.calls==1: raise ControlledStop()
                self.calls+=1
                return np.zeros((count,144),np.float32), {'final':np.zeros((1,144),np.float32)}
        class Study:
            def __init__(self,stop): self.models={'synthetic':Denoiser(a)}; self.sampler=FakeSampler(stop)
        for stop in (False,True):
            with tempfile.TemporaryDirectory() as d:
                out=Path(d); manifest=b'unit fixture'
                (out/'approved_manifest.json').write_bytes(manifest)
                (out/'source_receipt.json').write_text(json.dumps(dict(status='verified',manifest_sha256=hashlib.sha256(manifest).hexdigest())))
                config=dict(samples_per_model=1024,batch_size=64,resources={},wall_guard_seconds=60,seed_base=900)
                with patch.dict('os.environ', {k:'1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}):
                    run(lambda:Study(stop),out,config)
                r=json.loads((out/'results.json').read_text())
                self.assertEqual(r['status'],'paused' if stop else 'complete')
                self.assertEqual(r['summaries']['synthetic']['samples'],64 if stop else 1024)
                rows=json.loads((out/'batches.json').read_text())
                for row in rows:
                    p=out/f"synthetic_batch_{row['batch']:02d}.npz"
                    self.assertEqual(row['sha256'],hashlib.sha256(p.read_bytes()).hexdigest())
                self.assertEqual((out/'status.json').read_bytes(),(out/'results.json').read_bytes())
