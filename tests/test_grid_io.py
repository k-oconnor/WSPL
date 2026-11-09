import json,tempfile,unittest,hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from skillforge.grid_domain import enumerate_domain,prepare_sets,encode_grid
from skillforge.grid_io import render_prompt,palette_ids,summarize


class GridTests(unittest.TestCase):
    def test_domain_preservation_and_translation(self):
        rows=enumerate_domain();self.assertEqual(len(rows),560)
        self.assertEqual(len({r['id'] for r in rows}),560)
        for row in rows:
            a,b=row['input'],row['output'];color=row['color']
            self.assertEqual(a.count(color),b.count(color));self.assertEqual(a.index(3),b.index(3))
            self.assertNotEqual(a,b);self.assertEqual(a.count(3),1);self.assertEqual(b.count(3),1)
            old=sorted(divmod(i,4) for i,x in enumerate(a) if x==color);new=sorted(divmod(i,4) for i,x in enumerate(b) if x==color)
            offsets={(rr-r,cc-c) for (r,c),(rr,cc) in zip(old,new)}
            self.assertEqual(len(offsets),1);dr,dc=offsets.pop();self.assertEqual(abs(dr)+abs(dc),1)
            mr,mc=divmod(a.index(3),4)
            self.assertLess(sum(abs(mr-r)+abs(mc-c) for r,c in new),sum(abs(mr-r)+abs(mc-c) for r,c in old))

    def test_split_disjoint_balance(self):
        sets=prepare_sets();self.assertEqual(sets,prepare_sets());ids=[]
        for name,rows in sets.items():
            ids += [r['id'] for r in rows]
            for d in ('up','down','left','right'):
                for c in (1,2):self.assertEqual(sum(r['direction']==d and r['color']==c for r in rows),len(rows)//8)
        self.assertEqual(len(ids),len(set(ids)));self.assertTrue(all(r['shape'].startswith('L') for r in sets['movement_novel']))

    def test_serialization_no_target_inference(self):
        data=prepare_sets();row=data['movement_novel'][0]
        text=render_prompt(row['input'],'movement',data['demonstrations'])
        self.assertTrue(text.endswith('Input: '+encode_grid(row['input'])+'\nOutput:'))
        with self.assertRaises(ValueError):encode_grid([9]*16)
        with self.assertRaises(ValueError):palette_ids(SimpleNamespace(encode=lambda *a,**k:[1,2]))

    def test_runner_and_failed_copy_gate(self):
        from run_grid_io import run
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'approved_manifest.json').write_text('{}');h=hashlib.sha256(b'{}').hexdigest()
            (p/'source_receipt.json').write_text(json.dumps(dict(status='verified',manifest_sha256=h)))
            data=prepare_sets();(p/'data.json').write_text(json.dumps(data))
            def generate(study,grid,task,demos,mode):return dict(tokens=[0]*16,decoded='0'*16,format_valid=True,output=[0]*16,native_argmax_legal=[True]*16)
            c=dict(resources={},model_sha256='fixed',data=str(p/'data.json'),wall_guard_seconds=60)
            with patch('run_grid_io.model_hash',return_value='fixed'),patch('run_grid_io.generate',generate):run(lambda:SimpleNamespace(model=None),p,c)
            result=json.loads((p/'results.json').read_text());self.assertEqual(result['status'],'complete');self.assertFalse(result['prerequisite']['copy_gate_passed'])
            self.assertEqual(result['summary']['movement_novel']['palette_only']['n'],64)


if __name__=='__main__':unittest.main()
