import unittest,tempfile,json,hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from run_arc_attention_transfer import run,summarize


class TransferTests(unittest.TestCase):
    def test_summaries(self):
        base=[dict(id='a',correct=False),dict(id='b',correct=True)]
        new=[dict(id='a',correct=True,raw_correct=True),dict(id='b',correct=False,raw_correct=False)]
        self.assertEqual(summarize(new,base),dict(n=2,correct=1,raw_correct=1,corrected=['a'],broken=['b']))

    def test_frozen_candidates_all_arms_records_and_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'approved_manifest.json').write_text('{}');h=hashlib.sha256(b'{}').hexdigest()
            (p/'source_receipt.json').write_text(json.dumps(dict(manifest_sha256=h)))
            bank=dict(K=np.zeros((6,3,64),np.float32),V=np.ones((6,3,64),np.float32),alpha=np.zeros((6,3),np.float32))
            np.savez(p/'bank.npz',**bank);np.savez(p/'key.npz',K=np.ones(64,np.float32))
            row=dict(id='fixture',question='synthetic conduction fixture',choices=dict(text=['a','b'],label=['A','B']),answerKey='A')
            (p/'data.jsonl').write_text(json.dumps(row)+'\n');calls=[]
            def score(study,row,bank,mode):
                self.assertTrue((p/'frozen_candidates.json').exists());calls.append(mode)
                return dict(id=row['id'],correct=True,raw_correct=True)
            config=dict(resources={},wall_guard_seconds=100,model_sha256='frozen',parent_bank=str(p/'bank.npz'),ridge_bank=str(p/'bank.npz'),curve_keys={'grid_key':str(p/'key.npz'),'astra_key':str(p/'key.npz')},datasets={'challenge':str(p/'data.jsonl')})
            with patch('run_arc_attention_transfer.score_boundary',score),patch('run_arc_attention_transfer.model_hash',return_value='frozen'):
                run(lambda:SimpleNamespace(model=None),p,config)
            result=json.loads((p/'results.json').read_text())
            self.assertEqual(result['status'],'complete');self.assertEqual(len(calls),6)
            self.assertEqual(calls.count('off'),1);self.assertEqual(calls.count('forced'),1)
            self.assertEqual(len(result['summary']['challenge']),6)


if __name__=='__main__':unittest.main()
