import json,tempfile,unittest,hashlib
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from skillforge.relational_program import family,OPTIONS,fit_bias,apply_bias,pair_summary,decision
from skillforge.joint_attention import metrics
from test_joint_attention import bank


class RelationTests(unittest.TestCase):
    def test_physics_and_pairs(self):
        a=family('acquisition');r=family('reserved')
        self.assertEqual((len(a),len(r)),(24,48));self.assertFalse({x['id'] for x in a}&{x['id'] for x in r})
        for row in a+r:
            state=row['physical_state'];answer=row['choices']['text'][row['choices']['label'].index(row['answerKey'])]
            if not state['contact'] or state['left']==state['right']:self.assertEqual(answer,'neither object')
            elif row['group']=='changes':self.assertEqual(answer,'both objects')
            else:
                left_warms=state['left']<state['right'];want_left=left_warms if row['group']=='warms' else not left_warms
                self.assertEqual(answer,'left object' if want_left else 'right object')
        correct=[dict(id=row['id'],correct=True) for row in r]
        self.assertEqual(pair_summary(correct,r),dict(n=24,both_correct=24,answer_changing_pairs=20))

    def test_bias_fixed_offsets_and_other_answers(self):
        rows=family('acquisition');records=[]
        for row in rows:
            gold=row['choices']['label'].index(row['answerKey']);records.append(dict(id=row['id'],gold=gold,normalized_logprobs=[-1.,-2.,-3.,-4.],correct=False))
        b=fit_bias(records,rows);self.assertEqual(b,fit_bias(records,rows));self.assertEqual(b['candidate_count'],625)
        self.assertAlmostEqual(sum(b['offsets']),0)
        extra=dict(id='other',choices=dict(text=['cat','dog']));score=dict(id='other',gold=0,normalized_logprobs=[-1.,-2.],correct=True)
        applied=apply_bias(records+[score],rows+[extra],b)
        self.assertEqual(applied[-1]['normalized_logprobs'],score['normalized_logprobs'])

    def test_decision_does_not_count_same_score_as_joint_evidence(self):
        rows=family('reserved')+[dict(id='keep',group='retention')]
        records=[dict(id=row['id'],correct=True) for row in rows]
        outputs={name:records for name in ('untouched','joint','wsp_only','asp_only','answer_bias')}
        scores={name:metrics(records,records,rows) for name in outputs}
        pairs={name:pair_summary(records,rows) for name in outputs}
        d=decision(scores,pairs,outputs,rows);self.assertFalse(d['continue_conceptual_asp'])
        self.assertFalse(d['criteria']['both_components_help'])

    def test_runner_freezes_program_and_bias_before_reserved(self):
        from run_relational_program import run
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'approved_manifest.json').write_text('{}');digest=hashlib.sha256(b'{}').hexdigest()
            (p/'source_receipt.json').write_text(json.dumps(dict(status='verified',manifest_sha256=digest)))
            acquired=family('acquisition');reserved_rows=family('reserved')+[dict(id='keep',group='retention',choices=dict(text=['cat','dog'],label=['A','B']),answerKey='A')]
            (p/'acq.json').write_text(json.dumps(acquired));reserved=p/'reserved.json';reserved.write_text(json.dumps(reserved_rows))
            (p/'guide.md').write_text('fixture');(p/'local.json').write_text(json.dumps(dict(codex_cli='fake')))
            np.savez(p/'parent.npz',**bank());reference=bank();reference['answer_token_ids']=np.array([1],np.float32)
            read=Path.read_text;seen=[]
            def guarded(path,*args,**kwargs):
                if path==reserved:
                    for name in ('frozen.json','program.json','program.npz','frozen_answer_bias.json'):self.assertTrue((p/name).exists())
                return read(path,*args,**kwargs)
            def score(row):
                gold=row['choices']['label'].index(row['answerKey']);return dict(id=row['id'],gold=gold,correct=True,normalized_logprobs=[-1.]*len(row['choices']['text']))
            class Program:
                def __init__(self,b):self.b=b
                def score(self,s,row,variant):seen.append(variant);return score(row)
            def collector(cli,catalog,instructions,folder,text,timeout,heartbeat):folder.mkdir();return 'FINAL',dict(thread_id='fixture')
            base=[score(row) for row in acquired]
            config=dict(resources={},runtime_verify='fake',linux_identity='fake',wall_guard_seconds=100,model_sha256='frozen',parent_bank=str(p/'parent.npz'),acquisition=str(p/'acq.json'),reserved=str(reserved),guide=str(p/'guide.md'),local_paths=str(p/'local.json'),catalog='fake',instructions='fake')
            with patch('run_relational_program.subprocess.run'),patch('run_relational_program.model_hash',return_value='frozen'),patch('run_relational_program.make_reference',return_value=(reference,base,base)),patch('run_relational_program.collect',collector),patch('run_relational_program.NeuralProgram',Program),patch.object(Path,'read_text',guarded):
                run(lambda:SimpleNamespace(model=None,tokenizer=SimpleNamespace(decode=lambda ids:'x')),p,config)
            result=json.loads((p/'results.json').read_text());self.assertEqual(result['status'],'complete');self.assertEqual(len(seen),4*49)
            self.assertFalse(result['decision']['continue_conceptual_asp'])


if __name__=='__main__':unittest.main()
