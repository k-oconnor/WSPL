"""Mock transcripts test experiment orchestration, not scientific performance."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT/'tools')]
from skillforge.authoring_study import ARMS,AuthoringTrial,digest,reference_program
from run_authoring_screen import AuthoringScreen,trial_specs,run
from prepare_authoring_requests import prepare
from test_authoring_study import text


def study():
    return dict(status='frozen',model_id='MOCK_ENGINEERING_ONLY',sampling_protocol='deterministic invalid mock',
                task_seeds=[1,2],draws=3,arms=list(ARMS),task_access_record='inspected unit fixtures',
                monetary_cap=1.,wall_guard_seconds=30)


def transcripts(config):
    result={}
    for trial_id,task,arm in trial_specs(config):
        trial=AuthoringTrial(task,arm,trial_id=trial_id,model_id=config['model_id'])
        result[trial_id]=dict(request_sha256=digest(trial.request()),turns=[],final_candidate=None)
    return result


class AuthoringScreenTests(unittest.TestCase):
    def test_missing_proposals_remain_failures_in_all_six_denominators(self):
        config=study()
        with tempfile.TemporaryDirectory() as tmp:
            result=AuthoringScreen().replay(config,transcripts(config),Path(tmp))
            self.assertEqual(result['trials'],36)
            self.assertTrue(all(a==dict(qualified=0,total=6) for a in result['arms'].values()))
            self.assertTrue((Path(tmp)/'results.json').exists())

    def test_feedback_chain_cannot_be_forged_or_task_substituted(self):
        config=study();bundle=transcripts(config);first=next(iter(bundle))
        bundle[first]['request_sha256']='0'*64
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'identity'):
                AuthoringScreen().replay(config,bundle,Path(tmp))
        bundle=transcripts(config)
        bundle[first]['turns']=[dict(candidate={},usage={},previous_feedback_sha256='forged')]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'preceding'):
                AuthoringScreen().replay(config,bundle,Path(tmp))

    def test_draft_budget_and_incomplete_pairing_block_screen(self):
        config=study()
        for key,value in [('status','draft'),('model_id',None),('monetary_cap',float('nan')),('task_access_record',None)]:
            invalid={**config,key:value}
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):AuthoringScreen().replay(invalid,{},Path(tmp))
        bundle=transcripts(config);bundle.pop(next(iter(bundle)))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):AuthoringScreen().replay(config,bundle,Path(tmp))

    def test_request_export_contains_no_reference_artifacts_or_final_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'requests';inventory=prepare(study(),output)
            self.assertEqual(len(inventory),39)
            self.assertTrue((output/'PARSER.md').exists())
            for trial_id,_,_ in trial_specs(study()):
                request=json.loads((output/f'{trial_id}.json').read_text())
                self.assertEqual(request['format'],'wspl.authoring_request.v1')
                if request['arm'].startswith('wspl_'):
                    self.assertIn('module',request['syntax'])
                    self.assertIn('not JSON IR',request['syntax'])
                self.assertNotIn('reference_program',request)
                self.assertNotIn('artifact',request)
                self.assertNotIn('withheld_inputs',request)
            with self.assertRaises(FileExistsError):prepare(study(),output)

    def test_direct_scientific_runner_requires_existing_gate_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):run(lambda:None,Path(tmp),{})

    def test_principal_contrasts_pair_by_task_draw_and_cap_failures_are_preserved(self):
        config=study();bundle=transcripts(config)
        for trial_id,task,arm in trial_specs(config):
            if arm=='wspl_on':
                bundle[trial_id]['turns']=[dict(candidate=text(task),
                    usage=dict(input_tokens=1,output_tokens=1,elapsed_seconds=0.,cost=0.),previous_feedback_sha256=None)]
        with tempfile.TemporaryDirectory() as tmp:
            result=AuthoringScreen().replay(config,bundle,Path(tmp))
            for contrast in result['principal_contrasts'].values():
                self.assertEqual(contrast['paired_differences'],[1]*6)
                self.assertEqual(contrast['mean'],1.)
        bundle=transcripts(config);first=next(iter(bundle))
        bundle[first]['turns']=[dict(candidate={},usage=dict(input_tokens=1,output_tokens=1,elapsed_seconds=0.,cost=2.),previous_feedback_sha256=None)]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'monetary cap'):
                AuthoringScreen().replay(config,bundle,Path(tmp))
            self.assertTrue((Path(tmp)/f'{first}.json').exists())
            self.assertFalse((Path(tmp)/'results.json').exists())


if __name__=='__main__':unittest.main()
