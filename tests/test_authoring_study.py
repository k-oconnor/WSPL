"""Inspected development fixtures; no scientific proposer trials."""
from fractions import Fraction as Q
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from skillforge.authoring_study import (
    AuthoringTrial, INPUTS, PARAMETERS, WIDTHS, build_candidate, cases,
    grade, mechanism, program, reference_program, ref, target,
)
from skillforge.wspl_core import ScalarProgram

USAGE = dict(input_tokens=10, output_tokens=10, elapsed_seconds=0.1, cost=0.)


def expression(task):
    def literal(i, pos):return INPUTS[i] if pos else f'(1-{INPUTS[i]})'
    clauses = ['max(0,'+'+'.join(literal(i,p) for i,p in c)+'-1)' for c in task['clauses']]
    return 'clip('+'+'.join(clauses)+',0,1)'


def sparse(artifact):
    return dict(format='wspl.sparse_weights.v0', layers=[
        dict(rows=[{str(i):v for i,v in enumerate(row) if Q(v)} for row in layer['weights']], bias=layer['bias'])
        for layer in artifact.to_dict()['layers']])


class AuthoringStudyTests(unittest.TestCase):
    def trial(self, arm='wspl_off', **kwargs):
        return AuthoringTrial(mechanism(1), arm, trial_id='unit-inspected', model_id='mock-no-provider', **kwargs)

    def test_all_three_representations_have_same_capacity_and_boolean_behavior(self):
        for seed in range(1,9):
            task = mechanism(seed)
            source = reference_program(task)
            a = build_candidate('wspl', source)
            for representation,candidate in [('wspl',source), ('code',expression(task)), ('weights',sparse(a))]:
                graph=build_candidate(representation,candidate)
                self.assertEqual(graph.to_dict()['allocation']['widths'],list(WIDTHS))
                self.assertEqual(graph.to_dict()['allocation']['parameters'],PARAMETERS)
                self.assertEqual(grade(task,graph,cases())['status'],'PROVED')
            # Independent source/runtime check includes non-Boolean probes;
            # finite-domain study proof still covers only Boolean cases.
            for x in [dict(zip(INPUTS,[Q(1,3)]*4)),dict(zip(INPUTS,[Q(0),Q(1,2),Q(1),Q(1,4)]))]:
                self.assertEqual(a.evaluate(x),ScalarProgram(source).evaluate(x))

    def test_feedback_off_cannot_see_verification_or_withheld_rows(self):
        for arm in ('wspl_off','wspl_on'):
            trial=self.trial(arm)
            feedback=trial.submit(reference_program(trial.task),USAGE)
            self.assertEqual(feedback['development']['evaluated'],4)
            self.assertEqual('verification' in feedback,arm.endswith('_on'))
            self.assertNotIn('withheld_inputs',json.dumps(feedback))
            final=trial.finalize()
            self.assertTrue(final['qualification'])
            self.assertEqual(final['non_development_inputs']['evaluated'],12)
            with self.assertRaises(ValueError):trial.finalize()
            with self.assertRaises(ValueError):trial.submit({},USAGE)

    def test_invalid_candidates_are_charged_and_cannot_change_contract(self):
        trial=self.trial()
        source=reference_program(trial.task)
        source['inputs']['power']['upper']='2'
        self.assertEqual(trial.submit(source,USAGE)['status'],'INVALID')
        final=trial.finalize()
        self.assertFalse(final['qualification'])
        self.assertEqual(final['total_tokens'],20)
        self.assertEqual(final['candidates'],1)

    def test_malformed_json_shapes_are_logged_as_invalid_not_silently_skipped(self):
        for arm,candidate in [('wspl_off',[]),('weights_on',None),('code_on',{}),
                              ('wspl_on',{'outputs':None})]:
            trial=self.trial(arm)
            self.assertEqual(trial.submit(candidate,USAGE)['status'],'INVALID')
            self.assertEqual(trial.finalize()['total_tokens'],20)

    def test_fallback_uses_last_valid_never_best_final_score(self):
        trial=self.trial()
        trial.submit(reference_program(trial.task),USAGE)
        trial.submit(program(ref('power')),USAGE)
        final=trial.finalize()
        self.assertEqual(final['selected_candidate'],1)
        self.assertFalse(final['qualification'])
        other=self.trial()
        other.submit(reference_program(other.task),USAGE)
        other.submit({},USAGE)
        self.assertFalse(other.finalize(1)['qualification'])

    def test_token_and_candidate_exhaustion_keep_failures_in_denominator(self):
        trial=self.trial(max_tokens=10)
        self.assertEqual(trial.submit(reference_program(trial.task),USAGE)['status'],'BUDGET_EXCEEDED')
        self.assertFalse(trial.finalize()['qualification'])
        trial=self.trial(max_candidates=1)
        trial.submit({},USAGE)
        with self.assertRaises(ValueError):trial.submit({},USAGE)

    def test_proposal_code_is_parsed_without_execution(self):
        for code in ["__import__('os').system('echo bad')",'power.__class__','[x for x in []]',
                     'power if lock else switch_a','power*switch_a','power / lock']:
            with self.assertRaises((ValueError,SyntaxError)):
                build_candidate('code',code)
        self.assertEqual(build_candidate('code','clip(1/2 * power,0,1)').evaluate(dict(zip(INPUTS,[0,0,1,0])))['next_open'],Q(1,2))

    def test_mutations_and_empty_domains_are_not_qualified(self):
        task=mechanism(1)
        a=build_candidate('wspl',reference_program(task))
        self.assertEqual(grade(task,a,cases())['correct'],16)
        with self.assertRaises(ValueError):grade(task,a,[])
        d=sparse(a);d['layers'][-1]['bias']=['1']
        self.assertEqual(grade(task,build_candidate('weights',d),cases())['status'],'REFUTED')
        d=sparse(a);d['layers'][0]['rows'][0]={'4':'1'}
        with self.assertRaises(ValueError):build_candidate('weights',d)


if __name__=='__main__':unittest.main()
