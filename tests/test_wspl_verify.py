"""Solver-backed real-graph engineering fixtures, not physical guarantees."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from skillforge.wspl_artifact import ReLUArtifact
from skillforge.wspl_verify import verify_contract,verify_suite


def example():return ReLUArtifact.load(ROOT/'examples/wspl/brake_patch.relu.json')
def linear(terms=None,bias='0',unit='1'):return dict(terms=terms or {},bias=bias,unit=unit)
def atom(op,key,value,unit='1'):return dict(op=op,left=linear({key:'1'},unit=unit),right=linear(bias=value,unit=unit))
def contract(artifact,assumption=None,conclusion=None):return dict(format='wspl.io_contract.v0',name='test',artifact_sha256=artifact.sha256,semantics='rational_real_graph',assumptions=assumption or {'op':'true'},ensures=conclusion or atom('eq','output.a','-1'))
def emergency(a):return contract(a,dict(op='and',args=[atom('le','input.d','1/5'),atom('ge','input.v','2/5')]))


@unittest.skipUnless(importlib.util.find_spec('z3'),'optional Z3 unavailable')
class VerifierTests(unittest.TestCase):
    def test_all_three_brake_contracts_are_nonvacuous_real_graph_proofs(self):
        a=example();suite=json.loads((ROOT/'examples/wspl/brake.contracts.json').read_text())
        report=verify_suite(a,suite)
        self.assertEqual(report['status'],'PROVED',report)
        self.assertEqual(len(report['results']),3)
        for r in report['results']:
            self.assertTrue(r['nonvacuous']);self.assertEqual(r['violation_query'],'unsat')
            self.assertFalse(r['independently_checked_proof_trace'])
            self.assertEqual(r['numerical_execution'],'not_established')
            self.assertEqual(r['source_equivalence'],'not_checked')

    def test_reduced_braking_gain_is_refuted_with_replayed_witness(self):
        d=example().to_dict()
        for row in d['layers'][1]['weights']:row[0]='-1';row[1]='1'
        a=ReLUArtifact(d);r=verify_contract(a,emergency(a))
        self.assertEqual(r['status'],'REFUTED',r);self.assertTrue(r['witness_validated'])
        x=r['counterexample']['inputs'];self.assertNotEqual(a.evaluate(x)['a'],-1)

    def test_wrong_input_column_order_refutes_actual_weights(self):
        d=example().to_dict()
        for row in d['layers'][0]['weights']:row[0],row[1]=row[1],row[0]
        a=ReLUArtifact(d);r=verify_contract(a,emergency(a))
        self.assertEqual(r['status'],'REFUTED',r);self.assertTrue(r['witness_validated'])

    def test_vacuity_and_stale_hash_are_invalid_not_proved(self):
        a=example();c=emergency(a);c['artifact_sha256']='0'*64
        self.assertEqual(verify_contract(a,c)['status'],'INVALID')
        c=contract(a,atom('lt','input.d','0'))
        r=verify_contract(a,c);self.assertEqual(r['status'],'INVALID');self.assertFalse(r['nonvacuous'])

    def test_budget_and_unsupported_execution_are_unknown(self):
        a=example()
        self.assertEqual(verify_contract(a,emergency(a),timeout_ms=0)['status'],'UNKNOWN')
        self.assertEqual(verify_contract(a,emergency(a),max_parameters=1)['status'],'UNKNOWN')
        c=emergency(a);c['semantics']='ieee_float32'
        self.assertEqual(verify_contract(a,c)['status'],'UNKNOWN')

    def test_units_unknown_ports_and_boolean_schema_are_rejected(self):
        a=example();c=emergency(a);c['ensures']['left']['unit']='m'
        self.assertEqual(verify_contract(a,c)['status'],'INVALID')
        c=contract(a,conclusion=atom('eq','output.missing','0'))
        self.assertEqual(verify_contract(a,c)['status'],'INVALID')
        c=contract(a,conclusion={'op':'and','args':[]})
        self.assertEqual(verify_contract(a,c)['status'],'INVALID')

    def test_approach_bounds_and_direction_contracts(self):
        a=ReLUArtifact.load(ROOT/'examples/wspl/approach_p.relu.json')
        report=verify_suite(a,json.loads((ROOT/'examples/wspl/approach.contracts.json').read_text()))
        self.assertEqual(report['status'],'PROVED',report)
        self.assertTrue(all(r['nonvacuous'] for r in report['results']))


if __name__=='__main__':unittest.main()
