"""Synthetic real-graph checks; run in the separately pinned proof runtime."""
import unittest
import importlib.util
from skillforge.controller_author import construct,teacher
from skillforge.wspl_parse import parse_module
from skillforge.wspl_equiv import check_equivalence


@unittest.skipUnless(importlib.util.find_spec('z3'),'requires the pinned proof runtime')
class MechanismProofTests(unittest.TestCase):
    def test_signed_rule_equivalence(self):
        spec=dict(kind='move',gain=-.4,speed=.04)
        artifact=construct(teacher(spec),spec)
        result=check_equivalence(parse_module(teacher(spec)).program,artifact)
        self.assertEqual(result['status'],'EQUIVALENT')

    def test_equivalent_independently_authored_source(self):
        spec=dict(kind='move',gain=-.4,speed=.04)
        source=teacher(spec)
        alternate=source.replace('  vx =','  let redundant: Real[0,0] normalized_zero = 0;\n  vx =',1)
        artifact=construct(alternate,spec)
        result=check_equivalence(parse_module(source).program,artifact,require_source_binding=False)
        self.assertFalse(result['bound_to_source'])
        self.assertEqual(result['status'],'EQUIVALENT')

    def test_wrong_sign_has_replayed_counterexample(self):
        wanted=dict(kind='move',gain=-.4,speed=.04)
        wrong=dict(wanted,gain=.4)
        artifact=construct(teacher(wrong),wrong);identity=artifact.sha256
        source=parse_module(teacher(wanted)).program
        self.assertEqual(check_equivalence(source,artifact)['status'],'INVALID')
        result=check_equivalence(source,artifact,require_source_binding=False)
        self.assertEqual(result['status'],'DIFFERENT')
        self.assertTrue(result['witness_validated'])
        self.assertFalse(result['bound_to_source'])
        self.assertFalse(result['source_binding_required'])
        self.assertEqual(artifact.sha256,identity)


if __name__=='__main__':unittest.main()
