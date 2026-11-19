import json
import unittest
from skillforge.retention_pilot import DEFAULT_SOURCE,compile_selector,diagnose,parse_request,validate_probe
from skillforge.adaptation_pilot import MappedWorld


class RetentionTests(unittest.TestCase):
    def test_observation_only_selection_on_return(self):
        s=compile_selector(DEFAULT_SOURCE.replace('original = 1','original = dx').replace('repaired = 0','repaired = dy'))
        for mapping,wanted in [([[1,0],[0,1]],'original'),([[0,-.8],[1.2,0]],'repaired'),([[1,0],[0,1]],'original')]:
            w=MappedWorld([-.8,-.8],{},bound=1.024);w.mapping=mapping
            selected,r=diagnose(w,[.005,0.],s)
            self.assertEqual(selected,wanted);self.assertEqual(r['physical_steps'],1)
            self.assertNotEqual(w.position,(-.8,-.8));self.assertEqual(set(r['inputs']),{'u','v','dx','dy'})
    def test_bounds_and_schema(self):
        for x in [[.02,0],[float('nan'),0],[],[True,0]]:
            with self.assertRaises(ValueError):validate_probe(x)
        for a in [{},dict(tool='finish',prediction=''),dict(tool='finish',prediction='ok',regime=0)]:
            with self.assertRaises(ValueError):parse_request(json.dumps(a))
    def test_port_and_capacity(self):
        with self.assertRaises(ValueError):compile_selector(DEFAULT_SOURCE.replace('input u:', 'input regime:'))
        with self.assertRaises(ValueError):compile_selector(DEFAULT_SOURCE.replace('Real[-1,1]','Real[0,1]'))
    def test_tie_is_original(self):
        s=compile_selector(DEFAULT_SOURCE.replace('original = 1','original = 0'))
        w=MappedWorld([0,0],{});w.mapping=[[0,-.8],[1.2,0]]
        self.assertEqual(diagnose(w,[0,0],s)[0],'original')


if __name__=='__main__':unittest.main()
