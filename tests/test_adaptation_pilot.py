import copy
import json
import unittest
from skillforge.adaptation_pilot import probe,candidate,parse_action,packet,episode,tasks
from skillforge.controller_author import make_spec,teacher,construct
from skillforge.closed_loop_control import teacher as executive


class AdaptationTests(unittest.TestCase):
    def setUp(self):
        self.specs={k:make_spec(k,0) for k in ('move','near','holding','close','open','safe')}
        self.specs['move'].update(gain=.73,speed=.07)
        self.specs['near'].update(radius=.037,metric='l1');self.specs['safe']['bound']=1.024
        self.blocks={k:construct(teacher(v),v) for k,v in self.specs.items()}
        self.mapping=[[0.,-.8],[1.2,0.]]
        self.source=teacher(self.specs['move']).replace('0.73 * x','0.6 * y').replace('0.73 * y','-0.9 * x')
    def test_probe_and_validation(self):
        r=probe([[.01,0.],[0.,.01]],self.mapping)
        self.assertEqual(r[0]['observed_displacement'],(0.,.012))
        self.assertEqual(r[1]['observed_displacement'],(-.008,0.))
        for c in [[[float('nan'),0]],[[1,0]],[]]:
            with self.assertRaises(ValueError):probe(c,self.mapping)
    def test_isolated_replacement(self):
        hashes={k:b.sha256 for k,b in self.blocks.items()}
        new=candidate(dict(tool='install_wspl',source=self.source),self.blocks['move'],self.specs['move'])
        for n in [3,5,8,12]:
            t=tasks(400+n,[n])[0]
            e=episode(t,dict(self.blocks,move=new),self.specs,self.mapping)
            while not e.finished:e.execute(executive(e.observation()))
            self.assertTrue(e.result(False)['task_success'])
        self.assertEqual(hashes,{k:b.sha256 for k,b in self.blocks.items()})
    def test_native_identity(self):
        old=self.blocks['move'];new=candidate(dict(tool='install_native',layers=old.to_dict()['layers']),old,self.specs['move'])
        self.assertEqual(old.evaluate({'x':'1','y':'-1'}),new.evaluate({'x':'1','y':'-1'}))
    def test_schema_and_fit_bounds(self):
        self.assertEqual(parse_action('{"tool":"finish","prediction":"retain"}')['tool'],'finish')
        for x in [{},dict(tool='finish',prediction=''),dict(tool='finish',prediction='a',code='x'),dict(tool='fit',prediction='x',updates=0,rows=[])]:
            with self.assertRaises(ValueError):parse_action(json.dumps(x))
    def test_prompt_does_not_contain_hidden_mechanism(self):
        text=packet(self.blocks['move'],self.specs,[],{})
        self.assertNotIn('1.2',text);self.assertNotIn('-0.8',text)
        self.assertIn('Last accepted',text)
    def test_partitions_distinct(self):
        sets=[{json.dumps(t,sort_keys=True) for t in tasks(s,[3]*8)} for s in [11000000,12000000,13000000]]
        for i in range(3):
            for j in range(i):self.assertFalse(sets[i]&sets[j])


if __name__=='__main__':unittest.main()
