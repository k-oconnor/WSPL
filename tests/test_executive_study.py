import json
import unittest
from skillforge.controller_author import KINDS, construct, make_spec, teacher as block_teacher
from skillforge.executive_study import dataset, make_task, make_expanded_task, teacher, score, prompt, order_goals


class ExecutiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs={k:make_spec(k,20) for k in KINDS}
        cls.blocks={k:construct(block_teacher(s),s) for k,s in cls.specs.items()}

    def test_corpus_sizes_and_prompt_disjointness(self):
        data=dataset()
        self.assertEqual([len(x) for x in data.values()],[16384,256,256])
        sets=[{r['messages'][0]['content'] for r in rows} for rows in data.values()]
        self.assertEqual([len(x) for x in sets],[16384,256,256])
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertEqual({len(r['task']['objects']) for r in data['train']},{2,3,4,5})

    def test_expanded_geometry_and_reference_compositions(self):
        for n in [2,5,6,7,8]:
            for seed in [10,20,30]:
                task=make_expanded_task(seed,n,renamed=seed==30)
                locations=list(task['objects'].values())+list(task['points'].values())
                for i,a in enumerate(locations):
                    for b in locations[i+1:]:
                        self.assertGreater(max(abs(x-y) for x,y in zip(a,b)),.1)
                result=score(teacher(task),task,self.blocks,self.specs,False)
                self.assertTrue(result['task_success'],result)

    def test_eight_object_jam_preserves_seven_deliveries(self):
        task=make_expanded_task(42,8,jam=True)
        result=score(teacher(task),task,self.blocks,self.specs,False)
        self.assertTrue(result['safe_failure'],result)
        self.assertEqual(len(result['delivered']),7)

    def test_training_and_test_identifier_formats(self):
        ordinary=make_expanded_task(5,5)
        novel=make_expanded_task(5,8,renamed=True)
        self.assertTrue(all(n[0].islower() for n in ordinary['objects']))
        self.assertTrue(all(n.startswith('Unit_') for n in novel['objects']))

    def test_reference_varied_compositions(self):
        for n in [2,3,4,5]:
            for seed in [1,4,9]:
                task=make_task(seed,n)
                result=score(teacher(task),task,self.blocks,self.specs,False)
                self.assertTrue(result['task_success'],result)

    def test_wrong_order_rejected_despite_final_goals(self):
        task=make_task(1,3)
        names=list(task['objects'])
        task['before']=[[names[0],names[1]],[names[1],names[2]]]
        text='\n'.join('transfer '+n+' to '+task['goals'][n] for n in reversed(names))+'\nmove_to home'
        result=score(text,task,self.blocks,self.specs,False)
        self.assertEqual(result['execution_status'],'success')
        self.assertFalse(result['task_success'])

    def test_home_only_is_not_success(self):
        result=score('move_to home',make_task(5,2),self.blocks,self.specs,False)
        self.assertEqual(result['execution_status'],'success')
        self.assertFalse(result['task_success'])

    def test_jam_is_failure_not_success_and_preserves_progress(self):
        task=make_task(5,4,jam=True)
        result=score(teacher(task),task,self.blocks,self.specs,False)
        self.assertTrue(result['safe_failure'],result)
        self.assertFalse(result['task_success'])
        self.assertEqual(len(result['delivered']),3)
        self.assertNotIn('jammed',prompt(task))

    def test_repeated_delivery_rejected(self):
        task=make_task(7,2)
        n=order_goals(task)[0]
        text=teacher(task)+'transfer '+n+' to '+task['goals'][n]+'\nmove_to home'
        result=score(text,task,self.blocks,self.specs,False)
        self.assertFalse(result['task_success'])
        self.assertIn('repeat_delivery',result['violations'])

    def test_jam_does_not_excuse_earlier_order_violation(self):
        task=make_task(11,3)
        a,b,c=list(task['objects'])
        task['before']=[[a,b],[b,c]]
        task['jammed']=[c]
        text='\n'.join('transfer '+n+' to '+task['goals'][n] for n in [b,a,c])
        result=score(text,task,self.blocks,self.specs,False)
        self.assertEqual(result['reason'],'budget_exhausted')
        self.assertFalse(result['safe_failure'])

    def test_cycle_rejected(self):
        task=make_task(1,2)
        a,b=task['objects']
        task['before']=[[a,b],[b,a]]
        with self.assertRaises(ValueError):teacher(task)

    def test_invalid_program_counted(self):
        result=score('print(hello)',make_task(1,2),self.blocks,self.specs,False)
        self.assertFalse(result['constructed'])
        self.assertFalse(result['task_success'])


if __name__=='__main__':unittest.main()
