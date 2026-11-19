import unittest
from skillforge.controller_author import KINDS,make_spec,construct,teacher as block_teacher
from skillforge.closed_loop_control import Episode,make_task,teacher,records,training_data,prompt,evaluation_tasks


class ClosedLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs={k:make_spec(k,20) for k in KINDS}
        cls.blocks={k:construct(block_teacher(s),s) for k,s in cls.specs.items()}

    def rollout(self,task,blocks=None,scale=1):
        e=Episode(task,blocks or self.blocks,self.specs,scale)
        while not e.finished:e.execute(teacher(e.observation()))
        return e

    def test_long_reference_and_new_names(self):
        for n in (2,5,8,12):
            e=self.rollout(make_task(42+n,n,renamed=True))
            self.assertTrue(e.result(False)['task_success'])

    def test_transient_failure_requires_explicit_retry(self):
        e=Episode(make_task(42,3,'slip'),self.blocks,self.specs)
        e.execute(teacher(e.observation()))
        self.assertEqual(e.last['status'],'failure')
        self.assertFalse(e.delivered)
        self.assertFalse(e.finished)
        self.assertEqual(len(e.decisions),1)
        e.execute(teacher(e.observation()))
        self.assertEqual(len(e.delivered),1)

    def test_combined_disturbance_both_events_actually_occur(self):
        e=self.rollout(make_task(42,8,'combined'))
        self.assertTrue(e.result(False)['task_success'])
        self.assertEqual({x['kind'] for x in e.events},{'moved','slip'})
        self.assertGreaterEqual(e.result(False)['failed_calls'],2)

    def test_blocked_destination_does_not_auto_select_another(self):
        t=make_task(43,3,'blocked');e=Episode(t,self.blocks,self.specs)
        target=t['event_object'];e.execute('transfer '+target+' to '+t['goals'][target])
        self.assertEqual(e.last['reason'],'destination_blocked')
        self.assertEqual(e.ticks,0)
        self.assertFalse(e.delivered)
        self.assertTrue(e.blocked())

    def test_inverted_mechanism_requires_one_local_block_replacement(self):
        t=make_task(49,3)
        self.assertFalse(self.rollout(t,scale=-1).result(False)['task_success'])
        spec=dict(self.specs['move'],gain=-self.specs['move']['gain'])
        changed=dict(self.blocks,move=construct(block_teacher(spec),spec))
        e=self.rollout(t,changed,-1)
        self.assertTrue(e.result(False)['task_success'])
        self.assertEqual(sum(self.blocks[k].sha256!=changed[k].sha256 for k in KINDS),1)

    def test_omissions_and_multiple_actions_are_not_repaired(self):
        for source in ['stop','move_to home\nstop']:
            e=Episode(make_task(12,3),self.blocks,self.specs);e.execute(source)
            self.assertTrue(e.finished)
            self.assertFalse(e.result(False)['task_success'])
            self.assertEqual(len(e.result(False)['missing']),3)

    def test_prompt_does_not_disclose_hidden_event_schedule(self):
        e=Episode(make_task(15,4,'combined'),self.blocks,self.specs)
        text=prompt(e.observation())
        for key in ['event_object','disturbance','motion_scale','triggered']:
            self.assertNotIn(key,text)
        self.assertNotIn('ready',e.observation())

    def test_nominal_planning_preserves_initial_public_information(self):
        for disturbance in ('none','slip','moved','blocked','combined'):
            task=make_task(16,4,disturbance)
            nominal=dict(task,disturbance='blocked' if disturbance in ('blocked','combined') else 'none')
            self.assertEqual(Episode(task,self.blocks,self.specs).observation(),
                             Episode(nominal,self.blocks,self.specs).observation())

    def test_independent_monitor_rejects_out_of_order_delivery(self):
        t=make_task(21,2);a,b=sorted(t['objects']);t['before']=[[a,b]]
        e=Episode(t,self.blocks,self.specs);e.execute('transfer '+b+' to '+t['goals'][b])
        self.assertIn('ordering',e.result(False)['violations'])
        self.assertFalse(e.result(False)['task_success'])

    def test_regrasp_of_completed_work_is_detected(self):
        t=make_task(12,2);t['before']=[];a=sorted(t['objects'])[0]
        e=Episode(t,self.blocks,self.specs);e.execute('transfer '+a+' to '+t['goals'][a])
        e.execute('grasp '+a)
        self.assertIn('moved_after_delivery',e.result(False)['violations'])

    def test_repeated_blocked_call_is_bounded(self):
        t=make_task(43,2,'blocked');e=Episode(t,self.blocks,self.specs)
        a=t['event_object']
        while not e.finished:e.execute('transfer '+a+' to '+t['goals'][a])
        self.assertEqual(e.error,'decision_budget')
        self.assertLessEqual(len(e.decisions),e.decision_budget+1)

    def test_whole_mission_split_and_no_combined_training(self):
        data=training_data(self.blocks,self.specs,train_rows=40,dev_rows=20)
        self.assertEqual([len(v) for v in data.values()],[40,20])
        self.assertFalse({r['mission_id'] for r in data['train']}&{r['mission_id'] for r in data['development']})
        self.assertTrue(all(len(r['messages'][1]['content'].splitlines())==1 for rows in data.values() for r in rows))

    def test_reserved_panel_structure_includes_all_dependency_families(self):
        tasks=evaluation_tasks()
        self.assertEqual(len(tasks),144)
        self.assertEqual({t['dependency_family'] for t in tasks if t['panel']=='long12'},
                         {'chain','independent','fan_out','fan_in','random_dag'})


if __name__=='__main__':unittest.main()
