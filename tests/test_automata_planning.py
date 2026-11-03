"""Finite engineering cross-checks, not learned/physical benchmark results."""
import copy
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from skillforge.ltlf import parse,evaluate_closed_trace
from skillforge.automata import compile_dfa,CompilationLimit
from skillforge.planning import FiniteModel,synthesize,execute,PlanningLimit,ContractViolation,InfeasibleTask
from skillforge.__main__ import task_formula

def ap(n):return dict(op='ap',name=n)
def op(n,*args):return dict(op=n,args=list(args))
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def traces(atoms,max_length):
    letters=[dict(zip(atoms,bits)) for bits in itertools.product((False,True),repeat=len(atoms))]
    for n in range(1,max_length+1):yield from itertools.product(letters,repeat=n)


class CompilerTests(unittest.TestCase):
    def test_exhaustive_short_trace_agreement_with_independent_semantics(self):
        p,q=ap('p'),ap('q');t=dict(op='true');f=dict(op='false')
        formulas=[p,t,f,op('X',t),op('Xw',f),op('not',op('X',p)),op('X',op('not',p)),
            op('F',op('G',p)),op('G',op('F',q)),op('U',p,q),op('not',op('U',p,q)),
            op('X',op('Xw',f)),op('Xw',op('X',t)),op('G',op('implies',p,op('X',q))),
            op('F',op('and',p,op('X',op('F',q)))),op('U',op('Xw',p),op('G',q))]
        for node in formulas:
            formula=parse(node);dfa=compile_dfa(formula)
            for trace in traces(sorted(formula.atoms),4):
                self.assertEqual(dfa.accepts(trace),evaluate_closed_trace(formula,trace),(node,trace))

    def test_seeded_mixed_formula_agreement(self):
        rng=random.Random(421)
        def formula(depth):
            if depth==0:return rng.choice([ap('p'),ap('q'),dict(op='true'),dict(op='false')])
            name=rng.choice(['not','X','Xw','F','G','U','and','or','implies'])
            args=[formula(depth-1) for _ in range(2 if name in ('U','and','or','implies') else 1)]
            return op(name,*args)
        for _ in range(32):
            f=parse(formula(3));dfa=compile_dfa(f)
            for trace in traces(sorted(f.atoms),3):self.assertEqual(dfa.accepts(trace),evaluate_closed_trace(f,trace),(f,trace))

    def test_nonempty_trace_contract_and_boolean_values(self):
        dfa=compile_dfa(dict(op='true'))
        self.assertNotIn(dfa.initial,dfa.accepting)
        with self.assertRaises(ValueError):dfa.accepts([])
        with self.assertRaises(ValueError):compile_dfa(ap('p')).accepts([dict(p=.8)])
        with self.assertRaises(ValueError):compile_dfa(ap('p')).step(0,{})

    def test_pending_eventual_is_completable_but_collision_is_irreversible(self):
        f=compile_dfa(op('F',ap('goal')));q=f.step(0,dict(goal=False))
        self.assertNotIn(q,f.accepting);self.assertIn(q,f.completable)
        g=compile_dfa(op('G',op('not',ap('collision'))));q=g.step(0,dict(collision=True))
        self.assertNotIn(q,g.completable)

    def test_compilation_limits_are_explicit(self):
        with self.assertRaises(CompilationLimit):compile_dfa(op('and',ap('p'),ap('q')),max_atoms=1)
        with self.assertRaises(CompilationLimit):compile_dfa(ap('p'),max_states=1)
        with self.assertRaises(CompilationLimit):compile_dfa(ap('p'),max_bdd_nodes=2)

    def test_deterministic_serialization_and_complete_transition_table(self):
        node=op('F',op('and',ap('p'),op('X',ap('q'))));a=compile_dfa(node);b=compile_dfa(node)
        self.assertEqual(a.to_dict(),b.to_dict())
        self.assertTrue(all(len(row)==4 and all(0<=q<len(a.transitions) for q in row) for row in a.transitions))

    def test_existing_motor_task_example_acceptance_matches_reference(self):
        task=read('examples/pick_place.task.json');trace=read('examples/pick_place.trace.json')
        self.assertTrue(compile_dfa(task_formula(task)).accepts(trace))


class PlanningTests(unittest.TestCase):
    def model(self):return FiniteModel.from_dict(read('examples/discrete_pickup.model.json'))
    def dfa(self):return compile_dfa(task_formula(read('examples/discrete_pickup.task.json')))

    def test_success_and_bounded_timeout_failure_recovery(self):
        model=self.model();strategy=synthesize(self.dfa(),model,'start');self.assertTrue(strategy.realizable)
        for selector in (lambda s,a,o:o[0],lambda s,a,o:o[-1]):
            r=execute(strategy,model,selector);self.assertTrue(r['accepted'])
            self.assertTrue(evaluate_closed_trace(self.dfa().formula,r['trace']))
            self.assertEqual(r['states'][-1],'raised');self.assertEqual(len(r['trace']),len(r['actions'])+1)
        worst=execute(strategy,model,lambda s,a,o:o[-1])
        self.assertEqual(worst['actions'],['approach','retry_approach','grasp','retry_grasp','lift'])

    def test_unbounded_adversarial_retry_is_not_assumed_to_succeed(self):
        m=FiniteModel({'s':{'goal':False},'g':{'goal':True}}, {'s':{'try':['s','g']}}, {'try':['invoke_try']})
        strategy=synthesize(compile_dfa(op('F',ap('goal'))),m,'s')
        self.assertFalse(strategy.realizable)
        with self.assertRaises(InfeasibleTask):execute(strategy,m)

    def test_unavoidable_permanent_failure_is_unrealizable(self):
        m=FiniteModel({'s':{'goal':False},'dead':{'goal':False},'g':{'goal':True}}, {'s':{'try':['dead','g']}}, {'try':[]})
        self.assertFalse(synthesize(compile_dfa(op('F',ap('goal'))),m,'s').realizable)

    def test_command_guard_prevents_an_unsafe_shortcut(self):
        states={'s':{'holding':False,'goal':False},'h':{'holding':True,'goal':False},'g':{'holding':True,'goal':True}}
        model=FiniteModel(states,{'s':{'lift':['g'],'grasp':['h']},'h':{'lift':['g']}},{'lift':['invoke_lift'],'grasp':['invoke_grasp']})
        f=op('and',op('F',ap('goal')),op('G',op('implies',ap('invoke_lift'),ap('holding'))))
        r=execute(synthesize(compile_dfa(f),model,'s'),model)
        self.assertEqual(r['actions'],['grasp','lift'])

    def test_impossible_collision_constraint_is_not_reported_as_success(self):
        m=FiniteModel({'s':{'goal':False,'collision':False},'g':{'goal':True,'collision':True}}, {'s':{'go':['g']}},{'go':[]})
        f=op('and',op('F',ap('goal')),op('G',op('not',ap('collision'))))
        self.assertFalse(synthesize(compile_dfa(f),m,'s').realizable)

    def test_stop_has_one_terminal_observation_and_no_fake_action(self):
        m=self.model();r=execute(synthesize(compile_dfa(dict(op='true')),m,'start'),m)
        self.assertEqual(r['actions'],[]);self.assertEqual(len(r['trace']),1)
        self.assertTrue(all(not r['trace'][0][p] for p in m.controlled))

    def test_execution_budget_and_out_of_contract_outcomes_are_distinct(self):
        m=self.model();s=synthesize(self.dfa(),m,'start')
        r=execute(s,m,max_actions=0);self.assertEqual(r['status'],'budget_exhausted');self.assertFalse(r['accepted'])
        with self.assertRaises(ContractViolation):execute(s,m,lambda state,action,outcomes:'collision')

    def test_model_changes_and_undeclared_atoms_are_rejected(self):
        m=self.model();s=synthesize(self.dfa(),m,'start');d=read('examples/discrete_pickup.model.json')
        d['assumptions'].append('New external assumption')
        with self.assertRaises(ContractViolation):execute(s,FiniteModel.from_dict(d))
        with self.assertRaises(ValueError):synthesize(compile_dfa(ap('magic')),m,'start')

    def test_model_is_snapshot_and_rejects_invalid_structure(self):
        d=read('examples/discrete_pickup.model.json');m=FiniteModel.from_dict(d);old=m.identity
        d['states']['start']['collision']=True;self.assertEqual(m.identity,old)
        with self.assertRaises(TypeError):m.states['start']['collision']=True
        for states,edges,commands in [({'s':None},{},{}),({'s':{'p':.2}},{},{}),({'s':{'p':False}},{'s':{'go':['missing']}},{'go':[]}),({'s':{'p':False}},{},{'go':['p']})]:
            with self.assertRaises(ValueError):FiniteModel(states,edges,commands)

    def test_planning_limit_is_not_unrealizability(self):
        with self.assertRaises(PlanningLimit):synthesize(self.dfa(),self.model(),'start',max_product_states=1)

    def test_winning_rank_decreases_for_every_declared_outcome(self):
        m=self.model();s=synthesize(self.dfa(),m,'start')
        for node,action in s.policy.items():
            if action is None:continue
            world,q=node;nxt=s.dfa.step(q,m.valuation(world,action))
            self.assertTrue(all(s.rank[(o,nxt)]<s.rank[node] for o in m.transitions[world][action]))


class CLITests(unittest.TestCase):
    def run_cli(self,*args):
        env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')
        return subprocess.run([sys.executable,'-m','skillforge',*map(str,args)],cwd=ROOT,env=env,capture_output=True,text=True)

    def test_plan_execute_cli_worst_declared_outcomes(self):
        p=self.run_cli('plan','examples/discrete_pickup.task.json','examples/discrete_pickup.model.json','--initial','start','--execute','--outcome-choice','last')
        self.assertEqual(p.returncode,0,p.stderr);r=json.loads(p.stdout)
        self.assertTrue(r['realizable']);self.assertEqual(r['execution']['status'],'accepted')
        self.assertIn('retry_grasp',r['execution']['actions'])

    def test_compile_cli_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'dfa.json'
            p=self.run_cli('compile','examples/pick_place.task.json','--output',output)
            self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(output.read_text())['kind'],'ltlf_dfa')
            self.assertNotEqual(self.run_cli('compile','examples/pick_place.task.json','--output',output).returncode,0)

    def test_limit_and_unrealizable_exit_codes_differ(self):
        p=self.run_cli('compile','examples/pick_place.task.json','--max-atoms',1)
        self.assertEqual(p.returncode,2);self.assertEqual(json.loads(p.stderr)['status'],'construction_limit')
        with tempfile.TemporaryDirectory() as d:
            task=read('examples/discrete_pickup.task.json');task['formula']=dict(op='false');p=Path(d)/'task.json';p.write_text(json.dumps(task))
            result=self.run_cli('plan',p,'examples/discrete_pickup.model.json','--initial','start')
            self.assertEqual(result.returncode,3);self.assertFalse(json.loads(result.stdout)['realizable'])

    def test_clock_mismatch_and_vocabulary_rejected(self):
        p=self.run_cli('plan','examples/pick_place.task.json','examples/discrete_pickup.model.json','--initial','start')
        self.assertEqual(p.returncode,1);self.assertIn('clocks differ',p.stderr)
        task=read('examples/discrete_pickup.task.json');task['controllable'].append('holding')
        with self.assertRaises(ValueError):task_formula(task)
        for document in ([],dict(kind='task',schema_version=True)):
            with self.assertRaises(ValueError):task_formula(document)
        for document in ([],dict(schema_version=True)):
            with self.assertRaises(ValueError):FiniteModel.from_dict(document)

    def test_finite_model_schema_and_example(self):
        from jsonschema import Draft202012Validator
        schema=read('schemas/finite_model.schema.json');Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(read('examples/discrete_pickup.model.json'))
        Draft202012Validator(read('schemas/task.schema.json')).validate(read('examples/discrete_pickup.task.json'))


if __name__=='__main__':unittest.main()
