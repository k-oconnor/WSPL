"""Strong reachability strategies in a supplied finite skill-effect model.

All declared outcomes are adversarial; no fairness/eventual-success assumption.
This is finite model-constrained synthesis, not unrestricted LTLf synthesis.
"""
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from .automata import DFA


class PlanningLimit(RuntimeError):pass
class InfeasibleTask(RuntimeError):pass
class ContractViolation(RuntimeError):pass


class FiniteModel:
    def __init__(self,states,transitions,commands,clock='abstract_tick',assumptions=()):
        if not isinstance(states,Mapping) or not states or not isinstance(transitions,Mapping) or not isinstance(commands,Mapping) or clock not in ('abstract_tick','motor_tick'):raise ValueError('Invalid finite model')
        if any(not isinstance(s,str) or not s for s in states):raise ValueError('State IDs must be nonempty strings')
        if any(not isinstance(labels,Mapping) for labels in states.values()):raise ValueError('Expected proposition maps')
        observed=set(next(iter(states.values())))
        for labels in states.values():
            if set(labels)!=observed or any(type(v) is not bool for v in labels.values()):raise ValueError('State propositions must be consistent and Boolean')
        if any(not isinstance(a,str) or not a for a in commands):raise ValueError('Invalid skill name')
        if any(not isinstance(bits,(list,tuple,set,frozenset)) for bits in commands.values()):raise ValueError('Commands must list propositions')
        if any(not isinstance(p,str) or not p for bits in commands.values() for p in bits):raise ValueError('Invalid command proposition')
        controlled=set().union(*(set(bits) for bits in commands.values()))
        if any(not isinstance(p,str) or not p for p in observed|controlled) or observed&controlled:raise ValueError('Observed/controlled propositions overlap or are invalid')
        if set(transitions)-set(states):raise ValueError('Unknown source state')
        copied={}
        for s in states:
            choices={}
            if not isinstance(transitions.get(s,{}),Mapping):raise ValueError('Expected available-skill mapping')
            for action,outcomes in transitions.get(s,{}).items():
                if action not in commands or not isinstance(outcomes,(list,tuple)) or not outcomes or any(not isinstance(o,str) or o not in states for o in outcomes):raise ValueError('Invalid skill outcome')
                choices[action]=tuple(dict.fromkeys(outcomes))
            copied[s]=MappingProxyType(choices)
        self.states=MappingProxyType({s:MappingProxyType(dict(v)) for s,v in states.items()})
        self.transitions=MappingProxyType(copied);self.commands=MappingProxyType({a:frozenset(bits) for a,bits in commands.items()})
        if not isinstance(assumptions,(list,tuple)) or any(not isinstance(a,str) for a in assumptions):raise ValueError('Invalid assumptions')
        self.observed=frozenset(observed);self.controlled=frozenset(controlled);self.clock=clock;self.assumptions=tuple(assumptions)

    @classmethod
    def from_dict(cls,data):
        if not isinstance(data,dict) or set(data)!={'schema_version','kind','clock','states','transitions','commands','assumptions'} or type(data['schema_version']) is not int or data['schema_version']!=1 or data['kind']!='finite_skill_model':raise ValueError('Invalid finite model document')
        return cls(data['states'],data['transitions'],data['commands'],data['clock'],data['assumptions'])

    def to_dict(self):
        return dict(clock=self.clock,states={s:dict(v) for s,v in self.states.items()},
            transitions={s:{a:list(v) for a,v in choices.items()} for s,choices in self.transitions.items()},
            commands={a:sorted(bits) for a,bits in self.commands.items()},assumptions=list(self.assumptions))

    @property
    def identity(self):return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def valuation(self,state,action=None):
        if state not in self.states or (action is not None and action not in self.transitions[state]):raise ContractViolation('Unavailable skill/state')
        bits=frozenset() if action is None else self.commands[action]
        return {**self.states[state],**{p:p in bits for p in self.controlled}}


@dataclass(frozen=True)
class Strategy:
    dfa: DFA
    initial: tuple[str,int]
    model_identity: str
    policy: object
    rank: object
    explored_states: int

    @property
    def realizable(self):return self.initial in self.rank

    def to_dict(self):
        return dict(schema_version=1,kind='finite_model_strategy',realizable=self.realizable,initial=list(self.initial),
            model_sha256=self.model_identity,explored_states=self.explored_states,
            guarantee='Strong finite reachability under all supplied outcomes; no fairness assumption',
            policy=[dict(world=s,automaton=q,action=self.policy[(s,q)],rank=r) for (s,q),r in sorted(self.rank.items())])


def synthesize(dfa,model,initial,max_product_states=10000):
    if initial not in model.states or max_product_states<1:raise ValueError('Invalid planning initial state/limit')
    if set(dfa.alphabet)-model.observed-model.controlled:raise ValueError('Formula references undeclared propositions')
    start=(initial,dfa.initial);queue=deque([start]);seen={start};edges={};rank={};policy={}
    while queue:
        node=queue.popleft();s,q=node
        if dfa.step(q,model.valuation(s)) in dfa.accepting:rank[node]=0;policy[node]=None
        choices={}
        for action,outcomes in model.transitions[s].items():
            nxt=dfa.step(q,model.valuation(s,action));targets=tuple((o,nxt) for o in outcomes);choices[action]=targets
            for target in targets:
                if target not in seen:
                    if len(seen)>=max_product_states:raise PlanningLimit('Finite product-state limit exceeded')
                    seen.add(target);queue.append(target)
        edges[node]=choices
    # Least fixed point excludes adversarial failure loops even if success is possible.
    while True:
        additions={}
        for node,choices in edges.items():
            if node in rank:continue
            candidates=[(1+max(rank[t] for t in targets),action) for action,targets in choices.items() if all(t in rank for t in targets)]
            if candidates:additions[node]=min(candidates)
        if not additions:break
        for node,(distance,action) in additions.items():rank[node]=distance;policy[node]=action
    return Strategy(dfa,start,model.identity,MappingProxyType(policy),MappingProxyType(rank),len(seen))


def execute(strategy,model,choose_outcome=None,max_actions=100):
    if model.identity!=strategy.model_identity:raise ContractViolation('Skill-effect model changed after synthesis')
    if not strategy.realizable:raise InfeasibleTask('No strong strategy under the supplied outcome model')
    if max_actions<0:raise ValueError('Invalid action budget')
    node=strategy.initial;trace=[];actions=[];states=[node[0]]
    for step in range(max_actions+1):
        s,q=node;action=strategy.policy[node]
        if action is None:
            trace.append(model.valuation(s));accepted=strategy.dfa.accepts(trace)
            if not accepted:raise ContractViolation('Synthesized STOP failed automaton acceptance')
            return dict(status='accepted',trace=trace,actions=actions,states=states,accepted=True,terminal_valuations=1)
        if step==max_actions:
            trace.append(model.valuation(s))
            return dict(status='budget_exhausted',trace=trace,actions=actions,states=states,accepted=strategy.dfa.accepts(trace),terminal_valuations=1)
        valuation=model.valuation(s,action);trace.append(valuation);q_next=strategy.dfa.step(q,valuation)
        outcomes=model.transitions[s][action];outcome=outcomes[0] if choose_outcome is None else choose_outcome(s,action,outcomes)
        if outcome not in outcomes:raise ContractViolation('Observed outcome is outside the skill contract')
        next_node=(outcome,q_next)
        if next_node not in strategy.rank or strategy.rank[next_node]>=strategy.rank[node]:raise ContractViolation('Strategy progress invariant failed')
        actions.append(action);states.append(outcome);node=next_node
    raise AssertionError('Unreachable')
