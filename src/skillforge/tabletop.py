"""Independent finite tabletop macro-effect fixture; no neural/robot execution.

No requested task or hierarchy is used in generating available actions/effects.
One abstract tick is an entire pickup/place/home macro outcome. Grasp failures
have an explicit finite remaining budget, never an implicit fairness promise.
"""
from collections import deque
from .hierarchy import Hierarchy

LOCATIONS=('source','buffer','tray','held')


def tabletop_model(blocked_access=False,movable_blocker=True,failure_budget=0,failure_object='b',max_states=2000):
    if type(blocked_access) is not bool or type(movable_blocker) is not bool or type(max_states) is not int or max_states<1:raise ValueError('Invalid world configuration')
    if type(failure_budget) is not int or not 0<=failure_budget<=2 or failure_object not in ('a','b'):raise ValueError('Invalid fixture failure configuration')
    initial=('source','source',True,failure_budget,'none')
    commands={**{f'pickup_{o}':[f'invoke_pickup_{o}'] for o in ('a','b')},
        **{f'place_{o}_{t}':[f'invoke_place_{o}'] for o in ('a','b') for t in LOCATIONS[:-1]},'return_home':['invoke_return_home']}
    def outcomes(state):
        a,b,home,left,event=state;locations=[a,b];result={}
        if 'held' not in locations:
            for i,obj in enumerate(('a','b')):
                if (obj=='a' and not movable_blocker) or (obj=='b' and blocked_access and a=='source'):continue
                new=locations.copy();new[i]='held';targets=[(*new,False,left,'picked_'+obj)]
                if left and obj==failure_object:targets.append((a,b,False,left-1,'grasp_failed'))
                result['pickup_'+obj]=targets
            if not home:result['return_home']=[(a,b,True,left,'home')]
        else:
            i=locations.index('held');obj=('a','b')[i]
            for target in LOCATIONS[:-1]:
                new=locations.copy();new[i]=target
                result[f'place_{obj}_{target}']=[(*new,False,left,f'placed_{obj}_{target}')]
        return result
    def labels(state):
        a,b,home,left,event=state
        return {**{f'{obj}_at_{place}':locations==place for obj,locations in [('a',a),('b',b)] for place in LOCATIONS[:-1]},
            **{f'picked_{obj}':event=='picked_'+obj for obj in ('a','b')},
            **{f'placed_{obj}_{place}':event==f'placed_{obj}_{place}' for obj in ('a','b') for place in LOCATIONS[:-1]},
            'home':home,'access_clear':a!='source','holding_a':a=='held','holding_b':b=='held','grasp_failed':event=='grasp_failed','collision':False}
    ids={initial:'s000'};queue=deque([initial]);states={};edges={}
    while queue:
        state=queue.popleft();key=ids[state];states[key]=labels(state);edges[key]={}
        for action,targets in outcomes(state).items():
            row=[]
            for target in targets:
                if target not in ids:
                    if len(ids)>=max_states:raise ValueError('Tabletop fixture state limit exceeded')
                    ids[target]=f's{len(ids):03d}';queue.append(target)
                row.append(ids[target])
            edges[key][action]=row
    return dict(schema_version=1,kind='finite_skill_model',clock='abstract_tick',states=states,transitions=edges,commands=commands,
        assumptions=['Supplied two-object macro-effect world; not a physics model or a learned skill library.',
            'Pickup/place are abstract subtask outcomes; their primitive execution is not modeled.',
            f'At most {failure_budget} grasp failures for object {failure_object}; remaining retries then succeed.',
            'One held object; source, buffer and tray have sufficient noncolliding capacity.']),ids[initial]


def build_case(suite,case_id):
    if not isinstance(suite,dict) or type(suite.get('schema_version')) is not int or suite['schema_version']!=1 or suite.get('kind')!='hierarchical_fixture_suite':raise ValueError('Invalid hierarchy suite')
    cases=suite['cases']
    if not isinstance(cases,list) or any(not isinstance(c,dict) for c in cases):raise ValueError('Invalid case list')
    if len({c['case_id'] for c in cases})!=len(cases):raise ValueError('Duplicate case IDs')
    case=next((c for c in cases if c['case_id']==case_id),None)
    if case is None:raise ValueError('Unknown hierarchical case')
    hierarchy=Hierarchy(suite['templates']);tree,stats=hierarchy.expand(case['root'])
    formula=hierarchy.formula(case['root'],case['terminal_goals']);model,initial=tabletop_model(**case['world'])
    controlled=sorted(set().union(*(set(p) for p in model['commands'].values())))
    if formula.atoms-set(next(iter(model['states'].values())))-set(controlled):raise ValueError('Hierarchy summary references unknown world proposition')
    task=dict(schema_version=1,kind='task',task_id=case_id,instruction=case['instruction'],
        observable=sorted(next(iter(model['states'].values()))),controllable=controlled,bindings={'object_a':'a','object_b':'b','tray':'tray','buffer':'buffer'},
        formula=formula.to_dict(),clock='abstract_tick',stop_requires_acceptance=True,status='example_not_executed')
    return dict(task=task,model=model,initial=initial,hierarchy=tree,statistics=stats,
        metadata=dict(case_id=case_id,split=case['split'],expected_realizable=case['expected_realizable'],
            hierarchy_sha256=hierarchy.identity,outcome_words=hierarchy.outcome_words(case['root']),
            evaluation_scope='Supplied hierarchy; flat planning over verified-by-definition macro summaries. No primitive hierarchical executor or physical skill execution.'))
