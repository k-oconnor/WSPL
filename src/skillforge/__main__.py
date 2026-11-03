"""Engineering compiler/planner CLI; no robot, learning or LLM API execution."""
import argparse
import json
from pathlib import Path
import sys
from .automata import compile_dfa,CompilationLimit
from .ltlf import parse
from .planning import FiniteModel,synthesize,execute,PlanningLimit,ContractViolation


def task_formula(document):
    if not isinstance(document,dict) or document.get('kind')!='task' or type(document.get('schema_version')) is not int or document['schema_version']!=1:raise ValueError('Expected typed task document')
    if document.get('clock') not in ('motor_tick','abstract_tick') or document.get('stop_requires_acceptance') is not True:raise ValueError('Invalid task clock/STOP rule')
    observed=document.get('observable');controlled=document.get('controllable')
    if not isinstance(observed,list) or not isinstance(controlled,list) or any(not isinstance(p,str) for p in observed+controlled):raise ValueError('Invalid proposition declarations')
    if len(set(observed+controlled))!=len(observed+controlled):raise ValueError('Duplicate/overlapping proposition roles')
    for name in observed+controlled:parse(dict(op='ap',name=name))
    formula=parse(document['formula'])
    if formula.atoms-set(observed+controlled):raise ValueError('Formula uses undeclared proposition')
    return formula


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    for name in ('compile','plan'):
        p=commands.add_parser(name);p.add_argument('task',type=Path);p.add_argument('--output',type=Path)
        p.add_argument('--max-atoms',type=int,default=8);p.add_argument('--max-states',type=int,default=512)
        if name=='plan':
            p.add_argument('model',type=Path);p.add_argument('--initial',required=True)
            p.add_argument('--max-product-states',type=int,default=10000);p.add_argument('--execute',action='store_true')
            p.add_argument('--outcome-choice',choices=['first','last'],default='first');p.add_argument('--max-actions',type=int,default=100)
    args=parser.parse_args(argv)
    try:
        task=json.loads(args.task.read_text(encoding='utf-8'));dfa=compile_dfa(task_formula(task),max_atoms=args.max_atoms,max_states=args.max_states)
        if args.command=='compile':result=dfa.to_dict();result['clock']=task['clock'];code=0
        else:
            model=FiniteModel.from_dict(json.loads(args.model.read_text(encoding='utf-8')))
            if model.clock!=task['clock']:raise ValueError('Task/model clocks differ')
            if set(task['observable'])-model.observed or set(task['controllable'])-model.controlled:raise ValueError('Task/model proposition roles differ')
            strategy=synthesize(dfa,model,args.initial,args.max_product_states);result=strategy.to_dict();result['clock']=model.clock
            result['assumptions']=list(model.assumptions)
            code=0 if strategy.realizable else 3
            if args.execute and strategy.realizable:
                choose=lambda state,action,outcomes:outcomes[0 if args.outcome_choice=='first' else -1]
                result['execution']=execute(strategy,model,choose,args.max_actions)
                if result['execution']['status']!='accepted':code=5
        text=json.dumps(result,indent=2,allow_nan=False)+'\n'
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            with args.output.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
        else:print(text,end='')
        return code
    except (CompilationLimit,PlanningLimit) as error:
        print(json.dumps(dict(status='construction_limit',error=str(error))),file=sys.stderr);return 2
    except ContractViolation as error:
        print(json.dumps(dict(status='contract_violation',error=str(error))),file=sys.stderr);return 4
    except (ValueError,KeyError,TypeError,OSError) as error:
        print(json.dumps(dict(status='invalid_input',error=str(error))),file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
