"""Frozen adapter attenuation and predeclared integration summaries."""
import copy,math
import numpy as np
from .capability_bridge import validate_plan
from .wsp_artifact import TensorArtifact
from .wsp_graph import node


def scaled(plan,alpha):
    if not 0<=alpha<=1:raise ValueError('strength')
    result=copy.deepcopy(plan)
    for name,document in result['branches'].items():
        artifact=TensorArtifact(document,max_elements=250000);arrays=artifact.arrays();graph=artifact.to_dict()['graph']
        if 'integration_gain' in arrays or any(n['id']=='integration_scale' for n in graph['nodes']):raise ValueError('reserved scale name')
        arrays['integration_gain']=np.array([alpha],np.float32)
        graph['nodes'].append(node('integration_scale','scale',{'x':graph['outputs']['y']},params={'gain':'integration_gain'}))
        graph['outputs']['y']='node:integration_scale'
        result['branches'][name]=TensorArtifact.from_arrays(arrays,graph=graph).to_dict()
    validate_plan(result);return result


def compact(report):
    return dict(prose_nll=report['prose_nll'],prose_targets=report['prose_targets'],
        grids={k:dict(n=v['n'],exact=v['exact'],examples=v['records'][:4]) for k,v in report['grids'].items()})


def choose_strength(reports):
    original=reports['1.0'];eligible=[]
    for key,r in reports.items():
        if r['grids']['move']['exact']>=math.ceil(.9*original['grids']['move']['exact']) and r['grids']['copy6']['exact']>=original['grids']['copy6']['exact']-1:
            eligible.append(key)
    return min(eligible,key=lambda key:(reports[key]['prose_nll'],-float(key)))


def assess(reports):
    base,old,new,scalar=[reports[k] for k in ('foundation','original','repair','scalar')]
    gap=old['prose_nll']-base['prose_nll']
    recovered=(old['prose_nll']-new['prose_nll'])/gap if gap>0 else None
    skill=all(new['grids'][k]['exact']>=math.ceil(.9*old['grids'][k]['exact']) for k in ('novel','larger'))
    copy=new['grids']['copy6']['exact']>=old['grids']['copy6']['exact']-1
    beyond=new['prose_nll']<scalar['prose_nll'] and all(new['grids'][k]['exact']>=scalar['grids'][k]['exact']-2 for k in ('novel','larger','copy6'))
    return dict(prose_gap_recovered=recovered,movement_retained=skill,copy_retained=copy,
        integration_pass=bool(recovered is not None and recovered>=.5 and skill and copy),beyond_scalar_control=beyond)
