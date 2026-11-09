"""One fixed attention/weight artifact, a relational fixture, and a bias control."""
from itertools import product
import numpy as np
from .arc_attention import score_boundary
from .joint_attention import weight_edit

OPTIONS=('left object','right object','neither object','both objects')
TASKS=('warms','cools','changes')


def make_program_reference(study,parent,rows):
    """Acquisition-only access to delivery inputs and downstream states."""
    import torch
    from .joint_attention import make_reference
    from .arc_eval import encode_choices
    from .attention_probe import intervention
    reference,base,initial=make_reference(study,parent,rows)
    queries=[];partitions=[]
    for row in rows:
        encoded=encode_choices(study.tokenizer,row);tokens,start=encoded[0]
        ids=torch.tensor([tokens[:start]],device='cuda');captured={}
        with torch.inference_mode(),intervention(capture=captured):
            study.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
        q=captured['Q'][2][:3].astype(np.float64);k=captured['native_K'][2][0].astype(np.float64)
        logits=q@k.T/8;peak=logits.max(1)
        queries.append(q);partitions.append(peak+np.log(np.exp(logits-peak[:,None]).sum(1)))
    reference['acquisition_Q']=np.asarray(queries,np.float32)
    reference['acquisition_logZ']=np.asarray(partitions,np.float32)
    return reference,base,initial


class NeuralProgram:
    """Fixed execution hierarchy; no access to labels, task IDs or a solver."""
    def __init__(self,bank):self.bank=bank

    def score(self,study,row,variant='joint'):
        if variant not in ('joint','wsp_only','asp_only','untouched'):raise ValueError('variant')
        with weight_edit(study.model,self.bank,variant in ('joint','wsp_only')):
            return score_boundary(study,row,self.bank,'retrieval' if variant in ('joint','asp_only') else 'off')


def family(split):
    # Each adjacent pair is a declared physical intervention. This label generator
    # is preparation/evaluation code, NEVER part of the deployed NeuralProgram.
    if split=='acquisition':
        pairs=[((80,20,True),(20,80,True)),((60,30,True),(60,30,False)),
               ((25,65,True),(25,65,False)),((40,40,True),(55,40,True))]
    elif split=='reserved':
        pairs=[((75,15,True),(15,75,True)),((54,32,True),(32,54,True)),
               ((68,24,True),(24,68,True)),((38,12,True),(12,38,True)),
               ((90,10,True),(90,10,False)),((11,67,True),(11,67,False)),
               ((34,34,True),(54,34,True)),((17,17,True),(17,47,True))]
    else:raise ValueError('split')
    rows=[]
    for pair,states in enumerate(pairs):
        for member,(left,right,contact) in enumerate(states):
            if split=='acquisition':
                setup=f'Two solid objects are isolated from their surroundings. The left object starts at {left} degrees Celsius; the right object starts at {right} degrees Celsius. '
                setup+=('They exchange heat through direct thermal contact. ' if contact else 'A perfect thermal insulator prevents any heat exchange between them. ')
            else:
                setup=f'In an enclosure that prevents heat exchange with the surroundings, the right solid object initially has temperature {right} degrees Celsius and the left solid object has temperature {left} degrees Celsius. '
                setup+=('A conducting connection allows heat to pass between the two objects. ' if contact else 'The two objects are separated by perfect thermal insulation, so no heat passes between them. ')
            active=contact and left!=right
            answers={'warms':('right object' if left>right else 'left object') if active else 'neither object',
                     'cools':('left object' if left>right else 'right object') if active else 'neither object',
                     'changes':'both objects' if active else 'neither object'}
            queries={'warms':'Which object initially increases in temperature?',
                     'cools':'Which object initially decreases in temperature?',
                     'changes':'Which objects initially change temperature?'}
            for task in TASKS:
                rows.append(dict(id=f'relation_{split}_{pair}_{member}_{task}',question=setup+queries[task],
                                 choices=dict(label=['A','B','C','D'],text=list(OPTIONS)),answerKey=['A','B','C','D'][OPTIONS.index(answers[task])],
                                 group=task,pair=f'{split}_{pair}_{task}',member=member,
                                 physical_state=dict(left=left,right=right,contact=contact)))
    return rows


def fit_bias(records,rows):
    """625 predeclared constant-offset candidates; acquisition likelihood only."""
    rows={r['id']:r for r in rows}
    chosen=[r for r in records if tuple(rows[r['id']]['choices']['text'])==OPTIONS]
    scores=np.asarray([r['normalized_logprobs'] for r in chosen],float);gold=np.asarray([r['gold'] for r in chosen])
    best=None
    for values in product((-2.,-1.,0.,1.,2.),repeat=4):
        b=np.asarray(values);b-=b.mean();z=scores+b
        maximum=z.max(1);loss=float(np.mean(maximum+np.log(np.exp(z-maximum[:,None]).sum(1))-z[np.arange(len(z)),gold]))
        criterion=(loss,float(b@b),values)
        if best is None or criterion<best[0]:best=(criterion,b)
    return dict(offsets=best[1].tolist(),acquisition_loss=best[0][0],candidate_count=625)


def apply_bias(records,rows,bias):
    rows={r['id']:r for r in rows};lookup=dict(zip(OPTIONS,bias['offsets']));result=[]
    for r in records:
        x=dict(r);scores=np.asarray(r['normalized_logprobs'])+np.asarray([lookup.get(t,0.) for t in rows[r['id']]['choices']['text']])
        x['normalized_logprobs']=scores.tolist();x['prediction']=int(scores.argmax());x['correct']=x['prediction']==x['gold'];result.append(x)
    return result


def pair_summary(records,rows):
    mapping={r['id']:r for r in records};pairs={}
    for row in rows:
        if 'pair' in row:pairs.setdefault(row['pair'],[]).append(row)
    if any(len(v)!=2 for v in pairs.values()):raise ValueError('pair coverage')
    return dict(n=len(pairs),both_correct=sum(all(mapping[r['id']]['correct'] for r in pair) for pair in pairs.values()),
                answer_changing_pairs=sum(pair[0]['answerKey']!=pair[1]['answerKey'] for pair in pairs.values()))


def decision(metrics,pairs,records,rows):
    family_n=sum(r['group']!='retention' for r in rows)
    def correct(name):return sum(g['correct'] for k,g in metrics[name]['groups'].items() if k!='retention')
    joint=correct('joint');reference=max(correct('untouched'),correct('answer_bias'))
    improvements=sum(metrics['joint']['groups'][task]['correct']>max(metrics['untouched']['groups'][task]['correct'],metrics['answer_bias']['groups'][task]['correct']) for task in TASKS)
    maps={name:{r['id']:r['correct'] for r in values} for name,values in records.items()}
    unique=[r['id'] for r in rows if r['group']!='retention' and maps['joint'][r['id']] and not maps['wsp_only'][r['id']] and not maps['asp_only'][r['id']] and not maps['untouched'][r['id']]]
    criteria=dict(family_gain_at_least_six=joint-reference>=6,improved_tasks_at_least_two=improvements>=2,
                  counterfactual_pair_gain_at_least_three=pairs['joint']['both_correct']-max(pairs['untouched']['both_correct'],pairs['answer_bias']['both_correct'])>=3,
                  retention_preserved=all(maps['joint'][r['id']] for r in rows if r['group']=='retention' and maps['untouched'][r['id']]),
                  both_components_help=joint>max(correct('wsp_only'),correct('asp_only')),
                  joint_unique_corrections_at_least_two=len(unique)>=2)
    return dict(criteria=criteria,continue_conceptual_asp=all(criteria.values()),family_n=family_n,joint_unique_corrections=unique,
                rule='Descriptive triage, not statistical significance. If unmet, park conceptual ASP under this interface; no automatic next run.')
