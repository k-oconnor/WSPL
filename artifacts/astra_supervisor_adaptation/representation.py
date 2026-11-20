import json, copy

CODES=['A','B','0','1','2','3','4','P','S']

def aliases(public):
    o=public['observation']
    return {m:chr(65+i) for i,m in enumerate(o['available_modules'])}, {g:str(i) for i,g in enumerate(o['requested_order'])}

def encode_action(public, action):
    ms,gs=aliases(public)
    if action['op']=='select':return ms[action['module']]
    if action['op']=='reach':return gs[action['goal']]
    return {'probe':'P','stop':'S'}[action['op']]

def decode_action(public, code):
    ms,gs=aliases(public)
    if code=='P':return {'op':'probe','command':[0.12,0,0],'ticks':6}
    if code=='S':return {'op':'stop'}
    if code in ms.values():return {'op':'select','module':next(m for m,a in ms.items() if a==code)}
    if code in gs.values():return {'op':'reach','goal':next(g for g,a in gs.items() if a==code)}
    raise ValueError('Invalid action code: '+str(code))

def render(public, arithmetic=False):
    o=public['observation'];t=o.get('telemetry',{});ms,gs=aliases(public)
    def mm(v):return [round(float(x)*1000,2) for x in v]
    state={'selected':ms.get(o.get('selected_module')), 'order':list(gs.values()),
        'visited':[gs.get(g,g) for g in o.get('observed_visits',[])],
        'position_mm':mm(t.get('nominal_tool_position_m',[0,0,0])),
        'goals_mm':{gs[g]:mm(v) for g,v in o['goals'].items()},
        'decisions_left':o.get('remaining_decisions'), 'ticks_left':t.get('remaining_steps')}
    last=copy.deepcopy(o.get('last'))
    if last:
        if 'module' in last:last['module']=ms.get(last['module'],last['module'])
        if 'goal' in last:last['goal']=gs.get(last['goal'],last['goal'])
        for key in list(last):
            if key=='displacement_m':last['displacement_mm']=mm(last.pop(key))
            elif key.endswith('_error_m'):last[key[:-2]+'_mm']=round(last.pop(key)*1000,2)
        if arithmetic and 'initial_measured_error_mm' in last:
            initial=last['initial_measured_error_mm'];end=last['measured_error_mm']
            last['error_reduction_mm']=round(initial-end,2)
            last['error_ratio']=round(end/max(initial,0.001),3)
    state['last']=last
    return ('Robot executive. Choose exactly one action code. Complete the ordered waypoints, then stop. '
        'A or B selects that documented frozen motor. 0,1,2,3,4 reaches that waypoint for up to 10 ticks. '
        'P probes raw command [+0.12,0,0] for 6 ticks without selecting a motor. S stops. '
        'Only use listed module and waypoint codes. Calibration can change. Use observed feedback, not inventory position. '
        'Coordinates and errors below are millimeters. A probe reports actual displacement, not a diagnosis.\n'
        +'Modules:\n'+'\n'.join(ms[m]+': '+public['documentation'][m] for m in ms)
        +'\nState:\n'+json.dumps(state,separators=(',',':'))+'\nAction code:')
