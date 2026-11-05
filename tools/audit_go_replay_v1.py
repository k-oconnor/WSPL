"""Audit stored Go episodes and independently score final boards; no new games."""
import json
from pathlib import Path
import sys
import numpy as np

MAIN=Path(__file__).resolve().parents[1]
RUN=MAIN.parent/'Skillforge-go-replay-v1'
sys.path[:0]=[str(RUN/'src'),str(RUN/'experiments')]
from source_consistency import verify
from skillforge.go_replay import sha,POLICIES
from run_go_replay import interval


def black_wins(board):
    board=board.reshape(9,9);black=int(np.count_nonzero(board==1));white=int(np.count_nonzero(board==-1));seen=set()
    for r in range(9):
        for c in range(9):
            if board[r,c] or (r,c) in seen:continue
            stack=[(r,c)];seen.add((r,c));region=0;boundary=set()
            while stack:
                a,b=stack.pop();region+=1
                for x,y in ((a-1,b),(a+1,b),(a,b-1),(a,b+1)):
                    if not (0<=x<9 and 0<=y<9):continue
                    if board[x,y]:boundary.add(int(board[x,y]))
                    elif (x,y) not in seen:seen.add((x,y));stack.append((x,y))
            if boundary=={1}:black+=region
            if boundary=={-1}:white+=region
    return black>white+7.5


def main():
    out=RUN/'results/go_replay_v1'
    receipt=json.loads((out/'source_receipt.json').read_text());r=json.loads((out/'results.json').read_text())
    assert receipt['status']==r['status']=='complete' and receipt['constructor_calls']==1
    m=verify(RUN,out/'approved_manifest.json','a0044b77433bd6262a77e93454be06d0721e32c6eecaa136307471d1d44d600c')
    for p,h in m['runner_config']['resources'].items():
        p=Path(p);p=p if p.is_absolute() else RUN/p
        assert sha(p)==h,p
    ref=Path(m['constructor_kwargs']['reference_root'])
    for name in POLICIES:
        with np.load(out/(name+'_actor.npz'),allow_pickle=False) as z:
            assert len(z.files)==14
            for k in z.files:assert z[k].tobytes()==np.load(ref/name/(k+'.npy'),allow_pickle=False).tobytes()
    records=[json.loads(s) for s in (out/'episodes.jsonl').read_text().splitlines()]
    ledger={(x['actor'],x['opponent'],x['episode']):x for x in records};assert len(ledger)==len(records)==12000
    archives={};scores=0;trace_moves=0;comparisons={}
    published=json.loads((MAIN/'data/csail_relu_reference_v1/exp6_go_from_scratch/eval/ppo_same_size/ppo_big_eval/h2h_lr_scan.json').read_text())
    for opponent in range(3):
        for name in ('budget_0','budget_50'):
            pair=f'{name}_vs_s{opponent}';rows=[]
            for start in range(0,2000,32):
                path=out/pair/f'batch_{start:04d}.npz';archives[str(path.relative_to(out))]=sha(path)
                with np.load(path,allow_pickle=False) as z:
                    ids=np.arange(start,min(start+32,2000));n=len(ids)
                    np.testing.assert_array_equal(z['episode_ids'],ids)
                    np.testing.assert_array_equal(z['actor_colors'],ids%2)
                    np.testing.assert_array_equal(z['seeds'],10300000+opponent*100000+ids)
                    assert z['actions'].shape==(162,n) and z['final_board'].shape==(n,81)
                    assert not np.any(z['invalid']) and np.all(z['terminal'])
                    assert np.isin(z['final_board'],[-1,0,1]).all()
                    for i,episode in enumerate(ids):
                        row=ledger[(name,opponent,int(episode))];rows.append(row);steps=int(z['steps'][i]);color=int(episode%2)
                        actions=z['actions'][:,i];assert 2<=steps<=162 and np.all((actions[:steps]>=0)&(actions[:steps]<=81)) and np.all(actions[steps:]==-1)
                        assert steps==row['steps'] and row['actor_color']==color and row['seed']==int(z['seeds'][i])
                        if z['psk'][i]:win=(steps%2)==color;reason='repetition'
                        else:
                            win=black_wins(z['final_board'][i])==(color==0)
                            reason='passes' if z['passes'][i]>=2 else 'horizon'
                            if reason=='passes':assert np.all(actions[steps-2:steps]==81)
                            else:assert steps==162
                        assert row['win']==win==(z['reward'][i]>0) and row['reason']==reason;scores+=1
                        if i<2:
                            boards=z['first_two_boards'][:,i]
                            assert not np.any(boards[0])
                            for t,a in enumerate(actions[:steps]):
                                if a<81:assert boards[t,a]==0
                                trace_moves+=1
            summary=r['summaries'][pair];wins=sum(x['win'] for x in rows)
            assert summary['games']==len(rows)==2000 and summary['wins']==wins
            assert summary['win_rate']==wins/2000 and summary['ci95_wilson']==interval(wins,2000)
            for color in (0,1):
                color_rows=[x for x in rows if x['actor_color']==color]
                assert len(color_rows)==1000 and summary['colors'][str(color)]==dict(games=1000,wins=sum(x['win'] for x in color_rows))
            for reason in ('passes','repetition','horizon'):assert summary['reasons'][reason]==sum(x['reason']==reason for x in rows)
            key=('v1_b0' if name=='budget_0' else 'v1_b50')+'_vs_lr3e-5'+('' if opponent==0 else '_s'+str(opponent))
            original=published[key]['score'];comparisons[pair]=dict(local=wins/2000,released=original,difference_pp=100*(wins/2000-original),ci95=summary['ci95_wilson'])
    audit=dict(status='passed',sources=len(m['files']),resources=len(m['runner_config']['resources']),episodes=scores,batches=len(archives),selected_trajectory_moves=trace_moves,
        independent_final_score_check=True,comparisons=comparisons,active_seconds=r['active_seconds'],finished_utc=receipt['finished_utc'],archive_hashes=archives,
        scope='Read-only arrays, actors, source/runtime, accounting and independent final-board scoring. No re-sampling or policy execution; not full trajectory rule replay.')
    (MAIN/'docs/wspl/go_replay_v1_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='archive_hashes'},indent=2))


if __name__=='__main__':main()
