"""Pinned PGX Go rules and batched FP32 actor replay; no implicit games."""
import hashlib
import json
from pathlib import Path
import numpy as np

POLICIES=('budget_0','budget_50','ppo_same_size_lr3e-5_s0','ppo_same_size_lr3e-5_s1','ppo_same_size_lr3e-5_s2')
WIDTHS=(243,512,512,512,512,512,512,82)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_actor(folder):
    folder=Path(folder)
    if json.loads((folder/'arch.json').read_text())!={'depth':6,'width':512}:raise ValueError('architecture differs')
    arrays=[]
    for i,(a,b) in enumerate(zip(WIDTHS,WIDTHS[1:]),1):
        name='L'+str(i) if i<7 else 'out'
        pair=[]
        for suffix,shape in (('W',(a,b)),('b',(b,))):
            x=np.load(folder/(name+'.'+suffix+'.npy'),allow_pickle=False)
            if x.shape!=shape or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('invalid actor tensor')
            pair.append(x)
        arrays.append(tuple(pair))
    return tuple(arrays)


def features(state):
    import jax.numpy as jnp
    sign=jnp.where(state.color==0,1,-1)
    return jnp.stack((state.board*sign>0,state.board*sign<0,jnp.full_like(state.board,state.color,dtype=jnp.bool_)),axis=-1).reshape(243).astype(jnp.float32)


def logits(weights,x):
    import jax.numpy as jnp
    for i,(w,b) in enumerate(weights):
        x=jnp.matmul(x,w,precision='highest')+b
        if i<6:x=jnp.maximum(x,0)
    return x


def make_game():
    from pgx._src.games.go import Game
    return Game(size=9,komi=7.5,history_length=1,max_termination_steps=162)


def make_batch_runner():
    import jax
    import jax.numpy as jnp
    game=make_game()
    def batch(actor,opponent,keys,actor_colors):
        states=jax.vmap(lambda _:game.init())(actor_colors)
        def tick(carry,t):
            states,invalid=carry
            active=~jax.vmap(game.is_terminal)(states)
            obs=jax.vmap(features)(states)
            a=logits(actor,obs);b=logits(opponent,obs)
            chosen=jnp.where((states.color==actor_colors)[:,None],a,b)
            legal=jax.vmap(game.legal_action_mask)(states)
            finite=jnp.isfinite(chosen).all(axis=1)
            safe=jnp.where(jnp.isfinite(chosen),chosen,0.)
            masked=jnp.where(legal,safe,-jnp.inf)
            step_keys=jax.vmap(lambda k:jax.random.fold_in(k,t))(keys)
            actions=jax.vmap(jax.random.categorical)(step_keys,masked).astype(jnp.int32)
            invalid=invalid | (active & (~finite | ~jnp.take_along_axis(legal,actions[:,None],axis=1)[:,0]))
            # Scalar conditional freezes finished games; subsequent scan ticks
            # never advance their move count or overwrite terminal rewards.
            next_states=jax.vmap(lambda s,a,on:jax.lax.cond(on,lambda:game.step(s,a),lambda:s))(states,actions,active)
            trace=dict(actions=jnp.where(active,actions,-1),first_two_boards=jnp.sign(states.board[:2]).astype(jnp.int8))
            return (next_states,invalid),trace
        (states,invalid),trace=jax.lax.scan(tick,(states,jnp.zeros_like(actor_colors,dtype=jnp.bool_)),jnp.arange(162))
        rewards=jax.vmap(game.rewards)(states)
        reward=jnp.take_along_axis(rewards,actor_colors[:,None],axis=1)[:,0]
        result=dict(reward=reward,steps=states.step_count,psk=states.is_psk,passes=states.consecutive_pass_count,
            terminal=jax.vmap(game.is_terminal)(states),invalid=invalid,final_board=jnp.sign(states.board).astype(jnp.int8))
        return result,trace
    return jax.jit(batch)


class GoReplayStudy:
    def __init__(self,reference_root,hashes):
        root=Path(reference_root)
        for rel,digest in hashes.items():
            if sha(root/rel)!=digest:raise ValueError('actor resource differs: '+rel)
        self.actors={name:load_actor(root/name) for name in POLICIES}
