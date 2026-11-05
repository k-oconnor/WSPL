"""Rule fixtures and synthetic inference only; no released-policy games."""
import importlib.util
import unittest
import numpy as np
HAS_GO=importlib.util.find_spec('pgx') is not None and importlib.util.find_spec('jax') is not None


@unittest.skipUnless(HAS_GO,'separate pinned WSL Go runtime required')
class GoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as jnp
        from skillforge.go_replay import make_game
        cls.jax=jax;cls.jnp=jnp;cls.game=make_game();cls.step=staticmethod(jax.jit(cls.game.step))
        cls.mask=staticmethod(jax.jit(cls.game.legal_action_mask));cls.reward=staticmethod(jax.jit(cls.game.rewards))

    def fixture(self,black=(),white=(),color=0):
        board=np.zeros((9,9),np.int32)
        for r,c in black:board[r,c]=1
        for r,c in white:board[r,c]=-1
        labels=np.zeros((9,9),np.int32);seen=set()
        for r in range(9):
            for c in range(9):
                if not board[r,c] or (r,c) in seen:continue
                todo=[(r,c)];group=[];seen.add((r,c))
                while todo:
                    a,b=todo.pop();group.append((a,b))
                    for aa,bb in ((a-1,b),(a+1,b),(a,b-1),(a,b+1)):
                        if 0<=aa<9 and 0<=bb<9 and (aa,bb) not in seen and board[aa,bb]==board[r,c]:
                            seen.add((aa,bb));todo.append((aa,bb))
                label=(min(a*9+b for a,b in group)+1)*board[r,c]
                for a,b in group:labels[a,b]=label
        return self.game.init()._replace(board=self.jnp.array(labels.ravel()),board_history=self.jnp.array(board.reshape(1,81)),step_count=self.jnp.int32(color))

    def test_observation_order_color_and_occupied(self):
        from skillforge.go_replay import features
        for color in (0,1):
            s=self.fixture([(1,2)],[(3,4)],color)
            actual=np.asarray(features(s)).reshape(9,9,3)
            np.testing.assert_array_equal(actual,self.game.observe(s))
            self.assertEqual(actual[1,2,0],1-color);self.assertEqual(actual[3,4,0],color)
            self.assertTrue(np.all(actual[:,:,2]==color))
            mask=np.asarray(self.mask(s));self.assertFalse(mask[11]);self.assertFalse(mask[31]);self.assertTrue(mask[81])

    def test_capture_and_suicide(self):
        s=self.fixture([(0,1)],[(0,0)])
        self.assertTrue(bool(self.mask(s)[9]));s=self.step(s,self.jnp.int32(9))
        self.assertEqual(int(s.board[0]),0);self.assertEqual(int(s.num_captured[0]),1)
        s=self.fixture(white=[(3,4),(5,4),(4,3),(4,5)])
        self.assertFalse(bool(self.mask(s)[40]))

    def test_simple_ko(self):
        s=self.fixture([(0,1),(1,0),(2,1)],[(1,1),(0,2),(2,2),(1,3)])
        s=self.step(s,self.jnp.int32(11))
        self.assertEqual(int(s.ko),10);self.assertFalse(bool(self.mask(s)[10]))
        s=self.step(s,self.jnp.int32(81));self.assertEqual(int(s.ko),-1)

    def test_repeated_board_loses_for_mover(self):
        from pgx._src.games.go import _compute_hash
        s=self.fixture(color=1);future=self.step(s,self.jnp.int32(0))
        s=s._replace(hash_history=s.hash_history.at[0].set(_compute_hash(future)))
        s=self.step(s,self.jnp.int32(0));self.assertTrue(bool(s.is_psk));self.assertTrue(bool(self.game.is_terminal(s)))
        np.testing.assert_array_equal(self.reward(s),[1,-1])

    def test_pass_horizon_scoring_komi(self):
        s=self.game.init();s=self.step(s,self.jnp.int32(81));self.assertFalse(bool(self.game.is_terminal(s)))
        s=self.step(s,self.jnp.int32(81));self.assertTrue(bool(self.game.is_terminal(s)))
        np.testing.assert_array_equal(self.reward(s),[-1,1])
        s=self.fixture(black=[(r,c) for r in range(9) for c in range(9)])._replace(step_count=self.jnp.int32(162))
        np.testing.assert_array_equal(self.reward(s),[1,-1]);self.assertTrue(bool(self.game.is_terminal(s)))
        self.assertFalse(bool(self.game.is_terminal(s._replace(step_count=self.jnp.int32(161)))))

    def test_full_shape_inference_numpy_reference(self):
        from skillforge.go_replay import WIDTHS,logits
        rng=np.random.default_rng(44);weights=tuple((rng.normal(0,.02,(a,b)).astype(np.float32),rng.normal(0,.02,b).astype(np.float32)) for a,b in zip(WIDTHS,WIDTHS[1:]))
        x=rng.integers(0,2,(4,243)).astype(np.float32);expected=x.copy()
        for i,(w,b) in enumerate(weights):
            expected=expected@w+b
            if i<6:expected=np.maximum(expected,0)
        actual=self.jax.jit(logits)(weights,x)
        np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-5)

    def test_batched_forced_pass_freezes_terminal_and_maps_colors(self):
        from skillforge.go_replay import WIDTHS,make_batch_runner
        weights=tuple((self.jnp.zeros((a,b)),self.jnp.zeros(b)) for a,b in zip(WIDTHS,WIDTHS[1:]))
        weights=weights[:-1]+((weights[-1][0],weights[-1][1].at[81].set(1000)),)
        run=make_batch_runner();colors=self.jnp.array([0,1]);keys=self.jax.random.split(self.jax.random.PRNGKey(5),2)
        result,trace=run(weights,weights,keys,colors)
        np.testing.assert_array_equal(result['steps'],[2,2]);np.testing.assert_array_equal(result['reward'],[-1,1])
        np.testing.assert_array_equal(trace['actions'][:2],np.full((2,2),81))
        self.assertTrue(np.all(np.asarray(trace['actions'][2:])==-1));self.assertFalse(np.any(result['invalid']))
        again,_=run(weights,weights,keys,colors)
        for k in result:np.testing.assert_array_equal(again[k],result[k])
