import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'experiments')]
from skillforge.authoring_sft import LoRALinear
from skillforge.fresh_robot import (panel, validate_panels, prompt_record, summarize,
                                   token_loss, motor_artifact)
from skillforge.wspl_artifact import ReLUArtifact
from skillforge.factory_pilot import values
import run_fresh_robot_parent as runner


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return 'xxx'

    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(c) % 128 for c in text]}

    def decode(self, tokens, **kwargs):
        return ''.join(chr(t) for t in tokens)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = LoRALinear(torch.nn.Linear(4, 128), rank=2, alpha=4)

    def gradient_checkpointing_disable(self):
        pass

    def forward(self, input_ids, **kwargs):
        assert kwargs['logits_to_keep'] == 1 and kwargs['use_cache'] is False
        x = torch.nn.functional.one_hot(input_ids[:, -1:] % 4, 4).float()
        return types.SimpleNamespace(logits=self.layer(x))


class FakeSimulator:
    closed = False
    failure = False

    def __init__(self, *args):
        self.calls = 0

    def call(self, r):
        if r['op'] == 'start':
            self.calls = 0
            return {'finished': False, 'seed': r['seed']}
        if r['op'] == 'act':
            self.calls += 1
            return {'finished': True}
        if r['op'] == 'finish':
            return {'success': not self.failure, 'decisions': self.calls}
        raise AssertionError(r)

    def close(self):
        self.closed = True
        FakeSimulator.closed = True


class FreshRobotTests(unittest.TestCase):
    def test_panel_disjointness(self):
        validate_panels(panel(10, 3), panel(20, 3))
        with self.assertRaises(ValueError):
            validate_panels(panel(10, 3), panel(12, 3))

    def test_compiled_motors_inverse(self):
        import numpy as np
        a = motor_artifact()
        x = [.001, -.002, .003]
        ordinary = values(ReLUArtifact(a['original']), x)
        repair = values(ReLUArtifact(a['repair']), x)
        mapping = np.array([[0, -.8, 0], [1.2, 0, 0], [0, 0, .9]])
        np.testing.assert_allclose(mapping @ repair, ordinary, rtol=1e-6, atol=1e-9)

    def test_single_target_and_no_truncation(self):
        self.assertEqual(prompt_record(Tokenizer(), 'x', 'A')['target'], 65)
        with self.assertRaises(ValueError):
            prompt_record(Tokenizer(), 'x', 'AB')
        with self.assertRaises(ValueError):
            prompt_record(Tokenizer(), 'x', 'A', max_tokens=2)

    def test_paired_summary_rejects_missing_duplicate_and_mismatched_states(self):
        rows = [dict(arm=a, mission=i, start_sha256=str(i), success=(a == 'ordinary'))
                for a in ('base', 'ordinary') for i in range(2)]
        self.assertTrue(summarize(rows, 2)['parent_ready'])
        self.assertEqual(summarize(rows, 2)['corrections'], 2)
        for bad in [rows[:-1], rows+[rows[0]], rows[:-1]+[dict(rows[-1], start_sha256='bad')]]:
            with self.assertRaises(ValueError):
                summarize(bad)

    def test_loss_only_backpropagates_to_factors(self):
        model = TinyModel()
        token_loss(model, dict(ids=[1, 2], target=3)).backward()
        self.assertIsNone(model.layer.base.weight.grad)
        self.assertGreater(float(model.layer.lora_B.grad.abs().sum()), 0)

    def run_fake(self, out):
        (out/'source_receipt.json').write_text(json.dumps({'manifest_sha256': 'fixture'}))
        model = TinyModel()
        config = dict(wall_guard_seconds=30, resources={}, train_seed=100, train_missions=2,
                      reserved_seed=200, reserved_missions=2, motor_artifact='unused',
                      learning_rate=.001, schedule_seed=5, batch_size=2, updates=2, minimum_success=2)
        def construct():
            return types.SimpleNamespace(model=model, tokenizer=Tokenizer())
        with patch.object(runner, 'Simulator', FakeSimulator), patch.object(runner, 'render', return_value='x'), \
                patch.object(runner, 'encode_action', return_value='A'), \
                patch.object(runner, 'decode_action', return_value={'op': 'stop'}), \
                patch.object(runner, 'ordinary_action', return_value={'op': 'stop'}):
            runner.run(construct, out, config)

    def test_full_fake_runner_freeze_provenance_and_cleanup(self):
        FakeSimulator.failure = FakeSimulator.closed = False
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            self.run_fake(out)
            result = json.loads((out/'results.json').read_text())
            self.assertEqual(result['author_calls'], 0)
            self.assertEqual(result['updates'], 2)
            self.assertEqual(result['summary']['missions_per_arm'], 2)
            self.assertFalse(result['follow_on_launched'])
            saved = torch.load(out/'adapter_final.pt', weights_only=True)
            self.assertEqual(saved['manifest_sha256'], 'fixture')
            self.assertEqual(len(json.loads((out/'episodes.json').read_text())), 4)
            self.assertTrue(FakeSimulator.closed)

    def test_teacher_failure_prevents_training(self):
        FakeSimulator.failure = True
        try:
            with tempfile.TemporaryDirectory() as folder:
                out = Path(folder)
                with self.assertRaisesRegex(ValueError, 'teacher failed'):
                    self.run_fake(out)
                self.assertFalse((out/'adapter_final.pt').exists())
                self.assertFalse((out/'training.jsonl').exists())
                self.assertTrue(FakeSimulator.closed)
        finally:
            FakeSimulator.failure = False


if __name__ == '__main__':
    unittest.main()
