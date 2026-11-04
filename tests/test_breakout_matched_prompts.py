import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('matched_prepare',ROOT/'tools/prepare_breakout_matched.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class PromptTests(unittest.TestCase):
    def test_only_selected_arm_differs_no_old_target_or_solution(self):
        native='old task edits\nTHE GAME: supplied rules\nTHE INPUT: raw pixels\nTHE NETWORK (old): tanh\nHOW TO WRITE: old interface'
        guide='all three interfaces and fixed ReLU target'
        p=module.prompts(native,guide)
        prefixes=[text.split('\n\nSELECTED ARM: ')[0] for text in p.values()]
        self.assertEqual(len(set(prefixes)),1)
        for arm,text in p.items():
            self.assertTrue(text.endswith('SELECTED ARM: '+arm+'\n'))
            self.assertIn('supplied rules',text);self.assertIn(guide,text)
            self.assertNotIn('old task',text);self.assertNotIn('old interface',text);self.assertNotIn('tanh',text)
