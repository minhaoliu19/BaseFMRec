import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_interest_lifecycle import config
from compare_paper_ml100k import build_model,shared_hash


class PaperComparisonTests(unittest.TestCase):
    def test_same_shared_initialization_and_training_rng(self):
        args=config(False)
        args.device='cpu'
        args.random_seed=1997
        baseline=build_model(args)
        baseline_rng=torch.get_rng_state().clone()
        args.interest_lifecycle=1
        lifecycle=build_model(args)
        self.assertEqual(shared_hash(baseline),shared_hash(lifecycle))
        self.assertTrue(torch.equal(baseline_rng,torch.get_rng_state()))
        self.assertIsNone(baseline.interest_lifecycle)
        self.assertIsNotNone(lifecycle.interest_lifecycle)


if __name__=='__main__':
    unittest.main()
