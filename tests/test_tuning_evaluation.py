import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tune_ml100k import evaluate
from trainer import metric_percent


class FixedRanking(torch.nn.Module):
    def forward(self, sequence, label, forward_mse_time=0, train_flag=False, history=None):
        torch.rand(1)  # Exercise evaluation RNG isolation.
        return None, sequence.float(), None, None, None, None

    def fm_rep_pre(self, representation):
        return representation.clone()


class TuningEvaluationTests(unittest.TestCase):
    def test_tail_batch_is_weighted_and_rng_is_restored(self):
        self.assertEqual(metric_percent([(1.,3),(0.,1)]),75.)
        scores=torch.arange(24,0,-1).float().repeat(4,1)
        labels=torch.tensor([[1],[1],[1],[23]])
        dataset=TensorDataset(scores,labels)
        args=SimpleNamespace(device='cpu',random_seed=1997,metric_ks=[5,10,20])
        model=FixedRanking().train()
        before=torch.get_rng_state().clone()
        metrics=evaluate(model,DataLoader(dataset,batch_size=3),args)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(model.training)
        for value in metrics.values():
            self.assertAlmostEqual(value,75.,places=4)
        other=evaluate(model,DataLoader(dataset,batch_size=2),args)
        for key in metrics:
            self.assertAlmostEqual(metrics[key],other[key],places=4)


if __name__=='__main__':
    unittest.main()
