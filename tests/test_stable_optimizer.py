import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from stable_optimizer import CPUAdam


class StableOptimizerTests(unittest.TestCase):
    def test_matches_clipped_cpu_adam_and_resume(self):
        a=torch.nn.Parameter(torch.tensor([1.,2.,3.]))
        b=torch.nn.Parameter(a.detach().clone())
        optimizer=CPUAdam([a],lr=.001,clip_value=.5)
        reference=torch.optim.Adam([b],lr=.001,foreach=False)
        for step in range(4):
            gradient=torch.tensor([2.,-.2,.1])*(step+1)
            a.grad=gradient.clone()
            b.grad=gradient.clamp(-.5,.5)
            optimizer.step()
            reference.step()
            torch.testing.assert_close(a,b)
        c=torch.nn.Parameter(a.detach().clone())
        resumed=CPUAdam([c],lr=.001,clip_value=.5)
        resumed.load_state_dict(optimizer.state_dict())
        c.grad=torch.tensor([.1,.1,.1])
        b.grad=c.grad.clone()
        resumed.step()
        reference.step()
        torch.testing.assert_close(c,b)

    def test_rejects_nonfinite_gradient_without_weight_update(self):
        parameter=torch.nn.Parameter(torch.ones(3))
        optimizer=CPUAdam([parameter])
        parameter.grad=torch.tensor([1.,float('nan'),2.])
        with self.assertRaises(FloatingPointError):
            optimizer.step()
        torch.testing.assert_close(parameter,torch.ones(3))


if __name__=='__main__':
    unittest.main()
