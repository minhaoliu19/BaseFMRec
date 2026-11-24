"""Run with: .venv/Scripts/python.exe -m unittest discover -s tests -v"""
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from interest_lifecycle import InteractionLifecycle
from utils import TrainDataset, ValDataset, TestDataset, lifecycle_collate
from model import Att_FM_model, create_model_FM


def config(enabled=True):
    return SimpleNamespace(hidden_size=8, item_num=32, max_len=3,
        emb_dropout=0., dropout=0., mask_ratio=1., interest_lifecycle=enabled,
        num_interests=2, interest_promote=[2.,3.,4.], interest_degrade_gap=3.,
        interest_repromote=3., interest_temperature=.5, num_blocks=2, last=1,
        lambda_uncertainty=.001, eps=.001, sample_N=2, eps_reverse=.001,
        m_logNorm=1., s_logNorm=.6, s_modsamp=1., sampling_method='mode')


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1997)
        self.module=InteractionLifecycle(8,2,(2.,3.,4.),3.,3.)

    def stats(self, trace, padding=0):
        assignments=F.one_hot(torch.tensor([trace]),2).float()
        valid=torch.ones(1,len(trace),dtype=torch.bool)
        if padding:
            assignments=F.pad(assignments,(0,0,0,padding))
            valid=F.pad(valid,(0,padding))
        return self.module.state_from_assignments(assignments,valid)[1]

    def test_five_levels(self):
        for n in range(1,5):
            self.assertEqual(self.stats([0]*n)['states'].tolist(),[[n,0]])

    def test_demote_floor_and_reactivation(self):
        trace=[0]*4+[1]*3
        self.assertEqual(self.stats(trace)['states'][0,0],3)
        for n in (1,2):
            self.assertEqual(self.stats(trace+[0]*n)['states'][0,0],3)
        self.assertEqual(self.stats(trace+[0]*3)['states'][0,0],4)
        self.assertEqual(self.stats(trace+[1]*100)['states'][0,0],3)

    def test_second_gap_resets_recovery(self):
        trace=[0]*4+[1]*3+[0]*2+[1]*3+[0]
        result=self.stats(trace)
        self.assertEqual(result['states'][0,0],3)
        self.assertEqual(result['recovery_count'][0,0],1)
        self.assertEqual(self.stats(trace+[0]*2)['states'][0,0],4)

    def test_padding_not_counted(self):
        for trace in ([0]*4,[0]*4+[1]*3,[0]*4+[1]*3+[0]*2):
            a,b=self.stats(trace),self.stats(trace,10)
            for key in ('states','counts','gaps','recovery_count'):
                torch.testing.assert_close(a[key],b[key])

    def test_vectorized_replay_matches_reference(self):
        rng=random.Random(42)
        for _ in range(50):
            trace=[rng.randrange(2) for _ in range(60)]
            count,last,mature,active,recovery=[0]*2,[-1]*2,[False]*2,[False]*2,[0]*2
            for position,interest in enumerate(trace):
                for k in range(2):
                    if mature[k] and position-last[k]-1>=3:
                        active[k]=False
                if mature[interest] and position-last[interest]-1>=3:
                    recovery[interest]=0
                count[interest]+=1
                if mature[interest] and not active[interest]:
                    recovery[interest]+=1
                    active[interest]=recovery[interest]>=3
                if not mature[interest] and count[interest]>=4:
                    mature[interest]=active[interest]=True
                last[interest]=position
            expected=[(4 if active[k] and len(trace)-1-last[k]<3 else 3)
                      if mature[k] else min(count[k],3) for k in range(2)]
            self.assertEqual(self.stats(trace)['states'].tolist(),[expected])

    def test_full_prefix_excludes_labels(self):
        sample=TrainDataset({0:[1,2,3,4,5,6]},2,True)[0]
        self.assertEqual(sample[0].tolist(),[4,5])
        self.assertEqual(sample[2].tolist(),[1,2,3,4,5])
        val=ValDataset({1:[1,2,3]},{1:[4]},2,True)[0]
        test=TestDataset({1:[1,2,3]},{1:[4]},{1:[5]},2,True)[0]
        self.assertEqual(val[2].tolist(),[1,2,3])
        self.assertEqual(test[2].tolist(),[1,2,3,4])
        self.assertEqual(len(TrainDataset({0:[1,2]},2)[0]),2)

    def controlled_embedding(self):
        embedding=torch.nn.Embedding(10,8)
        with torch.no_grad():
            embedding.weight[1]=self.module.prototypes[0]
            embedding.weight[2]=self.module.prototypes[1]
        return embedding

    def test_memory_outside_recent_window(self):
        embedding=self.controlled_embedding()
        ids=torch.tensor([[2,2,2]])
        self.module(embedding(ids),ids,torch.tensor([[1,1,1,1,2,2,2]]),embedding)
        self.assertEqual(self.module.last_stats['states'][0,0],3)
        self.assertTrue(self.module.last_stats['ever_mature'][0,0])

    def test_backward_thresholds_and_prototypes(self):
        embedding=self.controlled_embedding()
        history=torch.tensor([[1,1,1,1,2,2,2,1,1]])
        seq=history[:,-3:]
        result=self.module(embedding(seq),seq,history,embedding)
        result.square().mean().backward()
        for name,p in self.module.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(float(p.grad.abs().sum()),0,name)

    def test_batch_order_invariance(self):
        embedding=torch.nn.Embedding(20,8)
        seq=torch.tensor([[1,2,3],[6,7,8]])
        histories=torch.nn.utils.rnn.pad_sequence(
            [torch.tensor([1,2,3]),torch.tensor([4,5,6,7,8])],batch_first=True)
        output=self.module(embedding(seq),seq,histories,embedding)
        reverse=self.module(embedding(seq.flip(0)),seq.flip(0),histories.flip(0),embedding)
        torch.testing.assert_close(output,reverse.flip(0))

    def test_fmrec_train_inference_and_baseline(self):
        for enabled in (False,True):
            args=config(enabled)
            model=Att_FM_model(create_model_FM(args),args)
            examples=[TrainDataset({0:[1,2,3,4,5]},3,enabled)[0],
                      TrainDataset({0:[2,3,4]},3,enabled)[0]]
            batch=lifecycle_collate(examples) if enabled else (
                torch.stack([x[0] for x in examples]),torch.stack([x[1] for x in examples]))
            history=batch[2] if enabled else None
            reconstruction,rep,_,_,fm_loss,_=model(batch[0],batch[1],history=history)
            loss=reconstruction+fm_loss+.2*model.loss_fm_ce(rep,batch[1])
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            model.eval()
            with torch.no_grad():
                out=model(batch[0],batch[1],train_flag=False,history=history)[1]
            self.assertEqual(out.shape,(2,8))
            self.assertTrue(torch.isfinite(out).all())


if __name__=='__main__':
    unittest.main()
