"""Validation-only parameter search; evaluate test once after selection.

All epochs consume every training example. Wall time is checked between epochs.
The FMRec objective and flow implementation are unchanged.

中文说明：搜索仅依据验证集 NDCG@20，选定候选后才使用测试集。
预算在整轮边界判断，可能为完成最后一轮而略超预算；结果只代表已执行搜索范围。
"""
import os
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')
import argparse
import copy
import json
import pickle
import random
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from model import Att_FM_model, create_model_FM
from trainer import forward_batch, hrs_and_ndcgs_k
from utils import Data_Train, Data_Val, Data_Test
from stable_optimizer import CPUAdam


ROOT = Path(__file__).resolve().parents[1]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    # 即便检查点被加载到 GPU，PyTorch 的随机数状态仍须以 CPU ByteTensor 恢复。
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state[3]])


def evaluate(model, loader, args):
    """User-weighted metrics, fixed validation noise, no change to training RNG."""
    saved_rng = rng_state()
    # 验证噪声固定，结束后恢复 RNG，避免验证频率改变后续训练随机序列。
    was_training = model.training
    try:
        seed_all(args.random_seed + 10000)
        model.eval()
        sums = {f'{name}@{k}': 0. for name in ('HR','NDCG') for k in args.metric_ks}
        users = 0
        with torch.no_grad():
            for batch in loader:
                batch = [x.to(args.device) for x in batch]
                rep = forward_batch(model, batch)[1]
                scores = model.fm_rep_pre(rep)
                # ID 0 is padding, not a candidate item in ML-100K.
                scores[:,0] = -torch.inf
                metrics = hrs_and_ndcgs_k(scores, batch[1], args.metric_ks)
                count = batch[1].size(0)
                users += count
                for key,value in metrics.items():
                    sums[key] += value * count
        return {key: value/users*100 for key,value in sums.items()}
    finally:
        model.train(was_training)
        restore_rng(saved_rng)


def make_args(candidate, cli, data):
    values = dict(dataset='ml-100k', log_file=str(cli.output)+'/', random_seed=1997,
        max_len=50, device=cli.device, num_gpu=1, batch_size=256,
        hidden_size=32, dropout=.1, emb_dropout=.3, hidden_act='gelu',
        num_blocks=4, epochs=cli.max_epochs, decay_step=15, gamma=.5,
        metric_ks=[5,10,20], optimizer='Adam', lr=.001, grad_clip_value=1.,
        weight_decay=0., momentum=None, lambda_uncertainty=.001,
        eval_interval=1, patience=6, eps=.001, sample_N=30, lambda_t=.001,
        dropout_c=.1, eps_reverse=.001, m_logNorm=1., s_logNorm=.6,
        s_modsamp=1., last=2, mask_ratio=1., sampling_method='mode',
        Loss_Alpha=.2, Loss_Beta=.4, interest_lifecycle=1, num_interests=8,
        interest_promote=[2.,4.,8.], interest_degrade_gap=20.,
        interest_repromote=3., interest_temperature=.5,
        item_num=len(data['smap'])+1, user_num=len(data['train'])+1,optimizer_device='cpu')
    values.update(candidate)
    return SimpleNamespace(**values)


def train_epoch(model, loader, optimizer, args):
    """完整遍历一个训练 epoch；仅覆盖全部样本后才返回可计入的轮次。"""
    model.train()
    total, seen = 0., 0
    repaired_elements = 0
    for batch_index, batch in enumerate(loader):
        batch = [x.to(args.device) for x in batch]
        optimizer.zero_grad(set_to_none=True)
        before_forward_rng = rng_state()
        reconstruction, rep, _, _, fm_loss, _ = forward_batch(model,batch,True)
        loss = fm_loss + args.Loss_Alpha*model.loss_fm_ce(rep,batch[1]) + args.Loss_Beta*reconstruction
        if not torch.isfinite(loss):
            diagnostic = {'batch_index':batch_index,
                'loss_components':{'fm':float(fm_loss.detach()),
                    'ce':float(model.loss_fm_ce(rep,batch[1]).detach()),
                    'reconstruction':float(reconstruction.detach())},
                'nonfinite_parameters':[name for name,p in model.named_parameters()
                                        if not torch.isfinite(p).all()]}
            torch.save(dict(model_state_dict=model.state_dict(),args=vars(args),batch=batch,
                            before_forward_rng=before_forward_rng,
                            diagnostic=diagnostic),Path(args.log_file)/'debug_nonfinite.pt')
            print('NONFINITE',json.dumps(diagnostic),flush=True)
            raise FloatingPointError('Non-finite forward loss; diagnostic saved, no batch skipped')
        loss.backward()
        # Retain the existing local trainer's gradient safety policy and report it.
        if not isinstance(optimizer,CPUAdam):
            repaired_in_batch = torch.zeros((),device=args.device,dtype=torch.long)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    gradient = parameter.grad
                    repaired_in_batch += (~torch.isfinite(gradient)).sum()
                    gradient.nan_to_num_(nan=0.,posinf=args.grad_clip_value,neginf=-args.grad_clip_value)
                    gradient.clamp_(-args.grad_clip_value,args.grad_clip_value)
            repaired_elements += int(repaired_in_batch.item())
        optimizer.step()
        finite_weights=torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()
        if not finite_weights:
            diagnostic={'batch_index':batch_index,'bad_parameters':
                [name for name,p in model.named_parameters() if not torch.isfinite(p).all()]}
            torch.save(dict(model_state_dict=model.state_dict(),args=vars(args),batch=batch,
                            optimizer=optimizer.state_dict(),diagnostic=diagnostic),
                       Path(args.log_file)/'debug_optimizer.pt')
            raise FloatingPointError('Non-finite weights immediately after optimizer.step: '+str(diagnostic))
        count = batch[1].size(0)
        total += float(loss.detach().item())*count
        seen += count
    if seen != len(loader.dataset):
        raise RuntimeError('An epoch did not cover all training examples')
    return total/seen, seen, repaired_elements


def write_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--max_minutes',type=float,default=30.)
    parser.add_argument('--screen_epochs',type=int,default=6)
    parser.add_argument('--max_epochs',type=int,default=60)
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--resume_checkpoint',type=Path,default=None)
    parser.add_argument('--candidates_json',type=Path,default=None)
    parser.add_argument('--threads',type=int,default=8)
    cli=parser.parse_args()
    if cli.output is None:
        cli.output=ROOT/'src'/'log'/'ml100k_tuning'/datetime.now().strftime('%Y%m%d_%H%M%S')
    cli.output.mkdir(parents=True,exist_ok=True)
    with (ROOT/'datasets'/'data'/'ml-100k'/'dataset.pkl').open('rb') as f:
        data=pickle.load(f)
    torch.set_num_threads(cli.threads)
    candidates=[{'hidden_size':32,'lr':.001},
                {'hidden_size':64,'lr':.001},
                {'hidden_size':32,'lr':.0005}]
    if cli.candidates_json:
        candidates=json.loads(cli.candidates_json.read_text(encoding='utf-8'))
    results=[]
    start=time.monotonic()
    deadline=start+cli.max_minutes*60
    print('OUTPUT',str(cli.output),flush=True)
    print('Search: validation NDCG@20 only; test is held out until final selection.',flush=True)

    def run(candidate, index, epochs, resume=None):
        seed_all(1997)
        args=make_args(candidate,cli,data)
        loader=Data_Train(data['train'],args).get_pytorch_dataloaders()
        val_loader=Data_Val(data['train'],data['val'],args).get_pytorch_dataloaders()
        model=Att_FM_model(create_model_FM(args),args).to(args.device)
        optimizer=CPUAdam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay,
                          clip_value=args.grad_clip_value)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,args.decay_step,args.gamma)
        checkpoint=cli.output/f'trial_{index}_best.pt'
        best=-float('inf')
        history=[]
        first=0
        bad=0
        stopped_early=False
        if resume:
            saved=torch.load(resume,map_location=args.device)
            torch.save(saved,checkpoint)
            model.load_state_dict(saved['model_state_dict'])
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            first=saved['epoch']+1
            best=saved['validation']['NDCG@20']
            restore_rng(saved['rng'])
            history=list(saved['history'])
        for epoch in range(first,epochs):
            if time.monotonic()>=deadline and (history or results):
                break
            epoch_start=time.monotonic()
            loss,count,repairs=train_epoch(model,loader,optimizer,args)
            scheduler.step()
            metrics=evaluate(model,val_loader,args)
            row=dict(epoch=epoch+1,loss=loss,validation=metrics,train_examples=count,
                     nonfinite_gradient_elements_replaced=repairs,
                     seconds=time.monotonic()-epoch_start)
            history.append(row)
            score=metrics['NDCG@20']
            print(f'Trial {index} h={args.hidden_size} lr={args.lr} epoch={epoch+1} '
                  f'loss={loss:.4f} val_HR20={metrics["HR@20"]:.4f}% '
                  f'val_NDCG20={score:.4f}% seconds={row["seconds"]:.1f} repairs={repairs}',flush=True)
            if score>best:
                best=score
                bad=0
                torch.save(dict(model_state_dict=model.state_dict(),args=vars(args),epoch=epoch,
                    validation=metrics,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    rng=rng_state(),history=history),checkpoint)
            else:
                bad+=1
            write_json(cli.output/f'trial_{index}_history.json',history)
            if bad>=args.patience:
                stopped_early=True
                print(f'Trial {index}: early stop after {bad} validation checks without improvement.',flush=True)
                break
        del model,optimizer,scheduler,loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not checkpoint.exists():
            return None
        saved=torch.load(checkpoint,map_location='cpu')
        return dict(trial=index,candidate=candidate,checkpoint=str(checkpoint),
                    best_epoch=saved['epoch']+1,validation=saved['validation'],
                    epochs_run=len(history),stop_reason='early_stop' if stopped_early else
                    ('epoch_limit' if len(history)>=epochs else 'time_budget'))

    for index,candidate in enumerate(candidates,1):
        if time.monotonic()>=deadline and results:
            break
        result=run(candidate,index,cli.screen_epochs,
                   cli.resume_checkpoint if index==1 else None)
        if result:
            results.append(result)
            write_json(cli.output/'search_progress.json',results)
    winner=max(results,key=lambda x:x['validation']['NDCG@20'])
    # 验证集上的最佳配置不等于全局最优；不同候选实际轮数会写入报告。
    if (time.monotonic()<deadline and winner.get('stop_reason')=='epoch_limit'
            and winner['epochs_run']<cli.max_epochs):
        continuation=run(winner['candidate'],winner['trial'],cli.max_epochs,winner['checkpoint'])
        if continuation:
            results=[continuation if x['trial']==winner['trial'] else x for x in results]
    winner=max(results,key=lambda x:x['validation']['NDCG@20'])
    saved=torch.load(winner['checkpoint'],map_location=cli.device)
    args=SimpleNamespace(**saved['args'])
    model=Att_FM_model(create_model_FM(args),args).to(cli.device)
    model.load_state_dict(saved['model_state_dict'])
    test_loader=Data_Test(data['train'],data['val'],data['test'],args).get_pytorch_dataloaders()
    test_metrics=evaluate(model,test_loader,args)
    # Portable checkpoint compatible with inspect_interest.py, without optimizer/RNG.
    torch.save({'model_state_dict':model.state_dict(),'args':vars(args),
                'epoch':saved['epoch'],'validation':saved['validation']},cli.output/'selected_model.pt')
    report=dict(selection='validation NDCG@20',metric_units='percent',
                metric_aggregation='user weighted; padding excluded; fixed evaluation RNG',
                trials=results,winner=winner,test_metrics=test_metrics,
                args=vars(args),elapsed_minutes=(time.monotonic()-start)/60,
                note='Best observed within this search, not a global optimum. One seed only.')
    write_json(cli.output/'tuning_report.json',report)
    print('FINAL_REPORT',json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
