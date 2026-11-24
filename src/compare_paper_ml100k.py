"""Paired ML-100K experiment at the paper's 128-dimensional configuration.

Same shared initialization, data, objective, update limit and evaluation protocol.
This is an adapted local reproduction, not a claim of exact published reproduction.

中文说明：128 维本地对照顺序执行基线与生命周期模型，匹配共享权重和随机起点。
本地训练上限、早停和评估聚合口径仍可能与论文不同，不能称为已认证完全复现。
"""
import argparse
import copy
import hashlib
import json
import math
import pickle
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch

from interest_lifecycle import InteractionLifecycle
from model import Att_FM_model, create_model_FM
from stable_optimizer import CPUAdam
from trainer import forward_batch
from tune_ml100k import seed_all, rng_state, restore_rng, evaluate, make_args
from utils import Data_Train, Data_Val, Data_Test

ROOT=Path(__file__).resolve().parents[1]


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def save_checkpoint(path,value):
    # 临时文件写完再替换，降低长任务中途退出留下半个检查点的风险。
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temporary)
    temporary.replace(path)


def build_args(cli,data,enabled):
    args=make_args(dict(hidden_size=128,batch_size=512,lr=.001,
        interest_lifecycle=int(enabled),grad_clip_value=0.,decay_step=100,gamma=.1,
        eval_interval=2,patience=8),cli,data)
    args.epochs=cli.epochs
    args.random_seed=cli.seed
    return args


def build_model(args):
    seed_all(args.random_seed)
    base_args=copy.copy(args)
    base_args.interest_lifecycle=0
    model=Att_FM_model(create_model_FM(base_args),base_args).to(args.device)
    if args.interest_lifecycle:
        # Creating extra parameters must not shift the shared training RNG stream.
        saved=rng_state()
        seed_all(args.random_seed+33)
        model.interest_lifecycle=InteractionLifecycle(args.hidden_size,args.num_interests,
            args.interest_promote,args.interest_degrade_gap,args.interest_repromote,
            args.interest_temperature).to(args.device)
        restore_rng(saved)
    return model


def shared_hash(model):
    """只校验共享主干参数，新增生命周期参数不参与两组初始化一致性比较。"""
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        if not name.startswith('interest_lifecycle.'):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def batch_update(model,batch,optimizer,args):
    # 三项损失和原权重保持不变，生命周期参数通过同一目标联合学习。
    batch=[x.to(args.device) for x in batch]
    optimizer.zero_grad(set_to_none=True)
    reconstruction,rep,_,_,fm_loss,_=forward_batch(model,batch,True)
    loss=fm_loss+args.Loss_Alpha*model.loss_fm_ce(rep,batch[1])+args.Loss_Beta*reconstruction
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite forward loss; stopped without skipping a batch')
    loss.backward()
    optimizer.step()  # Strict finite-gradient checks; no replacement of bad values.
    return float(loss.detach()),batch[1].size(0)


def benchmark(cli,data):
    result={}
    for enabled in (False,True):
        name='fmrec_baseline' if not enabled else 'fmrec_lifecycle'
        args=build_args(cli,data,enabled)
        model=build_model(args)
        model.train()
        loader=Data_Train(data['train'],args).get_pytorch_dataloaders()
        optimizer=CPUAdam(model.parameters(),lr=args.lr,clip_value=0.)
        measurements=[]
        for index,batch in enumerate(loader):
            start=time.monotonic()
            loss,count=batch_update(model,batch,optimizer,args)
            measurements.append(time.monotonic()-start)
            if index+1>=cli.benchmark_batches:
                break
        seconds=sum(measurements[1:])/max(1,len(measurements)-1)
        result[name]={'batch_seconds':seconds,'batches_per_epoch':len(loader),
            'estimated_epoch_minutes':seconds*len(loader)/60,
            'benchmark_batches':len(measurements),'last_loss':loss}
        print(name,json.dumps(result[name]),flush=True)
        del model,optimizer,loader
    write_json(cli.output/'speed_estimate.json',result)


def train_arm(cli,data,name,enabled,status):
    args=build_args(cli,data,enabled)
    directory=cli.output/name
    directory.mkdir(parents=True,exist_ok=True)
    model=build_model(args)
    initial_hash=shared_hash(model)
    train_loader=Data_Train(data['train'],args).get_pytorch_dataloaders()
    val_loader=Data_Val(data['train'],data['val'],args).get_pytorch_dataloaders()
    optimizer=CPUAdam(model.parameters(),lr=args.lr,clip_value=0.)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,args.decay_step,args.gamma)
    best=-math.inf
    bad=0
    history=[]
    first=0
    latest=directory/'latest.pt'
    if latest.exists():
        # 只从完整轮次恢复；改变配置后须另建实验目录，不能混用历史结果。
        saved=torch.load(latest,map_location=args.device)
        if saved['args']!=vars(args):
            raise ValueError('Resume config differs from checkpoint; use a new output directory')
        model.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        restore_rng(saved['rng'])
        first=saved['epoch']+1
        best=saved['best_score']
        bad=saved['bad_checks']
        history=saved['history']
    status.update(arm=name,state='training',epoch=first,initial_shared_hash=initial_hash)
    write_json(cli.output/'status.json',status)
    for epoch in range(first,cli.epochs):
        if bad>=args.patience:
            break
        started=time.monotonic()
        model.train()
        seen=0
        total=0.
        for index,batch in enumerate(train_loader):
            loss,count=batch_update(model,batch,optimizer,args)
            total+=loss*count
            seen+=count
            if index%20==0 or index+1==len(train_loader):
                elapsed=time.monotonic()-started
                status.update(epoch=epoch+1,batch=index+1,batches=len(train_loader),
                    train_examples_seen=seen,loss=loss,epoch_seconds=elapsed,
                    epoch_remaining_seconds=elapsed/(index+1)*(len(train_loader)-index-1),
                    updated_at=datetime.now().isoformat())
                write_json(cli.output/'status.json',status)
                print(f'{name} epoch={epoch+1} batch={index+1}/{len(train_loader)} loss={loss:.4f}',flush=True)
        if seen!=len(train_loader.dataset):
            raise RuntimeError('Incomplete training epoch')
        scheduler.step()
        metrics=None
        if (epoch+1)%args.eval_interval==0 or epoch==cli.epochs-1:
            metrics=evaluate(model,val_loader,args)
            score=metrics['NDCG@20']
            if score>best:
                best=score
                bad=0
                save_checkpoint(directory/'best.pt',dict(model_state_dict=model.state_dict(),args=vars(args),
                    epoch=epoch,validation=metrics))
            else:
                bad+=1
        history.append(dict(epoch=epoch+1,loss=total/seen,validation=metrics,
                            train_examples=seen,seconds=time.monotonic()-started))
        save_checkpoint(latest,dict(model_state_dict=model.state_dict(),args=vars(args),epoch=epoch,
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(),
            best_score=best,bad_checks=bad,history=history))
        write_json(directory/'history.json',history)
        print(name,'EPOCH',json.dumps(history[-1]),flush=True)
    saved=torch.load(directory/'best.pt',map_location=args.device)
    model.load_state_dict(saved['model_state_dict'])
    # Test only after this arm's validation-selected checkpoint is frozen.
    test_loader=Data_Test(data['train'],data['val'],data['test'],args).get_pytorch_dataloaders()
    metrics=evaluate(model,test_loader,args)
    result=dict(arm=name,args=vars(args),initial_shared_hash=initial_hash,
                epochs_completed=len(history),best_epoch=saved['epoch']+1,
                validation=saved['validation'],test=metrics,
                stop_reason='early_stop' if bad>=args.patience else 'epoch_limit')
    write_json(directory/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['benchmark','train'],default='train')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--seed',type=int,default=1997)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--device',choices=['cpu'],default='cpu')
    parser.add_argument('--benchmark_batches',type=int,default=6)
    cli=parser.parse_args()
    cli.max_epochs=cli.epochs
    cli.output=cli.output.resolve()
    cli.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(cli.threads)
    data_path=ROOT/'datasets'/'data'/'ml-100k'/'dataset.pkl'
    with data_path.open('rb') as stream:
        data=pickle.load(stream)
    status=dict(state='starting',started_at=datetime.now().isoformat(),
        dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        purpose='128d paired baseline and lifecycle comparison; not exact publication reproduction',
        max_epochs_per_arm=cli.epochs,metric_units='percent',selection='validation NDCG@20')
    if cli.mode=='benchmark':
        benchmark(cli,data)
        return
    try:
        results=[]
        for name,enabled in [('fmrec_baseline',False),('fmrec_lifecycle',True)]:
            result_file=cli.output/name/'result.json'
            if result_file.exists():
                result=json.loads(result_file.read_text(encoding='utf-8'))
                if result['args']!=vars(build_args(cli,data,enabled)):
                    raise ValueError('Completed result config differs; use a new output directory')
            else:
                result=train_arm(cli,data,name,enabled,status)
            results.append(result)
        if results[0]['initial_shared_hash']!=results[1]['initial_shared_hash']:
            raise RuntimeError('Shared initialization hashes differ')
        paper={'HR@5':9.3338,'HR@10':15.4934,'HR@20':24.3146,
               'NDCG@5':5.6848,'NDCG@10':7.6571,'NDCG@20':9.8158}
        report=dict(experiments=results,published_fmrec_percent=paper,
            difference_lifecycle_minus_baseline={k:results[1]['test'][k]-results[0]['test'][k] for k in paper},
            difference_lifecycle_minus_paper={k:results[1]['test'][k]-paper[k] for k in paper},
            note='One seed; no significance claim. Same local data/evaluation across arms; published protocol may differ.')
        write_json(cli.output/'comparison.json',report)
        status.update(state='completed',updated_at=datetime.now().isoformat())
        write_json(cli.output/'status.json',status)
    except BaseException:
        status.update(state='failed',error=traceback.format_exc(),updated_at=datetime.now().isoformat())
        write_json(cli.output/'status.json',status)
        raise


if __name__=='__main__':
    main()
