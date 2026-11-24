"""Budgeted paired subset screening, NOT a published full-data reproduction."""
# 中文说明：按用户历史长度分层抽样，保留被选用户的全部历史和原物品候选集合。
# 两组完成同一轮后才提交指标；不足以完成的一对轮次不能算作有效对照。
import argparse
import copy
import hashlib
import json
import math
import pickle
import random
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch

from compare_paper_ml100k import build_args,build_model,shared_hash,batch_update,write_json,save_checkpoint
from stable_optimizer import CPUAdam
from tune_ml100k import rng_state,restore_rng,evaluate
from utils import Data_Train,Data_Val,Data_Test

ROOT=Path(__file__).resolve().parents[1]
ACTIVE_OUTPUT=None


class BudgetReached(Exception):
    pass


def select_users(train,count,seed):
    """固定种子，在十个历史长度分层中抽取用户，不按点击或结果挑选样本。"""
    ordered=sorted(train,key=lambda user:(len(train[user]),user))
    rng=random.Random(seed)
    selected=[]
    for group in range(10):
        bucket=ordered[len(ordered)*group//10:len(ordered)*(group+1)//10]
        wanted=count//10+int(group<count%10)
        selected.extend(rng.sample(bucket,wanted))
    return sorted(selected)


def main():
    global ACTIVE_OUTPUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--users',type=int,default=150)
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--minutes',type=float,default=17)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--seed',type=int,default=1997)
    cli=parser.parse_args()
    started=time.monotonic()
    deadline=started+cli.minutes*60
    cli.device='cpu'
    cli.max_epochs=cli.epochs
    cli.output=cli.output.resolve()
    ACTIVE_OUTPUT=cli.output
    cli.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(cli.threads)
    source=ROOT/'datasets'/'data'/'ml-100k'/'dataset.pkl'
    with source.open('rb') as stream:
        original=pickle.load(stream)
    if not 10<=cli.users<=len(original['train']):
        raise ValueError('users must be between 10 and the full user count')
    users=select_users(original['train'],cli.users,cli.seed)
    data=copy.copy(original)
    for split in ('train','val','test'):
        # 只在内存中过滤同一批用户，不覆盖原 pickle，也不重排用户历史。
        data[split]={user:original[split][user] for user in users}
    examples=sum(len(sequence)-1 for sequence in data['train'].values())
    manifest=dict(mode='subset screening, not full ML-100K',seed=cli.seed,users=users,
        selection='10 user-history-length strata; fixed sample within each stratum',
        user_count=len(users),train_examples=examples,train_interactions=sum(map(len,data['train'].values())),
        val_examples=len(users),test_examples=len(users),catalog_items=len(data['smap']),
        original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_unchanged=True,max_epochs=cli.epochs,budget_minutes=cli.minutes)
    write_json(cli.output/'subset_manifest.json',manifest)
    arms=[]
    for name,enabled in [('fmrec_baseline',False),('fmrec_lifecycle',True)]:
        args=build_args(cli,data,enabled)
        args.eval_interval=1
        model=build_model(args)
        arms.append(dict(name=name,args=args,model=model,
            optimizer=CPUAdam(model.parameters(),lr=args.lr,clip_value=0.),
            train=Data_Train(data['train'],args).get_pytorch_dataloaders(),
            val=Data_Val(data['train'],data['val'],args).get_pytorch_dataloaders(),
            test=Data_Test(data['train'],data['val'],data['test'],args).get_pytorch_dataloaders(),
            rng=rng_state(),best=-math.inf,history=[],initial_hash=shared_hash(model)))
    if arms[0]['initial_hash']!=arms[1]['initial_hash']:
        raise RuntimeError('Shared initial parameters differ')
    pair_times=[]
    completed=0
    partial_discarded=False
    status=dict(state='training',started_at=datetime.now().isoformat(),
                subset_users=len(users),train_examples=examples,completed_paired_epochs=0)
    write_json(cli.output/'status.json',status)
    print('SUBSET',json.dumps(manifest),flush=True)
    for epoch in range(cli.epochs):
        # 依据已完成配对轮次的耗时预留余量，避免最后只训练完其中一个模型。
        if pair_times and deadline-time.monotonic()<max(pair_times)*1.25+60:
            break
        pair_started=time.monotonic()
        pending=[]
        try:
            for arm in arms:
                restore_rng(arm['rng'])
                model=arm['model']
                model.train()
                total=0.
                seen=0
                for index,batch in enumerate(arm['train']):
                    if time.monotonic()>deadline-60:
                        raise BudgetReached()
                    loss,count=batch_update(model,batch,arm['optimizer'],arm['args'])
                    total+=loss*count
                    seen+=count
                    if index%5==0 or index+1==len(arm['train']):
                        status.update(arm=arm['name'],epoch=epoch+1,batch=index+1,
                            batches=len(arm['train']),loss=loss,
                            elapsed_seconds=time.monotonic()-started,
                            updated_at=datetime.now().isoformat())
                        write_json(cli.output/'status.json',status)
                        print(arm['name'],f'epoch={epoch+1} batch={index+1}/{len(arm["train"])} loss={loss:.4f}',flush=True)
                if seen!=examples:
                    raise RuntimeError('Incomplete epoch')
                metrics=evaluate(model,arm['val'],arm['args'])
                arm['rng']=rng_state()
                pending.append(dict(epoch=epoch+1,loss=total/seen,validation=metrics,train_examples=seen))
        except BudgetReached:
            # 不完整配对不计入比较，最终加载此前已提交的最佳检查点。
            partial_discarded=True
            break
        # Commit only after BOTH models completed the same paired round.
        for arm,row in zip(arms,pending):
            arm['history'].append(row)
            write_json(cli.output/(arm['name']+'_history.json'),arm['history'])
            if row['validation']['NDCG@20']>arm['best']:
                arm['best']=row['validation']['NDCG@20']
                save_checkpoint(cli.output/(arm['name']+'_best.pt'),
                    dict(model_state_dict=arm['model'].state_dict(),args=vars(arm['args']),
                         epoch=epoch+1,validation=row['validation']))
            print('VALIDATION',arm['name'],json.dumps(row),flush=True)
        completed+=1
        pair_times.append(time.monotonic()-pair_started)
        status['completed_paired_epochs']=completed
        write_json(cli.output/'status.json',status)
    if not completed:
        raise RuntimeError('No complete paired epoch within budget; no comparison can be claimed')
    results=[]
    for arm in arms:
        saved=torch.load(cli.output/(arm['name']+'_best.pt'),map_location='cpu')
        arm['model'].load_state_dict(saved['model_state_dict'])
        test_metrics=evaluate(arm['model'],arm['test'],arm['args'])
        results.append(dict(name=arm['name'],best_epoch=saved['epoch'],
            epochs_completed=completed,validation=saved['validation'],test=test_metrics,
            args=vars(arm['args']),initial_shared_hash=arm['initial_hash']))
    paper={'HR@5':9.3338,'HR@10':15.4934,'HR@20':24.3146,
           # 论文值仅供参考；子集测试不能用于宣称超过论文全量结果。
           'NDCG@5':5.6848,'NDCG@10':7.6571,'NDCG@20':9.8158}
    report=dict(mode='rapid subset screening',manifest=manifest,experiments=results,
        completed_paired_epochs=completed,partial_pair_discarded=partial_discarded,
        elapsed_minutes=(time.monotonic()-started)/60,
        difference_lifecycle_minus_baseline_percentage_points={
            key:results[1]['test'][key]-results[0]['test'][key] for key in paper},
        published_full_data_values_reference_only=paper,
        direct_published_performance_comparison_valid=False,
        caveat='150-user subset and short training; one seed; no statistical superiority claim.')
    write_json(cli.output/'quick_comparison.json',report)
    lines=['# ML-100K 用户子集快速对照','',
        f'用户数：{len(users)}；每组完整训练轮数：{completed}；总耗时：{report["elapsed_minutes"]:.2f} 分钟。',
        '', '原数据未覆盖；完整历史保留；候选物品仍为原目录。两组共享相同初始主干权重。',
        '', '| 测试指标（%） | 无生命周期 | 有生命周期 | 差值（百分点） |',
        '|---|---:|---:|---:|']
    for key in paper:
        lines.append(f'| {key} | {results[0]["test"][key]:.4f} | {results[1]["test"][key]:.4f} | {report["difference_lifecycle_minus_baseline_percentage_points"][key]:+.4f} |')
    lines+=['','这是子集短程筛查，不能据此与论文全量指标直接比较，不能证明统计显著或最终收益。']
    (cli.output/'QUICK_RESULT.md').write_text('\n'.join(lines),encoding='utf-8')
    status.update(state='completed',elapsed_seconds=time.monotonic()-started,
                  completed_paired_epochs=completed,updated_at=datetime.now().isoformat())
    write_json(cli.output/'status.json',status)
    print('FINAL_REPORT',json.dumps(report),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception:
        if ACTIVE_OUTPUT is not None:
            path=ACTIVE_OUTPUT/'status.json'
            status=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
            status.update(state='failed',error=traceback.format_exc(),updated_at=datetime.now().isoformat())
            write_json(path,status)
        raise
