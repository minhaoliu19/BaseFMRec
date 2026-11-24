"""Inspect a trained checkpoint's lifecycle using training prefixes only."""
# 中文说明：固定模型参数后重放训练历史，不进行优化、不使用验证或测试答案。
# 输出的兴趣编号是潜在原型 ID，不等于人工标注的电影品类或心理学阶段真值。
import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn.utils.rnn import pad_sequence

from model import Att_FM_model, create_model_FM


def main():
    parser = argparse.ArgumentParser(description='Replay S0-S4 states on observed training interactions')
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--user', type=int, default=None, help='Local user ID; default first training user')
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--output', type=Path, default=None)
    cli = parser.parse_args()
    saved = torch.load(cli.checkpoint, map_location=cli.device)
    args = SimpleNamespace(**saved['args'])
    if not args.interest_lifecycle:
        raise ValueError('Checkpoint has no lifecycle module')
    model = Att_FM_model(create_model_FM(args), args).to(cli.device)
    model.load_state_dict(saved['model_state_dict'])
    model.eval()
    dataset_path = Path(__file__).resolve().parent.parent/'datasets'/'data'/args.dataset/'dataset.pkl'
    with dataset_path.open('rb') as f:
        dataset = pickle.load(f)
    user = cli.user if cli.user is not None else next(iter(dataset['train']))
    if user not in dataset['train']:
        raise ValueError('User ID is not in this dataset: ' + str(user))
    interactions = dataset['train'][user]
    # 同一模型下逐前缀观察状态，避免把跨训练 epoch 的原型变化误当成用户演化。
    records = []
    with torch.no_grad():
        # Bounded batches; do not discard old events or read val/test targets.
        for start in range(1, len(interactions)+1, 16):
            lengths = list(range(start, min(start+16, len(interactions)+1)))
            prefixes = [torch.tensor(interactions[:n], device=cli.device) for n in lengths]
            history = pad_sequence(prefixes, batch_first=True)
            local = torch.tensor([[0]*max(0,args.max_len-n)+interactions[:n][-args.max_len:]
                                  for n in lengths], device=cli.device)
            representation = model.LayerNorm(model.item_embeddings(local))
            model.interest_lifecycle(representation, local, history, model.item_embeddings)
            stats = model.interest_lifecycle.last_stats
            for row, n in enumerate(lengths):
                records.append({'observed_interactions':n, 'item_id':interactions[n-1],
                    'states':stats['states'][row].cpu().tolist(),
                    'counts':stats['counts'][row].cpu().tolist(),
                    'unrelated_gaps':stats['gaps'][row].cpu().tolist(),
                    'ever_mature':stats['ever_mature'][row].cpu().tolist(),
                    'recovery_count':stats['recovery_count'][row].cpu().tolist(),
                    'weights':stats['weights'][row].cpu().tolist()})
    changes = []
    previous = [0]*args.num_interests
    for record in records:
        for k, state in enumerate(record['states']):
            if state != previous[k]:
                changes.append({'position':record['observed_interactions'], 'interest':k,
                                'from':previous[k], 'to':state})
        previous = record['states']
    parameters = model.interest_lifecycle.summary()
    parameters.pop('state_counts', None)
    report = {'dataset':args.dataset, 'user':user,
              'note':'Latent interest IDs, not category labels. Training prefix replay under fixed checkpoint parameters.',
              'parameters':parameters, 'transitions':changes, 'history':records}
    if cli.output:
        cli.output.parent.mkdir(parents=True,exist_ok=True)
        cli.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'dataset':args.dataset,'user':user,'events':len(records),
                      'parameters':report['parameters'],'transitions':changes},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
