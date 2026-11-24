"""训练入口：解析参数、读取数据、构造模型并启动训练。

ML-100K 可自动读取根目录保存的参数，命令行选项优先；dry_run 只显示配置。
配置中的 epochs 是上限，不表示已完成轮数。device 与 optimizer_device 分开，
允许 GPU 前后向配合 CPU 参数更新；该组合本身不保证 GPU 数值稳定。
"""
import os
# GTX 1650 + this PyTorch CUDA build can report asynchronous kernel errors
# during in-place gradient limiting. Synchronize CUDA launches so failures are
# attributed correctly in IDE runs too. Synchronization does not guarantee
# numerical stability; the selected ML-100K configuration may use CPU instead.
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
import datetime
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import logging
import time
import pickle
import json
from pathlib import Path
from utils import Data_Train, Data_Val, Data_Test, Data_CHLS
from model import create_model_FM, Att_FM_model
from trainer import model_train, LSHT_inference
from collections import Counter

import itertools
from copy import deepcopy
import multiprocessing

# Device selection is controlled by ``--device``.  Do not hard-code a GPU
# index here: doing so makes the program fail on workstations with one GPU.


parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', default='amazon_beauty', help='Dataset name: amazon_beauty, steam, ml-100k, yelp')
parser.add_argument('--dataset', default='ml-100k', help='Dataset name: amazon_beauty, steam, ml-100k, yelp')
parser.add_argument('--log_file', default='log/', help='log dir path')
parser.add_argument('--random_seed', type=int, default=1997, help='Random seed')  
parser.add_argument('--max_len', type=int, default=10, help='The max length of sequence')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPU')
parser.add_argument('--cpu_threads',type=int,default=8,help='CPU compute threads')
parser.add_argument('--batch_size', type=int, default=512, help='Batch Size')  
parser.add_argument("--hidden_size", default=16, type=int, help="hidden size of model")
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout of representation')
parser.add_argument('--emb_dropout', type=float, default=0.3, help='Dropout of item embedding')
parser.add_argument("--hidden_act", default="gelu", type=str) # gelu relu
parser.add_argument('--num_blocks', type=int, default=4, help='Number of Transformer blocks')
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training')
parser.add_argument('--decay_step', type=int, default=100, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20], help='ks for Metric@k')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--optimizer_device',choices=['cpu','cuda'],default='cpu',
                    help='CPU Adam update; GPU still computes model forward and backward')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--grad_clip_value', type=float, default=1.0,
                    help='Element-wise gradient limit; set to 0 to disable clipping')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
parser.add_argument('--lambda_uncertainty', type=float, default=0.001, help='uncertainty weight')
parser.add_argument('--eval_interval', type=int, default=99, help='the number of epoch to eval')
parser.add_argument('--patience', type=int, default=5, help='the number of epoch to wait before early stop')
parser.add_argument('--selection_metric',choices=['HR@5','HR@10','HR@20','NDCG@5','NDCG@10','NDCG@20'],
                    default='NDCG@20',help='Single validation metric for checkpoint selection')
parser.add_argument('--report_train_metrics',type=int,choices=[0,1],default=0,
                    help='Optional expensive training-set inference after all training epochs')
parser.add_argument('--eps', type=float, default=0.001, help='start step')
parser.add_argument('--sample_N', type=int, default=2, help='Euler calculate steps')
parser.add_argument('--lambda_t', type=float, default=0.001, help='scale of condition')
parser.add_argument('--dropout_c', type=float, default=0.1, help='scale of condition')
parser.add_argument('--eps_reverse', type=float, default=0.001, help='reverse start step')
parser.add_argument('--m_logNorm', type=float, default=1.0, help='Logit-Normal Sampling mean')
parser.add_argument('--s_logNorm', type=float, default=0.6, help='Logit-Normal Sampling Variance')
parser.add_argument('--s_modsamp', type=float, default=1.0, help='Mode_sample_timestep scale parameter')
parser.add_argument('--last', type=int, default=2, help='last H Get')
parser.add_argument('--mask_ratio', type=float, default=1.0, help='Balanced positive-negative class ratio')
parser.add_argument('--sampling_method', type=str, default='mode', choices=['uniform', 'logit_normal', 'mode', 'cosmap'])
parser.add_argument('--Loss_Alpha', type=float, default=0.2, help='Loss parameter')
parser.add_argument('--Loss_Beta', type=float, default=0.4, help='Loss parameter')
parser.add_argument('--interest_lifecycle', type=int, choices=[0, 1], default=1,
                    help='Enable interaction-index S0-S4 lifecycle; 0 reproduces baseline path')
parser.add_argument('--num_interests', type=int, default=8, help='Number of learned latent interests')
parser.add_argument('--interest_promote', nargs=3, type=float, default=[2., 4., 8.],
                    help='Initial evidence thresholds for S2 S3 S4 (learned during training)')
parser.add_argument('--interest_degrade_gap', type=float, default=20.,
                    help='Initial unrelated-interaction gap for S4 -> S3 (learned)')
parser.add_argument('--interest_repromote', type=float, default=3.,
                    help='Initial related-interaction evidence for S3 -> S4 recovery (learned)')
parser.add_argument('--interest_temperature', type=float, default=0.5,
                    help='Soft routing and transition surrogate temperature')
parser.add_argument('--config',type=Path,
                    default=Path(__file__).resolve().parents[1]/'ml100k_best_config.json',
                    help='Optional saved ML-100K tuning defaults; explicit CLI options override')
parser.add_argument('--dry_run',action='store_true',help='Print resolved configuration without training')
preliminary,_=parser.parse_known_args()
# 第一次解析只寻找数据集及配置位置，第二次解析保证显式 CLI 参数覆盖默认值。
if preliminary.dataset == 'ml-100k' and preliminary.config.exists():
    with preliminary.config.open('r',encoding='utf-8') as config_file:
        selected_config=json.load(config_file)
    valid_keys={action.dest for action in parser._actions}
    unknown=set(selected_config)-valid_keys
    if unknown:
        raise ValueError('Unknown config keys: '+str(unknown))
    parser.set_defaults(**selected_config)
args = parser.parse_args()
if args.cpu_threads < 1:
    raise ValueError('cpu_threads must be >= 1')
torch.set_num_threads(args.cpu_threads)

print(args)
if args.config.exists() and args.dataset=='ml-100k':
    print('Loaded tuning defaults:',args.config)
if args.dry_run:
    raise SystemExit(0)

if not os.path.exists(args.log_file):
    os.makedirs(args.log_file)
if not os.path.exists(args.log_file + args.dataset):
    os.makedirs(args.log_file + args.dataset )


logging.basicConfig(level=logging.INFO, filename=args.log_file + args.dataset + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log',
                    datefmt='%Y/%m/%d %H:%M:%S', format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s', filemode='w')
logger = logging.getLogger(__name__)
logger.info(args)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def item_num_create(args, item_num):
    args.item_num = item_num + 1
    return args

def user_num_create(args, user_num):
    args.user_num = user_num + 1
    return args

def main(args,logger):    
    """只读取既有 pickle；不在训练入口重写划分或覆盖原始数据。"""
    fix_random_seed_as(args.random_seed)
    path_data = Path(__file__).resolve().parent.parent / 'datasets' / 'data' / args.dataset / 'dataset.pkl'
    with open(path_data, 'rb') as f:
        data_raw = pickle.load(f)
    
    args = item_num_create(args, len(data_raw['smap']))
    args = user_num_create(args, len(data_raw['train'].items()))

    tra_data = Data_Train(data_raw['train'], args)
    val_data = Data_Val(data_raw['train'], data_raw['val'], args)
    test_data = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'], args)
    tra_data_loader = tra_data.get_pytorch_dataloaders()
    val_data_loader = val_data.get_pytorch_dataloaders()
    test_data_loader = test_data.get_pytorch_dataloaders()
    FM_rec = create_model_FM(args)
    rec_fm_joint_model = Att_FM_model(FM_rec, args)
    
    best_model, test_results = model_train(tra_data_loader, val_data_loader, test_data_loader, rec_fm_joint_model, args, logger)


if __name__ == '__main__':
    main(args,logger)
