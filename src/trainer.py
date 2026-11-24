"""训练与评估组织层。

每轮完整处理训练样本，按验证集单一主指标选模型；测试集不用于选择。
指标按用户数加权而不是对 batch 均值直接平均，输出单位为百分数。
CPUAdam 路径遇到非有限梯度会报错；旧 GPU 更新路径仍保留历史数值保护。
"""
import torch.nn as nn
import torch.optim as optim
import datetime
import torch
import numpy as np
import copy
import time
import pickle
import json
from pathlib import Path

from contextlib import contextmanager
from stable_optimizer import CPUAdam


def forward_batch(model, batch, train_flag=False, forward_mse_time=0):
    """统一转发双张量基线或带完整历史的三张量样本。"""
    history = batch[2] if len(batch) > 2 else None
    return model(batch[0], batch[1], forward_mse_time,
                 train_flag=train_flag, history=history)


def metric_percent(values):
    """values 为 (批次指标均值, 用户数)，避免最后一个小 batch 被过度加权。"""
    return round(sum(value*count for value,count in values)/sum(count for _,count in values)*100,4)


def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        if getattr(args,'optimizer_device','cuda') == 'cpu':
            return CPUAdam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay,
                           clip_value=args.grad_clip_value)
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError


def cal_hr(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    hr = [hit[:, :ks[i]].sum().item()/label.size()[0] for i in range(len(ks))]
    return hr


def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k-1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg/max_dcg).mean().item())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit/log2).sum(dim=-1)
    return rel


def hrs_and_ndcgs_k(scores, labels, ks):
    scores=scores.detach().clone()
    scores[:,0]=-torch.inf  # padding ID is not a candidate item
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    hr = cal_hr(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics[f'HR@{k}'] = hr_temp
        metrics[f'NDCG@{k}'] = ndcg_temp
    return metrics  


def LSHT_inference(model_joint, args, data_loader):
    device = args.device
    model_joint = model_joint.to(device)
    with torch.no_grad():
        test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        test_metrics_dict_mean = {}
        for test_batch in data_loader:
            test_batch = [x.to(device) for x in test_batch]
            scores_rec, rep_fm, _, _, _, _ = forward_batch(model_joint, test_batch)
            scores_rec_fm = model_joint.fm_rep_pre(rep_fm)
            metrics = hrs_and_ndcgs_k(scores_rec_fm, test_batch[1], [5, 10, 20])
            for k, v in metrics.items():
                test_metrics_dict[k].append((v,test_batch[1].size(0)))
    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = metric_percent(values_temp)
        test_metrics_dict_mean[key_temp] = values_mean
    print(test_metrics_dict_mean)


@contextmanager
def timer(name, logger=None):
    start = time.time()
    yield
    end = time.time()
    elapsed = end - start
    if logger:
        logger.info(f"{name} time: {elapsed:.4f} 秒")
    else:
        print(f"{name} time: {elapsed:.4f} 秒")




def model_train(tra_data_loader, val_data_loader, test_data_loader, model_joint, args, logger):
    epochs = args.epochs
    if epochs < 1 or args.eval_interval < 1:
        raise ValueError('epochs and eval_interval must be >= 1')
    device = args.device
    metric_ks = args.metric_ks
    Loss_Alpha = args.Loss_Alpha
    Loss_Beta = args.Loss_Beta
    model_joint = model_joint.to(device)
    # is_parallel = args.num_gpu > 1
    # if is_parallel:
    #     model_joint = nn.DataParallel(model_joint)
    optimizer = optimizers(model_joint, args)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
    
    best_metrics_dict = {f'Best_HR@{k}': 0.0 for k in metric_ks}
    best_metrics_dict.update({f'Best_NDCG@{k}': 0.0 for k in metric_ks})

    best_epoch = {f'Best_epoch_HR@{k}': -1 for k in metric_ks}
    best_epoch.update({f'Best_epoch_NDCG@{k}': -1 for k in metric_ks})

    bad_count = 0

    average_epoch_losses = []
    best_model = None
    best_selection_value = -float('inf')
    
    for epoch_temp in range(epochs):        
        print('Epoch: {}'.format(epoch_temp))
        logger.info('Epoch: {}'.format(epoch_temp))
        model_joint.train()
        
        epoch_losses = []

        forward_mse_time = 0.0

        flag_update = 0
        for index_temp, train_batch in enumerate(tra_data_loader):

            train_batch = [x.to(device) for x in train_batch]

            optimizer.zero_grad()

            loss_mse, fm_rep, weights, t, loss_FM_mse, z0 = forward_batch(
                model_joint, train_batch, train_flag=True, forward_mse_time=forward_mse_time)

            ##########X0_pred
            loss_fm_value = model_joint.loss_fm_ce(fm_rep, train_batch[1])

            loss_FM = loss_FM_mse

            # ######V_pred
            # loss_fm_value = model_joint.loss_x0_pred(fm_rep, loss_mse, z0)
            
            # loss_fm_value = model_joint.loss_EDM_MSE(fm_rep, t, z0)
          
            ##########X0_pred
            # loss_all = loss_beta * loss_FM + (1- loss_beta) * loss_fm_value +  Loss_Beta * loss_mse

            loss_all = loss_FM + Loss_Alpha * loss_fm_value +  Loss_Beta * loss_mse

            # ##########V_pred
            # loss_all = loss_fm_value


            # loss_all = loss_mse

            # Keep the original objective unchanged. A non-finite forward
            # value cannot produce a meaningful update, so fail explicitly
            # instead of silently treating that batch as trained.
            if not torch.isfinite(loss_all):
                raise FloatingPointError(
                    '[%d/%d] Non-finite loss before backward pass.' % (index_temp, len(tra_data_loader))
                )

            loss_all.backward()

            # Element-wise clipping retains every batch while preventing an
            # extreme or non-finite derivative from corrupting model weights.
            if args.grad_clip_value > 0 and not isinstance(optimizer,CPUAdam):
                for parameter in model_joint.parameters():
                    if parameter.grad is not None:
                        torch.nan_to_num_(
                            parameter.grad,
                            nan=0.0,
                            posinf=args.grad_clip_value,
                            neginf=-args.grad_clip_value,
                        )
                        parameter.grad.clamp_(-args.grad_clip_value, args.grad_clip_value)

            optimizer.step()

            epoch_losses.append(loss_all.item())

            if index_temp % int(len(tra_data_loader) / 5 + 1) == 0:
                print('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all.item()))
                logger.info('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all.item()))

        if not epoch_losses:
            raise RuntimeError('All batches in this epoch were numerically invalid.')
        average_loss = sum(epoch_losses) / len(epoch_losses)
        average_epoch_losses.append(average_loss)
        print("Average loss in epoch {}: {:.4f}".format(epoch_temp, average_loss))
        logger.info("Average loss in epoch {}: {:.4f}".format(epoch_temp, average_loss))
        if model_joint.interest_lifecycle is not None:
            lifecycle_summary = model_joint.interest_lifecycle.summary()
            print('Lifecycle (last training batch):', lifecycle_summary)
            logger.info('Lifecycle (last training batch): %s', lifecycle_summary)

        lr_scheduler.step()

        # Always validate the last epoch, including a one-epoch verification run.
        if (epoch_temp + 1) % args.eval_interval == 0 or epoch_temp == epochs - 1:
            print('start predicting: ', datetime.datetime.now())
            logger.info('start predicting: {}'.format(datetime.datetime.now()))
            model_joint.eval()
            with torch.random.fork_rng(devices=[0] if device=='cuda' else []), torch.no_grad():
                torch.manual_seed(args.random_seed+10000)
                metrics_dict = {f'HR@{k}': [] for k in metric_ks}
                metrics_dict.update({f'NDCG@{k}': [] for k in metric_ks})

                for val_batch in val_data_loader:
                    val_batch = [x.to(device) for x in val_batch]

                    scores_rec, rep_fm, _, _, _, _ = forward_batch(model_joint, val_batch)
                    scores_rec_fm = model_joint.fm_rep_pre(rep_fm)    ### inner_production
                    # scores_rec_fm = model_joint.routing_rep_pre(rep_fm)   ### routing_rep_pre
                    
                    metrics = hrs_and_ndcgs_k(scores_rec_fm, val_batch[1], metric_ks)
                    for k, v in metrics.items():
                        metrics_dict[k].append((v,val_batch[1].size(0)))
                        
            for key_temp, values_temp in metrics_dict.items():
                values_mean = metric_percent(values_temp)
                if values_mean > best_metrics_dict['Best_' + key_temp]:
                    best_metrics_dict['Best_' + key_temp] = values_mean
                    best_epoch['Best_epoch_' + key_temp] = epoch_temp
                    
            selection_metric=getattr(args,'selection_metric','NDCG@20')
            # 各指标历史最大值仅用于显示；实际检查点只由预先指定主指标决定。
            if selection_metric not in metrics_dict:
                raise ValueError('selection_metric must be included in metric_ks')
            selection_value=metric_percent(metrics_dict[selection_metric])
            flag_update=int(selection_value>best_selection_value)
            if flag_update:
                best_selection_value=selection_value
                bad_count=0
            if flag_update == 0:
                bad_count += 1
            else:
                print(best_metrics_dict)
                print(best_epoch)
                logger.info(best_metrics_dict)
                logger.info(best_epoch)
                best_model = copy.deepcopy(model_joint)
            if bad_count >= args.patience:
                break
            
    
    logger.info(best_metrics_dict)
    logger.info(best_epoch)
        
    if best_model is None:
        best_model = copy.deepcopy(model_joint)
    
    
    top_100_item = []
    with torch.random.fork_rng(devices=[0] if device=='cuda' else []), torch.no_grad():
        torch.manual_seed(args.random_seed+10000)
        test_metrics_dict = {f'HR@{k}': [] for k in metric_ks}
        test_metrics_dict.update({f'NDCG@{k}': [] for k in metric_ks})

        test_metrics_dict_mean = {}
        for test_batch in test_data_loader:
            test_batch = [x.to(device) for x in test_batch]

            scores_rec, rep_fm, _, _, _, _ = forward_batch(best_model, test_batch)
            scores_rec_fm = best_model.fm_rep_pre(rep_fm)   ### Inner Production
            # scores_rec_fm = best_model.routing_rep_pre(rep_fm)   ### routing
            
            _, indices = torch.topk(scores_rec_fm, k=100)
            top_100_item.append(indices)

            metrics = hrs_and_ndcgs_k(scores_rec_fm, test_batch[1], metric_ks)
            for k, v in metrics.items():
                test_metrics_dict[k].append((v,test_batch[1].size(0)))
    
    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = metric_percent(values_temp)
        test_metrics_dict_mean[key_temp] = values_mean



    top_100_item_train = []
    with torch.no_grad():
        train_metrics_dict = {f'HR@{k}': [] for k in metric_ks}
        train_metrics_dict.update({f'NDCG@{k}': [] for k in metric_ks})

        train_metrics_dict_mean = {}
        for train_batch in tra_data_loader if getattr(args,'report_train_metrics',True) else []:
            train_batch = [x.to(device) for x in train_batch]

            scores_rec, rep_fm, _, _, _, _ = forward_batch(best_model, train_batch)
            scores_rec_fm = best_model.fm_rep_pre(rep_fm)   ### Inner Production
            # scores_rec_fm = best_model.routing_rep_pre(rep_fm)   ### routing
            
            _, indices = torch.topk(scores_rec_fm, k=100)
            top_100_item_train.append(indices)

            metrics = hrs_and_ndcgs_k(scores_rec_fm, train_batch[1], metric_ks)
            for k, v in metrics.items():
                train_metrics_dict[k].append((v,train_batch[1].size(0)))

    for key_temp, values_temp in train_metrics_dict.items():
        if values_temp:
            values_mean = metric_percent(values_temp)
            train_metrics_dict_mean[key_temp] = values_mean

    print('Train------------------------------------------------------')
    logger.info('Train------------------------------------------------------')
    print(train_metrics_dict_mean)
    logger.info(train_metrics_dict_mean)


    print('Test------------------------------------------------------')
    logger.info('Test------------------------------------------------------')
    print(test_metrics_dict_mean)
    logger.info(test_metrics_dict_mean)

    print('Best Eval---------------------------------------------------------')
    logger.info('Best Eval---------------------------------------------------------')
    print(best_metrics_dict)
    print(best_epoch)
    logger.info(best_metrics_dict)
    logger.info(best_epoch)

    print(args)
            

    if best_model.interest_lifecycle is not None:
        output_dir = Path(args.log_file) / args.dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        # last_stats holds diagnostics only; do not retain it in the checkpoint.
        saved_args={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
        report = {'args': saved_args, 'test_metrics': test_metrics_dict_mean,
                  'last_batch_lifecycle': best_model.interest_lifecycle.summary()}
        run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        torch.save({'model_state_dict': best_model.state_dict(), 'args': vars(args)},
                   output_dir / ('lifecycle_' + run_id + '.pt'))
        with (output_dir / ('lifecycle_' + run_id + '.json')).open('w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return best_model, test_metrics_dict_mean
    
