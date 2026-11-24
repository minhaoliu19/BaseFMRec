# ML-100K：128 维 FMRec 与兴趣生命周期对照

本轮目标：先比较相同本地条件下原始 FMRec 主干与加入兴趣生命周期的模型，再将两者测试指标与论文表 2 数值并列。不能在实验结束前声称提升。

## 固定配置

- 数据：现有 ML-100K，943 用户、1682 物品、原训练／验证／测试划分，不修改 pickle。
- 维度 128、序列窗口 50、batch=512、4 个 Transformer block、4 个注意力头、30 步 Euler 采样。
- Adam 学习率 0.001，100 轮衰减一次、倍率 0.1，α=0.2、β=0.4。
- CPU float32；GPU 之前有数值异常，当前不作为可靠训练路径。
- 不进行梯度裁剪，贴近官方 Adam 更新；非有限梯度会报错，不能替换或跳过 batch。
- 每组最多 60 轮，每 2 轮验证，验证 NDCG@20 连续 8 次不改善则早停。
- seed=1997，两组共享参数的初始权重及训练随机数起点相同，并记录共享权重 SHA256。
- 两组只改变兴趣生命周期开关；开启组额外接收完整历史记忆，这也是改进方法的一部分。
- 指标按用户加权，排除 padding，固定评估随机状态。测试仅在各组模型经验证集选定后使用。

## 与官方完全复现的边界

对齐的是论文主要模型和优化超参数。官方代码默认最大 500 轮，本轮为本机可承受的 60 轮上限，验证频率／早停口径也不同；本地 CPU 环境、数据预处理来源与评估聚合方式未证明与论文逐项完全一致。

因此 `fmrec_baseline` 表示本地 FMRec 主干对照，而不是已经认证的论文复现。若 60 轮仍在改善，不能称为收敛。单个种子不支持显著性结论。

## 运行及进度

```powershell
Set-Location D:\CodexWorkSpace\FMRec
.\.venv\Scripts\python.exe .\src\compare_paper_ml100k.py --mode train --device cpu --epochs 60 --threads 4 --output .\src\log\paper_comparison_128_20261124
```

运行顺序：先 `fmrec_baseline`（无生命周期），再 `fmrec_lifecycle`（开启生命周期），不同时抢占 CPU。

- `status.json`：当前组、轮次、batch 进度和该轮剩余时间估计，每 20 个 batch 更新。
- `speed_estimate.json`：测速结果，不能当作训练指标。
- `fmrec_baseline/history.json` 与 `fmrec_lifecycle/history.json`：完整轮次记录。
- 每组 `latest.pt`：最近完整轮次及优化器、调度器和随机状态，支持断点续训。
- 每组 `best.pt`：验证集 NDCG@20 最优模型。
- 每组 `result.json`：该组完成后的验证／测试指标。
- `comparison.json`：两组全部完成后生成，同时列出本地组间差值和相对论文数值的差值（百分点）。

同一命令再次执行时，会从最近完整轮次恢复；若配置与保存值不同则拒绝恢复，避免混淆。已完成的组直接读取结果，不会重新训练或重新选参。

CPU 每轮较慢，两组充分训练可能需要数小时甚至一天以上。最终以测速和实际 epoch 耗时更新估计，初始几轮不能当作最终指标。
