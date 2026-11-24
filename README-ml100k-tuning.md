# ML-100K 参数搜索与稳定性说明

本轮工作日期：2026-11-24。目标为在本机、完整本地 ML-100K 上寻找已观察到的较优配置，不能保证全局最优。

## 实验协议

- 保留现有数据文件和训练／验证／测试划分。
- 每个完成的 epoch 必须遍历全部 97,171 个训练前缀样本，不截断生命周期历史，不跳过异常 batch。
- 保留 FMRec 的流匹配、交叉熵和重建联合目标，以及五等级兴趣模块。
- 参数选择仅使用验证集 NDCG@20；候选模型最终选定后才评估测试集。
- 验证使用固定随机状态，结束后恢复训练随机状态；不同 epoch 不因验证消耗随机数而改变训练过程。
- HR/NDCG 按用户数汇总，排除 padding=0 的候选物品；单位为百分数。旧验证日志使用 batch 均值，因此不能将前后数值的差异完全归因于参数变化。
- 每轮存储验证曲线，验证主指标改善时保存模型。搜索预算在 epoch 边界检查，可能为完成整轮而稍超预算。

## 为什么增加稳定更新器

扩大模型后的 GPU 试验出现非有限损失。已保存的 `debug_optimizer.pt` 中，Adam 的 `exp_avg_sq` 出现负值（最小约 -2.787×10^13），该量按定义不能为负；随后对应权重产生 NaN。

使用单张量 GPU Adam 仍出现问题。`src/stable_optimizer.py` 的 CPUAdam 保留模型 GPU 前后向，将梯度检查、裁剪、Adam 更新放到 CPU，再同步参数；但本机 GPU 前后向路径随后仍出现非有限梯度，因此又启动完整 CPU 对照。

这些证据表明本机当前 GPU 计算路径仍不稳定，尚未证明具体是驱动、PyTorch、设备还是某个底层算子的原因，不能宣称已彻底修好 GPU。正式调参使用拒绝非有限梯度的路径，没有以替换 NaN 的方式取得结果。

CPUAdam 的数值更新已与标准 CPU Adam 进行测试对照；非有限梯度会在更新前报错。它是运行稳定性适配，不是新增推荐算法核心。

## 关键文件

- `src/tune_ml100k.py`：参数搜索、完整 epoch、固定验证、检查点及最终报告。
- `ml100k_tuning_candidates.json`：候选模型维度、学习率、batch size、裁剪幅度。
- `src/stable_optimizer.py`：CPU Adam 更新器。
- `src/main.py`：支持自动加载根目录 `ml100k_best_config.json`（文件存在时），显式 CLI 参数覆盖文件默认值。
- `src/trainer.py`：支持 CPUAdam，并以单一验证指标选模型及早停。
- `tests/test_tuning_evaluation.py`：尾批用户加权与随机状态恢复测试。
- `tests/test_stable_optimizer.py`：更新一致性和拒绝非有限梯度测试。

## 运行搜索

```powershell
Set-Location D:\CodexWorkSpace\FMRec
.\.venv\Scripts\python.exe .\src\tune_ml100k.py --device cpu --max_minutes 30 --screen_epochs 3 --max_epochs 60 --candidates_json .\ml100k_tuning_candidates.json --threads 8
```

GPU 当前存在上述异常，不应仅因为可识别 CUDA 就视为稳定。命令中的预算和 epoch 上限可以自行调整；增加预算能获得更充分训练，但仍需要以验证集选择模型。

如有完整检查点，可用 `--resume_checkpoint CHECKPOINT` 恢复模型、优化器、学习率调度器和随机状态；候选配置须与检查点匹配。

## 模型与结果

正常完成的搜索目录包含：

- `trial_N_history.json`：每轮损失、验证指标、全量样本数、耗时。
- `trial_N_best.pt`：该候选按验证集主指标选出的最佳检查点。
- `selected_model.pt`：最终选定模型，可用于 `inspect_interest.py`。
- `tuning_report.json`：实际搜索范围、选中配置、验证结果及最终一次测试结果。

GPU 失败的目录属于诊断记录，不等于完成的有效实验。最终文档和推荐配置以正常完成的 CPU 报告为准。各候选若因预算停止在不同 epoch，报告会保留实际轮数，不视为严格相同预算的架构对比。

本轮不构成升降级机制相对同配置原始 FMRec 的收益证明；这仍需开启／关闭模块的控制变量实验与多个种子验证。

## 初始模型的统一口径参照

最初一轮模型（16 维、窗口 10、2 步采样）的固定检查点已在 CPU、固定评估随机状态、用户加权且排除 padding 的条件下重新评估：

| 指标 | 初始模型重测（%） |
|---|---:|
| HR@5 | 0.212089 |
| HR@10 | 1.060445 |
| HR@20 | 2.439024 |
| NDCG@5 | 0.112578 |
| NDCG@10 | 0.380923 |
| NDCG@20 | 0.732113 |

该参照也包含兴趣升降级模块，不能视为关闭模块的原始 FMRec 基线。与最终模型的差异还包括训练轮数、窗口、维度和采样步数。
