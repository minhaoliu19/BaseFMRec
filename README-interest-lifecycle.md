# S0—S4 交互位置兴趣升降级实验

## 已实现的范围

保持所有 `datasets/data/*/dataset.pkl` 不变，只增加代码。保留 FMRec 原有
直线路径、流匹配/推荐/重建损失及 ODE 采样，在历史编码后注入兴趣记忆和权重。
本模块默认开启，使用 `--interest_lifecycle 0` 可关闭并走原模型路径。

这里的“间隔”是其间出现了多少次**其他潜在兴趣的交互**，不是天数。
潜在兴趣由可训练原型与物品 embedding 的余弦相似度学习，不等于已知商品品类。
每次交互在前向计算中分配给最相近的一个原型；反向使用 softmax 近似梯度。

## 状态和规则

| 状态 | 实现含义 |
|---|---|
| S0 | 该历史前缀尚未给此潜在兴趣提供交互证据；不代表负反馈 |
| S1 | 已发生相关交互 |
| S2 | 累计证据达到第二级阈值（初始 2） |
| S3 | 累计证据达到第三级阈值（初始 4），或成熟兴趣降级 |
| S4 | 累计证据达到成熟阈值（初始 8），且当前活跃 |

成熟后若连续 G 次其他兴趣交互（G 初始为 20），S4 降到 S3，保留成熟记忆。
降级后的首次相关交互重新开始恢复计数，累计 N 次相关交互（N 初始为 3）恢复 S4。
恢复过程中若再次出现 G 次其他交互，恢复证据重新计数。曾经成熟的兴趣不低于 S3。
未成熟兴趣当前按累计证据晋升，不实现额外的 S2→S1 等降级规则。

晋升阈值、G、N、等级基础权重、兴趣原型和条件残差强度均参与优化。
阈值为连续参数：若学到 N=3.07，硬前向需 4 次相关交互。等级权重单调递增，
S0=0、S4=1，但最终推荐仍由 FMRec 生成表示计算。离散路由和转移采用直通估计：
前向严格按五态执行，反向以 sigmoid/softmax 近似，并非离散判断的精确梯度。
状态没有人工真值标签，目前完全依靠原推荐目标训练；有效性须经消融实验确认。

## 历史及防泄漏

近期序列仍截断至 `max_len`。另向模块传递**完整可见历史前缀**，不截断长期记忆。
训练前缀不含当前标签；验证只使用训练历史；测试使用训练历史与验证交互。
状态随每个样本重放，不跨 batch 存储或随打乱的训练顺序更新。
“曾经成熟”指当前模型参数下重放该前缀曾满足成熟条件；不是跨训练 epoch 的固定标签。

## 运行

以下命令从项目根目录运行，使用本地 Python 3.10 环境和 GPU。

```powershell
Set-Location D:\CodexWorkSpace\FMRec
.\.venv\Scripts\python.exe .\src\main.py --dataset ml-100k --device cuda --interest_lifecycle 1 --epochs 1 --batch_size 512 --log_file src/log/lifecycle/
```

直接在 IDE 启动 `src/main.py` 也会启用模块。现有默认参数仍是小规模验证设置
（max_len=10、hidden_size=16、epochs=1、sample_N=2），不是正式论文复现设置。

对应基线：同一命令改为 `--interest_lifecycle 0 --log_file src/log/baseline/`。
多轮对照应保持数据、超参数和种子一致；新增模型初始化会消耗不同随机数，
正式结论应使用多个种子，不应以单轮结果宣称优于基线。

可选初始参数：

```text
--num_interests 8
--interest_promote 2 4 8
--interest_degrade_gap 20
--interest_repromote 3
--interest_temperature 0.5
```

每轮打印最后一个训练 batch 的 S0—S4 统计及学习后阈值。这是“样本×潜在兴趣”数量，
不是全数据集的唯一兴趣数量。训练结束后输出 `lifecycle_*.pt` 模型和同名 JSON 报告。
最后一轮始终验证，因此 epochs=1 也会更新 Best Eval。测试集不参与最佳模型选择。
当前沿用原项目 HR/NDCG 的批次均值统计及百分数输出格式。

## 检查单个用户的状态轨迹

将 CHECKPOINT 替换为实际模型路径；默认选择训练集第一个用户，也可加 `--user 1`。

```powershell
.\.venv\Scripts\python.exe .\src\inspect_interest.py --checkpoint CHECKPOINT --output src/log/interest-trace.json
```

脚本只读取该用户的训练历史，输出每次交互后的状态、间隔、计数、权重及所有状态转移。
它在固定 checkpoint 下重放历史，不代表训练期间真实用户的心理状态标签。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖晋升、保护降级、重新激活、重复间隔重置、padding、完整历史、标签隔离、
batch 顺序不变性、阈值/原型梯度、开启和关闭模块的 FMRec 训练推理。

已在 ML-100K 的全部 190 个 batch 上完成一轮 GPU 验证，平均损失 3.9496，
验证和测试正常结束。初始晋升阈值 [2,4,8] 更新为约 [2.0648,4.1488,8.2335]，
G 更新为约 19.9281，N 更新为约 3.0723。该结果仅证明运行与参数更新有效，
尚未证明推荐效果提升。其余三个数据集尚未进行完整训练验证；完整历史带来额外显存开销，
长序列数据可调低 batch_size，但不应跳过样本或截断生命周期历史。
