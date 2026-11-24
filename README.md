# FMRec

Flow-Matching based Sequential Recommender Model

本项目基于 [FMRec 官方实现](https://github.com/FengLiu-1/FMRec)，增加按交互位置
计算的 S0—S4 兴趣生命周期模块及可复核的实验脚本。
原论文：[Flow Matching Based Sequential Recommender Model，IJCAI 2025](https://www.ijcai.org/proceedings/2025/0346.pdf)。

## 当前实现

- 兴趣晋升、S4→S3 保护降级、降级后的相关交互恢复；阈值采用近似梯度训练。
- 完整可见历史用于兴趣记忆，近期窗口仍输入原 FMRec 主干。
- 保留原流匹配目标和采样计算；状态不是心理学真值标签。
- 完整历史不包含当前预测标签；实验按验证集选择模型。
- 本地 CPU 稳定更新、参数搜索、128 维对照及用户子集快速实验。

## 文件入口

| 文件 | 作用 |
|---|---|
| `src/main.py` | 常规训练；支持配置文件和生命周期开关 |
| `src/interest_lifecycle.py` | 五态状态、交互间隔、恢复证据与兴趣记忆 |
| `src/fmrec.py` | FMRec 流匹配与 Euler 采样核心 |
| `src/tune_ml100k.py` | 验证集选参和检查点 |
| `src/compare_paper_ml100k.py` | 128 维、有／无生命周期的本地对照 |
| `src/quick_compare_ml100k.py` | 固定用户子集、等轮次、限定预算的快速筛查 |
| `src/inspect_interest.py` | 固定模型下重放用户兴趣状态轨迹 |
| `tests/` | 状态规则、梯度、评估和公平性测试 |

## 运行前准备

使用 Python 3.10，安装 `requirements.txt`，另外安装适合本机的 PyTorch。
本地验证环境使用 PyTorch 2.1.2。将已预处理数据放到
`datasets/data/<dataset>/dataset.pkl`；数据来源和处理方式见 `datasets/readme.md`。
本仓库不上传数据文件、虚拟环境、日志或模型权重。

```powershell
python src/main.py --dry_run
python src/main.py --dataset ml-100k --interest_lifecycle 1 --device cpu
python -m unittest discover -s tests -v
```

`ml100k_best_config.json` 保存的是一次有限搜索的配置，不代表全局最优。
其中日志目录为原开发机器的绝对路径；换机器后需设置 `--log_file` 指向可写目录。
关闭模块可用 `--interest_lifecycle 0`；显式命令行参数覆盖配置默认值。

## 结果解释与详细说明

- [生命周期实现说明](README-interest-lifecycle.md)
- [本地环境说明](README-local.md)
- [调参协议及数值稳定性](README-ml100k-tuning.md)
- [初步全量调参记录](ML100K_TUNING_RESULT.md)
- [128 维对照协议](PAPER_COMPARISON_128.md)

小样本、单种子和短程训练结果不能证明优于论文。此前较大配置的 GPU 运行
出现过数值异常，当前不能把单轮 GPU 能运行理解为多轮训练已经稳定。
