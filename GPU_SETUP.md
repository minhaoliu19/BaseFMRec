# FMRec GPU 环境

项目目录：E:\lmh\FMRec
Python：.venv\Scripts\python.exe（Python 3.13）
PyTorch：2.9.1+cu128
Torchvision：0.24.1+cu128
CUDA runtime：12.8（随 PyTorch 安装，无需另外安装 CUDA Toolkit）
GPU：NVIDIA GeForce RTX 4090

## 启动

在 PowerShell 中运行：

```powershell
cd E:\lmh\FMRec
.\run_gpu.ps1
```

默认使用 amazon_beauty，最多 500 轮，每 20 轮评估，沿用官方默认超参数。
可选数据集：amazon_beauty、ml-100k、steam、yelp。

```powershell
.\run_gpu.ps1 -Dataset ml-100k -Epochs 2 -EvalInterval 1
.\.venv\Scripts\python.exe .\verify_gpu.py
```

启动脚本从自身位置定位项目，设置 CUDA_VISIBLE_DEVICES=0，固定使用项目虚拟环境。
训练日志、best_model.pt 和 results.json 保存到 runs 下。
由于原程序会自动追加数据集名，默认输出路径为 runs/amazon_beauty/amazon_beauty/。

## 已做的兼容性调整

- 原 main.py 将 GPU 编号写死为 7，改为默认 0。
- 数据路径以 main.py 所在目录定位。
- 删除未使用的 timm 和 matplotlib 导入。
- 初始化 best_model，防止短训练无评估时变量未定义；测试前显式切换 eval 模式。
- 训练结束保存模型权重、运行参数与测试指标。

GPU 矩阵计算和反向传播已通过。MovieLens 两轮训练用于验证完整流程，不能代表论文收敛结果。
依赖版本保存在 requirements-lock.txt；GPU 检查结果保存在 gpu_verification.json。

官方源码：https://github.com/FengLiu-1/FMRec
源码版本：1c39b17e59a08980c8372e4f861167e0a259c349

MovieLens 两轮 GPU 验证已完整完成（退出码 0），模型与指标已保存到 runs/gpu-check/ml-100k/。测试 HR@10=3.3026%，NDCG@10=1.5812%。这些数值仅用于环境验证。
