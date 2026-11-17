# BaseFMRec

基于 [FMRec 官方实现](https://github.com/FengLiu-1/FMRec) 的复现与二次开发仓库。

原论文：Flow Matching Based Sequential Recommender Model（IJCAI 2025）。
基础源码版本：`1c39b17e59a08980c8372e4f861167e0a259c349`。

## 环境安装

已验证 Python 3.13、PyTorch 2.9.1+cu128、RTX 4090。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe verify_gpu.py
```

## 训练

```powershell
.\run_gpu.ps1
.\run_gpu.ps1 -Dataset ml-100k -Epochs 2 -EvalInterval 1
```

默认实验为 amazon_beauty，最多 500 轮，每 20 轮验证一次。
也支持 ml-100k、steam、yelp；预处理数据位于 `datasets/data/`。
模型、参数、指标和日志保存到 `runs/<dataset>/<dataset>/`。

## 本仓库修改

- 默认使用 GPU 0，数据路径相对于源码定位。
- 移除未使用的第三方导入。
- 修复短训练时 best_model 未初始化的问题，并在测试前进入 eval 模式。
- 训练结束保存 best_model.pt 和 results.json。
- 补充 GPU 检查、启动脚本和依赖版本清单。
- 排除本地环境、缓存、训练日志和模型产物。

MovieLens 两轮 GPU 训练、验证和测试已跑通；此检查不代表论文最终指标复现。
详细本机运行记录见 [GPU_SETUP.md](GPU_SETUP.md)。
