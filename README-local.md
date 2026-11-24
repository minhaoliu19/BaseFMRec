# FMRec local setup and run

> 2026-11-24 update: larger GPU training runs showed non-finite gradients and
> invalid Adam second moments on this workstation. See `README-ml100k-tuning.md`.
> A one-epoch GPU smoke test is not evidence of stable multi-epoch GPU training.
> When present, `ml100k_best_config.json` supplies the selected ML-100K defaults.

This directory contains the authors' official implementation of **FMRec**
(IJCAI 2025), plus the bundled preprocessed datasets.

## Environment

The local environment is `.venv` and uses Python 3.10 with PyTorch 2.1.2
and CUDA 12.1. It has been verified on the NVIDIA GeForce GTX 1650. Activate
it from PowerShell:

```powershell
Set-Location D:\CodexWorkSpace\FMRec
.\.venv\Scripts\Activate.ps1
```

## Run

Run from the `src` directory because the authors' program resolves datasets
relative to that directory. The following is a quick GPU validation run:

```powershell
Set-Location D:\CodexWorkSpace\FMRec\src
python main.py --dataset ml-100k --device cuda --epochs 1 --eval_interval 99 --batch_size 128
```

For GPU training, use `--device cuda`, for example:

```powershell
python main.py --dataset amazon_beauty --device cuda
```

Available bundled datasets: `amazon_beauty`, `ml-100k`, `steam`, and `yelp`.
Logs are written under `src/log/`.

The original source pinned GPU number 7 unconditionally. That line was
removed so the `--device` parameter works on a single-GPU computer. The
unused Matplotlib import was also removed so startup has no font-cache
permission dependency.
