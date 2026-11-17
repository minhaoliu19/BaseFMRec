import json
from pathlib import Path
import torch
import torchvision
assert torch.cuda.is_available(), 'CUDA is unavailable'
x = torch.randn(1024, 1024, device='cuda', requires_grad=True)
loss = (x @ x.T).square().mean()
loss.backward()
torch.cuda.synchronize()
result = {'torch': torch.__version__, 'torchvision': torchvision.__version__, 'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'forward_backward': 'passed'}
print(json.dumps(result, indent=2))
Path(__file__).with_name('gpu_verification.json').write_text(json.dumps(result, indent=2))
