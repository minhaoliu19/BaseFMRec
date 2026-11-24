"""CPU Adam updates with GPU forward/backward on this workstation."""
# 中文说明：这是运行适配，不改变 FMRec 损失或兴趣策略。
# 参数镜像和 Adam 二阶矩保存在 CPU；模型本身也可以完全运行在 CPU。
import torch


class CPUAdam(torch.optim.Adam):
    def __init__(self,parameters,lr=.001,weight_decay=0.,clip_value=1.):
        self.live_parameters=list(parameters)
        self.cpu_parameters=[torch.nn.Parameter(p.detach().cpu().clone()) for p in self.live_parameters]
        self.clip_value=clip_value
        super().__init__(self.cpu_parameters,lr=lr,weight_decay=weight_decay,foreach=False)

    def zero_grad(self,set_to_none=True):
        for p in self.live_parameters:
            if set_to_none:
                p.grad=None
            elif p.grad is not None:
                p.grad.zero_()
        super().zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self,closure=None):
        # 先检查所有梯度，任何一项非有限都在更新前失败，不替换成零或跳过样本。
        if closure is not None:
            raise ValueError('Closures unsupported')
        for live,cpu in zip(self.live_parameters,self.cpu_parameters):
            cpu.grad=None if live.grad is None else live.grad.detach().to('cpu',copy=True)
            if cpu.grad is not None:
                if not torch.isfinite(cpu.grad).all():
                    raise FloatingPointError('Non-finite gradient before CPU Adam; no update applied')
                if self.clip_value>0:
                    cpu.grad.clamp_(-self.clip_value,self.clip_value)
        result=super().step()
        # 更新完成后同步回实际模型；原生 Adam 的状态可继续用于断点恢复。
        for live,cpu in zip(self.live_parameters,self.cpu_parameters):
            if not torch.isfinite(cpu).all():
                raise FloatingPointError('Non-finite CPU Adam weight')
            live.copy_(cpu.to(live.device))
        return result

    def load_state_dict(self,state_dict):
        super().load_state_dict(state_dict)
        for live,cpu in zip(self.live_parameters,self.cpu_parameters):
            cpu.data.copy_(live.detach().cpu())
        for group in self.param_groups:
            group['foreach']=False
