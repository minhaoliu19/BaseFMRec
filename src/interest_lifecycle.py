"""Interaction-index S0--S4 lifecycle (no wall-clock timestamps required).

Replay each *observed prefix*, rather than mutating a global user cache. This
supports shuffled minibatches and prevents validation/test target leakage.
Hard routing/transitions are used in the forward pass; smooth surrogate gates
provide gradients for routing, promotion, inactivity and reactivation thresholds.
These are modelling assumptions, not labelled psychological interest states.

中文说明：S0=无证据，S1=触发，S2=维持，S3=中期或成熟后降级，S4=成熟活跃。
这里按“其间发生多少次其他兴趣交互”计算衰退，不能解释为经过多少天。
硬路由和五态用于前向计算，softmax/sigmoid 仅提供反向代理梯度。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def inverse_softplus(value):
    return math.log(math.expm1(value))


class InteractionLifecycle(nn.Module):
    def __init__(self, hidden_size, num_interests=8, promote=(2., 4., 8.),
                 degrade_gap=20., repromote=3., temperature=0.5):
        super().__init__()
        if num_interests < 2 or temperature <= 0:
            raise ValueError('num_interests >= 2 and temperature > 0 are required')
        if not (1 < promote[0] < promote[1] < promote[2]):
            raise ValueError('Promotion thresholds must satisfy 1 < S2 < S3 < S4')
        if degrade_gap < 1 or repromote < 1:
            raise ValueError('Gap and reactivation thresholds must be >= 1')
        self.num_interests = num_interests
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.empty(num_interests, hidden_size))
        nn.init.normal_(self.prototypes, std=hidden_size ** -0.5)
        increments = (promote[0] - 1, promote[1] - promote[0], promote[2] - promote[1])
        self.raw_promote = nn.Parameter(torch.tensor([inverse_softplus(x) for x in increments]))
        self.raw_gap = nn.Parameter(torch.tensor(inverse_softplus(max(degrade_gap - 1, 1e-4))))
        self.raw_repromote = nn.Parameter(torch.tensor(inverse_softplus(max(repromote - 1, 1e-4))))
        self.raw_weight_increments = nn.Parameter(torch.tensor(
            [inverse_softplus(x) for x in (0.15, 0.20, 0.30, 0.35)]))
        self.context_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.context_projection.weight)
        # A small, nonzero residual makes the new module trainable immediately.
        self.residual_logit = nn.Parameter(torch.tensor(-2.0))
        self.last_stats = None

    def thresholds(self):
        # softplus 正增量确保晋升阈值有序；当前是全局参数，不是每用户独立阈值。
        promotion = 1 + F.softplus(self.raw_promote).cumsum(0)
        return promotion, 1 + F.softplus(self.raw_gap), 1 + F.softplus(self.raw_repromote)

    def state_weights(self):
        # 等级基础权重有序；S0=0、S4=1，其余权重可以学习。
        weights = F.softplus(self.raw_weight_increments).cumsum(0)
        return torch.cat((weights.new_zeros(1), weights / weights[-1]))

    def state_from_assignments(self, assignments, valid):
        """Vectorized replay of hard, one-hot semantic interactions [B,H,K].

        At an occurrence following >= G unrelated events, reset recovery evidence
        if the interest was already mature. Each subsequent related event counts
        toward N. Another long gap resets it again. Mature memory never falls
        below S3. Counts and gaps are based on unpadded positions, not max_len.
        """
        promote, degrade, repromote = self.thresholds()
        hits = assignments * valid.unsqueeze(-1)
        counts = hits.cumsum(1)
        total = counts[:, -1]
        # positions 从 1 起算；补零位置由 valid 屏蔽，不贡献交互计数。
        positions = torch.arange(1, hits.size(1) + 1, device=hits.device).view(1, -1, 1)
        occurred = hits.detach() > 0.5
        latest = torch.where(occurred, positions, 0).cummax(1).values
        previous = F.pad(latest[:, :-1], (0, 0, 1, 0))
        between = positions - previous - 1
        before = counts - hits
        resets = (occurred & (previous > 0) & (between >= degrade.detach())
                  & (before.detach() >= promote[-1].detach()))
        # 恢复期间再次出现长间隔会重置证据；不把很久以前的回访次数反复使用。
        reset_count = torch.where(resets, before, 0).amax(1)
        recovery_count = total - reset_count
        last_position = valid.sum(1, keepdim=True)
        gap = (last_position - latest[:, -1]).clamp_min(0).to(hits.dtype)
        seen = total.detach() > 0
        ever_mature = total.detach() >= promote[-1].detach()
        # “曾成熟”由当前参数重放此前缀推断，并非跨 epoch 永久锁定的外部标签。
        recovery_needed = resets.any(1)
        active = ((gap < degrade.detach())
                  & (~recovery_needed | (recovery_count.detach() >= repromote.detach())))

        hard_state = seen.long()
        for boundary in promote.detach():
            hard_state = hard_state + (total.detach() >= boundary).long()
        hard_state = torch.where(ever_mature & ~active, 3, hard_state)
        # 保护规则：成熟兴趣不活跃时最多退到 S3，历史表示仍然保留。

        # Smooth gates in backward, exact discrete state/floor in forward.
        t = self.temperature
        g1 = torch.sigmoid((total - 0.5) / t) * seen
        g2, g3, g4 = [torch.sigmoid((total - b) / t) * seen for b in promote]
        active_soft = torch.sigmoid((degrade - gap) / t)
        recovery_soft = torch.sigmoid((recovery_count - repromote + 0.5) / t)
        active_soft = active_soft * torch.where(recovery_needed, recovery_soft, 1.)
        soft = torch.stack((1 - g1, g1 - g2, g2 - g3,
                            g3 - g4 + g4 * (1 - active_soft), g4 * active_soft), -1)
        hard = F.one_hot(hard_state, 5).to(hits.dtype)
        state_prob = hard + soft - soft.detach()
        # 数值前向等于 hard；反向沿 soft 计算，不是离散状态机的精确梯度。
        return state_prob, {
            'states': hard_state.detach(), 'counts': total.detach(),
            'gaps': gap.detach(), 'ever_mature': ever_mature,
            'recovery_count': recovery_count.detach(),
            'promote': promote.detach(), 'degrade_gap': degrade.detach(),
            'repromote': repromote.detach(),
        }

    def forward(self, sequence_rep, sequence_ids, history_ids, embedding):
        valid = history_ids.ne(0)
        history_rep = embedding(history_ids)
        normalized_prototypes = F.normalize(self.prototypes, dim=-1)
        logits = F.linear(F.normalize(history_rep, dim=-1), normalized_prototypes)
        soft_assignment = F.softmax(logits / self.temperature, dim=-1)
        hard_assignment = F.one_hot(soft_assignment.argmax(-1), self.num_interests).to(history_rep.dtype)
        assignment = (hard_assignment + soft_assignment - soft_assignment.detach()) * valid.unsqueeze(-1)
        probabilities, stats = self.state_from_assignments(assignment, valid)
        weights = probabilities @ self.state_weights()
        mass = assignment.sum(1)
        memory = torch.einsum('bhk,bhd->bkd', assignment, history_rep) / mass.clamp_min(1).unsqueeze(-1)
        # 完整历史按潜在兴趣聚合成均值记忆；这里没有使用人工电影类型标签。
        context = (weights.unsqueeze(-1) * memory).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-6)
        local_logits = F.linear(F.normalize(embedding(sequence_ids), dim=-1), normalized_prototypes)
        local_assignment = F.softmax(local_logits / self.temperature, dim=-1)
        local_weight = (local_assignment * weights.unsqueeze(1)).sum(-1)
        strength = self.residual_logit.sigmoid()
        # 小幅残差注入，不直接把等级当作最终候选物品得分。
        # Apply after LayerNorm: a scalar multiplier before normalization cancels.
        enriched = sequence_rep * (1 + strength * local_weight.unsqueeze(-1))
        enriched = enriched + strength * self.context_projection(context).unsqueeze(1) * sequence_ids.ne(0).unsqueeze(
            -1)
        self.last_stats = {**stats, 'weights': weights.detach(),
                           'state_weights': self.state_weights().detach()}
        return enriched

    def summary(self):
        if self.last_stats is None:
            return {}
        stats = self.last_stats
        states = stats['states']
        return {
            'state_counts': {f'S{s}': int((states == s).sum().item()) for s in range(5)},
            'promotion_thresholds': stats['promote'].cpu().tolist(),
            'degrade_gap': float(stats['degrade_gap'].item()),
            'repromote_threshold': float(stats['repromote'].item()),
            'state_weights': stats['state_weights'].cpu().tolist(),
        }
