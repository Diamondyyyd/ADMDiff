import torch
from torch import nn
import torch.nn.functional as F


class OffsetSubtraction(nn.Module):
    def __init__(self, window_size, feature_num, d, tau=0.1, alpha=0.5):
        super(OffsetSubtraction, self).__init__()
        self.d = d
        self.tau = tau
        self.alpha = alpha

        # 原始时间索引 (考虑 pad 后的长度)
        init_index = torch.arange(window_size).unsqueeze(-1).unsqueeze(-1) + d
        init_index = init_index.repeat(1, feature_num, 2 * d + 1)

        # 构造延迟
        delay = torch.tensor(
            [0] + [i for i in range(1, d + 1)] + [-i for i in range(1, d + 1)],
            dtype=torch.long
        )
        delay = delay.unsqueeze(0).unsqueeze(0).repeat(window_size, feature_num, 1)

        # padding 后索引范围 [0, window_size + 2d - 1]
        self.index = init_index + delay  

    def forward(self, subed, sub):
        batch_size = subed.shape[0]
        device = sub.device

        # pad，保证索引合法
        sub = F.pad(sub, (0, 0, self.d, self.d), mode="reflect")

        index = self.index.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)

        # gather
        sub = torch.gather(
            sub.unsqueeze(-1).repeat(1, 1, 1, 2 * self.d + 1), 
            dim=1, 
            index=index
        )

        res = subed.unsqueeze(-1).repeat(1, 1, 1, 2 * self.d + 1) - sub
        diff = self.alpha * torch.abs(res) + (1 - self.alpha) * (res ** 2)

        weights = F.softmax(-diff / self.tau, dim=-1)
        res = (res * weights).sum(dim=-1)

        return res
