import torch
import torch.nn.functional as F
from torch import nn

from ddecomposition.attention import OrdAttention, MixAttention



class TemporalTransformerBlock(nn.Module):
    def __init__(self, model_dim, ff_dim, atten_dim, head_num, dropout):
        super(TemporalTransformerBlock, self).__init__()
        self.attention = OrdAttention(model_dim, atten_dim, head_num, dropout, True)

        self.conv1 = nn.Conv1d(in_channels=model_dim, out_channels=ff_dim, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = self.attention(x, x, x)

        residual = x.clone()
        x = self.activation(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))

        return self.norm(x + residual)


class SpatialTransformerBlock(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, head_num, dropout):
        super(SpatialTransformerBlock, self).__init__()
        self.attention = OrdAttention(window_size, atten_dim, head_num, dropout, True)

        self.conv1 = nn.Conv1d(in_channels=model_dim, out_channels=ff_dim, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.attention(x, x, x)
        x = x.permute(0, 2, 1)

        residual = x.clone()
        x = self.activation(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))

        return self.norm(x + residual)


class SpatialTemporalTransformerBlock(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, head_num, dropout):
        super(SpatialTemporalTransformerBlock, self).__init__()
        self.time_block = TemporalTransformerBlock(model_dim, ff_dim, atten_dim, head_num, dropout)
        self.feature_block = SpatialTransformerBlock(window_size, model_dim, ff_dim, atten_dim, head_num, dropout)

        self.conv1 = nn.Conv1d(in_channels=2 * model_dim, out_channels=ff_dim, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

        self.norm1 = nn.LayerNorm(2 * model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        time_x = self.time_block(x)
        feature_x = self.feature_block(x)
        x = self.norm1(torch.cat([time_x, feature_x], dim=-1))

        x = self.activation(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))

        return self.norm2(x)


class DecompositionBlock(nn.Module):
    def __init__(self, model_dim, ff_dim, atten_dim, feature_num, head_num, dropout):
        super(DecompositionBlock, self).__init__()
        self.mixed_attention = MixAttention(model_dim, atten_dim, head_num, dropout, False)
        self.ordinary_attention = OrdAttention(model_dim, atten_dim, head_num, dropout, True)

        # FFN
        self.conv1 = nn.Conv1d(in_channels=model_dim, out_channels=ff_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=1)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")

        # 输出映射 [B,L,D] -> [B,L,F]
        self.fc1 = nn.Linear(model_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, feature_num)

        # 显式低通滤波
        self.lowpass = nn.Conv1d(in_channels=model_dim, out_channels=model_dim,
                                 kernel_size=5, padding=2, groups=model_dim, bias=False)
        nn.init.constant_(self.lowpass.weight, 1.0 / 5.0)

        # 门控机制
        self.gate = nn.Linear(model_dim, model_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, trend, time):
        """
        trend: [B, L, D]
        time:  [B, L, D]
        return:
            stable: [B, L, F]
            trend:  [B, L, D]
        """
        # -------- 稳定成分提取 --------
        stable = self.mixed_attention(trend, time, trend, time, time)  
        stable = self.ordinary_attention(stable, stable, stable)        

        residual = stable.clone()
        stable = self.activation(self.conv1(stable.permute(0, 2, 1)))   
        stable = self.dropout(self.conv2(stable).permute(0, 2, 1))      
        stable = self.norm1(stable + residual)

        # 显式低通滤波
        smooth = self.lowpass(stable.permute(0, 2, 1)).permute(0, 2, 1) 
        stable = 0.7 * stable + 0.3 * smooth

        # 趋势更新
        gate_val = torch.sigmoid(self.gate(trend))                      
        trend = self.norm2(trend - gate_val * stable)                   

        # 输出稳定成分到原始特征空间
        stable = self.fc2(self.activation(self.fc1(stable)))            

        return stable, trend
