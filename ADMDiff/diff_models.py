# diff_models.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer

class _SafeOps:
    @staticmethod
    def clamp_nan(x, minv=-1e4, maxv=1e4):
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        return torch.clamp(x, min=minv, max=maxv)


class SEModule(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, L)
        b, c, l = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class LinearSelfAttention(nn.Module):
    def __init__(self, channels, nheads, dropout=0.0):
        super().__init__()
        assert channels % nheads == 0
        self.channels = channels
        self.nheads = nheads
        self.head_dim = channels // nheads

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)

        # init
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _phi(x):

        return F.elu(x) + 1.0

    def forward(self, x):
        """
        x: (B, S, C)
        returns: (B, S, C)
        """
        B, S, C = x.shape
        x = _SafeOps.clamp_nan(x)

        q = self.q_proj(x).view(B, S, self.nheads, self.head_dim).transpose(1, 2)  
        k = self.k_proj(x).view(B, S, self.nheads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.nheads, self.head_dim).transpose(1, 2)

        qf = self._phi(q)  
        kf = self._phi(k)  

    
        kv = torch.einsum('bhsd,bhse->bhde', kf, v)  
        kf_sum = kf.sum(dim=2)  

   
        out_raw = torch.einsum('bhsd,bhde->bhse', qf, kv)

       
        z = torch.einsum('bhsd,bhd->bhs', qf, kf_sum).clamp(min=1e-6)  

        out = out_raw / z.unsqueeze(-1)  
        out = out.transpose(1, 2).contiguous().view(B, S, C)  
        out = self.out_proj(self.dropout(out))
        out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, channels, mult=4, dropout=0.0):
        super().__init__()
        inner = channels * mult
        self.net = nn.Sequential(
            nn.Linear(channels, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, channels),
            nn.Dropout(dropout)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = _SafeOps.clamp_nan(x)
        return self.net(x)


class EfficientTransformerBlock(nn.Module):
    def __init__(self, channels, nheads, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = LinearSelfAttention(channels, nheads, dropout=dropout)
        self.norm2 = nn.LayerNorm(channels)
        self.ff = FeedForward(channels, mult=4, dropout=dropout)

    def forward(self, x):
        x = _SafeOps.clamp_nan(x)
        h = self.norm1(x)
        a = self.attn(h)
        x = x + a
        x = x + self.ff(self.norm2(x))
        x = _SafeOps.clamp_nan(x)
        return x


class EfficientTimeTransformer(nn.Module):
    """时间轴 Transformer：接收 (B*K, C, L) 并返回 (B*K, C, L)"""
    def __init__(self, channels, nheads, depth=1, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([EfficientTransformerBlock(channels, nheads, dropout=dropout) for _ in range(depth)])

    def forward(self, x):  
        BK, C, L = x.shape
        if L == 1:
            return x
        x = x.transpose(1, 2).contiguous()  # (B*K, L, C)
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(1, 2).contiguous()  # (B*K, C, L)
        return x

class EfficientFeatureTransformer(nn.Module):
    """特征轴 Transformer：接收 (B*L, C, K) 并返回 (B*L, C, K)"""
    def __init__(self, channels, nheads, depth=1, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([EfficientTransformerBlock(channels, nheads, dropout=dropout) for _ in range(depth)])

    def forward(self, x):  # x: (B*L, C, K)
        BL, C, K = x.shape
        if K == 1:
            return x
        x = x.transpose(1, 2).contiguous()  
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(1, 2).contiguous()  # (B*L, C, K)
        return x



class MultiScaleTemporalConv(nn.Module):
    def __init__(self, channels, kernel_sizes=[3, 5, 7], dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, k, padding=k // 2) for k in kernel_sizes
        ])
        self.proj = nn.Conv1d(channels * len(kernel_sizes), channels, 1)
        self.dropout = nn.Dropout(dropout)

        # init
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        nn.init.kaiming_normal_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        """
        x: (B, C, L)
        returns: (B, C, L)
        """
        outs = []
        for conv in self.convs:
            outs.append(conv(x))
        out = torch.cat(outs, dim=1) 
        out = self.proj(out)
        return self.dropout(out)


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads,
                 time_depth=1, feat_depth=1, use_mstc=False):
        super().__init__()
        self.channels = channels
        self.use_mstc = use_mstc   

        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.strategy_projection  = nn.Linear(diffusion_embedding_dim, channels)

        # projections
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection  = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection= Conv1d_with_init(2 * channels, 2 * channels, 1)

        # efficient transformers
        self.time_layer = EfficientTimeTransformer(channels, nheads, depth=time_depth, dropout=0.1)
        self.feature_layer = EfficientFeatureTransformer(channels, nheads, depth=feat_depth, dropout=0.1)

    
        if use_mstc:
            self.mstc = MultiScaleTemporalConv(channels, kernel_sizes=[3, 5, 7], dropout=0.1)

        self.se_mid = SEModule(2 * channels)
        self.cond_gate = nn.Sequential(nn.Linear(2 * channels, 2 * channels, bias=False), nn.Sigmoid())
        self.fusion_layernorm = nn.LayerNorm(2 * channels)
        self.skip_norm = nn.GroupNorm(1, channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y_ = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        out = self.time_layer(y_)  

       
        if self.use_mstc:
            out = self.mstc(out)  

        out = out.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return out

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y_ = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        out = self.feature_layer(y_)
        out = out.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return out

    def forward(self, x, cond_info, diffusion_emb, strategy_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x_flat = x.reshape(B, channel, K * L)

        diffusion_proj = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        strategy_proj  = self.strategy_projection(strategy_emb).unsqueeze(-1)

        y = x_flat + diffusion_proj + strategy_proj

        temporal_out = self.forward_time(y, base_shape)    
        feature_out  = self.forward_feature(temporal_out, base_shape)  

       
        y_fused = feature_out

        # mid projection
        y_mid = self.mid_projection(y_fused)  
        y_mid = self.se_mid(y_mid)

        cond = cond_info.reshape(B, cond_info.size(1), K * L)
        cond = self.cond_projection(cond)  

        y_pooled = F.adaptive_avg_pool1d(y_mid, 1).squeeze(-1)     
        cond_pooled = F.adaptive_avg_pool1d(cond, 1).squeeze(-1)
        fusion_input = self.fusion_layernorm(y_pooled + cond_pooled)
        fusion_gate = self.cond_gate(fusion_input).unsqueeze(-1)  
        y_mid = y_mid + fusion_gate * cond

        # output projection
        y_out = self.output_projection(y_mid)  
        residual, skip = torch.chunk(y_out, 2, dim=1)  

        residual = residual.reshape(B, channel, K, L)
        skip = skip.reshape(B, channel, K, L)

        skip_normalized = self.skip_norm(skip)
        out = (x + residual) / math.sqrt(2.0)

        return out, skip_normalized



class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1).float()
        frequencies = 10.0 ** (torch.arange(dim).float() / max(1, dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table  # (T, dim*2)


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.strategy_embedding = nn.Embedding(2, config['diffusion_embedding_dim'])

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.output_se = SEModule(self.channels)

        # build residual blocks
        layers = config["layers"]
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                side_dim=config["side_dim"],
                channels=self.channels,
                diffusion_embedding_dim=config["diffusion_embedding_dim"],
                nheads=config["nheads"]
            )
            for _ in range(layers)
        ])

    
        self.register_parameter("skip_weights", nn.Parameter(torch.zeros(layers)))

    def forward(self, x, cond_info, diffusion_step, strategy_type):
        """
        x: (B, inputdim, K, L)
        cond_info: (B, cond_dim, K, L)
        diffusion_step: indices (B,) or scalar index
        strategy_type: (B,)
        """
        B, inputdim, K, L = x.shape

        x_flat = x.reshape(B, inputdim, K * L)
        xp = self.input_projection(x_flat)  # (B, channels, K*L)
        xp = F.relu(xp).reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)
        strategy_emb = self.strategy_embedding(strategy_type)

       
        skip_sum = None
        cur = xp

        weights = torch.softmax(self.skip_weights.to(xp.device), dim=0) 

        for i, layer in enumerate(self.residual_layers):
            cur, skip_conn = layer(cur, cond_info, diffusion_emb, strategy_emb)
            skip_conn = torch.clamp(skip_conn, -10.0, 10.0)
            w = weights[i]
            if skip_sum is None:
                skip_sum = skip_conn * w
            else:
                skip_sum = skip_sum + skip_conn * w
           
            del skip_conn

        if skip_sum is None:
            agg = torch.zeros_like(cur)
        else:
           
            agg = skip_sum

        agg_flat = agg.reshape(B, self.channels, K * L)
        if torch.isnan(agg_flat).any() or torch.isinf(agg_flat).any():
            agg_flat = torch.clamp(agg_flat, -5.0, 5.0)

        agg_flat = self.output_se(agg_flat)
        out = self.output_projection1(agg_flat)
        out = F.relu(out)
        out = self.output_projection2(out)  
        out = out.reshape(B, K, L)

        out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        return out


