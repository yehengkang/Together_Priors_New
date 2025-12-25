import torch
import torch.nn as nn
import torch.nn.functional as F

# 如果你已有 Conv 实现，可以替换；这里给出一个简单的 1x1 conv 作为通道变换占位
class Conv1x1(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class TransformerLayer(nn.Module):
    """
    单层 Transformer（Self-Attention + MLP），符合标准实现。
    输入/输出 embedding 维度为 `c`，使用 residual+LayerNorm。
    """
    def __init__(self, c, num_heads, mlp_ratio=4.0, dropout=0.0, attn_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads, dropout=attn_dropout)
        self.norm2 = nn.LayerNorm(c)

        hidden_dim = int(c * mlp_ratio)
        self.fc1 = nn.Linear(c, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, c)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        x: (seq_len, batch, c)  -- 注意：这里遵循 nn.MultiheadAttention 的默认格式
        """
        # --- Self-Attention block ---
        x_res = x
        x_norm = self.norm1(x.permute(1,0,2)).permute(1,0,2)  # LayerNorm expects (batch, seq, dim)
        # x_norm shape -> (seq_len, batch, c)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm)  # 返回 (attn_output, attn_weights)
        # 使用 attn_out，并显式接收 attn_weights 以避免 DDP 中出现 unused params 问题
        x = x_res + self.dropout(attn_out)

        # --- MLP block ---
        x_res2 = x
        x_norm2 = self.norm2(x.permute(1,0,2)).permute(1,0,2)
        # MLP 在 (seq_len, batch, c) 上逐 token 处理：先转换到 (seq_len*batch, c) 形式更高效也清晰
        # 但这里直接在最后一维上作用也可以
        mlp_out = self.fc2(self.dropout(self.activation(self.fc1(x_norm2))))
        x = x_res2 + self.dropout(mlp_out)

        return x  # (seq_len, batch, c)

class TransformerBlock(nn.Module):
    """
    将 2D 特征 (B, C, H, W) 转为 token 序列，添加可学习位置编码，经过若干 ViTTransformerLayer，再还原回 (B,C,H,W)。
    """
    def __init__(self, c1, c2, num_heads, num_layers, use_conv_proj=True, dropout=0.1):
        super().__init__()
        self.conv = Conv1x1(c1, c2) if (use_conv_proj and c1 != c2) else (nn.Identity() if c1 == c2 else Conv1x1(c1, c2))
        self.c2 = c2
        self.num_layers = num_layers

        # 位置编码：按序列长度（H*W）学习，固定的输入尺寸H*W=400
        N = 400
        # 位置编码维度为 (1, seq_len, c2)，在 forward 时会广播到 batch
        self.pos_embed = nn.Parameter(torch.zeros(1, N, self.c2))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer 层堆叠
        self.layers = nn.ModuleList([TransformerLayer(c2, num_heads, dropout=dropout) for _ in range(num_layers)])

    def forward(self, x):
        """
        x: (B, C_in, H, W)
        returns: (B, C_out, H, W)
        """
        B, C, H, W = x.shape
        x = self.conv(x)  # (B, c2, H, W)

        # 转成 (B, N, C) 形式的 token 序列，其中 N = H*W
        N = H * W
        # flatten -> (B, c2, N) then permute -> (B, N, c2)
        tokens = x.flatten(2).permute(0, 2, 1)  # (B, N, c2)

        tokens = tokens + self.pos_embed  # (B, N, c2)
        tokens = self.pos_dropout(tokens)

        # MultiheadAttention expects (seq_len, batch, embed_dim)
        tokens = tokens.permute(1, 0, 2)  # (N, B, c2)

        # 通过多层 TransformerLayer
        for layer in self.layers:
            tokens = layer(tokens)  # 仍是 (N, B, c2)

        # 恢复回 (B, c2, H, W)
        tokens = tokens.permute(1, 0, 2)  # (B, N, c2)
        out = tokens.permute(0, 2, 1).reshape(B, self.c2, H, W)  # (B, c2, H, W)

        return out
