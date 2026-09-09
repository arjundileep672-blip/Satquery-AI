"""
SatQuery AI - Bundled ChangeFormer V6 Architecture

Minimal ChangeFormer V6 implementation compatible with pretrained LEVIR-CD weights.

Source: https://github.com/wgcban/ChangeFormer
Paper: "A Transformer-Based Siamese Network for Change Detection"
       Bandara & Patel, IGARSS 2022

Inputs:  Two 3-channel RGB images (T1, T2) — same spatial size
Output:  Segmentation logits [B, 2, H, W] (channel 0 = unchanged, 1 = changed)

The output is NOT a calibrated probability.
Use torch.softmax(pred, dim=1)[:, 1] to get predicted change scores [0, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class _MixFFN(nn.Module):
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.Linear(dim, hidden)
        self.dw  = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        B, N, C = x.shape
        x2 = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x2 = self.act(self.dw(x2))
        x2 = rearrange(x2, 'b c h w -> b (h w) c')
        x = x + x2
        return self.fc2(x)


class _EfficientSelfAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int = 1, sr_ratio: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q  = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr   = nn.Conv2d(dim, dim, sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
            x_ = self.sr(x_)
            x_ = rearrange(x_, 'b c h w -> b (h w) c')
            x_ = self.norm(x_)
        else:
            x_ = x
        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = _EfficientSelfAttn(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = _MixFFN(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x


class _PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, patch_size: int, stride: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x, H, W


class _MixTransformerEncoder(nn.Module):
    """Lightweight Mix-Transformer encoder (MiT-B0 scale)."""

    EMBED = [32, 64, 160, 256]
    HEADS = [1, 2, 5, 8]
    SR    = [8, 4, 2, 1]
    DEPTH = [2, 2, 2, 2]

    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.patch_embeds = nn.ModuleList()
        self.blocks       = nn.ModuleList()
        self.norms        = nn.ModuleList()
        in_dim = in_ch
        for i, out_dim in enumerate(self.EMBED):
            ps = 7 if i == 0 else 3
            st = 4 if i == 0 else 2
            self.patch_embeds.append(_PatchEmbed(in_dim, out_dim, ps, st))
            self.blocks.append(nn.ModuleList([
                _TransformerBlock(out_dim, self.HEADS[i], self.SR[i])
                for _ in range(self.DEPTH[i])
            ]))
            self.norms.append(nn.LayerNorm(out_dim))
            in_dim = out_dim

    def forward(self, x: torch.Tensor):
        outs = []
        for embed, blks, norm in zip(self.patch_embeds, self.blocks, self.norms):
            x, H, W = embed(x)
            for blk in blks:
                x = blk(x, H, W)
            x = norm(x)
            x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
            outs.append(x)
        return outs


class _ChangeDecoder(nn.Module):
    """Lightweight decoder: difference features -> segmentation map."""

    def __init__(self, embed_dims, num_classes: int = 2, decoder_dim: int = 256):
        super().__init__()
        self.fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d * 2, decoder_dim, 1, bias=False),
                nn.BatchNorm2d(decoder_dim),
                nn.ReLU(inplace=True),
            )
            for d in embed_dims
        ])
        self.predict = nn.Conv2d(decoder_dim * len(embed_dims), num_classes, 1)

    def forward(self, f1_list, f2_list, target_size):
        feats = []
        ref_size = f1_list[0].shape[2:]
        for f1, f2, fuse in zip(f1_list, f2_list, self.fuse):
            diff = torch.abs(f1 - f2)
            cat  = torch.cat([f1, diff], dim=1)
            out  = fuse(cat)
            out  = F.interpolate(out, size=ref_size, mode="bilinear", align_corners=False)
            feats.append(out)
        x = self.predict(torch.cat(feats, dim=1))
        return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)


class ChangeFormerV6(nn.Module):
    """
    ChangeFormer V6 — Siamese Transformer for bi-temporal change detection.

    Usage:
        model = ChangeFormerV6()
        pred = model(img_t1, img_t2)        # [B, 2, H, W]
        change_score = torch.softmax(pred, dim=1)[:, 1]  # [B, H, W]

    Output interpretation:
        - pred[:, 0]: unchanged score
        - pred[:, 1]: change score (NOT a calibrated probability)
    """

    def __init__(self, in_ch: int = 3, num_classes: int = 2):
        super().__init__()
        self.encoder = _MixTransformerEncoder(in_ch)
        self.decoder = _ChangeDecoder(
            embed_dims=_MixTransformerEncoder.EMBED,
            num_classes=num_classes,
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        H, W = x1.shape[2], x1.shape[3]
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        return self.decoder(f1, f2, (H, W))
