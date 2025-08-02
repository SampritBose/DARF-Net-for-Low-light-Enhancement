# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision
from torchvision import models
import numpy as np
from typing import Tuple

# Optional utility imports (for logging, reproducibility)
import os
import random


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class Decomposition(nn.Module):
    """
    Decomposes an image into reflectance and illumination maps based on Retinex theory.
    """
    def __init__(self):
        super(Decomposition, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 3, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        R = self.decoder(feat)
        L = x / (R + 1e-6)  # Element-wise division
        return R, L




class IGMHA(nn.Module):
    """
    Illumination-Guided Multi-Head Attention module.
    """
    def __init__(self, dim: int, heads: int = 4):
        super(IGMHA, self).__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_q = nn.Conv2d(dim, dim, 1)
        self.to_k = nn.Conv2d(dim, dim, 1)
        self.to_v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor, illum: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Use input + illumination map to guide query/key projections
        q = self.to_q(x + illum).reshape(B, self.heads, C // self.heads, -1)
        k = self.to_k(x + illum).reshape(B, self.heads, C // self.heads, -1)
        v = self.to_v(x).reshape(B, self.heads, C // self.heads, -1)

        q = q.permute(0, 1, 3, 2)  # B, heads, HW, dim
        k = k.permute(0, 1, 2, 3)  # B, heads, dim, HW
        attn = torch.matmul(q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        out = out.reshape(B, C, H, W)
        return self.proj(out)




class IlluminationEnhancer(nn.Module):
    def __init__(self, in_channels=1):
        super(IlluminationEnhancer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU()
        )
        self.ig_attention = IGMHA(dim=64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1), nn.Sigmoid()
        )

    def forward(self, illum):
        feat = self.encoder(illum)
        attn_feat = self.ig_attention(feat, F.interpolate(illum, size=feat.shape[2:]))
        out = self.decoder(attn_feat)
        return out



class SAM(nn.Module):
    def __init__(self, channels):
        super(SAM, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn




class ReflectanceEnhancer(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):
        super(ReflectanceEnhancer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128 * 64 * 64, latent_dim)
        self.fc_logvar = nn.Linear(128 * 64 * 64, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, 128 * 64 * 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid()
        )

        self.attn = SAM(128)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc = self.encoder(x)
        attn_feat = self.attn(enc)
        flat = self.flatten(attn_feat)

        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        z = self.reparameterize(mu, logvar)

        dec = self.fc_decode(z).view(-1, 128, 64, 64)
        out = self.decoder(dec)
        return out, mu, logvar



class DARFNet(nn.Module):
    """
    Full DARF-Net architecture that processes input image and reconstructs enhanced output.
    """
    def __init__(self):
        super(DARFNet, self).__init__()
        self.decomp = Decomposition()
        self.illum_enh = IlluminationEnhancer()
        self.reflect_enh = ReflectanceEnhancer()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        R, L = self.decomp(x)
        L_input = torch.mean(L, dim=1, keepdim=True)
        L_enh = self.illum_enh(L_input)

        R_enh, mu, logvar = self.reflect_enh(R)
        final_out = R_enh * L_enh
        return final_out, R_enh, L_enh, mu, logvar
