"""
model.py
========

Architektura konvoluční sítě pro kolorizaci.

Použit je jednoduchý U-Net (Ronneberger et al., 2015) přizpůsobený rozlišení
32x32. U-Net je v úlohách image-to-image (segmentace, kolorizace) standardem,
protože:

* enkodér zachycuje sémantický kontext (jaký objekt to zhruba je),
* dekodér rekonstruuje prostorové detaily,
* skip-connections zachovávají lokální strukturu, která je pro pixelovou úlohu klíčová.

Vstup:  L kanál tvaru (B, 1, 32, 32)
Výstup: ab kanály tvaru (B, 2, 32, 32) v rozsahu (-1, 1) díky aktivaci tanh

Inicializace vah: Kaiming He (vhodná pro ReLU nonlinearity).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Dvojice Conv-BatchNorm-ReLU, základní stavební blok U-Netu."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ColorizationNet(nn.Module):
    """
    Mini U-Net se 3 stupni downsamplingu (32 -> 16 -> 8 -> 4) a symetrickým
    dekodérem se skip-connections.

    Pro `base_channels = 32` má cca 1.9 M parametrů.
    Pro `base_channels = 16` má cca 0.5 M parametrů (rychlejší trénink).
    """

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )

        # Enkodér
        self.enc1 = _conv_block(1, c1)            # 32x32
        self.enc2 = _conv_block(c1, c2)           # 16x16
        self.enc3 = _conv_block(c2, c3)           # 8x8
        self.bottleneck = _conv_block(c3, c4)     # 4x4

        self.pool = nn.MaxPool2d(2)

        # Dekodér: ConvTranspose pro upsampling, pak conv_block po concatenaci skip
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = _conv_block(c4, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = _conv_block(c3, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = _conv_block(c2, c1)

        # Výstupní vrstva: 2 kanály (a, b), aktivace tanh -> rozsah (-1, 1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(c1, 2, kernel_size=1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming He init pro Conv/ConvTranspose vrstvy, standard pro ReLU."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Enkodér
        e1 = self.enc1(x)                         # (B, c1, 32, 32)
        e2 = self.enc2(self.pool(e1))             # (B, c2, 16, 16)
        e3 = self.enc3(self.pool(e2))             # (B, c3,  8,  8)
        b = self.bottleneck(self.pool(e3))        # (B, c4,  4,  4)

        # Dekodér se skip-connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)                  # (B, 2, 32, 32)

    def count_parameters(self) -> int:
        """Vrátí počet trénovatelných parametrů."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
