"""
metrics.py
==========

Metriky pro kvantitativní hodnocení kolorizace.

MSE (Mean Squared Error)
    Reportovaná na dvou úrovních:
    1. **ab-MSE** – přímo na normalizovaných ab-kanálech. Tuto hodnotu
       optimalizujeme při tréninku.
    2. **RGB-MSE** – po rekonstrukci L + predikovaných ab zpět do RGB.
       Je interpretovatelnější (pixelová chyba ve viditelném prostoru).

PSNR (Peak Signal-to-Noise Ratio)
    Standardní metrika kvality obrazu v dB:

        PSNR = 10 * log10(MAX² / MSE)

    Pro obrázky v rozsahu [0, 1] je MAX = 1, tedy PSNR_dB = -10*log10(MSE).
    Vyšší PSNR znamená věrnější rekonstrukci. Pro kolorizaci jsou hodnoty
    typicky 20–30 dB.
"""

from __future__ import annotations

from typing import Dict

import math
import torch
from torch.utils.data import DataLoader

from .data_utils import lab_to_rgb_tensor


@torch.no_grad()
def evaluate_rgb_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str | torch.device = "cpu",
) -> Dict[str, float]:
    """
    Spočte MSE a PSNR v RGB prostoru přes celý loader.

    Vrací slovník s klíči: 'rgb_mse', 'rgb_psnr_db', 'ab_mse'.
    """
    device = torch.device(device)
    model.eval().to(device)

    sum_sq_ab = 0.0
    count_ab = 0
    sum_sq_rgb = 0.0
    count_rgb = 0

    for L, ab in loader:
        L_dev = L.to(device)
        ab_pred = model(L_dev).cpu()

        # MSE na ab-kanálech
        sum_sq_ab += torch.sum((ab_pred - ab) ** 2).item()
        count_ab += ab.numel()

        # Rekonstrukce do RGB a MSE tam
        rgb_gt = lab_to_rgb_tensor(L, ab)
        rgb_pred = lab_to_rgb_tensor(L, ab_pred)

        sum_sq_rgb += torch.sum((rgb_pred - rgb_gt) ** 2).item()
        count_rgb += rgb_gt.numel()

    ab_mse = sum_sq_ab / count_ab
    rgb_mse = sum_sq_rgb / count_rgb

    if rgb_mse > 0.0:
        psnr_db = -10.0 * math.log10(rgb_mse)        # MAX = 1
    else:
        psnr_db = float("inf")

    return {"rgb_mse": rgb_mse, "rgb_psnr_db": psnr_db, "ab_mse": ab_mse}
