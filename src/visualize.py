"""
visualize.py
============

Pomocné funkce pro vizualizaci dat, průběhu trénování a výsledků kolorizace.
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_utils import lab_to_rgb_tensor


def _to_numpy_img(t: torch.Tensor) -> np.ndarray:
    """Tensor (C, H, W) v [0, 1] -> NumPy (H, W, C) pro imshow."""
    return t.detach().cpu().permute(1, 2, 0).numpy()


def _strip_ticks(ax: plt.Axes) -> None:
    """Skryje ticks, ale ponechá rámeček a ylabel funkční."""
    ax.set_xticks([])
    ax.set_yticks([])


def show_sample_grid(
    loader: DataLoader,
    n: int = 6,
    title: str = "Ukázky vstupních dat (L) a barevných cílů (ab → RGB)",
) -> None:
    """Vykreslí n párů: vstupní L (šedá) vs. ground-truth RGB."""
    L_batch, ab_batch = next(iter(loader))
    L_batch, ab_batch = L_batch[:n], ab_batch[:n]
    rgb_batch = lab_to_rgb_tensor(L_batch, ab_batch)

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    fig.suptitle(title)

    for i in range(n):
        axes[0, i].imshow(L_batch[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        _strip_ticks(axes[0, i])
        axes[1, i].imshow(_to_numpy_img(rgb_batch[i]))
        _strip_ticks(axes[1, i])

    axes[0, 0].set_ylabel("L (vstup)", fontsize=11)
    axes[1, 0].set_ylabel("RGB (cíl)", fontsize=11)

    plt.tight_layout()
    plt.show()


def plot_training_history(history: Dict[str, List[float]]) -> None:
    """
    Vykreslí dva grafy vedle sebe:
      (a) trénovací a validační ztráta,
      (b) průběh learning rate.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history["train_loss"], "o-", label="Trénovací MSE")
    ax1.plot(epochs, history["val_loss"], "s-", label="Validační MSE")
    if "best_epoch" in history and history["best_epoch"]:
        ax1.axvline(
            history["best_epoch"], color="green", ls="--", alpha=0.6,
            label=f"Nejlepší epocha ({history['best_epoch']})",
        )
    ax1.set_xlabel("Epocha")
    ax1.set_ylabel("MSE na ab-kanálech")
    ax1.set_title("Vývoj ztrátové funkce")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if "lr" in history:
        ax2.plot(epochs, history["lr"], "d-", color="purple")
        ax2.set_xlabel("Epocha")
        ax2.set_ylabel("Learning rate")
        ax2.set_yscale("log")
        ax2.set_title("Průběh learning rate (ReduceLROnPlateau)")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.axis("off")

    plt.tight_layout()
    plt.show()


def show_colorization_results(
    model: torch.nn.Module,
    loader: DataLoader,
    n: int = 8,
    device: str | torch.device = "cpu",
    title: str = "Výsledky kolorizace na testovacích datech",
) -> None:
    """
    Vykreslí trojice: vstup (L), predikce modelu (RGB), ground-truth (RGB).
    """
    device = torch.device(device)
    model.eval().to(device)

    L_batch, ab_batch = next(iter(loader))
    L_batch, ab_batch = L_batch[:n], ab_batch[:n]

    with torch.no_grad():
        ab_pred = model(L_batch.to(device)).cpu()

    rgb_gt = lab_to_rgb_tensor(L_batch, ab_batch)
    rgb_pred = lab_to_rgb_tensor(L_batch, ab_pred)

    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
    fig.suptitle(title)

    for i in range(n):
        axes[0, i].imshow(L_batch[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(_to_numpy_img(rgb_pred[i]))
        axes[2, i].imshow(_to_numpy_img(rgb_gt[i]))
        for r in range(3):
            _strip_ticks(axes[r, i])

    axes[0, 0].set_ylabel("Vstup (L)", fontsize=11)
    axes[1, 0].set_ylabel("Predikce", fontsize=11)
    axes[2, 0].set_ylabel("Ground truth", fontsize=11)

    plt.tight_layout()
    plt.show()
