"""
train.py
========

Trénovací a evaluační procedury.

Ztrátová funkce: MSE mezi predikovanými a skutečnými ab-kanály.
Optimalizátor:   Adam (osvědčená volba pro CNN, rychlá konvergence).
LR scheduler:    ReduceLROnPlateau – pokud se validační ztráta dlouho nelepší,
                 sníží learning rate. Pomáhá s doladěním vah v pozdějších epochách.
Best-model:      Po každé epoše porovnáme validační ztrátu s dosavadním minimem.
                 Pokud je lepší, uložíme snapshot vah. Na konec se obnoví
                 nejlepší stav – nevracíme tedy nutně poslední epochu.
"""

from __future__ import annotations

import copy
import time
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    """
    Jedna epocha. Pokud je optimizer = None, jede se v eval režimu (validace/test).
    Vrací průměrnou ztrátu na vzorek.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_samples = 0.0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for L, ab in loader:
            L, ab = L.to(device), ab.to(device)
            pred = model(L)
            loss = criterion(pred, ab)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = L.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

    return total_loss / total_samples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 20,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    scheduler_patience: int = 3,
    scheduler_factor: float = 0.5,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Hlavní trénovací smyčka.

    - Po každé epoše vyhodnotí val. ztrátu.
    - Pokud val. ztráta překoná dosavadní minimum, uloží snapshot vah.
    - ReduceLROnPlateau snižuje lr o `scheduler_factor`, pokud se val. ztráta
      `scheduler_patience` epoch nezlepšila.
    - Na konci vrátí model s nejlepším val. snapshotem (in-place).

    Returns
    -------
    history : dict
        Slovník 'train_loss', 'val_loss', 'lr', 'best_epoch'.
    """
    device = torch.device(device)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "best_epoch": 0,
    }

    best_val = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = 0

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = _run_epoch(model, val_loader, criterion, None, device)
        scheduler.step(val_loss)
        dt = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if verbose:
            tag = " *" if improved else ""
            print(
                f"Epoch {epoch:02d}/{n_epochs} | "
                f"train MSE: {train_loss:.4f} | val MSE: {val_loss:.4f} | "
                f"lr: {current_lr:.1e} | {dt:.1f}s{tag}"
            )

    history["best_epoch"] = best_epoch

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(
                f"\nObnoven nejlepší model z epochy {best_epoch} "
                f"(val MSE = {best_val:.4f})."
            )

    return history


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device = "cpu",
) -> float:
    """Vrátí průměrné MSE na ab-kanálech (typicky test loader)."""
    device = torch.device(device)
    model.to(device)
    return _run_epoch(model, loader, nn.MSELoss(), None, device)
