"""
data_utils.py
=============

Modul pro načítání dat a konverze mezi barevnými prostory RGB a CIE LAB.

Klíčová myšlenka kolorizace v LAB prostoru:
- Kanál L (Lightness, 0–100) odpovídá jasu, tj. obsahuje informaci,
  kterou má i černobílý obrázek.
- Kanály a, b (přibližně −128 až 127) nesou barevnou informaci.
- Úloha kolorizace = naučit mapování L -> (a, b).

Hodnoty jsou v tomto modulu normalizovány do rozsahů vhodných pro NN:
- L  -> [0, 1]   (děleno 100)
- ab -> [-1, 1]  (děleno 110; pro sRGB obrázky se hodnoty bezpečně vejdou)

Pro efektivitu jsou všechny vzorky převedeny do LAB jednou předem
(funkce `get_dataloaders`) a uloženy jako TensorDataset. Tím se vyhneme
opakovanému volání `skimage.color.rgb2lab` v každé iteraci trénovací smyčky.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from skimage import color
import torchvision
import torchvision.transforms as T
from torchvision.datasets.utils import extract_archive


# Normalizační konstanty pro LAB (viz docstring modulu)
L_NORM = 100.0
AB_NORM = 110.0


# ---------------------------------------------------------------------------
# Konverze barevných prostorů
# ---------------------------------------------------------------------------
def rgb_to_lab_tensor(rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Konvertuje RGB tensor do normalizovaného LAB.

    Parameters
    ----------
    rgb : torch.Tensor
        Tensor tvaru (B, 3, H, W) v rozsahu [0, 1].

    Returns
    -------
    L : torch.Tensor
        Tensor tvaru (B, 1, H, W), hodnoty v [0, 1].
    ab : torch.Tensor
        Tensor tvaru (B, 2, H, W), hodnoty přibližně v [-1, 1].
    """
    rgb_np = rgb.detach().cpu().numpy().transpose(0, 2, 3, 1)
    # Vektorizovaně: rgb2lab zpracuje celé pole (B, H, W, 3) jedním voláním
    # (kanály jsou poslední osa, vedoucí osy se zpracují nezávisle). Tím
    # odpadá per-obrázková Python smyčka a její fixní režie.
    lab_np = color.rgb2lab(rgb_np)

    L = lab_np[..., 0:1] / L_NORM
    ab = lab_np[..., 1:3] / AB_NORM

    L = torch.from_numpy(L).permute(0, 3, 1, 2).float()
    ab = torch.from_numpy(ab).permute(0, 3, 1, 2).float()
    return L, ab


def lab_to_rgb_tensor(L: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    """
    Inverzní konverze: normalizované LAB -> RGB v [0, 1].

    Parameters
    ----------
    L : torch.Tensor
        Tensor tvaru (B, 1, H, W) v [0, 1].
    ab : torch.Tensor
        Tensor tvaru (B, 2, H, W) v přibližně [-1, 1].

    Returns
    -------
    rgb : torch.Tensor
        Tensor tvaru (B, 3, H, W) v [0, 1] (zaclamped).

    Poznámka
    --------
    Pokud predikované hodnoty (a, b) leží mimo sRGB gamut, skimage je tiše
    cropne – během tréninku se to děje běžně. Vzniká přitom UserWarning,
    který zde lokálně potlačujeme, aby výstup notebooku zůstal přehledný.
    """
    L_np = (L.detach().cpu().numpy() * L_NORM).transpose(0, 2, 3, 1)
    ab_np = (ab.detach().cpu().numpy() * AB_NORM).transpose(0, 2, 3, 1)
    lab_np = np.concatenate([L_np, ab_np], axis=-1)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Conversion from CIE-LAB.*",
            category=UserWarning,
        )
        # Vektorizovaně přes celé pole (B, H, W, 3), viz rgb_to_lab_tensor.
        rgb_np = color.lab2rgb(lab_np)

    rgb_np = np.clip(rgb_np, 0.0, 1.0)
    return torch.from_numpy(rgb_np).permute(0, 3, 1, 2).float()



# ---------------------------------------------------------------------------
# Předpočítání LAB nad celou podmnožinou
# ---------------------------------------------------------------------------
def _precompute_lab_dataset(
    subset: Subset, chunk_size: int = 256, num_workers: int | None = None
) -> TensorDataset:
    """
    Projde celý Subset, převede všechny obrázky do LAB a vrátí TensorDataset
    s tensory (L, ab).

    Convert probíhá po dávkách (`chunk_size`) a dávky se zpracují **paralelně
    ve více vláknech**. Numpy/skimage ufunkce uvnitř `rgb2lab` při výpočtu
    uvolňují GIL, takže vlákna reálně běží na více jádrech (na rozdíl od
    procesů zde nevzniká režie spawnu ani picklingu velkých polí). Výsledek je
    numericky identický s jednovláknovým zpracováním.

    Parameters
    ----------
    num_workers : int | None
        Počet vláken. None = automaticky `min(os.cpu_count(), 8)`.
    """
    n = len(subset)
    ranges = [(s, min(s + chunk_size, n)) for s in range(0, n, chunk_size)]

    def _convert_chunk(rng: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        start, end = rng
        rgb_batch = torch.stack([subset[i][0] for i in range(start, end)])  # (b, 3, H, W)
        return rgb_to_lab_tensor(rgb_batch)

    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)

    if num_workers > 1 and len(ranges) > 1:
        # ThreadPoolExecutor.map zachovává pořadí vstupu -> konkatenace je správná
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            results = list(ex.map(_convert_chunk, ranges))
    else:
        results = [_convert_chunk(r) for r in ranges]

    L_all = torch.cat([L for L, _ in results], dim=0)
    ab_all = torch.cat([ab for _, ab in results], dim=0)
    return TensorDataset(L_all, ab_all)


# ---------------------------------------------------------------------------
# Lokální příprava datasetu (BEZ stahování)
# ---------------------------------------------------------------------------
def _ensure_cifar_extracted(data_root: str, verbose: bool = True) -> None:
    """
    Zajistí, že je CIFAR-10 lokálně k dispozici v `data_root`, **bez jakéhokoli
    stahování ze sítě**.

    - Pokud už existuje rozbalená složka (`cifar-10-batches-py/`), nedělá nic.
    - Pokud chybí, ale je přítomen tar archiv (`cifar-10-python.tar.gz`),
      rozbalí ho **lokálně**.
    - Pokud není ani jedno, vyhodí srozumitelnou chybu.

    Stahování je tím pádem zcela vyřazené (bylo extrémně pomalé).
    """
    cls = torchvision.datasets.CIFAR10
    extracted = os.path.join(data_root, cls.base_folder)   # cifar-10-batches-py
    archive = os.path.join(data_root, cls.filename)        # cifar-10-python.tar.gz

    if os.path.isdir(extracted):
        return
    if os.path.isfile(archive):
        if verbose:
            print(f"Rozbaluji lokální archiv '{cls.filename}' (bez stahování) ...")
        extract_archive(archive, data_root)
        return
    raise FileNotFoundError(
        f"CIFAR-10 nenalezen ve složce '{data_root}'. Stahování je vypnuté – "
        f"vlož sem archiv '{cls.filename}' nebo rozbalenou složku "
        f"'{cls.base_folder}/'."
    )


# ---------------------------------------------------------------------------
# Tvorba DataLoaderů
# ---------------------------------------------------------------------------
def get_dataloaders(
    data_root: str = "./data",
    n_train: int = 5000,
    n_val: int = 1000,
    n_test: int = 1000,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Připraví trénovací, validační a testovací DataLoader nad CIFAR-10
    s předpočítanými LAB hodnotami.

    CIFAR-10 je vybrán jako standardní akademický dataset:
    - 60 000 barevných obrázků 32x32 pix,
    - veřejně dostupný přes torchvision,
    - velikost přesně odpovídá výpočetnímu omezení.

    Pro úsporu zdrojů použijeme náhodné podmnožiny se zafixovaným seedem.
    """
    transform = T.Compose([T.ToTensor()])         # PIL -> tensor v [0, 1]

    # Žádné stahování ze sítě: použij lokální data (tar v případě potřeby rozbal)
    _ensure_cifar_extracted(data_root, verbose=verbose)

    full_train = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=transform
    )
    full_test = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=False, transform=transform
    )

    rng = np.random.default_rng(seed)

    train_idx = rng.choice(len(full_train), size=n_train + n_val, replace=False)
    train_part, val_part = train_idx[:n_train], train_idx[n_train:]
    test_idx = rng.choice(len(full_test), size=n_test, replace=False)

    if verbose:
        print("Předpočítávám LAB reprezentaci ...")

    train_ds = _precompute_lab_dataset(Subset(full_train, train_part.tolist()))
    val_ds   = _precompute_lab_dataset(Subset(full_train, val_part.tolist()))
    test_ds  = _precompute_lab_dataset(Subset(full_test,  test_idx.tolist()))

    if verbose:
        print(f"Hotovo: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Reprodukovatelný shuffle
    gen = torch.Generator()
    gen.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=gen,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader
