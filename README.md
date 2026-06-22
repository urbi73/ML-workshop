# Kolorování černobílých fotografií – semestrální projekt

Projekt do workshopu Machine Learning, magisterský obor Datové analýzy.

## Struktura projektu

```
colorization_project/
├── colorization_notebook.ipynb   # hlavní akademický notebook
├── requirements.txt              # závislosti
├── README.md                     # tento soubor
└── src/                          # modulární Python kód
    ├── __init__.py
    ├── data_utils.py             # načítání dat + konverze RGB ↔ LAB + předpočet
    ├── model.py                  # U-Net architektura s Kaiming init
    ├── train.py                  # trénink s best-model checkpointingem a LR scheduler
    ├── metrics.py                # PSNR + RGB-MSE v lidsky viditelném prostoru
    └── visualize.py              # vizualizační utility
```

## Instalace

Doporučuji vytvořit virtuální prostředí:

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

## Spuštění

```bash
jupyter notebook colorization_notebook.ipynb
```

Notebook stáhne CIFAR-10 (~170 MB) při prvním spuštění do složky `./data`.

## Klíčové implementační volby

- **Předpočítání LAB**: konverze RGB→LAB proběhne **jednou** při tvorbě
  DataLoaderů, ne v každé iteraci → výrazně rychlejší epochy.
- **Best-model checkpointing**: `train_model` sleduje minimum val. MSE
  a na konec obnoví váhy nejlepší epochy – nevrací nutně poslední epochu.
- **LR scheduler**: `ReduceLROnPlateau` (factor=0.5, patience=3) doladí lr,
  pokud konvergence zpomalí.
- **Dvě sady metrik**:
  - ab-MSE: stejná jednotka jako loss (řízení tréninku),
  - RGB-MSE a PSNR (dB): interpretovatelné v lidsky viditelném prostoru.
- **Reprodukovatelnost**: seed 42 pro NumPy, PyTorch i shuffle v DataLoaderu.

## Očekávaná doba trénování

- CPU (4 jádra, např. střední notebook): ~3–5 minut na 20 epoch při 5 000
  trénovacích vzorcích.
- CUDA GPU: pod 1 minutu.

Pokud je výpočetní výkon ještě omezenější, lze v notebooku snížit:
- `n_train` na 2 000,
- `N_EPOCHS` na 10,
- `base_channels` v `ColorizationNet` na 16 (~0.5 M parametrů).
