"""NASA C-MAPSS loading plus a synthetic generator with the same schema.

The real dataset is a set of whitespace-separated text files (train_FD001.txt, ...).
Download it from the NASA Prognostics Data Repository and place the files in
``data/CMAPSS/``. The synthetic generator exists so tests, CI and demos work
without the download.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SETTING_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTING_COLS, *SENSOR_COLS]

# Sensors that actually vary in FD001 (the others are constant).
FD001_SENSORS = [
    "s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21",
]  # fmt: skip


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, usecols=range(26), names=COLUMNS)


def load_cmapss(
    data_dir: Path | str, subset: str = "FD001"
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return ``(train, test, rul_true)`` for a subset such as ``FD001``."""
    d = Path(data_dir)
    files = {
        "train": d / f"train_{subset}.txt",
        "test": d / f"test_{subset}.txt",
        "rul": d / f"RUL_{subset}.txt",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"C-MAPSS files not found: {', '.join(missing)}")
    rul = pd.read_csv(files["rul"], header=None).iloc[:, 0].to_numpy()
    return _read(files["train"]), _read(files["test"]), rul


def generate_synthetic(n_units: int = 60, seed: int = 0) -> pd.DataFrame:
    """Run-to-failure data shaped like C-MAPSS.

    Per-sensor baselines/slopes come from a fixed RNG so every call (any seed) shares
    the same "physics"; only lifetimes and noise differ between seeds.
    """
    phys = np.random.default_rng(42)
    base = {s: phys.uniform(20, 600) for s in SENSOR_COLS}
    slope = {s: phys.choice([-1, 1]) * phys.uniform(0.01, 0.05) * base[s] for s in FD001_SENSORS}
    noise = {s: 0.004 * base[s] for s in SENSOR_COLS}

    rng = np.random.default_rng(seed)
    frames = []
    for unit in range(1, n_units + 1):
        life = int(rng.integers(150, 300))
        t = np.arange(1, life + 1)
        degradation = (t / life) ** 2.5
        data: dict[str, np.ndarray] = {"unit": np.full(life, unit), "cycle": t}
        data["op1"] = rng.normal(0, 0.002, life)
        data["op2"] = rng.normal(0, 0.0003, life)
        data["op3"] = np.full(life, 100.0)
        for s in SENSOR_COLS:
            trend = slope[s] * degradation if s in slope else 0.0
            data[s] = base[s] + trend + rng.normal(0, noise[s], life)
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)[COLUMNS]


def row_to_payload(row: pd.Series) -> dict:
    """Convert a C-MAPSS row into the JSON body expected by ``POST /readings``."""
    return {
        "engine_id": int(row["unit"]),
        "cycle": int(row["cycle"]),
        "settings": [float(row[c]) for c in SETTING_COLS],
        "sensors": {c: float(row[c]) for c in SENSOR_COLS},
    }
