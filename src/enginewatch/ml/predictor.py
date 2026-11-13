"""A deliberately simple baseline RUL model.

The model is not the point of this project (see the thesis repo for research models);
it is a replaceable component behind the ``RulPredictor`` interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from enginewatch.ml.features import build_features, make_rul_targets


class MissingSensorsError(ValueError):
    def __init__(self, missing: list[str]):
        super().__init__(f"Reading is missing sensors required by the model: {missing}")
        self.missing = missing


@dataclass
class RulPredictor:
    model: Any
    sensors: list[str]
    window: int
    rul_cap: int

    def predict_frame(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(X), 0, self.rul_cap)

    def predict_one(self, history: list[dict[str, float]], cycles: list[int]) -> float:
        """Predict RUL for the newest reading; ``history`` holds recent readings, oldest first."""
        missing = sorted({s for s in self.sensors for h in history if s not in h})
        if missing:
            raise MissingSensorsError(missing)
        df = pd.DataFrame(history)
        df["unit"] = 1
        df["cycle"] = cycles
        X = build_features(df, self.sensors, self.window).iloc[[-1]]
        return float(self.predict_frame(X)[0])

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "sensors": self.sensors,
                "window": self.window,
                "rul_cap": self.rul_cap,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> RulPredictor:
        return cls(**joblib.load(path))


def train_model(
    df: pd.DataFrame,
    sensors: list[str],
    window: int = 5,
    rul_cap: int = 125,
    seed: int = 0,
    val_fraction: float = 0.2,
) -> tuple[RulPredictor, dict[str, float]]:
    """Fit the baseline. Returns the predictor and metrics from a held-out-engine split."""
    X = build_features(df, sensors, window)
    y = make_rul_targets(df, rul_cap).loc[X.index]
    units = df.loc[X.index, "unit"]

    rng = np.random.default_rng(seed)
    unique = units.unique()
    n_val = max(1, int(len(unique) * val_fraction))
    val_units = rng.choice(unique, size=n_val, replace=False)
    val_mask = units.isin(val_units)

    def new_model() -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=seed)

    probe = new_model().fit(X[~val_mask], y[~val_mask])
    pred = np.clip(probe.predict(X[val_mask]), 0, rul_cap)
    rmse = float(np.sqrt(np.mean((pred - y[val_mask]) ** 2)))

    final = new_model().fit(X, y)
    metrics = {"val_rmse": rmse, "n_units": float(len(unique)), "n_rows": float(len(X))}
    return RulPredictor(final, list(sensors), window, rul_cap), metrics
