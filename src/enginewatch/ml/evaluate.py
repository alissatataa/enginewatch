"""Standard C-MAPSS evaluation: RMSE and the asymmetric NASA scoring function."""

from __future__ import annotations

import numpy as np
import pandas as pd

from enginewatch.ml.features import build_features
from enginewatch.ml.predictor import RulPredictor


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Late predictions (pred > true) are penalised more heavily than early ones."""
    d = y_pred - y_true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def evaluate_on_test(predictor: RulPredictor, test_df: pd.DataFrame, rul_true: np.ndarray) -> dict:
    X = build_features(test_df, predictor.sensors, predictor.window)
    last_idx = test_df.groupby("unit")["cycle"].idxmax()
    y_pred = predictor.predict_frame(X.loc[last_idx])
    y_true = np.minimum(rul_true, predictor.rul_cap)
    return {
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "nasa_score": nasa_score(y_true, y_pred),
        "n_engines": int(len(y_true)),
    }
