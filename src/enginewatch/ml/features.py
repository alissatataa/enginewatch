"""Feature engineering shared by training and online inference.

Using one function for both paths guarantees there is no train/serve skew.
"""

from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame, sensors: list[str], window: int) -> pd.DataFrame:
    """Per-engine rolling mean of each sensor, plus the cycle counter.

    ``df`` needs columns ``unit``, ``cycle`` and the sensor columns. The result is
    aligned to ``df.index`` in (unit, cycle) order.
    """
    df = df.sort_values(["unit", "cycle"])
    rolled = (
        df.groupby("unit")[sensors]
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .reindex(df.index)
    )
    rolled["cycle"] = df["cycle"]
    return rolled


def make_rul_targets(df: pd.DataFrame, cap: int) -> pd.Series:
    """Piece-wise linear RUL target: cycles to failure, clipped at ``cap``."""
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    return (max_cycle - df["cycle"]).clip(upper=cap).astype(float)
