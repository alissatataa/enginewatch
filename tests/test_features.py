import numpy as np
import pytest

from enginewatch.data.cmapss import FD001_SENSORS, generate_synthetic
from enginewatch.ml.features import build_features, make_rul_targets
from enginewatch.ml.predictor import MissingSensorsError, RulPredictor, train_model


@pytest.fixture(scope="module")
def df():
    return generate_synthetic(n_units=6, seed=3)


def test_rul_targets_end_at_zero_and_are_capped(df):
    y = make_rul_targets(df, cap=125)
    assert y.max() == 125
    last = df.groupby("unit")["cycle"].idxmax()
    assert (y.loc[last] == 0).all()


def test_rolling_features_do_not_leak_across_engines(df):
    X = build_features(df, FD001_SENSORS, window=5)
    first_rows = df.groupby("unit")["cycle"].idxmin()
    # First cycle of each engine: rolling mean equals the raw value (no bleed from previous engine)
    np.testing.assert_allclose(X.loc[first_rows, "s2"], df.loc[first_rows, "s2"])


def test_online_prediction_matches_offline(df):
    predictor, _ = train_model(df, FD001_SENSORS, window=5)
    unit = df[df["unit"] == 2].reset_index(drop=True)
    offline = predictor.predict_frame(build_features(unit, FD001_SENSORS, 5))
    i = 80
    hist = unit.iloc[i - 4 : i + 1]
    online = predictor.predict_one(hist[FD001_SENSORS].to_dict("records"), hist["cycle"].tolist())
    assert online == pytest.approx(offline[i])


def test_missing_sensor_raises(df):
    predictor, _ = train_model(df, FD001_SENSORS, window=5)
    with pytest.raises(MissingSensorsError):
        predictor.predict_one([{"s2": 1.0}], [1])


def test_model_roundtrip(tmp_path, df):
    predictor, _ = train_model(df, FD001_SENSORS, window=5)
    predictor.save(tmp_path / "m.joblib")
    loaded = RulPredictor.load(tmp_path / "m.joblib")
    assert loaded.sensors == predictor.sensors and loaded.window == 5
