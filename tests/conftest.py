import pytest
from fastapi.testclient import TestClient

from enginewatch.api.main import create_app
from enginewatch.config import Settings
from enginewatch.data.cmapss import FD001_SENSORS, generate_synthetic
from enginewatch.ml.predictor import train_model


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    df = generate_synthetic(n_units=15, seed=1)
    predictor, _ = train_model(df, FD001_SENSORS, window=5)
    path = tmp_path_factory.mktemp("model") / "rul.joblib"
    predictor.save(path)
    return path


def _settings(tmp_path, model_file: str) -> Settings:
    return Settings(database_url=f"sqlite:///{tmp_path / 'test.db'}", rul_model_path=model_file)


@pytest.fixture
def client(tmp_path, model_path):
    with TestClient(create_app(_settings(tmp_path, str(model_path)))) as c:
        yield c


@pytest.fixture
def client_no_model(tmp_path):
    with TestClient(create_app(_settings(tmp_path, str(tmp_path / "missing.joblib")))) as c:
        yield c
