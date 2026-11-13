import pytest

from enginewatch.data.cmapss import generate_synthetic, row_to_payload


@pytest.fixture(scope="module")
def run_to_failure():
    """One engine's full life from a seed the model never saw."""
    df = generate_synthetic(n_units=1, seed=99)
    return [row_to_payload(r) for _, r in df.iterrows()]


def test_health_reports_model(client, client_no_model):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    assert client_no_model.get("/health").json()["model_loaded"] is False


def test_full_engine_lifecycle_raises_alerts(client, run_to_failure):
    for p in run_to_failure:
        assert client.post("/readings", json=p).status_code == 201

    fleet = client.get("/engines").json()
    assert len(fleet) == 1 and fleet[0]["cycle"] == len(run_to_failure)
    assert fleet[0]["status"] in {"warning", "critical"}

    history = client.get("/engines/1/history").json()
    assert len(history) == len(run_to_failure)
    assert history[0]["rul"] > history[-1]["rul"]  # health degrades over life

    alerts = client.get("/alerts").json()
    levels = [a["level"] for a in alerts]
    assert "warning" in levels or "critical" in levels
    assert len(alerts) <= 2  # at most one warning and one critical: latched, not spam


def test_duplicate_reading_conflicts(client, run_to_failure):
    assert client.post("/readings", json=run_to_failure[0]).status_code == 201
    assert client.post("/readings", json=run_to_failure[0]).status_code == 409


def test_validation_errors(client, run_to_failure):
    bad = {**run_to_failure[0], "settings": [1.0]}
    assert client.post("/readings", json=bad).status_code == 422
    missing = {**run_to_failure[0], "sensors": {"s2": 1.0}}
    resp = client.post("/readings", json=missing)
    assert resp.status_code == 422 and "missing sensors" in resp.json()["detail"]


def test_without_model_data_is_stored_but_rul_is_null(client_no_model, run_to_failure):
    resp = client_no_model.post("/readings", json=run_to_failure[0])
    assert resp.status_code == 201
    assert resp.json()["rul"] is None and resp.json()["status"] == "unknown"
    assert client_no_model.get("/alerts").json() == []


def test_unknown_engine_history_404(client):
    assert client.get("/engines/999/history").status_code == 404


def test_dashboard_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "EngineWatch" in r.text
