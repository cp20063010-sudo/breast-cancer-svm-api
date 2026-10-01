from fastapi.testclient import TestClient
from app.main import app, metadata

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True

def test_predict_rejects_missing_features():
    r = client.post("/predict", json={"features": {"mean radius": 10.0}})
    assert r.status_code == 422

def test_metadata_has_30_features():
    r = client.get("/metadata")
    assert r.status_code == 200
    assert len(r.json()["feature_names"]) == 30