"""API 엔드포인트 테스트."""

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import API_KEYS, model_service
from src.api.main import app


@pytest.fixture(autouse=True)
def _load_model():
    """테스트 전 모델 로드."""
    if not model_service.is_loaded:
        try:
            model_service.load()
        except FileNotFoundError:
            pytest.skip("Model file not found")


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def api_key():
    return next(iter(API_KEYS))


SAMPLE_PLAYER = {
    "Age": 25,
    "Gender": "Male",
    "Location": "USA",
    "GameGenre": "RPG",
    "GameDifficulty": "Medium",
    "PlayTimeHours": 120.5,
    "SessionsPerWeek": 5,
    "AvgSessionDurationMinutes": 45.0,
    "PlayerLevel": 32,
    "AchievementsUnlocked": 15,
    "InGamePurchases": 8,
}


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_single_success(client, api_key):
    resp = client.post(
        "/api/v1/predict/single",
        json=SAMPLE_PLAYER,
        headers={"X-API-Key": api_key},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert 0 <= data["churn_probability"] <= 1
    assert isinstance(data["churn_prediction"], bool)
    assert data["risk_level"] in ("low", "medium", "high", "critical")


def test_predict_single_no_api_key(client):
    resp = client.post("/api/v1/predict/single", json=SAMPLE_PLAYER)
    assert resp.status_code == 401


def test_predict_single_invalid_key(client):
    resp = client.post(
        "/api/v1/predict/single",
        json=SAMPLE_PLAYER,
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


def test_predict_batch(client, api_key):
    resp = client.post(
        "/api/v1/predict/batch",
        json={"players": [SAMPLE_PLAYER, SAMPLE_PLAYER]},
        headers={"X-API-Key": api_key},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["predictions"]) == 2


def test_predict_single_validation_error(client, api_key):
    bad_player = {**SAMPLE_PLAYER, "Age": -1}
    resp = client.post(
        "/api/v1/predict/single",
        json=bad_player,
        headers={"X-API-Key": api_key},
    )
    assert resp.status_code == 422
