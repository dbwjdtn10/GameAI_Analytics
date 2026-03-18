"""JWT 인증 테스트."""

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import model_service
from src.api.main import app


@pytest.fixture(autouse=True)
def _load_model():
    if not model_service.is_loaded:
        try:
            model_service.load()
        except FileNotFoundError:
            pytest.skip("Model file not found")


@pytest.fixture
def client():
    return TestClient(app)


def test_login_success(client):
    resp = client.post(
        "/auth/token",
        data={"username": "admin", "password": "gameai2024"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["expires_in"] > 0


def test_login_wrong_password(client):
    resp = client.post(
        "/auth/token",
        data={"username": "admin", "password": "wrong"},
    )
    assert resp.status_code == 401


def test_login_nonexistent_user(client):
    resp = client.post(
        "/auth/token",
        data={"username": "nobody", "password": "test"},
    )
    assert resp.status_code == 401


def test_token_contains_expected_fields(client):
    resp = client.post(
        "/auth/token",
        data={"username": "analyst", "password": "analyst2024"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data

    from jose import jwt

    from src.config import JWT_ALGORITHM, JWT_SECRET_KEY

    payload = jwt.decode(data["access_token"], JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    assert payload["sub"] == "analyst"
    assert payload["role"] == "viewer"
