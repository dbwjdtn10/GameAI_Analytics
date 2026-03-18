"""Prometheus 메트릭 & 헬스 체크 확장 테스트."""

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


def test_metrics_endpoint_exists(client):
    """Prometheus /metrics 엔드포인트 존재 확인."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "http_request_duration" in resp.text or "HELP" in resp.text


def test_health_includes_components(client):
    """확장 헬스 체크에 components 필드 포함 확인."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "components" in data
    assert "model" in data["components"]


def test_request_id_header(client):
    """응답에 X-Request-ID 헤더 포함 확인."""
    resp = client.get("/health")
    assert "X-Request-ID" in resp.headers


def test_process_time_header(client):
    """응답에 X-Process-Time-Ms 헤더 포함 확인."""
    resp = client.get("/health")
    assert "X-Process-Time-Ms" in resp.headers
