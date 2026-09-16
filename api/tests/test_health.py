"""Pruebas del arranque y la ruta de salud."""

from fastapi.testclient import TestClient

from api.main import app


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_docs() -> None:
    with TestClient(app) as client:
        response = client.get("/docs")

    assert response.status_code == 200
    assert "swagger-ui" in response.text
