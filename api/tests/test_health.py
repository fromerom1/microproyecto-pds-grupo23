"""Pruebas del arranque y la ruta de salud."""

from fastapi.testclient import TestClient

from api.main import app


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_reporta_cohorte_inexistente(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setenv("API_COHORTE_PATH", str(tmp_path / "inexistente.csv"))
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "error"
    assert "API_COHORTE_PATH" in caplog.text


def test_model_info_reporta_cohorte_inexistente(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("API_COHORTE_PATH", str(tmp_path / "inexistente.csv"))
    with TestClient(app) as client:
        response = client.get("/model/info")

    assert response.status_code == 503
    assert "API_COHORTE_PATH" in response.json()["detail"]


def test_health_reporta_cohorte_incompatible(tmp_path, monkeypatch) -> None:
    cohorte = tmp_path / "cohorte.csv"
    cohorte.write_text("otra_columna\n1\n", encoding="utf-8")
    monkeypatch.setenv("API_COHORTE_PATH", str(cohorte))
    with TestClient(app) as client:
        response = client.get("/health")
        info = client.get("/model/info")

    assert response.status_code == 503
    assert response.json()["status"] == "error"
    assert info.status_code == 503
    assert "cohorte" in info.json()["detail"]


def test_docs() -> None:
    with TestClient(app) as client:
        response = client.get("/docs")

    assert response.status_code == 200
    assert "swagger-ui" in response.text
