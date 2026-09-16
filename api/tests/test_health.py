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
    assert response.json()["detail"] == "La cohorte de referencia no está disponible."
    assert "API_COHORTE_PATH" in caplog.text


def test_model_info_reporta_cohorte_inexistente(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("API_COHORTE_PATH", str(tmp_path / "inexistente.csv"))
    with TestClient(app) as client:
        response = client.get("/model/info")

    assert response.status_code == 503
    assert response.json()["detail"] == "La cohorte de referencia no está disponible."


def test_health_reporta_cohorte_incompatible(tmp_path, monkeypatch) -> None:
    cohorte = tmp_path / "cohorte.csv"
    cohorte.write_text("otra_columna\n1\n", encoding="utf-8")
    monkeypatch.setenv("API_COHORTE_PATH", str(cohorte))
    with TestClient(app) as client:
        response = client.get("/health")
        info = client.get("/model/info")

    assert response.status_code == 503
    assert response.json()["status"] == "error"
    assert response.json()["detail"] == "La cohorte no contiene las variables requeridas."
    assert info.status_code == 503
    assert info.json()["detail"] == response.json()["detail"]


def test_errores_internos_no_se_exponen(monkeypatch) -> None:
    def fallar(*args, **kwargs):
        raise RuntimeError("detalle interno")

    monkeypatch.setattr("api.api.modelo_listo", fallar)
    with TestClient(app) as client:
        salud = client.get("/health")
        info = client.get("/model/info")

    assert salud.status_code == 503
    assert salud.json()["detail"] == "El servicio no está disponible."
    assert info.status_code == 503
    assert info.json()["detail"] == "El servicio no está disponible."


def test_docs() -> None:
    with TestClient(app) as client:
        response = client.get("/docs")

    assert response.status_code == 200
    assert "swagger-ui" in response.text
