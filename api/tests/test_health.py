"""Pruebas del arranque y la ruta de salud."""

import json
from shutil import copyfile

import pandas as pd
from fastapi.testclient import TestClient

from api import config
from api.config import directorio_modelos
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


def test_health_reporta_cohorte_sin_filas(cohorte_sintetica) -> None:
    datos = pd.read_csv(cohorte_sintetica).iloc[:0]
    datos.to_csv(cohorte_sintetica, index=False, encoding="utf-8")

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert "filas" in response.json()["detail"]


def test_health_reporta_referencia_invalida(cohorte_sintetica) -> None:
    datos = pd.read_csv(cohorte_sintetica)
    datos["gestage_final"] = "no-numero"
    datos.to_csv(cohorte_sintetica, index=False, encoding="utf-8")

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert "referencia" in response.json()["detail"]


def test_health_indica_como_obtener_escalera(tmp_path, monkeypatch) -> None:
    modelos = directorio_modelos()
    monkeypatch.setenv("API_MODELS_DIR", str(modelos))
    monkeypatch.delenv("API_COHORTE_PATH")
    monkeypatch.setattr(config, "RAIZ_PROYECTO", tmp_path)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert "model_dataset_escalera.csv" in response.json()["detail"]
    assert "dvc pull" in response.json()["detail"]


def copiar_variante(tmp_path, horizonte: str, variante: str):
    origen = directorio_modelos()
    nombre = f"model_stunting_{horizonte}_{variante}"
    copyfile(origen / f"{nombre}.joblib", tmp_path / f"{nombre}.joblib")
    ruta_json = tmp_path / f"{nombre}.json"
    copyfile(origen / f"{nombre}.json", ruta_json)
    return ruta_json


def test_metadatos_incompletos_devuelven_503(tmp_path, monkeypatch) -> None:
    ruta = copiar_variante(tmp_path, "24m", "B")
    metadata = json.loads(ruta.read_text(encoding="utf-8"))
    metadata.pop("target")
    ruta.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setenv("API_MODELS_DIR", str(tmp_path))

    with TestClient(app) as client:
        salud = client.get("/health")
        info = client.get("/model/info")

    assert salud.status_code == 503
    assert "target" in salud.json()["detail"]
    assert info.status_code == 503
    assert "target" in info.json()["detail"]


def test_health_revisa_todas_las_variantes(tmp_path, monkeypatch) -> None:
    copiar_variante(tmp_path, "24m", "B")
    ruta = copiar_variante(tmp_path, "24m", "cv")
    metadata = json.loads(ruta.read_text(encoding="utf-8"))
    metadata.pop("familia")
    ruta.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setenv("API_MODELS_DIR", str(tmp_path))

    with TestClient(app) as client:
        salud = client.get("/health")

    assert salud.status_code == 503
    assert "cv" in salud.json()["detail"]
    assert "familia" in salud.json()["detail"]


def test_curva_vacia_devuelve_503(tmp_path, monkeypatch) -> None:
    ruta = copiar_variante(tmp_path, "24m", "B")
    metadata = json.loads(ruta.read_text(encoding="utf-8"))
    metadata["curva_capacidad"] = []
    ruta.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setenv("API_MODELS_DIR", str(tmp_path))

    with TestClient(app) as client:
        salud = client.get("/health")

    assert salud.status_code == 503
    assert "curva_capacidad" in salud.json()["detail"]


def test_directorio_modelos_invalido_devuelve_503(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("API_MODELS_DIR", str(tmp_path / "sin_modelos"))

    with TestClient(app) as client:
        salud = client.get("/health")
        info = client.get("/model/info")

    assert salud.status_code == 503
    assert info.status_code == 503
    assert "modelos" in salud.json()["detail"]


def test_health_no_omite_json_sin_horizonte(tmp_path, monkeypatch) -> None:
    copiar_variante(tmp_path, "24m", "B")
    ruta = copiar_variante(tmp_path, "24m", "cv")
    metadata = json.loads(ruta.read_text(encoding="utf-8"))
    metadata.pop("horizonte")
    ruta.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setenv("API_MODELS_DIR", str(tmp_path))

    with TestClient(app) as client:
        salud = client.get("/health")

    assert salud.status_code == 503
    assert "24m/cv" in salud.json()["detail"]
    assert "horizonte" in salud.json()["detail"]
