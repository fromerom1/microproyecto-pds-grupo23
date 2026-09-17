import json

import numpy as np
import pandas as pd
import pytest

from api.config import directorio_modelos, host, puerto, ruta_cohorte
from api.modelos import modelos_disponibles, seleccionar_variante
from src.predict import ModeloRiesgo


def test_cohorte_sintetica_cubre_cada_variante(cohorte_sintetica) -> None:
    datos = pd.read_csv(cohorte_sintetica)
    assert len(datos) == 3
    for horizonte, variantes in modelos_disponibles().items():
        for variante in variantes:
            modelo = ModeloRiesgo(
                horizonte, variante, cohorte_path=cohorte_sintetica
            )
            assert set(modelo.features).issubset(datos.columns)
            esperada = modelo._prep.transform(datos[modelo.features]).mean(axis=0)
            np.testing.assert_allclose(modelo._referencia, esperada)


def test_descubre_variantes_nuevas(tmp_path, monkeypatch) -> None:
    for variante in ("B", "C", "calibrado"):
        nombre = f"model_stunting_18m_{variante}"
        (tmp_path / f"{nombre}.json").write_text(
            json.dumps({"horizonte": "18m", "features": ["x"]}), encoding="utf-8"
        )
        (tmp_path / f"{nombre}.joblib").touch()
    (tmp_path / "model_stunting_18m_incompleto.json").write_text(
        json.dumps({"horizonte": "18m"}), encoding="utf-8"
    )
    monkeypatch.setenv("API_MODELS_DIR", str(tmp_path))

    assert modelos_disponibles() == {"18m": ["B", "C", "calibrado"]}
    assert seleccionar_variante("18m") == "B"
    assert seleccionar_variante("18m", "calibrado") == "calibrado"


def test_configuracion_por_entorno(tmp_path, monkeypatch) -> None:
    cohorte = tmp_path / "cohorte.csv"
    cohorte.touch()
    monkeypatch.setenv("API_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("API_COHORTE_PATH", str(cohorte))
    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("API_PORT", "8765")

    assert directorio_modelos() == tmp_path
    assert ruta_cohorte() == cohorte
    assert host() == "127.0.0.1"
    assert puerto() == 8765


def test_cohorte_configurada_debe_existir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("API_COHORTE_PATH", str(tmp_path / "inexistente.csv"))

    with pytest.raises(FileNotFoundError, match="API_COHORTE_PATH"):
        ruta_cohorte()


def test_host_y_puerto_predeterminados(monkeypatch) -> None:
    monkeypatch.delenv("API_HOST", raising=False)
    monkeypatch.delenv("API_PORT", raising=False)

    assert host() == "0.0.0.0"
    assert puerto() == 8000


def test_no_usa_dataset_base_como_respaldo(tmp_path, monkeypatch) -> None:
    from api import config

    procesados = tmp_path / "data" / "processed"
    procesados.mkdir(parents=True)
    (procesados / "model_dataset.csv").write_text("x\n1\n", encoding="utf-8")
    monkeypatch.delenv("API_COHORTE_PATH")
    monkeypatch.setattr(config, "RAIZ_PROYECTO", tmp_path)

    with pytest.raises(FileNotFoundError, match="model_dataset_escalera.csv"):
        ruta_cohorte()
