import json

import pytest

from api.config import directorio_modelos, host, puerto, ruta_cohorte
from api.modelos import modelos_disponibles, seleccionar_variante


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
