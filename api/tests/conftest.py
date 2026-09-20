import json

import pandas as pd
import pytest

from api.config import directorio_modelos
from api.modelos import modelos_disponibles
from src.preprocessing import CATEGORIAS


@pytest.fixture(autouse=True)
def cohorte_sintetica(tmp_path, monkeypatch):
    monkeypatch.delenv("API_MODELS_DIR", raising=False)
    features = set()
    for horizonte, variantes in modelos_disponibles().items():
        for variante in variantes:
            ruta = directorio_modelos() / f"model_stunting_{horizonte}_{variante}.json"
            metadata = json.loads(ruta.read_text(encoding="utf-8"))
            features.update(metadata["features"])

    registros = []
    for indice in range(3):
        registro = {}
        for campo in sorted(features):
            if campo in CATEGORIAS:
                opciones = CATEGORIAS[campo]
                registro[campo] = opciones[indice % len(opciones)]
            elif campo == "gestage_final":
                registro[campo] = 36.0 + indice
            else:
                registro[campo] = float(indice - 1)
        registros.append(registro)

    ruta = tmp_path / "cohorte_sintetica.csv"
    pd.DataFrame(registros).to_csv(ruta, index=False, encoding="utf-8")
    monkeypatch.setenv("API_COHORTE_PATH", str(ruta))
    return ruta
