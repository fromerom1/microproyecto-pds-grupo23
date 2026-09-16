"""Descubrimiento y carga de modelos disponibles para la API."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from api.config import directorio_modelos, ruta_cohorte

if TYPE_CHECKING:
    from src.predict import ModeloRiesgo


def _artefactos_disponibles(models_dir: Path) -> dict[str, list[str]]:
    disponibles: dict[str, list[str]] = {}
    if not models_dir.is_dir():
        return disponibles

    for ruta_json in models_dir.glob("model_stunting_*.json"):
        try:
            metadata = json.loads(ruta_json.read_text(encoding="utf-8"))
            horizonte = metadata["horizonte"]
        except (OSError, ValueError, KeyError, TypeError):
            continue

        prefijo = f"model_stunting_{horizonte}_"
        if not ruta_json.stem.startswith(prefijo):
            continue
        variante = ruta_json.stem.removeprefix(prefijo)
        if variante and ruta_json.with_suffix(".joblib").is_file():
            disponibles.setdefault(str(horizonte), []).append(variante)

    return {horizonte: sorted(variantes) for horizonte, variantes in sorted(disponibles.items())}


def modelos_disponibles() -> dict[str, list[str]]:
    """Lista horizontes y variantes con JSON y joblib emparejados."""
    return _artefactos_disponibles(directorio_modelos())


def seleccionar_variante(horizonte: str, variante: str | None = None) -> str:
    """Valida la variante solicitada o elige su alternativa predeterminada."""
    disponibles = modelos_disponibles()
    variantes = disponibles.get(horizonte)
    if not variantes:
        raise ValueError(f"No existe un modelo para horizonte='{horizonte}'.")
    if variante is not None:
        if variante not in variantes:
            raise ValueError(
                f"No existe un modelo para horizonte='{horizonte}', variante='{variante}'."
            )
        return variante
    for preferida in ("B", "cv"):
        if preferida in variantes:
            return preferida
    return variantes[0]


@lru_cache(maxsize=16)
def _cargar_modelo(
    horizonte: str, variante: str, models_dir: Path, cohorte_path: Path
) -> "ModeloRiesgo":
    from src.predict import ModeloRiesgo

    return ModeloRiesgo(
        horizonte=horizonte,
        variante=variante,
        models_dir=models_dir,
        cohorte_path=cohorte_path,
    )


def obtener_modelo(horizonte: str = "24m", variante: str | None = None) -> "ModeloRiesgo":
    """Reutiliza el ModeloRiesgo solicitado segun la configuracion actual."""
    variante_elegida = seleccionar_variante(horizonte, variante)
    return _cargar_modelo(
        horizonte, variante_elegida, directorio_modelos(), ruta_cohorte()
    )
