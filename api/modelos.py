"""Descubrimiento y carga de modelos disponibles para la API."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from api.config import directorio_modelos, ruta_cohorte

if TYPE_CHECKING:
    from src.predict import ModeloRiesgo


CLAVES_METADATOS = {
    "horizonte", "features", "target", "config", "familia", "n_entrenamiento",
    "prevalencia", "umbral_alto", "umbral_medio", "curva_capacidad",
    "prob_cohorte_ordenada", "cv", "data_md5", "sklearn", "creado",
}


class MetadatosInvalidos(ValueError):
    pass


def validar_metadatos(ruta: Path, horizonte: str, variante: str) -> None:
    nombre = f"{horizonte}/{variante}"
    try:
        metadata = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise MetadatosInvalidos(f"No se pudieron leer los metadatos de {nombre}.") from error
    if not isinstance(metadata, dict):
        raise MetadatosInvalidos(f"Metadatos inválidos de {nombre}.")
    faltantes = sorted(CLAVES_METADATOS - metadata.keys())
    if faltantes:
        raise MetadatosInvalidos(
            f"Metadatos incompletos de {nombre}: {', '.join(faltantes)}."
        )
    if metadata["horizonte"] != horizonte:
        raise MetadatosInvalidos(f"Horizonte inconsistente en {nombre}.")
    if not isinstance(metadata["features"], list) or not metadata["features"]:
        raise MetadatosInvalidos(f"Lista de features inválida en {nombre}.")
    curva = metadata["curva_capacidad"]
    if not isinstance(curva, list) or not curva:
        raise MetadatosInvalidos(f"curva_capacidad vacía o inválida en {nombre}.")
    claves_curva = {"capacidad", "sensibilidad", "vpp", "umbral"}
    if any(not isinstance(punto, dict) or not claves_curva <= punto.keys() for punto in curva):
        raise MetadatosInvalidos(f"curva_capacidad incompleta en {nombre}.")
    prob = metadata["prob_cohorte_ordenada"]
    if not isinstance(prob, list) or not prob:
        raise MetadatosInvalidos(f"prob_cohorte_ordenada vacía o inválida en {nombre}.")
    if not isinstance(metadata["cv"], dict):
        raise MetadatosInvalidos(f"cv inválido en {nombre}.")


def _artefactos_disponibles(models_dir: Path) -> dict[str, list[str]]:
    disponibles: dict[str, list[str]] = {}
    if not models_dir.is_dir():
        return disponibles

    for ruta_json in models_dir.glob("model_stunting_*.json"):
        if not ruta_json.with_suffix(".joblib").is_file():
            continue
        nombre = ruta_json.stem.removeprefix("model_stunting_")
        horizonte, separador, variante = nombre.partition("_")
        if not separador or not horizonte or not variante:
            continue
        disponibles.setdefault(horizonte, []).append(variante)

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
    validar_metadatos(
        models_dir / f"model_stunting_{horizonte}_{variante}.json", horizonte, variante
    )
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
