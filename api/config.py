"""Configuracion de ejecucion para el servicio API."""

from __future__ import annotations

import os
from pathlib import Path


RAIZ_PROYECTO = Path(__file__).resolve().parents[1]


def directorio_modelos() -> Path:
    """Devuelve el directorio de artefactos configurado actualmente."""
    return Path(os.environ.get("API_MODELS_DIR", RAIZ_PROYECTO / "models"))


def ruta_cohorte() -> Path:
    """Devuelve la cohorte de referencia configurada actualmente."""
    configurada = os.environ.get("API_COHORTE_PATH")
    if configurada:
        ruta = Path(configurada)
        if not ruta.is_file():
            raise FileNotFoundError("API_COHORTE_PATH no apunta a un archivo.")
        return ruta
    escalera = RAIZ_PROYECTO / "data" / "processed" / "model_dataset_escalera.csv"
    if not escalera.is_file():
        raise FileNotFoundError("Falta data/processed/model_dataset_escalera.csv.")
    return escalera


def host() -> str:
    """Devuelve el host de escucha configurado actualmente."""
    return os.environ.get("API_HOST", "0.0.0.0")


def puerto() -> int:
    """Devuelve el puerto de escucha configurado actualmente."""
    valor = os.environ.get("API_PORT", "8000")
    try:
        numero = int(valor)
    except ValueError as error:
        raise ValueError("API_PORT debe ser un entero valido.") from error
    if not 1 <= numero <= 65535:
        raise ValueError("API_PORT debe estar entre 1 y 65535.")
    return numero
