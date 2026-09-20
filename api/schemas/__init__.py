"""Esquemas de entrada y salida de la API."""

from api.schemas.health import Health
from api.schemas.error import ErrorRespuesta
from api.schemas.modelo import InfoModelo, ModelosDisponibles, OpcionesModelo, PuntoOperacion
from api.schemas.prediccion import LoteEntrada, PrediccionRespuesta, RegistroEntrada, ResultadosLote

__all__ = [
    "Health",
    "ErrorRespuesta",
    "InfoModelo",
    "LoteEntrada",
    "ModelosDisponibles",
    "OpcionesModelo",
    "PrediccionRespuesta",
    "PuntoOperacion",
    "RegistroEntrada",
    "ResultadosLote",
]
