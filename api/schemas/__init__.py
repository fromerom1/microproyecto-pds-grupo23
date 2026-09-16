"""Esquemas de entrada y salida de la API."""

from api.schemas.health import Health
from api.schemas.modelo import InfoModelo, OpcionesModelo, PuntoOperacion
from api.schemas.prediccion import LoteEntrada, PrediccionRespuesta, RegistroEntrada, ResultadosLote

__all__ = [
    "Health",
    "InfoModelo",
    "LoteEntrada",
    "OpcionesModelo",
    "PrediccionRespuesta",
    "PuntoOperacion",
    "RegistroEntrada",
    "ResultadosLote",
]
