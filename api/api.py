"""Rutas públicas de la API."""

from fastapi import APIRouter

from api.schemas import Health

api_router = APIRouter()


@api_router.get("/health", response_model=Health)
def health() -> Health:
    """Confirma que el servicio está respondiendo."""
    return Health(status="ok")
