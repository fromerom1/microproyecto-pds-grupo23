"""Rutas públicas de la API."""

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from api.modelos import modelos_disponibles, obtener_modelo
from api.schemas import (
    Health,
    InfoModelo,
    LoteEntrada,
    OpcionesModelo,
    PrediccionRespuesta,
    PuntoOperacion,
    RegistroEntrada,
    ResultadosLote,
)

api_router = APIRouter()


@api_router.get("/health", response_model=Health)
def health() -> Health:
    """Confirma que el servicio está respondiendo."""
    return Health(status="ok")


def modelo_actual(horizonte: str = "24m", variante: str | None = None):
    try:
        return obtener_modelo(horizonte, variante)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


def validar_registro(registro: dict, modelo, permitir_extras: bool = False) -> None:
    faltantes = [campo for campo in modelo.features if campo not in registro]
    if faltantes:
        raise HTTPException(status_code=422, detail={"campos_faltantes": faltantes})

    if not permitir_extras:
        extras = sorted(set(registro) - set(modelo.features))
        if extras:
            raise HTTPException(status_code=422, detail={"campos_desconocidos": extras})

    opciones = modelo.opciones()
    for campo in modelo.features:
        valor = registro[campo]
        if valor is not None and campo in opciones and valor not in opciones[campo]:
            raise HTTPException(status_code=422, detail={"categoria_invalida": campo})


@api_router.get("/model/info", response_model=InfoModelo)
def info_modelo(modelo=Depends(modelo_actual)) -> dict:
    return {
        **modelo.info,
        "features": modelo.features,
        "features_extra": modelo.features_extra,
        "variantes_disponibles": modelos_disponibles()[modelo.horizonte],
    }


@api_router.get("/model/options", response_model=OpcionesModelo)
def opciones_modelo(modelo=Depends(modelo_actual)) -> dict:
    opciones = modelo.opciones()
    return {campo: opciones[campo] for campo in modelo.features if campo in opciones}


@api_router.post("/predict", response_model=PrediccionRespuesta)
def predecir(entrada: RegistroEntrada, modelo=Depends(modelo_actual)) -> dict:
    validar_registro(entrada.registro, modelo)
    try:
        return modelo.predecir(entrada.registro)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail="Registro incompatible con el modelo.") from error


@api_router.post("/predict/batch", response_model=ResultadosLote)
def predecir_lote(
    entrada: LoteEntrada,
    capacidad: float | None = Query(default=None, ge=0, le=1),
    modelo=Depends(modelo_actual),
) -> list[dict]:
    for registro in entrada.registros:
        validar_registro(registro, modelo, permitir_extras=True)
    try:
        resultados = modelo.predecir_lote(pd.DataFrame(entrada.registros), capacidad)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail="Lote incompatible con el modelo.") from error
    return resultados.astype(object).where(pd.notna(resultados), None).to_dict(orient="records")


@api_router.get("/model/operating-point", response_model=PuntoOperacion)
def punto_operacion(
    capacidad: float = Query(ge=0, le=1),
    modelo=Depends(modelo_actual),
) -> dict:
    return modelo.punto_operacion(capacidad)
