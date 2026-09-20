"""Rutas públicas de la API."""

import logging
import os

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from api.config import ruta_cohorte
from api.modelos import (
    MetadatosInvalidos,
    modelos_disponibles,
    obtener_modelo,
    seleccionar_variante,
)
from api.schemas import (
    ErrorRespuesta,
    Health,
    InfoModelo,
    LoteEntrada,
    ModelosDisponibles,
    OpcionesModelo,
    PrediccionRespuesta,
    PuntoOperacion,
    RegistroEntrada,
    ResultadosLote,
)

api_router = APIRouter()
logger = logging.getLogger(__name__)
RESPUESTAS_MODELO = {
    404: {"model": ErrorRespuesta, "description": "Modelo no encontrado"},
    503: {"model": ErrorRespuesta, "description": "Modelo o cohorte no disponibles"},
}


class ServicioNoDisponible(Exception):
    pass


def motivo_503(error: Exception) -> str:
    if isinstance(error, ServicioNoDisponible):
        return str(error)
    return "El servicio no está disponible."


def modelo_listo(horizonte: str = "24m", variante: str | None = None):
    try:
        cohorte = ruta_cohorte()
    except FileNotFoundError as error:
        if os.environ.get("API_COHORTE_PATH"):
            raise ServicioNoDisponible("La cohorte de referencia no está disponible.") from error
        raise ServicioNoDisponible(
            "Falta data/processed/model_dataset_escalera.csv. "
            "Ejecuta dvc pull con credenciales del equipo; si no aparece, dvc repro features."
        ) from error
    except Exception as error:
        raise ServicioNoDisponible("La cohorte de referencia no está disponible.") from error
    try:
        modelo = obtener_modelo(horizonte, variante)
    except MetadatosInvalidos as error:
        raise ServicioNoDisponible(str(error)) from error
    except Exception as error:
        raise ServicioNoDisponible("No se pudo cargar el modelo seleccionado.") from error
    try:
        datos = pd.read_csv(cohorte)
    except Exception as error:
        raise ServicioNoDisponible("No se pudo leer la cohorte de referencia.") from error
    if datos.empty:
        raise ServicioNoDisponible("La cohorte de referencia no tiene filas.")
    if not set(modelo.features).issubset(datos.columns):
        raise ServicioNoDisponible("La cohorte no contiene las variables requeridas.")
    try:
        transformados = modelo._prep.transform(datos[modelo.features])
        referencia = np.asarray(transformados.mean(axis=0), dtype=float).reshape(-1)
        referencia_modelo = np.asarray(modelo._referencia, dtype=float).reshape(-1)
    except Exception as error:
        raise ServicioNoDisponible("No se pudo construir la referencia de la cohorte.") from error
    if not np.isfinite(referencia).all() or not np.allclose(referencia, referencia_modelo):
        raise ServicioNoDisponible("No se pudo construir la referencia de la cohorte.")
    return modelo


@api_router.get(
    "/health",
    response_model=Health,
    response_model_exclude_none=True,
    responses={503: {"model": Health}},
)
def health() -> Health | JSONResponse:
    """Comprueba que modelo y cohorte esten disponibles."""
    try:
        disponibles = modelos_disponibles()
        if not disponibles:
            raise ServicioNoDisponible("No hay modelos disponibles.")
        for horizonte, variantes in disponibles.items():
            for variante in variantes:
                modelo_listo(horizonte, variante)
    except Exception as error:
        logger.exception("Fallo en /health: %s", error)
        return JSONResponse(
            status_code=503,
            content=Health(status="error", detail=motivo_503(error)).model_dump(),
        )
    return Health(status="ok")


def modelo_actual(horizonte: str = "24m", variante: str | None = None):
    try:
        seleccionar_variante(horizonte, variante)
    except ValueError as error:
        if not modelos_disponibles():
            raise HTTPException(status_code=503, detail="No hay modelos disponibles.") from error
        raise HTTPException(status_code=404, detail=str(error)) from error
    try:
        return modelo_listo(horizonte, variante)
    except Exception as error:
        raise HTTPException(status_code=503, detail=motivo_503(error)) from error


def validar_registro(registro: dict, modelo, permitir_extras: bool = False) -> None:
    faltantes = [campo for campo in modelo.features if campo not in registro]
    if faltantes:
        raise HTTPException(status_code=422, detail={"campos_faltantes": faltantes})

    if not permitir_extras:
        extras = sorted(set(registro) - set(modelo.features))
        if extras:
            raise HTTPException(status_code=422, detail={"campos_desconocidos": extras})

    opciones = modelo.opciones()
    numericas = {
        campo
        for nombre, _, columnas in modelo._prep.transformers_
        if nombre.startswith("num")
        for campo in columnas
    }
    for campo in modelo.features:
        valor = registro[campo]
        if campo in numericas and valor is not None and (
            isinstance(valor, bool) or not isinstance(valor, (int, float))
        ):
            raise HTTPException(status_code=422, detail={"numero_invalido": campo})
        if valor is not None and campo in opciones and valor not in opciones[campo]:
            raise HTTPException(status_code=422, detail={"categoria_invalida": campo})


def normalizar_faltantes(registro: dict) -> dict:
    return {campo: np.nan if valor is None else valor for campo, valor in registro.items()}


@api_router.get(
    "/model/info",
    response_model=InfoModelo,
    responses=RESPUESTAS_MODELO,
)
def info_modelo(modelo=Depends(modelo_actual)) -> dict:
    return {
        **modelo.info,
        "features": modelo.features,
        "features_extra": modelo.features_extra,
        "variantes_disponibles": modelos_disponibles()[modelo.horizonte],
        "curva_capacidad": modelo.meta["curva_capacidad"],
    }


@api_router.get(
    "/model/variants",
    response_model=ModelosDisponibles,
    responses={503: RESPUESTAS_MODELO[503]},
)
def variantes_modelo() -> dict[str, list[str]]:
    disponibles = modelos_disponibles()
    if not disponibles:
        raise HTTPException(status_code=503, detail="No hay modelos disponibles.")
    return disponibles


@api_router.get("/model/options", response_model=OpcionesModelo, responses=RESPUESTAS_MODELO)
def opciones_modelo(modelo=Depends(modelo_actual)) -> dict:
    opciones = modelo.opciones()
    return {campo: opciones[campo] for campo in modelo.features if campo in opciones}


@api_router.post("/predict", response_model=PrediccionRespuesta, responses=RESPUESTAS_MODELO)
def predecir(entrada: RegistroEntrada, modelo=Depends(modelo_actual)) -> dict:
    validar_registro(entrada.registro, modelo)
    try:
        resultado = modelo.predecir(normalizar_faltantes(entrada.registro))
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail="Registro incompatible con el modelo.") from error
    return {**resultado, "variante": modelo.variante}


@api_router.post(
    "/predict/batch",
    response_model=ResultadosLote,
    response_model_exclude_unset=True,
    responses=RESPUESTAS_MODELO,
)
def predecir_lote(
    entrada: LoteEntrada,
    capacidad: float | None = Query(default=None, gt=0, le=1),
    modelo=Depends(modelo_actual),
) -> list[dict]:
    for registro in entrada.registros:
        validar_registro(registro, modelo, permitir_extras=True)
    try:
        registros = [normalizar_faltantes(registro) for registro in entrada.registros]
        resultados = modelo.predecir_lote(pd.DataFrame(registros), capacidad)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail="Lote incompatible con el modelo.") from error
    resultados["variante"] = modelo.variante
    return resultados.astype(object).where(pd.notna(resultados), None).to_dict(orient="records")


@api_router.get(
    "/model/operating-point", response_model=PuntoOperacion, responses=RESPUESTAS_MODELO
)
def punto_operacion(
    capacidad: float = Query(gt=0, le=1),
    modelo=Depends(modelo_actual),
) -> dict:
    capacidades = [punto["capacidad"] for punto in modelo.meta["curva_capacidad"]]
    minimo, maximo = min(capacidades), max(capacidades)
    if not minimo <= capacidad <= maximo:
        raise HTTPException(
            status_code=422,
            detail=f"capacidad debe estar entre {minimo} y {maximo} para este modelo.",
        )
    return modelo.punto_operacion(capacidad)
