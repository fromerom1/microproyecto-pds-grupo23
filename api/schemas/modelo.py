from typing import Any

from pydantic import BaseModel, RootModel


class PuntoOperacion(BaseModel):
    capacidad: float
    sensibilidad: float
    vpp: float
    umbral: float


class InfoModelo(BaseModel):
    horizonte: str
    variante: str
    peldano: str
    n_features: int
    target: str
    config: str | dict[str, Any]
    familia: str
    n_entrenamiento: int
    prevalencia: float
    roc_auc_cv: float | None
    roc_auc_ci: list[float | None]
    pr_auc_cv: float | None
    umbral_alto: float
    umbral_medio: float
    data_md5: str
    sklearn: str
    creado: str
    features: list[str]
    features_extra: list[str]
    variantes_disponibles: list[str]
    curva_capacidad: list[PuntoOperacion]


class OpcionesModelo(RootModel[dict[str, list[str]]]):
    pass


class ModelosDisponibles(RootModel[dict[str, list[str]]]):
    pass
