from typing import Any

from pydantic import BaseModel, ConfigDict, Field, RootModel

Valor = str | int | float | bool | None


class RegistroEntrada(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registro: dict[str, Valor] = Field(min_length=1)


class LoteEntrada(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registros: list[dict[str, Valor]] = Field(min_length=1)


class Contribucion(BaseModel):
    variable: str
    etiqueta: str
    valor: str | float | None
    contribucion: float


class PrediccionRespuesta(BaseModel):
    horizonte: str
    probabilidad: float
    banda: str
    percentil: float
    contribuciones: list[Contribucion]
    unidad_contribucion: str


class ResultadosLote(RootModel[list[dict[str, Any]]]):
    pass
