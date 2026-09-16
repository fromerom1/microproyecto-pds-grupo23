"""Respuesta del chequeo de salud."""

from typing import Literal

from pydantic import BaseModel


class Health(BaseModel):
    status: Literal["ok"]
