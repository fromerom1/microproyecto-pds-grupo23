from pydantic import BaseModel


class ErrorRespuesta(BaseModel):
    detail: str
