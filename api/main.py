"""Aplicación FastAPI; ejecutar desde la raíz del repositorio."""

from fastapi import FastAPI

from api.api import api_router

app = FastAPI(title="API de riesgo nutricional - Grupo 23")
app.include_router(api_router)
