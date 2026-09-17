"""Punto de entrada para ``python -m api``."""

import uvicorn

from api.config import host, puerto


def main() -> None:
    """Inicia el servidor HTTP con la configuracion actual."""
    uvicorn.run("api.main:app", host=host(), port=puerto())


if __name__ == "__main__":
    main()
