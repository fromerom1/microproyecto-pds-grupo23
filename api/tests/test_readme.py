from pathlib import Path


README = Path(__file__).resolve().parents[1] / "README.md"


def test_comandos_normales_sin_alias_obligatorio() -> None:
    contenido = README.read_text(encoding="utf-8")
    instrucciones = contenido.split("## Nota para Windows")[0]

    assert "python -m uvicorn api.main:app --reload" in instrucciones
    assert "python -m pytest -q" in instrucciones
    assert "subst X:" not in instrucciones


def test_documenta_origen_de_categorias() -> None:
    contenido = README.read_text(encoding="utf-8")

    assert "CATEGORIAS" in contenido
    assert "src/preprocessing.py" in contenido
    assert "Los JSON no definen las categorías" in contenido
