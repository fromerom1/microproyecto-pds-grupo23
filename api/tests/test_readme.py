from pathlib import Path


README = Path(__file__).resolve().parents[1] / "README.md"


def test_comandos_de_arranque_y_pruebas() -> None:
    contenido = README.read_text(encoding="utf-8")

    assert "python -m api" in contenido
    assert "python -m pytest" in contenido
    assert "dvc pull" in contenido
    assert "dvc repro features" in contenido
    assert ".venv\\Scripts\\python.exe" in contenido
    assert "Activate.ps1" not in contenido
