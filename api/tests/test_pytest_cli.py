import os
import subprocess
import sys
from pathlib import Path


def test_pytest_importa_api_desde_raiz() -> None:
    ejecutable = Path(sys.executable).with_name(
        "pytest.exe" if os.name == "nt" else "pytest"
    )
    resultado = subprocess.run(
        [str(ejecutable), "--collect-only", "-q", "api/tests/test_health.py::test_health"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert resultado.returncode == 0, resultado.stdout + resultado.stderr
