from fastapi.testclient import TestClient

from api.main import app
from api.modelos import obtener_modelo
from api.tests.test_model_endpoints import registro_demo


def test_openapi_declara_404_y_503() -> None:
    with TestClient(app) as client:
        esquema = client.get("/openapi.json").json()

    rutas = {
        ("/model/info", "get"),
        ("/model/options", "get"),
        ("/predict", "post"),
        ("/predict/batch", "post"),
        ("/model/operating-point", "get"),
    }
    for ruta, metodo in rutas:
        respuestas = esquema["paths"][ruta][metodo]["responses"]
        assert {"404", "503"} <= respuestas.keys()
    assert "503" in esquema["paths"]["/model/variants"]["get"]["responses"]


def test_openapi_describe_fila_del_lote() -> None:
    with TestClient(app) as client:
        esquema = client.get("/openapi.json").json()

    modelos = esquema["components"]["schemas"]
    assert modelos["ResultadosLote"]["items"]["$ref"].endswith("/FilaLote")
    fila = modelos["FilaLote"]
    assert {"probabilidad", "banda", "percentil", "ranking", "variante"} <= set(
        fila["required"]
    )
    assert fila["additionalProperties"] is True


def test_lote_sin_capacidad_conserva_null_y_omite_seguimiento() -> None:
    registro = registro_demo(obtener_modelo())
    registro["sga"] = None
    registro["newid"] = None
    with TestClient(app) as client:
        response = client.post("/predict/batch", json={"registros": [registro]})

    assert response.status_code == 200
    assert response.json()[0]["sga"] is None
    assert response.json()[0]["newid"] is None
    assert "seguimiento" not in response.json()[0]
