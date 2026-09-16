import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.modelos import modelos_disponibles
from src.predict import ModeloRiesgo


@pytest.fixture(scope="module")
def modelo() -> ModeloRiesgo:
    return ModeloRiesgo("24m")


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    with TestClient(app) as client:
        yield client


def registro_demo(modelo: ModeloRiesgo) -> dict:
    registro = {
        "enrol_hiv_status_cat": "Negative",
        "momage_cat": "25 and less",
        "educ_cat_n": "Primary or below",
        "marital_cat": "Married",
        "wealth_quintile": "lowest quintile",
        "depression": "no depression",
        "mom_muac_cat": "normal",
        "b1_sex": "female",
        "preterm": "yes",
        "hfia_enr": "severe",
        "parity": "multi",
        "enrol_anemia": "no",
        "caesarean": "no",
        "sga": "yes",
        "lbw": "yes",
        "gestage_final": 36.0,
    }
    for feature, value in {"zlen_nac": -1.4, "zwei_nac": -1.1, "zhc_nac": -0.6}.items():
        if feature in modelo.features:
            registro[feature] = value
    return registro


def test_predict_matches_model_with_defaults(cliente: TestClient, modelo: ModeloRiesgo) -> None:
    registro = registro_demo(modelo)
    response = cliente.post("/predict", json={"registro": registro})

    assert response.status_code == 200
    assert response.json() == modelo.predecir(registro)


def test_predict_accepts_horizon_and_variant(cliente: TestClient, modelo: ModeloRiesgo) -> None:
    variantes = modelos_disponibles()[modelo.horizonte]
    alternativa = next((valor for valor in variantes if valor != modelo.variante), None)
    if alternativa is None:
        pytest.skip("No hay otra variante disponible.")
    expected_model = ModeloRiesgo(modelo.horizonte, alternativa)
    registro = {
        campo: valor
        for campo, valor in registro_demo(modelo).items()
        if campo in expected_model.features
    }
    if set(expected_model.features) - set(registro):
        pytest.skip("La variante requiere campos ajenos al caso demo.")
    response = cliente.post(
        f"/predict?horizonte={modelo.horizonte}&variante={alternativa}",
        json={"registro": registro},
    )

    assert response.status_code == 200
    assert response.json() == expected_model.predecir(registro)


def test_predict_accepts_other_horizon(cliente: TestClient, modelo: ModeloRiesgo) -> None:
    alternativo = next(
        (valor for valor in modelos_disponibles() if valor != modelo.horizonte), None
    )
    if alternativo is None:
        pytest.skip("No hay otro horizonte disponible.")
    expected_model = ModeloRiesgo(alternativo)
    registro = {
        campo: valor
        for campo, valor in registro_demo(modelo).items()
        if campo in expected_model.features
    }
    if set(expected_model.features) - set(registro):
        pytest.skip("El horizonte requiere campos ajenos al caso demo.")

    response = cliente.post(
        f"/predict?horizonte={alternativo}", json={"registro": registro}
    )

    assert response.status_code == 200
    assert response.json() == expected_model.predecir(registro)


def test_predict_rejects_invalid_body(cliente: TestClient) -> None:
    response = cliente.post("/predict", json={"registro": []})

    assert response.status_code == 422


def test_predict_rejects_missing_feature(cliente: TestClient, modelo: ModeloRiesgo) -> None:
    registro = registro_demo(modelo)
    registro.pop(modelo.features[-1])

    response = cliente.post("/predict", json={"registro": registro})

    assert response.status_code == 422
    assert modelo.features[-1] in response.json()["detail"]["campos_faltantes"]


def test_predict_rejects_unknown_category(cliente: TestClient, modelo: ModeloRiesgo) -> None:
    registro = registro_demo(modelo)
    campo = next(iter(modelo.opciones()))
    registro[campo] = "desconocida"

    response = cliente.post("/predict", json={"registro": registro})

    assert response.status_code == 422
    assert response.json()["detail"] == {"categoria_invalida": campo}


def test_predict_rejects_unknown_field(cliente: TestClient, modelo: ModeloRiesgo) -> None:
    registro = registro_demo(modelo)
    registro["campo_ajeno"] = "valor"

    response = cliente.post("/predict", json={"registro": registro})

    assert response.status_code == 422
    assert response.json()["detail"] == {"campos_desconocidos": ["campo_ajeno"]}


def test_predict_rejects_invalid_number(cliente: TestClient, modelo: ModeloRiesgo) -> None:
    if not modelo.features_extra:
        pytest.skip("El modelo no tiene campos numéricos adicionales.")
    registro = registro_demo(modelo)
    registro[modelo.features_extra[0]] = "invalido"

    response = cliente.post("/predict", json={"registro": registro})

    assert response.status_code == 422


def test_predict_batch_matches_model_and_sorts_risk(
    cliente: TestClient, modelo: ModeloRiesgo
) -> None:
    primero = registro_demo(modelo)
    segundo = registro_demo(modelo)
    segundo["gestage_final"] = 40.0
    primero["newid"] = "A"
    segundo["newid"] = "B"
    registros = [primero, segundo]
    expected = modelo.predecir_lote(
        pd.DataFrame(registros), capacidad=0.5
    ).to_dict(orient="records")
    response = cliente.post(
        "/predict/batch?capacidad=0.5", json={"registros": registros}
    )

    assert response.status_code == 200
    assert response.json() == expected
    assert [row["ranking"] for row in response.json()] == [1, 2]


def test_predict_batch_rejects_empty_list(cliente: TestClient) -> None:
    response = cliente.post("/predict/batch", json={"registros": []})

    assert response.status_code == 422


@pytest.mark.parametrize("capacidad", [-0.01, 1.01])
def test_predict_batch_rejects_capacity_outside_range(
    cliente: TestClient, modelo: ModeloRiesgo, capacidad: float
) -> None:
    response = cliente.post(
        f"/predict/batch?capacidad={capacidad}",
        json={"registros": [registro_demo(modelo)]},
    )

    assert response.status_code == 422


def test_model_info_matches_model_and_exposes_features(
    cliente: TestClient, modelo: ModeloRiesgo
) -> None:
    response = cliente.get("/model/info")
    expected = {
        **modelo.info,
        "features": modelo.features,
        "features_extra": modelo.features_extra,
        "variantes_disponibles": modelos_disponibles()[modelo.horizonte],
    }

    assert response.status_code == 200
    assert response.json() == expected


def test_model_options_are_filtered_to_model_features(
    cliente: TestClient, modelo: ModeloRiesgo
) -> None:
    response = cliente.get("/model/options")
    expected = {
        key: values
        for key, values in ModeloRiesgo.opciones().items()
        if key in modelo.features
    }

    assert response.status_code == 200
    assert response.json() == expected


def test_operating_point_matches_model(cliente: TestClient, modelo: ModeloRiesgo) -> None:
    capacity = 0.2
    response = cliente.get(f"/model/operating-point?capacidad={capacity}")

    assert response.status_code == 200
    assert response.json() == modelo.punto_operacion(capacity)


def test_operating_point_rejects_invalid_capacity(cliente: TestClient) -> None:
    response = cliente.get("/model/operating-point?capacidad=1.1")

    assert response.status_code == 422


def test_model_info_reports_missing_model(cliente: TestClient) -> None:
    response = cliente.get("/model/info?horizonte=99m")

    assert response.status_code == 404
