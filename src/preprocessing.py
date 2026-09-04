"""
Contrato de modelado y preprocesamiento — fuente única de verdad.

Todo lo que necesita saber un modelo sobre los datos vive aqui: que columnas
son features, cual es el target, y como se transforman. Lo importan el
entrenamiento (train_stunting.py), la evaluacion (evaluate_cv.py), la
prediccion (predict.py) y, mas adelante, la API.

Por que existe este modulo y no esta todo dentro de cada script:
    Un Pipeline de scikit-learn se serializa con pickle, y pickle guarda la
    RUTA de importacion de cada clase. Si `Winsorizer` se define dentro de un
    notebook o de un script ejecutado directamente, queda registrada como
    `__main__.Winsorizer`, y ningun otro proceso puede volver a cargar el
    modelo. Definiendola aqui queda como `src.preprocessing.Winsorizer`, que
    es importable desde el tablero, la API o un contenedor.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Contrato de modelado (EDA, Paso 9.1): 16 variables basales, todas conocidas
# en el momento del ingreso al programa. Ninguna posterior al parto.
# ---------------------------------------------------------------------------
CAT_SIN_MISSING = [
    "enrol_hiv_status_cat", "momage_cat", "educ_cat_n", "marital_cat",
    "wealth_quintile", "depression", "mom_muac_cat", "b1_sex", "preterm",
]
CAT_CON_MISSING = ["hfia_enr", "parity", "enrol_anemia", "caesarean", "sga", "lbw"]
NUMERICAS = ["gestage_final"]

FEATURES = CAT_SIN_MISSING + CAT_CON_MISSING + NUMERICAS
TARGETS = {"24m": "stunted_24", "12m": "stunted_12"}
TARGET = TARGETS["24m"]          # horizonte primario
ID_COL = "newid"
SEED = 42

# Etiquetas legibles para el tablero y las figuras
FEATURE_LABELS = {
    "enrol_hiv_status_cat": "Estado VIH materno",
    "momage_cat":           "Edad materna",
    "educ_cat_n":           "Educación materna",
    "marital_cat":          "Estado marital",
    "wealth_quintile":      "Quintil de riqueza",
    "depression":           "Depresión materna",
    "mom_muac_cat":         "Nutrición materna (MUAC)",
    "b1_sex":               "Sexo del bebé",
    "preterm":              "Prematuro",
    "hfia_enr":             "Inseguridad alimentaria",
    "parity":               "Paridad",
    "enrol_anemia":         "Anemia materna",
    "caesarean":            "Cesárea",
    "sga":                  "Pequeño para edad gestacional",
    "lbw":                  "Bajo peso al nacer",
    "gestage_final":        "Edad gestacional (sem)",
}

# Valores validos por variable categorica, tal como vienen en el dataset.
# El tablero los usa para construir los selectores sin inventar categorias.
CATEGORIAS = {
    "enrol_hiv_status_cat": ["Negative", "Positive"],
    "momage_cat":           ["25 and less", "26 to 35", "35 and more"],
    "educ_cat_n":           ["Primary or below", "Secondary and above"],
    "marital_cat":          ["Married", "Other"],
    "wealth_quintile":      ["lowest quintile", "quintile2", "quintile3", "quintile4", "highest quintile"],
    "depression":           ["no depression", "mild", "moderate or severe"],
    "mom_muac_cat":         ["undernutrition", "normal", "above normal"],
    "b1_sex":               ["female", "male"],
    "preterm":              ["no", "yes"],
    "hfia_enr":             ["secured", "mild", "moderate", "severe"],
    "parity":               ["nulli", "multi"],
    "enrol_anemia":         ["no", "yes"],
    "caesarean":            ["no", "yes"],
    "sga":                  ["no", "yes"],
    "lbw":                  ["no", "yes"],
}


class Winsorizer(BaseEstimator, TransformerMixin):
    """Recorta edades gestacionales biologicamente implausibles (EDA, Paso 6)."""

    def __init__(self, low: float = 25.0, high: float = 44.0):
        self.low, self.high = low, high

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low, self.high)

    def get_feature_names_out(self, input_features=None):
        # Sin esto, ColumnTransformer.get_feature_names_out() falla y no se
        # pueden mapear importancias ni valores SHAP a la variable original.
        return np.asarray(input_features, dtype=object)


def build_preprocessor(numericas_extra: list[str] | None = None) -> ColumnTransformer:
    """Preprocesamiento completo. Se ajusta SOLO con el fold de entrenamiento:
    el Pipeline que lo contiene garantiza que imputacion y escalado no vean test.

    `numericas_extra`: columnas numericas adicionales (p. ej. los z-scores de las
    visitas tempranas del experimento escalera). Se imputan por mediana y se
    escalan, pero NO se winsorizan: ese recorte es especifico de la edad
    gestacional."""
    pipe_num = Pipeline([
        ("winsor", Winsorizer(25.0, 44.0)),
        ("imputa", SimpleImputer(strategy="median")),
        ("escala", StandardScaler()),
    ])
    pipe_num_extra = Pipeline([
        ("imputa", SimpleImputer(strategy="median")),
        ("escala", StandardScaler()),
    ])
    pipe_cat_ok = Pipeline([
        ("imputa", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    # Para estas variables el faltante es informativo (EDA, Paso 4.4): se
    # conserva como categoria propia en vez de imputarse.
    pipe_cat_na = Pipeline([
        ("imputa", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    bloques = [
        ("num",    pipe_num,    NUMERICAS),
        ("cat_ok", pipe_cat_ok, CAT_SIN_MISSING),
        ("cat_na", pipe_cat_na, CAT_CON_MISSING),
    ]
    if numericas_extra:
        bloques.append(("num_extra", pipe_num_extra, list(numericas_extra)))
    return ColumnTransformer(bloques)


def variable_original(nombre_transformado: str) -> str:
    """'cat_ok__wealth_quintile_quintile2' -> 'wealth_quintile'; 'num__gestage_final' -> 'gestage_final'.
    Permite agregar importancias o valores SHAP por variable original."""
    prefijo, resto = nombre_transformado.split("__", 1)
    if prefijo in ("num", "num_extra"):
        return resto
    # el nombre de la categoria puede contener '_' (p. ej. 'no_depression');
    # se busca la variable mas larga que sea prefijo del nombre
    candidatas = [f for f in CAT_SIN_MISSING + CAT_CON_MISSING if resto.startswith(f + "_")]
    return max(candidatas, key=len) if candidatas else resto


def cargar_dataset(path, target: str = TARGET):
    """Lee el dataset plano (una fila por bebe), descarta filas sin target y
    devuelve (X, y, df) con y binaria (1 = 'yes')."""
    import pandas as pd
    df = pd.read_csv(path)
    faltan = [c for c in [ID_COL] + FEATURES + [target] if c not in df.columns]
    if faltan:
        raise ValueError(f"El dataset no tiene las columnas: {faltan}")
    df = df[df[target].notna()].reset_index(drop=True)
    y = (df[target] == "yes").astype(int)
    return df[FEATURES].copy(), y, df
