#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microproyecto PDS - Grupo 23 - Entrega 2
Prediccion temprana de stunting (LAZ < -2) a 24 meses.
Pipeline de cleansing/transformacion + modelo base + MLflow.

Contrato de modelado (EDA Paso 9):
  * 16 features basales (10 maternas + 6 nacimiento) -> alerta en tiempo cero
  * Target primario: stunted_24
  * Particion AGRUPADA por bebe (newid) 70/15/15
  * Imputacion/balanceo SOLO dentro del fold de entrenamiento
  * Metricas: F1, PR-AUC, ROC-AUC, sensibilidad

Uso local: python train_stunting.py
Uso VM:    python3 train_stunting.py --n-estimators 500 --max-depth 0 --max-features 8
           (max-depth 0 = None/arboles sin limite)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # VM sin interfaz grafica
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, average_precision_score, roc_auc_score,
                             recall_score, accuracy_score, ConfusionMatrixDisplay)

import mlflow
import mlflow.sklearn

SEED = 42

# ---------------------------------------------------------------------------
# 1. CONFIGURACION (contrato de modelado del EDA)
# ---------------------------------------------------------------------------
CAT_SIN_MISSING = ['enrol_hiv_status_cat', 'momage_cat', 'educ_cat_n', 'marital_cat',
                   'wealth_quintile', 'depression', 'mom_muac_cat', 'b1_sex', 'preterm']
CAT_CON_MISSING = ['hfia_enr', 'parity', 'enrol_anemia', 'caesarean', 'sga', 'lbw']
NUMERICAS       = ['gestage_final']
FEATURES = CAT_SIN_MISSING + CAT_CON_MISSING + NUMERICAS
TARGET   = 'stunted_24'


# ---------------------------------------------------------------------------
# 2. PIPELINE DE PREPROCESAMIENTO (cleansing + normalizacion + transformacion)
# ---------------------------------------------------------------------------
class Winsorizer(BaseEstimator, TransformerMixin):
    """Recorta edades gestacionales biologicamente implausibles (>44 sem, EDA Paso 6)."""
    def __init__(self, low=25.0, high=44.0):
        self.low, self.high = low, high
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low, self.high)


def build_preprocessor():
    """
    Decisiones del EDA implementadas (se ajustan SOLO con el fold de train):
      * gestage_final: winsorizacion + imputacion mediana + estandarizacion
      * categoricas con faltantes: categoria 'missing' = indicador de faltante
      * categoricas completas: one-hot con handle_unknown (categorias raras)
    """
    pipe_num = Pipeline([
        ("winsor", Winsorizer(25.0, 44.0)),
        ("imputa", SimpleImputer(strategy="median")),
        ("escala", StandardScaler()),
    ])
    pipe_cat_ok = Pipeline([
        ("imputa", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pipe_cat_na = Pipeline([
        ("imputa", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num",    pipe_num,    NUMERICAS),
        ("cat_ok", pipe_cat_ok, CAT_SIN_MISSING),
        ("cat_na", pipe_cat_na, CAT_CON_MISSING),
    ])


# ---------------------------------------------------------------------------
# 3. PARTICION AGRUPADA POR BEBE (anti-leakage longitudinal)
# ---------------------------------------------------------------------------
def grouped_split(df, seed=SEED):
    g1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    tr, tmp_pos = next(g1.split(df, groups=df["newid"]))
    tmp = df.iloc[tmp_pos]
    g2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    va, te = next(g2.split(tmp, groups=tmp["newid"]))
    return df.iloc[tr], tmp.iloc[va], tmp.iloc[te]


# ---------------------------------------------------------------------------
# 4. ENTRENAMIENTO + METRICAS + MLFLOW
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="model_dataset.csv")
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=6)     # 0 = sin limite
    ap.add_argument("--max-features", type=int, default=4)
    args = ap.parse_args()

    # --- carga y target binario (positivo = 'yes') ---
    df = pd.read_csv(args.data)
    df = df.assign(y=(df[TARGET] == "yes").astype(int))

    train, val, test = grouped_split(df)
    print(f"Split agrupado por bebe: train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"Prevalencia target: train={train.y.mean():.2f}, val={val.y.mean():.2f}, test={test.y.mean():.2f}")

    # --- pipeline completo: el preprocesamiento vive DENTRO del artefacto ---
    modelo = Pipeline([
        ("prep", build_preprocessor()),
        ("clf",  RandomForestClassifier(
                    n_estimators=args.n_estimators,
                    max_depth=None if args.max_depth == 0 else args.max_depth,
                    max_features=args.max_features,
                    class_weight="balanced",
                    random_state=SEED)),
    ])

    modelo.fit(train[FEATURES], train.y)     # fitting SOLO con train

    def evalua(X, ytrue, tag):
        p  = modelo.predict(X)
        pr = modelo.predict_proba(X)[:, 1]
        m = {f"{tag}_acc":   accuracy_score(ytrue, p),
             f"{tag}_f1":    f1_score(ytrue, p, zero_division=0),
             f"{tag}_sens":  recall_score(ytrue, p, zero_division=0),
             f"{tag}_prauc": average_precision_score(ytrue, pr),
             f"{tag}_auc":   roc_auc_score(ytrue, pr)}
        print({k: round(v, 3) for k, v in m.items()})
        return m

    print("\n[VALIDACION]");  m_val  = evalua(val[FEATURES],  val.y,  "val")
    print("[TEST]");         m_test = evalua(test[FEATURES], test.y, "test")

    # --- figuras para el analisis visual del reporte (artefactos MLflow) ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_predictions(test.y, modelo.predict(test[FEATURES]),
                                            ax=axes[0], cmap="Blues")
    axes[0].set_title("Matriz de confusion (test)")

    nombres = modelo.named_steps["prep"].get_feature_names_out()
    imp = modelo.named_steps["clf"].feature_importances_
    def orig_var(n):
        pref, rest = n.split("__")
        return rest if pref == "num" else rest.rsplit("_", 1)[0]
    serie = (pd.Series(imp, index=[orig_var(n) for n in nombres])
             .groupby(level=0).sum().sort_values(ascending=False).head(10))
    axes[1].barh(serie.index[::-1], serie.values[::-1], color="#e67e22")
    axes[1].set_title("Top 10 variables (importancia)")
    plt.tight_layout()
    fig.savefig("run_summary.png", dpi=120, bbox_inches="tight")

    # --- MLflow (misma logica del Taller 4) ---
    # En la VM: correr el servidor en la MISMA carpeta del script:
    #   mlflow server -h 0.0.0.0 -p 8050 --allowed-hosts "localhost:8050,<IP>:8050"
    # mlflow.set_tracking_uri("http://<IP_PUBLICA>:8050")
    mlflow.set_experiment("stunting-baseline-rf")
    with mlflow.start_run(run_name=f"rf_{args.n_estimators}_d{args.max_depth}_f{args.max_features}"):
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth if args.max_depth else None)
        mlflow.log_param("max_features", args.max_features)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("bloque_features", "16 basales (tiempo cero)")
        mlflow.log_param("target", TARGET)
        mlflow.log_param("split", "grouped 70/15/15 por newid")
        mlflow.log_param("seed", SEED)
        for k, v in {**m_val, **m_test}.items():
            mlflow.log_metric(k, v)
        mlflow.log_artifact("run_summary.png")
        try:
            mlflow.sklearn.log_model(modelo, name="stunting-rf")
        except TypeError:                      # MLflow 2.22 (VM del taller) usa artifact_path
            mlflow.sklearn.log_model(modelo, artifact_path="stunting-rf")
        print("\nRun registrada en MLflow.")


if __name__ == "__main__":
    main()