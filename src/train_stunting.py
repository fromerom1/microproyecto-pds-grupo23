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

Experimentos MLflow multi-familia:
  * --model rf : Random Forest   (n_estimators, max_depth, max_features)
  * --model lr : Regr. Logistica (C, penalty)   <- regularizacion como "hiperparametro"
  * --model gb : Gradient Boosting (n_estimators, max_depth)

Uso (desde la raiz del repo, para que src/ sea importable):
    python -m src.train_stunting --model lr --C 0.1
    python -m src.train_stunting --model rf --n-estimators 500 --max-depth 0 --max-features 8
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # VM sin interfaz grafica
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, average_precision_score, roc_auc_score,
                             recall_score, accuracy_score, ConfusionMatrixDisplay)

import mlflow
import mlflow.sklearn

SEED = 42

# ---------------------------------------------------------------------------
# 1-2. CONTRATO DE MODELADO Y PREPROCESAMIENTO
# ---------------------------------------------------------------------------
# Viven en src/preprocessing.py (fuente unica de verdad). Motivo: si Winsorizer
# se define aqui, al ejecutar `python train_stunting.py` queda pickleada como
# `__main__.Winsorizer` y el modelo registrado en MLflow no puede cargarse desde
# ningun otro proceso (tablero, API, contenedor). Importada desde un modulo
# queda como `src.preprocessing.Winsorizer`, que si es importable.
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.preprocessing import (CAT_SIN_MISSING, CAT_CON_MISSING, NUMERICAS,   # noqa: E402,F401
                               FEATURES, TARGET, Winsorizer, build_preprocessor)


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
# 4. CONSTRUCCION DEL CLASIFICADOR SEGUN CLI
# ---------------------------------------------------------------------------
def build_classifier(args):
    """Familia de modelo elegible por CLI para los experimentos MLflow."""
    if args.model == "lr":
        # solver liblinear soporta l1 y l2; C pequeno = mas regularizacion
        return LogisticRegression(C=args.C, penalty=args.penalty, solver="liblinear",
                                  max_iter=1000, class_weight="balanced",
                                  random_state=SEED)
    if args.model == "gb":
        return GradientBoostingClassifier(n_estimators=args.n_estimators,
                                          max_depth=3 if args.max_depth == 0 else args.max_depth,
                                          learning_rate=0.05, random_state=SEED)
    return RandomForestClassifier(n_estimators=args.n_estimators,
                                  max_depth=None if args.max_depth == 0 else args.max_depth,
                                  max_features=args.max_features,
                                  class_weight="balanced", random_state=SEED)


def importancias(clf, pre):
    """Importancia agregada por variable original:
    arboles -> feature_importances_; lineales -> |coef_|; si no, None."""
    nombres = pre.get_feature_names_out()
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0])
    else:
        return None
    def orig_var(n):
        pref, rest = n.split("__")
        return rest if pref == "num" else rest.rsplit("_", 1)[0]
    return (pd.Series(imp, index=[orig_var(n) for n in nombres])
            .groupby(level=0).sum().sort_values(ascending=False))


# ---------------------------------------------------------------------------
# 5. ENTRENAMIENTO + METRICAS + MLFLOW
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="model_dataset.csv")
    ap.add_argument("--model", default="rf", choices=["rf", "lr", "gb"],
                    help="Familia del modelo: rf (default), lr o gb")
    # Hiperparametros RF / GB
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=6)      # 0 = sin limite (rf)
    ap.add_argument("--max-features", type=int, default=4)
    # Hiperparametros LR (regularizacion)
    ap.add_argument("--C", type=float, default=1.0)          # menor C = mas regularizacion
    ap.add_argument("--penalty", default="l2", choices=["l1", "l2"])
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
        ("clf",  build_classifier(args)),
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
    print("[TEST]");          m_test = evalua(test[FEATURES], test.y, "test")

    # --- figuras para el analisis visual del reporte (artefactos MLflow) ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_predictions(test.y, modelo.predict(test[FEATURES]),
                                            ax=axes[0], cmap="Blues")
    axes[0].set_title(f"Matriz de confusion (test) - {args.model}")
    serie = importancias(modelo.named_steps["clf"], modelo.named_steps["prep"])
    if serie is not None:
        top = serie.head(10)
        axes[1].barh(top.index[::-1], top.values[::-1], color="#e67e22")
        axes[1].set_title("Top 10 variables (importancia)")
    else:
        axes[1].axis("off")
    plt.tight_layout()
    fig.savefig("run_summary.png", dpi=120, bbox_inches="tight")

    # --- MLflow (misma logica del Taller 4) ---
    # En la VM: servidor en la MISMA carpeta del script:
    #   mlflow server -h 0.0.0.0 -p 8050 --allowed-hosts "localhost:8050,<IP>:8050"
    mlflow.set_experiment("stunting-baseline-multi")
    run_name = (f"lr_C{args.C}_{args.penalty}" if args.model == "lr"
                else f"{args.model}_{args.n_estimators}_d{args.max_depth}_f{args.max_features}")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model", args.model)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth if args.max_depth else None)
        mlflow.log_param("max_features", args.max_features)
        mlflow.log_param("C", args.C)
        mlflow.log_param("penalty", args.penalty)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("bloque_features", "16 basales (tiempo cero)")
        mlflow.log_param("target", TARGET)
        mlflow.log_param("split", "grouped 70/15/15 por newid")
        mlflow.log_param("seed", SEED)
        for k, v in {**m_val, **m_test}.items():
            mlflow.log_metric(k, v)
        mlflow.log_artifact("run_summary.png")
        try:
            mlflow.sklearn.log_model(modelo, name="stunting-model")
        except TypeError:                      # MLflow 2.22 (VM del taller) usa artifact_path
            mlflow.sklearn.log_model(modelo, artifact_path="stunting-model")
        print("\nRun registrada en MLflow.")


if __name__ == "__main__":
    main()
