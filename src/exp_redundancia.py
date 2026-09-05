#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microproyecto PDS - Grupo 23 - Entrega 2
Experimento: que pasa al quitar la redundancia preterm / gestage_final.

Motivacion (chequeos de calidad, seccion 1 del informe):
    'preterm' es una funcion determinista de 'gestage_final' -prematuro es
    'edad gestacional < 37 semanas'- y el cruce no arroja una sola
    discrepancia en los 333 bebes. Ambas variables entran hoy al contrato
    de 16 basales, de modo que el modelo recibe la misma informacion dos
    veces: colinealidad para la logistica y dilucion de la importancia
    entre las dos columnas en los modelos de arbol.

El experimento compara cuatro contratos de variables sobre el mismo
pipeline y la misma particion, con la evaluacion que ya usa el equipo
(validacion cruzada estratificada repetida 5x10):

    A  completo   las 16 basales (contrato actual)
    B  sin_pt     16 menos 'preterm'      -> se queda la continua
    C  sin_gest   16 menos 'gestage_final'-> se queda la binaria
    D  ninguna    16 menos las dos        -> control

Si A, B y C son equivalentes dentro del intervalo, la redundancia no
cuesta desempeno pero sobra por parsimonia e interpretabilidad. Si B o C
mejora, la colinealidad si estaba estorbando. D dice cuanto aporta el
bloque gestacional en conjunto.

Uso local:  python exp_redundancia.py --data <ruta>/model_dataset.csv
Uso en VM:  python3 exp_redundancia.py --data data/processed/model_dataset.csv \
                --tracking-uri http://<IP-PUBLICA>:8050
"""

import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

SEED = 42

# --- contrato del EDA (las 16 basales de tiempo cero) -----------------------
CAT_SIN_MISSING = ['enrol_hiv_status_cat', 'momage_cat', 'educ_cat_n',
                   'marital_cat', 'wealth_quintile', 'depression',
                   'mom_muac_cat', 'b1_sex', 'preterm']
CAT_CON_MISSING = ['hfia_enr', 'parity', 'enrol_anemia', 'caesarean', 'sga', 'lbw']
NUMERICAS = ['gestage_final']
TARGET = 'stunted_24'

CONTRATOS = {
    'A_completo': [],                          # nada que quitar
    'B_sin_preterm': ['preterm'],
    'C_sin_gestage': ['gestage_final'],
    'D_sin_ambas': ['preterm', 'gestage_final'],
}


class Winsorizer(BaseEstimator, TransformerMixin):
    """Recorta edades gestacionales implausibles (EDA: 4 casos fuera de 25-44)."""

    def __init__(self, low=25.0, high=44.0):
        self.low, self.high = low, high

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low, self.high)


def construye_preprocesador(cat_ok, cat_na, num):
    bloques = []
    if num:
        bloques.append(("num", Pipeline([
            ("winsor", Winsorizer()),
            ("imputa", SimpleImputer(strategy="median")),
            ("escala", StandardScaler()),
        ]), num))
    if cat_ok:
        bloques.append(("cat_ok", Pipeline([
            ("imputa", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_ok))
    if cat_na:
        bloques.append(("cat_na", Pipeline([
            ("imputa", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_na))
    return ColumnTransformer(bloques)


def evalua(df, quitar, modelo_nombre, cv):
    """Devuelve media, desviacion e intervalo del 95 % de ROC-AUC y PR-AUC."""
    cat_ok = [c for c in CAT_SIN_MISSING if c not in quitar]
    cat_na = [c for c in CAT_CON_MISSING if c not in quitar]
    num = [c for c in NUMERICAS if c not in quitar]
    feats = cat_ok + cat_na + num

    clf = (LogisticRegression(max_iter=2000, class_weight="balanced",
                              random_state=SEED)
           if modelo_nombre == "logistica" else
           RandomForestClassifier(n_estimators=400, max_depth=6, max_features=4,
                                  class_weight="balanced", random_state=SEED))

    pipe = Pipeline([("prep", construye_preprocesador(cat_ok, cat_na, num)),
                     ("clf", clf)])

    res = cross_validate(pipe, df[feats], df["y"], cv=cv,
                         scoring=["roc_auc", "average_precision"],
                         n_jobs=-1, error_score="raise")
    salida = {"n_variables": len(feats)}
    for metrica, clave in [("roc_auc", "test_roc_auc"),
                           ("pr_auc", "test_average_precision")]:
        v = res[clave]
        salida[metrica] = {
            "media": float(v.mean()),
            "desv": float(v.std()),
            "ic_bajo": float(np.percentile(v, 2.5)),
            "ic_alto": float(np.percentile(v, 97.5)),
        }
    return salida


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/model_dataset.csv")
    ap.add_argument("--tracking-uri", default=None,
                    help="URI de MLflow; si se omite, registra en ./mlruns")
    ap.add_argument("--salida-fig", default="fig_redundancia.png")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    antes = len(df)
    # los bebes sin desenlace no son negativos: se excluyen, no se imputan
    df = df[df[TARGET].notna()].reset_index(drop=True)
    df["y"] = (df[TARGET] == "yes").astype(int)
    print(f"Bebes: {antes} en el archivo, {len(df)} con desenlace a 24 meses "
          f"({antes - len(df)} sin dato, excluidos)")
    print(f"Prevalencia: {df.y.mean():.3f}")

    # verificacion de la redundancia sobre los datos de este experimento
    g = pd.to_numeric(df["gestage_final"], errors="coerce")
    pt = df["preterm"].astype(str).str.strip().str.lower()
    incoherentes = int((((pt == "yes") & (g >= 37)) |
                        ((pt == "no") & (g < 37))).sum())
    print(f"Cruce preterm vs gestage_final < 37: {incoherentes} discrepancias")

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    resultados = {}
    for modelo in ("logistica", "random_forest"):
        print(f"\n=== {modelo} ===")
        print(f"{'contrato':<16}{'vars':>5}{'ROC-AUC':>26}{'PR-AUC':>26}")
        for nombre, quitar in CONTRATOS.items():
            r = evalua(df, quitar, modelo, cv)
            resultados[f"{modelo}|{nombre}"] = r
            ra, pa = r["roc_auc"], r["pr_auc"]
            print(f"{nombre:<16}{r['n_variables']:>5}"
                  f"{ra['media']:>12.3f} [{ra['ic_bajo']:.2f}-{ra['ic_alto']:.2f}]"
                  f"{pa['media']:>12.3f} [{pa['ic_bajo']:.2f}-{pa['ic_alto']:.2f}]")

    # --- figura comparativa ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    etiquetas = list(CONTRATOS)
    for ax, modelo in zip(axes, ("logistica", "random_forest")):
        medias = [resultados[f"{modelo}|{n}"]["roc_auc"]["media"] for n in etiquetas]
        bajos = [resultados[f"{modelo}|{n}"]["roc_auc"]["ic_bajo"] for n in etiquetas]
        altos = [resultados[f"{modelo}|{n}"]["roc_auc"]["ic_alto"] for n in etiquetas]
        err = [np.array(medias) - np.array(bajos), np.array(altos) - np.array(medias)]
        ax.errorbar(range(len(etiquetas)), medias, yerr=err, fmt="o",
                    capsize=5, color="#2c6fad")
        ax.axhline(0.5, ls="--", lw=0.8, color="#999999")
        ax.set_xticks(range(len(etiquetas)))
        ax.set_xticklabels([e.split("_", 1)[1] for e in etiquetas],
                           rotation=20, ha="right", fontsize=8)
        ax.set_title(modelo, fontsize=10)
        ax.set_ylabel("ROC-AUC (CV 5x10, IC 95 %)" if modelo == "logistica" else "")
        ax.set_ylim(0.35, 0.85)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    fig.suptitle("Efecto de retirar la redundancia preterm / gestage_final",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(args.salida_fig, dpi=150, bbox_inches="tight")
    print(f"\nFigura -> {args.salida_fig}")

    # --- MLflow ---
    try:
        import mlflow
        if args.tracking_uri:
            mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment("stunting-redundancia-gestacional")
        for clave, r in resultados.items():
            modelo, contrato = clave.split("|")
            with mlflow.start_run(run_name=f"{modelo}_{contrato}"):
                mlflow.log_param("modelo", modelo)
                mlflow.log_param("contrato", contrato)
                mlflow.log_param("variables_retiradas",
                                 ",".join(CONTRATOS[contrato]) or "ninguna")
                mlflow.log_param("n_variables", r["n_variables"])
                mlflow.log_param("cv", "RepeatedStratifiedKFold 5x10")
                mlflow.log_param("target", TARGET)
                for m in ("roc_auc", "pr_auc"):
                    for k, v in r[m].items():
                        mlflow.log_metric(f"{m}_{k}", v)
                mlflow.log_metric("discrepancias_preterm_gestage", incoherentes)
        mlflow.log_artifact(args.salida_fig) if mlflow.active_run() else None
        print("Runs registradas en MLflow.")
    except Exception as e:                       # el experimento vale sin MLflow
        print(f"[aviso] no se registro en MLflow: {e}")

    with open("resultados_redundancia.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print("Resultados -> resultados_redundancia.json")


if __name__ == "__main__":
    main()
