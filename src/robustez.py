#!/usr/bin/env python3
"""
Robustez de la seleccion de modelo y del punto de operacion.

Complementa a evaluate_cv.py y experimento_escalera.py respondiendo cuatro
preguntas que el barrido anterior deja abiertas:

    1. BASELINES. Un ROC-AUC de 0.72 no significa nada sin un punto de
       comparacion. Se evaluan, con la misma CV y las mismas metricas, la
       regla clinica que hoy se usaria sin modelo (marcar si PEG o BPN) y
       un clasificador trivial. Todo el valor del proyecto es superarlas.

    2. BRECHA DE OPTIMISMO. evaluate_cv.py comenta que "con n=333 la varianza
       de la evaluacion domina sobre cualquier ajuste fino de hiperparametros".
       Es plausible, pero nadie lo midio. Aqui se mide: una busqueda de
       hiperparametros evaluada con CV simple (optimista, porque los mismos
       folds eligen y puntuan) contra la misma busqueda con CV ANIDADA
       (honesta, porque el bucle externo nunca participa en la eleccion).
       La diferencia es cuanto del "aprendizaje" era ruido.

    3. REGLA DE SELECCION. evaluate_cv.py usa como piso `media - DE_entre_folds`,
       pero la regla de un error estandar usa el ERROR ESTANDAR DE LA MEDIA
       (DE/sqrt(k)), unas 7 veces mas estrecho con 50 folds. Con el piso
       implementado pasan las siete configuraciones y la seleccion devuelve
       siempre el primer elemento del diccionario. Se contrastan los tres
       criterios para verificar si la eleccion del equipo sobrevive.

    4. UMBRALES DE BANDA. evaluate_cv.py fija los umbrales de banda del tablero
       con `final.predict_proba(X)` sobre los mismos datos de entrenamiento.
       Las probabilidades en muestra estan mas dispersas que las reales, asi
       que las bandas exageran la separacion. Se recalculan out-of-fold y se
       cuantifica el desplazamiento. De paso se evalua la calibracion con y
       sin CalibratedClassifierCV, que es la limitacion que la Entrega 2
       declara y aplaza.

No modifica ningun archivo de los companeros: importa preprocessing.py y
features.py, y escribe en figures/04_robustez/ y models/ con nombres propios.

Uso:
    python -m src.robustez                          # peldano B, 24m, completo
    python -m src.robustez --target 12m
    python -m src.robustez --rapido                 # sin RF en la busqueda (~1 min)
    python -m src.robustez --peldano A

    MLflow remoto (EC2):  export MLFLOW_TRACKING_URI=http://<ip>:8050
    Sin esa variable registra en ./mlruns (local).
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform, randint
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold)
from sklearn.pipeline import Pipeline

import mlflow

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.preprocessing import (FEATURE_LABELS, FEATURES, ID_COL, SEED,  # noqa: E402
                               TARGETS, build_preprocessor)
from src.features import (ETIQUETA_PELDANO, ETIQUETAS_EXTRA,  # noqa: E402
                          columnas_peldano, numericas_peldano)

warnings.filterwarnings("ignore")

AZUL, ROJO, GRIS, VERDE, NARANJA = "#1f5f8b", "#a33235", "#8a929e", "#2c6e49", "#c1732a"
ETIQUETAS = {**FEATURE_LABELS, **ETIQUETAS_EXTRA}


# ===========================================================================
# 1. Baselines
# ===========================================================================
class ReglaClinica(BaseEstimator, ClassifierMixin):
    """La regla que hoy se usaria sin modelo: marcar al bebe si es pequeno para
    la edad gestacional (PEG) o si tiene bajo peso al nacer (BPN).

    Es la vara real del proyecto. Un modelo que no la supere no justifica su
    complejidad, por bueno que parezca su ROC-AUC en abstracto.

    Opera sobre el DataFrame crudo (antes del preprocesamiento) porque necesita
    las categorias originales 'yes'/'no'. Devuelve un score de 0, 1 o 2 segun
    cuantas condiciones se cumplan, escalado a [0, 1]: con dos niveles de riesgo
    el ordenamiento es menos plano que con una marca binaria.
    """

    def __init__(self, columnas=("sga", "lbw")):
        self.columnas = columnas

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        # Prevalencia observada en cada nivel de la regla: convierte el conteo
        # en una probabilidad interpretable en vez de un score arbitrario.
        s = self._score(X)
        self.tasas_ = {}
        for nivel in (0, 1, 2):
            m = s == nivel
            self.tasas_[nivel] = float(np.asarray(y)[m].mean()) if m.sum() > 0 else float(np.asarray(y).mean())
        return self

    def _score(self, X):
        return sum((X[c].astype(str).str.lower() == "yes").astype(int) for c in self.columnas).to_numpy()

    def predict_proba(self, X):
        s = self._score(X)
        p = np.array([self.tasas_.get(int(v), 0.0) for v in s])
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def baselines():
    """Dos varas de medir. Ninguna aprende estructura: son el suelo."""
    return {
        "baseline_trivial": lambda: DummyClassifier(strategy="prior", random_state=SEED),
        "baseline_clinico": lambda: ReglaClinica(),
    }


# ===========================================================================
# 2. Espacios de busqueda
# ===========================================================================
# Deliberadamente estrechos y sesgados hacia la regularizacion fuerte. Con 95
# positivos, un espacio amplio no encuentra el mejor modelo: encuentra la
# combinacion que mejor se ajusta al ruido de los folds de validacion. El
# numero de muestras (n_iter) se mantiene bajo por la misma razon.
ESPACIOS = {
    "lr": (
        lambda: LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED),
        {"clf__C": loguniform(1e-3, 1e1), "clf__penalty": ["l1", "l2"], "clf__solver": ["liblinear"]},
        12,
    ),
    "rf": (
        lambda: RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=1),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [2, 3, 4, 6],
         "clf__min_samples_leaf": randint(5, 30), "clf__max_features": ["sqrt", 0.5]},
        12,
    ),
    "gb": (
        lambda: GradientBoostingClassifier(random_state=SEED),
        {"clf__n_estimators": [80, 150, 250], "clf__learning_rate": loguniform(1e-2, 2e-1),
         "clf__max_depth": [2, 3], "clf__min_samples_leaf": randint(5, 30),
         "clf__subsample": [0.8, 1.0]},
        12,
    ),
}


# ===========================================================================
# Metricas
# ===========================================================================
def punto_operacion(y_true, prob, capacidad):
    """Misma definicion que evaluate_cv.py: marcar el `capacidad` de mayor
    riesgo y medir que se logra. Se replica en vez de importarse para que este
    modulo sea autocontenido si el otro cambia."""
    n_marcar = max(1, int(round(capacidad * len(prob))))
    orden = np.argsort(-prob)
    marcados = np.zeros(len(prob), dtype=bool)
    marcados[orden[:n_marcar]] = True
    pos = np.asarray(y_true) == 1
    sens = (marcados & pos).sum() / max(1, pos.sum())
    vpp = (marcados & pos).sum() / n_marcar
    return float(sens), float(vpp)


def metricas(y_true, prob, capacidad):
    sens, vpp = punto_operacion(y_true, prob, capacidad)
    cap = int(capacidad * 100)
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        f"sens_at_{cap}": sens,
        f"vpp_at_{cap}": vpp,
    }


def resumir(folds: list[dict]) -> dict:
    df = pd.DataFrame(folds)
    out = {}
    for c in df.columns:
        v = df[c].to_numpy(dtype=float)
        out[f"{c}_mean"] = float(v.mean())
        out[f"{c}_std"] = float(v.std(ddof=1))
        out[f"{c}_se"] = float(v.std(ddof=1) / np.sqrt(len(v)))
        out[f"{c}_ci_lo"] = float(np.percentile(v, 2.5))
        out[f"{c}_ci_hi"] = float(np.percentile(v, 97.5))
    return out


# ===========================================================================
# Evaluacion simple (una configuracion fija, CV repetida)
# ===========================================================================
def evaluar_fijo(constructor, X, y, cv, capacidad, extra, crudo=False):
    """`crudo=True` para la regla clinica, que necesita las columnas sin
    transformar. El resto pasa por el preprocesamiento del equipo."""
    folds, n = [], len(y)
    suma, cuenta = np.zeros(n), np.zeros(n)
    for tr, te in cv.split(X, y):
        est = constructor() if crudo else Pipeline([("prep", build_preprocessor(extra)), ("clf", constructor())])
        est.fit(X.iloc[tr], y.iloc[tr])
        prob = est.predict_proba(X.iloc[te])[:, 1]
        folds.append(metricas(y.iloc[te], prob, capacidad))
        suma[te] += prob
        cuenta[te] += 1
    return folds, suma / np.maximum(cuenta, 1)


def comparar_pareado(folds_a, folds_b, metrica="pr_auc"):
    """Diferencia PAREADA entre dos configuraciones sobre LOS MISMOS folds.

    Comparar medias de dos CV distintas es invalido: parte de la diferencia
    viene de que los folds no son los mismos. Con folds identicos la varianza
    entre particiones se cancela, y la diferencia por fold estima directamente
    la ventaja de A sobre B. Es la comparacion que decide si la busqueda de
    hiperparametros aporta algo real sobre la eleccion del equipo.
    """
    a = np.array([f[metrica] for f in folds_a])
    b = np.array([f[metrica] for f in folds_b])
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {
        "n_folds": int(len(d)),
        "media_a": float(a.mean()), "media_b": float(b.mean()),
        "dif_media": float(d.mean()), "dif_se": float(se),
        "dif_ic_lo": float(d.mean() - 1.96 * se), "dif_ic_hi": float(d.mean() + 1.96 * se),
        "folds_gana_a_pct": float((d > 0).mean() * 100),
    }


# ===========================================================================
# Busqueda de hiperparametros: no anidada (optimista) vs anidada (honesta)
# ===========================================================================
def buscar(familia, X, y, cv_interna, extra, n_iter, scoring="average_precision"):
    base, espacio, _ = ESPACIOS[familia]
    pipe = Pipeline([("prep", build_preprocessor(extra)), ("clf", base())])
    return RandomizedSearchCV(pipe, espacio, n_iter=n_iter, scoring=scoring,
                              cv=cv_interna, random_state=SEED, n_jobs=-1, refit=True)


def evaluar_anidado(familia, X, y, cv_externa, k_interno, capacidad, extra, n_iter):
    """CV anidada: el bucle interno elige hiperparametros, el externo puntua.
    El fold externo NUNCA participa en la eleccion, por eso su score es honesto.

    Devuelve tambien el mejor score interno de cada fold: su media es la
    estimacion OPTIMISTA (la que reportaria una busqueda sin anidar) y la
    diferencia contra el score externo es la brecha de optimismo.
    """
    folds, internos, elegidos = [], [], []
    n = len(y)
    suma, cuenta = np.zeros(n), np.zeros(n)
    for tr, te in cv_externa.split(X, y):
        cv_in = StratifiedKFold(n_splits=k_interno, shuffle=True, random_state=SEED)
        bus = buscar(familia, X, y, cv_in, extra, n_iter)
        bus.fit(X.iloc[tr], y.iloc[tr])
        prob = bus.predict_proba(X.iloc[te])[:, 1]
        folds.append(metricas(y.iloc[te], prob, capacidad))
        internos.append(float(bus.best_score_))
        elegidos.append({k.replace("clf__", ""): v for k, v in bus.best_params_.items()})
        suma[te] += prob
        cuenta[te] += 1
    return folds, np.array(internos), elegidos, suma / np.maximum(cuenta, 1)


# ===========================================================================
# Criterios de seleccion
# ===========================================================================
def comparar_criterios(tabla: pd.DataFrame, metrica: str, orden_simplicidad: list[str], k_folds: int):
    """Contrasta el piso implementado en evaluate_cv.py con la regla de un error
    estandar correcta. Devuelve una fila por criterio con su piso, sus
    candidatas y la configuracion que elegiria."""
    top = tabla[f"{metrica}_mean"].idxmax()
    m, de = tabla.loc[top, f"{metrica}_mean"], tabla.loc[top, f"{metrica}_std"]
    criterios = {
        "implementado (media - DE entre folds)": m - de,
        f"1-SE correcto (DE/sqrt({k_folds}))": m - de / np.sqrt(k_folds),
        "1-SE conservador (DE/sqrt(5))": m - de / np.sqrt(5),
        "maximo (sin regla)": m - 1e-12,
    }
    filas = []
    for nombre, piso in criterios.items():
        cand = [c for c in orden_simplicidad if tabla.loc[c, f"{metrica}_mean"] >= piso]
        filas.append({"criterio": nombre, "piso": round(float(piso), 4),
                      "n_candidatas": len(cand),
                      "candidatas": ", ".join(cand),
                      "elegida": cand[0] if cand else top})
    return pd.DataFrame(filas), top


# ===========================================================================
# Figuras
# ===========================================================================
def fig_brecha(res_nested: dict, path: Path, metrica="pr_auc"):
    fams = list(res_nested)
    interno = [res_nested[f]["interno_mean"] for f in fams]
    externo = [res_nested[f]["resumen"][f"{metrica}_mean"] for f in fams]
    err = [res_nested[f]["resumen"][f"{metrica}_se"] for f in fams]
    x = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - 0.19, interno, 0.38, label="CV interna (optimista)", color=NARANJA, alpha=.9)
    ax.bar(x + 0.19, externo, 0.38, yerr=err, label="CV anidada (honesta)", color=AZUL,
           alpha=.9, ecolor=GRIS, capsize=4)
    for i, (a, b) in enumerate(zip(interno, externo)):
        ax.annotate(f"−{a - b:.3f}", (i, max(a, b) + 0.015), ha="center", fontsize=9, color=ROJO)
    ax.set_xticks(x); ax.set_xticklabels(fams)
    ax.set_ylabel(metrica.upper().replace("_", "-"))
    ax.set_title("Brecha de optimismo de la búsqueda de hiperparámetros",
                 loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.3)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_baselines(tabla: pd.DataFrame, metrica: str, prevalencia: float, path: Path):
    t = tabla.sort_values(f"{metrica}_mean")
    colores = [ROJO if i.startswith("baseline") else AZUL for i in t.index]
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(t) + 1.8))
    yy = np.arange(len(t))
    ax.errorbar(t[f"{metrica}_mean"], yy,
                xerr=[t[f"{metrica}_mean"] - t[f"{metrica}_ci_lo"],
                      t[f"{metrica}_ci_hi"] - t[f"{metrica}_mean"]],
                fmt="none", ecolor=GRIS, capsize=4)
    ax.scatter(t[f"{metrica}_mean"], yy, color=colores, zorder=3, s=45)
    ref = 0.5 if metrica == "roc_auc" else prevalencia
    ax.axvline(ref, ls="--", color=GRIS, lw=1,
               label="Azar (0.5)" if metrica == "roc_auc" else f"Prevalencia ({prevalencia:.2f})")
    ax.set_yticks(yy); ax.set_yticklabels([ETIQ_CONF.get(i, i) for i in t.index])
    ax.set_xlabel(f"{metrica.upper().replace('_', '-')} — media e IC95 % entre folds")
    ax.set_title("Modelos frente a las varas de referencia", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, loc="lower right"); ax.grid(axis="x", alpha=.3)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_calibracion(y, oof_sin, oof_con, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (prob, etiqueta) in zip(axes, [(oof_sin, "Sin calibrar"), (oof_con, "Calibrada (isotónica)")]):
        pt, pp = calibration_curve(y, prob, n_bins=5, strategy="quantile")
        ax.plot([0, 1], [0, 1], "--", color=GRIS, lw=1)
        ax.plot(pp, pt, "o-", color=AZUL, lw=2)
        ax.set_xlabel("Probabilidad predicha"); ax.set_ylabel("Fracción observada")
        ax.set_title(f"{etiqueta} — Brier = {brier_score_loss(y, prob):.3f}",
                     loc="left", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=.3)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_umbrales(prob_in, prob_oof, cap, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(0, 1, 26)
    ax.hist(prob_in, bins=bins, alpha=.55, label="En muestra (usada hoy)", color=NARANJA)
    ax.hist(prob_oof, bins=bins, alpha=.55, label="Out-of-fold (honesta)", color=AZUL)
    for p, c, l in [(prob_in, NARANJA, "en muestra"), (prob_oof, AZUL, "out-of-fold")]:
        u = float(np.quantile(p, 1 - cap))
        ax.axvline(u, color=c, ls="--", lw=1.8, label=f"Umbral alto {l} = {u:.3f}")
    ax.set_xlabel("Probabilidad estimada"); ax.set_ylabel("Bebés")
    ax.set_title("Umbral de banda alta: en muestra vs. out-of-fold",
                 loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9); ax.grid(axis="y", alpha=.3)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


ETIQ_CONF = {
    "baseline_trivial": "Trivial (prevalencia)",
    "baseline_clinico": "Regla clínica (PEG o BPN)",
    "lr_C0.1": "LR C=0.1 (elegida por el equipo)",
    "lr_buscada": "LR con búsqueda anidada",
    "rf_buscada": "RF con búsqueda anidada",
    "gb_buscada": "GB con búsqueda anidada",
}


# ===========================================================================
# Principal
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data/processed/model_dataset_escalera.csv"))
    ap.add_argument("--peldano", default="B", choices=list("ABCD"))
    ap.add_argument("--target", default="24m", choices=list(TARGETS))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-repeats", type=int, default=10, help="repeticiones de la CV de evaluacion")
    ap.add_argument("--n-repeats-nested", type=int, default=2, help="repeticiones del bucle EXTERNO de la CV anidada")
    ap.add_argument("--k-interno", type=int, default=4)
    ap.add_argument("--capacity", type=float, default=0.20)
    ap.add_argument("--rapido", action="store_true", help="excluye random forest de la busqueda anidada")
    ap.add_argument("--experiment", default="stunting-robustez")
    ap.add_argument("--owner", default=None, help="etiqueta de autoria en MLflow")
    ap.add_argument("--out-dir", default=str(ROOT / "figures/04_robustez"))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    args = ap.parse_args()

    target = TARGETS[args.target]
    cap, cap_pct = args.capacity, int(args.capacity * 100)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir); models_dir.mkdir(parents=True, exist_ok=True)
    owner = args.owner or getpass.getuser()

    # ---- datos ------------------------------------------------------------
    cols = columnas_peldano(args.peldano)
    extra = [c for c in numericas_peldano(args.peldano) if c != "gestage_final"]
    df = pd.read_csv(args.data)
    faltan = [c for c in cols + [target] if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en {args.data}: {faltan}")
    df = df[df[target].notna()].reset_index(drop=True)
    X, y = df[cols].copy(), (df[target] == "yes").astype(int)
    data_hash = hashlib.md5(Path(args.data).read_bytes()).hexdigest()

    print(f"MLflow tracking: {os.environ.get('MLFLOW_TRACKING_URI') or './mlruns (local)'}")
    print(f"Peldaño {args.peldano} ({ETIQUETA_PELDANO[args.peldano]}) — {len(cols)} variables")
    print(f"Dataset: n={len(y)}  positivos={int(y.sum())}  prevalencia={y.mean():.3f}  target={target}")
    print(f"CV evaluacion: {args.n_splits}x{args.n_repeats}   CV anidada: externa {args.n_splits}x{args.n_repeats_nested}, interna {args.k_interno}\n")

    cv_eval = RepeatedStratifiedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats, random_state=SEED)
    cv_ext = RepeatedStratifiedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats_nested, random_state=SEED)
    k_eval = args.n_splits * args.n_repeats

    mlflow.set_experiment(args.experiment)
    run_name = f"robustez_{args.peldano}_{args.target}_cap{cap_pct}"
    with mlflow.start_run(run_name=run_name) as padre:
        mlflow.set_tags({"tipo": "robustez", "owner": owner, "mlflow.user": owner,
                         "peldano": args.peldano, "horizonte": args.target})
        mlflow.log_params({
            "peldano": args.peldano, "n_variables": len(cols), "target": target,
            "n": len(y), "positivos": int(y.sum()), "prevalencia": round(float(y.mean()), 4),
            "n_splits": args.n_splits, "n_repeats": args.n_repeats,
            "n_repeats_nested": args.n_repeats_nested, "k_interno": args.k_interno,
            "capacidad": cap, "seed": SEED, "data_md5": data_hash,
            "sklearn": sklearn.__version__, "rapido": args.rapido,
        })

        resultados, oofs = {}, {}

        # ---- 1. baselines --------------------------------------------------
        print("[1/4] Baselines")
        for nombre, ctor in baselines().items():
            crudo = nombre == "baseline_clinico"
            folds, oof = evaluar_fijo(ctor, X, y, cv_eval, cap, extra, crudo=crudo)
            resultados[nombre] = resumir(folds); oofs[nombre] = oof
            r = resultados[nombre]
            print(f"  {ETIQ_CONF[nombre]:32s} ROC {r['roc_auc_mean']:.3f}  PR {r['pr_auc_mean']:.3f}  "
                  f"sens@{cap_pct}% {r[f'sens_at_{cap_pct}_mean']:.2f}  VPP@{cap_pct}% {r[f'vpp_at_{cap_pct}_mean']:.2f}")

        # ---- 2. modelo del equipo (referencia) ------------------------------
        print("\n[2/4] Configuración elegida por el equipo")
        ctor_equipo = lambda: LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=SEED)
        folds, oof_eq = evaluar_fijo(ctor_equipo, X, y, cv_eval, cap, extra)
        resultados["lr_C0.1"] = resumir(folds); oofs["lr_C0.1"] = oof_eq
        r = resultados["lr_C0.1"]
        print(f"  {ETIQ_CONF['lr_C0.1']:32s} ROC {r['roc_auc_mean']:.3f}  PR {r['pr_auc_mean']:.3f}  "
              f"sens@{cap_pct}% {r[f'sens_at_{cap_pct}_mean']:.2f}  VPP@{cap_pct}% {r[f'vpp_at_{cap_pct}_mean']:.2f}")

        # ---- 3. busqueda de hiperparametros con CV anidada -------------------
        familias = ["lr", "gb"] if args.rapido else ["lr", "rf", "gb"]
        print(f"\n[3/4] Búsqueda de hiperparámetros con CV anidada ({', '.join(familias)})")
        # Referencia sobre LOS MISMOS folds externos: sin esto la comparacion
        # contra la busqueda mezclaria dos esquemas de CV distintos (50 folds
        # contra 10) y la diferencia seria en parte un artefacto de particion.
        folds_eq_ext, _ = evaluar_fijo(ctor_equipo, X, y, cv_ext, cap, extra)

        nested, pareados = {}, {}
        for f in familias:
            n_iter = ESPACIOS[f][2]
            folds, internos, elegidos, oof = evaluar_anidado(
                f, X, y, cv_ext, args.k_interno, cap, extra, n_iter)
            pareados[f] = comparar_pareado(folds, folds_eq_ext)
            res = resumir(folds)
            nested[f] = {"resumen": res, "interno_mean": float(internos.mean()),
                         "interno_std": float(internos.std(ddof=1)), "elegidos": elegidos}
            resultados[f"{f}_buscada"] = res; oofs[f"{f}_buscada"] = oof
            brecha = internos.mean() - res["pr_auc_mean"]
            pa = pareados[f]
            print(f"  {f}: PR-AUC interna {internos.mean():.3f} → anidada {res['pr_auc_mean']:.3f} "
                  f"(brecha {brecha:+.3f})   ROC anidada {res['roc_auc_mean']:.3f}")
            print(f"      vs LR C=0.1 en los mismos {pa['n_folds']} folds: "
                  f"{pa['dif_media']:+.3f} [{pa['dif_ic_lo']:+.3f}, {pa['dif_ic_hi']:+.3f}]  "
                  f"gana en {pa['folds_gana_a_pct']:.0f} % de los folds")
            with mlflow.start_run(run_name=f"anidada_{f}", nested=True):
                mlflow.set_tags({"owner": owner, "mlflow.user": owner, "familia": f})
                mlflow.log_params({"n_iter": n_iter, "k_interno": args.k_interno,
                                   "espacio": str(list(ESPACIOS[f][1]))})
                mlflow.log_metrics({**res, "interno_mean": internos.mean(),
                                    "brecha_optimismo_pr_auc": brecha,
                                    **{f"pareado_{k}": v for k, v in pa.items()}})
                p = out / f"hiperparametros_elegidos_{f}_{args.target}.json"
                p.write_text(json.dumps(elegidos, indent=2, default=str), encoding="utf-8")
                mlflow.log_artifact(str(p))

        # ---- tabla comparativa ----------------------------------------------
        tabla = pd.DataFrame(resultados).T
        tabla.index.name = "configuracion"
        p_tabla = out / f"comparacion_robustez_{args.peldano}_{args.target}.csv"
        tabla.to_csv(p_tabla); mlflow.log_artifact(str(p_tabla))
        for m in ("roc_auc", "pr_auc"):
            p = out / f"baselines_{m}_{args.peldano}_{args.target}.png"
            fig_baselines(tabla, m, float(y.mean()), p); mlflow.log_artifact(str(p))
        p = out / f"brecha_optimismo_{args.peldano}_{args.target}.png"
        fig_brecha(nested, p); mlflow.log_artifact(str(p))

        # ---- 4. criterios de seleccion --------------------------------------
        print("\n[4/4] Criterios de selección, umbrales y calibración")
        # Solo configuraciones medidas sobre EL MISMO esquema de folds (el externo
        # de la CV anidada). Mezclar aqui las de 50 folds haria que el piso del
        # criterio se calculara con una varianza que no les corresponde.
        tabla_ext = pd.DataFrame({
            "lr_C0.1": resumir(folds_eq_ext),
            **{f"{f}_buscada": nested[f]["resumen"] for f in familias},
        }).T
        k_ext = args.n_splits * args.n_repeats_nested
        orden = [c for c in ["lr_C0.1", "lr_buscada", "gb_buscada", "rf_buscada"] if c in tabla_ext.index]
        crit, top = comparar_criterios(tabla_ext.loc[orden], "pr_auc", orden, k_ext)
        print(f"\n  Máximo por PR-AUC: {top}")
        print(crit.to_string(index=False))
        p_crit = out / f"criterios_seleccion_{args.peldano}_{args.target}.csv"
        crit.to_csv(p_crit, index=False); mlflow.log_artifact(str(p_crit))
        mlflow.set_tag("eleccion_estable", str(crit["elegida"].nunique() == 1))

        # ---- umbrales de banda: en muestra vs out-of-fold --------------------
        final = Pipeline([("prep", build_preprocessor(extra)), ("clf", ctor_equipo())]).fit(X, y)
        prob_in = final.predict_proba(X)[:, 1]
        prob_oof = oofs["lr_C0.1"]
        u_in = float(np.quantile(prob_in, 1 - cap))
        u_oof = float(np.quantile(prob_oof, 1 - cap))
        print(f"\n  Umbral banda alta — en muestra {u_in:.3f}  |  out-of-fold {u_oof:.3f}  "
              f"(desplazamiento {u_oof - u_in:+.3f})")
        p = out / f"umbrales_{args.peldano}_{args.target}.png"
        fig_umbrales(prob_in, prob_oof, cap, p); mlflow.log_artifact(str(p))

        # ---- calibracion -----------------------------------------------------
        ctor_cal = lambda: CalibratedClassifierCV(
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=SEED),
            method="isotonic", cv=4)
        folds_cal, oof_cal = evaluar_fijo(ctor_cal, X, y, cv_eval, cap, extra)
        res_cal = resumir(folds_cal)
        resultados["lr_C0.1_calibrada"] = res_cal
        print(f"  Brier — sin calibrar {resultados['lr_C0.1']['brier_mean']:.3f}  "
              f"calibrada {res_cal['brier_mean']:.3f}  |  prevalencia {y.mean():.3f}  "
              f"(referencia: predecir siempre la prevalencia da {y.mean()*(1-y.mean()):.3f})")
        p = out / f"calibracion_{args.peldano}_{args.target}.png"
        fig_calibracion(y.to_numpy(), oof_eq, oof_cal, p); mlflow.log_artifact(str(p))

        # ---- metricas al padre y metadatos ----------------------------------
        for k, v in resultados["lr_C0.1"].items():
            mlflow.log_metric(f"equipo_{k}", v)
        for k, v in res_cal.items():
            mlflow.log_metric(f"calibrada_{k}", v)
        mlflow.log_metrics({
            "baseline_clinico_pr_auc": resultados["baseline_clinico"]["pr_auc_mean"],
            "baseline_clinico_roc_auc": resultados["baseline_clinico"]["roc_auc_mean"],
            f"baseline_clinico_vpp_at_{cap_pct}": resultados["baseline_clinico"][f"vpp_at_{cap_pct}_mean"],
            "umbral_alto_en_muestra": u_in, "umbral_alto_oof": u_oof,
            "desplazamiento_umbral": u_oof - u_in,
        })

        meta = {
            "peldano": args.peldano, "variables": cols, "target": target,
            "n": int(len(y)), "positivos": int(y.sum()), "prevalencia": float(y.mean()),
            "capacidad": cap,
            "umbral_alto_en_muestra": u_in, "umbral_alto_oof": u_oof,
            "umbral_medio_oof": float(np.quantile(prob_oof, max(0.0, 1 - cap - 0.30))),
            "prob_cohorte_oof_ordenada": sorted(float(v) for v in prob_oof),
            "criterios_seleccion": crit.to_dict(orient="records"),
            "brechas_optimismo": {f: nested[f]["interno_mean"] - nested[f]["resumen"]["pr_auc_mean"]
                                  for f in nested},
            "comparacion_pareada_vs_lr_C0.1": pareados,
            "metricas": {k: v for k, v in resultados.items()},
            "data_md5": data_hash, "sklearn": sklearn.__version__, "owner": owner,
            "creado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        p_meta = models_dir / f"robustez_{args.peldano}_{args.target}.json"
        p_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        mlflow.log_artifact(str(p_meta))

        print(f"\nFiguras y tablas: {out}")
        print(f"Metadatos: {p_meta}")
        print(f"Run MLflow: {padre.info.run_id}")


if __name__ == "__main__":
    main()
