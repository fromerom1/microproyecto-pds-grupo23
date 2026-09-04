#!/usr/bin/env python3
"""
Evaluacion robusta de modelos con validacion cruzada repetida + MLflow.

Por que existe este script y no basta con train_stunting.py:
    Con 333 bebes y ~95 positivos, un unico split 70/15/15 deja ~49 bebes en
    test. Medido con 20 semillas distintas, el ROC-AUC de test del mismo
    modelo oscila entre 0.43 y 0.81. Ese numero no describe al modelo,
    describe a la semilla. La validacion cruzada estratificada repetida
    (5 folds x 10 repeticiones = 50 evaluaciones) da una media con intervalo
    de confianza, que es lo que la rubrica llama "evaluar adecuadamente".

Que hace:
    1. Para cada configuracion de modelo, CV estratificada repetida.
    2. Por fold: ROC-AUC, PR-AUC, Brier, y el punto de operacion por
       capacidad (sensibilidad y VPP al marcar el X % de mayor riesgo).
    3. Agrega media, desviacion e IC95 % por configuracion.
    4. Predicciones out-of-fold -> curvas ROC/PR, calibracion y curva
       sensibilidad-vs-capacidad (la que usa el tablero).
    5. Estabilidad de la importancia de variables entre folds.
    6. Reajusta la mejor configuracion con TODOS los datos y la guarda en
       models/ con los umbrales de banda, lista para el tablero y la API.
    7. Todo registrado en MLflow: un run padre por barrido, un run hijo por
       configuracion, figuras y tablas como artefactos.

Uso:
    python -m src.evaluate_cv                       # 24 meses, 5x10, capacidad 20 %
    python -m src.evaluate_cv --target 12m
    python -m src.evaluate_cv --n-repeats 20 --capacity 0.25

    MLflow remoto (EC2):  export MLFLOW_TRACKING_URI=http://<ip>:8050
    Sin esa variable registra en ./mlruns (local).
"""

from __future__ import annotations

import argparse
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
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn

# Permite ejecutar como `python -m src.evaluate_cv` (paquete) o `python src/evaluate_cv.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.preprocessing import (FEATURE_LABELS, FEATURES, SEED, TARGETS,       # noqa: E402
                               build_preprocessor, cargar_dataset, variable_original)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuraciones a comparar. Es una comparacion de familias con una
# regularizacion razonable, no una busqueda exhaustiva: con n=333 la varianza
# de la evaluacion domina sobre cualquier ajuste fino de hiperparametros.
# ---------------------------------------------------------------------------
def configuraciones():
    return {
        "lr_C0.1":  lambda: LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=SEED),
        "lr_C1":    lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED),
        "lr_C10":   lambda: LogisticRegression(C=10.0, class_weight="balanced", max_iter=2000, random_state=SEED),
        "rf_d4":    lambda: RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=5,
                                                   class_weight="balanced", random_state=SEED, n_jobs=-1),
        "rf_d6":    lambda: RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5,
                                                   class_weight="balanced", random_state=SEED, n_jobs=-1),
        "rf_full":  lambda: RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=5,
                                                   class_weight="balanced", random_state=SEED, n_jobs=-1),
        "gb_d3":    lambda: GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3,
                                                       random_state=SEED),
    }


def familia(nombre: str) -> str:
    return {"lr": "logistic_regression", "rf": "random_forest", "gb": "gradient_boosting"}[nombre.split("_")[0]]


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def punto_operacion(y_true: np.ndarray, prob: np.ndarray, capacidad: float):
    """Marca el `capacidad` (fraccion) de mayor probabilidad y mide que logra.
    Devuelve sensibilidad, VPP y el umbral de probabilidad implicito."""
    n_marcar = max(1, int(round(capacidad * len(prob))))
    orden = np.argsort(-prob)
    marcados = np.zeros(len(prob), dtype=bool)
    marcados[orden[:n_marcar]] = True
    positivos = y_true == 1
    sens = (marcados & positivos).sum() / max(1, positivos.sum())
    vpp = (marcados & positivos).sum() / n_marcar
    umbral = float(prob[orden[n_marcar - 1]])
    return float(sens), float(vpp), umbral


def curva_capacidad(y_true, prob, capacidades=np.arange(0.05, 0.55, 0.05)):
    filas = []
    for c in capacidades:
        s, v, u = punto_operacion(y_true, prob, float(c))
        filas.append({"capacidad": round(float(c), 2), "sensibilidad": s, "vpp": v, "umbral": u})
    return pd.DataFrame(filas)


def importancia_por_variable(pipe: Pipeline) -> pd.Series | None:
    clf, pre = pipe.named_steps["clf"], pipe.named_steps["prep"]
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0])
    else:
        return None
    nombres = [variable_original(n) for n in pre.get_feature_names_out()]
    return pd.Series(imp, index=nombres).groupby(level=0).sum()


def ic95(v: np.ndarray):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# ---------------------------------------------------------------------------
# Evaluacion de una configuracion
# ---------------------------------------------------------------------------
def evaluar_config(nombre, constructor, X, y, cv, capacidad):
    n = len(y)
    suma_oof = np.zeros(n)
    cuenta_oof = np.zeros(n)
    folds, importancias = [], []

    for k, (tr, te) in enumerate(cv.split(X, y)):
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", constructor())])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        prob = pipe.predict_proba(X.iloc[te])[:, 1]
        yt = y.iloc[te].to_numpy()
        sens, vpp, umbral = punto_operacion(yt, prob, capacidad)
        folds.append({
            "fold": k,
            "roc_auc": roc_auc_score(yt, prob),
            "pr_auc": average_precision_score(yt, prob),
            "brier": brier_score_loss(yt, prob),
            f"sens_at_{int(capacidad*100)}": sens,
            f"vpp_at_{int(capacidad*100)}": vpp,
            "umbral_capacidad": umbral,
        })
        suma_oof[te] += prob
        cuenta_oof[te] += 1
        imp = importancia_por_variable(pipe)
        if imp is not None:
            importancias.append(imp)

    df_folds = pd.DataFrame(folds)
    oof = suma_oof / np.maximum(cuenta_oof, 1)
    df_imp = pd.DataFrame(importancias) if importancias else None
    return df_folds, oof, df_imp


def resumen(df_folds: pd.DataFrame) -> dict:
    out = {}
    for col in df_folds.columns:
        if col == "fold":
            continue
        v = df_folds[col].to_numpy()
        lo, hi = ic95(v)
        out[f"{col}_mean"] = float(v.mean())
        out[f"{col}_std"] = float(v.std(ddof=1))
        out[f"{col}_ci_lo"] = lo
        out[f"{col}_ci_hi"] = hi
    return out


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
AZUL, ROJO, GRIS, VERDE = "#1f5f8b", "#a33235", "#8a929e", "#2c6e49"


def fig_comparacion(tabla: pd.DataFrame, metrica: str, titulo: str, path: Path):
    t = tabla.sort_values(f"{metrica}_mean")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(t) + 1.6))
    y = np.arange(len(t))
    ax.errorbar(t[f"{metrica}_mean"], y,
                xerr=[t[f"{metrica}_mean"] - t[f"{metrica}_ci_lo"], t[f"{metrica}_ci_hi"] - t[f"{metrica}_mean"]],
                fmt="o", color=AZUL, ecolor=GRIS, capsize=4, lw=1.5)
    if metrica == "roc_auc":
        ax.axvline(0.5, ls="--", color=ROJO, lw=1, label="Azar (0.5)")
        ax.legend(loc="lower right", frameon=False)
    ax.set_yticks(y); ax.set_yticklabels(t.index)
    ax.set_xlabel(f"{metrica.upper().replace('_', '-')} — media e IC95 % sobre los folds")
    ax.set_title(titulo, loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=.3)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def fig_curvas_oof(y, oof, nombre, capacidad, path: Path):
    fpr, tpr, _ = roc_curve(y, oof)
    prec, rec, _ = precision_recall_curve(y, oof)
    prob_true, prob_pred = calibration_curve(y, oof, n_bins=5, strategy="quantile")
    cc = curva_capacidad(y, oof)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    ax = axes[0]
    ax.plot(fpr, tpr, color=AZUL, lw=2, label=f"AUC = {roc_auc_score(y, oof):.3f}")
    ax.plot([0, 1], [0, 1], "--", color=ROJO, lw=1); ax.set_xlabel("1 − especificidad"); ax.set_ylabel("Sensibilidad")
    ax.set_title("ROC (out-of-fold)", loc="left", fontweight="bold"); ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(rec, prec, color=AZUL, lw=2, label=f"PR-AUC = {average_precision_score(y, oof):.3f}")
    ax.axhline(y.mean(), ls="--", color=ROJO, lw=1, label=f"Prevalencia = {y.mean():.2f}")
    ax.set_xlabel("Sensibilidad"); ax.set_ylabel("VPP (precisión)")
    ax.set_title("Precisión–sensibilidad", loc="left", fontweight="bold"); ax.legend(frameon=False)

    ax = axes[2]
    ax.plot(prob_pred, prob_true, "o-", color=AZUL, lw=2)
    ax.plot([0, 1], [0, 1], "--", color=GRIS, lw=1)
    ax.set_xlabel("Probabilidad predicha"); ax.set_ylabel("Fracción observada")
    ax.set_title(f"Calibración (Brier = {brier_score_loss(y, oof):.3f})", loc="left", fontweight="bold")

    ax = axes[3]
    ax.plot(cc.capacidad * 100, cc.sensibilidad * 100, "o-", color=AZUL, lw=2, label="Sensibilidad")
    ax.plot(cc.capacidad * 100, cc.vpp * 100, "s-", color=VERDE, lw=2, label="VPP")
    ax.axvline(capacidad * 100, ls="--", color=ROJO, lw=1)
    ax.set_xlabel("% de la cohorte marcada para seguimiento"); ax.set_ylabel("%")
    ax.set_title("Punto de operación por capacidad", loc="left", fontweight="bold"); ax.legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=.3)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.suptitle(f"Mejor configuración: {nombre}", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return cc


def fig_estabilidad_importancias(df_imp: pd.DataFrame, nombre, path: Path):
    m = df_imp.mean().sort_values(); s = df_imp.std()
    etiquetas = [FEATURE_LABELS.get(v, v) for v in m.index]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(m) + 1.5))
    ax.barh(etiquetas, m.values, xerr=s[m.index].values, color=AZUL, ecolor=GRIS, capsize=3, alpha=.9)
    ax.set_xlabel("Importancia agregada por variable — media ± DE entre folds")
    ax.set_title(f"Estabilidad de la importancia de variables ({nombre})", loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=.3)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data/processed/model_dataset.csv"))
    ap.add_argument("--target", default="24m", choices=list(TARGETS))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--capacity", type=float, default=0.20,
                    help="Fraccion de la cohorte que el programa puede vigilar de cerca (punto de operacion)")
    ap.add_argument("--experiment", default="stunting-evaluacion-cv")
    ap.add_argument("--rank-by", default="pr_auc", choices=["pr_auc", "roc_auc"])
    ap.add_argument("--out-dir", default=str(ROOT / "figures/03_cv"))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    args = ap.parse_args()

    target = TARGETS[args.target]
    X, y, df = cargar_dataset(args.data, target)
    data_hash = hashlib.md5(Path(args.data).read_bytes()).hexdigest()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir); models_dir.mkdir(parents=True, exist_ok=True)
    cap_pct = int(args.capacity * 100)

    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    print(f"MLflow tracking: {uri or './mlruns (local)'}")
    print(f"Dataset: n={len(y)}  positivos={int(y.sum())}  prevalencia={y.mean():.3f}  target={target}")
    print(f"CV: {args.n_splits} folds x {args.n_repeats} repeticiones = {args.n_splits*args.n_repeats} evaluaciones por configuracion\n")

    cv = RepeatedStratifiedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats, random_state=SEED)
    mlflow.set_experiment(args.experiment)

    filas, oofs, imps = {}, {}, {}
    with mlflow.start_run(run_name=f"cv_{args.n_splits}x{args.n_repeats}_{args.target}_cap{cap_pct}") as padre:
        mlflow.log_params({
            "target": target, "n": len(y), "positivos": int(y.sum()), "prevalencia": round(float(y.mean()), 4),
            "n_splits": args.n_splits, "n_repeats": args.n_repeats, "capacidad": args.capacity,
            "features": "16 basales (tiempo cero)", "seed": SEED, "data_md5": data_hash,
            "sklearn": sklearn.__version__, "rank_by": args.rank_by,
        })
        mlflow.set_tag("tipo", "evaluacion_cv")

        for nombre, constructor in configuraciones().items():
            df_folds, oof, df_imp = evaluar_config(nombre, constructor, X, y, cv, args.capacity)
            res = resumen(df_folds)
            filas[nombre] = res; oofs[nombre] = oof; imps[nombre] = df_imp
            print(f"  {nombre:9s}  ROC-AUC {res['roc_auc_mean']:.3f} [{res['roc_auc_ci_lo']:.2f}, {res['roc_auc_ci_hi']:.2f}]"
                  f"   PR-AUC {res['pr_auc_mean']:.3f}   sens@{cap_pct}% {res[f'sens_at_{cap_pct}_mean']:.2f}"
                  f"   VPP@{cap_pct}% {res[f'vpp_at_{cap_pct}_mean']:.2f}")

            with mlflow.start_run(run_name=nombre, nested=True):
                clf = constructor()
                mlflow.log_params({"config": nombre, "familia": familia(nombre), **{
                    k: v for k, v in clf.get_params().items()
                    if k in ("C", "penalty", "n_estimators", "max_depth", "min_samples_leaf", "learning_rate", "class_weight")
                }})
                mlflow.log_metrics(res)
                p = out / f"folds_{nombre}.csv"; df_folds.to_csv(p, index=False); mlflow.log_artifact(str(p))

        # ---- comparacion y seleccion ----------------------------------------
        tabla = pd.DataFrame(filas).T
        tabla.index.name = "config"
        tabla = tabla.sort_values(f"{args.rank_by}_mean", ascending=False)
        p_tabla = out / f"comparacion_{args.target}.csv"; tabla.to_csv(p_tabla); mlflow.log_artifact(str(p_tabla))
        for m in ("roc_auc", "pr_auc"):
            p = out / f"comparacion_{m}_{args.target}.png"
            fig_comparacion(tabla, m, f"Comparación de configuraciones — stunting a {args.target}", p)
            mlflow.log_artifact(str(p))

        # Regla de un error estandar: entre las configuraciones cuya media esta a
        # menos de una DE de la mejor, se elige la MAS SIMPLE (el orden de
        # `configuraciones()` va de menor a mayor complejidad). Con n=333 las
        # diferencias entre familias son mas pequenas que la varianza de la
        # evaluacion, y la simplicidad compra interpretabilidad gratis.
        top = tabla.index[0]
        piso = tabla.loc[top, f"{args.rank_by}_mean"] - tabla.loc[top, f"{args.rank_by}_std"]
        candidatas = [c for c in configuraciones() if tabla.loc[c, f"{args.rank_by}_mean"] >= piso]
        mejor = candidatas[0]
        print(f"\nMaximo por {args.rank_by}: {top} ({tabla.loc[top, f'{args.rank_by}_mean']:.3f})")
        print(f"Indistinguibles (a < 1 DE): {candidatas}")
        print(f"Elegida por la regla de un error estandar (la mas simple): {mejor}")
        mlflow.set_tag("mejor_config", mejor)
        mlflow.set_tag("maximo_config", top)
        mlflow.log_param("regla_seleccion", "one-standard-error, mas simple entre indistinguibles")
        for k, v in filas[mejor].items():
            mlflow.log_metric(f"mejor_{k}", v)

        # ---- curvas OOF y curva de capacidad del mejor ----------------------
        p = out / f"curvas_oof_{mejor}_{args.target}.png"
        cc = fig_curvas_oof(y.to_numpy(), oofs[mejor], mejor, args.capacity, p)
        mlflow.log_artifact(str(p))
        p_cc = out / f"curva_capacidad_{args.target}.csv"; cc.to_csv(p_cc, index=False); mlflow.log_artifact(str(p_cc))

        if imps[mejor] is not None:
            p = out / f"importancias_{mejor}_{args.target}.png"
            fig_estabilidad_importancias(imps[mejor], mejor, p); mlflow.log_artifact(str(p))
            p_imp = out / f"importancias_{mejor}_{args.target}.csv"
            pd.DataFrame({"media": imps[mejor].mean(), "de": imps[mejor].std()}).sort_values("media", ascending=False).to_csv(p_imp)
            mlflow.log_artifact(str(p_imp))

        # ---- reajuste final con todos los datos -> artefacto para el tablero -
        final = Pipeline([("prep", build_preprocessor()), ("clf", configuraciones()[mejor]())]).fit(X, y)
        prob_cohorte = final.predict_proba(X)[:, 1]
        # Bandas por posicion en la cohorte: alto = top `capacidad`, medio = siguiente 30 %
        umbral_alto = float(np.quantile(prob_cohorte, 1 - args.capacity))
        umbral_medio = float(np.quantile(prob_cohorte, 1 - args.capacity - 0.30))
        meta = {
            "target": target, "horizonte": args.target, "config": mejor, "familia": familia(mejor),
            "features": FEATURES, "n_entrenamiento": int(len(y)), "prevalencia": float(y.mean()),
            "capacidad": args.capacity, "umbral_alto": umbral_alto, "umbral_medio": umbral_medio,
            "cv": {"n_splits": args.n_splits, "n_repeats": args.n_repeats, **filas[mejor]},
            "curva_capacidad": cc.to_dict(orient="records"),
            "prob_cohorte_ordenada": sorted(float(v) for v in prob_cohorte),
            "data_md5": data_hash, "sklearn": sklearn.__version__,
            "creado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        p_model = models_dir / f"model_stunting_{args.target}_cv.joblib"
        p_meta = models_dir / f"model_stunting_{args.target}_cv.json"
        joblib.dump(final, p_model)
        p_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        mlflow.log_artifact(str(p_meta))
        try:
            mlflow.sklearn.log_model(final, name="stunting-model-cv")
        except TypeError:
            mlflow.sklearn.log_model(final, artifact_path="stunting-model-cv")
        print(f"Modelo final guardado: {p_model}\nMetadatos: {p_meta}")
        print(f"Umbrales de banda (cohorte): alto >= {umbral_alto:.3f}  |  medio >= {umbral_medio:.3f}")
        print(f"Run padre MLflow: {padre.info.run_id}")


if __name__ == "__main__":
    main()
