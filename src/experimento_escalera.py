#!/usr/bin/env python3
"""
Experimento escalera: cuanto mejora la prediccion a medida que llega informacion
del seguimiento temprano.

Responde la pregunta de la seccion 10 de la propuesta de grado: "en que momento
la prediccion se vuelve lo suficientemente confiable?". Cuatro peldanos
acumulativos (ver src/features.py):

    A  16 basales (contrato actual)                 -> ingreso
    B  + z-scores al nacer                          -> ingreso, ya medidos
    C  + z-scores semana 3 + velocidad de LAZ        -> primer control
    D  + z-scores mes 3 + velocidad de LAZ           -> tercer mes

Cada peldano se evalua con la misma validacion cruzada estratificada repetida de
evaluate_cv.py (5x10 = 50 folds), para dos horizontes (12 y 24 meses) y dos
familias (la logistica elegida y un random forest de control). Todo a MLflow.

Uso:
    python -m src.experimento_escalera                  # 5x10, ambos horizontes
    python -m src.experimento_escalera --n-repeats 3    # prueba rapida
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.preprocessing import FEATURE_LABELS, ID_COL, SEED, TARGETS, build_preprocessor   # noqa: E402
from src.features import (ETIQUETA_PELDANO, ETIQUETAS_EXTRA, OUT as DATA_ESCALERA,       # noqa: E402
                          columnas_peldano, numericas_peldano)
from src.evaluate_cv import (curva_capacidad, fig_estabilidad_importancias,              # noqa: E402
                             importancia_por_variable, punto_operacion, resumen)
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score      # noqa: E402

warnings.filterwarnings("ignore")

AZUL, ROJO, GRIS, VERDE = "#1f5f8b", "#a33235", "#8a929e", "#2c6e49"

MODELOS = {
    "lr_C0.1": lambda: LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=SEED),
    "rf_d6":   lambda: RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5,
                                              class_weight="balanced", random_state=SEED, n_jobs=-1),
}


def evaluar_peldano(peldano, constructor, df, target, cv, capacidad):
    cols = columnas_peldano(peldano)
    extra = [c for c in numericas_peldano(peldano) if c != "gestage_final"]
    d = df[df[target].notna()].reset_index(drop=True)
    X, y = d[cols], (d[target] == "yes").astype(int)
    folds, imps = [], []
    suma, cuenta = np.zeros(len(y)), np.zeros(len(y))
    for k, (tr, te) in enumerate(cv.split(X, y)):
        pipe = Pipeline([("prep", build_preprocessor(extra)), ("clf", constructor())])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        prob = pipe.predict_proba(X.iloc[te])[:, 1]
        yt = y.iloc[te].to_numpy()
        sens, vpp, _ = punto_operacion(yt, prob, capacidad)
        folds.append({"fold": k, "roc_auc": roc_auc_score(yt, prob),
                      "pr_auc": average_precision_score(yt, prob),
                      "brier": brier_score_loss(yt, prob),
                      f"sens_at_{int(capacidad*100)}": sens, f"vpp_at_{int(capacidad*100)}": vpp})
        suma[te] += prob; cuenta[te] += 1
        imp = importancia_por_variable(pipe)
        if imp is not None:
            imps.append(imp)
    return pd.DataFrame(folds), suma / np.maximum(cuenta, 1), (pd.DataFrame(imps) if imps else None), X, y, extra


def fig_escalera(tabla: pd.DataFrame, modelo: str, capacidad: float, path: Path):
    """Panel de tres metricas vs peldano, una linea por horizonte."""
    metricas = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), (f"sens_at_{int(capacidad*100)}", f"Sensibilidad al {int(capacidad*100)} %")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    x = np.arange(4)
    for ax, (m, titulo) in zip(axes, metricas):
        for h, color, marker in (("24m", ROJO, "o"), ("12m", AZUL, "s")):
            t = tabla[(tabla.modelo == modelo) & (tabla.horizonte == h)].set_index("peldano").reindex(list("ABCD"))
            ax.errorbar(x, t[f"{m}_mean"], yerr=[t[f"{m}_mean"] - t[f"{m}_ci_lo"], t[f"{m}_ci_hi"] - t[f"{m}_mean"]],
                        fmt=f"{marker}-", color=color, lw=2, capsize=4, ms=7, label=f"{h[:-1]} meses")
        if m == "roc_auc":
            ax.axhline(0.5, ls="--", color=GRIS, lw=1)
        ax.set_xticks(x); ax.set_xticklabels(["A\nbasales", "B\n+ nacer", "C\n+ sem 3", "D\n+ mes 3"])
        ax.set_title(titulo, loc="left", fontweight="bold"); ax.grid(alpha=.3)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle(f"Experimento escalera — {modelo} · media e IC95 % sobre 50 folds", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DATA_ESCALERA))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--capacity", type=float, default=0.20)
    ap.add_argument("--experiment", default="stunting-escalera")
    ap.add_argument("--out-dir", default=str(ROOT / "figures/04_escalera"))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    args = ap.parse_args()

    if not Path(args.data).exists():
        sys.exit(f"No existe {args.data}. Ejecuta primero: python -m src.features")
    df = pd.read_csv(args.data)
    data_hash = hashlib.md5(Path(args.data).read_bytes()).hexdigest()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir); models_dir.mkdir(parents=True, exist_ok=True)
    cap = int(args.capacity * 100)
    cv = RepeatedStratifiedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats, random_state=SEED)

    print(f"MLflow tracking: {os.environ.get('MLFLOW_TRACKING_URI', '') or './mlruns (local)'}")
    print(f"CV: {args.n_splits}x{args.n_repeats} · capacidad {cap} %\n")
    mlflow.set_experiment(args.experiment)

    filas, oofs, imps, datos = [], {}, {}, {}
    with mlflow.start_run(run_name=f"escalera_{args.n_splits}x{args.n_repeats}_cap{cap}") as padre:
        mlflow.log_params({"n_splits": args.n_splits, "n_repeats": args.n_repeats, "capacidad": args.capacity,
                           "data_md5": data_hash, "sklearn": sklearn.__version__, "seed": SEED,
                           "peldanos": "A basales | B +nacer | C +sem3 | D +mes3"})
        mlflow.set_tag("tipo", "escalera")

        for h, target in TARGETS.items():
            print(f"── Horizonte {h} ({target}) ──")
            for nombre, ctor in MODELOS.items():
                for p in "ABCD":
                    df_f, oof, df_imp, X, y, extra = evaluar_peldano(p, ctor, df, target, cv, args.capacity)
                    r = resumen(df_f)
                    filas.append({"horizonte": h, "modelo": nombre, "peldano": p, "n_features": X.shape[1], **r})
                    oofs[(h, nombre, p)] = (oof, y); imps[(h, nombre, p)] = df_imp; datos[(h, nombre, p)] = (X, y, extra)
                    print(f"  {nombre:8s} {p}  {X.shape[1]:2d} feats  ROC-AUC {r['roc_auc_mean']:.3f} "
                          f"[{r['roc_auc_ci_lo']:.2f}, {r['roc_auc_ci_hi']:.2f}]  PR-AUC {r['pr_auc_mean']:.3f}  "
                          f"sens@{cap} {r[f'sens_at_{cap}_mean']:.2f}  VPP@{cap} {r[f'vpp_at_{cap}_mean']:.2f}")
                    with mlflow.start_run(run_name=f"{h}_{nombre}_{p}", nested=True):
                        mlflow.log_params({"horizonte": h, "target": target, "modelo": nombre, "peldano": p,
                                           "peldano_desc": ETIQUETA_PELDANO[p], "n_features": X.shape[1],
                                           "features_extra": ",".join(extra) or "-"})
                        mlflow.log_metrics(r)
                        pf = out / f"folds_{h}_{nombre}_{p}.csv"; df_f.to_csv(pf, index=False); mlflow.log_artifact(str(pf))
            print()

        tabla = pd.DataFrame(filas)
        p_t = out / "escalera_resultados.csv"; tabla.to_csv(p_t, index=False); mlflow.log_artifact(str(p_t))

        for nombre in MODELOS:
            pf = out / f"escalera_{nombre}.png"
            fig_escalera(tabla, nombre, args.capacity, pf); mlflow.log_artifact(str(pf))

        # --- ganancia de cada peldano respecto al anterior (logistica) ---
        print("Ganancia por peldano (ROC-AUC, lr_C0.1):")
        for h in TARGETS:
            t = tabla[(tabla.modelo == "lr_C0.1") & (tabla.horizonte == h)].set_index("peldano").reindex(list("ABCD"))
            deltas = t.roc_auc_mean.diff().round(3).to_dict()
            print(f"  {h}: " + "  ".join(f"{p}={t.loc[p,'roc_auc_mean']:.3f}" + (f" ({deltas[p]:+.3f})" if p != "A" else "") for p in "ABCD"))
            for p in "ABCD":
                mlflow.log_metric(f"{h}_lr_roc_auc_{p}", t.loc[p, "roc_auc_mean"])

        # --- importancias del peldano D con logistica: que z-scores pesan ---
        for h in TARGETS:
            df_imp = imps[(h, "lr_C0.1", "D")]
            if df_imp is not None:
                etiquetas = {**FEATURE_LABELS, **ETIQUETAS_EXTRA}
                df_imp = df_imp.rename(columns=etiquetas)
                pf = out / f"importancias_D_{h}.png"
                fig_estabilidad_importancias(df_imp, f"lr_C0.1 · peldaño D · {h}", pf); mlflow.log_artifact(str(pf))

        # --- guardar el modelo del peldano B (tiempo cero completo) por horizonte ---
        # B es el mejor candidato para el tablero: usa solo informacion del ingreso,
        # igual que A, pero incluye la antropometria al nacer que el contrato omitia.
        for h, target in TARGETS.items():
            X, y, extra = datos[(h, "lr_C0.1", "B")]
            final = Pipeline([("prep", build_preprocessor(extra)), ("clf", MODELOS["lr_C0.1"]())]).fit(X, y)
            prob = final.predict_proba(X)[:, 1]
            oof, _ = oofs[(h, "lr_C0.1", "B")]
            cc = curva_capacidad(y.to_numpy(), oof)
            r = tabla[(tabla.horizonte == h) & (tabla.modelo == "lr_C0.1") & (tabla.peldano == "B")].iloc[0]
            meta = {
                "target": target, "horizonte": h, "config": "lr_C0.1", "familia": "logistic_regression",
                "peldano": "B", "peldano_desc": ETIQUETA_PELDANO["B"],
                "features": list(X.columns), "features_extra": extra,
                "n_entrenamiento": int(len(y)), "prevalencia": float(y.mean()), "capacidad": args.capacity,
                "umbral_alto": float(np.quantile(prob, 1 - args.capacity)),
                "umbral_medio": float(np.quantile(prob, 1 - args.capacity - 0.30)),
                "cv": {"n_splits": args.n_splits, "n_repeats": args.n_repeats,
                       **{k: float(v) for k, v in r.items() if isinstance(v, (int, float, np.floating))}},
                "curva_capacidad": cc.to_dict(orient="records"),
                "prob_cohorte_ordenada": sorted(float(v) for v in prob),
                "data_md5": data_hash, "sklearn": sklearn.__version__,
                "creado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            pm = models_dir / f"model_stunting_{h}_B.joblib"; joblib.dump(final, pm)
            pj = models_dir / f"model_stunting_{h}_B.json"; pj.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            mlflow.log_artifact(str(pj))
            print(f"Guardado {pm.name} (peldaño B, {h})")

        print(f"\nRun padre MLflow: {padre.info.run_id}")


if __name__ == "__main__":
    main()
