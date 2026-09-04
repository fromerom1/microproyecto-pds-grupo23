"""
Modulo de prediccion: el contrato que consumen el tablero y la API.

Carga el modelo entrenado por evaluate_cv.py junto con sus metadatos (umbrales
de banda, distribucion de la cohorte, curva de capacidad) y expone tres
operaciones, que son las mismas del contrato de API en Mockup/TRAZABILIDAD.md:

    predecir(registro)      -> POST /predict        un bebe
    predecir_lote(df)       -> POST /predict/batch  una cohorte, ordenada por riesgo
    info                    -> GET  /model/info     version, metricas CV, umbrales

Ejemplo:
    from src.predict import ModeloRiesgo
    m = ModeloRiesgo("24m")
    r = m.predecir({"educ_cat_n": "Primary or below", "sga": "yes", ...})
    r["probabilidad"], r["banda"], r["percentil"], r["contribuciones"]

Prueba rapida desde consola:
    python -m src.predict --demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.preprocessing import (CATEGORIAS, FEATURE_LABELS, FEATURES, ID_COL,   # noqa: E402
                               variable_original)
try:
    from src.features import ETIQUETAS_EXTRA                                    # noqa: E402
except Exception:                                                               # pragma: no cover
    ETIQUETAS_EXTRA = {}
ETIQUETAS = {**FEATURE_LABELS, **ETIQUETAS_EXTRA}

BANDAS = ("bajo", "medio", "alto")


class ModeloRiesgo:
    """Envuelve el pipeline entrenado y sus metadatos. Una instancia por horizonte."""

    def __init__(self, horizonte: str = "24m", variante: str | None = None,
                 models_dir: Path | str = ROOT / "models",
                 cohorte_path: Path | str | None = None):
        """
        variante:
            "B"  -> peldano B del experimento escalera: 16 basales + z-scores al nacer.
                    Toda la informacion es del momento del ingreso. Es el modelo por
                    defecto si existe (src/experimento_escalera.py lo genera).
            "cv" -> 16 basales solamente (src/evaluate_cv.py).
            None -> "B" si esta disponible, si no "cv".
        """
        models_dir = Path(models_dir)
        self.horizonte = horizonte
        if variante is None:
            variante = "B" if (models_dir / f"model_stunting_{horizonte}_B.joblib").exists() else "cv"
        self.variante = variante
        self.path_modelo = models_dir / f"model_stunting_{horizonte}_{variante}.joblib"
        self.path_meta = models_dir / f"model_stunting_{horizonte}_{variante}.json"
        if not self.path_modelo.exists():
            raise FileNotFoundError(
                f"No existe {self.path_modelo}. Ejecuta primero: python -m src.evaluate_cv --target {horizonte}"
                f" (variante cv) o python -m src.experimento_escalera (variante B)")
        self.pipeline = joblib.load(self.path_modelo)
        self.meta = json.loads(self.path_meta.read_text(encoding="utf-8"))
        # Columnas que espera el modelo: las 16 basales y, en la variante B, los z-scores al nacer
        self.features = list(self.meta.get("features", FEATURES))
        self.features_extra = list(self.meta.get("features_extra", []))
        if cohorte_path is None:
            esc = ROOT / "data/processed/model_dataset_escalera.csv"
            cohorte_path = esc if esc.exists() else ROOT / "data/processed/model_dataset.csv"
        self._prep = self.pipeline.named_steps["prep"]
        self._clf = self.pipeline.named_steps["clf"]
        self._nombres = [variable_original(n) for n in self._prep.get_feature_names_out()]
        self._cohorte_ordenada = np.asarray(self.meta.get("prob_cohorte_ordenada", []), dtype=float)
        self._referencia = self._cargar_referencia(cohorte_path)
        self._explicador = None   # se construye la primera vez que hace falta

    # ---------------------------------------------------------------- utilidades
    def _cargar_referencia(self, cohorte_path):
        """Media de las features transformadas de la cohorte: punto de referencia
        para las contribuciones de la regresion logistica. Si no hay cohorte, ceros."""
        try:
            if cohorte_path and Path(cohorte_path).exists():
                df = pd.read_csv(cohorte_path)
                if all(f in df.columns for f in self.features):
                    return self._prep.transform(df[self.features]).mean(axis=0)
        except Exception:
            pass
        return np.zeros(len(self._nombres))

    def _a_dataframe(self, registro: dict | pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame([registro]) if isinstance(registro, dict) else registro.copy()
        faltan = [f for f in self.features if f not in df.columns]
        for f in faltan:                       # el pipeline imputa; solo hay que crear la columna
            df[f] = np.nan
        return df[self.features]

    def banda(self, prob: float) -> str:
        if prob >= self.meta["umbral_alto"]:
            return "alto"
        if prob >= self.meta["umbral_medio"]:
            return "medio"
        return "bajo"

    def percentil(self, prob: float) -> float:
        """Posicion del bebe dentro de la cohorte de entrenamiento (0-100)."""
        if self._cohorte_ordenada.size == 0:
            return float("nan")
        return float(100.0 * np.searchsorted(self._cohorte_ordenada, prob, side="right") / self._cohorte_ordenada.size)

    # ---------------------------------------------------------------- explicabilidad
    def _contribuciones(self, X_t: np.ndarray) -> tuple[np.ndarray, str]:
        """Contribucion de cada feature transformada a la prediccion.
        Regresion logistica: coef * (x - referencia), exacto, en log-odds.
        Arboles: valores SHAP (TreeExplainer) sobre P(clase 1), en probabilidad."""
        if hasattr(self._clf, "coef_"):
            return self._clf.coef_[0] * (X_t - self._referencia), "log-odds"
        import shap
        if self._explicador is None:
            self._explicador = shap.TreeExplainer(self._clf)
        sv = self._explicador.shap_values(X_t)
        if isinstance(sv, list):              # versiones antiguas: lista por clase
            sv = sv[1]
        elif sv.ndim == 3:                    # (n, features, clases)
            sv = sv[:, :, 1]
        return sv, "probabilidad"

    def _agregar_por_variable(self, contrib_fila: np.ndarray, registro: pd.Series) -> list[dict]:
        s = pd.Series(contrib_fila, index=self._nombres).groupby(level=0).sum()
        out = []
        for var, val in s.items():
            valor = registro.get(var, np.nan)
            out.append({
                "variable": var,
                "etiqueta": ETIQUETAS.get(var, var),
                "valor": None if pd.isna(valor) else (float(valor) if isinstance(valor, (int, float, np.floating)) else str(valor)),
                "contribucion": float(val),
            })
        out.sort(key=lambda d: -abs(d["contribucion"]))
        return out

    # ---------------------------------------------------------------- API publica
    def predecir(self, registro: dict, con_explicacion: bool = True) -> dict:
        X = self._a_dataframe(registro)
        prob = float(self.pipeline.predict_proba(X)[0, 1])
        res = {
            "horizonte": self.horizonte,
            "probabilidad": prob,
            "banda": self.banda(prob),
            "percentil": self.percentil(prob),
        }
        if con_explicacion:
            X_t = self._prep.transform(X)
            contrib, unidad = self._contribuciones(X_t)
            res["contribuciones"] = self._agregar_por_variable(contrib[0], X.iloc[0])
            res["unidad_contribucion"] = unidad
        return res

    def predecir_lote(self, df: pd.DataFrame, capacidad: float | None = None) -> pd.DataFrame:
        """Devuelve el df con probabilidad, banda, percentil y ranking, ordenado de
        mayor a menor riesgo. Si se da `capacidad`, marca el top correspondiente."""
        X = self._a_dataframe(df)
        prob = self.pipeline.predict_proba(X)[:, 1]
        out = df.copy()
        out["probabilidad"] = prob
        out["banda"] = [self.banda(p) for p in prob]
        out["percentil"] = [self.percentil(p) for p in prob]
        out = out.sort_values("probabilidad", ascending=False).reset_index(drop=True)
        out["ranking"] = np.arange(1, len(out) + 1)
        if capacidad is not None:
            n = max(1, int(round(capacidad * len(out))))
            out["seguimiento"] = out["ranking"] <= n
        return out

    def punto_operacion(self, capacidad: float) -> dict:
        """Sensibilidad y VPP esperados a esa capacidad, interpolados de la curva
        out-of-fold de la validacion cruzada (no del ajuste en la cohorte)."""
        curva = pd.DataFrame(self.meta["curva_capacidad"])
        c = float(np.clip(capacidad, curva.capacidad.min(), curva.capacidad.max()))
        return {
            "capacidad": c,
            "sensibilidad": float(np.interp(c, curva.capacidad, curva.sensibilidad)),
            "vpp": float(np.interp(c, curva.capacidad, curva.vpp)),
            "umbral": float(np.interp(c, curva.capacidad, curva.umbral)),
        }

    @property
    def info(self) -> dict:
        cv = self.meta.get("cv", {})
        return {
            "horizonte": self.horizonte,
            "variante": self.variante,
            "peldano": self.meta.get("peldano_desc", "A · basales (ingreso)"),
            "n_features": len(self.features),
            "target": self.meta["target"],
            "config": self.meta["config"],
            "familia": self.meta["familia"],
            "n_entrenamiento": self.meta["n_entrenamiento"],
            "prevalencia": self.meta["prevalencia"],
            "roc_auc_cv": cv.get("roc_auc_mean"),
            "roc_auc_ci": [cv.get("roc_auc_ci_lo"), cv.get("roc_auc_ci_hi")],
            "pr_auc_cv": cv.get("pr_auc_mean"),
            "umbral_alto": self.meta["umbral_alto"],
            "umbral_medio": self.meta["umbral_medio"],
            "data_md5": self.meta["data_md5"],
            "sklearn": self.meta["sklearn"],
            "creado": self.meta["creado"],
        }

    @staticmethod
    def opciones() -> dict:
        """Categorias validas por variable, para construir formularios."""
        return dict(CATEGORIAS)


# ---------------------------------------------------------------------------
def _demo():
    m = ModeloRiesgo("24m")
    print("Modelo:", json.dumps(m.info, indent=2, ensure_ascii=False))
    caso = {
        "enrol_hiv_status_cat": "Negative", "momage_cat": "25 and less", "educ_cat_n": "Primary or below",
        "marital_cat": "Married", "wealth_quintile": "lowest quintile", "depression": "no depression",
        "mom_muac_cat": "normal", "b1_sex": "female", "preterm": "yes", "hfia_enr": "severe",
        "parity": "multi", "enrol_anemia": "no", "caesarean": "no", "sga": "yes", "lbw": "yes",
        "gestage_final": 36.0,
    }
    if m.features_extra:
        caso.update({"zlen_nac": -1.4, "zwei_nac": -1.1, "zhc_nac": -0.6})
    r = m.predecir(caso)
    print(f"\nCaso de ejemplo -> probabilidad {r['probabilidad']:.3f} | banda {r['banda']} | percentil {r['percentil']:.0f}")
    print(f"Contribuciones ({r['unidad_contribucion']}):")
    for c in r["contribuciones"][:8]:
        print(f"  {c['etiqueta']:32s} {str(c['valor']):22s} {c['contribucion']:+.3f}")
    print("\nPunto de operacion al 20 %:", {k: round(v, 3) for k, v in m.punto_operacion(0.20).items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        _demo()
    else:
        ap.print_help()
