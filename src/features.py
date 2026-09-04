"""
Construccion del dataset del experimento escalera a partir del CSV crudo.

El dataset plano de modelado (data/processed/model_dataset.csv) contiene solo las
16 variables basales del contrato. Pero el CSV longitudinal tiene, en cada visita,
los z-scores de talla (zlen), peso (zwei) y perimetro cefalico (zhc). Esos
z-scores en las visitas TEMPRANAS son informacion legitima para predecir el
desenlace a 12 o 24 meses, y responden la pregunta que la propuesta de grado
formula en su seccion 10: "en que momento la prediccion se vuelve lo
suficientemente confiable?"

Peldanos (bloques de features acumulativos):

    A  basales        16 variables del contrato                       -> ingreso
    B  + nacer        A + zlen/zwei/zhc en 'delivery'                  -> ingreso (*)
    C  + semana 3     B + zlen/zwei/zhc en 'week-3'  + velocidad LAZ   -> primer control
    D  + mes 3        C + zlen/zwei/zhc en 'month-3' + velocidad LAZ   -> tercer mes

(*) Los z-scores al nacer se miden en el parto: al momento del ingreso al programa
    ya se conocen. El contrato original los omitia; el peldano B mide cuanto costo
    esa omision al modelo de "tiempo cero".

Anti-fuga: ningun peldano usa informacion posterior al mes 3. El desenlace mas
temprano que se predice es a 12 meses. No se usan las variables binarias
`stunted` de las visitas tempranas: el z-score continuo las contiene.

Uso:
    python -m src.features            # escribe data/processed/model_dataset_escalera.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.preprocessing import FEATURES, ID_COL, TARGETS   # noqa: E402

RAW = ROOT / "data/plosmed_data_newid.csv"
OUT = ROOT / "data/processed/model_dataset_escalera.csv"

ZSCORES = ["zlen", "zwei", "zhc"]
VISITAS_TEMPRANAS = {"delivery": "nac", "week-3": "s3", "month-3": "m3"}

# Columnas que agrega cada peldano respecto al anterior
PELDANOS = {
    "A": [],
    "B": [f"{z}_nac" for z in ZSCORES],
    "C": [f"{z}_s3" for z in ZSCORES] + ["dlaz_s3"],
    "D": [f"{z}_m3" for z in ZSCORES] + ["dlaz_m3"],
}
ETIQUETA_PELDANO = {
    "A": "A · basales (ingreso)",
    "B": "B · + z-scores al nacer",
    "C": "C · + semana 3",
    "D": "D · + mes 3",
}

ETIQUETAS_EXTRA = {
    "zlen_nac": "LAZ al nacer", "zwei_nac": "WAZ al nacer", "zhc_nac": "HCZ al nacer",
    "zlen_s3": "LAZ semana 3", "zwei_s3": "WAZ semana 3", "zhc_s3": "HCZ semana 3",
    "zlen_m3": "LAZ mes 3", "zwei_m3": "WAZ mes 3", "zhc_m3": "HCZ mes 3",
    "dlaz_s3": "Δ LAZ nacer→sem 3", "dlaz_m3": "Δ LAZ nacer→mes 3",
}


def columnas_peldano(peldano: str) -> list[str]:
    """Features acumuladas hasta ese peldano (A ⊂ B ⊂ C ⊂ D)."""
    extra = []
    for p in "ABCD":
        extra += PELDANOS[p]
        if p == peldano:
            break
    return FEATURES + extra


def numericas_peldano(peldano: str) -> list[str]:
    """Las columnas anadidas son todas numericas; gestage_final ya lo es."""
    return ["gestage_final"] + [c for c in columnas_peldano(peldano) if c not in FEATURES]


def construir(raw_path: Path = RAW) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    # --- bloque basal: una fila por bebe (las 16 features no cambian entre visitas) ---
    base = (df.sort_values([ID_COL, "visit"])
              .drop_duplicates(ID_COL)[[ID_COL] + FEATURES]
              .reset_index(drop=True))

    # --- z-scores en las visitas tempranas, a formato ancho ---
    for visita, sufijo in VISITAS_TEMPRANAS.items():
        sub = (df[df.visit == visita][[ID_COL] + ZSCORES]
                 .drop_duplicates(ID_COL)
                 .rename(columns={z: f"{z}_{sufijo}" for z in ZSCORES}))
        base = base.merge(sub, on=ID_COL, how="left")

    # --- velocidad de crecimiento en talla (cambio de LAZ desde el nacimiento) ---
    base["dlaz_s3"] = base["zlen_s3"] - base["zlen_nac"]
    base["dlaz_m3"] = base["zlen_m3"] - base["zlen_nac"]

    # --- targets: estado en month-12 y month-24 ---
    for h, col in TARGETS.items():
        visita = {"12m": "month-12", "24m": "month-24"}[h]
        t = (df[df.visit == visita][[ID_COL, "stunted"]]
               .drop_duplicates(ID_COL).rename(columns={"stunted": col}))
        base = base.merge(t, on=ID_COL, how="left")

    return base


def main():
    out = construir()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Escrito {OUT.relative_to(ROOT)}: {out.shape[0]} bebes x {out.shape[1]} columnas")
    print("\nCobertura de las columnas nuevas (% no nulo):")
    nuevas = [c for p in "BCD" for c in PELDANOS[p]]
    print((out[nuevas].notna().mean() * 100).round(1).to_string())
    print("\nTargets:")
    for col in TARGETS.values():
        print(f"  {col}: {out[col].value_counts(dropna=False).to_dict()}")
    # coherencia con el dataset plano existente
    plano = ROOT / "data/processed/model_dataset.csv"
    if plano.exists():
        p = pd.read_csv(plano)
        iguales = (p.set_index(ID_COL)[FEATURES].sort_index()
                     .equals(out.set_index(ID_COL)[FEATURES].sort_index()))
        print(f"\nLas 16 basales coinciden con model_dataset.csv: {iguales}")


if __name__ == "__main__":
    main()
