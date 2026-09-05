#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microproyecto PDS - Grupo 23 - Entrega 2
Chequeos de calidad del dataset de modelado.

Cierra los dos huecos que la rubrica senalo en la exploracion de la
Entrega 1 ("no hay correlaciones entre predictores ni chequeos de
consistencia o duplicados"):

  1. Duplicados      identificador repetido y filas con perfil identico.
  2. Consistencia    rangos plausibles y coherencia entre variables que
                     se derivan una de otra.
  3. Asociacion      V de Cramer entre cada par de predictores, para
                     detectar redundancia. El EDA de la Entrega 1 cruzo
                     cada predictor contra el objetivo, nunca los
                     predictores entre si, que es donde vive la
                     colinealidad.

Hallazgo principal: 'preterm' es una funcion determinista de
'gestage_final' (prematuro = edad gestacional < 37 semanas). El cruce no
arroja una sola discrepancia, de modo que ambas columnas aportan la misma
informacion al modelo. El efecto de retirarla se mide en
src/exp_redundancia.py.

Uso:  python src/checks_calidad.py
      python src/checks_calidad.py --data <ruta> --fig <ruta.png>
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- contrato de modelado del EDA: las 16 variables basales -----------------
FEATURES = [
    "enrol_hiv_status_cat", "momage_cat", "educ_cat_n", "marital_cat",
    "hfia_enr", "wealth_quintile", "depression", "mom_muac_cat", "parity",
    "enrol_anemia", "b1_sex", "gestage_final", "caesarean", "preterm",
    "sga", "lbw",
]
ID = "newid"
CONTINUAS = ["gestage_final"]
GESTA_MIN, GESTA_MAX, CORTE_PRETERM = 25.0, 44.0, 37.0


# ---------------------------------------------------------------------------
# 1. DUPLICADOS Y CONSISTENCIA
# ---------------------------------------------------------------------------
def duplicados(df):
    id_rep = int(df[ID].duplicated().sum())
    perfil_rep = int(df.duplicated(subset=FEATURES).sum())
    print("1. DUPLICADOS")
    print(f"   filas                                  : {len(df)}")
    print(f"   identificadores repetidos              : {id_rep}")
    print(f"   filas con perfil basal identico        : {perfil_rep}")
    return {"filas": len(df), "id_repetidos": id_rep, "perfil_repetido": perfil_rep}


def consistencia(df):
    g = pd.to_numeric(df["gestage_final"], errors="coerce")
    pt = df["preterm"].astype(str).str.strip().str.lower()

    faltantes = int(g.isna().sum())
    fuera = int(((g < GESTA_MIN) | (g > GESTA_MAX)).sum())
    # prematuro deberia ser exactamente 'edad gestacional < 37 semanas'
    discrepancias = int((((pt == "yes") & (g >= CORTE_PRETERM)) |
                         ((pt == "no") & (g < CORTE_PRETERM))).sum())

    print("\n2. CONSISTENCIA")
    print(f"   gestage_final faltante                 : {faltantes}")
    print(f"   gestage_final fuera de {GESTA_MIN:.0f}-{GESTA_MAX:.0f} semanas    : {fuera}")
    print(f"   preterm incoherente con gestage < {CORTE_PRETERM:.0f}   : {discrepancias}")
    if discrepancias == 0:
        print("   -> preterm se deriva por completo de gestage_final")
    return {"gestage_faltante": faltantes, "gestage_fuera_rango": fuera,
            "preterm_discrepancias": discrepancias}


# ---------------------------------------------------------------------------
# 2. ASOCIACION ENTRE PREDICTORES (V de Cramer)
# ---------------------------------------------------------------------------
def cramers_v(a, b):
    """V de Cramer con la correccion de sesgo de Bergsma. None si no aplica."""
    tabla = pd.crosstab(a, b)
    if tabla.shape[0] < 2 or tabla.shape[1] < 2:
        return None
    n = tabla.to_numpy().sum()
    if n == 0:
        return None
    obs = tabla.to_numpy(dtype=float)
    esp = np.outer(obs.sum(1), obs.sum(0)) / n
    chi2 = ((obs - esp) ** 2 / esp).sum()
    phi2 = chi2 / n
    r, k = tabla.shape
    phi2c = max(0.0, phi2 - (r - 1) * (k - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    den = min(rc - 1, kc - 1)
    return float(np.sqrt(phi2c / den)) if den > 0 else None


def discretiza(serie):
    """Las continuas se agrupan en cuartiles para poder cruzarlas."""
    v = pd.to_numeric(serie, errors="coerce")
    return pd.qcut(v, 4, duplicates="drop").astype(str).where(v.notna())


def matriz_asociacion(df, ruta_fig):
    datos = {c: (discretiza(df[c]) if c in CONTINUAS
                 else df[c].astype(str).str.strip().replace("nan", np.nan))
             for c in FEATURES}

    M = pd.DataFrame(np.eye(len(FEATURES)), index=FEATURES, columns=FEATURES)
    pares = []
    for x, y in itertools.combinations(FEATURES, 2):
        par = pd.DataFrame({"x": datos[x], "y": datos[y]}).dropna()
        v = cramers_v(par["x"], par["y"]) if len(par) else None
        if v is None:
            v = np.nan
        M.loc[x, y] = M.loc[y, x] = v
        pares.append((v, x, y))

    pares = [p for p in pares if not np.isnan(p[0])]
    pares.sort(reverse=True)

    print("\n3. ASOCIACION ENTRE PREDICTORES (V de Cramer)")
    print(f"   pares evaluados                        : {len(pares)}")
    print("\n   Asociaciones mas fuertes:")
    for v, x, y in pares[:8]:
        marca = "  <-- redundancia" if v >= 0.9 else ("  <- moderada" if v >= 0.5 else "")
        print(f"     {x:<22}{y:<22}{v:.2f}{marca}")

    altas = [p for p in pares if p[0] >= 0.5]
    print(f"\n   pares con V >= 0.50                    : {len(altas)}")

    # --- figura ---
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    im = ax.imshow(M.to_numpy(dtype=float), cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_yticks(range(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=90, fontsize=7.5)
    ax.set_yticklabels(FEATURES, fontsize=7.5)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            val = M.iat[i, j]
            if not np.isnan(val) and i != j and val >= 0.30:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.5, color="black")
    ax.set_title("Asociación entre predictores (V de Cramér)", fontsize=11, pad=10)
    # La V subestima el par preterm / gestage_final: al discretizar la continua
    # en cuartiles se pierde el corte de 37 semanas que define la binaria. La
    # dependencia real la establece el chequeo de consistencia (0 discrepancias).
    fig.text(0.5, -0.02,
             "La dependencia entre preterm y gestage_final no se aprecia aquí: al agrupar la continua en "
             "cuartiles\nse pierde el corte de 37 semanas que define la binaria. La establece el chequeo de "
             "consistencia.",
             ha="center", va="top", fontsize=7.5, color="#444444")
    fig.colorbar(im, ax=ax, fraction=0.045, label="V de Cramér")
    plt.tight_layout()
    Path(ruta_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_fig, dpi=150, bbox_inches="tight")
    print(f"\n   Figura -> {ruta_fig}")
    return pares


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/model_dataset.csv")
    ap.add_argument("--fig", default="figures/01_EDA/fig7_asociacion_predictores.png")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    print(f"Dataset: {args.data}  ({df.shape[0]} filas x {df.shape[1]} columnas)\n")

    duplicados(df)
    consistencia(df)
    matriz_asociacion(df, args.fig)


if __name__ == "__main__":
    main()
