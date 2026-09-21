"""
Tablero de apoyo a la priorizacion de seguimiento nutricional.
Micro-proyecto PDS - Grupo 23 - Entrega 2/3.

Tres vistas, las mismas de la maqueta (Mockup/):
    Cohorte           que tan grande es el problema y en quien se concentra
    Riesgo individual la prediccion para un bebe y por que
    Priorizacion      a quien ver primero, dada la capacidad del programa

El tablero NO importa scikit-learn ni el modelo directamente: consume la API
de api/ (GET /model/variants, GET /model/info, POST /predict, POST
/predict/batch, GET /model/operating-point), como exige el enunciado ("el
tablero que consume el modelo a traves de esa API"). La unica excepcion es el
panel de importancia de variables para la variante "B": necesita coeficientes
del pipeline que la API todavia no expone como endpoint, asi que carga el
artefacto local solo para ese panel y se omite si no esta disponible (por
ejemplo si el tablero corre contra una API remota sin modelos locales).

Arranque (con la API corriendo aparte, ver api/README.md):
    python -m streamlit run app/dashboard.py

Variables de entorno:
    DASHBOARD_API_URL   base de la API (por defecto http://localhost:8000)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.preprocessing import CATEGORIAS, TARGETS                         # noqa: E402

API_URL = os.environ.get("DASHBOARD_API_URL", "http://localhost:8000").rstrip("/")

# ---------------------------------------------------------------------------
st.set_page_config(page_title="Riesgo nutricional temprano · Grupo 23",
                   page_icon="🍼", layout="wide", initial_sidebar_state="expanded")

AZUL, ROJO, AMBAR, VERDE, GRIS = "#1f5f8b", "#a33235", "#b4700e", "#2c6e49", "#8a929e"
COLOR_BANDA = {"alto": ROJO, "medio": AMBAR, "bajo": VERDE}
ETIQ_BANDA = {"alto": "Riesgo alto", "medio": "Riesgo medio", "bajo": "Riesgo bajo"}

# Etiquetas legibles para los valores de las categorias (el modelo recibe el valor crudo)
ETIQ_VALOR = {
    "Negative": "Negativo", "Positive": "Positivo",
    "25 and less": "25 años o menos", "26 to 35": "26 a 35 años", "35 and more": "Más de 35 años",
    "Primary or below": "Primaria o menos", "Secondary and above": "Secundaria o más",
    "Married": "Casada / unión", "Other": "Otro",
    "lowest quintile": "Q1 (más bajo)", "quintile2": "Q2", "quintile3": "Q3", "quintile4": "Q4", "highest quintile": "Q5 (más alto)",
    "no depression": "Sin depresión", "mild": "Leve", "moderate or severe": "Moderada o severa",
    "undernutrition": "Desnutrición", "normal": "Normal", "above normal": "Por encima de lo normal",
    "female": "Femenino", "male": "Masculino",
    "secured": "Asegurada", "moderate": "Moderada", "severe": "Severa",
    "nulli": "Nulípara", "multi": "Multípara",
    "no": "No", "yes": "Sí",
}
def etq(v): return ETIQ_VALOR.get(v, v)


# ---------------------------------------------------------------------------
# Cliente de la API (GET /model/variants, /model/info, /model/operating-point,
# POST /predict, /predict/batch — el contrato completo esta en api/README.md)
# ---------------------------------------------------------------------------
class ApiError(RuntimeError):
    """La API no respondio, o respondio con un error de negocio (4xx/5xx)."""


def _api_call(method: str, path: str, **kwargs) -> dict | list:
    try:
        r = httpx.request(method, f"{API_URL}{path}", timeout=kwargs.pop("timeout", 10), **kwargs)
    except httpx.HTTPError as error:
        raise ApiError(f"No se pudo conectar con la API en {API_URL}{path} ({error}).") from error
    if r.status_code >= 400:
        try:
            detalle = r.json().get("detail", r.text)
        except Exception:
            detalle = r.text
        raise ApiError(f"{path} respondió {r.status_code}: {detalle}")
    return r.json()


@st.cache_data(ttl=30, show_spinner=False)
def api_variantes() -> dict[str, list[str]]:
    """GET /model/variants -> horizontes y variantes disponibles."""
    return _api_call("GET", "/model/variants")


@st.cache_data(ttl=30, show_spinner=False)
def api_info(horizonte: str, variante: str) -> dict:
    """GET /model/info -> metadatos del modelo, incluida la curva de capacidad."""
    return _api_call("GET", "/model/info", params={"horizonte": horizonte, "variante": variante})


def api_predecir(horizonte: str, variante: str, registro: dict) -> dict:
    """POST /predict -> probabilidad, banda, percentil, contribuciones y variante usada."""
    return _api_call("POST", "/predict", params={"horizonte": horizonte, "variante": variante},
                     json={"registro": registro})


@st.cache_data(ttl=30, show_spinner="Consultando la API…")
def api_predecir_lote(horizonte: str, variante: str, registros: list[dict],
                      capacidad: float) -> list[dict]:
    """POST /predict/batch -> lista ordenada por riesgo, con banda y variante por fila."""
    return _api_call("POST", "/predict/batch",
                     params={"horizonte": horizonte, "variante": variante, "capacidad": capacidad},
                     json={"registros": registros}, timeout=30)


def api_punto_operacion(horizonte: str, variante: str, capacidad: float,
                        curva_capacidad: list[dict]) -> dict:
    """GET /model/operating-point -> sensibilidad/VPP/umbral esperados a esa capacidad.

    La API no recorta la capacidad al rango valido (lo hace explicito en su
    contrato); se ajusta aqui contra la propia curva del modelo, igual que
    antes hacia ModeloRiesgo.punto_operacion localmente.
    """
    curva = pd.DataFrame(curva_capacidad)
    capacidad_ajustada = float(np.clip(capacidad, curva.capacidad.min(), curva.capacidad.max()))
    return _api_call("GET", "/model/operating-point",
                     params={"horizonte": horizonte, "variante": variante,
                             "capacidad": capacidad_ajustada})


def _registros_seguros(df: pd.DataFrame) -> list[dict]:
    """dict JSON-serializable por fila: NaN -> None, escalares numpy -> nativos de Python."""
    # astype(object) ANTES del where: en una columna float, `where(..., None)` no
    # guarda None —float no puede— y deja el NaN, con lo que json.dumps revienta
    # con "Out of range float values are not JSON compliant". Pasa con la cohorte
    # real, donde 48 de 333 filas tienen algun faltante. Es el mismo patron que
    # api/api.py usa al serializar la respuesta del lote.
    limpio = df.astype(object).where(df.notna(), None)
    registros = []
    for fila in limpio.to_dict(orient="records"):
        registros.append({k: (v.item() if isinstance(v, np.generic) else v) for k, v in fila.items()})
    return registros


@st.cache_resource(show_spinner=False)
def _modelo_local(horizonte: str, variante: str):
    """Solo para el panel de importancia de variables (coeficientes/SHAP del
    pipeline): la API no lo expone todavia. Si no hay artefactos locales
    (p. ej. el tablero corre contra una API remota), el panel se omite."""
    try:
        from src.predict import ModeloRiesgo
        return ModeloRiesgo(horizonte, variante)
    except Exception:
        return None


@st.cache_data
def cargar_cohorte() -> pd.DataFrame:
    esc = ROOT / "data/processed/model_dataset_escalera.csv"
    return pd.read_csv(esc if esc.exists() else ROOT / "data/processed/model_dataset.csv")


@st.cache_data(ttl=30, show_spinner="Consultando la API…")
def scores_cohorte(horizonte: str, variante: str, capacidad: float) -> pd.DataFrame:
    registros = _registros_seguros(cargar_cohorte())
    resultados = api_predecir_lote(horizonte, variante, registros, capacidad)
    return pd.DataFrame(resultados)


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
try:
    disponibles = api_variantes()
except ApiError as error:
    st.error(f"No se pudo conectar con la API de predicción en `{API_URL}`.\n\n"
             f"{error}\n\nVerifica que esté corriendo (`python -m api` desde la raíz del "
             f"repositorio) o ajusta `DASHBOARD_API_URL`.")
    st.stop()
if not disponibles:
    st.error("La API respondió pero no reporta modelos disponibles.")
    st.stop()

with st.sidebar:
    st.title("Riesgo nutricional temprano")
    st.caption("Cohorte de seguimiento · Grupo 23")
    horizonte = st.radio("Horizonte de predicción", list(disponibles.keys()),
                         format_func=lambda h: f"{h[:-1]} meses", horizontal=True)
    variante = st.radio("Variante del modelo", disponibles[horizonte], horizontal=True,
                        help="B: 16 basales + Z-scores al nacer (experimento escalera). "
                             "cv: 16 basales solamente, validación cruzada.")
    capacidad = st.slider("Capacidad de seguimiento estrecho", 5, 50, 20, 5,
                          format="%d%%",
                          help="Porcentaje de la cohorte que el programa puede vigilar de cerca. "
                               "Define el punto de operación y las bandas de riesgo.") / 100
    st.divider()
    try:
        info = api_info(horizonte, variante)
    except ApiError as error:
        st.error(f"No se pudo obtener /model/info de la API.\n\n{error}")
        st.stop()
    st.caption("**Modelo**")
    st.caption(f"{info['peldano']} · {info['n_features']} variables · variante `{info['variante']}`")
    st.code(f"{info['config']}  ·  {info['familia']}\n"
            f"ROC-AUC (CV 5×10): {info['roc_auc_cv']:.2f}  [{info['roc_auc_ci'][0]:.2f}, {info['roc_auc_ci'][1]:.2f}]\n"
            f"PR-AUC (CV):       {info['pr_auc_cv']:.2f}\n"
            f"n = {info['n_entrenamiento']}  ·  datos {info['data_md5'][:8]}\n"
            f"sklearn {info['sklearn']}", language=None)

cohorte = cargar_cohorte()
try:
    scores = scores_cohorte(horizonte, variante, capacidad)
    po = api_punto_operacion(horizonte, variante, capacidad, info["curva_capacidad"])
except ApiError as error:
    st.error(f"La API falló al calcular los puntajes de la cohorte.\n\n{error}")
    st.stop()

# ---------------------------------------------------------------------------
tab_cohorte, tab_individual, tab_prior = st.tabs(["Cohorte", "Riesgo individual", "Priorización"])


# ═══════════════════════════════ COHORTE ════════════════════════════════════
with tab_cohorte:
    st.subheader("Panorama de la cohorte")
    st.caption("Dimensiona el problema y muestra en quién se concentra. Las cifras de la cohorte "
               "provienen del dataset; las de riesgo, del modelo (vía la API).")

    n = len(cohorte)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bebés en la cohorte", n)
    c2.metric("Prematuros", int((cohorte.preterm == "yes").sum()),
              f"{100*(cohorte.preterm == 'yes').mean():.1f} % de la cohorte", delta_color="off")
    c3.metric("Bajo peso al nacer", int((cohorte.lbw == "yes").sum()),
              f"{100*(cohorte.lbw == 'yes').mean():.1f} % de la cohorte", delta_color="off")
    c4.metric(f"Marcados para seguimiento ({int(capacidad*100)} %)",
              int(scores.seguimiento.sum()))

    izq, der = st.columns([1.1, 1])
    with izq:
        st.markdown("**Distribución del riesgo estimado**")
        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        ax.hist(scores.probabilidad, bins=20, color=AZUL, alpha=.85, edgecolor="white")
        ax.axvline(info["umbral_alto"], color=ROJO, ls="--", lw=1.5, label="Umbral riesgo alto")
        ax.axvline(info["umbral_medio"], color=AMBAR, ls="--", lw=1.5, label="Umbral riesgo medio")
        ax.set_xlabel("Probabilidad estimada de desnutrición crónica"); ax.set_ylabel("Bebés")
        ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.3)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        conteo = scores.banda.value_counts().reindex(["alto", "medio", "bajo"]).fillna(0).astype(int)
        st.dataframe(pd.DataFrame({"Banda": [ETIQ_BANDA[b] for b in conteo.index],
                                   "Bebés": conteo.values,
                                   "% de la cohorte": (100 * conteo.values / n).round(1)}),
                     hide_index=True, use_container_width=True)

    with der:
        st.markdown("**Variables que más pesan en el modelo**")
        modelo_local = _modelo_local(horizonte, variante)
        p_imp = ROOT / f"figures/03_cv/importancias_{info['config']}_{horizonte}.csv"
        if variante == "cv" and p_imp.exists():
            imp = pd.read_csv(p_imp, index_col=0).sort_values("media")
            xerr, xlabel = imp.de, "Importancia agregada · media ± DE entre 50 folds"
        elif modelo_local is not None:
            # |coeficiente| agregado por variable original, del modelo en uso
            from src.preprocessing import variable_original
            clf, pre = modelo_local.pipeline.named_steps["clf"], modelo_local.pipeline.named_steps["prep"]
            nombres = [variable_original(n) for n in pre.get_feature_names_out()]
            imp = (pd.Series(np.abs(clf.coef_[0]), index=nombres).groupby(level=0).sum()
                     .rename("media").to_frame().sort_values("media"))
            xerr, xlabel = None, "|coeficiente| agregado por variable · modelo en uso"
        else:
            imp = None
        if imp is not None:
            from src.predict import ETIQUETAS
            fig, ax = plt.subplots(figsize=(6.2, 4.6))
            ax.barh([ETIQUETAS.get(v, v) for v in imp.index], imp.media, xerr=xerr,
                    color=AZUL, ecolor=GRIS, capsize=3, alpha=.9)
            ax.set_xlabel(xlabel)
            ax.grid(axis="x", alpha=.3)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
            fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
        else:
            st.caption("Panel no disponible: requiere acceso local a los artefactos del modelo "
                       "(la API todavía no expone importancias de variables como endpoint).")

    st.divider()
    st.markdown("**Contexto del análisis exploratorio**")
    st.caption("Cuándo se abre la ventana de intervención y cómo divergen las trayectorias. "
               "Figuras del EDA (`notebooks/01_EDA.ipynb`).")
    f1, f2 = st.columns(2)
    for col, nombre, cap in ((f1, "fig3_prevalencia_por_visita.png", "Prevalencia de stunting por visita"),
                             (f2, "fig5_laz_trayectoria.png", "Trayectoria de LAZ según desenlace a 24 meses")):
        p = ROOT / "figures/01_EDA" / nombre
        if p.exists():
            col.image(str(p), caption=cap, use_container_width=True)


# ═══════════════════════════ RIESGO INDIVIDUAL ══════════════════════════════
with tab_individual:
    st.subheader("Estimación de riesgo al ingreso")
    st.caption(f"{info['n_features']} variables, todas conocidas en el momento del ingreso al programa. "
               "Ninguna posterior al parto.")

    def selector(col, var, etiqueta, default=None):
        ops = CATEGORIAS[var]
        idx = ops.index(default) if default in ops else 0
        return col.selectbox(etiqueta, ops, index=idx, format_func=etq, key=f"f_{var}")

    with st.form("formulario_riesgo"):
        cm, cn = st.columns(2)
        with cm:
            st.markdown("**Perfil materno al enrolamiento** · 10")
            a, b = st.columns(2)
            reg = {}
            reg["momage_cat"]           = selector(a, "momage_cat", "Edad materna")
            reg["educ_cat_n"]           = selector(b, "educ_cat_n", "Educación")
            reg["marital_cat"]          = selector(a, "marital_cat", "Estado marital")
            reg["wealth_quintile"]      = selector(b, "wealth_quintile", "Quintil de riqueza")
            reg["hfia_enr"]             = selector(a, "hfia_enr", "Inseguridad alimentaria")
            reg["mom_muac_cat"]         = selector(b, "mom_muac_cat", "Nutrición materna (MUAC)", "normal")
            reg["enrol_anemia"]         = selector(a, "enrol_anemia", "Anemia")
            reg["depression"]           = selector(b, "depression", "Depresión")
            reg["parity"]               = selector(a, "parity", "Paridad")
            reg["enrol_hiv_status_cat"] = selector(b, "enrol_hiv_status_cat", "Estado VIH")
        with cn:
            usa_z = bool(info.get("features_extra"))
            st.markdown(f"**Condiciones del nacimiento** · {6 + (3 if usa_z else 0)}")
            a, b = st.columns(2)
            reg["b1_sex"]        = selector(a, "b1_sex", "Sexo")
            reg["gestage_final"] = b.number_input("Edad gestacional (semanas)", 25.0, 44.0, 39.0, 0.5, key="f_gest")
            reg["caesarean"]     = selector(a, "caesarean", "Cesárea")
            reg["preterm"]       = selector(b, "preterm", "Prematuro")
            reg["sga"]           = selector(a, "sga", "Pequeño para edad gestacional")
            reg["lbw"]           = selector(b, "lbw", "Bajo peso al nacer")
            if usa_z:
                st.caption("Antropometría al nacer, en puntajes Z (OMS). Se miden en el parto y "
                           "son el predictor más fuerte del experimento escalera.")
                z1, z2, z3 = st.columns(3)
                reg["zlen_nac"] = z1.number_input("LAZ al nacer", -6.0, 6.0, -0.5, 0.1, key="f_zlen",
                                                  help="Talla para la edad. < −2 indica retraso en el crecimiento.")
                reg["zwei_nac"] = z2.number_input("WAZ al nacer", -6.0, 6.0, -0.5, 0.1, key="f_zwei",
                                                  help="Peso para la edad.")
                reg["zhc_nac"]  = z3.number_input("HCZ al nacer", -6.0, 6.0, 0.0, 0.1, key="f_zhc",
                                                  help="Perímetro cefálico para la edad.")
        enviado = st.form_submit_button("Calcular riesgo", type="primary", use_container_width=True)

    if enviado:
        try:
            r = api_predecir(horizonte, variante, reg)
        except ApiError as error:
            st.error(f"La API no pudo calcular la predicción.\n\n{error}")
            st.stop()
        band = r["banda"]
        g, e = st.columns([1, 1.6])
        with g:
            st.markdown(
                f"<div style='border:1px solid #e3e7ec;border-radius:8px;padding:18px;text-align:center'>"
                f"<div style='font-size:12px;color:{GRIS}'>Probabilidad estimada de desnutrición crónica a {horizonte[:-1]} meses</div>"
                f"<div style='font-size:46px;font-weight:700;color:{COLOR_BANDA[band]};line-height:1.1'>{r['probabilidad']:.2f}</div>"
                f"<div style='display:inline-block;margin-top:6px;padding:3px 12px;border-radius:20px;"
                f"background:{COLOR_BANDA[band]}22;color:{COLOR_BANDA[band]};font-weight:600;font-size:12px;letter-spacing:.06em'>"
                f"{ETIQ_BANDA[band].upper()}</div>"
                f"<div style='font-size:12px;color:{GRIS};margin-top:8px'>Percentil {r['percentil']:.0f} de la cohorte</div>"
                f"</div>", unsafe_allow_html=True)
            st.caption(f"Con capacidad del {int(capacidad*100)} %, este bebé "
                       f"{'**entra**' if r['probabilidad'] >= po['umbral'] else '**no entra**'} "
                       f"en el grupo de seguimiento estrecho (umbral {po['umbral']:.2f}).")
            st.caption(f"Estimado con la variante **{r['variante']}** del modelo "
                       f"({info['peldano']}).")
        with e:
            st.markdown("**Qué empuja esta estimación**")
            contrib = pd.DataFrame(r["contribuciones"])
            contrib = contrib[contrib.contribucion.abs() > 1e-6].head(10).iloc[::-1]
            fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(contrib) + 1))
            colores = [ROJO if v > 0 else VERDE for v in contrib.contribucion]
            def etiqueta_barra(c):
                v = c.valor
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return f"{c.etiqueta} · {v:g}"
                return f"{c.etiqueta} · {etq(str(v))}"
            labels = [etiqueta_barra(c) for c in contrib.itertuples()]
            ax.barh(labels, contrib.contribucion, color=colores)
            ax.axvline(0, color=GRIS, lw=1)
            ax.set_xlabel(f"← reduce el riesgo · aumenta el riesgo →   ({r['unidad_contribucion']})")
            ax.grid(axis="x", alpha=.3)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
            fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        st.warning("**Estimación de apoyo, no diagnóstico.** Indica prioridad de seguimiento y no sustituye la "
                   "valoración clínica ni la medición antropométrica. El modelo se entrenó sobre una cohorte de "
                   "Kenia con 9 % de prematuros; su transferencia a población canguro requiere validación.",
                   icon="⚠️")


# ═══════════════════════════════ PRIORIZACIÓN ═══════════════════════════════
with tab_prior:
    st.subheader("Lista priorizada de seguimiento")
    st.caption(f"Punto de operación con capacidad del {int(capacidad*100)} % · "
               f"umbral de probabilidad {po['umbral']:.2f}. Las métricas son las de la validación cruzada, "
               f"no las del ajuste en la cohorte.")

    n_marc = int(scores.seguimiento.sum())
    positivos_est = int(round(po["sensibilidad"] * (cohorte[TARGETS[horizonte]] == "yes").sum()))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bebés marcados", n_marc, f"{int(capacidad*100)} % de {len(scores)}", delta_color="off")
    c2.metric("Sensibilidad esperada", f"{100*po['sensibilidad']:.0f} %",
              help="De los bebés que desarrollarán la condición, qué fracción queda dentro del grupo marcado (CV).")
    c3.metric("Valor predictivo positivo", f"{100*po['vpp']:.0f} %",
              help="De los bebés marcados, qué fracción desarrollará la condición (CV).")
    c4.metric("Prevalencia de base", f"{100*info['prevalencia']:.0f} %",
              f"{100*(po['vpp']-info['prevalencia']):+.0f} pp vs. azar")

    izq, der = st.columns([1.5, 1])
    with izq:
        solo_marcados = st.toggle("Mostrar solo los marcados", value=True)
        tabla = scores if not solo_marcados else scores[scores.seguimiento]
        vista = pd.DataFrame({
            "#": tabla.ranking,
            "ID": tabla["newid"],
            "Prob.": tabla.probabilidad.round(3),
            "Banda": tabla.banda.map(ETIQ_BANDA),
            "Variante": tabla.variante,
            "Prematuro": tabla.preterm.map(etq).fillna("—"),
            "PEG": tabla.sga.map(etq).fillna("—"),
            "Bajo peso": tabla.lbw.map(etq).fillna("—"),
            "Quintil": tabla.wealth_quintile.map(etq).fillna("—"),
            "Seguimiento": np.where(tabla.seguimiento, "Sí", "—"),
        })
        st.dataframe(vista, hide_index=True, use_container_width=True, height=440)
        st.download_button("Descargar lista priorizada (CSV)",
                           tabla.drop(columns=["seguimiento"]).assign(seguimiento=tabla.seguimiento.map({True: "si", False: "no"}))
                                .to_csv(index=False).encode("utf-8"),
                           file_name=f"lista_priorizada_{horizonte}_{variante}_cap{int(capacidad*100)}.csv", mime="text/csv")
    with der:
        st.markdown("**Cómo cambia el punto de operación con la capacidad**")
        curva = pd.DataFrame(info["curva_capacidad"])
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        ax.plot(curva.capacidad * 100, curva.sensibilidad * 100, "o-", color=AZUL, lw=2, label="Sensibilidad")
        ax.plot(curva.capacidad * 100, curva.vpp * 100, "s-", color=VERDE, lw=2, label="VPP")
        ax.axhline(100 * info["prevalencia"], ls=":", color=GRIS, lw=1, label="Prevalencia")
        ax.axvline(capacidad * 100, ls="--", color=ROJO, lw=1.5)
        ax.set_xlabel("% de la cohorte marcada"); ax.set_ylabel("%")
        ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.3)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
        st.caption("Marcar a más bebés sube la sensibilidad y baja el VPP. La capacidad real del "
                   "programa, no un umbral estadístico, es lo que fija el punto. Curva servida por "
                   "`GET /model/info`.")
