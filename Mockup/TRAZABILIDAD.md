# Trazabilidad de la maqueta con la pregunta de negocio

> **Actualización (Entrega 2, 4 de septiembre de 2026).** Este documento describe la
> maqueta de la Entrega 1 y su contrato de **16 variables basales**. El experimento
> escalera de la Entrega 2 mostró que la antropometría al nacer (LAZ, WAZ y HCZ) es
> el predictor más fuerte disponible al ingreso, y el contrato del tablero real pasó a
> **19 variables**. El contrato vigente está en `src/preprocessing.py` y el tablero
> en `app/dashboard.py`; ver `README.md` y `CHANGELOG.md`.

Maqueta publicada en
**https://fromerom1.github.io/microproyecto-pds-grupo23/Mockup/**

Documento de apoyo a `README.md`. Mientras el README describe **qué es** cada
elemento de la interfaz, este documento justifica **por qué existe**: qué parte de
la pregunta de negocio atiende y con qué hallazgo del análisis exploratorio se
sustenta.

El enunciado de la Entrega 1 pide una maqueta «donde se identifiquen claramente
sus elementos **y su relación con la pregunta de negocio a resolver**». Esta es esa
relación, elemento por elemento.

## La pregunta

> ¿Es posible predecir, a partir del perfil materno al enrolamiento y los datos del
> nacimiento, si un bebé desarrollará desnutrición crónica (stunting) durante sus
> primeros 24 meses de vida, **para priorizar** intervenciones nutricionales y de
> cuidado canguro?

La frase tiene tres partes, y cada vista de la maqueta atiende una:

| Parte de la pregunta | Vista |
|---|---|
| *…perfil materno al enrolamiento y datos del nacimiento…* | Vista de Ingreso |
| *…si un bebé desarrollará desnutrición crónica…* | Vista de Detalle |
| *…para priorizar…* | Vista Principal |

## Vista Principal (Resumen / Triage)

| Elemento | Qué parte de la pregunta atiende | Sustento |
|---|---|---|
| Indicadores de cohorte | Dimensionan el problema y la población objetivo del Método Madre Canguro dentro de la cohorte | 333 bebés · 31 prematuros (9,3 %) · 20 con bajo peso al nacer (6,0 %) |
| Distribución de riesgo (dona) | Muestra cuántos casos hay que atender en cada banda, que es lo que determina la carga de trabajo del programa | Salida del modelo |
| Distribución por tramo y riesgo | Muestra **cuándo** se abre la ventana de intervención: el riesgo se concentra después del mes 12 | Prevalencia 8,5 % al nacer → 15,0 % a 12m → 29,4 % a 24m |
| Lista de cohorte ordenada por riesgo | Es la traducción operativa de «priorizar»: convierte probabilidades en un orden de atención | Decisión de diseño |

La lista ordenada de mayor a menor riesgo es el elemento central de esta vista.
Un programa de seguimiento con recursos limitados no necesita saber la
probabilidad de cada niño: necesita saber **a quién ver primero**.

## Vista de Ingreso (Nuevo Paciente)

| Elemento | Qué parte de la pregunta atiende | Sustento |
|---|---|---|
| Perfil materno (10 variables) | Es literalmente el «perfil materno al enrolamiento» del enunciado | Contrato de modelado, Paso 9.1 del EDA |
| Datos del nacimiento (6 variables) | Son los «datos del nacimiento». Prematurez y bajo peso se derivan de la edad gestacional y el peso capturados | Contrato de modelado, Paso 9.1 |

Las 16 variables son **todas** conocidas en el momento del ingreso al programa.
Ninguna variable posterior al parto entra en el formulario, y esa restricción no
es cosmética: es lo que permite emitir la alerta en tiempo cero, que es el
escenario de mayor valor operativo. El EDA descartó por esta razón la
circunferencia braquial del niño (`zac`), que no existe en el período neonatal.

Las variables con mayor señal univariada en el EDA están todas presentes:
PEG (+29,7 pp), quintil de riqueza (+27,1), bajo peso al nacer (+21,6),
educación materna (+12,0) e inseguridad alimentaria (+11,3).

## Vista de Detalle (Dashboard Individual)

| Elemento | Qué parte de la pregunta atiende | Sustento |
|---|---|---|
| Medidor de riesgo | Es la predicción: la probabilidad de desnutrición crónica a los 24 meses | Salida de `POST /predict` |
| Factores de riesgo (SHAP) | Sin esto la estimación no es auditable por un neonatólogo. Responde al objetivo de «modelos interpretables» de la propuesta de la Fundación Canguro | Propuesta, objetivos específicos |
| Proyección de LAZ | Justifica que la predicción es viable: las trayectorias divergen desde el nacimiento, meses antes de cruzar el umbral clínico | Los bebés con stunting a 24m nacen con LAZ ≈ −1,2 frente a ≈ −0,2 |
| Umbral crítico en −2,0 | Es la definición clínica exacta del stunting, la misma que usa el target del modelo | OMS · `stunted` = LAZ < −2 |

El gráfico SHAP muestra contribuciones en ambas direcciones. Un neonatólogo
necesita ver tanto lo que agrava como lo que protege: si el riesgo alto de un niño
se explica por variables modificables, hay margen de intervención; si se explica
solo por condiciones del nacimiento, la conducta es distinta.

## Relación entre el tablero y el modelo

El enunciado exige que el tablero use el modelo **a través de la API**, no
importándolo directamente. Este es el contrato que necesitan las tres vistas:

| Endpoint | Consumido por | Devuelve |
|---|---|---|
| `POST /predict` | Vista de Detalle | probabilidad, banda de riesgo, contribuciones SHAP |
| `POST /predict/batch` | Vista Principal | probabilidades de la cohorte, ordenadas |
| `GET /model/info` | Encabezado | versión del modelo, hash DVC de los datos, métricas de validación |
| `GET /health` | Encabezado | estado de la API |

El preprocesamiento vive **dentro del artefacto del modelo**, no en el tablero: la
API recibe las categorías en crudo tal como salen del formulario y se encarga de
codificarlas. Así el tablero no puede desincronizarse del modelo cuando cambie el
pipeline, que es el error más común en este tipo de integraciones.

## Alcance y limitaciones

Conviene que queden declaradas en el reporte y, en la versión funcional, visibles
en la propia interfaz:

- **Es una herramienta de priorización, no de diagnóstico.** Indica a quién vigilar
  de cerca; no sustituye la valoración clínica ni la medición antropométrica.
- **Transferencia limitada.** El modelo se entrena sobre una cohorte de Kenia donde
  solo el 9,3 % de los niños es prematuro y el 6,0 % tiene bajo peso al nacer. En un
  Programa Madre Canguro esa proporción es del 100 % por definición del programa.
  Su aplicación a población canguro requiere validación.
- **Horizonte.** El desenlace del dataset está a 24 meses; el protocolo de la
  Fundación Canguro trabaja a 12 meses de edad corregida. El modelo puede
  reportarse en ambos horizontes, y conviene declarar cuál se está mostrando.

## Sobre los datos de la maqueta

Las cifras **descriptivas** —total de la cohorte, prematuros, bajo peso al nacer,
prevalencias por tramo y trayectorias de LAZ— provienen del análisis exploratorio
en `notebooks/01_EDA.ipynb`.

Las cifras de **desempeño del modelo** (probabilidades, bandas de riesgo, valores
SHAP) y los identificadores de los pacientes son **ilustrativos**: el modelo se
entrena en la semana 4 y esos valores se reemplazan por los reales en la Entrega 2.
