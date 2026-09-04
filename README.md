# Predicción temprana de desnutrición crónica infantil

Micro-proyecto — Proyecto Desarrollo de Soluciones, MAIA Uniandes 2026-2, Grupo 23.
En colaboración con la **Fundación Canguro**.

**Maqueta publicada:** https://fromerom1.github.io/microproyecto-pds-grupo23/Mockup/

---

## El problema

La desnutrición crónica infantil es predominantemente **adquirida**: el 82 % de los
niños que la presentan a los 24 meses nacieron sin ella. Eso abre una ventana de
intervención durante el primer año de vida. El problema es que cuando el retraso en
la talla ya es visible en la curva de crecimiento, lleva meses desarrollándose y es
difícil de revertir, y los recursos de seguimiento son limitados.

### Pregunta de negocio

> ¿Es posible predecir, a partir del perfil materno al enrolamiento y los datos del
> nacimiento, si un bebé desarrollará desnutrición crónica (stunting) durante sus
> primeros 24 meses de vida, para priorizar intervenciones nutricionales y de
> cuidado canguro?

### Producto esperado

Un prototipo funcional compuesto por modelos supervisados empaquetados, una API que
sirve las inferencias y un tablero que consume el modelo **a través de esa API**,
todo desplegado en contenedores Docker.

---

## Datos

Cohorte observacional publicada en PLOS Medicine: *Differences in growth trajectories
in breastfed HIV-exposed uninfected and HIV-unexposed infants in Kenya*
(Zenodo, record 15867528).

| | |
|---|---|
| Bebés | 333 |
| Visitas | 2.934 en 9 momentos: nacimiento, semanas 3 y 6, meses 3, 6, 9, 12, 18 y 24 |
| Variables | 39 |
| Target | `stunted` = LAZ < −2 |

Se eligió como **proxy** del problema de la Fundación Canguro porque es longitudinal,
contiene las condiciones neonatales de interés del Método Madre Canguro (prematuro,
bajo peso al nacer, pequeño para edad gestacional) y registra determinantes sociales
y nutricionales modificables.

**Limitación de transferencia:** en esta cohorte solo el 9,3 % de los niños es
prematuro y el 6,0 % tiene bajo peso al nacer. En un Programa Madre Canguro esa
proporción es del 100 % por definición del programa.

### Obtener los datos

El CSV no está en Git: se versiona con DVC contra un bucket S3.

```bash
pip install "dvc[s3]"
dvc pull
```

Requiere credenciales de AWS con acceso al remoto declarado en `.dvc/config`. Si el
remoto no está disponible, el dataset es público: puede descargarse de Zenodo y
colocarse en `data/plosmed_data_newid.csv`.

---

## Estructura del repositorio

```
.dvc/                       configuración de DVC (remoto S3)
data/plosmed_data_newid.csv.dvc   dataset crudo, versionado con DVC (el CSV no está en Git)
data/processed/             dataset plano de modelado (una fila por bebé, 16 features + targets)
notebooks/01_EDA.ipynb      análisis exploratorio
notebooks/02_modelado.ipynb baseline de modelado (LR / RF / GB)
src/preprocessing.py        contrato de modelado y preprocesamiento — fuente única de verdad
src/train_stunting.py       entrenamiento con un split y registro en MLflow (barrido de hiperparámetros)
src/evaluate_cv.py          evaluación robusta: CV estratificada repetida, IC, punto de operación, MLflow
src/features.py             dataset del experimento escalera desde el CSV crudo (z-scores tempranos)
src/experimento_escalera.py cuánto mejora la predicción con cada visita de seguimiento
src/predict.py              módulo de predicción: el contrato que consumen el tablero y la API
app/dashboard.py            tablero Streamlit (tres vistas de la maqueta)
models/                     modelos entrenados (.joblib) y sus metadatos (.json)
figures/                    EDA (01_EDA), baseline (02_model), CV (03_cv), escalera (04_escalera)
Mockup/                     maqueta del prototipo (SPA) — ver Mockup/README.md
docs/                       documentos de las entregas y guías (MLFLOW_EC2.md)
requirements.txt            dependencias, con las del artefacto del modelo fijadas
CHANGELOG.md                registro de cambios del proyecto
```

## Cómo reproducir el análisis exploratorio

```bash
pip install -r requirements.txt
dvc pull
cd notebooks
jupyter notebook 01_EDA.ipynb
```

El notebook usa rutas relativas a su propia carpeta, así que debe ejecutarse desde
`notebooks/`. Las figuras que genera quedan en `notebooks/figures/`; las versionadas
para el reporte están en `figures/01_EDA/`.

## Cómo entrenar, evaluar y usar el modelo

Todo se ejecuta **desde la raíz del repositorio** con `python -m`, para que `src/` sea
importable. Es lo que garantiza que el modelo serializado pueda abrirse después desde
el tablero, la API o un contenedor.

```bash
# Evaluación robusta (CV estratificada 5x10, 7 configuraciones) — ~4 min
python -m src.evaluate_cv --target 24m
python -m src.evaluate_cv --target 12m

# Experimento escalera: requiere el CSV crudo (dvc pull o descarga de Zenodo) — ~6 min
python -m src.features                  # construye data/processed/model_dataset_escalera.csv
python -m src.experimento_escalera      # 4 peldaños x 2 horizontes x 2 familias, CV 5x10

# Barrido de una configuración con un solo split (un run por corrida)
python -m src.train_stunting --model lr --C 0.1

# Prueba rápida del módulo de predicción
python -m src.predict --demo

# Tablero
streamlit run app/dashboard.py
```

`evaluate_cv` deja en `models/model_stunting_<horizonte>_cv.joblib` el modelo de 16
variables basales; `experimento_escalera` deja en `models/model_stunting_<horizonte>_B.joblib`
el del peldaño B (16 basales + z-scores al nacer). En ambos casos el `.json` contiguo
guarda umbrales de banda, curva sensibilidad-vs-capacidad y métricas de CV. **El tablero
y `predict.py` usan B si existe**, porque es el mejor modelo con información disponible
al ingreso.

### El experimento escalera

Cuatro bloques acumulativos de variables, cada uno un momento clínico real, evaluados con
la misma CV 5×10. Responde a *¿en qué momento la predicción se vuelve confiable?*

| Peldaño | Variables | ROC-AUC 24 m | ROC-AUC 12 m |
|---|---|---|---|
| A · basales (ingreso) | 16 | 0.58 [0.48, 0.72] | 0.68 [0.57, 0.81] |
| B · + z-scores al nacer | 19 | **0.72** [0.61, 0.84] | **0.79** [0.67, 0.88] |
| C · + semana 3 | 23 | 0.75 [0.63, 0.87] | 0.81 [0.67, 0.90] |
| D · + mes 3 | 27 | 0.79 [0.68, 0.89] | 0.85 [0.73, 0.94] |

El salto grande está en **B**: la antropometría al nacer, que se mide en el parto y ya se
conoce al ingreso. El contrato original de 16 variables la omitía. El random forest
reproduce la misma escalera, así que el patrón no depende de la familia de modelo.

**MLflow.** Sin configuración, los scripts registran en `./mlruns` (local). Para
registrar en el servidor del equipo en EC2:

```bash
export MLFLOW_TRACKING_URI=http://<ip>:8050      # PowerShell: $env:MLFLOW_TRACKING_URI = "..."
```

La guía completa de la máquina, los pantallazos que pide la rúbrica y cómo detenerla
sin terminarla está en [`docs/MLFLOW_EC2.md`](docs/MLFLOW_EC2.md).

### Por qué validación cruzada y no un solo split

Con 333 bebés, un split 70/15/15 deja ~49 en test. Medido con 20 semillas distintas, el
ROC-AUC de test del mismo modelo oscila entre **0.43 y 0.81**: ese número describe a la
semilla, no al modelo. La CV estratificada repetida (50 evaluaciones) da una media con
intervalo de confianza, y la regla de un error estándar elige, entre las configuraciones
estadísticamente indistinguibles, la más simple.

---

## Maqueta del prototipo

En `Mockup/`. Aplicación de página única con tres vistas —resumen y priorización,
ingreso de paciente y detalle individual— construida con HTML5, CSS y JavaScript
con Chart.js.

- **En vivo:** https://fromerom1.github.io/microproyecto-pds-grupo23/Mockup/
- **Qué es cada elemento:** [`Mockup/README.md`](Mockup/README.md)
- **Por qué existe cada elemento**, y contrato de la API:
  [`Mockup/TRAZABILIDAD.md`](Mockup/TRAZABILIDAD.md)

---

## Entregas

| | Fecha | Contenido |
|---|---|---|
| Entrega 1 | 23 ago 2026 | Problema, pregunta de negocio, datos, EDA, maqueta, repositorios |
| Entrega 2 | 6 sep 2026 | Modelos, experimentos en MLflow, tablero |
| Entrega 3 | 22 sep 2026 | Modelos empaquetados, API, tablero en Docker, manuales, video |

---

## Pendientes técnicos

Backlog conocido, para no perderlo de vista al entrar en la fase de modelado:

- **Pipeline de DVC.** Hoy DVC versiona un archivo suelto; no existe `dvc.yaml` con
  etapas, así que no hay linaje entre datos crudos, features y modelo. Necesario
  desde la semana 4.
- **Calibración.** Con `class_weight="balanced"` el modelo sobreestima las
  probabilidades (curva de calibración en `figures/03_cv/`). Para priorizar importa
  el orden, no el valor absoluto, pero si el tablero muestra probabilidades conviene
  calibrar (`CalibratedClassifierCV`) antes de la Entrega 3.
- **Peldaños C y D en el tablero.** El tablero usa B (ingreso). Para C y D haría falta
  una vista de "actualizar riesgo en el control" que reciba los z-scores de la visita.
- **API.** `src/predict.py` ya expone el contrato (`predecir`, `predecir_lote`,
  `punto_operacion`, `info`); falta envolverlo en FastAPI y contenerizar.

## Nota sobre la publicación

GitHub Pages publica desde la raíz, lo que expone **todo** el repositorio por URL.
Hoy no representa un problema porque los datos son de una cohorte pública. Cuando se
incorporen datos del Observatorio Canguro —historias clínicas anonimizadas bajo
acuerdo de confidencialidad— esto debe revisarse antes de recibirlos.

## Equipo

Grupo 23 · Maestría en Inteligencia Artificial, Universidad de los Andes.

Cada integrante documenta su aporte a través de los commits del repositorio. Para que
GitHub los atribuya correctamente, el correo de `git config user.email` debe estar
registrado en la cuenta de GitHub de cada quien.
