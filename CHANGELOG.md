# Registro de cambios

Cambios relevantes del proyecto, con la razón detrás de cada uno. El orden es del
más reciente al más antiguo.

---

## 4 de septiembre de 2026 — Evaluación robusta, módulo de predicción y tablero

*Rama `feat/evaluacion-cv-y-tablero` · Alejandro Mesa*

### Añadido

- **`src/preprocessing.py`.** Contrato de modelado y preprocesamiento como fuente
  única de verdad: las 16 features, el target, las categorías válidas, `Winsorizer`
  y `build_preprocessor()`. Lo importan entrenamiento, evaluación, predicción y
  tablero.
- **`src/evaluate_cv.py`.** Evaluación con validación cruzada estratificada repetida
  (5×10) para siete configuraciones de tres familias. Por fold: ROC-AUC, PR-AUC,
  Brier y punto de operación por capacidad (sensibilidad y VPP al marcar el 20 % de
  mayor riesgo). Agrega media, DE e IC95 %; curvas ROC/PR/calibración out-of-fold;
  curva sensibilidad-vs-capacidad; estabilidad de importancias entre folds. Selección
  por la regla de un error estándar. Reajusta la elegida con toda la cohorte y la
  guarda en `models/` con sus metadatos. Todo en MLflow: un run padre por barrido,
  un run hijo por configuración.
- **`src/predict.py`.** `ModeloRiesgo` con `predecir`, `predecir_lote`,
  `punto_operacion` e `info` — el mismo contrato de la API en
  `Mockup/TRAZABILIDAD.md`. Contribuciones por variable (coeficiente × valor para la
  logística, SHAP para árboles), banda de riesgo, percentil en la cohorte.
- **`app/dashboard.py`.** Tablero Streamlit con las tres vistas de la maqueta,
  consumiendo `predict.py`: cohorte (indicadores reales, distribución de riesgo,
  importancias, figuras del EDA), riesgo individual (formulario de 16 variables,
  probabilidad, banda, percentil, contribuciones en ambas direcciones, aviso
  clínico) y priorización (control de capacidad, lista ordenada, métricas del punto
  de operación tomadas de la CV, curva de capacidad, descarga CSV).
- **`docs/MLFLOW_EC2.md`.** Guía para la máquina EC2 con MLflow, los pantallazos que
  exige la rúbrica y cómo detenerla sin terminarla.
- **`figures/03_cv/`** y **`models/model_stunting_{24m,12m}_cv.{joblib,json}`.**
- **`src/features.py`.** Construye `data/processed/model_dataset_escalera.csv` desde el
  CSV crudo: las 16 basales más los z-scores de talla, peso y perímetro cefálico en
  las visitas `delivery`, `week-3` y `month-3`, y la velocidad de LAZ. Verifica que
  las basales coincidan con `model_dataset.csv`.
- **`src/experimento_escalera.py`.** Cuatro peldaños acumulativos (A basales → B +
  nacer → C + semana 3 → D + mes 3), dos horizontes, dos familias, CV 5×10. Figura
  de escalera con IC, importancias del peldaño D, y guarda el modelo del peldaño B
  como `models/model_stunting_{24m,12m}_B.{joblib,json}`. Experimento MLflow
  `stunting-escalera`.
- **`predict.py` y el tablero usan el peldaño B por defecto** si existe (con `cv`
  como respaldo). El formulario de riesgo individual añade tres campos: LAZ, WAZ y
  HCZ al nacer. `build_preprocessor()` acepta `numericas_extra`.

### Corregido

- **Los modelos guardados no podían cargarse fuera del notebook.** `Winsorizer`
  estaba definida dentro del notebook y del script, así que el pickle la
  referenciaba como `__main__.Winsorizer` y cualquier otro proceso — tablero, API,
  contenedor — fallaba al abrirlo. Ahora vive en `src/preprocessing.py` y se
  serializa como `src.preprocessing.Winsorizer`. `train_stunting.py` la importa de
  ahí (cambio de ~15 líneas, sin tocar su lógica); verificado que el modelo que
  registra en MLflow carga desde otro proceso.
- **`Winsorizer.get_feature_names_out()`.** Sin él, `ColumnTransformer` no podía
  devolver los nombres de las features transformadas y no se podían mapear
  importancias ni contribuciones a la variable original.
- **`requirements.txt`.** Fijadas las versiones que participan en el pickle del
  modelo (numpy, pandas, scikit-learn, joblib). Añadido `streamlit`.
- **`.gitignore`.** `mlruns/` y `mlartifacts/`.

### Hallazgos

- **Un solo split 70/15/15 no discrimina.** Con 20 semillas distintas, el ROC-AUC de
  test del mismo modelo va de 0.43 a 0.81. Las métricas de `model_metadata.json`
  (RF: test 0.62) están dentro de ese rango de ruido.
- **Con CV, todas las familias son indistinguibles a 24 meses** (ROC-AUC 0.57–0.60,
  IC que roza 0.5). La logística iguala al random forest; el gradient boosting
  sobreajusta. Se elige `lr_C0.1` por simplicidad e interpretabilidad.
- **A 12 meses el modelo discrimina mejor** (ROC-AUC 0.68–0.70, IC [0.57, 0.81]).
  Coherente con el EDA: el 82 % de los casos a 24 meses son adquiridos después del
  nacimiento, así que las variables basales pierden poder en ese horizonte.
- **El modelo sobreestima probabilidades** (efecto de `class_weight="balanced"`).
  El orden es válido para priorizar; el valor absoluto no está calibrado.
- **Punto de operación al 20 %** (24 m, 16 basales): sensibilidad 27 %, VPP 40 %
  frente a una prevalencia del 29 %. Lift modesto pero real.
- **Escalera (regresión logística, ROC-AUC).** 24 m: A 0.58 → B 0.72 (+0.14) → C 0.75
  → D 0.79. 12 m: A 0.68 → B 0.79 (+0.11) → C 0.81 → D 0.85. **El salto grande es
  B, la antropometría al nacer, que ya se conoce al ingreso**: el contrato de 16
  variables la omitía y con ella el modelo de tiempo cero pasa de rozar el azar a
  discriminar con claridad. Al 20 % de capacidad, B sube el VPP de 39 % a 52 %
  (24 m). Con la talla real al nacer en el modelo, la bandera binaria de PEG pierde
  peso: es redundante.
- El random forest reproduce la misma escalera; el patrón no depende de la familia.

---

## 23 de agosto de 2026 — Trazabilidad y coherencia de la maqueta

*Rama `fix/maqueta-consistencia-datos` · Alejandro Mesa*

### Añadido

- **`Mockup/TRAZABILIDAD.md`.** El enunciado pide una maqueta «donde se identifiquen
  claramente sus elementos **y su relación con la pregunta de negocio a resolver**».
  `Mockup/README.md` explica qué es cada elemento; este documento explica por qué
  existe: qué parte de la pregunta atiende y con qué hallazgo del EDA se sustenta.
  Incluye el contrato de la API y una sección de alcance y limitaciones.
- **`Mockup/vendor/chart.umd.min.js`.** Chart.js v4.5.1 (MIT), copiado del paquete
  oficial de npm.
- **`.nojekyll`** en la raíz.
- **`README.md`** del repositorio, que era un archivo de dos líneas.
- **`CHANGELOG.md`**, este archivo.

### Corregido

- **Cifras de cohorte incoherentes con el dataset.** La vista principal declaraba
  1.245 pacientes, 320 en gestación y 925 neonatales. El dataset descrito en el
  reporte tiene 333 bebés y no contiene registros de etapa gestacional: la primera
  visita de la cohorte es el nacimiento. Un lector que comparara la sección de datos
  con la de la maqueta habría detectado la contradicción.

  | Antes | Ahora | Origen |
  |---|---|---|
  | Total Pacientes 1.245 | 333 | dimensión del dataset |
  | Gestación 320 | Prematuros 31 | `preterm` = yes, 9,3 % |
  | Neonatales 925 | Bajo peso al nacer 20 | `lbw` = yes, 6,0 % |
  | En Riesgo 186 | En Riesgo Alto 67 | ilustrativo, ~20 % de 333 |

- **Tramos del gráfico de barras.** Pasaron de categorías gestacionales inexistentes
  a visitas reales de seguimiento (nacimiento, 0-6, 6-12, 12-18, 18-24 meses), con la
  distribución de riesgo creciendo con la edad, que es el hallazgo del EDA: la
  prevalencia salta de 8,5 % al nacer a 15,0 % a los 12 meses y 29,4 % a los 24.
- **Etiquetas de edad corregida.** «Edad Corregida» pasó a «Edad de seguimiento» y
  «Meses de Edad Corregida» a «Meses de seguimiento». La cohorte de Kenia no es de
  prematuros y no maneja edad corregida; ese concepto aplicará cuando se incorporen
  datos del Observatorio Canguro.
- **Formulario de ingreso.** Capturaba 7 variables; el contrato de modelado del Paso
  9.1 del EDA define 16. Se agregaron edad materna, estado marital, inseguridad
  alimentaria del hogar, estado nutricional materno (MUAC), paridad, estado VIH, sexo
  del recién nacido y tipo de parto. Prematurez y bajo peso se derivan de la edad
  gestacional y el peso, de modo que 15 controles cubren las 16 variables.
- **Gráfico SHAP.** Mostraba cuatro barras, todas positivas. Un gráfico SHAP con una
  sola dirección pierde la mitad del argumento de explicabilidad: un clínico necesita
  ver qué protege tanto como qué agrava. Se agregaron contribuciones negativas y el
  color depende del signo.
- **Dependencia de internet.** Chart.js venía de `cdn.jsdelivr.net`; sin conexión los
  tres gráficos desaparecían. Ahora se sirve desde `Mockup/vendor/`.
- **Degradación de los iconos.** Material Symbols son ligaduras tipográficas: si la
  fuente no carga, el navegador imprime el nombre literal del icono («bar_chart»,
  «person_add») en letra grande. Ahora se ocultan por CSS y se revelan solo si la
  fuente está realmente disponible, comprobado por medición de ancho —
  `document.fonts.check()` no sirve para esto porque devuelve `true` aunque la familia
  no exista, ya que cuenta la fuente de reemplazo.

### Por qué `.nojekyll`

GitHub Pages publica desde la raíz del repositorio, de modo que GitHub procesa el
sitio con Jekyll. Jekyll filtra ciertas rutas antes de publicar, entre ellas varias
que empiezan por `vendor/`, y este cambio añade `Mockup/vendor/chart.umd.min.js`. Si
el filtro lo alcanzara, el archivo daría 404 en el sitio publicado y los gráficos
desaparecerían solo en producción, mientras en local se verían bien. Un `.nojekyll`
vacío hace que Pages sirva el repositorio tal cual y elimina el riesgo por completo.

### Verificación

Renderizado con navegador headless en dos escenarios:

| | Con internet | Sin internet |
|---|---|---|
| Chart.js | carga | carga |
| Los tres gráficos | renderizan | renderizan |
| Iconos | visibles | ocultos, sin texto crudo |
| Errores de JavaScript | ninguno | ninguno |
| Indicadores | 333 / 31 / 20 / 67 | 333 / 31 / 20 / 67 |

La navegación entre vistas y los datos de ejemplo de la tabla quedan intactos. No se
modificaron la estructura de las vistas, la paleta ni los componentes.

---

## 23 de agosto de 2026 — Maqueta del prototipo

Primera versión del mockup: aplicación de página única con tres vistas (resumen y
triage, ingreso de paciente, detalle individual), en HTML5, CSS y JavaScript con
Chart.js. Ver `Mockup/README.md`.

## 23 de agosto de 2026 — Análisis exploratorio

Notebook `notebooks/01_EDA.ipynb` y las 15 figuras de `figures/01_EDA/`. Caracteriza
la estructura longitudinal y la calidad de los datos, la dinámica temporal del target
y las asociaciones tempranas con el desenlace, y cierra con el contrato de modelado
que fija las 16 variables basales y las reglas anti-fuga.

## 20–21 de agosto de 2026 — Infraestructura

Inicialización del repositorio, configuración de DVC con remoto en S3 e
incorporación del dataset de la cohorte de Kenia.
