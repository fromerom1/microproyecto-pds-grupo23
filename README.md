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
.dvc/                    configuración de DVC (remoto S3)
data/                    dataset versionado con DVC (el CSV no está en Git)
docs/                    documentos de las entregas
figures/01_EDA/          figuras generadas por el análisis exploratorio
notebooks/01_EDA.ipynb   análisis exploratorio
Mockup/                  maqueta del prototipo (SPA) — ver Mockup/README.md
requirements.txt         dependencias de Python
CHANGELOG.md             registro de cambios del proyecto
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
- **Estructura `src/`.** Todo el código vive en el notebook. La Entrega 3 pide
  empaquetar modelos y servirlos por API, y eso no se importa desde un `.ipynb`.
- **Versiones en `requirements.txt`.** Sin fijar. La imagen de Docker que se
  construya en septiembre no será necesariamente la misma que corre hoy.
- **Salida del notebook sin versionar.** `data/processed/model_dataset.csv` se genera
  en el Paso 9.2 pero no está en Git ni en DVC; debería ser salida de una etapa del
  pipeline y no un efecto secundario del notebook.
- **Tamaño de muestra.** 333 bebés, con 95 casos positivos a 24 meses y 47 a 12
  meses. Una partición train/test única no discrimina: hace falta validación cruzada
  estratificada repetida con intervalos por bootstrap, reportar AUC-PR además de
  AUC-ROC, y fijar un punto de operación por capacidad de seguimiento.

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
