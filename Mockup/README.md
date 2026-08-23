# Mockup - Observatorio Canguro (Evaluación de Riesgo Nutricional)

Este directorio contiene el prototipo (mockup) del frontend de la aplicación web de riesgo nutricional. La interfaz fue construida utilizando HTML5, CSS puro y JavaScript (apoyado en Chart.js para visualizaciones).

**En vivo:** https://fromerom1.github.io/microproyecto-pds-grupo23/Mockup/

Este documento describe **qué es** cada elemento de la interfaz. Para **por qué existe** cada uno —qué parte de la pregunta de negocio atiende y con qué hallazgo del EDA se sustenta— y para el contrato de la API, ver [`TRAZABILIDAD.md`](TRAZABILIDAD.md).

## Objetivo del Mockup

Proveer una herramienta visual (Tablero de Consulta) que permita al personal médico detectar e interpretar tempranamente la probabilidad de que un niño presente **desnutrición crónica (stunting)** a los 24 meses de edad, basándose en un modelo de aprendizaje automático alimentado por datos recolectados durante su ingreso al programa.

---

## Arquitectura de la Interfaz

La aplicación sigue un formato SPA (Single Page Application) dividida en 3 vistas principales orientadas a un flujo de trabajo clínico eficiente.

### 1. Vista Principal (Triage / Resumen)
Es la pantalla de inicio del sistema. Proporciona una visión general de la cohorte actual de pacientes y ayuda a priorizar los casos más críticos.
- **Tarjetas de KPI (Indicadores Clave):** Muestra el volumen total de pacientes (333, la dimensión del dataset), los prematuros (31) y los de bajo peso al nacer (20) —que son la población objetivo del Método Madre Canguro dentro de la cohorte— y destaca el número de pacientes en Riesgo Alto.
- **Visualización de Datos:** Gráficos que muestran la distribución general de riesgo (Dona) y la cantidad de pacientes por tramo de seguimiento con su distribución de riesgo interna (Barras Apiladas). Los tramos corresponden a las visitas que existen en la cohorte: nacimiento, 0-6, 6-12, 12-18 y 18-24 meses.
- **Lista de Pacientes:** Una tabla de los pacientes activos, ordenados de mayor a menor probabilidad de riesgo para facilitar el triage.

### 2. Vista de Ingreso (Nuevo Paciente)
Un formulario limpio e intuitivo diseñado para capturar la información clave que alimenta el modelo predictivo al momento del ingreso.
Recoge las **16 variables basales** definidas en el contrato de modelado (Paso 9.1 del EDA), todas conocidas en el momento del ingreso al programa.

- **Perfil Materno (10):** nivel educativo, quintil de riqueza, edad materna, estado marital, inseguridad alimentaria del hogar (HFIA), estado nutricional materno (MUAC), paridad, estado VIH, anemia y depresión.
- **Datos del Nacimiento (6):** sexo, edad gestacional, tipo de parto, peso al nacer e indicador de Pequeño para Edad Gestacional (SGA). La prematurez y el bajo peso al nacer se derivan de la edad gestacional y el peso, de modo que 15 controles cubren las 16 variables.

### 3. Vista de Detalle (Dashboard Individual)
El expediente analítico completo de un solo bebé. Combina información clínica actual con el análisis del modelo predictivo para la toma de decisiones.
- **Medidor de Riesgo:** Un termómetro visual (Gauge) que indica el puntaje de probabilidad de desnutrición crónica.
- **Factores de Riesgo (Explicabilidad):** Un gráfico SHAP de barras horizontales que expone la "caja negra" del modelo, indicando qué variables específicas (por ejemplo, ser SGA o pertenecer al quintil de riqueza 1) están aumentando o disminuyendo el riesgo del bebé. Las contribuciones se muestran en ambas direcciones y el color depende del signo: un clínico necesita ver qué protege tanto como qué agrava.
- **Seguimiento Longitudinal (Proyección LAZ):** Un gráfico de tendencia que proyecta la Talla para la Edad (Puntaje Z) a lo largo de 24 meses, con una línea de umbral crítico roja en `-2.0`, permitiendo anticipar el deterioro nutricional de manera temprana.

---

## Especificaciones de Diseño (Design System)
- **Tipografía:** Inter (clara, legible y profesional).
- **Colores Principales:** 
  - Azul Institucional (`#1A73E8`) para elementos primarios y alertas de riesgo bajo.
  - Rojo Intenso (`#D32F2F`) para alertas de riesgo alto.
  - Rojo Suave / Coral (`#E57373`) para riesgo medio.
  - Tonos grises ultra claros y neutros para fondos y separadores.
- **Estilo Visual:** Diseño tipo *dashboard*, plano, limpio, sin sombras pesadas, con bordes sutiles y priorizando la reducción de "ruido visual" para facilitar la interpretación de datos clínicos complejos.

---

## Dependencias

- **Chart.js v4.5.1** — servido desde `vendor/chart.umd.min.js`, no desde un CDN, para
  que los gráficos se vean también sin conexión.
- **Inter** y **Material Symbols Rounded** — desde Google Fonts. Los iconos son
  ligaduras tipográficas: si la fuente no carga, el navegador imprimiría el nombre
  literal del icono. Se ocultan por CSS y `app.js` los revela solo si la fuente está
  realmente disponible, así que sin conexión la interfaz se ve limpia en lugar de rota.

No hay proceso de compilación: se abre `index.html` en cualquier navegador.

## Sobre los datos de la maqueta

Las cifras **descriptivas** —total de la cohorte, prematuros, bajo peso al nacer,
prevalencias por tramo y trayectorias de LAZ— provienen del análisis exploratorio en
`notebooks/01_EDA.ipynb`.

Las cifras de **desempeño del modelo** (probabilidades, bandas de riesgo, valores
SHAP) y los identificadores de los pacientes son **ilustrativos**: el modelo se
entrena en la semana 4 y esos valores se reemplazan por los reales en la Entrega 2.
