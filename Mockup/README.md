# Mockup - Observatorio Canguro (Evaluación de Riesgo Nutricional)

Este directorio contiene el prototipo (mockup) del frontend de la aplicación web de riesgo nutricional. La interfaz fue construida utilizando HTML5, CSS puro y JavaScript (apoyado en Chart.js para visualizaciones).

## Objetivo del Mockup

Proveer una herramienta visual (Tablero de Consulta) que permita al personal médico detectar e interpretar tempranamente la probabilidad de que un niño presente **desnutrición crónica (stunting)** a los 24 meses de edad, basándose en un modelo de aprendizaje automático alimentado por datos recolectados durante su ingreso al programa.

---

## Arquitectura de la Interfaz

La aplicación sigue un formato SPA (Single Page Application) dividida en 3 vistas principales orientadas a un flujo de trabajo clínico eficiente.

### 1. Vista Principal (Triage / Resumen)
Es la pantalla de inicio del sistema. Proporciona una visión general de la cohorte actual de pacientes y ayuda a priorizar los casos más críticos.
- **Tarjetas de KPI (Indicadores Clave):** Muestra el volumen total de pacientes, separados en etapas (Gestación y Neonatales) y destaca el número de pacientes en Riesgo Crítico.
- **Visualización de Datos:** Gráficos que muestran la distribución general de riesgo (Dona) y la cantidad de pacientes por tramos de edad con su distribución de riesgo interna (Barras Apiladas).
- **Lista de Pacientes:** Una tabla de los pacientes activos, ordenados de mayor a menor probabilidad de riesgo para facilitar el triage.

### 2. Vista de Ingreso (Nuevo Paciente)
Un formulario limpio e intuitivo diseñado para capturar la información clave que alimenta el modelo predictivo al momento del ingreso.
- **Perfil Materno:** Captura de variables socioeconómicas y clínicas de la madre (Nivel educativo, Quintil de riqueza, Anemia, Depresión).
- **Datos del Nacimiento:** Captura de las condiciones neonatales de interés para el Método Madre Canguro (Edad gestacional, Peso al nacer, Indicador de Pequeño para Edad Gestacional - SGA).

### 3. Vista de Detalle (Dashboard Individual)
El expediente analítico completo de un solo bebé. Combina información clínica actual con el análisis del modelo predictivo para la toma de decisiones.
- **Medidor de Riesgo:** Un termómetro visual (Gauge) que indica el puntaje de probabilidad de desnutrición crónica.
- **Factores de Riesgo (Explicabilidad):** Un gráfico SHAP de barras horizontales que expone la "caja negra" del modelo, indicando qué variables específicas (por ejemplo, ser SGA o pertenecer al quintil de riqueza 1) están aumentando o disminuyendo el riesgo del bebé.
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
