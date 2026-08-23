# Registro de cambios

Cambios relevantes del proyecto, con la razón detrás de cada uno. El orden es del
más reciente al más antiguo.

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
