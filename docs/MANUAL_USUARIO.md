# Manual de usuario — Tablero de riesgo nutricional temprano

Micro-proyecto PDS · MAIA Uniandes 2026-2 · Grupo 23 · Entrega 3

---

## 1. Qué hace el tablero, y qué no

El tablero apoya una decisión concreta: **a qué bebés del programa ver primero**, cuando
los cupos de seguimiento estrecho son menos que los bebés en riesgo. Estima, a partir del
perfil materno al enrolamiento y de los datos del nacimiento, la probabilidad de que un
bebé desarrolle desnutrición crónica (talla baja para la edad) en sus primeros 12 o 24
meses, y ordena la cohorte por esa estimación.

**No es una herramienta de diagnóstico.** No reemplaza la valoración clínica ni la
medición antropométrica. Un riesgo bajo no descarta el problema y un riesgo alto no lo
confirma: lo que el tablero ofrece es un **orden de prioridad** mejor que el azar, no un
veredicto sobre un niño.

El modelo se entrenó con una cohorte de Kenia en la que el 9,3 % de los niños es prematuro.
En un Programa Madre Canguro esa proporción es del 100 %, de modo que su uso en población
canguro exige validación previa con datos propios.

---

## 2. Cómo abrirlo

| Dónde | Dirección |
|---|---|
| Desplegado en la nube | `http://<IP-pública-de-la-instancia>:8501` |
| En su propio equipo | `http://localhost:8501`, tras levantar el entorno |

Para levantarlo usted mismo, siga `docs/MANUAL_INSTALACION.md`. No hace falta instalar
Python ni nada más: basta un navegador.

Si al abrir aparece *«No se pudo conectar con la API de predicción»*, el servicio de
inferencia no está disponible. No es un problema del tablero; avise a quien administre el
despliegue.

---

## 3. La barra lateral: los tres controles que mandan

Todo lo que ve a la derecha depende de estos tres ajustes. Cámbielos y las tres vistas se
recalculan solas.

**Horizonte de predicción.** *12 meses* o *24 meses*. Es el momento en que se evalúa el
desenlace. A 12 meses el modelo acierta más (ROC-AUC 0,79 contra 0,72), pero a 24 meses la
condición es más frecuente y más difícil de revertir. Si no tiene una razón para lo
contrario, trabaje con 24 meses.

**Variante del modelo.**

| Variante | Qué usa | Cuándo usarla |
|---|---|---|
| **B** | 19 variables: las 16 basales más los tres puntajes Z al nacer | La predeterminada. Úsela siempre que tenga la antropometría del parto |
| **cv** | Solo las 16 variables basales | Únicamente si no dispone de los puntajes Z al nacer |

La diferencia es grande: a 24 meses, B alcanza un ROC-AUC de 0,72 contra 0,58 de `cv`. La
talla, el peso y el perímetro cefálico al nacer se miden en el parto y ya se conocen al
ingreso, así que rara vez hay motivo para usar `cv`.

**Capacidad de seguimiento estrecho.** El porcentaje de la cohorte que su programa
realmente puede vigilar de cerca, entre 5 % y 50 %. **Este es el control más importante y el
más malinterpretado:** no es un ajuste de precisión del modelo, es la declaración de cuántos
cupos tiene. Define el umbral de probabilidad a partir del cual un bebé queda marcado, y con
él la sensibilidad y el valor predictivo positivo que puede esperar. Póngalo en el número
real de su programa, no en el que le dé mejores cifras.

Debajo de los controles, una ficha resume el modelo en uso: peldaño, número de variables,
familia, ROC-AUC de la validación cruzada con su intervalo de confianza, tamaño de
entrenamiento y versión de scikit-learn. Sirve para saber exactamente qué está viendo.

---

## 4. Vista «Cohorte» — dimensionar el problema

Responde a *¿qué tan grande es esto y en quién se concentra?*

**Los cuatro indicadores de arriba.** Bebés en la cohorte, prematuros, bajo peso al nacer y
cuántos quedan marcados con la capacidad que usted fijó. Los tres primeros salen del
conjunto de datos; el cuarto, del modelo.

**Distribución del riesgo estimado.** Un histograma de las probabilidades de toda la
cohorte, con dos líneas verticales: el umbral de riesgo alto (rojo) y el de riesgo medio
(ámbar). Le dice si los bebés se agrupan en un extremo o se reparten, que es lo que
determina si la priorización discrimina de verdad.

**Conteo por banda.** Cuántos bebés caen en riesgo alto, medio y bajo, y qué porcentaje
representan.

**Variables que más pesan en el modelo.** El aporte agregado de cada variable. Léalo como
*qué mira el modelo*, no como una relación causal: que el quintil de riqueza pese mucho no
significa que el dinero cause por sí solo la desnutrición.

**Contexto del análisis exploratorio.** Dos figuras al pie: cómo crece la prevalencia visita
a visita y cómo se separan las trayectorias de talla. Justifican por qué la ventana de
intervención es el primer año.

---

## 5. Vista «Riesgo individual» — un bebé concreto

Responde a *¿cuál es el riesgo de este bebé, y por qué?*

### Llenar el formulario

Son 19 campos repartidos en dos columnas. Todos son conocidos **en el momento del ingreso al
programa**; ninguno exige esperar a un control posterior.

- **Perfil materno al enrolamiento (10):** edad, educación, estado marital, quintil de
  riqueza, inseguridad alimentaria del hogar, estado nutricional materno (MUAC), anemia,
  depresión, paridad y estado VIH.
- **Condiciones del nacimiento (6):** sexo, edad gestacional en semanas, cesárea,
  prematuridad, pequeño para la edad gestacional y bajo peso al nacer.
- **Antropometría al nacer (3), solo en la variante B:** LAZ (talla para la edad), WAZ (peso
  para la edad) y HCZ (perímetro cefálico para la edad), en puntajes Z de la OMS. Un valor
  por debajo de −2 indica retraso; 0 es el promedio de referencia.

Los desplegables ya traen las categorías válidas. Cuando termine, pulse **Calcular riesgo**.

### Leer el resultado

**El número grande** es la probabilidad estimada, entre 0 y 1. **La etiqueta de color** es la
banda —alto, medio o bajo— según los umbrales del modelo. **El percentil** sitúa al bebé
dentro de la cohorte: percentil 85 significa que su riesgo es mayor que el del 85 % de los
bebés.

Debajo, una línea le dice si ese bebé **entra o no en el grupo de seguimiento** con la
capacidad que usted fijó, y con qué umbral.

**«Qué empuja esta estimación»** desglosa la contribución de cada variable. Las barras a la
derecha (rojas) aumentan el riesgo; las de la izquierda (verdes) lo reducen. La longitud es
la magnitud del aporte. Esta es la parte que hace la estimación discutible con criterio
clínico: si el modelo marca a un bebé por razones que usted sabe que no aplican, esa
información vale.

---

## 6. Vista «Priorización» — a quién ver primero

Responde a *¿a quién llamo, y qué puedo esperar de esa lista?*

**Las cuatro métricas de arriba** describen el punto de operación con la capacidad elegida:

| Métrica | Qué significa |
|---|---|
| Bebés marcados | Cuántos entran en la lista |
| Sensibilidad esperada | De los bebés que desarrollarán la condición, qué fracción está en la lista |
| Valor predictivo positivo | De los bebés de la lista, qué fracción desarrollará la condición |
| Prevalencia de base | Qué fracción la desarrollaría eligiendo al azar |

La comparación que importa es **VPP contra prevalencia de base**: es cuánto mejora la
priorización frente a no usar el modelo. A 24 meses con capacidad del 20 %, el VPP es 49 %
contra una prevalencia del 29 %.

Estas cifras vienen de la validación cruzada, no del ajuste sobre la cohorte, así que son
una expectativa honesta y no una promesa optimista.

**La tabla** lista a los bebés ordenados por riesgo, con su identificador, probabilidad,
banda y las variables de contexto (prematuro, pequeño para la edad gestacional, bajo peso,
quintil). El interruptor *Mostrar solo los marcados* alterna entre la lista de seguimiento y
la cohorte completa. **Descargar lista priorizada (CSV)** la exporta para trabajarla fuera.

**La curva de capacidad**, a la derecha, es la herramienta de negociación: muestra cómo
suben y bajan sensibilidad y VPP al ampliar o reducir los cupos. Ampliar la lista alcanza a
más casos pero hace que una fracción mayor de las visitas sean a bebés que no lo
necesitaban. No existe un punto óptimo estadístico: lo fija la capacidad real del programa.

---

## 7. Cómo usar esto bien

- **Fije la capacidad primero**, con el número real de cupos, y lea los resultados después.
  Al revés es elegir el dato que mejor se ve.
- **La lista es un punto de partida, no una orden.** Un bebé que no aparece puede necesitar
  seguimiento por razones que el modelo no ve.
- **Revise las contribuciones** antes de actuar sobre un caso individual.
- **Las probabilidades no están calibradas finamente**: son buenas para ordenar, menos para
  leerse como «este bebé tiene un 68 % de probabilidad». Confíe en el orden y en la banda más
  que en el decimal.
- **Vuelva a evaluar cuando lleguen datos nuevos.** Esta versión usa información del ingreso;
  las mediciones de los controles tempranos mejoran la predicción y se incorporarán después.

---

## 8. Problemas frecuentes

| Qué ve | Qué pasa |
|---|---|
| *«No se pudo conectar con la API de predicción»* | El servicio de inferencia está caído. Avise a quien administre el despliegue |
| *«La API respondió pero no reporta modelos disponibles»* | Los artefactos del modelo no están donde el servicio los busca |
| El panel de variables dice *«Panel no disponible»* | El tablero corre contra una API remota sin artefactos locales. El resto funciona normal |
| La página tarda al mover un control | Cada cambio recalcula la cohorte completa contra la API. Son unos segundos |
| La dirección dejó de responder | Si está desplegado en una instancia de laboratorio, la IP pública cambia cada vez que la máquina se detiene y se vuelve a encender |
