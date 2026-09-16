# API de riesgo nutricional

Servicio FastAPI sobre `src.predict.ModeloRiesgo`. Requiere Python 3.11 o superior. Recibe las categorías originales; el modelo se encarga del preprocesamiento. Ejecutar desde la raíz del repositorio para que el artefacto pueda importar `src.preprocessing.Winsorizer`.

## Arranque

En Windows, desde PowerShell y la raíz del repositorio:

```powershell
py -3.12 -m venv ".venv"
& ".\.venv\Scripts\Activate.ps1"
python -m pip install -r "requirements.txt"
python -m api
```

En macOS o Linux, desde la raíz del repositorio:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m api
```

Por defecto escucha en `0.0.0.0:8000`; la documentación interactiva queda en [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Para recarga durante desarrollo: `python -m uvicorn api.main:app --reload`.

`python -m api` admite `API_HOST` y `API_PORT`. `API_MODELS_DIR` cambia la carpeta de modelos y `API_COHORTE_PATH` el CSV de referencia. Si no se configuran, se usan `models/` y `data/processed/model_dataset_escalera.csv` (o `model_dataset.csv` si no existe el primero). El contenedor debe incluir `src/`, `models/` y los datos de referencia.

## Contrato

Todas las rutas de modelo aceptan `horizonte` y `variante` como parámetros de URL. `horizonte` vale `24m` si se omite. La variante predeterminada es `B` si existe; si no, `cv` o la primera disponible. Las variantes y los campos se descubren de cada pareja `.json`/`.joblib` en `models/`.

| Ruta | Entrada | Salida |
| --- | --- | --- |
| `GET /health` | Ninguna | `status`; 503 con la causa si modelo o cohorte no están disponibles |
| `GET /model/info` | Horizonte y variante opcionales | Datos del modelo y variables; 503 con la causa si falla la configuración |
| `GET /model/options` | Horizonte y variante opcionales | Categorías conocidas de sus variables |
| `POST /predict` | `registro` y parámetros opcionales | Probabilidad, banda, percentil y contribuciones |
| `POST /predict/batch` | `registros`, `capacidad` opcional entre 0 y 1 | Lista ordenada con ranking; `seguimiento` si se indica capacidad |
| `GET /model/operating-point` | `capacidad` dentro de la curva del modelo | Capacidad, sensibilidad, VPP y umbral esperados |

Los registros deben incluir todas las `features` de la variante consultada. `/predict` rechaza campos adicionales; `/predict/batch` permite identificadores y conserva los campos recibidos. `null` se entrega como dato faltante al modelo; los booleanos en campos numéricos devuelven `422`. Los JSON no definen las categorías: `/model/options` usa `ModeloRiesgo.opciones()`, que toma `CATEGORIAS` de `src/preprocessing.py`. Si se agregan categorías, esa fuente debe actualizarse por quien mantenga el preprocesamiento; los campos nuevos sin opciones conocidas se entregan al pipeline sin inventar restricciones. Una combinación inexistente devuelve `404`; un registro o una capacidad inválidos, `422`. Para `/model/operating-point`, el rango válido sale de `curva_capacidad` en el JSON del modelo y no se recorta silenciosamente.

### Ejemplos reales del modelo actual `24m/B`

`GET /health` responde:

```json
{"status":"ok"}
```

Si `API_COHORTE_PATH` apunta a un archivo inexistente, responde `503`:

```json
{"status":"error","detail":"La cohorte de referencia no está disponible."}
```

En ese caso, `GET /model/info` también responde `503` con `{"detail":"La cohorte de referencia no está disponible."}`. Los detalles técnicos se registran en el log, no en la respuesta.

`GET /model/info` responde, entre otros campos:

```json
{"horizonte":"24m","variante":"B","n_features":19,"roc_auc_cv":0.7216018306636157,"features_extra":["zlen_nac","zwei_nac","zhc_nac"],"variantes_disponibles":["B","cv"]}
```

`GET /model/options` responde, entre otras categorías:

```json
{"b1_sex":["female","male"],"sga":["no","yes"]}
```

Petición a `POST /predict`:

```json
{
  "registro": {
    "enrol_hiv_status_cat": "Negative",
    "momage_cat": "25 and less",
    "educ_cat_n": "Primary or below",
    "marital_cat": "Married",
    "wealth_quintile": "lowest quintile",
    "depression": "no depression",
    "mom_muac_cat": "normal",
    "b1_sex": "female",
    "preterm": "yes",
    "hfia_enr": "severe",
    "parity": "multi",
    "enrol_anemia": "no",
    "caesarean": "no",
    "sga": "yes",
    "lbw": "yes",
    "gestage_final": 36.0,
    "zlen_nac": -1.4,
    "zwei_nac": -1.1,
    "zhc_nac": -0.6
  }
}
```

Extracto de la respuesta real (se muestran dos de 19 contribuciones):

```json
{
  "horizonte": "24m",
  "probabilidad": 0.6795943714877184,
  "banda": "alto",
  "percentil": 85.44891640866874,
  "unidad_contribucion": "log-odds",
  "contribuciones": [
    {"variable":"zlen_nac","etiqueta":"LAZ al nacer","valor":-1.4,"contribucion":0.6135545951848757},
    {"variable":"gestage_final","etiqueta":"Edad gestacional (sem)","valor":36.0,"contribucion":-0.22855677201861166}
  ]
}
```

Petición a `POST /predict/batch?capacidad=0.2` con un registro (mismos campos del ejemplo anterior, más `newid`):

```json
{"registros":[{"newid":"A","enrol_hiv_status_cat":"Negative","momage_cat":"25 and less","educ_cat_n":"Primary or below","marital_cat":"Married","wealth_quintile":"lowest quintile","depression":"no depression","mom_muac_cat":"normal","b1_sex":"female","preterm":"yes","hfia_enr":"severe","parity":"multi","enrol_anemia":"no","caesarean":"no","sga":"yes","lbw":"yes","gestage_final":36.0,"zlen_nac":-1.4,"zwei_nac":-1.1,"zhc_nac":-0.6}]}
```

Extracto de la respuesta real (también conserva los campos recibidos):

```json
[{"newid":"A","probabilidad":0.6795943714877184,"banda":"alto","percentil":85.44891640866874,"ranking":1,"seguimiento":true}]
```

`GET /model/operating-point?capacidad=0.2` responde:

```json
{"capacidad":0.2,"sensibilidad":0.3368421052631579,"vpp":0.49230769230769234,"umbral":0.6173863300707529}
```

La capacidad es la proporción que puede recibir seguimiento. La sensibilidad estima qué parte de los casos se alcanzaría y el VPP qué parte de los priorizados sería un caso. Las contribuciones usan la unidad indicada por el modelo; no siempre son probabilidades. Esta herramienta prioriza seguimiento, no diagnostica.

## Pruebas

Desde la raíz del repositorio, con `.venv` activado:

```text
python -m pytest -q
```

Las pruebas comparan la API con `ModeloRiesgo`, incluido el registro de `python -m src.predict --demo`; no dependen de probabilidades fijadas a mano.

## Docker

Cuando el equipo tenga una imagen con dependencias, `src/`, `models/`, datos y la raíz del repositorio como directorio de trabajo, reemplazar `NOMBRE_IMAGEN` por su nombre:

```text
docker run --rm -p 8000:8000 --entrypoint python NOMBRE_IMAGEN -m api
docker run --rm --entrypoint python NOMBRE_IMAGEN -m pytest -q
```

Esos comandos ejecutan `python -m api` y `python -m pytest -q` dentro del contenedor. El Dockerfile lo prepara el equipo de despliegue.

## Nota para Windows

Si la ruta del repositorio supera el límite de Windows para bibliotecas instaladas, se puede crear un alias temporal antes de crear `.venv`:

```powershell
$repo = (Get-Location).Path
subst X: "$repo"
Set-Location "X:\"
```

Repetir desde `X:\` los comandos de arranque y pruebas anteriores. `X:` debe estar libre; al terminar, salir de esa unidad y ejecutar `subst X: /D`.
