# API de riesgo nutricional

Servicio FastAPI sobre `src.predict.ModeloRiesgo`. Recibe las categorías originales; el modelo se encarga del preprocesamiento. Ejecutar desde la raíz del repositorio para que el artefacto pueda importar `src.preprocessing.Winsorizer`.

## Arranque

En PowerShell, desde la raíz del repositorio:

```powershell
py -3.12 -m venv ".venv"
$repo = (Get-Location).Path
subst X: "$repo"
Set-Location "X:\"
& "X:\.venv\Scripts\python.exe" -m pip install -r "requirements.txt"
& "X:\.venv\Scripts\python.exe" -m api
```

`X:` es un alias temporal para evitar el límite de rutas largas de Windows en este repositorio; debe estar libre. Al terminar: `subst X: /D`. En un entorno sin ese límite, basta activar `.venv`, instalar `requirements.txt` y ejecutar `python -m api`. Por defecto escucha en `0.0.0.0:8000`; la documentación interactiva queda en [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Para recarga durante desarrollo: `python -m uvicorn api.main:app --reload`.

`python -m api` admite `API_HOST` y `API_PORT`. `API_MODELS_DIR` cambia la carpeta de modelos y `API_COHORTE_PATH` el CSV de referencia. Si no se configuran, se usan `models/` y `data/processed/model_dataset_escalera.csv` (o `model_dataset.csv` si no existe el primero). El contenedor debe incluir `src/`, `models/` y los datos de referencia.

## Contrato

Todas las rutas de modelo aceptan `horizonte` y `variante` como parámetros de URL. `horizonte` vale `24m` si se omite. La variante predeterminada es `B` si existe; si no, `cv` o la primera disponible. Las variantes y los campos se descubren de cada pareja `.json`/`.joblib` en `models/`.

| Ruta | Entrada | Salida |
| --- | --- | --- |
| `GET /health` | Ninguna | `status` |
| `GET /model/info` | Horizonte y variante opcionales | Datos del modelo, `features`, `features_extra`, `variantes_disponibles` |
| `GET /model/options` | Horizonte y variante opcionales | Categorías conocidas de sus variables |
| `POST /predict` | `registro` y parámetros opcionales | Probabilidad, banda, percentil y contribuciones |
| `POST /predict/batch` | `registros`, `capacidad` opcional entre 0 y 1 | Lista ordenada con ranking; `seguimiento` si se indica capacidad |
| `GET /model/operating-point` | `capacidad` entre 0 y 1 | Capacidad, sensibilidad, VPP y umbral esperados |

Los registros deben incluir todas las `features` de la variante consultada. `/predict` rechaza campos adicionales; `/predict/batch` permite identificadores y conserva los campos recibidos. Las categorías conocidas se validan con `/model/options`. Los campos nuevos sin categorías declaradas se entregan al pipeline sin inventar restricciones. Una combinación inexistente devuelve `404`; un registro o una capacidad inválidos, `422`.

### Ejemplos reales del modelo actual `24m/B`

`GET /health` responde:

```json
{"status":"ok"}
```

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

Desde `X:\`, ejecutar `& "X:\.venv\Scripts\python.exe" -m pytest -q`. Las pruebas comparan la API con `ModeloRiesgo`, incluido el registro de `python -m src.predict --demo`; no dependen de probabilidades fijadas a mano.
