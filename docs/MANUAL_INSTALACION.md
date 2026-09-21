# Manual de instalación — tablero y API de riesgo nutricional

Micro-proyecto PDS · MAIA Uniandes 2026-2 · Grupo 23 · Entrega 3

Este manual instala el prototipo completo: el **tablero** y la **API** que lo alimenta,
cada uno en su propio contenedor Docker. Sirve tanto para levantarlo en un equipo
personal como para desplegarlo en una máquina de AWS EC2.

---

## 1. Qué se instala

```
                  ┌──────────────────────────┐
  navegador  ───►  │  tablero   :8501         │   Streamlit
                  │  (Streamlit)             │
                  └───────────┬──────────────┘
                              │ HTTP  (red interna de Docker)
                  ┌───────────▼──────────────┐
                  │  api       :8000         │   FastAPI + modelo empaquetado
                  │  (FastAPI)               │
                  └───────────┬──────────────┘
                              │ lectura
                  ┌───────────▼──────────────┐
                  │  data/processed/  (volumen de solo lectura)
                  └──────────────────────────┘
```

| Componente | Qué es | Puerto |
|---|---|---|
| `api` | Servicio FastAPI que sirve las predicciones del modelo. Lleva instalado el paquete `modelo-stunting`, construido desde este mismo repositorio | 8000 |
| `tablero` | Interfaz Streamlit con las tres vistas del prototipo. Obtiene **todas** las predicciones llamando a la API | 8501 |
| cohorte de referencia | `data/processed/model_dataset_escalera.csv`. **No viaja dentro de las imágenes**: se monta como volumen de solo lectura y está versionada en S3 con DVC | — |

La API necesita la cohorte para calcular percentiles y contribuciones; el tablero la
necesita para las cifras de la vista *Cohorte* y para armar el lote que envía a
`POST /predict/batch`. Por eso el volumen se monta en los dos servicios.

---

## 2. Requisitos

| | Mínimo | Recomendado |
|---|---|---|
| Docker Engine | 24 | 29 o superior |
| Docker Compose | v2 (plugin `docker compose`) | v2.30 o superior |
| RAM | 2 GiB | 4 GiB — con menos, el `pip install` del build puede morir por memoria (§6) |
| Disco libre | 6 GiB | 20 GiB |
| Python (solo para `dvc pull`) | 3.11 | 3.12 |

No hace falta instalar Python, scikit-learn ni Streamlit en la máquina anfitriona: todo
eso vive dentro de las imágenes. Python local se usa únicamente para traer los datos
con DVC, y el paso 3.2 explica cómo evitarlo si no lo tiene.

Para el despliegue en EC2 hacen falta además credenciales de AWS Academy Learner Lab
con acceso al bucket `s3://maia-microproyecto-g23-dvc-20262` (región `us-east-2`).

---

## 3. Instalación local

### 3.1 Clonar el repositorio

En **Windows con PowerShell**, fuera de OneDrive para evitar que la sincronización
bloquee archivos durante el build:

```powershell
mkdir $HOME\repos -Force
cd $HOME\repos
git clone https://github.com/fromerom1/microproyecto-pds-grupo23.git
cd microproyecto-pds-grupo23
```

En **Linux o macOS**:

```bash
git clone https://github.com/fromerom1/microproyecto-pds-grupo23.git
cd microproyecto-pds-grupo23
```

### 3.2 Traer la cohorte de referencia

Los datos están versionados con DVC contra un bucket de S3, así que `dvc pull`
necesita credenciales de AWS con acceso a ese bucket. Si no las tiene configuradas,
falla con `Unable to locate credentials`: créelas primero en `~/.aws/credentials`
(en Windows, `%USERPROFILE%\.aws\credentials`) con el formato del paso §4.4, o use
una de las alternativas del final de esta sección.

Con Python y credenciales disponibles:

```powershell
# Windows PowerShell — ajuste el número de versión al que tenga (py -0 los lista)
py -3.12 -m venv ".venv"
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install "dvc[s3]"
& ".\.venv\Scripts\python.exe" -m dvc pull
```

```bash
# Linux / macOS
python3 -m venv .venv
. .venv/bin/activate
pip install "dvc[s3]"
dvc pull
```

Debe quedar el archivo `data/processed/model_dataset_escalera.csv`. Compruébelo:

```powershell
Get-ChildItem data\processed        # Windows
```
```bash
ls -l data/processed                # Linux / macOS
```

**Si no tiene Python instalado**, tráigalos con un contenedor desechable —
`$HOME/.aws` se monta para que DVC use sus credenciales:

```bash
docker run --rm -v "$PWD":/repo -v "$HOME/.aws":/root/.aws:ro -w /repo \
  python:3.12-slim sh -c 'pip install -q "dvc[s3]" && dvc pull'
```

**Si el bucket no está disponible** (credenciales del Learner Lab caducadas, por
ejemplo), el dataset crudo es público y se puede reconstruir el procesado:

1. Descargue la cohorte de Zenodo (registro 15867528) a `data/plosmed_data_newid.csv`.
2. Reconstruya el CSV procesado:
   ```bash
   docker run --rm -v "$PWD":/repo -w /repo python:3.12-slim \
     sh -c 'pip install -q -r requirements.txt && python -m src.features'
   ```

Y si otra máquina del equipo ya tiene el CSV, cópielo directamente — pesa 68 KB:

```bash
scp -i C:\aws\labsuser.pem data/processed/model_dataset_escalera.csv \
    ubuntu@<IP_PUBLICA>:~/microproyecto-pds-grupo23/data/processed/
```

### 3.3 Construir y levantar

Desde la raíz del repositorio:

```bash
docker compose up -d --build
```

La primera vez tarda entre 4 y 10 minutos: construye el wheel del modelo e instala
numpy, pandas, scikit-learn y Streamlit en las dos imágenes. Las siguientes veces
reutiliza las capas de `pip` y baja a menos de un minuto.

### 3.4 Verificar

```bash
docker compose ps
```

El servicio `api` debe aparecer como `healthy` y `tablero` como `running`. Si `api`
queda en `starting` más de un minuto o pasa a `unhealthy`, vaya a §6.

```bash
curl http://localhost:8000/health          # {"status":"ok"}
curl http://localhost:8000/model/variants  # {"12m":["B","cv"],"24m":["B","cv"]}
```

En PowerShell, `curl` es un alias de `Invoke-WebRequest`; use:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Y abra en el navegador:

- **Tablero:** <http://localhost:8501>
- **Documentación interactiva de la API:** <http://localhost:8000/docs>

El tablero debe mostrar las tres pestañas —Cohorte, Riesgo individual y
Priorización— con datos. Si muestra *"No se pudo conectar con la API de
predicción"*, vaya a §6.

---

## 4. Despliegue en AWS EC2

Los pasos asumen AWS Academy Learner Lab, que es el entorno del curso.

### 4.1 Lanzar la instancia

Consola AWS → **EC2** → **Launch instances**:

| Campo | Valor |
|---|---|
| Name | `g23-entrega3` |
| AMI | **Ubuntu Server 24.04 LTS** |
| Tipo | **t3.medium** (2 vCPU, 4 GiB). En `t3.small` el build necesita swap (§6) |
| Key pair | `vockey` |
| Almacenamiento | **20 GiB** gp3 — los 8 GiB por defecto no alcanzan para las dos imágenes |

En **Network settings → Edit**, el grupo de seguridad necesita tres reglas de entrada:

| Tipo | Puerto | Origen | Para qué |
|---|---|---|---|
| SSH | 22 | Anywhere | administración |
| Custom TCP | 8000 | Anywhere | API |
| Custom TCP | 8501 | Anywhere | tablero |

Anote la **IPv4 pública** de la instancia.

> 📎 Pantallazo: la instancia en la consola, con su IP pública visible.
> → `docs/soportes/entrega3/01_ec2_instancia.png`

### 4.2 Conectarse

```powershell
ssh -i C:\aws\labsuser.pem ubuntu@<IP_PUBLICA>
```

### 4.3 Instalar Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
newgrp docker                      # aplica el grupo sin cerrar la sesión
docker version && docker compose version
```

El script oficial instala el motor y el plugin `docker compose`, y deja el servicio
habilitado en el arranque. Eso último importa: cuando la instancia se vuelva a
encender, los contenedores se levantan solos.

### 4.4 Clonar el repositorio y traer los datos

```bash
git clone https://github.com/fromerom1/microproyecto-pds-grupo23.git
cd microproyecto-pds-grupo23
```

Copie las credenciales del panel **AWS Details → AWS CLI** del Learner Lab:

```bash
mkdir -p ~/.aws && nano ~/.aws/credentials
```

```ini
[default]
aws_access_key_id=...
aws_secret_access_key=...
aws_session_token=...
region=us-east-2
```

```bash
sudo apt-get update && sudo apt-get install -y python3-venv
python3 -m venv ~/.venv-dvc
~/.venv-dvc/bin/pip install -q "dvc[s3]"
~/.venv-dvc/bin/dvc pull
ls -l data/processed/model_dataset_escalera.csv
```

> Ubuntu 24.04 no permite `pip install` sobre el Python del sistema (PEP 668); de ahí
> el entorno virtual. DVC solo se usa para traer los datos, no para servirlos.

### 4.5 Construir y levantar

```bash
docker compose up -d --build
docker compose ps
docker images
```

> 📎 Pantallazos, con el prompt `ubuntu@ip-172-31-…` visible:
> `docker images` → `02_docker_images.png`
> `docker compose ps` con `api` en `healthy` → `03_docker_compose_ps.png`

### 4.6 Verificar desde afuera

```bash
curl -s http://localhost:8000/health
```

Y en el navegador, contra la **IP pública**:

- Tablero: `http://<IP_PUBLICA>:8501`
- API: `http://<IP_PUBLICA>:8000/docs`

> 📎 Pantallazos con la IP pública visible en la barra de direcciones:
> `/docs` de la API → `04_api_docs.png`
> tablero, vista Cohorte → `05_tablero_cohorte.png`
> tablero, vista Riesgo individual con una predicción calculada → `06_tablero_riesgo.png`
> tablero, vista Priorización → `07_tablero_priorizacion.png`

### 4.7 Detener sin terminar

El enunciado de la entrega lo pide de forma explícita: *«mantenga las máquinas y/o
servicios empleados para el despliegue detenidos, no los termine»*.

```bash
docker compose stop
```

Y en la consola: **EC2 → Instance state → Stop instance**. **No use *Terminate***:
eso borra la instancia y el volumen, y con él los datos que trajo `dvc pull`.

Para volver a levantarlo: **Start instance**, y como el motor de Docker arranca con la
máquina y los servicios están declarados `restart: unless-stopped`, los contenedores
vuelven por su cuenta. Solo cambia la IP pública, salvo que se le haya asignado una
IP elástica.

> 📎 Pantallazo: la instancia en estado `Stopped`. → `08_ec2_detenida.png`

---

## 5. Operación

```bash
docker compose logs -f api          # registros de la API
docker compose logs -f tablero      # registros del tablero
docker compose restart tablero      # reiniciar un servicio
docker compose stop                 # detener, conservando los contenedores
docker compose down                 # detener y eliminar los contenedores
docker compose up -d --build        # reconstruir tras un git pull
```

Actualizar a la última versión del código:

```bash
git pull
docker compose up -d --build
```

El wheel del modelo se reconstruye dentro del build, así que un cambio en `src/` o en
`models/` queda recogido sin pasos adicionales.

Para obtener el wheel por fuera de Docker —por ejemplo, para el soporte de la entrega—:

```bash
pip install build
python -m build --wheel          # deja dist/modelo_stunting-0.1.0-py3-none-any.whl
```

> 📎 Pantallazo: la construcción del wheel y `dist/` con el archivo.
> → `09_wheel_modelo.png`

Las pruebas de la API también se pueden correr dentro del contenedor:

```bash
docker compose run --rm --entrypoint python api -m pytest -q
```

---

## 6. Solución de problemas

| Síntoma | Causa | Qué hacer |
|---|---|---|
| `api` queda `unhealthy`; los registros dicen *«Falta data/processed/model_dataset_escalera.csv»* | No se trajo la cohorte, o el volumen no está montado | Repita §3.2 y confirme que el archivo existe en el anfitrión antes de `docker compose up` |
| El tablero muestra *«No se pudo conectar con la API de predicción»* | La API no está sana; el tablero no arranca hasta que lo esté | `docker compose ps` y `docker compose logs api` |
| El build muere con `Killed` o `MemoryError` durante `pip install` | Poca RAM (típico en `t3.micro` y `t3.small`) | Agregue swap y reintente: `sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile` |
| `docker: permission denied … /var/run/docker.sock` | El usuario no está en el grupo `docker` | `sudo usermod -aG docker $USER` y `newgrp docker` |
| El navegador no carga nada en `:8501` o `:8000` | Faltan las reglas de entrada del grupo de seguridad | §4.1 |
| `dvc pull` falla con `ExpiredToken` o `InvalidAccessKeyId` | Credenciales del Learner Lab caducadas (duran unas horas) | Vuelva a copiarlas del panel **AWS Details** y repita |
| `no space left on device` durante el build | Disco de la instancia lleno | `docker system prune -af` y, si persiste, amplíe el volumen a 20 GiB |
| En Windows, el build falla leyendo archivos | OneDrive sincronizando el repositorio | Clone fuera de OneDrive (§3.1) o pause la sincronización |
| `docker compose` no se reconoce en PowerShell | Docker Desktop no está corriendo | Ábralo y espere a que el motor quede en verde |
| El puerto 8000 u 8501 ya está en uso | Otro proceso los ocupa | Cambie la publicación en `docker-compose.yml`, p. ej. `"8600:8000"` |

---

## 7. Desinstalación

```bash
docker compose down --rmi local -v     # contenedores, imágenes locales y volúmenes
```

Los datos de `data/processed/` quedan en el disco porque el volumen es un directorio
del repositorio, no un volumen administrado por Docker. Bórrelos a mano si hace falta.

---

## 8. Notas de diseño

Tres decisiones que conviene conocer antes de modificar las imágenes:

1. **La cohorte no está en la imagen.** Se monta de solo lectura. Hoy los datos son de
   una cohorte pública, pero cuando entren datos del Observatorio Canguro —historias
   clínicas anonimizadas bajo acuerdo de confidencialidad— publicar una imagen con los
   datos adentro sería un problema. La decisión se toma ahora para no tener que
   deshacerla después.
2. **Las imágenes instalan el paquete `modelo-stunting`, no copian `src/`.** El wheel se
   construye en la primera etapa del `Dockerfile` desde este repositorio y se instala
   en el directorio de trabajo con `pip install --target /app`. Queda en `/app/src`,
   que es la ruta donde `src/predict.py` espera encontrar `models/` y `data/`.
3. **El tablero lleva `scikit-learn` y `models/` a bordo** solo para el panel de
   importancia de variables, que necesita los coeficientes del pipeline. Todo lo demás
   lo obtiene de la API. Cuando la API exponga `GET /model/importances`, esa imagen
   puede adelgazar: basta quitar `scikit-learn`, `joblib` y el `COPY models/`.

Las versiones de `numpy`, `pandas`, `scikit-learn` y `joblib` están fijadas en los tres
archivos de dependencias porque participan en el pickle del modelo: un `.joblib` guardado
con una versión de scikit-learn no tiene garantía de abrir con otra.
