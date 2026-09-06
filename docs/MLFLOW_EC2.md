# MLflow en AWS EC2 — guía de la Entrega 2

La rúbrica exige **"pantallazos de experimentos registrados en MLflow en una máquina
de AWS EC2 (debe ser visible el usuario e IP de la máquina en EC2, y la IP en
MLflow)"** y pide dejar la máquina **detenida, no terminada**. Esta guía cubre eso y
solo eso.

El flujo es el mismo del Taller 4: una EC2 con Ubuntu 24.04, MLflow como servidor,
y los scripts del repositorio registrando runs contra él.

> **El equipo ya tiene una máquina.** El barrido de `train_stunting.py` (experimento
> `stunting-baseline-multi`, 6 runs, `figures/03_mlflow/`) se registró el 3 de
> septiembre en una EC2 con usuario `ubuntu`. Lo primero es registrar ahí mismo
> `evaluate_cv` y la escalera (sección 4, preferiblemente por SSH en la misma máquina) para que los tres experimentos
> queden en un solo servidor. Ojo con dos cosas: la IP pública cambia si la máquina se
> detuvo, y los pantallazos de `figures/03_mlflow/` están recortados **sin la barra de
> direcciones**, así que no cumplen por sí solos el requisito de "IP visible en
> MLflow" — hay que retomarlos con la URL a la vista (sección 5).

---

## 1. Lanzar la instancia

En la consola de EC2 (AWS Academy Learner Lab funciona):

| Campo | Valor |
|---|---|
| AMI | Ubuntu Server 24.04 LTS |
| Tipo | `t3.medium` (la CV 5×10 con 7 configuraciones tarda ~4 min; en `t2.micro` tarda mucho más) |
| Par de claves | el del laboratorio (`vockey` en Learner Lab) |
| Grupo de seguridad | entrada TCP **22** (SSH) y TCP **8050** (MLflow) |

Para el puerto 8050, "Mi IP" es lo correcto si solo tú vas a entrar; si el equipo va
a ver la interfaz desde varias redes, `0.0.0.0/0` durante la sesión de trabajo y se
cierra al terminar. MLflow 2.22 no tiene autenticación: no lo dejes abierto a
internet más tiempo del necesario.

Anota la **IP pública** de la instancia. En Learner Lab cambia cada vez que la
máquina se detiene y arranca, así que los pantallazos hay que tomarlos en una misma
sesión.

## 2. Preparar la máquina

```bash
ssh -i vockey.pem ubuntu@<IP-PUBLICA>

sudo apt update && sudo apt install -y python3-pip python3-venv git
git clone https://github.com/fromerom1/microproyecto-pds-grupo23.git
cd microproyecto-pds-grupo23
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

La instalación tarda unos minutos (scikit-learn, mlflow, shap, streamlit).

## 3. Levantar el servidor de MLflow

Desde la raíz del repositorio, en una terminal que se queda abierta:

```bash
source .venv/bin/activate
mlflow server \
  --host 0.0.0.0 --port 8050 \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts
```

`--artifacts-destination` (y no `--default-artifact-root`) hace que los artefactos
viajen **a través del servidor**. Con `--default-artifact-root ./mlartifacts` los runs
lanzados desde otra máquina escribirían sus figuras en el disco de esa otra máquina y
la interfaz en la EC2 los mostraría vacíos. Si el servidor ya está corriendo con
`--default-artifact-root`, la opción A de la sección 4 (correr en la propia EC2)
sigue funcionando; la B no subiría artefactos.

> **Ojo:** `--allowed-hosts` es una opción de MLflow 3.x y **no existe en 2.22.5**
> (la versión fijada en `requirements.txt`). Con 2.22.5 basta `--host 0.0.0.0`. Si el
> servidor que ya corre en la EC2 es 3.x y rechaza la conexión por la IP pública
> (error de *host* inválido), ahí sí: `--allowed-hosts "<IP-PUBLICA>:8050"`. Los
> scripts del repo (cliente 2.22.5) registran sin problema contra un servidor 3.x.

Abre en el navegador `http://<IP-PUBLICA>:8050`. Debe verse la interfaz vacía.

Para que el servidor sobreviva a que cierres la sesión SSH, usa `tmux` o `nohup`:

```bash
nohup mlflow server --host 0.0.0.0 --port 8050 \
  --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts \
  > mlflow.log 2>&1 &
```

## 4. Registrar los experimentos

**Opción A — en la propia máquina** (la más simple; es lo que hizo el barrido de
`stunting-baseline-multi` y garantiza que los artefactos queden en la EC2).
Segunda terminal SSH, desde la raíz del repo, con el clon actualizado (`git pull`):

```bash
source .venv/bin/activate
export MLFLOW_TRACKING_URI=http://127.0.0.1:8050

# Evaluacion robusta: CV 5x10, 7 configuraciones, un run padre + 7 hijos
python -m src.evaluate_cv --target 24m
python -m src.evaluate_cv --target 12m

# Experimento escalera, un run padre + 16 hijos. El dataset de la escalera ya esta
# versionado en data/processed/; `features` solo hace falta para regenerarlo (dvc pull)
python -m src.experimento_escalera

# Barrido con split unico (script de Yeisson), un run por corrida: las seis
# configuraciones del experimento stunting-baseline-multi. OJO: --data es
# obligatorio desde la raiz, el default del script es relativo a src/.
D=data/processed/model_dataset.csv
python -m src.train_stunting --data $D --model rf --n-estimators 200 --max-depth 6 --max-features 4
python -m src.train_stunting --data $D --model rf --n-estimators 100 --max-depth 3 --max-features 2
python -m src.train_stunting --data $D --model rf --n-estimators 500 --max-depth 0 --max-features 8
python -m src.train_stunting --data $D --model lr --C 1.0
python -m src.train_stunting --data $D --model lr --C 0.1
python -m src.train_stunting --data $D --model gb --n-estimators 150 --max-depth 3 --max-features 4
```

**Opción B — desde tu portátil**, registrando en la EC2. Solo cambia la URI:

```powershell
$env:MLFLOW_TRACKING_URI = "http://<IP-PUBLICA>:8050"   # la IP de la EC2 del equipo
python -m src.evaluate_cv --target 24m
python -m src.evaluate_cv --target 12m
python -m src.experimento_escalera        # usa data/processed/model_dataset_escalera.csv, ya versionado
```

Los parámetros y métricas llegan siempre; las figuras, CSV de folds y el modelo
solo si el servidor se levantó con `--artifacts-destination` (sección 3).

## 5. Los pantallazos que pide la rúbrica

Tómalos **en la misma sesión**, con la IP visible, y **sin recortar la barra de
direcciones del navegador**: es la única parte de la captura que prueba que MLflow
corre en la EC2 y no en el portátil. Cuatro capturas cubren el requisito:

1. **Consola de EC2 → Instances**, con la instancia seleccionada: se ve el
   *Public IPv4 address* y, arriba a la derecha, el usuario de AWS.
2. **Terminal SSH** con el prompt `ubuntu@ip-XXX-XXX-XXX-XXX` visible y el
   servidor de MLflow corriendo (la salida de `mlflow server` o el `mlflow.log`).
3. **MLflow → lista de experimentos** en el navegador, con la barra de direcciones
   mostrando `http://<IP-PUBLICA>:8050`. Deben verse `stunting-evaluacion-cv`,
   `stunting-escalera` y `stunting-baseline-multi`.
4. **MLflow → un run abierto**, con las métricas (`roc_auc_mean`, `pr_auc_mean`,
   `sens_at_20_mean`…) y los artefactos (`curvas_oof_*.png`, `comparacion_*.png`).
   Los runs padre `cv_5x10_24m_cap20` y `escalera_5x10_cap20` son los mejores
   candidatos: muestran la comparación completa y los runs hijos anidados debajo.

Guárdalos en `docs/soportes/entrega2/` y referéncialos desde el reporte.

## 6. Al terminar: detener, no terminar

Consola de EC2 → instancia → **Instance state → Stop instance**. *Terminate* la
borra y con ella el `mlflow.db` con todos los runs. La rúbrica lo dice explícitamente.

En Learner Lab, además, **End Lab** detiene las instancias por sí solo; lo que hay
que evitar es terminarlas.

## Solución de problemas

| Síntoma | Causa probable |
|---|---|
| El navegador no carga `http://<IP>:8050` | El grupo de seguridad no abre el 8050, o MLflow se levantó sin `--host 0.0.0.0` |
| `mlflow: error: no such option: --allowed-hosts` | Esa opción es de MLflow 3.x; con 2.22.5 se omite |
| `Can't get attribute 'Winsorizer' on <module '__main__'>` al cargar un modelo | El modelo se guardó desde un notebook o script ejecutado directo. Los scripts del repo ya importan `Winsorizer` desde `src/preprocessing.py`; ejecútalos con `python -m src.<script>` desde la raíz |
| La IP cambió después de reiniciar | Normal en Learner Lab. Repetir los pantallazos, o asociar una Elastic IP si el laboratorio lo permite |
| `evaluate_cv` tarda mucho | Son 7 × 50 ajustes. En `t3.medium` ~4 min; `--n-repeats 3` para una prueba rápida |
