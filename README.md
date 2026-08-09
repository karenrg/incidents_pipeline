# Pipeline de Análisis de Datos

Pipeline modular y reproducible para el análisis descriptivo de datasets estructurados. Fue diseñado para analizar incidentes de inteligencia artificial, pero su arquitectura basada en mapeo de columnas permite adaptarlo a cualquier dominio que comparta una estructura similar: eventos con texto, fecha, categoría y región.

Desarrollado como parte de una tesis de maestría en Ciencia de Datos.

---

## ¿Qué hace este pipeline?

Dado un CSV, el pipeline:

1. **Carga y valida** el dataset, adaptando los nombres de columna a un esquema interno configurable.
2. **Preprocesa** los datos: limpia texto, parsea fechas, filtra por año y región.
3. **Procesa lenguaje natural**: tokeniza, lematiza y detecta keywords temáticas.
4. **Analiza sentimiento** (opcional): clasifica el texto de cada registro como negativo alto, medio o bajo.
5. **Calcula métricas** descriptivas: distribuciones, rankings, tendencias temporales, índice de severidad.
6. **Genera un reporte HTML interactivo** con explorador de columnas, filtros dinámicos, scatter plots, tabla cruzada y wordclouds.

---

## Cómo correr el pipeline

### Opción A — Google Colab (recomendado, sin instalación)

1. Abrí [`pipeline_runner_colab.ipynb`](pipeline_runner_colab.ipynb) en Google Colab.
2. Ejecutá la **Celda 1** (Setup). Colab se reinicia automáticamente — es normal.
3. Editá la **Celda 2 (Configuración)** con la ruta de tu CSV y el mapeo de columnas.
4. Ejecutá el resto de las celdas en orden.
5. La **Celda 12** descarga el reporte HTML y el JSON de métricas.

> 💡 Para usar análisis de sentimiento con GPU: `Entorno de ejecución → Cambiar tipo de entorno → T4 GPU`

### Opción B — Entorno local

Requiere **Python 3.10 o superior**.

```bash
git clone https://github.com/karenrg/incidents_pipeline
cd incidents_pipeline
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Luego editá `config/params.yaml` y ejecutá `pipeline_runner.ipynb`.

---

## Configuración

### Mínimo necesario

Solo necesitás indicar la ruta de tu CSV y qué columna cumple cada rol:

```yaml
# config/params.yaml
data:
  source_path: "data/raw/mi_dataset.csv"
  column_mapping:
    text_data:  "descripcion"   # Texto principal (requerido)
    event_date: "fecha"         # Fecha (requerido)
    geo_zone:   "region"        # Región (opcional)
    # ... el resto son opcionales
```

Con solo eso, el pipeline corre completo con valores por defecto para todo lo demás.

### Análisis de sentimiento (opcional)

Desactivado por defecto. Para activarlo:

```yaml
sentiment:
  enabled: true
  backend: "openai"        # "openai" o "transformer"
  openai_model: "gpt-4o-mini"
  openai_api_key: ""       # O usá la variable de entorno OPENAI_API_KEY
```

| Backend | Descripción | Requiere |
|---------|-------------|----------|
| `openai` | GPT-4o-mini vía API (más preciso) | API key de OpenAI |
| `transformer` | Modelo de HuggingFace configurable (`model_name` en params.yaml). Default: `cardiffnlp/twitter-roberta-base-sentiment-latest` | Nada adicional |

**Para configurar la API key de OpenAI:**

```bash
# Variable de entorno (recomendado — no queda en el código)
export OPENAI_API_KEY="sk-proj-..."   # Linux/Mac
set OPENAI_API_KEY=sk-proj-...        # Windows CMD
```

En Colab, usá **Secrets** (ícono 🔑 en el panel izquierdo) con el nombre `OPENAI_API_KEY`.

---

## Estructura del proyecto

```
incidents-pipeline/
│
├── config/
│   └── params.yaml                 # Configuración del pipeline (editá esto)
│
├── src/
│   ├── __init__.py                 # Logging y semillas globales
│   ├── ingestion.py                # Carga del CSV y validación de columnas
│   ├── preprocessing.py            # Limpieza, fechas, filtros, columnas multilabel
│   ├── nlp.py                      # Tokenización, lematización, stopwords
│   ├── sentiment.py                # Análisis de sentimiento (3 backends)
│   ├── analysis.py                 # Cálculo de métricas descriptivas
│   └── html_report.py              # Reporte HTML interactivo
│
├── data/
│   ├── raw/                        # Poné tu CSV aquí
│   └── processed/                  # Dataset procesado (generado automáticamente)
│
├── outputs/
│   └── reports/                    # HTML y metrics.json (generados)
│
├── tests/                          # Tests unitarios (pytest)
│
├── pipeline_runner.ipynb           # Notebook para correr en local
├── pipeline_runner_colab.ipynb     # Notebook para correr en Google Colab
├── requirements.txt                # Dependencias para entorno local
└── requirements_colab.txt          # Dependencias para Google Colab
```

---

## Archivos generados

| Archivo | Descripción |
|---------|-------------|
| `data/processed/incidents_processed.parquet` | Dataset enriquecido con columnas calculadas |
| `outputs/reports/metrics.json` | Todas las métricas en formato JSON |
| `outputs/reports/interactive_report.html` | Reporte interactivo (abrir en el navegador) |

> Los directorios `data/processed/` y `outputs/` están en `.gitignore` porque son artefactos generados que se reproducen corriendo el pipeline.

---

## Datasets de referencia

El pipeline fue desarrollado y probado con datasets de incidentes de IA, pero puede adaptarse a cualquier CSV con estructura similar (texto, fecha, categoría, región). Datasets usados en el desarrollo:

| Dataset | Fuente |
|---------|--------|
| OECD AI Incidents Monitor (AIM) | [oecd.ai](https://oecd.ai) |
| AI Incident Database (AIID) | [incidentdatabase.ai](https://incidentdatabase.ai) |

Para adaptarlo a otro dataset, solo cambiás el `column_mapping` en `params.yaml`.

---

## Tests

```bash
pytest
```

Cubren ingestion (mapeo de columnas, validaciones), preprocesamiento (parseo de listas, fechas, filtros) y sentiment (clasificación por umbrales).

---

## Buenas prácticas implementadas

- **Único punto de configuración**: todo en `config/params.yaml`, sin constantes dispersas en el código.
- **Sin secretos en el repositorio**: las API keys se leen de variables de entorno o Secrets de Colab.
- **Reproducibilidad**: semilla global configurable (`random_state`) y dependencias con versiones fijadas.
- **Modularidad**: cada módulo en `src/` tiene una responsabilidad única y puede testearse por separado.
- **Sensible a datos faltantes**: columnas opcionales vacías no rompen el pipeline.
- **Logging estructurado**: todos los pasos loguean con nivel y contexto para facilitar el debugging.
