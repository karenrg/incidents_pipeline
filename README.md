# Pipeline de Análisis de Incidentes de IA

Pipeline modular y reproducible para analizar datasets de incidentes de inteligencia artificial. A partir de un CSV, genera métricas descriptivas, visualizaciones y dos tipos de reporte: un PDF ejecutivo y un explorador HTML interactivo.

Desarrollado como parte de una tesis de maestría.

---

## ¿Qué hace este pipeline?

Dado un CSV con incidentes de IA, el pipeline:

1. **Carga y valida** el dataset, adaptando los nombres de columna a un esquema interno.
2. **Preprocesa** los datos: limpia texto, parsea fechas, filtra por año y región.
3. **Procesa lenguaje natural**: tokeniza, lematiza y detecta keywords temáticas (ej: salud mental).
4. **Analiza sentimiento** (opcional): clasifica el texto de cada incidente como negativo alto, medio o bajo.
5. **Calcula métricas** descriptivas: distribuciones, rankings, tendencias temporales, índice de severidad.
6. **Genera reportes**:
   - `executive_report.pdf` — reporte ejecutivo con gráficos en alta resolución.
   - `interactive_report.html` — explorador interactivo con filtros, scatter plots, wordclouds y más.

---

## Cómo correr el pipeline

### Opción A — Google Colab (recomendado, sin instalación)

1. Abrí [`pipeline_runner_colab.ipynb`](pipeline_runner_colab.ipynb) en Google Colab.
2. Ejecutá la **Celda 1** (Setup). Colab se reinicia automáticamente — es normal.
3. Editá la **Celda 2 (Configuración)** con la ruta de tu CSV y las columnas de tu dataset.
4. Ejecutá el resto de las celdas en orden.
5. Al final, la **Celda 13** descarga los reportes generados.

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

Solo necesitás indicar la ruta de tu CSV y qué columna de tu dataset cumple cada rol:

```yaml
# config/params.yaml
data:
  source_path: "data/raw/mi_dataset.csv"
  column_mapping:
    text_data:  "descripcion"   # Texto principal del incidente
    event_date: "fecha"         # Fecha del incidente
    geo_zone:   "region"        # Región (opcional)
    # ... resto de columnas son opcionales
```

Con solo eso, el pipeline corre completo con valores por defecto para todo lo demás.

### Análisis de sentimiento (opcional)

Desactivado por defecto. Para activarlo:

```yaml
sentiment:
  enabled: true
  backend: "openai"        # "openai", "transformer" o "traditional_ml"
  openai_model: "gpt-4o-mini"
  openai_api_key: ""       # O usá la variable de entorno OPENAI_API_KEY
```

| Backend | Descripción | Requiere |
|---------|-------------|----------|
| `openai` | GPT-4o-mini vía API (más preciso) | API key de OpenAI |
| `transformer` | Modelo HuggingFace (~500 MB de descarga) | Nada adicional |
| `traditional_ml` | TF-IDF + SVM, sin descarga | Nada adicional |

**Para configurar la API key de OpenAI:**

```bash
# Opción recomendada: variable de entorno (no queda en el código)
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
│   ├── visualization.py            # Gráficos PNG y reporte PDF
│   └── html_report.py              # Reporte HTML interactivo
│
├── data/
│   ├── raw/                        # Poné tu CSV aquí
│   └── processed/                  # Dataset procesado (generado automáticamente)
│
├── outputs/
│   ├── figures/                    # Gráficos PNG individuales (generados)
│   └── reports/                    # PDF, HTML y metrics.json (generados)
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
| `outputs/reports/executive_report.pdf` | Reporte ejecutivo con metodología y gráficos |
| `outputs/reports/interactive_report.html` | Explorador interactivo (abrir en el navegador) |
| `outputs/figures/*.png` | Gráficos individuales en 300 DPI |

> Los directorios `data/processed/` y `outputs/` están en `.gitignore` porque son artefactos generados que se reproducen corriendo el pipeline.

---

## Datasets compatibles

El pipeline funciona con cualquier CSV de incidentes de IA. Está probado con:

| Dataset | Fuente |
|---------|--------|
| AI Incident Database (AIID) | [incidentdatabase.ai](https://incidentdatabase.ai) |
| OECD AI Incidents Monitor (AIM) | [oecd.ai](https://oecd.ai) |

Para adaptarlo a otro dataset, solo cambiás el `column_mapping` en `params.yaml` (o en la celda de configuración del notebook Colab).

---

## Tests

```bash
pytest
```

Cubren ingestion (mapeo de columnas, validaciones), preprocesamiento (parseo de listas, fechas, filtros) y sentiment (clasificación por umbrales, backend traditional_ml).

---

## Buenas prácticas implementadas

- **Único punto de configuración**: todo en `config/params.yaml`, sin constantes dispersas en el código.
- **Sin secretos en el repositorio**: las API keys se leen de variables de entorno o Secrets de Colab.
- **Reproducibilidad**: semilla global configurable (`random_state`) y dependencias con versiones fijadas.
- **Modularidad**: cada módulo en `src/` tiene una responsabilidad única y puede testearse por separado.
- **Sensible a datos faltantes**: columnas opcionales vacías no rompen el pipeline.
- **Logging estructurado**: todos los pasos loguean con nivel y contexto para facilitar el debugging.
