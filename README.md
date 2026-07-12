# AI Incidents Pipeline — OECD AIM

Pipeline modular y reproducible para el análisis de incidentes de IA del
*OECD AI Incidents Monitor* (AIM). Transforma el dataset crudo en un dataset
enriquecido, un almacén de métricas en JSON y un reporte ejecutivo en PDF con
las figuras y tablas utilizadas en el capítulo de resultados de la tesis.

## Estructura del proyecto

```
ai-incidents-pipeline/
├── config/params.yaml        # Único punto de parametrización
├── src/
│   ├── __init__.py            # Logging y semillas globales
│   ├── ingestion.py           # Carga y validación del dataset
│   ├── preprocessing.py       # Limpieza, fechas, filtros, multilabel
│   ├── nlp.py                 # Normalización, tokenización, lematización
│   ├── sentiment.py           # Backends de sentimiento/negatividad
│   ├── analysis.py            # Métricas descriptivas (JSON metrics store)
│   └── visualization.py       # Figuras y reporte PDF
├── tests/                      # Tests unitarios (pytest)
├── data/raw/                   # Dataset fuente (oecd_aim.csv)
├── data/processed/             # Dataset procesado (parquet)
├── outputs/figures/            # Figuras generadas (PNG)
├── outputs/reports/            # metrics.json y executive_report.pdf
├── pipeline_runner.ipynb       # Notebook de ejecución end-to-end
└── requirements.txt
```

## Instalación

Se requiere Python 3.10 o superior.

```bash
cd ai-incidents-pipeline
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

La primera ejecución descarga automáticamente los recursos de NLTK
necesarios (`stopwords`, `wordnet`, `omw-1.4`, `punkt_tab`, `vader_lexicon`).

## Configuración

Todos los parámetros del pipeline (rutas de datos, mapeo de columnas,
filtros temporales/geográficos, configuración de NLP, backend de
sentimiento, umbrales de clasificación y parámetros de análisis/reportes)
se definen en `config/params.yaml`. Este es el único punto de
parametrización: no hay valores fijos en el código.

Backends de sentimiento disponibles (`sentiment.backend`):

- `transformer`: modelo HuggingFace (`sentiment.model_name`).
- `openai`: API de OpenAI (requiere la variable de entorno `OPENAI_API_KEY`).
- `traditional_ml`: TF-IDF + SVM, con pseudo-etiquetas generadas con NLTK VADER.

## Ejecución del pipeline

Abrir y ejecutar `pipeline_runner.ipynb` de principio a fin. El notebook:

1. Carga `config/params.yaml` y fija las semillas globales (`random`,
   `numpy`, `torch`) a partir de `random_state`.
2. Ejecuta ingestión, preprocesamiento, NLP y análisis de sentimiento.
3. Guarda el dataset procesado en `data/processed/incidents_processed.parquet`.
4. Calcula las métricas descriptivas y las guarda en
   `outputs/reports/metrics.json`.
5. Genera las figuras en `outputs/figures/` y el reporte ejecutivo en
   `outputs/reports/executive_report.pdf`.
6. Imprime `Pipeline completado.` al finalizar.

## Tests

```bash
pytest
```

Los tests cubren la ingestión (mapeo de columnas, validaciones), el
preprocesamiento (parseo de listas, fechas, filtros, recategorización de
daño, binarización multilabel) y el análisis de sentimiento (clasificación
por umbrales, backend `traditional_ml`, evaluación contra anotaciones
humanas).

## Salidas generadas

- `data/processed/incidents_processed.parquet`: dataset enriquecido y
  estandarizado.
- `outputs/reports/metrics.json`: almacén de métricas con todas las
  agregaciones utilizadas en las figuras y tablas.
- `outputs/reports/executive_report.pdf`: reporte ejecutivo con
  metodología, tablas estadísticas y figuras.
- `outputs/figures/*.png`: figuras individuales en alta resolución.
