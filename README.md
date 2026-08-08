# AI Incidents Analysis Pipeline

Pipeline modular y reproducible para el análisis de incidentes de inteligencia artificial.
Transforma un dataset crudo (OECD AIM, AIID u otro compatible) en un dataset enriquecido,
un almacén de métricas en JSON y un reporte ejecutivo en PDF con las figuras y tablas
utilizadas en el capítulo de resultados de la tesis.

## Estructura del proyecto

```
incidents-pipeline/
├── config/
│   └── params.yaml               # Único punto de parametrización
├── src/
│   ├── __init__.py               # Logging y semillas globales
│   ├── ingestion.py              # Carga y validación del dataset (CSV)
│   ├── preprocessing.py          # Limpieza, fechas, filtros, multilabel
│   ├── nlp.py                    # Normalización, tokenización, lematización
│   ├── sentiment.py              # Backends de análisis de sentimiento
│   ├── analysis.py               # Métricas descriptivas (JSON metrics store)
│   └── visualization.py          # Figuras y reporte ejecutivo PDF
├── tests/                        # Tests unitarios (pytest)
├── data/
│   ├── raw/                      # Dataset fuente en CSV
│   └── processed/                # Dataset procesado en Parquet (generado)
├── outputs/
│   ├── figures/                  # Figuras PNG (generadas)
│   └── reports/                  # metrics.json y executive_report.pdf (generados)
├── pipeline_runner.ipynb         # Notebook de ejecución local
├── pipeline_runner_colab.ipynb   # Notebook de ejecución en Google Colab
├── requirements.txt              # Dependencias con versiones fijadas
└── requirements_colab.txt        # Dependencias adicionales para Colab
```

## Datasets soportados

El pipeline usa un sistema de mapeo de columnas (`config/params.yaml`) que permite
adaptarlo a cualquier dataset CSV sin modificar el código:

| Dataset | Fuente | Archivo de configuración |
|---------|--------|--------------------------|
| OECD AI Incidents Monitor (AIM) | [OECD](https://oecd.ai) | `schema_type: "OECD"` en params.yaml |
| AI Incident Database (AIID) | [incidentdatabase.ai](https://incidentdatabase.ai) | `schema_type: "AIID"` en params.yaml |

Para cambiar de dataset, se edita la sección `data:` en `config/params.yaml`.

## Instalación (entorno local)

Se requiere **Python 3.10 o superior**.

```bash
git clone https://github.com/karenrg/incidents_pipeline.git
cd incidents_pipeline
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

La primera ejecución descarga automáticamente los recursos de NLTK necesarios
(`wordnet`, `omw-1.4`, `stopwords`).

## Ejecución en Google Colab

Usá `pipeline_runner_colab.ipynb`. Las instrucciones están en la primera celda del notebook.

**Pasos:**
1. Abrí el notebook en Colab.
2. Ejecutá la **Celda 1** (Setup). Colab se reinicia automáticamente — es normal.
3. Después del reinicio, ejecutá desde la **Celda 2** en adelante.
4. Editá la **Celda 4 (Configuración)** si querés cambiar algún parámetro.

> Tip: activar GPU en `Entorno de ejecución → Cambiar tipo de entorno → T4 GPU`
> reduce el tiempo del paso de sentimiento de ~10 min a ~2 min.

## Configuración

Todos los parámetros se definen en `config/params.yaml`:

| Sección | Qué controla |
|---------|-------------|
| `data` | Ruta del CSV, tipo de esquema, mapeo de columnas, columnas multivaluadas |
| `filters` | Rango de años y regiones a incluir |
| `nlp` | Idioma, stopwords personalizadas, mapa de normalización, keywords de salud mental |
| `sentiment` | Backend, modelo, batch size, umbrales de clasificación |
| `analysis` | Grupos vulnerables, tipos de daño, N para rankings, ventanas de media móvil |
| `wordclouds` | Nubes de palabras a generar (con filtros opcionales por columna) |
| `reporting` | Título del reporte, fuente de datos, DPI de figuras, directorios de salida |

### Backends de sentimiento

| Backend | Descripción | Requisito |
|---------|-------------|-----------|
| `transformer` | Modelo HuggingFace (`cardiffnlp/twitter-roberta-base-sentiment-latest`) | ninguno adicional |
| `openai` | API de OpenAI (`gpt-4o-mini` por defecto) | API key (ver abajo) |
| `traditional_ml` | TF-IDF + SVM con pseudo-etiquetas VADER | ninguno adicional |

### Configurar la API key de OpenAI

Hay dos formas (usar solo una):

**Opción A — Variable de entorno (recomendado):**
```bash
export OPENAI_API_KEY="sk-proj-..."   # Linux/Mac
set OPENAI_API_KEY=sk-proj-...        # Windows CMD
```

**Opción B — Campo en params.yaml (solo para uso local, no commitear):**
```yaml
sentiment:
  backend: "openai"
  openai_api_key: "sk-proj-..."   # ← vaciar antes de hacer git commit
```

## Ejecución del pipeline (local)

Abrí y ejecutá `pipeline_runner.ipynb` de principio a fin. El notebook:

1. Lee `config/params.yaml` y fija las semillas globales.
2. Ejecuta ingestión → preprocesamiento → NLP → análisis de sentimiento.
3. Guarda el dataset procesado en `data/processed/incidents_processed.parquet`.
4. Calcula métricas descriptivas y las guarda en `outputs/reports/metrics.json`.
5. Genera figuras en `outputs/figures/` y el reporte en `outputs/reports/executive_report.pdf`.

## Descripción de módulos

### `src/ingestion.py`
Lee el CSV fuente, intenta múltiples encodings y separadores, y renombra las columnas
al esquema interno del pipeline usando `column_mapping` de `params.yaml`.
Loga advertencias si columnas críticas son nulas o si hay filas duplicadas.

### `src/preprocessing.py`
- Convierte columnas multivaluadas (strings tipo `"['a','b']"`) a listas Python.
- Parsea fechas, genera columnas `year` y `year_month`, filtra por rango de años.
- Filtra por regiones geográficas configuradas.
- Aplica recategorización de tipos de daño.
- Binariza columnas multilabel (one-hot) para el branch ML.

### `src/nlp.py`
Normalización Unicode, tokenización, lematización con WordNet, remoción de stopwords
(NLTK + stopwords personalizadas), mapa de normalización configurable.
Calcula un flag binario `mental_health_flag` basado en keywords configuradas.

### `src/sentiment.py`
Tres backends intercambiables vía `params.yaml`:
- **TransformerSentiment**: pipeline de HuggingFace con clasificación en batches.
- **OpenAISentiment**: prompt al API de OpenAI con respuesta estructurada en JSON.
- **MLSentiment**: TF-IDF + SVM con pseudo-etiquetas generadas por NLTK VADER.

Todos devuelven un score continuo de negatividad en `[0, 1]` que se clasifica en
Alta / Media / Baja según umbrales configurados.

### `src/analysis.py`
Calcula y serializa a JSON todas las métricas descriptivas:
evolución regional, concentración acumulada, distribución de principios,
tipos de daño, industrias, stakeholders, grupos vulnerables, cronología de daño,
índice de severidad, análisis de salud mental, negatividad por principio en el tiempo.

### `src/visualization.py`
Genera todas las figuras en PNG (matplotlib) y ensambla el reporte ejecutivo en PDF (fpdf2).
Usa `config["reporting"]["data_source_label"]` y `config["reporting"]["report_title"]`
para que el reporte sea genérico respecto al dataset usado.

## Tests

```bash
pytest
```

Los tests cubren ingestion (mapeo de columnas, validaciones), preprocesamiento
(parseo de listas, fechas, filtros, binarización multilabel) y análisis de sentimiento
(clasificación por umbrales, backend traditional_ml, evaluación contra anotaciones humanas).

## Salidas generadas

| Archivo | Descripción |
|---------|-------------|
| `data/processed/incidents_processed.parquet` | Dataset enriquecido y estandarizado |
| `outputs/reports/metrics.json` | Almacén de métricas (todas las agregaciones) |
| `outputs/reports/executive_report.pdf` | Reporte ejecutivo con metodología y figuras |
| `outputs/figures/*.png` | Figuras individuales en alta resolución (300 DPI) |

> Los directorios `data/processed/` y `outputs/` están en `.gitignore` ya que son
> artefactos generados que se reproducen ejecutando el pipeline.
