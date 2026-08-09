"""Interactive HTML report — EDA + Analysis tabs.

Tab 1 (EDA):      Overview cards + per-column explorer with Plotly chart on click.
Tab 2 (Analysis): All pre-computed pipeline metrics as Plotly charts + dynamic
                  word-cloud builder + dynamic filter explorer.

Language toggle (ES/EN) is built into the HTML.
No new Python dependencies beyond what is already in requirements.txt.
Plotly.js and wordcloud2.js are loaded from CDN when the report is opened.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_html_report(df: pd.DataFrame, metrics: dict, config: dict) -> Path:
    """Generate the interactive HTML report and return its path."""
    eda_stats = _compute_eda_stats(df, config)
    records   = _prepare_records(df, config)
    col_roles = config["analysis"]["columns"]
    reporting = config["reporting"]

    # Inject column_mapping into reporting so _render_html can build COL_LABELS
    reporting = dict(reporting)
    reporting["_col_mapping"] = config.get("data", {}).get("column_mapping", {})

    html = _render_html(eda_stats, records, col_roles, metrics, reporting)

    out_dir = Path(reporting["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "interactive_report.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info("Interactive HTML report → %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if hasattr(obj, "isoformat"):    return str(obj)
        return super().default(obj)


def _safe(v):
    if isinstance(v, list):        return [_safe(x) for x in v]
    if isinstance(v, np.bool_):    return bool(v)
    if isinstance(v, np.integer):  return int(v)
    if isinstance(v, np.floating): return None if np.isnan(v) else float(v)
    if isinstance(v, float):       return None if np.isnan(v) else v
    if isinstance(v, pd.Period):   return str(v)
    if v is pd.NaT:                return None
    if hasattr(v, "isoformat"):    return str(v)
    try:
        if pd.isna(v): return None
    except (TypeError, ValueError):
        pass
    if not isinstance(v, (str, int, float, bool, type(None))):
        return str(v)
    return v


def _excl_eda(config: dict) -> set[str]:
    """Columns excluded from the EDA column explorer (too heavy / not useful).

    column_mapping keys are the INTERNAL df column names (e.g. 'text_data').
    Values are original CSV names (e.g. 'description') — we must exclude by key.
    """
    mapping = config.get("data", {}).get("column_mapping", {})
    text_col = "text_data" if "text_data" in mapping else next(iter(mapping), "")
    return {text_col, config["analysis"]["columns"].get("tokens", "tokens")}


def _excl_records(config: dict) -> set[str]:
    """Columns excluded from embedded DATA_RECORDS.
    text_data is now included so the wordcloud can use full descriptions."""
    return set()


def _count_duplicates(df: pd.DataFrame) -> int:
    hashable = [c for c in df.columns
                if not df[c].dropna().apply(lambda x: isinstance(x, list)).any()]
    return int(df[hashable].duplicated().sum()) if hashable else 0


def _compute_eda_stats(df: pd.DataFrame, config: dict) -> dict:
    excl = _excl_eda(config)
    stats: dict = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_cells": int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().sum().sum() / max(df.shape[0] * df.shape[1], 1) * 100, 1),
        "duplicates": _count_duplicates(df),
        "columns": [],
    }
    for col in df.columns:
        if col in excl:
            continue
        s = df[col]
        is_list_col = s.dropna().apply(lambda x: isinstance(x, list)).any()
        entry: dict = {
            "name": col,
            "null_count": int(s.isnull().sum()),
            "null_pct": round(s.isnull().mean() * 100, 1),
        }
        if is_list_col:
            entry["col_type"] = "list"
            exploded = s.explode().dropna()
            entry["unique_count"] = int(exploded.nunique())
            vc = exploded.value_counts().head(20)
            entry["top_values"] = [str(x) for x in vc.index.tolist()]
            entry["top_counts"]  = vc.values.tolist()
        elif pd.api.types.is_numeric_dtype(s):
            entry["col_type"] = "numeric"
            clean = s.dropna()
            entry["unique_count"] = int(s.nunique())
            if len(clean) > 0:
                entry["mean"]   = round(float(clean.mean()), 3)
                entry["median"] = round(float(clean.median()), 3)
                entry["std"]    = round(float(clean.std()), 3)
                entry["min"]    = round(float(clean.min()), 3)
                entry["max"]    = round(float(clean.max()), 3)
                n_bins = min(30, max(5, int(clean.nunique())))
                hist, edges = np.histogram(clean, bins=n_bins)
                entry["hist_counts"] = hist.tolist()
                entry["hist_edges"]  = [round(float(e), 3) for e in edges.tolist()]
            else:
                for k in ("mean", "median", "std", "min", "max"):
                    entry[k] = None
        else:
            entry["col_type"] = "categorical"
            entry["unique_count"] = int(s.nunique())
            vc = s.value_counts().head(20)
            entry["top_values"] = [str(x) for x in vc.index.tolist()]
            entry["top_counts"]  = vc.values.tolist()
        stats["columns"].append(entry)
    return stats


def _prepare_records(df: pd.DataFrame, config: dict) -> list[dict]:
    """Embed all columns except raw text (tokens ARE included for wordcloud)."""
    excl = _excl_records(config)
    cols = [c for c in df.columns if c not in excl]
    return [{col: _safe(row[col]) for col in cols} for _, row in df[cols].iterrows()]


# ─────────────────────────────────────────────────────────────────────────────
# HTML rendering
# ─────────────────────────────────────────────────────────────────────────────

def _render_html(
    eda_stats: dict,
    records: list,
    col_roles: dict,
    metrics: dict,
    reporting: dict,
) -> str:
    title  = reporting.get("report_title", "Análisis de Incidentes de IA")
    source = reporting.get("data_source_label", "")

    eda_json     = json.dumps(eda_stats,  cls=_NpEncoder, ensure_ascii=False)
    data_json    = json.dumps(records,    cls=_NpEncoder, ensure_ascii=False)
    roles_json   = json.dumps(col_roles,  ensure_ascii=False)
    metrics_json = json.dumps(metrics,    cls=_NpEncoder, ensure_ascii=False)
    # COL_LABELS: internal column name → original CSV column name (for display in selectors)
    col_mapping   = reporting.get("_col_mapping", {})
    col_labels_json = json.dumps(col_mapping, ensure_ascii=False)
    html_cfg     = reporting.get("html_report", {})
    html_cfg_json = json.dumps({
        "filter_columns":       html_cfg.get("filter_columns", []),
        "range_filter_columns": html_cfg.get("range_filter_columns", ["year"]),
        "kpi_labels":           html_cfg.get("kpi_labels", {}),
    }, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<script src="https://cdn.jsdelivr.net/npm/wordcloud@1.2.2/src/wordcloud2.js"></script>
<style>{_CSS}</style>
</head>
<body>
<header>
  <div class="header-inner">
    <div>
      <h1>{title}</h1>
      <span class="badge">{source}</span>
    </div>
    <div class="lang-toggle">
      <button class="lang-btn active" id="btn-es" onclick="setLang('es')">ES</button>
      <button class="lang-btn"        id="btn-en" onclick="setLang('en')">EN</button>
    </div>
  </div>
</header>

<div class="tab-bar">
  <button class="tab-btn active" id="tabbtn-eda"      onclick="showTab('eda',this)"      data-i18n="tab_eda">🔍 Exploración del dato</button>
  <button class="tab-btn"        id="tabbtn-analysis" onclick="showTab('analysis',this)" data-i18n="tab_analysis">📊 Análisis del dataset</button>
</div>

<!-- ═══ TAB EDA ══════════════════════════════════════════════════════════ -->
<div id="tab-eda" class="tab-content">

  <!-- 1. Resumen de columnas -->
  <section>
    <h2 class="section-title" data-i18n="col_summary_title">Resumen de columnas</h2>
    <div id="col-summary-table" class="summary-table-wrap"></div>
  </section>

  <!-- 2. Completitud -->
  <section>
    <h2 class="section-title" data-i18n="col_completeness">Completitud por columna</h2>
    <p class="hint" data-i18n="col_completeness_hint">% de registros con valor no nulo.</p>
    <div class="chart-card" style="max-width:100%">
      <div id="eda-completeness" style="height:320px"></div>
    </div>
  </section>

  <!-- 3. Explorador de columnas con KPIs -->
  <section class="overview-grid" id="eda-overview-cards"></section>
  <section class="col-explorer">
    <h2 data-i18n="col_explorer_title">Explorador de columnas</h2>
    <p class="hint" data-i18n="col_explorer_hint">Hacé click en una columna para ver su perfil detallado.</p>
    <div class="col-cards" id="col-cards"></div>
    <div id="col-detail" class="col-detail hidden">
      <div id="col-detail-stats"></div>
      <div id="col-chart-type-btns" class="chart-type-btns"></div>
      <div id="col-detail-chart"></div>
    </div>
  </section>

  <!-- 4. Scatter Plot -->
  <section>
    <h2 class="section-title" data-i18n="section_scatter">Scatter Plot</h2>
    <p class="hint" data-i18n="section_scatter_hint">Seleccioná dos columnas numéricas para explorar su relación.</p>
    <div class="filter-panel">
      <div class="filter-builder" style="flex-wrap:wrap;gap:10px">
        <div style="display:flex;flex-direction:column;gap:4px">
          <label style="font-size:.78rem;color:#666;font-weight:600" data-i18n="scatter_x">Eje X</label>
          <select id="scatter-x" class="filter-builder select" style="min-width:170px"></select>
        </div>
        <div style="display:flex;flex-direction:column;gap:4px">
          <label style="font-size:.78rem;color:#666;font-weight:600" data-i18n="scatter_y">Eje Y</label>
          <select id="scatter-y" class="filter-builder select" style="min-width:170px"></select>
        </div>
        <div style="display:flex;flex-direction:column;gap:4px">
          <label style="font-size:.78rem;color:#666;font-weight:600" data-i18n="scatter_color">Color por</label>
          <select id="scatter-color" class="filter-builder select" style="min-width:170px"></select>
        </div>
        <button class="btn-add" style="align-self:flex-end" onclick="renderScatter()" data-i18n="btn_generate">Generar</button>
      </div>
    </div>
    <div class="chart-card"><div id="scatter-chart" class="chart-area"></div></div>
  </section>

  <!-- 5. Tabla cruzada -->
  <section>
    <h2 class="section-title" data-i18n="section_crosstab">Tabla cruzada</h2>
    <p class="hint" data-i18n="section_crosstab_hint">Seleccioná dos columnas categóricas para ver su co-ocurrencia.</p>
    <div class="filter-panel">
      <div class="filter-builder" style="flex-wrap:wrap;gap:10px">
        <div style="display:flex;flex-direction:column;gap:4px">
          <label style="font-size:.78rem;color:#666;font-weight:600" data-i18n="crosstab_row">Filas</label>
          <select id="cross-row" class="filter-builder select" style="min-width:170px"></select>
        </div>
        <div style="display:flex;flex-direction:column;gap:4px">
          <label style="font-size:.78rem;color:#666;font-weight:600" data-i18n="crosstab_col">Columnas</label>
          <select id="cross-col" class="filter-builder select" style="min-width:170px"></select>
        </div>
        <button class="btn-add" style="align-self:flex-end" onclick="renderCrosstab()" data-i18n="btn_generate">Generar</button>
      </div>
    </div>
    <div class="chart-card"><div id="crosstab-chart" class="chart-area"></div></div>
  </section>

</div>

<!-- ═══ TAB ANALYSIS ═════════════════════════════════════════════════════ -->
<div id="tab-analysis" class="tab-content" style="display:none">

  <!-- Summary KPI cards -->
  <section class="overview-grid" id="analysis-cards"></section>

  <!-- Analysis filter bar -->
  <section class="analysis-filter-bar" id="analysis-filter-bar">
    <div class="af-group">
      <span class="af-label" data-i18n="af_year_range">Rango de años</span>
      <div class="af-year-inputs">
        <input type="number" id="af-year-min" class="af-year-input" placeholder="Desde" onchange="applyAnalysisFilters()">
        <span class="af-year-sep">—</span>
        <input type="number" id="af-year-max" class="af-year-input" placeholder="Hasta" onchange="applyAnalysisFilters()">
      </div>
    </div>
    <div class="af-group af-group-regions">
      <span class="af-label" data-i18n="af_regions">Regiones</span>
      <div id="af-regions" class="af-checkboxes"></div>
    </div>
    <button class="af-clear-btn" onclick="clearAnalysisFilters()" data-i18n="af_clear">Limpiar filtros</button>
  </section>

  <!-- Pre-computed pipeline charts -->
  <section>
    <h2 class="section-title" data-i18n="section_pipeline">Resultados del pipeline</h2>
    <p class="hint" data-i18n="section_pipeline_hint">Calculados sobre el conjunto completo configurado en params.yaml.</p>
    <div class="charts-grid" id="metrics-charts"></div>
  </section>

  <!-- Interactive word cloud -->
  <section>
    <h2 class="section-title" data-i18n="section_wordcloud">Nube de palabras dinámica</h2>
    <div class="wc-panel">
      <div class="wc-controls">
        <div class="wc-ctrl-row">
          <label data-i18n="wc_col_label">Columna de texto</label>
          <input type="text" id="wc-col-search"
                 placeholder="Buscar columna..."
                 oninput="searchWcCol(this.value)"
                 autocomplete="off" class="filter-search-input" style="margin-bottom:4px">
          <select id="wc-col-select" onchange="onWcColChange()">
            <option value="" data-i18n-opt="select_col">— Seleccioná una columna —</option>
          </select>
        </div>
        <div id="wc-filter-wrap" class="wc-filter-wrap hidden">
          <label data-i18n="wc_filter_label">Filtrar filas por columna</label>
          <select id="wc-filter-col" onchange="onWcFilterColChange()">
            <option value="" data-i18n-opt="select_col">— Columna de filtro (opcional) —</option>
          </select>
          <div id="wc-filter-vals" class="filter-vals-list" style="max-height:100px;min-width:260px"></div>
        </div>
        <button class="btn-add" onclick="generateWordcloud()" data-i18n="wc_generate">Generar nube</button>
      </div>
      <div class="wc-canvas-wrap">
        <canvas id="wc-canvas" width="800" height="380"></canvas>
        <p id="wc-placeholder" class="wc-placeholder" data-i18n="wc_placeholder">Seleccioná una columna y hacé click en "Generar nube".</p>
      </div>
    </div>
  </section>

  <!-- Dynamic explorer -->
  <section>
    <h2 class="section-title" data-i18n="section_explorer">Explorador dinámico</h2>
    <p class="hint" data-i18n="section_explorer_hint">Aplicá filtros para explorar subconjuntos del dataset.</p>
    <div class="filter-panel">
      <div class="filter-builder">
        <input type="text" id="filter-col-search"
               placeholder="Buscar columna..."
               oninput="searchFilterCol(this.value)"
               autocomplete="off" class="filter-search-input">
        <select id="filter-col" onchange="onFilterColChange()">
          <option value="" data-i18n-opt="select_col">— Seleccioná una columna —</option>
        </select>
        <div id="filter-vals-wrap" class="filter-vals-wrap hidden">
          <div id="filter-vals-list" class="filter-vals-list"></div>
          <button class="btn-add" onclick="addFilter()" data-i18n="add_filter">Agregar filtro</button>
        </div>
      </div>
      <div id="active-filters" class="active-filters">
        <span class="no-filters" data-i18n="no_filters">Sin filtros — mostrando todos los registros.</span>
      </div>
    </div>
    <div id="record-banner" class="record-banner"></div>
    <div class="charts-grid" id="dynamic-charts"></div>
  </section>
</div>

<script>
/* ─── Embedded data ─────────────────────────────────────────────────── */
const EDA_STATS    = {eda_json};
const DATA_RECORDS = {data_json};
const COL_ROLES    = {roles_json};
const METRICS      = {metrics_json};
const HTML_CONFIG  = {html_cfg_json};
/* internal col name → original CSV name (e.g. text_data → description) */
const COL_LABELS   = {col_labels_json};
/* Display label for a column: original CSV name if available, else internal */
const colLabel = c => COL_LABELS[c] || c;

/* ─── i18n ───────────────────────────────────────────────────────────── */
const LANGS = {{
  es: {{
    tab_eda: '🔍 Exploración del dato',
    tab_analysis: '📊 Análisis del dataset',
    section_pipeline: 'Resultados del pipeline',
    section_pipeline_hint: 'Calculados sobre el conjunto completo configurado en params.yaml.',
    section_wordcloud: 'Nube de palabras dinámica',
    section_explorer: 'Explorador dinámico',
    section_explorer_hint: 'Aplicá filtros para explorar subconjuntos del dataset.',
    col_explorer_title: 'Explorador de columnas',
    col_explorer_hint: 'Hacé click en una columna para ver su perfil detallado.',
    total_incidents: 'Incidentes totales', period: 'Período',
    eda_kpi_rows_sub: 'filas en el dataset',
    eda_kpi_cols: 'Columnas', eda_kpi_cols_sub: 'columnas en el dataset',
    eda_kpi_missing_sub: 'celdas con valor nulo',
    eda_kpi_dups: 'Filas duplicadas', eda_kpi_dups_sub: 'filas idénticas en todas sus columnas',
    col_search_placeholder: 'Buscar columna...',
    type_lbl: 'Tipo', unique_lbl: 'Únicos', nulls_lbl: 'Nulos',
    mean_lbl: 'Media', median_lbl: 'Mediana', std_lbl: 'Desv. estándar', minmax_lbl: 'Mín / Máx',
    numeric_lbl: 'Numérico', categorical_lbl: 'Categórico', multilabel_lbl: 'Multilabel',
    frequency: 'Frecuencia', incidents: 'Incidentes', year_lbl: 'Año',
    month_lbl: 'Mes', cum_pct: '% acumulado', severity_lbl: 'Severidad promedio',
    neg_score: 'Score de negatividad', pct_total: '% del total',
    select_col: '— Seleccioná una columna —',
    add_filter: 'Agregar filtro', clear_all: 'Limpiar todo',
    no_filters: 'Sin filtros — mostrando todos los registros.',
    no_data: 'Sin datos para mostrar',
    wc_col_label: 'Columna de texto', wc_filter_label: 'Filtrar filas por columna',
    wc_generate: 'Generar nube',
    wc_placeholder: 'Seleccioná una columna y hacé click en "Generar nube".',
    chart_revol: 'Evolución regional', chart_cumul: 'Concentración acumulada',
    chart_harmtype: 'Tipos de daño por región', chart_princip: 'Principios / Dominios de riesgo',
    chart_indust: 'Top industrias', chart_harmed: 'Grupos afectados por región',
    chart_vulner: 'Grupos vulnerables', chart_chrono: 'Cronología de daño',
    chart_sev: 'Índice de severidad', chart_mh: 'Incidentes de salud mental',
    chart_neg: 'Negatividad por principio', chart_tok: 'Tokens más frecuentes',
    records_of: 'registros', of_total: 'del total',
    col_summary_title: 'Resumen de columnas',
    col_completeness: 'Completitud por columna',
    col_completeness_hint: '% de registros con valor no nulo.',
    th_col: 'Columna', th_type: 'Tipo', th_records: 'Registros',
    th_unique: 'Únicos', th_nulls: 'Nulos',
    completeness_lbl: 'Completitud (%)',
    chart_histogram: 'Histograma', chart_boxplot: 'Box Plot',
    chart_freq: 'Frecuencia', chart_by_region: 'Por región', chart_trend: 'Tendencia',
    section_scatter: 'Scatter Plot', section_scatter_hint: 'Seleccioná dos columnas numéricas para explorar su relación.',
    scatter_x: 'Eje X', scatter_y: 'Eje Y', scatter_color: 'Color por',
    section_crosstab: 'Tabla cruzada', section_crosstab_hint: 'Seleccioná dos columnas categóricas para ver su co-ocurrencia.',
    crosstab_row: 'Filas', crosstab_col: 'Columnas',
    btn_generate: 'Generar', none_col: '(sin color)',
    af_year_range: 'Rango de años', af_regions: 'Regiones', af_clear: 'Limpiar filtros',
    af_active: 'Filtros activos',
  }},
  en: {{
    tab_eda: '🔍 Data Exploration',
    tab_analysis: '📊 Dataset Analysis',
    section_pipeline: 'Pipeline Results',
    section_pipeline_hint: 'Computed over the full dataset configured in params.yaml.',
    section_wordcloud: 'Dynamic Word Cloud',
    section_explorer: 'Dynamic Explorer',
    section_explorer_hint: 'Apply filters to explore dataset subsets.',
    col_explorer_title: 'Column Explorer',
    col_explorer_hint: 'Click on a column to see its detailed profile.',
    total_incidents: 'Total Incidents', period: 'Period',
    eda_kpi_rows_sub: 'rows in the dataset',
    eda_kpi_cols: 'Columns', eda_kpi_cols_sub: 'columns in the dataset',
    eda_kpi_missing_sub: 'cells with null value',
    eda_kpi_dups: 'Duplicate rows', eda_kpi_dups_sub: 'identical rows across all columns',
    col_search_placeholder: 'Search column...',
    type_lbl: 'Type', unique_lbl: 'Unique', nulls_lbl: 'Nulls',
    mean_lbl: 'Mean', median_lbl: 'Median', std_lbl: 'Std. dev.', minmax_lbl: 'Min / Max',
    numeric_lbl: 'Numeric', categorical_lbl: 'Categorical', multilabel_lbl: 'Multilabel',
    frequency: 'Frequency', incidents: 'Incidents', year_lbl: 'Year',
    month_lbl: 'Month', cum_pct: 'Cumulative %', severity_lbl: 'Avg. Severity',
    neg_score: 'Negativity score', pct_total: '% of total',
    select_col: '— Select a column —',
    add_filter: 'Add filter', clear_all: 'Clear all',
    no_filters: 'No filters — showing all records.',
    no_data: 'No data to display',
    wc_col_label: 'Text column', wc_filter_label: 'Filter rows by column',
    wc_generate: 'Generate cloud',
    wc_placeholder: 'Select a column and click "Generate cloud".',
    chart_revol: 'Regional Evolution', chart_cumul: 'Cumulative Concentration',
    chart_harmtype: 'Harm Types by Region', chart_princip: 'Principles / Risk Domains',
    chart_indust: 'Top Industries', chart_harmed: 'Affected Groups by Region',
    chart_vulner: 'Vulnerable Groups', chart_chrono: 'Harm Chronology',
    chart_sev: 'Severity Index', chart_mh: 'Mental Health Incidents',
    chart_neg: 'Negativity by Principle', chart_tok: 'Most Frequent Tokens',
    records_of: 'records', of_total: 'of total',
    col_summary_title: 'Column Summary',
    col_completeness: 'Completeness by Column',
    col_completeness_hint: '% of records with non-null value.',
    th_col: 'Column', th_type: 'Type', th_records: 'Records',
    th_unique: 'Unique', th_nulls: 'Nulls',
    completeness_lbl: 'Completeness (%)',
    chart_histogram: 'Histogram', chart_boxplot: 'Box Plot',
    chart_freq: 'Frequency', chart_by_region: 'By Region', chart_trend: 'Over Time',
    section_scatter: 'Scatter Plot', section_scatter_hint: 'Select two numeric columns to explore their relationship.',
    scatter_x: 'X Axis', scatter_y: 'Y Axis', scatter_color: 'Color by',
    section_crosstab: 'Cross-table', section_crosstab_hint: 'Select two categorical columns to see co-occurrence.',
    crosstab_row: 'Rows', crosstab_col: 'Columns',
    btn_generate: 'Generate', none_col: '(no color)',
    af_year_range: 'Year range', af_regions: 'Regions', af_clear: 'Clear filters',
    af_active: 'Active filters',
  }},
}};
let currentLang = 'es';
const t = (key) => LANGS[currentLang][key] || key;

function setLang(lang) {{
  currentLang = lang;
  document.getElementById('btn-es').classList.toggle('active', lang==='es');
  document.getElementById('btn-en').classList.toggle('active', lang==='en');
  // Update all data-i18n elements
  document.querySelectorAll('[data-i18n]').forEach(el => {{
    const k = el.dataset.i18n;
    if (LANGS[lang][k]) el.textContent = LANGS[lang][k];
  }});
  document.querySelectorAll('[data-i18n-opt]').forEach(el => {{
    const k = el.dataset.i18nOpt;
    if (LANGS[lang][k]) el.textContent = LANGS[lang][k];
  }});
  // Re-render everything that has language-dependent labels
  const analysisVisible = document.getElementById('tab-analysis').style.display !== 'none';
  if (analysisVisible) {{
    renderAnalysisCards();
    document.getElementById('metrics-charts').dataset.rendered = '';
    renderMetricsCharts();
    renderDynamicCharts();
  }}
  initEda();  // refresh EDA type badges
}}

/* ─── Plotly config (remove useless toolbar buttons) ────────────────── */
const PCFG = {{
  responsive: true,
  modeBarButtonsToRemove: ['lasso2d', 'select2d'],
  displaylogo: false,
}};

/* ─── Tab switching ─────────────────────────────────────────────────── */
function showTab(id, btn) {{
  document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + id).style.display = 'block';
  btn.classList.add('active');
  if (id === 'analysis') {{
    renderAnalysisCards();
    initAnalysisFilters();
    renderMetricsCharts();
    renderDynamicCharts();
    initWcPanel();
  }}
}}

/* ════════════════════════════════════════════════════════════════════════
   EDA TAB
   ════════════════════════════════════════════════════════════════════════ */
function initEda() {{
  const s = EDA_STATS;
  document.getElementById('eda-overview-cards').innerHTML = [
    {{ icon:'🗂️', value: s.n_rows.toLocaleString(),       label: t('total_incidents'), sub: t('eda_kpi_rows_sub') }},
    {{ icon:'📋', value: s.n_cols.toLocaleString(),          label: t('eda_kpi_cols'),    sub: t('eda_kpi_cols_sub') }},
    {{ icon:'❓', value: s.missing_pct + '%',               label: 'Missing',            sub: s.missing_cells.toLocaleString() + ' ' + t('eda_kpi_missing_sub') }},
    {{ icon:'📄', value: s.duplicates.toLocaleString(),    label: t('eda_kpi_dups'),    sub: t('eda_kpi_dups_sub') }},
  ].map(c => `<div class="stat-card"><span class="stat-icon">${{c.icon}}</span>
    <span class="stat-value">${{c.value}}</span>
    <span class="stat-label">${{c.label}}</span>
    <span class="stat-sub">${{c.sub}}</span></div>`).join('');

  renderCompletenessChart();
  renderColSummaryTable();

  const container = document.getElementById('col-cards');
  container.innerHTML = s.columns.map(col => {{
    const cls   = col.col_type==='numeric' ? 'badge-num' : col.col_type==='list' ? 'badge-list' : 'badge-cat';
    const label = col.col_type==='numeric' ? t('numeric_lbl') : col.col_type==='list' ? t('multilabel_lbl') : t('categorical_lbl');
    return `<div class="col-card" onclick="showColDetail('${{esc(col.name)}}')" id="colcard-${{CSS.escape(col.name)}}">
      <div class="col-card-name">${{col.name}}</div>
      <span class="type-badge ${{cls}}">${{label}}</span>
      <div class="col-card-stats">
        <span>${{t('unique_lbl')}}: <b>${{col.unique_count}}</b></span>
        <span>${{t('nulls_lbl')}}: <b>${{col.null_pct}}%</b></span>
      </div>
      <div class="null-bar"><div class="null-fill" style="width:${{col.null_pct}}%"></div></div>
    </div>`;
  }}).join('');

  initScatterPanel();
  initCrosstabPanel();
}}

function renderCompletenessChart() {{
  const cols = EDA_STATS.columns;
  const sorted = [...cols].sort((a,b) => (100 - b.null_pct) - (100 - a.null_pct));
  const completeness = sorted.map(c => +(100 - c.null_pct).toFixed(1));
  const colors = completeness.map(v => v >= 90 ? '#4ff7a6' : v >= 70 ? '#f7964f' : '#f74f6e');
  Plotly.newPlot('eda-completeness', [{{
    x: completeness,
    y: sorted.map(c => trunc(c.name, 40)),
    type: 'bar', orientation: 'h',
    marker: {{ color: colors }},
    text: completeness.map(v => v + '%'),
    textposition: 'outside',
    cliponaxis: false,
    hovertemplate: '%{{y}}: %{{x}}%<extra></extra>',
  }}], {{
    ...BL(t('completeness_lbl'), ''),
    margin: {{t:10, b:40, l:180, r:60}},
    xaxis: {{range:[0,110], ticksuffix:'%', showgrid:true}},
    yaxis: {{automargin:true}},
  }}, PCFG);
}}

function renderColSummaryTable() {{
  const cols = EDA_STATS.columns;
  const typeLabel = tp => tp==='numeric' ? t('numeric_lbl') : tp==='list' ? t('multilabel_lbl') : t('categorical_lbl');
  const typeCls   = tp => tp==='numeric' ? 'badge-num' : tp==='list' ? 'badge-list' : 'badge-cat';
  const rows = cols.map(c => `<tr>
    <td class="st-name">${{c.name}}</td>
    <td><span class="type-badge ${{typeCls(c.col_type)}}">${{typeLabel(c.col_type)}}</span></td>
    <td class="st-num">${{EDA_STATS.n_rows.toLocaleString()}}</td>
    <td class="st-num">${{c.unique_count.toLocaleString()}}</td>
    <td class="st-num ${{c.null_pct > 50 ? 'st-warn' : c.null_pct > 10 ? 'st-caution' : ''}}">${{c.null_count.toLocaleString()}} <span class="st-pct">(${{c.null_pct}}%)</span></td>
  </tr>`).join('');
  document.getElementById('col-summary-table').innerHTML = `
    <table class="summary-table">
      <thead><tr>
        <th data-i18n="th_col">${{t('th_col')}}</th>
        <th data-i18n="th_type">${{t('th_type')}}</th>
        <th data-i18n="th_records">${{t('th_records')}}</th>
        <th data-i18n="th_unique">${{t('th_unique')}}</th>
        <th data-i18n="th_nulls">${{t('th_nulls')}}</th>
      </tr></thead>
      <tbody>${{rows}}</tbody>
    </table>`;
}}

/* ── Column detail chart type toggle ────────────────────────────────── */
let _curColName = null;
let _curChartType = null;

function showColDetail(colName) {{
  _curColName = colName;
  document.querySelectorAll('.col-card').forEach(c => c.classList.remove('selected'));
  const card = document.getElementById('colcard-' + CSS.escape(colName));
  if (card) card.classList.add('selected');
  const col = EDA_STATS.columns.find(c => c.name === colName);
  if (!col) return;
  const detail = document.getElementById('col-detail');
  detail.classList.remove('hidden');

  let statsHtml = `<h3 class="detail-title">${{col.name}}</h3><div class="detail-stats">`;
  statsHtml += dsRow(t('type_lbl'), col.col_type) + dsRow(t('unique_lbl'), col.unique_count.toLocaleString())
             + dsRow(t('nulls_lbl'), `${{col.null_count.toLocaleString()}} (${{col.null_pct}}%)`);
  if (col.col_type === 'numeric') {{
    statsHtml += dsRow(t('mean_lbl'), col.mean ?? '—') + dsRow(t('median_lbl'), col.median ?? '—')
               + dsRow(t('std_lbl'), col.std ?? '—') + dsRow(t('minmax_lbl'), `${{col.min}} / ${{col.max}}`);
  }}
  statsHtml += '</div>';
  document.getElementById('col-detail-stats').innerHTML = statsHtml;

  // Chart type buttons
  const btns = col.col_type === 'numeric'
    ? [['hist', t('chart_histogram')], ['box', t('chart_boxplot')]]
    : [['freq', t('chart_freq')], ['region', t('chart_by_region')], ['trend', t('chart_trend')]];
  const defaultType = btns[0][0];
  document.getElementById('col-chart-type-btns').innerHTML = btns.map(([tp, lbl]) =>
    `<button class="chart-type-btn${{tp===defaultType?' active':''}}" onclick="renderColChart('${{esc(colName)}}','${{tp}}',this)">${{lbl}}</button>`
  ).join('');

  document.getElementById('col-detail-chart').innerHTML = '<div id="eda-chart" style="height:300px"></div>';
  renderColChart(colName, defaultType, null);
  detail.scrollIntoView({{behavior:'smooth', block:'nearest'}});
}}

function renderColChart(colName, chartType, btn) {{
  if (btn) {{
    document.querySelectorAll('.chart-type-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  }}
  _curChartType = chartType;
  const col = EDA_STATS.columns.find(c => c.name === colName);
  if (!col) return;
  document.getElementById('col-detail-chart').innerHTML = '<div id="eda-chart" style="height:300px"></div>';

  if (chartType === 'hist' && col.col_type === 'numeric' && col.hist_counts) {{
    const mid = col.hist_edges.slice(0,-1).map((e,i) => (e + col.hist_edges[i+1])/2);
    Plotly.newPlot('eda-chart', [{{ x: mid, y: col.hist_counts, type:'bar', marker:{{color:'#4f8ef7'}},
      hovertemplate: col.name + ': %{{x:,}}<br>' + t('frequency') + ': %{{y:,}}<extra></extra>' }}],
      {{ ...BL(col.name, t('frequency')), margin:{{t:10,b:50,l:60,r:20}} }}, PCFG);

  }} else if (chartType === 'box' && col.col_type === 'numeric') {{
    const vals = DATA_RECORDS.map(r => r[colName]).filter(v => v != null && !isNaN(+v)).map(Number);
    Plotly.newPlot('eda-chart', [{{ y: vals, type:'box', name: colName,
      marker:{{color:'#4f8ef7'}}, boxpoints:'outliers', jitter:0.3, pointpos:-1.8,
      hovertemplate: '%{{y:,}}<extra></extra>' }}],
      {{ ...BL('', col.name), margin:{{t:10,b:40,l:70,r:20}} }}, PCFG);

  }} else if (chartType === 'freq' && col.top_values) {{
    Plotly.newPlot('eda-chart', [{{ x: col.top_counts.slice().reverse(),
      y: col.top_values.slice().reverse().map(v => trunc(v, 50)),
      type:'bar', orientation:'h', marker:{{color:'#4f8ef7'}},
      customdata: col.top_values.slice().reverse(),
      hovertemplate: '%{{customdata}}: %{{x:,}}<extra></extra>' }}],
      {{ ...BL(t('frequency'), ''), margin:{{t:10,b:40,l:210,r:20}}, yaxis:{{automargin:true}} }}, PCFG);

  }} else if (chartType === 'region') {{
    const regionCol = COL_ROLES.region;
    if (!regionCol) {{ document.getElementById('eda-chart').innerHTML = noData(); return; }}
    const counts = {{}};
    DATA_RECORDS.forEach(r => {{
      const rg = r[regionCol]; if (!rg) return;
      const v  = r[colName];   if (v == null) return;
      (Array.isArray(v) ? v : [v]).forEach(val => {{
        const k = String(val);
        if (!counts[k]) counts[k] = {{}};
        counts[k][String(rg)] = (counts[k][String(rg)] || 0) + 1;
      }});
    }});
    const top10 = Object.entries(counts)
      .sort((a,b) => Object.values(b[1]).reduce((s,v)=>s+v,0) - Object.values(a[1]).reduce((s,v)=>s+v,0))
      .slice(0, 10);
    const regions = [...new Set(DATA_RECORDS.map(r=>r[regionCol]).filter(Boolean).map(String))].sort();
    Plotly.newPlot('eda-chart', regions.map((rg,i) => ({{
      x: top10.map(([k]) => trunc(k, 30)),
      y: top10.map(([k,v]) => v[rg]||0),
      name: rg, type:'bar', marker:{{color:C[i%C.length]}},
      customdata: top10.map(([k])=>k),
      hovertemplate: rg + ' – %{{customdata}}: %{{y:,}}<extra></extra>'
    }})), {{
      ...BL(colName, t('frequency')), barmode:'stack',
      showlegend:true, legend:{{orientation:'h',y:-0.28,x:0.5,xanchor:'center',font:{{size:9}}}},
      margin:{{t:10,b:90,l:60,r:20}}, xaxis:{{tickangle:-30}}
    }}, PCFG);

  }} else if (chartType === 'trend') {{
    const yearCol = COL_ROLES.year;
    if (!yearCol) {{ document.getElementById('eda-chart').innerHTML = noData(); return; }}
    const byCY = {{}};
    const years = new Set();
    DATA_RECORDS.forEach(r => {{
      const y = r[yearCol]; if (!y) return;
      years.add(y);
      const v = r[colName]; if (v == null) return;
      (Array.isArray(v) ? v : [v]).forEach(val => {{
        const k = String(val);
        if (!byCY[k]) byCY[k] = {{}};
        byCY[k][y] = (byCY[k][y]||0) + 1;
      }});
    }});
    const sortedYrs = [...years].sort();
    const top8 = Object.entries(byCY)
      .sort((a,b) => Object.values(b[1]).reduce((s,v)=>s+v,0) - Object.values(a[1]).reduce((s,v)=>s+v,0))
      .slice(0, 8);
    Plotly.newPlot('eda-chart', top8.map(([cat, byY], i) => ({{
      x: sortedYrs, y: sortedYrs.map(y => byY[y]||0),
      name: trunc(cat,25), type:'scatter', mode:'lines+markers',
      line:{{color:C[i%C.length],width:1.5}}, marker:{{size:4}},
      hovertemplate: cat + '<br>%{{x}}: %{{y:,}}<extra></extra>'
    }})), {{
      ...BL(t('year_lbl'), t('frequency')),
      showlegend:true, legend:{{orientation:'h',y:-0.28,x:0.5,xanchor:'center',font:{{size:9}}}},
      margin:{{t:10,b:90,l:60,r:20}}
    }}, PCFG);
    addLegendTooltips('eda-chart', top8.map(([k])=>k));
  }}
}}

/* ── Scatter Plot Explorer ──────────────────────────────────────────── */
function initScatterPanel() {{
  const numCols = EDA_STATS.columns.filter(c => c.col_type === 'numeric').map(c => c.name);
  const catCols = EDA_STATS.columns.filter(c => c.col_type !== 'numeric').map(c => c.name);
  const xSel = document.getElementById('scatter-x');
  const ySel = document.getElementById('scatter-y');
  const cSel = document.getElementById('scatter-color');
  numCols.forEach((c,i) => {{
    [xSel, ySel].forEach(sel => {{
      const o = document.createElement('option');
      o.value = c; o.textContent = colLabel(c);
      sel.appendChild(o);
    }});
  }});
  // Try sensible defaults
  if (numCols.length >= 2) {{ xSel.value = numCols[0]; ySel.value = numCols[1]; }}
  const none = document.createElement('option'); none.value = ''; none.textContent = t('none_col');
  cSel.appendChild(none);
  catCols.forEach(c => {{
    const o = document.createElement('option');
    o.value = c; o.textContent = colLabel(c);
    cSel.appendChild(o);
  }});
}}

function renderScatter() {{
  const xCol = document.getElementById('scatter-x').value;
  const yCol = document.getElementById('scatter-y').value;
  const cCol = document.getElementById('scatter-color').value;
  if (!xCol || !yCol) return;
  const el = document.getElementById('scatter-chart');
  const rows = DATA_RECORDS.filter(r => r[xCol] != null && r[yCol] != null && !isNaN(+r[xCol]) && !isNaN(+r[yCol]));

  if (cCol) {{
    const groups = {{}};
    rows.forEach(r => {{ const g = String(r[cCol]??'—'); if(!groups[g]) groups[g]=[]; groups[g].push(r); }});
    const traces = Object.entries(groups).slice(0,12).map(([g,rs],i) => ({{
      x: rs.map(r=>+r[xCol]), y: rs.map(r=>+r[yCol]), name: trunc(g,25),
      type:'scatter', mode:'markers', marker:{{color:C[i%C.length],size:6,opacity:0.7}},
      hovertemplate: g+'<br>'+xCol+': %{{x:,}}<br>'+yCol+': %{{y:,}}<extra></extra>'
    }}));
    Plotly.newPlot('scatter-chart', traces, {{
      ...BL(colLabel(xCol), colLabel(yCol)), showlegend:true,
      legend:{{orientation:'h',y:-0.2,x:0.5,xanchor:'center',font:{{size:9}}}},
      margin:{{t:10,b:90,l:70,r:20}}
    }}, PCFG);
    addLegendTooltips('scatter-chart', Object.keys(groups).slice(0,12));
  }} else {{
    Plotly.newPlot('scatter-chart', [{{
      x: rows.map(r=>+r[xCol]), y: rows.map(r=>+r[yCol]),
      type:'scatter', mode:'markers', marker:{{color:'#4f8ef7',size:6,opacity:0.7}},
      hovertemplate: xCol+': %{{x:,}}<br>'+yCol+': %{{y:,}}<extra></extra>'
    }}], {{ ...BL(colLabel(xCol), colLabel(yCol)), margin:{{t:10,b:40,l:70,r:20}} }}, PCFG);
  }}
}}

/* ── Crosstab Heatmap ───────────────────────────────────────────────── */
function initCrosstabPanel() {{
  const cols = EDA_STATS.columns.filter(c => c.col_type !== 'numeric' && c.unique_count <= 30).map(c => c.name);
  ['cross-row','cross-col'].forEach(id => {{
    const sel = document.getElementById(id);
    cols.forEach(c => {{
      const o = document.createElement('option');
      o.value = c; o.textContent = colLabel(c);
      sel.appendChild(o);
    }});
  }});
  if (cols.length >= 2) {{
    document.getElementById('cross-row').value = cols[0];
    document.getElementById('cross-col').value = cols.length > 1 ? cols[1] : cols[0];
  }}
}}

function renderCrosstab() {{
  const rowCol = document.getElementById('cross-row').value;
  const colCol = document.getElementById('cross-col').value;
  if (!rowCol || !colCol) return;

  const matrix = {{}};
  const rowVals = new Set(), colVals = new Set();
  DATA_RECORDS.forEach(r => {{
    const rv = r[rowCol]; const cv = r[colCol];
    if (rv == null || cv == null) return;
    const rvs = Array.isArray(rv) ? rv : [rv];
    const cvs = Array.isArray(cv) ? cv : [cv];
    rvs.forEach(rk => {{
      cvs.forEach(ck => {{
        const rk2=String(rk), ck2=String(ck);
        rowVals.add(rk2); colVals.add(ck2);
        if (!matrix[rk2]) matrix[rk2] = {{}};
        matrix[rk2][ck2] = (matrix[rk2][ck2]||0) + 1;
      }});
    }});
  }});

  const rows = [...rowVals].sort((a,b)=>a.localeCompare(b,undefined,{{numeric:true}})).slice(0,20);
  const cols = [...colVals].sort((a,b)=>a.localeCompare(b,undefined,{{numeric:true}})).slice(0,20);
  const z = rows.map(r => cols.map(c => matrix[r]?.[c] || 0));
  const h = Math.max(300, rows.length * 28 + 80);
  document.getElementById('crosstab-chart').style.height = h + 'px';

  Plotly.newPlot('crosstab-chart', [{{
    z, x: cols.map(c=>trunc(c,20)), y: rows.map(r=>trunc(r,20)),
    type:'heatmap', colorscale:'Blues', showscale:true,
    hovertemplate: rowCol+': %{{y}}<br>'+colCol+': %{{x}}<br>Count: %{{z:,}}<extra></extra>',
    zmin:0
  }}], {{
    ...BL(colLabel(colCol), colLabel(rowCol)),
    margin:{{t:10,b:120,l:160,r:60}},
    xaxis:{{tickangle:-35}}, yaxis:{{automargin:true}}
  }}, PCFG);
}}

/* ════════════════════════════════════════════════════════════════════════
   ANALYSIS — Filter bar (year range + region checkboxes)
   ════════════════════════════════════════════════════════════════════════ */
let ANALYSIS_FILTERS = {{ yearMin: null, yearMax: null, regions: [] }};

function initAnalysisFilters() {{
  // Populate year inputs with dataset range as placeholder
  const allYears = (METRICS.regional_evolution || []).map(r => +r.year).filter(Boolean);
  if (allYears.length) {{
    const minY = Math.min(...allYears), maxY = Math.max(...allYears);
    const minEl = document.getElementById('af-year-min');
    const maxEl = document.getElementById('af-year-max');
    minEl.min = minY; minEl.max = maxY; minEl.placeholder = minY;
    maxEl.min = minY; maxEl.max = maxY; maxEl.placeholder = maxY;
  }}

  // Populate region checkboxes (only once)
  const regBox = document.getElementById('af-regions');
  if (regBox.children.length) return;
  const allRegions = (METRICS.summary?.regions || []).slice().sort();
  if (!allRegions.length) {{
    regBox.closest('.af-group').style.display = 'none';
    return;
  }}
  regBox.innerHTML = allRegions.map(r =>
    `<label class="af-check-label">
       <input type="checkbox" value="${{r}}" onchange="applyAnalysisFilters()"> ${{r}}
     </label>`
  ).join('');
}}

function applyAnalysisFilters() {{
  const minEl = document.getElementById('af-year-min');
  const maxEl = document.getElementById('af-year-max');
  ANALYSIS_FILTERS.yearMin = minEl.value ? +minEl.value : null;
  ANALYSIS_FILTERS.yearMax = maxEl.value ? +maxEl.value : null;
  ANALYSIS_FILTERS.regions = [...document.querySelectorAll('#af-regions input:checked')].map(el => el.value);
  // Invalidate cache and re-render
  document.getElementById('metrics-charts').dataset.rendered = '';
  renderMetricsCharts();
}}

function clearAnalysisFilters() {{
  document.getElementById('af-year-min').value = '';
  document.getElementById('af-year-max').value = '';
  document.querySelectorAll('#af-regions input').forEach(el => el.checked = false);
  ANALYSIS_FILTERS = {{ yearMin: null, yearMax: null, regions: [] }};
  document.getElementById('metrics-charts').dataset.rendered = '';
  renderMetricsCharts();
}}

/* ── Filter helpers ─────────────────────────────────────────────────── */
// Filter an array of rows by year field
function _afYears(rows, field) {{
  const {{ yearMin, yearMax }} = ANALYSIS_FILTERS;
  if (yearMin == null && yearMax == null) return rows;
  field = field || 'year';
  return rows.filter(r => {{
    const y = +(String(r[field] || '').slice(0, 4));
    return (!yearMin || y >= yearMin) && (!yearMax || y <= yearMax);
  }});
}}

// From a list of column names, keep only those that match active regions
function _afRegionCols(cols) {{
  const {{ regions }} = ANALYSIS_FILTERS;
  return regions.length ? cols.filter(c => regions.includes(c)) : cols;
}}

// Filter rows whose `field` value is in active regions
function _afRegionRows(rows, field) {{
  const {{ regions }} = ANALYSIS_FILTERS;
  return regions.length ? rows.filter(r => regions.includes(String(r[field] || ''))) : rows;
}}

/* ════════════════════════════════════════════════════════════════════════
   ANALYSIS — Summary KPI cards  (only stable non-hardcoded cards)
   ════════════════════════════════════════════════════════════════════════ */
function renderAnalysisCards() {{
  const s   = METRICS.summary || {{}};
  if (!s.n_incidents) return;
  const yr  = s.year_range || [];
  const kl  = HTML_CONFIG.kpi_labels || {{}};
  const cards = [
    {{ id:'kpi-incidents', icon:'📁', label: kl.n_incidents || t('total_incidents'), value: s.n_incidents.toLocaleString() }},
    {{ id:'kpi-period',    icon:'📅', label: kl.period      || t('period'),           value: yr.length===2 ? yr[0]+' – '+yr[1] : '—' }},
  ];
  document.getElementById('analysis-cards').innerHTML = cards.map(c =>
    `<div class="stat-card">
       <span class="stat-icon">${{c.icon}}</span>
       <span class="stat-value">${{c.value}}</span>
       <span class="stat-label editable-title" contenteditable="true"
             title="Click para editar"
             onblur="this.textContent=this.textContent.trim()||'—'"
             onkeydown="if(event.key==='Enter'){{this.blur();event.preventDefault()}}"
       >${{c.label}}</span>
     </div>`
  ).join('');
}}

/* ════════════════════════════════════════════════════════════════════════
   ANALYSIS — Pre-computed pipeline charts  (use METRICS, no re-filter)
   ════════════════════════════════════════════════════════════════════════ */
function renderMetricsCharts() {{
  const grid = document.getElementById('metrics-charts');
  if (grid.dataset.rendered === currentLang) return;
  grid.dataset.rendered = currentLang;

  const specs = [
    {{ id:'mc-revol',    title: t('chart_revol'),    fn: chartRegionalEvolution }},
    {{ id:'mc-cumul',    title: t('chart_cumul'),    fn: chartCumulative }},
    {{ id:'mc-harmtype', title: t('chart_harmtype'), fn: chartHarmTypes }},
    {{ id:'mc-princip',  title: t('chart_princip'),  fn: chartPrinciples }},
    {{ id:'mc-indust',   title: t('chart_indust'),   fn: chartIndustries }},
    {{ id:'mc-harmed',   title: t('chart_harmed'),   fn: chartStakeholders }},
    {{ id:'mc-vulner',   title: t('chart_vulner'),   fn: chartVulnerable }},
    {{ id:'mc-chrono',   title: t('chart_chrono'),   fn: chartHarmChronology }},
    {{ id:'mc-sev',      title: t('chart_sev'),      fn: chartSeverity }},
    {{ id:'mc-mh',       title: t('chart_mh'),       fn: chartMentalHealth }},
    {{ id:'mc-neg',      title: t('chart_neg'),      fn: chartNegativity }},
    {{ id:'mc-tok',      title: t('chart_tok'),      fn: chartTopTokens }},
  ];

  grid.innerHTML = specs.map(s =>
    `<div class="chart-card">
       <div class="chart-title editable-title"
            contenteditable="true"
            data-chart-id="${{s.id}}"
            title="Click para editar — el título aparecerá en la exportación PNG"
            onblur="syncChartTitle(this)"
            onkeydown="if(event.key==='Enter'){{this.blur();event.preventDefault()}}"
       >${{s.title}}</div>
       <div id="${{s.id}}" class="chart-area"></div>
     </div>`
  ).join('');
  specs.forEach(s => {{
    try {{ s.fn(s.id); }}
    catch(e) {{ document.getElementById(s.id).innerHTML = noData(); console.warn(s.id, e); }}
  }});
}}

/* ── Individual chart renderers (use METRICS, language-aware) ──────── */
function chartRegionalEvolution(id) {{
  const raw = METRICS.regional_evolution || [];
  if (!raw.length) {{ noDom(id); return; }}
  const rows = _afYears(raw, 'year');
  if (!rows.length) {{ noDom(id); return; }}
  const allRegions = Object.keys(raw[0]).filter(k => k !== 'year');
  const regions = _afRegionCols(allRegions);
  if (!regions.length) {{ noDom(id); return; }}
  Plotly.newPlot(id, regions.map((rg,i) => ({{
    x: rows.map(r=>r.year), y: rows.map(r=>r[rg]||0), name: rg, type:'bar',
    marker:{{color: C[i%C.length]}},
    hovertemplate: rg + '<br>' + t('year_lbl') + ': %{{x}}<br>' + t('incidents') + ': %{{y:,}}<extra></extra>'
  }})), {{
    ...BL(t('year_lbl'), t('incidents')), barmode:'stack',
    showlegend:true, legend:{{orientation:'h',y:-0.32,x:0.5,xanchor:'center',font:{{size:10}}}},
    margin:{{t:10,b:110,l:60,r:20}}
  }}, PCFG);
  addLegendTooltips(id, regions);
}}

function chartCumulative(id) {{
  const rows = _afYears(METRICS.cumulative_concentration || [], 'year');
  if (!rows.length) {{ noDom(id); return; }}
  Plotly.newPlot(id, [{{
    x: rows.map(r=>r.year), y: rows.map(r=>r.cumulative_pct),
    type:'scatter', mode:'lines+markers',
    line:{{color:'#4f8ef7',width:2}}, marker:{{size:5}},
    fill:'tozeroy', fillcolor:'rgba(79,142,247,0.08)',
    hovertemplate: t('year_lbl') + ': %{{x}}<br>' + t('cum_pct') + ': %{{y:.1f}}%<extra></extra>'
  }}], {{ ...BL(t('year_lbl'), t('cum_pct')), margin:{{t:10,b:40,l:70,r:20}} }}, PCFG);
}}

function chartHarmTypes(id) {{
  const raw = METRICS.harm_types_distribution || [];
  if (!raw.length) {{ noDom(id); return; }}
  const regionKey = Object.keys(raw[0])[0];
  const rows = _afRegionRows(raw, regionKey);
  if (!rows.length) {{ noDom(id); return; }}
  const hts   = Object.keys(raw[0]).filter(k => k !== regionKey);
  const regs  = rows.map(r => String(r[regionKey]));
  const lgHt = legendBottom(hts.length, regs.length);
  const htEl = document.getElementById(id);
  if (htEl) htEl.style.height = lgHt.height + 'px';
  Plotly.newPlot(id, hts.map((ht,i) => ({{
    x: rows.map(r=>r[ht]||0), y: regs, name: trunc(ht,30),
    type:'bar', orientation:'h', marker:{{color: C[i%C.length]}},
    customdata: hts.map(() => ht),
    hovertemplate: ht + '<br>%{{y}}: %{{x:,}}<extra></extra>'
  }})), {{
    ...BL('', ''), barmode:'stack',
    showlegend:true,
    legend:{{orientation:'h', y:lgHt.y, yanchor:'top', x:0.5, xanchor:'center', font:{{size:8}}}},
    margin:{{t:10, b:lgHt.b, l:120, r:20}}, yaxis:{{automargin:true}}
  }}, PCFG);
  addLegendTooltips(id, hts);
}}

function chartPrinciples(id) {{
  const raw = METRICS.principles_distribution || [];
  if (!raw.length) {{ noDom(id); return; }}
  const rk = Object.keys(raw[0])[0];
  const rows = _afRegionRows(raw, rk);
  if (!rows.length) {{ noDom(id); return; }}
  const ps = Object.keys(raw[0]).filter(k=>k!==rk);
  const sorted = ps.map(p=>[p, rows.reduce((s,r)=>s+(r[p]||0),0)]).sort((a,b)=>a[1]-b[1]);
  Plotly.newPlot(id, [{{
    x: sorted.map(([,v])=>v), y: sorted.map(([k])=>trunc(k,45)),
    type:'bar', orientation:'h', marker:{{color:'#4f8ef7'}},
    customdata: sorted.map(([k])=>k),
    hovertemplate: '%{{customdata}}: %{{x:,}}<extra></extra>'
  }}], {{ ...BL(t('incidents'),''), margin:{{t:10,b:40,l:220,r:20}}, yaxis:{{automargin:true}} }}, PCFG);
}}

function chartIndustries(id) {{
  const rows = (METRICS.top_industries||[]).slice().reverse();
  if (!rows.length) {{ noDom(id); return; }}
  Plotly.newPlot(id, [{{
    x: rows.map(r=>r.count), y: rows.map(r=>trunc(String(r.industry),45)),
    type:'bar', orientation:'h', marker:{{color:'#4fbcf7'}},
    customdata: rows.map(r=>String(r.industry)),
    hovertemplate: '%{{customdata}}: %{{x:,}}<extra></extra>'
  }}], {{ ...BL(t('incidents'),''), margin:{{t:10,b:40,l:210,r:20}}, yaxis:{{automargin:true}} }}, PCFG);
}}

function chartStakeholders(id) {{
  const raw = METRICS.stakeholders_distribution || [];
  if (!raw.length) {{ noDom(id); return; }}
  const rk = Object.keys(raw[0])[0];
  const rows = _afRegionRows(raw, rk);
  if (!rows.length) {{ noDom(id); return; }}
  const gs = Object.keys(raw[0]).filter(k=>k!==rk);
  const regs = rows.map(r=>String(r[rk]));
  const lgSt = legendBottom(gs.length, regs.length);
  const stEl = document.getElementById(id);
  if (stEl) stEl.style.height = lgSt.height + 'px';
  Plotly.newPlot(id, gs.map((g,i)=>({{
    x: rows.map(r=>r[g]||0), y: regs, name: trunc(g,25),
    type:'bar', orientation:'h', marker:{{color: C[i%C.length]}},
    hovertemplate: g + '<br>%{{y}}: %{{x:,}}<extra></extra>'
  }})), {{
    ...BL('', ''), barmode:'stack',
    showlegend:true,
    legend:{{orientation:'h', y:lgSt.y, yanchor:'top', x:0.5, xanchor:'center', font:{{size:8}}}},
    margin:{{t:10, b:lgSt.b, l:120, r:20}}, yaxis:{{automargin:true}}
  }}, PCFG);
  addLegendTooltips(id, gs);
}}

function chartVulnerable(id) {{
  const rows = [...(METRICS.vulnerable_groups_distribution||[])].sort((a,b)=>a.count-b.count);
  if (!rows.length) {{ noDom(id); return; }}
  Plotly.newPlot(id, [{{
    x: rows.map(r=>r.count), y: rows.map(r=>String(r.group)),
    type:'bar', orientation:'h', marker:{{color:'#a64ff7'}},
    hovertemplate: '%{{y}}: %{{x:,}}<extra></extra>'
  }}], {{ ...BL(t('incidents'),''), margin:{{t:10,b:40,l:170,r:20}}, yaxis:{{automargin:true}} }}, PCFG);
}}

function chartHarmChronology(id) {{
  const rows = _afYears(METRICS.harm_chronology || [], 'year');
  if (!rows.length) {{ noDom(id); return; }}
  const hts = Object.keys(METRICS.harm_chronology[0] || {{}}).filter(k=>k!=='year');
  Plotly.newPlot(id, hts.map((ht,i)=>({{
    x: rows.map(r=>r.year), y: rows.map(r=>r[ht]||0), name: trunc(ht,30),
    type:'scatter', mode:'lines+markers', line:{{color:C[i%C.length],width:2}}, marker:{{size:5}},
    hovertemplate: ht + '<br>' + t('year_lbl') + ': %{{x}}<br>' + t('incidents') + ': %{{y}}<extra></extra>'
  }})), {{
    ...BL(t('year_lbl'),t('incidents')),
    showlegend:true, legend:{{orientation:'h',y:-0.32,x:0.5,xanchor:'center',font:{{size:9}}}},
    margin:{{t:10,b:110,l:60,r:20}}
  }}, PCFG);
  addLegendTooltips(id, hts);
}}

function chartSeverity(id) {{
  const rows = _afYears(METRICS.harm_severity_index || [], 'year');
  if (!rows.length) {{ noDom(id); return; }}
  Plotly.newPlot(id, [{{
    x: rows.map(r=>r.year), y: rows.map(r=>r.avg_severity),
    type:'scatter', mode:'lines+markers',
    line:{{color:'#f7964f',width:2}}, marker:{{size:6}},
    fill:'tozeroy', fillcolor:'rgba(247,150,79,0.08)',
    hovertemplate: t('year_lbl') + ': %{{x}}<br>' + t('severity_lbl') + ': %{{y:.2f}}<extra></extra>'
  }}], {{ ...BL(t('year_lbl'),t('severity_lbl')), margin:{{t:10,b:40,l:80,r:20}} }}, PCFG);
}}

function chartMentalHealth(id) {{
  const rows = _afYears(METRICS.mental_health_summary || [], 'year');
  if (!rows.length) {{ noDom(id); return; }}
  const mhNames = [t('incidents'), t('pct_total')];
  Plotly.newPlot(id, [
    {{ x: rows.map(r=>r.year), y: rows.map(r=>r.n_incidents), name: mhNames[0],
       type:'bar', marker:{{color:'rgba(79,142,247,0.7)'}},
       hovertemplate: t('year_lbl') + ': %{{x}}<br>' + t('incidents') + ': %{{y:,}}<extra></extra>' }},
    {{ x: rows.map(r=>r.year), y: rows.map(r=>+(r.proportion*100).toFixed(1)),
       name: mhNames[1], type:'scatter', mode:'lines+markers', yaxis:'y2',
       line:{{color:'#f74f6e',width:2}}, marker:{{size:5}},
       hovertemplate: t('year_lbl') + ': %{{x}}<br>' + t('pct_total') + ': %{{y}}%<extra></extra>' }}
  ], {{
    ...BL(t('year_lbl'),t('incidents')),
    showlegend:true, legend:{{orientation:'h',y:-0.28,x:0.5,xanchor:'center',font:{{size:10}}}},
    margin:{{t:10,b:90,l:60,r:60}},
    yaxis2:{{title:t('pct_total'), overlaying:'y', side:'right', showgrid:false}}
  }}, PCFG);
  addLegendTooltips(id, mhNames);
}}

function chartNegativity(id) {{
  const rows = _afYears(METRICS.negativity_by_principle || [], 'year_month');
  if (!rows.length) {{ noDom(id); return; }}
  const ps = Object.keys(METRICS.negativity_by_principle[0] || {{}}).filter(k=>k!=='year_month'&&!k.includes('__ma'));
  Plotly.newPlot(id, ps.map((p,i)=>({{
    x: rows.map(r=>r.year_month), y: rows.map(r=>r[p]??null),
    name: trunc(p,25), type:'scatter', mode:'lines', connectgaps:false,
    line:{{color:C[i%C.length],width:1.5}},
    hovertemplate: p + '<br>%{{x}}: %{{y:.3f}}<extra></extra>'
  }})), {{
    ...BL(t('month_lbl'),t('neg_score')),
    showlegend:true, legend:{{orientation:'h',y:-0.35,x:0.5,xanchor:'center',font:{{size:9}}}},
    margin:{{t:10,b:110,l:70,r:20}}, xaxis:{{tickangle:-45,nticks:12}}
  }}, PCFG);
  addLegendTooltips(id, ps);
}}

function chartTopTokens(id) {{
  const rows = [...(METRICS.top_tokens||[])].slice(0,20).sort((a,b)=>a.frequency-b.frequency);
  if (!rows.length) {{ noDom(id); return; }}
  Plotly.newPlot(id, [{{
    x: rows.map(r=>r.frequency), y: rows.map(r=>r.token),
    type:'bar', orientation:'h', marker:{{color:'#4ff7a6'}},
    hovertemplate: '%{{y}}: %{{x:,}}<extra></extra>'
  }}], {{ ...BL(t('frequency'),''), margin:{{t:10,b:40,l:100,r:20}}, yaxis:{{automargin:true}} }}, PCFG);
}}

/* ════════════════════════════════════════════════════════════════════════
   WORDCLOUD — interactive, column-driven
   ════════════════════════════════════════════════════════════════════════ */
/* Return true if column is text/list-of-strings — good for wordcloud */
function _isTextCol(c) {{
  if (c.startsWith('mlb_')) return false;
  const samples = DATA_RECORDS.filter(r => r[c] != null).slice(0, 10);
  if (!samples.length) return false;
  const v = samples[0][c];
  if (Array.isArray(v)) return v.length === 0 || typeof v[0] === 'string';
  return typeof v === 'string';
}}

/* Return true if column is categorical — good for filter checkboxes.
   Excludes mlb_* (one-hot), numeric/boolean, and free-text (avg len > 200). */
function _isCatCol(c) {{
  if (c.startsWith('mlb_')) return false;
  const samples = DATA_RECORDS.filter(r => r[c] != null).slice(0, 20);
  if (!samples.length) return false;
  const v = samples[0][c];
  if (Array.isArray(v)) return v.length === 0 || typeof v[0] === 'string';
  if (typeof v !== 'string') return false;
  // Exclude free-text: avg length > 200 chars (typical for description-type columns)
  const avgLen = samples.reduce((s,r) => s + String(r[c]).length, 0) / samples.length;
  return avgLen <= 200;
}}

function initWcPanel() {{
  if (document.getElementById('wc-col-select').options.length > 1) return;
  const allCols = DATA_RECORDS.length ? Object.keys(DATA_RECORDS[0]) : [];
  // Sort alphabetically by display label
  const sortByLabel = arr => [...arr].sort((a,b) => colLabel(a).localeCompare(colLabel(b)));
  const wcCols  = sortByLabel(allCols.filter(_isTextCol));   // wordcloud: any text/list
  const catCols = sortByLabel(allCols.filter(_isCatCol));    // filter row: categorical only

  // Store for search
  window._wcAllCols  = wcCols;
  window._wcCatCols  = catCols;

  _buildWcColOptions(wcCols);

  const fltSel = document.getElementById('wc-filter-col');
  catCols.forEach(c => {{
    const o = document.createElement('option');
    o.value = c; o.textContent = colLabel(c);
    fltSel.appendChild(o);
  }});

  // Default: try to select tokens column
  const tokCol = COL_ROLES.tokens || 'tokens';
  const colSel = document.getElementById('wc-col-select');
  if (wcCols.includes(tokCol)) colSel.value = tokCol;
  document.getElementById('wc-filter-wrap').classList.remove('hidden');
}}

function _buildWcColOptions(cols) {{
  const colSel = document.getElementById('wc-col-select');
  const cur = colSel.value;
  colSel.innerHTML = `<option value="">${{t('select_col')}}</option>`;
  cols.forEach(c => {{
    const o = document.createElement('option');
    o.value = c;
    o.textContent = colLabel(c);   // show original CSV name (e.g. "description")
    if (c === cur) o.selected = true;
    colSel.appendChild(o);
  }});
}}

function searchWcCol(q) {{
  const cols = window._wcAllCols || [];
  const filtered = q ? cols.filter(c => colLabel(c).toLowerCase().includes(q.toLowerCase())
                                     || c.toLowerCase().includes(q.toLowerCase())) : cols;
  _buildWcColOptions(filtered);
}}

function onWcColChange() {{
  document.getElementById('wc-filter-wrap').classList.remove('hidden');
}}

function onWcFilterColChange() {{
  const col = document.getElementById('wc-filter-col').value;
  const container = document.getElementById('wc-filter-vals');
  if (!col) {{ container.innerHTML = ''; return; }}
  const vals = new Set();
  DATA_RECORDS.forEach(r => {{
    const v = r[col];
    if (Array.isArray(v)) v.forEach(x=>{{ if(x!=null) vals.add(String(x)); }});
    else if (v!=null) vals.add(String(v));
  }});
  const sorted = [...vals].sort((a,b)=>a.localeCompare(b,undefined,{{numeric:true}}));
  container.innerHTML = sorted.map(v =>
    `<label class="val-checkbox"><input type="checkbox" value="${{v.replace(/"/g,'&quot;')}}"> ${{trunc(v,50)}}</label>`
  ).join('');
}}

function generateWordcloud() {{
  const col = document.getElementById('wc-col-select').value;
  if (!col) return;

  // Row filter from wc-filter-col + checked values
  const fCol    = document.getElementById('wc-filter-col').value;
  const fVals   = [...document.querySelectorAll('#wc-filter-vals input:checked')].map(i=>i.value);
  let rows = DATA_RECORDS;
  if (fCol && fVals.length) {{
    rows = rows.filter(r => {{
      const v = r[fCol];
      if (Array.isArray(v)) return fVals.some(fv=>v.includes(fv));
      return fVals.includes(String(v??''));
    }});
  }}

  // Detect column type: list (already tokenized) or raw text (need to split)
  const firstVal = rows.find(r => r[col] != null)?.[col];
  const isListCol = Array.isArray(firstVal);

  // Build word frequencies
  const counts = {{}};
  // Basic stopwords for raw-text mode
  const STOP = new Set(['the','a','an','in','of','to','and','is','was','for','on','with',
    'that','it','at','by','from','as','be','are','were','has','had','have','this',
    'but','or','not','its','their','been','they','he','she','his','her','also','more']);
  rows.forEach(r => {{
    const v = r[col]; if (v==null) return;
    let words;
    if (isListCol) {{
      words = (Array.isArray(v) ? v : [v]).map(x => String(x).trim()).filter(Boolean);
    }} else {{
      // Raw text: lowercase, remove punctuation, split, filter short and stopwords
      words = String(v).toLowerCase()
        .replace(/[^a-z0-9\s]/g, ' ')
        .split(/\s+/)
        .filter(w => w.length > 2 && !STOP.has(w));
    }}
    words.forEach(w => {{ if(w) counts[w] = (counts[w]||0)+1; }});
  }});

  const list = Object.entries(counts).sort((a,b)=>b[1]-a[1]).slice(0,120);
  if (!list.length) {{ document.getElementById('wc-placeholder').textContent = t('no_data'); return; }}

  const canvas = document.getElementById('wc-canvas');
  document.getElementById('wc-placeholder').style.display = 'none';

  // Clear canvas
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const maxFreq = list[0][1];
  WordCloud(canvas, {{
    list: list,
    gridSize: Math.max(4, Math.round(canvas.width / 60)),
    weightFactor: size => Math.max(10, (size / maxFreq) * 72),
    fontFamily: 'system-ui, sans-serif',
    color: () => C[Math.floor(Math.random()*C.length)],
    backgroundColor: '#ffffff',
    rotateRatio: 0.3,
    shuffle: true,
  }});
}}

/* ════════════════════════════════════════════════════════════════════════
   DYNAMIC FILTER EXPLORER
   ════════════════════════════════════════════════════════════════════════ */
let activeFilters = [];
let ALL_FILTER_COLS = [];

function _buildFilterColOptions(cols) {{
  const sel = document.getElementById('filter-col');
  const cur = sel.value;
  sel.innerHTML = `<option value="" data-i18n-opt="select_col">${{t('select_col')}}</option>`;
  cols.forEach(c => {{
    const o = document.createElement('option');
    o.value = c;
    o.textContent = colLabel(c);   // show original CSV name where available
    if (c === cur) o.selected = true;
    sel.appendChild(o);
  }});
}}

function searchFilterCol(q) {{
  const filtered = q
    ? ALL_FILTER_COLS.filter(c => colLabel(c).toLowerCase().includes(q.toLowerCase())
                                || c.toLowerCase().includes(q.toLowerCase()))
    : ALL_FILTER_COLS;
  _buildFilterColOptions(filtered);
}}

function initFilterPanel() {{
  if (!DATA_RECORDS.length) return;
  const allCols = Object.keys(DATA_RECORDS[0]);
  const allowed = HTML_CONFIG.filter_columns || [];
  // Base candidate set: configured columns OR all categorical columns (no mlb_*, no long text)
  const candidates = allCols.filter(_isCatCol);
  const cols = allowed.length ? candidates.filter(c => allowed.includes(c)) : candidates;
  // Also include numeric range columns
  const rangeCols = (HTML_CONFIG.range_filter_columns || []).filter(c => allCols.includes(c) && !cols.includes(c));
  // Sort alphabetically by display label
  ALL_FILTER_COLS = [...cols, ...rangeCols].sort((a,b) => colLabel(a).localeCompare(colLabel(b)));
  _buildFilterColOptions(ALL_FILTER_COLS);
  // Update search placeholder from i18n
  const srch = document.getElementById('filter-col-search');
  if (srch) srch.placeholder = t('col_search_placeholder');
}}

function isRangeCol(col) {{
  return (HTML_CONFIG.range_filter_columns || []).includes(col);
}}

function onFilterColChange() {{
  const col  = document.getElementById('filter-col').value;
  const wrap = document.getElementById('filter-vals-wrap');
  const list = document.getElementById('filter-vals-list');
  if (!col) {{ wrap.classList.add('hidden'); return; }}

  if (isRangeCol(col)) {{
    // Detect min/max from data
    const nums = DATA_RECORDS.map(r => Number(r[col])).filter(v => !isNaN(v));
    const mn = nums.length ? Math.min(...nums) : 0;
    const mx = nums.length ? Math.max(...nums) : 100;
    list.innerHTML = `
      <div class="range-filter-inputs">
        <label>Min <input type="number" id="range-min" value="${{mn}}" min="${{mn}}" max="${{mx}}" style="width:70px;padding:3px 6px;border:1px solid #dde;border-radius:5px"></label>
        <span style="color:#888">—</span>
        <label>Max <input type="number" id="range-max" value="${{mx}}" min="${{mn}}" max="${{mx}}" style="width:70px;padding:3px 6px;border:1px solid #dde;border-radius:5px"></label>
      </div>`;
  }} else {{
    const vals = new Set();
    DATA_RECORDS.forEach(r => {{
      const v = r[col];
      if (Array.isArray(v)) v.forEach(x=>{{ if(x!=null) vals.add(String(x)); }});
      else if (v!=null) vals.add(String(v));
    }});
    const sorted = [...vals].sort((a,b)=>a.localeCompare(b,undefined,{{numeric:true}}));
    list.innerHTML = sorted.map(v =>
      `<label class="val-checkbox"><input type="checkbox" value="${{v.replace(/"/g,'&quot;')}}"> ${{trunc(v,55)}}</label>`
    ).join('');
  }}
  wrap.classList.remove('hidden');
}}

function addFilter() {{
  const col = document.getElementById('filter-col').value;
  if (!col) return;
  activeFilters = activeFilters.filter(f => f.column !== col);

  if (isRangeCol(col)) {{
    const mn = Number(document.getElementById('range-min')?.value);
    const mx = Number(document.getElementById('range-max')?.value);
    if (isNaN(mn) || isNaN(mx)) return;
    activeFilters.push({{column: col, type: 'range', min: mn, max: mx}});
  }} else {{
    const checked = [...document.querySelectorAll('#filter-vals-list input:checked')].map(i=>i.value);
    if (!checked.length) return;
    activeFilters.push({{column: col, type: 'values', values: checked}});
  }}

  renderFilterChips();
  renderDynamicCharts();
  document.getElementById('filter-col').value = '';
  document.getElementById('filter-vals-wrap').classList.add('hidden');
}}

function removeFilter(col) {{
  activeFilters = activeFilters.filter(f=>f.column!==col);
  renderFilterChips(); renderDynamicCharts();
}}

function clearAllFilters() {{
  activeFilters = []; renderFilterChips(); renderDynamicCharts();
}}

function renderFilterChips() {{
  const el = document.getElementById('active-filters');
  if (!activeFilters.length) {{
    el.innerHTML = `<span class="no-filters" data-i18n="no_filters">${{t('no_filters')}}</span>`;
    return;
  }}
  el.innerHTML = activeFilters.map(f => {{
    const label = f.type === 'range'
      ? `${{f.min}} – ${{f.max}}`
      : f.values.slice(0,3).join(', ') + (f.values.length>3 ? ` +${{f.values.length-3}} más` : '');
    return `<span class="chip"><b>${{f.column}}</b>: ${{label}}
      <button class="chip-remove" onclick="removeFilter('${{esc(f.column)}}')" title="Quitar">✕</button></span>`;
  }}).join('') + `<button class="btn-clear" onclick="clearAllFilters()" data-i18n="clear_all">${{t('clear_all')}}</button>`;
}}

function filteredData() {{
  if (!activeFilters.length) return DATA_RECORDS;
  return DATA_RECORDS.filter(row => activeFilters.every(f => {{
    const v = row[f.column];
    if (f.type === 'range') {{
      const n = Number(v);
      return !isNaN(n) && n >= f.min && n <= f.max;
    }}
    if (Array.isArray(v)) return f.values.some(fv => v.includes(fv));
    return f.values.includes(String(v ?? ''));
  }}));
}}

/* ── Prettify column name: "risk_subdomain" → "Risk Subdomain" ─────── */
function prettyCol(col) {{
  return col.replace(/_/g,' ').replace(/\b\w/g, c=>c.toUpperCase());
}}

function renderDynamicCharts() {{
  const rows  = filteredData();
  const total = DATA_RECORDS.length;
  const pct   = total ? Math.round(rows.length/total*100) : 100;
  document.getElementById('record-banner').textContent =
    `${{rows.length.toLocaleString()}} ${{t('records_of')}} (${{pct}}% ${{t('of_total')}})`;

  const grid = document.getElementById('dynamic-charts');
  const specs = [];
  // Each spec title auto-generated from column name, not hardcoded
  if (COL_ROLES.year) {{
    specs.push({{ id:'dc-year', title: prettyCol(COL_ROLES.year) + ' × ' + prettyCol(COL_ROLES.region||''), type:'year_region' }});
  }}
  const barCols = [
    ['region','region'], ['harm_type','harm_type'], ['industries','industries'],
    ['harmed','harmed'], ['tags','tags']
  ];
  barCols.forEach(([role]) => {{
    if (COL_ROLES[role]) specs.push({{ id:'dc-'+role, title: prettyCol(COL_ROLES[role]), type:'bar_count', col:COL_ROLES[role], topN:15 }});
  }});
  if (COL_ROLES.sentiment_score) {{
    specs.push({{ id:'dc-sent', title: prettyCol(COL_ROLES.sentiment_score), type:'hist', col:COL_ROLES.sentiment_score }});
  }}

  grid.innerHTML = specs.map(s =>
    `<div class="chart-card">
       <div class="chart-title editable-title"
            contenteditable="true"
            data-chart-id="${{s.id}}"
            title="Click para editar"
            onblur="syncChartTitle(this)"
            onkeydown="if(event.key==='Enter'){{this.blur();event.preventDefault()}}"
       >${{s.title}}</div>
       <div id="${{s.id}}" class="chart-area"></div>
     </div>`
  ).join('');

  specs.forEach(spec => {{
    try {{
      if (spec.type === 'year_region') {{
        const yrData = countPerYearRegion(rows, COL_ROLES.year, COL_ROLES.region);
        if (!yrData.years.length) {{ noDom(spec.id); return; }}
        Plotly.newPlot(spec.id, yrData.regions.map((rg,i)=>({{
          x: yrData.years, y: yrData.years.map(y=>(yrData.perYearRegion[y]||{{}})[rg]||0),
          name: rg, type:'bar', marker:{{color:C[i%C.length]}},
          hovertemplate: rg+'<br>%{{x}}: %{{y:,}}<extra></extra>'
        }})), {{
          ...BL(t('year_lbl'),t('incidents')), barmode:'stack',
          showlegend:true, legend:{{orientation:'h',y:-0.32,x:0.5,xanchor:'center',font:{{size:10}}}},
          margin:{{t:10,b:110,l:60,r:20}}
        }}, PCFG);
      }} else if (spec.type === 'bar_count') {{
        const entries = countBy(rows, spec.col, spec.topN);
        if (!entries.length) {{ noDom(spec.id); return; }}
        Plotly.newPlot(spec.id, [{{
          x: entries.map(([,v])=>v).reverse(),
          y: entries.map(([k])=>trunc(k,50)).reverse(),
          type:'bar', orientation:'h', marker:{{color:'#4f8ef7'}},
          customdata: entries.map(([k])=>k).reverse(),
          hovertemplate:'%{{customdata}}: %{{x:,}}<extra></extra>'
        }}], {{ ...BL(t('frequency'),''), margin:{{t:10,b:40,l:190,r:20}}, yaxis:{{automargin:true}} }}, PCFG);
      }} else if (spec.type === 'hist') {{
        const h = numericHist(rows, spec.col, 20);
        if (!h) {{ noDom(spec.id); return; }}
        const mid = h.edges.slice(0,-1).map((e,i)=>(e+h.edges[i+1])/2);
        Plotly.newPlot(spec.id, [{{
          x: mid, y: h.counts, type:'bar', marker:{{color:'#4f8ef7'}},
          hovertemplate:'Score: %{{x:.2f}}<br>'+t('frequency')+': %{{y}}<extra></extra>'
        }}], {{ ...BL(t('neg_score'),t('frequency')), margin:{{t:10,b:40,l:60,r:20}} }}, PCFG);
      }}
    }} catch(e) {{ noDom(spec.id); console.warn(spec.id,e); }}
  }});
}}

/* ─── Aggregation helpers ───────────────────────────────────────────── */
function countBy(rows, col, topN) {{
  const counts = {{}};
  rows.forEach(r => {{
    const v=r[col]; if(v==null) return;
    (Array.isArray(v)?v:[v]).forEach(x=>{{ if(x!=null) counts[String(x)]=(counts[String(x)]||0)+1; }});
  }});
  const sorted = Object.entries(counts).sort((a,b)=>b[1]-a[1]);
  return topN ? sorted.slice(0,topN) : sorted;
}}

function countPerYearRegion(rows, yearCol, regionCol) {{
  const regions={{}}, perYearRegion={{}};
  rows.forEach(r=>{{
    const y=r[yearCol], rg=r[regionCol]; if(y==null||rg==null) return;
    regions[String(rg)]=1;
    if(!perYearRegion[y]) perYearRegion[y]={{}};
    perYearRegion[y][String(rg)]=(perYearRegion[y][String(rg)]||0)+1;
  }});
  const years = Object.keys(perYearRegion).map(Number).sort();
  return {{years, regions:Object.keys(regions), perYearRegion}};
}}

function numericHist(rows, col, bins) {{
  const vals=rows.map(r=>r[col]).filter(v=>v!=null&&!isNaN(v));
  if(!vals.length) return null;
  const mn=Math.min(...vals), mx=Math.max(...vals);
  if(mn===mx) return {{counts:[vals.length], edges:[mn,mx+1]}};
  const step=(mx-mn)/bins;
  const edges=Array.from({{length:bins+1}},(_,i)=>mn+i*step);
  const counts=new Array(bins).fill(0);
  vals.forEach(v=>{{ let i=Math.floor((v-mn)/step); if(i>=bins) i=bins-1; counts[i]++; }});
  return {{counts,edges}};
}}

/* ─── Legend tooltips (SVG <title> → native browser tooltip on hover) ── */
function addLegendTooltips(id, names) {{
  setTimeout(() => {{
    const el = document.getElementById(id);
    if (!el) return;
    el.querySelectorAll('.legendtext').forEach((lt, i) => {{
      if (!names[i]) return;
      // Remove any existing title child
      const existing = lt.querySelector('title');
      if (existing) existing.remove();
      const titleEl = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      titleEl.textContent = names[i];
      lt.appendChild(titleEl);
      lt.style.cursor = 'help';
    }});
  }}, 350);
}}

/* ─── Editable chart titles ─────────────────────────────────────────── */
function syncChartTitle(el) {{
  const id    = el.dataset.chartId;
  const title = el.textContent.trim();
  if (!title) {{ el.textContent = el.dataset.defaultTitle || '—'; return; }}
  if (id && document.getElementById(id)) {{
    try {{
      Plotly.relayout(id, {{'title': {{text: title, font: {{size: 12, color: '#444'}}}}, 'margin.t': 42}});
    }} catch(e) {{}}
  }}
}}

/* ─── UI/layout helpers ─────────────────────────────────────────────── */
const C = ['#4f8ef7','#f7964f','#4fbcf7','#f74f6e','#a64ff7','#4ff7a6',
           '#f7e54f','#4f6ef7','#f74fc4','#7af74f','#f7af4f','#4ff7e5'];

/* Compute horizontal-legend layout (items below chart) with NO gap.
   n      = number of legend items
   nCats  = number of y-axis bars/categories (used to size the bar area)
   Returns {{ b, y, height }} where height is the div height to set. */
function legendBottom(n, nCats) {{
  nCats = nCats || 3;
  const rows = Math.ceil(n / 2);       // 2 items per row at font-size 8
  const L    = rows * 20 + 20;         // legend pixel height (empirical)
  // Minimum total height so bars get at least 50px each:
  // T = 67.5*nCats + L + 10  solves: H_plot >= nCats*50 with y=-0.35
  const T = Math.max(295, Math.round(67.5 * nCats + L + 10));
  // margin.b that exactly fits legend at y=-0.35 (no gap, no overflow):
  const b = Math.round((L + 0.35 * (T - 10)) / 1.35);
  return {{ b, y: -0.35, height: T }};
}}
const BL = (xt,yt) => ({{
  paper_bgcolor:'transparent', plot_bgcolor:'transparent',
  font:{{family:'system-ui,sans-serif',size:11}},
  xaxis:{{title:xt}}, yaxis:{{title:yt}}, showlegend:false,
}});
const noData  = () => `<p class="no-data">${{t('no_data')}}</p>`;
const noDom   = (id) => {{ document.getElementById(id).innerHTML = noData(); }};
const trunc   = (s, n) => String(s).length > n ? String(s).slice(0,n)+'…' : String(s);
const esc     = s => String(s).replace(/'/g,"\\'");
const dsRow   = (l,v) => `<div class="ds-row"><span>${{l}}</span><b>${{v}}</b></div>`;

/* ─── Bootstrap ─────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {{
  initEda();
  initFilterPanel();
  renderFilterChips();
}});
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#f5f6fa;color:#1a1a2e;min-height:100vh}

header{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);color:#fff;padding:18px 28px}
.header-inner{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px}
.header-inner h1{font-size:1.3rem;font-weight:700}
.badge{background:rgba(255,255,255,.15);border-radius:12px;padding:3px 12px;font-size:.78rem;color:#cdd;display:inline-block;margin-top:4px}
.lang-toggle{display:flex;gap:4px}
.lang-btn{background:rgba(255,255,255,.12);border:1.5px solid rgba(255,255,255,.25);color:#dde;border-radius:6px;padding:5px 12px;cursor:pointer;font-size:.82rem;font-weight:600;transition:all .15s}
.lang-btn.active{background:#4f8ef7;border-color:#4f8ef7;color:#fff}
.lang-btn:hover{background:rgba(255,255,255,.2)}

.tab-bar{display:flex;background:#fff;border-bottom:2px solid #e0e3ef;padding:0 24px}
.tab-btn{background:none;border:none;cursor:pointer;padding:13px 22px;font-size:.93rem;color:#666;border-bottom:3px solid transparent;margin-bottom:-2px;transition:color .15s,border-color .15s}
.tab-btn:hover{color:#333}
.tab-btn.active{color:#4f8ef7;border-bottom-color:#4f8ef7;font-weight:600}

.tab-content{padding:26px 30px;max-width:1400px;margin:0 auto}
section{margin-bottom:34px}
.section-title{font-size:1.05rem;font-weight:700;margin-bottom:5px;color:#16213e}
.hint{font-size:.83rem;color:#888;margin-bottom:12px}

.overview-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:14px;margin-bottom:26px}
.stat-card{background:#fff;border-radius:12px;padding:17px 15px;box-shadow:0 1px 6px rgba(0,0,0,.07);display:flex;flex-direction:column;align-items:center;text-align:center;gap:4px}
.stat-icon{font-size:1.6rem}
.stat-value{font-size:1.45rem;font-weight:700;color:#4f8ef7}
.stat-label{font-size:.75rem;color:#888;text-transform:uppercase;letter-spacing:.04em}
.stat-sub{font-size:.7rem;color:#bbb}

.col-explorer h2{font-size:1.05rem;margin-bottom:5px}
.col-cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(185px,1fr));gap:9px;margin-bottom:16px}
.col-card{background:#fff;border-radius:10px;padding:12px 13px;box-shadow:0 1px 4px rgba(0,0,0,.06);cursor:pointer;transition:box-shadow .15s,transform .1s;border:2px solid transparent}
.col-card:hover{box-shadow:0 4px 12px rgba(79,142,247,.2);transform:translateY(-1px)}
.col-card.selected{border-color:#4f8ef7}
.col-card-name{font-weight:600;font-size:.85rem;margin-bottom:4px;word-break:break-word}
.col-card-stats{display:flex;gap:8px;font-size:.74rem;color:#666;margin-top:6px}
.type-badge{display:inline-block;border-radius:5px;padding:1px 7px;font-size:.68rem;font-weight:600;letter-spacing:.03em}
.badge-num{background:#e8f0fe;color:#3366cc}
.badge-cat{background:#fce8f3;color:#9b3080}
.badge-list{background:#e6f4ea;color:#1e7e34}
.null-bar{height:3px;background:#eee;border-radius:2px;margin-top:6px}
.null-fill{height:100%;background:#f7964f;border-radius:2px}

.col-detail{background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 10px rgba(0,0,0,.08);margin-top:5px}
.col-detail.hidden{display:none}
.detail-title{font-size:1rem;font-weight:700;margin-bottom:11px;color:#16213e}
.detail-stats{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:7px;margin-bottom:16px}
.chart-type-btns{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}
.chart-type-btn{padding:5px 13px;border:1.5px solid #d0d5e8;border-radius:20px;background:#fff;color:#555;font-size:.78rem;cursor:pointer;transition:all .15s}
.chart-type-btn:hover{border-color:#4f8ef7;color:#4f8ef7}
.chart-type-btn.active{background:#4f8ef7;border-color:#4f8ef7;color:#fff;font-weight:600}
.ds-row{background:#f5f6fa;border-radius:7px;padding:8px 12px;display:flex;justify-content:space-between;font-size:.81rem}
.ds-row span{color:#888}

.charts-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:16px}
.analysis-filter-bar{display:flex;flex-wrap:wrap;align-items:flex-start;gap:20px;background:#f5f6fa;border:1px solid #e2e5f0;border-radius:10px;padding:14px 18px;margin-bottom:8px}
.af-group{display:flex;flex-direction:column;gap:6px}
.af-group-regions{flex:1;min-width:200px}
.af-label{font-size:.78rem;font-weight:600;color:#4a5568;text-transform:uppercase;letter-spacing:.04em}
.af-year-inputs{display:flex;align-items:center;gap:6px}
.af-year-input{width:80px;padding:5px 8px;border:1.5px solid #d0d5e8;border-radius:7px;font-size:.85rem;color:#1a202c;background:#fff}
.af-year-input:focus{outline:none;border-color:#4f8ef7}
.af-year-sep{color:#aaa;font-size:.9rem}
.af-checkboxes{display:flex;flex-wrap:wrap;gap:6px 14px;max-height:80px;overflow-y:auto}
.af-check-label{display:flex;align-items:center;gap:4px;font-size:.8rem;color:#2d3748;cursor:pointer;white-space:nowrap}
.af-check-label input{cursor:pointer;accent-color:#4f8ef7}
.af-clear-btn{align-self:flex-end;padding:6px 14px;background:#fff;border:1.5px solid #d0d5e8;border-radius:20px;font-size:.78rem;color:#666;cursor:pointer;transition:all .15s;white-space:nowrap}
.af-clear-btn:hover{border-color:#f74f6e;color:#f74f6e}
.chart-card{background:#fff;border-radius:12px;padding:15px 17px;box-shadow:0 1px 6px rgba(0,0,0,.07)}
.chart-card-tall{grid-column:span 2}
.chart-title{font-size:.88rem;font-weight:600;color:#444;margin-bottom:9px}
.chart-area{height:295px}
.chart-area-tall{height:380px}
.no-data{font-size:.83rem;color:#bbb;text-align:center;padding:40px 0}

/* Wordcloud */
.wc-panel{background:#fff;border-radius:12px;padding:18px 20px;box-shadow:0 1px 6px rgba(0,0,0,.07);display:flex;flex-wrap:wrap;gap:20px}
.wc-controls{display:flex;flex-direction:column;gap:12px;min-width:220px}
.wc-ctrl-row{display:flex;flex-direction:column;gap:4px}
.wc-ctrl-row label,.wc-filter-wrap label{font-size:.8rem;color:#666;font-weight:600}
.wc-controls select{padding:7px 10px;border-radius:7px;border:1.5px solid #dde;font-size:.85rem;background:#f9f9fc;min-width:200px}
.wc-filter-wrap{display:flex;flex-direction:column;gap:6px}
.wc-filter-wrap.hidden{display:none}
.wc-canvas-wrap{flex:1;min-width:300px;position:relative;display:flex;align-items:center;justify-content:center;background:#fafafa;border-radius:10px;border:1px solid #eee;overflow:hidden}
#wc-canvas{display:block;max-width:100%}
.wc-placeholder{position:absolute;font-size:.85rem;color:#aaa;pointer-events:none}

/* Filter panel */
.filter-panel{background:#fff;border-radius:12px;padding:16px 20px;box-shadow:0 1px 6px rgba(0,0,0,.07);margin-bottom:14px}
.filter-builder{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-start}
.filter-builder select{padding:7px 10px;border-radius:7px;border:1.5px solid #dde;font-size:.86rem;min-width:195px;background:#f9f9fc}
.filter-search-input{padding:7px 10px;border-radius:7px;border:1.5px solid #dde;font-size:.86rem;min-width:195px;background:#f9f9fc;transition:border-color .15s}
.filter-search-input:focus{outline:none;border-color:#4f8ef7;box-shadow:0 0 0 2px rgba(79,142,247,.15)}
.filter-vals-wrap{display:flex;flex-wrap:wrap;gap:9px;align-items:flex-start}
.filter-vals-wrap.hidden{display:none}
.filter-vals-list{display:flex;flex-wrap:wrap;gap:4px;max-height:120px;overflow-y:auto;padding:6px;background:#f5f6fa;border-radius:7px;border:1px solid #e0e3ef;min-width:230px}
.val-checkbox{display:flex;align-items:center;gap:4px;font-size:.78rem;cursor:pointer}
.val-checkbox input{cursor:pointer}
.btn-add{background:#4f8ef7;color:#fff;border:none;border-radius:7px;padding:7px 15px;cursor:pointer;font-size:.84rem;font-weight:600;white-space:nowrap;align-self:flex-end}
.btn-add:hover{background:#3a78e0}
.active-filters{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px;align-items:center}
.no-filters{font-size:.81rem;color:#aaa}
.chip{background:#e8f0fe;border-radius:20px;padding:4px 10px;font-size:.78rem;display:inline-flex;align-items:center;gap:5px}
.chip b{color:#3366cc}
.chip-remove{background:none;border:none;cursor:pointer;color:#3366cc;font-size:.8rem;padding:0;line-height:1}
.btn-clear{background:none;border:1.5px solid #ccc;border-radius:6px;padding:3px 9px;font-size:.76rem;cursor:pointer;color:#888}
.btn-clear:hover{border-color:#f7964f;color:#f7964f}
.record-banner{font-size:.81rem;color:#666;text-align:right;margin-bottom:7px}
.range-filter-inputs{display:flex;align-items:center;gap:8px;padding:6px;font-size:.8rem}
/* Summary table */
.summary-table-wrap{overflow-x:auto}
.summary-table{width:100%;border-collapse:collapse;font-size:.82rem;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 6px rgba(0,0,0,.07)}
.summary-table thead tr{background:#f0f2fa}
.summary-table th{text-align:left;padding:10px 13px;font-weight:700;color:#444;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em;border-bottom:2px solid #e0e3ef;white-space:nowrap}
.summary-table td{padding:9px 13px;border-bottom:1px solid #f0f2fa;color:#333;vertical-align:middle}
.summary-table tbody tr:hover{background:#f8f9ff}
.summary-table tbody tr:last-child td{border-bottom:none}
.st-name{font-weight:600;font-size:.83rem;color:#16213e}
.st-num{text-align:right;font-variant-numeric:tabular-nums}
.st-pct{color:#aaa;font-size:.75rem}
.st-warn{color:#e03030;font-weight:600}
.st-caution{color:#e07830}

.editable-title{cursor:text;border-radius:4px;padding:1px 4px;transition:background .15s;outline:none}
.editable-title:hover{background:rgba(79,142,247,.08)}
.editable-title:focus{background:rgba(79,142,247,.12);box-shadow:0 0 0 2px rgba(79,142,247,.3)}
.stat-label.editable-title{display:inline-block}
"""
