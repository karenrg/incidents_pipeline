"""Figure generation and executive PDF report assembly."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .nlp import token_frequencies

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_title(name: str) -> str:
    """Derive a human-readable title from a snake_case metric name."""
    return name.replace("_", " ").title()


def _save_figure(fig: matplotlib.figure.Figure, figures_dir: Path, name: str, dpi: int) -> Path:
    """Save a matplotlib figure to *figures_dir* and close it."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved figure '%s' to %s", name, path)
    return path


def _apply_filters(
    df: pd.DataFrame, filters: dict | None, multilabel_columns: list[str],
) -> pd.DataFrame:
    """Apply column-value filters to a DataFrame.

    For scalar columns the filter uses ``isin``; for list-valued
    (multilabel) columns it matches if *any* filter value appears in the
    row's list.
    """
    if not filters:
        return df
    filtered = df.copy()
    for col, values in filters.items():
        if col in multilabel_columns:
            mask = filtered[col].apply(
                lambda x: isinstance(x, list) and any(v in x for v in values)
            )
        else:
            mask = filtered[col].isin(values)
        filtered = filtered[mask]
    return filtered


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def _plot_stacked_bar(
    frame: pd.DataFrame,
    title: str,
    legend_title: str,
    figures_dir: Path,
    name: str,
    dpi: int,
    figsize: tuple[float, float] = (8, 4),
) -> Path:
    """Render a region-by-category stacked bar chart."""
    ordered = frame[frame.sum().sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=figsize)
    ordered.plot(kind="bar", stacked=True, colormap="Set3", alpha=0.9, width=0.8, ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("N. Incidents", fontsize=10)
    ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(alpha=0.3, linestyle="--", axis="y")
    plt.setp(ax.get_xticklabels(), rotation=0)
    return _save_figure(fig, figures_dir, name, dpi)


def plot_regional_evolution(metrics: dict, figures_dir: Path, dpi: int) -> Path:
    """Stacked area chart of incidents per year and region."""
    frame = pd.DataFrame(metrics["regional_evolution"]).set_index("year")

    fig, ax = plt.subplots(figsize=(8, 4))
    frame.plot.area(stacked=True, colormap="Set3", alpha=0.85, ax=ax)
    ax.set_title(_auto_title("regional_evolution"), fontsize=12)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("N. of Incidents", fontsize=10)
    ax.legend(title="Region", loc="upper left")
    ax.grid(alpha=0.3, linestyle="--")
    return _save_figure(fig, figures_dir, "regional_evolution", dpi)


def plot_cumulative_concentration(metrics: dict, figures_dir: Path, dpi: int) -> Path:
    """Line chart of the cumulative percentage of incidents over time."""
    frame = pd.DataFrame(metrics["cumulative_concentration"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(frame["year"], frame["cumulative_pct"], marker="o", linewidth=2)
    ax.set_title(_auto_title("cumulative_concentration"), fontsize=12)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Cumulative %", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    return _save_figure(fig, figures_dir, "cumulative_concentration", dpi)


def plot_principles_distribution(metrics: dict, config: dict, figures_dir: Path, dpi: int) -> Path:
    """Stacked bar chart of the most frequent tags/principles per region."""
    region = config["analysis"]["columns"]["region"]
    frame = pd.DataFrame(metrics["principles_distribution"]).set_index(region)
    return _plot_stacked_bar(
        frame,
        title=_auto_title("principles_distribution"),
        legend_title=_auto_title(config["analysis"]["columns"]["tags"]),
        figures_dir=figures_dir,
        name="principles_distribution",
        dpi=dpi,
    )


def plot_harm_types_distribution(metrics: dict, config: dict, figures_dir: Path, dpi: int) -> Path:
    """Stacked bar chart of harm types per region."""
    region = config["analysis"]["columns"]["region"]
    frame = pd.DataFrame(metrics["harm_types_distribution"]).set_index(region)
    return _plot_stacked_bar(
        frame,
        title=_auto_title("harm_types_distribution"),
        legend_title=_auto_title(config["analysis"]["columns"]["harm_type"]),
        figures_dir=figures_dir,
        name="harm_types_distribution",
        dpi=dpi,
    )


def plot_stakeholders_distribution(metrics: dict, config: dict, figures_dir: Path, dpi: int) -> Path:
    """Stacked bar chart of the most affected stakeholder groups per region."""
    region = config["analysis"]["columns"]["region"]
    frame = pd.DataFrame(metrics["stakeholders_distribution"]).set_index(region)
    return _plot_stacked_bar(
        frame,
        title=_auto_title("stakeholders_distribution"),
        legend_title=_auto_title(config["analysis"]["columns"]["harmed"]),
        figures_dir=figures_dir,
        name="stakeholders_distribution",
        dpi=dpi,
        figsize=(10, 4),
    )


def plot_vulnerable_groups(metrics: dict, figures_dir: Path, dpi: int) -> Path:
    """Pie chart of incidents affecting configured vulnerable groups."""
    frame = pd.DataFrame(metrics["vulnerable_groups_distribution"])
    colors = plt.get_cmap("Set3").colors[: len(frame)]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        frame["count"],
        labels=frame["group"],
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
        textprops={"fontsize": 9},
    )
    ax.set_title(_auto_title("vulnerable_groups"), fontsize=12, weight="bold")
    return _save_figure(fig, figures_dir, "vulnerable_groups", dpi)


def plot_harm_chronology(metrics: dict, config: dict, figures_dir: Path, dpi: int) -> Path:
    """Line chart of harm types over time."""
    frame = pd.DataFrame(metrics["harm_chronology"])
    types = config["analysis"]["harm_chronology_types"]

    fig, ax = plt.subplots(figsize=(9, 4))
    for harm_type in types:
        if harm_type in frame.columns:
            ax.plot(frame["year"], frame[harm_type], marker="o", linewidth=2, label=harm_type)
    ax.set_title(_auto_title("harm_chronology"), fontsize=13, weight="bold")
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("N. Incidents", fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(title=_auto_title(config["analysis"]["columns"]["harm_type"]))
    return _save_figure(fig, figures_dir, "harm_chronology", dpi)


def plot_negativity_by_principle(metrics: dict, config: dict, figures_dir: Path, dpi: int) -> Path:
    """Grid of monthly negativity-score evolution for top tags/principles."""
    frame = pd.DataFrame(metrics["negativity_by_principle"])
    primary_window = config["analysis"]["moving_average_windows"][0]

    principles = [col for col in frame.columns if col != "year_month" and "__ma" not in col]
    n_cols = 2
    n_rows = int(np.ceil(len(principles) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).reshape(-1)

    x_values = pd.to_datetime(frame["year_month"])
    for index, principle in enumerate(principles):
        ax = axes[index]
        ax.plot(
            x_values, frame[principle],
            marker="o", linestyle="--", linewidth=1,
            color="red", alpha=0.5, label="Monthly average",
        )
        rolling_column = f"{principle}__ma{primary_window}"
        if rolling_column in frame.columns:
            ax.plot(
                x_values, frame[rolling_column],
                linewidth=2, color="darkred",
                label=f"{primary_window}-month rolling average",
            )
        ax.set_title(principle, fontsize=11, weight="bold")
        ax.grid(alpha=0.3, linestyle="--")

    for ax in axes[len(principles):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(_auto_title("negativity_by_principle"), fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    return _save_figure(fig, figures_dir, "negativity_by_principle", dpi)


def plot_wordcloud(
    df: pd.DataFrame,
    config: dict,
    figures_dir: Path,
    dpi: int,
    wc_config: dict,
) -> Path:
    """Word cloud for a (possibly filtered) slice of the dataset."""
    from wordcloud import WordCloud

    multilabel_cols = config["data"].get("multilabel_columns", [])
    filtered_df = _apply_filters(df, wc_config.get("filters"), multilabel_cols)

    tokens_col = config["analysis"]["columns"]["tokens"]
    frequencies = token_frequencies(filtered_df, token_column=tokens_col)

    if not frequencies:
        logger.warning("Wordcloud '%s': no tokens after filtering — skipping", wc_config["name"])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=16)
        ax.axis("off")
        return _save_figure(fig, figures_dir, f"wordcloud_{wc_config['name']}", dpi)

    cloud = WordCloud(width=1200, height=800, background_color="white", colormap="viridis")
    cloud.generate_from_frequencies(frequencies)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        f"Word Cloud — {_auto_title(wc_config['name'])}",
        fontsize=13, weight="bold",
    )
    return _save_figure(fig, figures_dir, f"wordcloud_{wc_config['name']}", dpi)


# ---------------------------------------------------------------------------
# Figure orchestration
# ---------------------------------------------------------------------------

def generate_figures(df: pd.DataFrame, metrics: dict, config: dict) -> dict[str, Path]:
    """Generate all figures required for the results chapter.

    Returns a dict mapping figure names to saved PNG paths.
    """
    figures_dir = Path(config["reporting"]["figures_dir"])
    dpi = config["reporting"]["figures_dpi"]

    figures: dict[str, Path] = {
        "regional_evolution": plot_regional_evolution(metrics, figures_dir, dpi),
        "cumulative_concentration": plot_cumulative_concentration(metrics, figures_dir, dpi),
        "principles_distribution": plot_principles_distribution(metrics, config, figures_dir, dpi),
        "harm_types_distribution": plot_harm_types_distribution(metrics, config, figures_dir, dpi),
        "stakeholders_distribution": plot_stakeholders_distribution(metrics, config, figures_dir, dpi),
        "vulnerable_groups": plot_vulnerable_groups(metrics, figures_dir, dpi),
        "harm_chronology": plot_harm_chronology(metrics, config, figures_dir, dpi),
        "negativity_by_principle": plot_negativity_by_principle(metrics, config, figures_dir, dpi),
    }

    for wc_config in config.get("wordclouds", [{"name": "general"}]):
        name = f"wordcloud_{wc_config['name']}"
        figures[name] = plot_wordcloud(df, config, figures_dir, dpi, wc_config)

    return figures


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------

# Maps figure names to their corresponding metrics key.
_FIGURE_METRICS_MAP: dict[str, str] = {
    "regional_evolution": "regional_evolution",
    "cumulative_concentration": "cumulative_concentration",
    "principles_distribution": "principles_distribution",
    "harm_types_distribution": "harm_types_distribution",
    "stakeholders_distribution": "stakeholders_distribution",
    "vulnerable_groups": "vulnerable_groups_distribution",
    "harm_chronology": "harm_chronology",
    "negativity_by_principle": "negativity_by_principle",
    "harm_severity_index": "harm_severity_index",
    "mental_health_summary": "mental_health_summary",
    "top_industries": "top_industries",
    "top_tokens": "top_tokens",
}

# Columns to exclude from auto-generated tables (rolling average columns).
_TABLE_EXCLUDE_SUFFIXES = ("__ma",)


def _table_columns(records: list[dict]) -> list[str]:
    """Pick the columns to display in an auto-generated table."""
    if not records:
        return []
    all_cols = list(records[0].keys())
    return [c for c in all_cols if not any(c.find(suf) >= 0 for suf in _TABLE_EXCLUDE_SUFFIXES)]


def _write_table(pdf, title: str, records: list[dict], columns: list[str]) -> None:
    """Render a simple bordered table with a heading."""
    if not records or not columns:
        return

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, title, ln=True)

    available_width = pdf.epw
    col_width = available_width / len(columns)

    pdf.set_font("Helvetica", "B", 8)
    for column in columns:
        pdf.cell(col_width, 6, _auto_title(str(column))[:30], border=1)
    pdf.ln(6)

    pdf.set_font("Helvetica", "", 8)
    for record in records:
        for column in columns:
            value = record.get(column, "")
            if isinstance(value, float):
                value = f"{value:.2f}"
            text = str(value)[:30]
            pdf.cell(col_width, 6, text, border=1)
        pdf.ln(6)


def _write_summary(pdf, metrics: dict) -> None:
    """Render the dataset summary block."""
    summary = metrics["summary"]
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0, 6,
        f"Total incidents analyzed: {summary['n_incidents']}\n"
        f"Period covered: {summary['year_range'][0]}-{summary['year_range'][1]}\n"
        f"Regions included: {', '.join(summary['regions'])}",
    )


def _write_methodology(pdf, config: dict) -> None:
    """Render the methodology section."""
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Methodology", ln=True)

    pdf.set_font("Helvetica", "", 11)
    text = (
        f"Data source: {config['data'].get('schema_type', 'N/A')}.\n"
        f"Temporal scope: {config['filters']['year_range'][0]}-{config['filters']['year_range'][1]}.\n"
        f"Regions: {', '.join(config['filters']['regions'])}.\n\n"
        "Preprocessing: multi-label parsing, temporal and geographic filtering, "
        "harm-type recategorization, and multi-label binarization for the "
        "machine-learning branch.\n\n"
        "NLP: Unicode normalization, tokenization, NLTK WordNet lemmatization, "
        "stopword removal and a configurable normalization map.\n\n"
        f"Sentiment analysis backend: {config['sentiment']['backend']}. Negativity "
        f"scores are mapped to categorical labels using thresholds high="
        f"{config['sentiment']['thresholds']['high']} (Alta), medium="
        f"{config['sentiment']['thresholds']['medium']} (Media), otherwise Baja."
    )
    pdf.multi_cell(0, 6, text)


def _write_standalone_tables(pdf, metrics: dict) -> None:
    """Render statistical tables that have no associated chart."""
    _write_table(pdf, "Top Industries", metrics["top_industries"], ["industry", "count"])
    pdf.ln(4)
    _write_table(
        pdf, "Harm Severity Index by Year",
        metrics["harm_severity_index"], ["year", "avg_severity"],
    )
    pdf.ln(4)
    _write_table(
        pdf, "Mental-Health-Related Incidents by Year",
        metrics["mental_health_summary"],
        ["year", "n_incidents", "n_total", "proportion"],
    )
    pdf.ln(4)
    _write_table(
        pdf, "Top Tokens",
        metrics["top_tokens"][:20], ["token", "frequency"],
    )


def generate_report(metrics: dict, figures: dict[str, Path], config: dict) -> Path:
    """Assemble the executive PDF report.

    Each figure is followed by a data table with the underlying counts.
    """
    from fpdf import FPDF

    reporting = config["reporting"]

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Cover ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, reporting["report_title"], ln=True, align="C")
    pdf.ln(4)
    _write_summary(pdf, metrics)

    # --- Methodology ---
    pdf.add_page()
    _write_methodology(pdf, config)

    # --- Figures + companion tables ---
    for name, path in figures.items():
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, _auto_title(name), ln=True)
        pdf.image(str(path), w=pdf.epw)
        pdf.ln(3)

        metrics_key = _FIGURE_METRICS_MAP.get(name)
        if metrics_key and metrics_key in metrics:
            records = metrics[metrics_key]
            columns = _table_columns(records)
            _write_table(pdf, "", records, columns)

    # --- Tables without a chart ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Additional Statistics", ln=True)
    _write_standalone_tables(pdf, metrics)

    reports_dir = Path(reporting["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "executive_report.pdf"
    pdf.output(str(output_path))
    logger.info("Report written to %s", output_path)

    return output_path


def run_visualization(df: pd.DataFrame, metrics: dict, config: dict) -> Path:
    """Generate all figures and assemble the executive PDF report."""
    figures = generate_figures(df, metrics, config)
    return generate_report(metrics, figures, config)
