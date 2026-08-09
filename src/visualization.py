"""Figure generation and executive PDF report assembly."""

import io
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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


def _truncate_label(label: str, max_len: int = 28) -> str:
    """Truncate a long legend label to avoid overflow."""
    return label[:max_len] + "…" if len(label) > max_len else label


def _save_figure(fig: matplotlib.figure.Figure, figures_dir: Path, name: str, dpi: int) -> Path:
    """Save a matplotlib figure to *figures_dir* and close it."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure '%s' to %s", name, path)
    return path


def _apply_filters(
    df: pd.DataFrame, filters: dict | None, multilabel_columns: list[str],
) -> pd.DataFrame:
    """Apply column-value filters to a DataFrame."""
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


def _palette(n: int, cmap: str = "Set2") -> list:
    """Return n colors from a matplotlib colormap."""
    return [plt.get_cmap(cmap)(i / max(n - 1, 1)) for i in range(n)]


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
    figsize: tuple[float, float] = (9, 4),
) -> Path:
    """Stacked bar chart with region on x-axis, legend below."""
    ordered = frame[frame.sum().sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=figsize)
    ordered.plot(kind="bar", stacked=True, colormap="Set3", alpha=0.9, width=0.65, ax=ax)
    ax.set_title(title, fontsize=13, weight="bold", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("N. Incidents", fontsize=10)

    handles, labels = ax.get_legend_handles_labels()
    labels = [_truncate_label(lbl) for lbl in labels]
    n_cols = min(len(labels), 4)
    ax.legend(
        handles, labels,
        title=legend_title,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=n_cols,
        fontsize=8,
        frameon=True,
    )

    ax.grid(alpha=0.3, linestyle="--", axis="y")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return _save_figure(fig, figures_dir, name, dpi)


def plot_regional_evolution(metrics: dict, figures_dir: Path, dpi: int) -> Path:
    """Stacked area chart of incidents per year and region."""
    frame = pd.DataFrame(metrics["regional_evolution"]).set_index("year")

    fig, ax = plt.subplots(figsize=(9, 4))
    frame.plot.area(stacked=True, colormap="Set2", alpha=0.85, ax=ax)
    ax.set_title("Evolution of AI Incidents per Region", fontsize=13, weight="bold", pad=10)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("N. of Incidents", fontsize=10)
    ax.legend(
        title="Region",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=len(frame.columns),
        fontsize=9,
        frameon=True,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(alpha=0.3, linestyle="--")
    return _save_figure(fig, figures_dir, "regional_evolution", dpi)


def plot_cumulative_concentration(metrics: dict, figures_dir: Path, dpi: int) -> Path:
    """Line chart of the cumulative percentage of incidents over time."""
    frame = pd.DataFrame(metrics["cumulative_concentration"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(frame["year"], frame["cumulative_pct"], marker="o", linewidth=2.5, color="#2196F3")
    ax.fill_between(frame["year"], frame["cumulative_pct"], alpha=0.15, color="#2196F3")
    ax.set_title("Cumulative Concentration of AI Incidents Over Time", fontsize=13, weight="bold", pad=10)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Cumulative %", fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.6)
    return _save_figure(fig, figures_dir, "cumulative_concentration", dpi)


def plot_principles_distribution(metrics: dict, config: dict, figures_dir: Path, dpi: int) -> Path:
    """Stacked bar chart of the most frequent tags/principles per region."""
    region = config["analysis"]["columns"]["region"]
    frame = pd.DataFrame(metrics["principles_distribution"]).set_index(region)
    return _plot_stacked_bar(
        frame,
        title="AI Principles Affected by Incidents",
        legend_title="Principle",
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
        title="Types of Harm Caused by Incidents",
        legend_title="Harm Type",
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
        title="Distribution of Affected Stakeholders",
        legend_title="Stakeholder",
        figures_dir=figures_dir,
        name="stakeholders_distribution",
        dpi=dpi,
        figsize=(10, 4),
    )


def plot_vulnerable_groups(metrics: dict, figures_dir: Path, dpi: int) -> Path:
    """Horizontal bar chart of incidents affecting configured vulnerable groups."""
    frame = pd.DataFrame(metrics["vulnerable_groups_distribution"]).sort_values("count")
    colors = _palette(len(frame), "Set2")

    fig, ax = plt.subplots(figsize=(7, max(3, len(frame) * 0.7)))
    bars = ax.barh(frame["group"], frame["count"], color=colors, edgecolor="white", height=0.6)

    for bar, val in zip(bars, frame["count"]):
        ax.text(bar.get_width() + max(frame["count"]) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=10, fontweight="bold")

    ax.set_title("Vulnerable Groups Affected by AI Incidents", fontsize=13, weight="bold", pad=10)
    ax.set_xlabel("N. Incidents", fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(alpha=0.3, linestyle="--", axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    return _save_figure(fig, figures_dir, "vulnerable_groups", dpi)


def plot_harm_chronology(metrics: dict, config: dict, figures_dir: Path, dpi: int) -> Path:
    """Line chart of harm types over time."""
    frame = pd.DataFrame(metrics["harm_chronology"])
    types = config["analysis"]["harm_chronology_types"]
    colors = ["#E53935", "#1E88E5", "#43A047", "#FB8C00"]

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, harm_type in enumerate(types):
        if harm_type in frame.columns:
            ax.plot(
                frame["year"], frame[harm_type],
                marker="o", linewidth=2.5,
                color=colors[i % len(colors)],
                label=harm_type,
            )

    ax.set_title("Evolution of Harm Types Over Time", fontsize=13, weight="bold", pad=10)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("N. Incidents", fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(
        title="Harm Type",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(len(types), 2),
        fontsize=9,
        frameon=True,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    return _save_figure(fig, figures_dir, "harm_chronology", dpi)


def plot_top_industries(metrics: dict, config: dict, figures_dir: Path, dpi: int) -> Path:
    """Horizontal bar chart of top industries by incident count."""
    top_n = config["analysis"].get("top_n_industries", 10)
    frame = pd.DataFrame(metrics["top_industries"][:top_n]).sort_values("count")
    colors = _palette(len(frame), "Blues")

    fig, ax = plt.subplots(figsize=(9, max(4, len(frame) * 0.55)))
    bars = ax.barh(frame["industry"], frame["count"], color=colors, edgecolor="white", height=0.65)

    for bar, val in zip(bars, frame["count"]):
        ax.text(bar.get_width() + max(frame["count"]) * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)

    ax.set_title(f"Top {top_n} Industries by Incident Count", fontsize=13, weight="bold", pad=10)
    ax.set_xlabel("N. Incidents", fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(alpha=0.3, linestyle="--", axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    return _save_figure(fig, figures_dir, "top_industries", dpi)


def plot_harm_severity_index(metrics: dict, figures_dir: Path, dpi: int) -> Path:
    """Bar chart of average harm severity index per year."""
    frame = pd.DataFrame(metrics["harm_severity_index"])
    colors = ["#EF5350" if v > 0.3 else "#FFA726" if v > 0.1 else "#66BB6A"
              for v in frame["avg_severity"]]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(frame["year"].astype(str), frame["avg_severity"], color=colors,
                  edgecolor="white", width=0.65)

    for bar, val in zip(bars, frame["avg_severity"]):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Harm Severity Index by Year", fontsize=13, weight="bold", pad=10)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Average Severity Score", fontsize=10)
    max_val = frame["avg_severity"].replace([np.inf, -np.inf], np.nan).max()
    ax.set_ylim(0, max(float(max_val) * 1.2, 0.1) if pd.notna(max_val) else 1.0)
    ax.grid(alpha=0.3, linestyle="--", axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    return _save_figure(fig, figures_dir, "harm_severity_index", dpi)


def plot_mental_health(metrics: dict, figures_dir: Path, dpi: int) -> Path:
    """Bar + line chart of mental-health-related incidents by year."""
    frame = pd.DataFrame(metrics["mental_health_summary"])

    fig, ax1 = plt.subplots(figsize=(9, 4))

    ax1.bar(frame["year"].astype(str), frame["n_incidents"], color="#AB47BC",
            alpha=0.8, edgecolor="white", width=0.65, label="N. Incidents")
    ax1.set_ylabel("N. Incidents", fontsize=10, color="#AB47BC")
    ax1.tick_params(axis="y", labelcolor="#AB47BC")

    ax2 = ax1.twinx()
    ax2.plot(frame["year"].astype(str), frame["proportion"] * 100,
             color="#E91E63", marker="o", linewidth=2, label="% of Total")
    ax2.set_ylabel("% of Total Incidents", fontsize=10, color="#E91E63")
    ax2.tick_params(axis="y", labelcolor="#E91E63")

    ax1.set_title("Mental-Health-Related Incidents by Year", fontsize=13, weight="bold", pad=10)
    ax1.set_xlabel("Year", fontsize=10)
    ax1.grid(alpha=0.3, linestyle="--", axis="y")
    ax1.spines[["top", "right"]].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    return _save_figure(fig, figures_dir, "mental_health", dpi)


def plot_negativity_by_principle(metrics: dict, config: dict, figures_dir: Path, dpi: int) -> Path:
    """Grid of monthly negativity-score evolution for top tags/principles."""
    frame = pd.DataFrame(metrics["negativity_by_principle"])
    primary_window = config["analysis"]["moving_average_windows"][0]

    principles = [col for col in frame.columns if col != "year_month" and "__ma" not in col]
    n_cols = 2
    n_rows = int(np.ceil(len(principles) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 4 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).reshape(-1)

    x_values = pd.to_datetime(frame["year_month"])
    for index, principle in enumerate(principles):
        ax = axes[index]
        ax.plot(
            x_values, frame[principle],
            marker="o", linestyle="--", linewidth=1,
            color="#EF5350", alpha=0.5, markersize=4, label="Monthly avg",
        )
        rolling_column = f"{principle}__ma{primary_window}"
        if rolling_column in frame.columns:
            ax.plot(
                x_values, frame[rolling_column],
                linewidth=2, color="#B71C1C",
                label=f"{primary_window}-month rolling avg",
            )
        ax.set_title(principle, fontsize=11, weight="bold")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[len(principles):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Negativity Score by Principle Over Time", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
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

    label = _auto_title(wc_config["name"])

    if not frequencies:
        logger.warning("Wordcloud '%s': no tokens after filtering — skipping", wc_config["name"])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=16)
        ax.axis("off")
        return _save_figure(fig, figures_dir, f"wordcloud_{wc_config['name']}", dpi)

    cloud = WordCloud(width=1400, height=800, background_color="white",
                      colormap="viridis", max_words=120)
    cloud.generate_from_frequencies(frequencies)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word Cloud — {label}", fontsize=13, weight="bold", pad=8)
    return _save_figure(fig, figures_dir, f"wordcloud_{wc_config['name']}", dpi)


# ---------------------------------------------------------------------------
# Figure orchestration
# ---------------------------------------------------------------------------

def generate_figures(df: pd.DataFrame, metrics: dict, config: dict) -> dict[str, Path]:
    """Generate all figures required for the results chapter."""
    figures_dir = Path(config["reporting"]["figures_dir"])
    dpi = config["reporting"]["figures_dpi"]

    figures: dict[str, Path] = {
        "regional_evolution":        plot_regional_evolution(metrics, figures_dir, dpi),
        "cumulative_concentration":  plot_cumulative_concentration(metrics, figures_dir, dpi),
        "principles_distribution":   plot_principles_distribution(metrics, config, figures_dir, dpi),
        "harm_types_distribution":   plot_harm_types_distribution(metrics, config, figures_dir, dpi),
        "stakeholders_distribution": plot_stakeholders_distribution(metrics, config, figures_dir, dpi),
        "vulnerable_groups":         plot_vulnerable_groups(metrics, figures_dir, dpi),
        "harm_chronology":           plot_harm_chronology(metrics, config, figures_dir, dpi),
        "top_industries":            plot_top_industries(metrics, config, figures_dir, dpi),
        "harm_severity_index":       plot_harm_severity_index(metrics, figures_dir, dpi),
        "mental_health":             plot_mental_health(metrics, figures_dir, dpi),
        "negativity_by_principle":   plot_negativity_by_principle(metrics, config, figures_dir, dpi),
    }

    for wc_config in config.get("wordclouds", [{"name": "general"}]):
        name = f"wordcloud_{wc_config['name']}"
        figures[name] = plot_wordcloud(df, config, figures_dir, dpi, wc_config)

    return figures


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------

# Human-readable section titles per figure
_FIGURE_TITLES: dict[str, str] = {
    "regional_evolution":        "Evolution of AI Incidents per Region",
    "cumulative_concentration":  "Cumulative Concentration of AI Incidents Over Time",
    "principles_distribution":   "AI Principles Affected by Incidents",
    "harm_types_distribution":   "Types of Harm Caused by Incidents",
    "stakeholders_distribution": "Distribution of Affected Stakeholders",
    "vulnerable_groups":         "Vulnerable Groups",
    "harm_chronology":           "Evolution of Harm Types Over Time",
    "top_industries":            "Top Industries by Incident Count",
    "harm_severity_index":       "Harm Severity Index by Year",
    "mental_health":             "Mental-Health-Related Incidents",
    "negativity_by_principle":   "Negativity Score by Principle Over Time",
}

# Maps figure names to their corresponding metrics key (for optional data tables)
_FIGURE_METRICS_MAP: dict[str, str] = {
    "regional_evolution":        "regional_evolution",
    "cumulative_concentration":  "cumulative_concentration",
    "principles_distribution":   "principles_distribution",
    "harm_types_distribution":   "harm_types_distribution",
    "stakeholders_distribution": "stakeholders_distribution",
    "vulnerable_groups":         "vulnerable_groups_distribution",
    "harm_chronology":           "harm_chronology",
    "top_industries":            "top_industries",
    "harm_severity_index":       "harm_severity_index",
    "mental_health":             "mental_health_summary",
    "negativity_by_principle":   "negativity_by_principle",
}

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

    if title:
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
        f"Total incidents analyzed: {summary['n_incidents']:,}\n"
        f"Period covered: {summary['year_range'][0]}-{summary['year_range'][1]}\n"
        f"Regions included: {', '.join(summary['regions'])}",
    )


def _write_methodology(pdf, config: dict) -> None:
    """Render the methodology section."""
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Methodology", ln=True)

    reporting = config.get("reporting", {})
    source_label = reporting.get("data_source_label", "AI Incidents dataset")
    year_range = config["filters"]["year_range"]
    regions = config["filters"]["regions"]
    sentiment_cfg = config["sentiment"]

    pdf.set_font("Helvetica", "", 11)
    text = (
        f"Data source: {source_label}.\n"
        f"Temporal scope: {year_range[0]}-{year_range[1]}.\n"
        f"Regions: {', '.join(regions)}.\n\n"
        "Preprocessing: multi-label parsing, temporal and geographic filtering, "
        "harm-type recategorization, and multi-label binarization.\n\n"
        "NLP: Unicode normalization, tokenization, WordNet lemmatization, "
        "stopword removal and a configurable normalization map.\n\n"
        f"Sentiment analysis backend: {sentiment_cfg['backend']}. "
        f"Negativity scores are mapped to categorical labels using thresholds "
        f"high={sentiment_cfg['thresholds']['high']} (Alta), "
        f"medium={sentiment_cfg['thresholds']['medium']} (Media), otherwise Baja."
    )
    pdf.multi_cell(0, 6, text)


def _png_to_jpeg_buf(path: Path, max_px: int = 1400) -> io.BytesIO:
    """Convert a PNG to a JPEG BytesIO buffer for fpdf2.

    fpdf2 can hang on large PNGs with alpha channels. Converting to JPEG
    (RGB, no alpha, capped resolution) avoids the issue entirely.
    """
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if img.width > max_px:
        ratio = max_px / img.width
        img = img.resize((max_px, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return buf


def generate_report(metrics: dict, figures: dict[str, Path], config: dict) -> Path:
    """Assemble the executive PDF report."""
    from fpdf import FPDF

    reporting = config["reporting"]
    show_tables = reporting.get("show_data_tables", False)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Cover ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 14, reporting["report_title"], ln=True, align="C")
    pdf.ln(6)
    _write_summary(pdf, metrics)

    # --- Methodology ---
    pdf.add_page()
    _write_methodology(pdf, config)

    # --- Figures (+ optional companion tables) ---
    for name, path in figures.items():
        pdf.add_page()
        section_title = _FIGURE_TITLES.get(name) or _auto_title(name)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, section_title, ln=True)
        pdf.image(_png_to_jpeg_buf(path), w=pdf.epw)

        if show_tables:
            pdf.ln(3)
            metrics_key = _FIGURE_METRICS_MAP.get(name)
            if metrics_key and metrics_key in metrics:
                records = metrics[metrics_key]
                columns = _table_columns(records)
                _write_table(pdf, "", records, columns)

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
