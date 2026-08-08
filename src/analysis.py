"""Aggregations and statistical metrics computed over the processed dataset."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .nlp import token_frequencies
from .preprocessing import explode_for_descriptive

logger = logging.getLogger(__name__)


def _col(config: dict, role: str) -> str:
    """Return the DataFrame column name mapped to an analytical *role*."""
    return config["analysis"]["columns"][role]


def regional_evolution(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Count incidents per year and region."""
    year = _col(config, "year")
    region = _col(config, "region")
    return df.groupby([year, region]).size().unstack(fill_value=0).sort_index()


def cumulative_concentration(regional_counts: pd.DataFrame) -> pd.Series:
    """Cumulative percentage of incidents over time."""
    totals_per_year = regional_counts.sum(axis=1)
    cumulative_pct = (totals_per_year.cumsum() / totals_per_year.sum() * 100).round(2)
    return cumulative_pct.rename("cumulative_pct")


def principles_distribution(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Distribution of the most frequent tags/principles per region."""
    top_n = config["analysis"]["top_n_principles"]
    region = _col(config, "region")
    tags = _col(config, "tags")

    exploded = explode_for_descriptive(df, tags)
    top_principles = exploded[tags].value_counts().nlargest(top_n).index
    exploded = exploded[exploded[tags].isin(top_principles)]

    return exploded.groupby([region, tags]).size().unstack(fill_value=0)


def harm_types_distribution(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Distribution of harm types per region, excluding configured values."""
    region = _col(config, "region")
    harm_type = _col(config, "harm_type")
    exclude = config["analysis"].get("exclude_values", [])

    exploded = explode_for_descriptive(df, harm_type)
    exploded = exploded[~exploded[harm_type].isin(exclude)]
    return exploded.groupby([region, harm_type]).size().unstack(fill_value=0)


def top_industries(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Rank industries by overall incident count."""
    top_n = config["analysis"]["top_n_industries"]
    industries = _col(config, "industries")

    exploded = explode_for_descriptive(df, industries)
    counts = exploded[industries].value_counts().nlargest(top_n)
    return counts.rename_axis("industry").reset_index(name="count")


def stakeholders_distribution(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Distribution of the most affected stakeholder groups per region."""
    top_n = config["analysis"]["top_n_harmed"]
    region = _col(config, "region")
    harmed = _col(config, "harmed")
    exclude = config["analysis"].get("exclude_values", [])

    exploded = explode_for_descriptive(df, harmed)
    exploded = exploded[~exploded[harmed].isin(exclude)]

    counts = exploded.groupby([region, harmed]).size().unstack(fill_value=0)
    top_groups = counts.sum().nlargest(top_n).index
    return counts[top_groups]


def vulnerable_groups_distribution(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Count incidents affecting each configured vulnerable group."""
    vulnerable_groups = config["analysis"]["vulnerable_groups"]
    harmed = _col(config, "harmed")

    exploded = explode_for_descriptive(df, harmed)
    exploded = exploded[exploded[harmed].isin(vulnerable_groups)]

    counts = exploded[harmed].value_counts().reindex(vulnerable_groups, fill_value=0)
    return counts.rename_axis("group").reset_index(name="count")


def harm_chronology(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Yearly counts of configured harm types (e.g. physical vs. psychological)."""
    year = _col(config, "year")
    harm_type = _col(config, "harm_type")
    types = config["analysis"]["harm_chronology_types"]

    exploded = explode_for_descriptive(df, harm_type)
    exploded = exploded[exploded[harm_type].isin(types)]

    counts = exploded.groupby([year, harm_type]).size().unstack(fill_value=0)
    return counts.reindex(columns=types, fill_value=0).sort_index()


def negativity_by_principle(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Monthly average negativity score for the most frequent tags/principles."""
    top_n = config["analysis"]["top_n_principles"]
    windows = config["analysis"]["moving_average_windows"]
    tags = _col(config, "tags")
    year_month = _col(config, "year_month")
    score_col = _col(config, "sentiment_score")

    exploded = explode_for_descriptive(df, tags)
    top_principles = exploded[tags].value_counts().nlargest(top_n).index.tolist()
    exploded = exploded[exploded[tags].isin(top_principles)]

    monthly_avg = (
        exploded.groupby([year_month, tags])[score_col]
        .mean()
        .unstack(fill_value=np.nan)
        .sort_index()
    )

    result = monthly_avg.copy()
    for window in windows:
        rolling = monthly_avg.rolling(window=window, min_periods=1).mean()
        rolling.columns = [f"{column}__ma{window}" for column in rolling.columns]
        result = pd.concat([result, rolling], axis=1)

    result.index = result.index.astype(str)
    return result.rename_axis("year_month").reset_index()


def harm_severity_index(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Average yearly harm-severity index based on harm level."""
    severity_map = config["analysis"]["harm_level_severity"]
    year = _col(config, "year")
    harm_level = _col(config, "harm_level")

    def _max_severity(levels: object) -> float:
        if isinstance(levels, str):
            levels = [levels] if levels.strip() else []
        if not isinstance(levels, list) or not levels:
            return np.nan
        return max(severity_map.get(level, 0) for level in levels)

    severity = df[harm_level].apply(_max_severity)
    return (
        severity.groupby(df[year])
        .mean()
        .round(2)
        .rename("avg_severity")
        .rename_axis("year")
        .reset_index()
    )


def mental_health_summary(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Yearly count and proportion of mental-health-related incidents."""
    year = _col(config, "year")
    mh_flag = _col(config, "mental_health_flag")

    summary = df.groupby(year)[mh_flag].agg(n_incidents="sum", n_total="count")
    summary["proportion"] = (summary["n_incidents"] / summary["n_total"]).round(4)
    return summary.reset_index()


def top_tokens(df: pd.DataFrame, config: dict, n: int = 50) -> pd.DataFrame:
    """Rank the most frequent tokens across the dataset."""
    tokens_col = _col(config, "tokens")
    counter = token_frequencies(df, token_column=tokens_col)
    return pd.DataFrame(counter.most_common(n), columns=["token", "frequency"])


def _json_default(value: object) -> object:
    """Fallback JSON encoder for numpy/pandas scalar types."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def run_analysis(df: pd.DataFrame, config: dict) -> dict:
    """Compute all descriptive metrics and write the JSON metrics store."""
    year = _col(config, "year")
    region = _col(config, "region")
    regional = regional_evolution(df, config)

    metrics = {
        "summary": {
            "n_incidents": int(len(df)),
            "year_range": [int(df[year].min()), int(df[year].max())],
            "regions": sorted(df[region].unique().tolist()),
        },
        "regional_evolution": regional.reset_index().to_dict(orient="records"),
        "cumulative_concentration": cumulative_concentration(regional)
        .reset_index()
        .to_dict(orient="records"),
        "principles_distribution": principles_distribution(df, config)
        .reset_index()
        .to_dict(orient="records"),
        "harm_types_distribution": harm_types_distribution(df, config)
        .reset_index()
        .to_dict(orient="records"),
        "top_industries": top_industries(df, config).to_dict(orient="records"),
        "stakeholders_distribution": stakeholders_distribution(df, config)
        .reset_index()
        .to_dict(orient="records"),
        "vulnerable_groups_distribution": vulnerable_groups_distribution(df, config).to_dict(
            orient="records"
        ),
        "harm_chronology": harm_chronology(df, config).reset_index().to_dict(orient="records"),
        "negativity_by_principle": negativity_by_principle(df, config).to_dict(orient="records"),
        "harm_severity_index": harm_severity_index(df, config).to_dict(orient="records"),
        "mental_health_summary": mental_health_summary(df, config).to_dict(orient="records"),
        "top_tokens": top_tokens(df, config).to_dict(orient="records"),
    }

    reports_dir = Path(config["reporting"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=_json_default)
    logger.info("Metrics written to %s", output_path)

    return metrics
