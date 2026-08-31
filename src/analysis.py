"""Aggregations and statistical metrics computed over the processed dataset."""

import ast
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_list_value(v: object) -> list:
    """Convert a cell value to a list, handling string representations like \"['x', 'y']\"."""
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("["):
            try:
                parsed = ast.literal_eval(s)
                return parsed if isinstance(parsed, list) else [parsed]
            except (ValueError, SyntaxError):
                pass
        return [s] if s else []
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    return [v]

from .nlp import token_frequencies
from .preprocessing import explode_for_descriptive

logger = logging.getLogger(__name__)


def _col(config: dict, role: str) -> str:
    """Return the DataFrame column name mapped to an analytical *role*."""
    return config["analysis"]["columns"][role]


def _has_col(df: pd.DataFrame, config: dict, role: str) -> bool:
    """Return True if the column for *role* is present in *df*."""
    return _col(config, role) in df.columns


def regional_evolution(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Count incidents per year and region."""
    if not _has_col(df, config, "year") or not _has_col(df, config, "region"):
        return pd.DataFrame()
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
    if not _has_col(df, config, "tags"):
        return pd.DataFrame()
    top_n = config["analysis"]["top_n_principles"]
    tags = _col(config, "tags")

    exploded = explode_for_descriptive(df, tags)
    top_principles = exploded[tags].value_counts().nlargest(top_n).index
    exploded = exploded[exploded[tags].isin(top_principles)]

    if not _has_col(df, config, "region"):
        # Without region, return flat counts instead of region breakdown
        return exploded[tags].value_counts().rename_axis(tags).reset_index(name="count")

    region = _col(config, "region")
    return exploded.groupby([region, tags]).size().unstack(fill_value=0)


def harm_types_distribution(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Distribution of harm types per region, excluding configured values."""
    if not _has_col(df, config, "harm_type"):
        return pd.DataFrame()
    harm_type = _col(config, "harm_type")
    exclude = config["analysis"].get("exclude_values", [])

    exploded = explode_for_descriptive(df, harm_type)
    exploded = exploded[~exploded[harm_type].isin(exclude)]

    if not _has_col(df, config, "region"):
        return exploded[harm_type].value_counts().rename_axis(harm_type).reset_index(name="count")

    region = _col(config, "region")
    return exploded.groupby([region, harm_type]).size().unstack(fill_value=0)


def top_industries(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Rank industries by overall incident count."""
    if not _has_col(df, config, "industries"):
        return pd.DataFrame(columns=["industry", "count"])
    top_n = config["analysis"]["top_n_industries"]
    industries = _col(config, "industries")

    exploded = explode_for_descriptive(df, industries)
    counts = exploded[industries].value_counts().nlargest(top_n)
    return counts.rename_axis("industry").reset_index(name="count")


def stakeholders_distribution(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Distribution of the most affected stakeholder groups per region."""
    if not _has_col(df, config, "harmed"):
        return pd.DataFrame()
    top_n = config["analysis"]["top_n_harmed"]
    harmed = _col(config, "harmed")
    exclude = config["analysis"].get("exclude_values", [])

    exploded = explode_for_descriptive(df, harmed)
    exploded = exploded[~exploded[harmed].isin(exclude)]

    if not _has_col(df, config, "region"):
        counts = exploded[harmed].value_counts().nlargest(top_n)
        return counts.rename_axis(harmed).reset_index(name="count")

    region = _col(config, "region")
    counts = exploded.groupby([region, harmed]).size().unstack(fill_value=0)
    top_groups = counts.sum().nlargest(top_n).index
    return counts[top_groups]


def vulnerable_groups_distribution(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Count incidents affecting each configured vulnerable group."""
    vulnerable_groups = config["analysis"]["vulnerable_groups"]
    if not vulnerable_groups or not _has_col(df, config, "harmed"):
        return pd.DataFrame(columns=["group", "count"])
    harmed = _col(config, "harmed")

    exploded = explode_for_descriptive(df, harmed)
    exploded = exploded[exploded[harmed].isin(vulnerable_groups)]

    counts = exploded[harmed].value_counts().reindex(vulnerable_groups, fill_value=0)
    return counts.rename_axis("group").reset_index(name="count")


def harm_chronology(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Yearly counts of configured harm types (e.g. physical vs. psychological)."""
    if not _has_col(df, config, "year"):
        return pd.DataFrame()
    year = _col(config, "year")
    harm_type = _col(config, "harm_type")
    types = config["analysis"]["harm_chronology_types"]

    exploded = explode_for_descriptive(df, harm_type)
    exploded = exploded[exploded[harm_type].isin(types)]

    counts = exploded.groupby([year, harm_type]).size().unstack(fill_value=0)
    return counts.reindex(columns=types, fill_value=0).sort_index()


def negativity_by_principle_yearly(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Yearly average negativity score per top OECD principle, with 3-year rolling average.

    Mirrors the paper's Figure 13: temporal evolution of negativity across principles.
    """
    if not _has_col(df, config, "tags") or not _has_col(df, config, "year"):
        return pd.DataFrame()
    score_col = _col(config, "sentiment_score")
    if score_col not in df.columns:
        return pd.DataFrame()

    top_n = config["analysis"]["top_n_principles"]
    tags  = _col(config, "tags")
    year  = _col(config, "year")

    exploded = explode_for_descriptive(df, tags)
    top_principles = exploded[tags].value_counts().nlargest(top_n).index.tolist()
    exploded = exploded[exploded[tags].isin(top_principles)]

    yearly_avg = (
        exploded.groupby([year, tags])[score_col]
        .mean()
        .unstack(fill_value=np.nan)
        .sort_index()
    )

    # 3-year rolling average (as in the paper)
    rolling3 = yearly_avg.rolling(window=3, min_periods=1).mean()
    rolling3.columns = [f"{c}__ma3" for c in rolling3.columns]
    result = pd.concat([yearly_avg, rolling3], axis=1)
    result.index = result.index.astype(str)
    return result.rename_axis("year").reset_index()


def wc_tokens_by_col(df: pd.DataFrame, config: dict, top_n: int = 150) -> dict:
    """Pre-compute token frequency dicts per category value for the wordcloud filter.

    Returns a flat dict keyed as "col_role::value" (e.g. "geo_zone::North America"),
    plus an "all" key for the unfiltered wordcloud.
    """
    tokens_col = _col(config, "tokens")
    if tokens_col not in df.columns:
        return {}

    roles = ["region", "tags", "harm_type", "industries", "harmed"]
    result: dict = {}

    # Overall (no filter)
    counter = token_frequencies(df, token_column=tokens_col)
    result["all"] = dict(counter.most_common(top_n))

    for role in roles:
        col = _col(config, role)
        if col not in df.columns:
            continue
        # Explode if values are lists
        sample = df[col].dropna().iloc[:5] if len(df) >= 5 else df[col].dropna()
        is_list = sample.apply(lambda v: isinstance(v, list)).any()
        src = df.explode(col) if is_list else df
        for val, grp in src.groupby(col):
            if not val or (isinstance(val, float) and np.isnan(val)):
                continue
            cnt = token_frequencies(grp, token_column=tokens_col)
            if cnt:
                result[f"{col}::{val}"] = dict(cnt.most_common(top_n))

    return result


def negativity_by_principle(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Monthly average negativity score for the most frequent tags/principles."""
    if not _has_col(df, config, "tags") or not _has_col(df, config, "year_month"):
        return pd.DataFrame()
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
    if not _has_col(df, config, "year") or not _has_col(df, config, "harm_level"):
        return pd.DataFrame()
    severity_map = config["analysis"]["harm_level_severity"]
    year = _col(config, "year")
    harm_level = _col(config, "harm_level")

    def _max_severity(levels: object) -> float:
        parsed = _parse_list_value(levels)
        if not parsed:
            return np.nan
        return max(severity_map.get(str(level).strip(), 0) for level in parsed)

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
    if not _has_col(df, config, "year"):
        return pd.DataFrame()
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
    has_year   = _has_col(df, config, "year")
    has_region = _has_col(df, config, "region")
    year   = _col(config, "year")
    region = _col(config, "region")
    regional = regional_evolution(df, config)

    summary: dict = {"n_incidents": int(len(df))}
    if has_year:
        summary["year_range"] = [int(df[year].min()), int(df[year].max())]
    if has_region:
        summary["regions"] = sorted(df[region].dropna().unique().tolist())

    metrics = {
        "summary": summary,
        "regional_evolution": regional.reset_index().to_dict(orient="records") if not regional.empty else [],
        "cumulative_concentration": (
            cumulative_concentration(regional).reset_index().to_dict(orient="records")
            if not regional.empty else []
        ),
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
        "negativity_by_principle": (
            negativity_by_principle(df, config).to_dict(orient="records")
            if _col(config, "sentiment_score") in df.columns
            else []
        ),
        "negativity_by_principle_yearly": (
            negativity_by_principle_yearly(df, config).to_dict(orient="records")
            if _col(config, "sentiment_score") in df.columns
            else []
        ),
        "wc_tokens": wc_tokens_by_col(df, config),
        "harm_severity_index": harm_severity_index(df, config).to_dict(orient="records"),
        "mental_health_summary": mental_health_summary(df, config).to_dict(orient="records"),
        "top_tokens": top_tokens(df, config).to_dict(orient="records"),
    }

    reports_dir = Path(config["reporting"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Metrics JSON
    output_path = reports_dir / "metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=_json_default)
    logger.info("Metrics written to %s", output_path)

    # Processed dataset CSV — same data as the parquet, easier to open in Excel/Sheets
    csv_path = reports_dir / "incidents_processed.csv"
    # Exclude ML-only columns (mlb_*) and heavy list columns not useful in CSV
    csv_cols = [
        c for c in df.columns
        if not c.startswith("mlb_") and c not in ("tokens", "year_month")
    ]
    # Restore original column names from column_mapping (internal → original CSV name)
    rename_map = config.get("data", {}).get("column_mapping", {})
    df[csv_cols].rename(columns=rename_map).to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("Processed CSV written to %s", csv_path)

    return metrics
