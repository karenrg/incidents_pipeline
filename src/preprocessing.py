"""Cleaning, temporal/geographic standardization and multi-label handling."""

import ast
import logging

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)


def preprocess(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Run the full preprocessing stage.

    Applies, in order: multi-label string-to-list parsing, temporal
    standardization and year-range filtering, geographic filtering,
    text imputation, harm-type recategorization, and multi-label
    binarization (ML branch).

    Args:
        df: DataFrame with the internal schema produced by
            ``ingestion.load_and_validate``.
        config: Parsed pipeline configuration (``params.yaml``).

    Returns:
        Preprocessed DataFrame, filtered to the configured year range and
        regions, with multi-label columns converted to Python lists, derived
        temporal columns (``year``, ``year_month``), a ``harmtype_category``
        column, and one-hot multi-label columns prefixed with ``mlb_``.
    """
    df = parse_list_columns(df, config["data"]["multilabel_columns"])
    df = standardize_dates(df, config)
    df = filter_regions(df, config)
    df = impute_text_fields(df, ["text_data"])
    df = recategorize_harm(df, config)
    df = binarize_multilabel_columns(df, config["data"]["multilabel_columns"])
    return df


def parse_list_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert string representations of lists into real Python lists.

    Multi-valued columns (e.g. ``tags_list``, ``industries``, ``harmed``,
    ``harmlevel``, ``harmtype``) arrive as strings such as
    ``"['Robustness', 'Human Rights']"``. This converts each cell to an
    actual list, leaving non-list-like values wrapped in a single-element
    list and missing values as empty lists.

    Args:
        df: Input DataFrame.
        columns: Names of the multi-label columns to convert.

    Returns:
        Copy of ``df`` with the given columns converted to lists.
    """
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            logger.warning("Multi-label column '%s' not found; skipping", column)
            continue
        df[column] = df[column].apply(_safe_literal_eval)
    return df


def _safe_literal_eval(value: object) -> list:
    """Safely evaluate a value into a Python list.

    Args:
        value: Cell value, expected to be a list, a string representation
            of a list, a scalar, or a missing value.

    Returns:
        A Python list: parsed from the string, the value itself if already
        a list, ``[]`` for missing values, or ``[value]`` for any other
        scalar.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = ast.literal_eval(stripped)
                return parsed if isinstance(parsed, list) else [parsed]
            except (ValueError, SyntaxError):
                logger.debug("Could not parse list-like string: %r", value)
                return [value]
        return [value]
    if pd.isna(value):
        return []
    return [value]


def standardize_dates(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Convert ``event_date`` to datetime and filter by configured year range.

    Adds ``year`` (int) and ``year_month`` (monthly ``Period``) columns and
    drops rows whose year falls outside ``config['filters']['year_range']``
    or whose date could not be parsed.

    Args:
        df: Input DataFrame with an ``event_date`` column.
        config: Parsed pipeline configuration (``params.yaml``).

    Returns:
        Filtered copy of ``df`` with standardized temporal columns.
    """
    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

    n_unparseable = int(df["event_date"].isna().sum())
    if n_unparseable:
        logger.warning("%d rows have an unparseable event_date and will be dropped", n_unparseable)

    df["year"] = df["event_date"].dt.year
    df["year_month"] = df["event_date"].dt.to_period("M")

    year_range = config["filters"].get("year_range") or []
    if len(year_range) == 2:
        year_min, year_max = year_range
        mask = df["year"].between(year_min, year_max)
        n_dropped = int((~mask).sum())
        if n_dropped:
            logger.info(
                "Dropping %d rows outside year range [%d, %d]", n_dropped, year_min, year_max
            )
        return df[mask].copy()

    logger.info("No year_range filter applied — keeping all %d rows", len(df))
    return df.copy()


def filter_regions(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Filter rows to the regions listed in ``config['filters']['regions']``.

    Args:
        df: Input DataFrame with a ``geo_zone`` column.
        config: Parsed pipeline configuration (``params.yaml``).

    Returns:
        Filtered copy of ``df`` containing only rows whose ``geo_zone`` is
        in the configured region list.
    """
    regions = config["filters"].get("regions") or []
    if not regions:
        logger.info("No regions filter applied — keeping all %d rows", len(df))
        return df.copy()

    if "geo_zone" not in df.columns:
        logger.warning("geo_zone column not found — skipping region filter")
        return df.copy()

    mask = df["geo_zone"].isin(regions)
    n_dropped = int((~mask).sum())
    if n_dropped:
        logger.info("Dropping %d rows outside regions %s", n_dropped, regions)

    return df[mask].copy()


def impute_text_fields(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Replace missing values in text columns with empty strings.

    Args:
        df: Input DataFrame.
        columns: Names of text columns to impute.

    Returns:
        Copy of ``df`` with missing values in ``columns`` replaced by ``""``.
    """
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str)
    return df


def recategorize_harm(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Group fine-grained harm types into broader categories.

    Reads the harm-type column name from
    ``config['analysis']['columns']['harm_type']`` and adds a
    ``<column>_category`` column where every element is replaced by its
    aggregated category according to
    ``config['analysis']['harm_aggregations']``; values not part of any
    aggregation are kept unchanged.

    Args:
        df: Input DataFrame with the configured harm-type column.
        config: Parsed pipeline configuration (``params.yaml``).

    Returns:
        Copy of ``df`` with the additional category column.
    """
    df = df.copy()
    harm_col = config["analysis"]["columns"]["harm_type"]
    aggregations = config["analysis"].get("harm_aggregations", {})

    reverse_map = {}
    for category, members in aggregations.items():
        for member in members:
            reverse_map[member] = category

    df[f"{harm_col}_category"] = df[harm_col].apply(
        lambda items: [reverse_map.get(item, item) for item in items]
        if isinstance(items, list)
        else items
    )
    return df


def explode_for_descriptive(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Explode a multi-label column for frequency-based descriptive analysis.

    Each list element becomes its own row so that ``value_counts`` produces
    unbiased frequencies (the descriptive branch of the dual multi-label
    handling).

    Args:
        df: Input DataFrame.
        column: Name of the multi-label (list-valued) column to explode.

    Returns:
        Copy of ``df`` exploded on ``column`` with empty/missing entries
        removed.

    Raises:
        KeyError: If ``column`` is not present in ``df``.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found for explode")

    exploded = df.explode(column)
    exploded = exploded[exploded[column].notna()]
    return exploded


def binarize_multilabel_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Encode multi-label columns with ``MultiLabelBinarizer`` (ML branch).

    For each column, adds one binary indicator column per observed label,
    named ``mlb_<column>__<label>``.

    Args:
        df: Input DataFrame with list-valued columns.
        columns: Names of the multi-label columns to encode.

    Returns:
        Copy of ``df`` with the additional one-hot encoded columns appended.
    """
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            continue

        values = df[column].apply(lambda x: x if isinstance(x, list) else [])
        binarizer = MultiLabelBinarizer()
        encoded = binarizer.fit_transform(values)

        encoded_df = pd.DataFrame(
            encoded,
            columns=[f"mlb_{column}__{label}" for label in binarizer.classes_],
            index=df.index,
        )
        df = pd.concat([df, encoded_df], axis=1)
    return df
