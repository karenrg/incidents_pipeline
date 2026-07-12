"""Data ingestion and validation for the OECD AI Incidents Monitor dataset."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Canonical column names produced by `load_and_validate` after applying the
# YAML-driven `column_mapping`.
INTERNAL_SCHEMA_COLUMNS = ["text_data", "event_date", "geo_zone", "tags_list"]

# Encodings attempted (in order) when reading the source CSV.
_CANDIDATE_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1"]


def load_and_validate(config: dict) -> pd.DataFrame:
    """Load the incidents CSV and map it to the internal schema.

    Reads the CSV referenced by ``config['data']['source_path']`` handling
    encoding/separator variations, renames the columns listed in
    ``config['data']['column_mapping']`` to the internal schema
    (``text_data``, ``event_date``, ``geo_zone``, ``tags_list``), and logs
    warnings for critical columns that are fully null and for duplicate rows.

    Args:
        config: Parsed pipeline configuration (``params.yaml``).

    Returns:
        DataFrame with the internal schema columns alongside the original
        dataset columns.

    Raises:
        FileNotFoundError: If the source CSV does not exist.
        KeyError: If a column referenced in ``column_mapping`` is missing
            from the source dataset.
    """
    data_config = config["data"]
    source_path = Path(data_config["source_path"])

    df = _read_csv_robust(source_path)
    df = _apply_column_mapping(df, data_config["column_mapping"])

    _check_critical_nulls(df, INTERNAL_SCHEMA_COLUMNS)
    _report_duplicates(df)

    return df


def _read_csv_robust(path: Path) -> pd.DataFrame:
    """Read a CSV file trying several encodings and sniffing the separator.

    Args:
        path: Path to the CSV file.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the path does not exist.
        UnicodeDecodeError: If none of the candidate encodings can decode
            the file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Source data file not found: {path}")

    last_error: UnicodeDecodeError | None = None
    for encoding in _CANDIDATE_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=encoding, sep=None, engine="python")
            logger.info("Loaded %s rows from %s (encoding=%s)", len(df), path, encoding)
            return df
        except UnicodeDecodeError as error:
            last_error = error
            continue

    raise last_error


def _apply_column_mapping(df: pd.DataFrame, column_mapping: dict) -> pd.DataFrame:
    """Rename source columns to the internal schema.

    Args:
        df: Raw DataFrame as loaded from the source CSV.
        column_mapping: Mapping of ``internal_name -> source_column_name``.

    Returns:
        DataFrame with the mapped columns renamed in place (original,
        unmapped columns are preserved).

    Raises:
        KeyError: If a source column referenced by the mapping is not
            present in ``df``.
    """
    rename_map = {}
    for internal_name, source_name in column_mapping.items():
        if source_name not in df.columns:
            raise KeyError(
                f"Column mapping error: source column '{source_name}' "
                f"(for internal field '{internal_name}') not found in dataset. "
                f"Available columns: {df.columns.tolist()}"
            )
        rename_map[source_name] = internal_name

    return df.rename(columns=rename_map)


def _check_critical_nulls(df: pd.DataFrame, columns: list[str]) -> None:
    """Log a warning for critical columns that are entirely null.

    Args:
        df: DataFrame to validate.
        columns: Names of critical columns to check.
    """
    for column in columns:
        if column not in df.columns:
            logger.warning("Critical column '%s' not present in dataset", column)
            continue

        null_fraction = df[column].isna().mean()
        if null_fraction == 1.0:
            logger.warning("Critical column '%s' is 100%% null", column)
        elif null_fraction > 0:
            logger.info("Column '%s' has %.1f%% null values", column, null_fraction * 100)


def _report_duplicates(df: pd.DataFrame) -> None:
    """Log the number of fully duplicated rows found in the dataset.

    Args:
        df: DataFrame to inspect.
    """
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        logger.warning("Found %d duplicate rows in the dataset", duplicate_count)
    else:
        logger.info("No duplicate rows found")
