"""Unit tests for src.ingestion."""

import pandas as pd
import pytest

from src.ingestion import (
    INTERNAL_SCHEMA_COLUMNS,
    _check_critical_nulls,
    _report_duplicates,
    load_and_validate,
)


@pytest.fixture
def config(tmp_path):
    """Minimal configuration pointing to a small synthetic CSV."""
    csv_path = tmp_path / "incidents.csv"
    pd.DataFrame(
        {
            "summary": ["Incident A", "Incident B", "Incident B"],
            "date": ["2021-01-01", "2022-06-15", "2022-06-15"],
            "region": ["Europe", "Asia", "Asia"],
            "principles": ["['Robustness']", "['Fairness']", "['Fairness']"],
            "industries": ["['Health']", "['Finance']", "['Finance']"],
            "harmed": ["['Workers']", "['Children']", "['Children']"],
            "harmlevel": ["['Hazard']", "['Injury']", "['Injury']"],
            "harmtype": ["['Physical']", "['Psychological']", "['Psychological']"],
        }
    ).to_csv(csv_path, index=False)

    return {
        "data": {
            "source_path": str(csv_path),
            "column_mapping": {
                "text_data": "summary",
                "event_date": "date",
                "geo_zone": "region",
                "tags_list": "principles",
            },
        }
    }


def test_load_and_validate_renames_columns_to_internal_schema(config):
    df = load_and_validate(config)

    for column in INTERNAL_SCHEMA_COLUMNS:
        assert column in df.columns

    # Untouched OECD-specific columns must remain available.
    assert "industries" in df.columns
    assert "harmtype" in df.columns


def test_load_and_validate_missing_file_raises(config):
    config["data"]["source_path"] = "does_not_exist.csv"

    with pytest.raises(FileNotFoundError):
        load_and_validate(config)


def test_load_and_validate_missing_mapping_column_raises(config):
    config["data"]["column_mapping"]["text_data"] = "does_not_exist"

    with pytest.raises(KeyError):
        load_and_validate(config)


def test_check_critical_nulls_warns_on_fully_null_column(caplog):
    df = pd.DataFrame({"text_data": [None, None]})

    with caplog.at_level("WARNING"):
        _check_critical_nulls(df, ["text_data"])

    assert "100%" in caplog.text


def test_report_duplicates_warns_when_duplicates_present(caplog):
    df = pd.DataFrame({"a": [1, 1], "b": [2, 2]})

    with caplog.at_level("WARNING"):
        _report_duplicates(df)

    assert "duplicate" in caplog.text.lower()
