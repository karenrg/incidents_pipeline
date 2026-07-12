"""Unit tests for src.preprocessing."""

import pandas as pd
import pytest

from src.preprocessing import (
    _safe_literal_eval,
    binarize_multilabel_columns,
    explode_for_descriptive,
    filter_regions,
    impute_text_fields,
    parse_list_columns,
    recategorize_harm,
    standardize_dates,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("['Robustness', 'Fairness']", ["Robustness", "Fairness"]),
        ([], []),
        (["already_list"], ["already_list"]),
        (float("nan"), []),
        ("Robustness", ["Robustness"]),
        ("[invalid", ["[invalid"]),
    ],
)
def test_safe_literal_eval(value, expected):
    assert _safe_literal_eval(value) == expected


def test_parse_list_columns_converts_string_lists():
    df = pd.DataFrame({"tags_list": ["['A', 'B']", "['C']"]})

    result = parse_list_columns(df, ["tags_list"])

    assert result["tags_list"].tolist() == [["A", "B"], ["C"]]


def test_parse_list_columns_warns_on_missing_column(caplog):
    df = pd.DataFrame({"other": [1, 2]})

    with caplog.at_level("WARNING"):
        result = parse_list_columns(df, ["tags_list"])

    assert "tags_list" not in result.columns
    assert "tags_list" in caplog.text


def test_standardize_dates_adds_columns_and_filters_year_range():
    df = pd.DataFrame({"event_date": ["2010-01-01", "2020-01-01", "not-a-date"]})
    config = {"filters": {"year_range": [2015, 2025]}}

    result = standardize_dates(df, config)

    assert result["year"].tolist() == [2020]
    assert "year_month" in result.columns


def test_filter_regions_keeps_only_configured_regions():
    df = pd.DataFrame({"geo_zone": ["Europe", "Asia", "Africa"]})
    config = {"filters": {"regions": ["Europe", "Asia"]}}

    result = filter_regions(df, config)

    assert sorted(result["geo_zone"].unique()) == ["Asia", "Europe"]


def test_impute_text_fields_fills_missing_with_empty_string():
    df = pd.DataFrame({"text_data": ["hello", None]})

    result = impute_text_fields(df, ["text_data"])

    assert result["text_data"].tolist() == ["hello", ""]


def test_recategorize_harm_groups_physical_and_psychological_into_health():
    df = pd.DataFrame({"harmtype": [["Physical", "Privacy"], ["Psychological"]]})
    config = {
        "analysis": {
            "columns": {"harm_type": "harmtype"},
            "harm_aggregations": {"Health": ["Physical", "Psychological"]},
        }
    }

    result = recategorize_harm(df, config)

    assert result["harmtype_category"].tolist() == [["Health", "Privacy"], ["Health"]]


def test_explode_for_descriptive_expands_list_column():
    df = pd.DataFrame({"tags_list": [["A", "B"], ["C"]]})

    result = explode_for_descriptive(df, "tags_list")

    assert result["tags_list"].tolist() == ["A", "B", "C"]


def test_explode_for_descriptive_missing_column_raises():
    df = pd.DataFrame({"other": [1]})

    with pytest.raises(KeyError):
        explode_for_descriptive(df, "tags_list")


def test_binarize_multilabel_columns_creates_one_hot_columns():
    df = pd.DataFrame({"tags_list": [["A", "B"], ["B"]]})

    result = binarize_multilabel_columns(df, ["tags_list"])

    assert result["mlb_tags_list__A"].tolist() == [1, 0]
    assert result["mlb_tags_list__B"].tolist() == [1, 1]
