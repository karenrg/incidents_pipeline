"""Unit tests for src.sentiment."""

import json

import pandas as pd
import pytest

from src.sentiment import (
    MLSentiment,
    build_analyzer,
    classify_sentiment,
    evaluate_sentiment,
    run_sentiment,
)

THRESHOLDS = {"high": 0.7, "medium": 0.4}

POSITIVE_TEXTS = [
    "This is a wonderful and safe AI system that helps everyone.",
    "The new feature improves accuracy and is great for users.",
]
NEGATIVE_TEXTS = [
    "A terrible disaster caused severe harm and injuries to many people.",
    "This horrible incident led to widespread fear and damage.",
]


@pytest.mark.parametrize(
    "score, expected",
    [
        (0.9, "Alta"),
        (0.7, "Alta"),
        (0.5, "Media"),
        (0.4, "Media"),
        (0.1, "Baja"),
    ],
)
def test_classify_sentiment_thresholds(score, expected):
    assert classify_sentiment(score, THRESHOLDS) == expected


def test_build_analyzer_unknown_backend_raises():
    config = {"sentiment": {"backend": "unknown"}, "random_state": 42}

    with pytest.raises(ValueError):
        build_analyzer(config)


def test_openai_backend_without_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = {"sentiment": {"backend": "openai", "openai_model": "gpt-4o-mini"}, "random_state": 42}

    with pytest.raises(OSError):
        build_analyzer(config)


def test_ml_sentiment_scores_are_in_unit_range():
    analyzer = MLSentiment(random_state=42)

    scores = analyzer.score(POSITIVE_TEXTS + NEGATIVE_TEXTS)

    assert len(scores) == 4
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_run_sentiment_adds_score_and_label_columns():
    config = {
        "sentiment": {
            "backend": "traditional_ml",
            "thresholds": THRESHOLDS,
            "evaluation": {"enabled": False},
        },
        "random_state": 42,
    }
    df = pd.DataFrame({"text_data": POSITIVE_TEXTS + NEGATIVE_TEXTS})

    result = run_sentiment(df, config)

    assert "sentiment_score" in result.columns
    assert "sentiment_label" in result.columns
    assert set(result["sentiment_label"]).issubset({"Alta", "Media", "Baja"})


def test_evaluate_sentiment_disabled_returns_none():
    config = {"sentiment": {"evaluation": {"enabled": False}}}
    df = pd.DataFrame({"sentiment_label": ["Alta", "Baja"]})

    assert evaluate_sentiment(df, config) is None


def test_evaluate_sentiment_writes_metrics_json(tmp_path):
    config = {
        "sentiment": {"evaluation": {"enabled": True, "human_label_column": "human_label"}},
        "reporting": {"reports_dir": str(tmp_path)},
    }
    df = pd.DataFrame(
        {
            "sentiment_label": ["Alta", "Baja", "Media"],
            "human_label": ["Alta", "Baja", "Baja"],
        }
    )

    metrics = evaluate_sentiment(df, config)

    output_path = tmp_path / "sentiment_evaluation.json"
    assert output_path.exists()
    with open(output_path, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["labels"] == metrics["labels"]
