"""Sentiment/negativity scoring backends, classification and evaluation."""

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SentimentAnalyzer(ABC):
    """Abstract interface for sentiment-negativity scoring backends."""

    @abstractmethod
    def score(self, texts: list[str]) -> list[float]:
        """Compute a negativity score in ``[0, 1]`` for each input text.

        Args:
            texts: List of raw text values to score.

        Returns:
            List of continuous negativity scores, one per input text.
        """
        raise NotImplementedError


class TransformerSentiment(SentimentAnalyzer):
    """Sentiment backend based on a HuggingFace text-classification model."""

    def __init__(self, model_name: str, batch_size: int) -> None:
        """Load the HuggingFace text-classification pipeline.

        Args:
            model_name: Name or path of the HuggingFace model
                (e.g. ``cardiffnlp/twitter-roberta-base-sentiment-latest``).
            batch_size: Batch size used during inference.
        """
        from transformers import pipeline

        self._batch_size = batch_size
        self._pipeline = pipeline("text-classification", model=model_name, top_k=None)

    def score(self, texts: list[str]) -> list[float]:
        """Return the negative-class probability for each text.

        Args:
            texts: List of raw text values to score.

        Returns:
            List of negativity scores (probability mass assigned to the
            "negative" label by the model).
        """
        cleaned = [text if isinstance(text, str) and text.strip() else "" for text in texts]
        outputs = self._pipeline(cleaned, batch_size=self._batch_size, truncation=True)

        scores = []
        for output in outputs:
            negative_score = next(
                (item["score"] for item in output if "neg" in item["label"].lower()),
                0.0,
            )
            scores.append(float(negative_score))
        return scores


class OpenAISentiment(SentimentAnalyzer):
    """Sentiment backend that queries the OpenAI Chat Completions API."""

    def __init__(self, model_name: str) -> None:
        """Initialize the OpenAI client.

        Args:
            model_name: OpenAI model identifier (e.g. ``gpt-4o-mini``).

        Raises:
            EnvironmentError: If ``OPENAI_API_KEY`` is not set.
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise OSError(
                "OPENAI_API_KEY environment variable is required for the 'openai' sentiment backend"
            )

        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name

    def score(self, texts: list[str]) -> list[float]:
        """Return a negativity score per text using the OpenAI API.

        Args:
            texts: List of raw text values to score.

        Returns:
            List of negativity scores in ``[0, 1]``.
        """
        return [self._score_one(text) for text in texts]

    def _score_one(self, text: object) -> float:
        """Score a single text via a JSON-mode chat completion.

        Args:
            text: Raw text value.

        Returns:
            Negativity score in ``[0, 1]``; ``0.0`` for empty input.
        """
        if not isinstance(text, str) or not text.strip():
            return 0.0

        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rate the negativity of the following news text on a "
                        "continuous scale from 0 (not negative) to 1 (extremely "
                        'negative). Respond as JSON: {"negativity_score": <float>}.'
                    ),
                },
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        payload = json.loads(response.choices[0].message.content)
        return float(payload["negativity_score"])


class MLSentiment(SentimentAnalyzer):
    """Classic TF-IDF + SVM sentiment backend.

    Bootstraps training labels with NLTK's VADER lexicon (negative vs.
    non-negative) and fits a linear SVM on TF-IDF features, since no
    human-labeled training set is provided by default.
    """

    def __init__(self, random_state: int) -> None:
        """Prepare the VADER lexicon used to bootstrap pseudo-labels.

        Args:
            random_state: Seed for the underlying SVM classifier.
        """
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer

        try:
            nltk.data.find("sentiment/vader_lexicon")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)

        self._vader = SentimentIntensityAnalyzer()
        self._random_state = random_state

    def score(self, texts: list[str]) -> list[float]:
        """Fit a TF-IDF + SVM model on VADER pseudo-labels and score texts.

        Args:
            texts: List of raw text values to score.

        Returns:
            List of negativity scores: predicted probability of the
            "negative" pseudo-class.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import SVC

        cleaned = [text if isinstance(text, str) else "" for text in texts]
        pseudo_labels = np.array(
            [1 if self._vader.polarity_scores(t)["compound"] <= -0.05 else 0 for t in cleaned]
        )

        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
        features = vectorizer.fit_transform(cleaned)

        if len(set(pseudo_labels)) < 2:
            logger.warning(
                "Traditional ML sentiment: only one pseudo-label class present; "
                "returning constant scores"
            )
            return [float(label) for label in pseudo_labels]

        classifier = SVC(kernel="linear", probability=True, random_state=self._random_state)
        classifier.fit(features, pseudo_labels)

        probabilities = classifier.predict_proba(features)
        negative_index = list(classifier.classes_).index(1)
        return [float(row[negative_index]) for row in probabilities]


def build_analyzer(config: dict) -> SentimentAnalyzer:
    """Instantiate the sentiment backend selected in the configuration.

    Args:
        config: Parsed pipeline configuration (``params.yaml``).

    Returns:
        A :class:`SentimentAnalyzer` implementation matching
        ``config['sentiment']['backend']``.

    Raises:
        ValueError: If the configured backend is not recognized.
    """
    sentiment_config = config["sentiment"]
    backend = sentiment_config["backend"]

    if backend == "transformer":
        return TransformerSentiment(
            model_name=sentiment_config["model_name"],
            batch_size=sentiment_config.get("batch_size", 32),
        )
    if backend == "openai":
        return OpenAISentiment(model_name=sentiment_config["openai_model"])
    if backend == "traditional_ml":
        return MLSentiment(random_state=config.get("random_state", 42))

    raise ValueError(f"Unknown sentiment backend: '{backend}'")


def classify_sentiment(score: float, thresholds: dict) -> str:
    """Map a continuous negativity score to a categorical label.

    Args:
        score: Negativity score in ``[0, 1]``.
        thresholds: Dict with ``high`` and ``medium`` thresholds.

    Returns:
        ``"Alta"`` if ``score >= thresholds['high']``, ``"Media"`` if
        ``score >= thresholds['medium']``, otherwise ``"Baja"``.
    """
    if score >= thresholds["high"]:
        return "Alta"
    if score >= thresholds["medium"]:
        return "Media"
    return "Baja"


def evaluate_sentiment(df: pd.DataFrame, config: dict) -> dict | None:
    """Evaluate predicted sentiment labels against human annotations.

    Only runs if ``config['sentiment']['evaluation']['enabled']`` is
    ``True`` and the configured human-label column is present in ``df``.
    Writes the resulting metrics to
    ``<reporting.reports_dir>/sentiment_evaluation.json``.

    Args:
        df: DataFrame containing ``sentiment_label`` and, if available, the
            human-label column.
        config: Parsed pipeline configuration (``params.yaml``).

    Returns:
        Dict with ``labels``, ``confusion_matrix`` and
        ``classification_report``, or ``None`` if evaluation did not run.
    """
    evaluation_config = config["sentiment"]["evaluation"]
    if not evaluation_config.get("enabled", False):
        return None

    human_label_column = evaluation_config["human_label_column"]
    if human_label_column not in df.columns:
        logger.warning(
            "Sentiment evaluation enabled but column '%s' not found; skipping evaluation",
            human_label_column,
        )
        return None

    from sklearn.metrics import classification_report, confusion_matrix

    y_true = df[human_label_column]
    y_pred = df["sentiment_label"]
    labels = sorted(set(y_true.dropna()) | set(y_pred.dropna()))

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    metrics = {
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }

    reports_dir = Path(config["reporting"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "sentiment_evaluation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Sentiment evaluation metrics written to %s", output_path)

    return metrics


def run_sentiment(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Score sentiment, classify it, and optionally evaluate against labels.

    Args:
        df: Input DataFrame with a ``text_data`` column.
        config: Parsed pipeline configuration (``params.yaml``).

    Returns:
        Copy of ``df`` with ``sentiment_score`` (continuous negativity) and
        ``sentiment_label`` (``"Alta"``/``"Media"``/``"Baja"``) columns added.
    """
    df = df.copy()

    analyzer = build_analyzer(config)
    df["sentiment_score"] = analyzer.score(df["text_data"].tolist())

    thresholds = config["sentiment"]["thresholds"]
    df["sentiment_label"] = df["sentiment_score"].apply(lambda s: classify_sentiment(s, thresholds))

    evaluate_sentiment(df, config)

    return df
