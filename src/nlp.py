"""Text normalization, tokenization, lemmatization and keyword detection."""

import logging
import re
import unicodedata
from collections import Counter

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

_NON_ALNUM_RE = re.compile(r"[^\w\s]")
_REQUIRED_NLTK_RESOURCES = {
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}


def process_text(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Run the NLP pipeline: normalize, tokenize, lemmatize and tag.

    Adds two columns to the DataFrame:

    - ``tokens``: list of cleaned, lemmatized tokens derived from
      ``text_data`` (stopwords removed, normalization map applied, tokens
      shorter than ``config['nlp']['min_token_length']`` dropped).
    - ``mental_health_flag``: ``1`` if any of
      ``config['nlp']['mental_health_keywords']`` appears in ``text_data``
      (case-insensitive), else ``0``.

    Args:
        df: Input DataFrame with a ``text_data`` column.
        config: Parsed pipeline configuration (``params.yaml``).

    Returns:
        Copy of ``df`` with the ``tokens`` and ``mental_health_flag``
        columns added.
    """
    _ensure_nltk_resources()

    nlp_config = config["nlp"]
    language = nlp_config["language"]
    min_token_length = nlp_config.get("min_token_length", 3)
    normalization_map = nlp_config.get("normalization_map", {})
    keywords = [kw.lower() for kw in nlp_config.get("mental_health_keywords", [])]

    stop_words = set(stopwords.words(language)) | set(nlp_config.get("custom_stopwords", []))
    lemmatizer = WordNetLemmatizer()

    df = df.copy()
    df["tokens"] = df["text_data"].apply(
        lambda text: tokenize_and_lemmatize(
            text, stop_words, lemmatizer, normalization_map, min_token_length
        )
    )
    df["mental_health_flag"] = df["text_data"].apply(
        lambda text: int(_contains_keyword(text, keywords))
    )
    return df


def _ensure_nltk_resources() -> None:
    """Download required NLTK corpora if not already present."""
    for package, resource_path in _REQUIRED_NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info("Downloading NLTK resource '%s'", package)
            nltk.download(package, quiet=True)


def normalize_text(text: object) -> str:
    """Lowercase, strip accents and remove non-alphanumeric characters.

    Args:
        text: Raw text value (non-strings are treated as empty).

    Returns:
        Cleaned, lowercase ASCII text containing only word characters and
        whitespace.
    """
    if not isinstance(text, str):
        return ""

    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    return _NON_ALNUM_RE.sub("", normalized.lower())


def tokenize_and_lemmatize(
    text: object,
    stop_words: set[str],
    lemmatizer: WordNetLemmatizer,
    normalization_map: dict[str, str],
    min_token_length: int,
) -> list[str]:
    """Normalize, tokenize, lemmatize and filter a single text value.

    Args:
        text: Raw text value.
        stop_words: Set of stopwords (base language + custom) to remove.
        lemmatizer: Lemmatizer instance applied to each token.
        normalization_map: Manual lemma overrides (e.g. ``deepfakes`` ->
            ``deepfake``).
        min_token_length: Minimum token length to keep.

    Returns:
        List of cleaned, lemmatized tokens.
    """
    cleaned_tokens = []
    for token in normalize_text(text).split():
        if len(token) < min_token_length:
            continue

        lemma = lemmatizer.lemmatize(token)
        lemma = normalization_map.get(lemma, lemma)

        if lemma not in stop_words:
            cleaned_tokens.append(lemma)

    return cleaned_tokens


def _contains_keyword(text: object, keywords: list[str]) -> bool:
    """Check whether any keyword appears in ``text`` (case-insensitive).

    Args:
        text: Raw text value.
        keywords: Lowercased keywords/phrases to look for.

    Returns:
        ``True`` if any keyword is a substring of the lowercased text.
    """
    if not isinstance(text, str) or not text:
        return False

    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def token_frequencies(df: pd.DataFrame, token_column: str = "tokens") -> Counter:
    """Aggregate token frequencies across the dataset for word clouds.

    Args:
        df: DataFrame containing a column of token lists.
        token_column: Name of the column with token lists.

    Returns:
        ``Counter`` mapping each token to its total frequency.
    """
    counter: Counter = Counter()
    for tokens in df[token_column]:
        counter.update(tokens)
    return counter
