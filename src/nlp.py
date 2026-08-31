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

_NON_ALNUM_RE       = re.compile(r"[^\w\s]")
_NON_ALNUM_ES_RE    = re.compile(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ]")  # preserva chars españoles
_REQUIRED_NLTK_RESOURCES = {
    "stopwords": "corpora/stopwords",
    "wordnet":   "corpora/wordnet",
    "omw-1.4":   "corpora/omw-1.4",
}


# ─────────────────────────────────────────────────────────────────────────────
# Lemmatizer factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_lemmatizer(language: str):
    """Return a (type, lemmatizer) tuple appropriate for *language*.

    For English: NLTK WordNetLemmatizer.
    For Spanish: spaCy ``es_core_news_sm`` if available, otherwise NLTK
    SnowballStemmer as fallback (already bundled with NLTK).
    """
    if language == "spanish":
        # Try spaCy first (best Spanish lemmatization)
        try:
            import spacy  # noqa: PLC0415
            try:
                nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
            except OSError:
                # Model not installed — download it automatically (~15 MB, once)
                import subprocess  # noqa: PLC0415
                import sys  # noqa: PLC0415
                logger.info("Descargando modelo spaCy es_core_news_sm (~15 MB)...")
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", "es_core_news_sm"],
                    check=True,
                    capture_output=True,
                )
                nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
            logger.info("Using spaCy es_core_news_sm for Spanish lemmatization")
            return ("spacy", nlp)
        except Exception:
            pass  # fall through to SnowballStemmer

        # Fallback: NLTK SnowballStemmer (no extra download needed)
        from nltk.stem import SnowballStemmer  # noqa: PLC0415
        logger.info("spaCy no disponible — usando SnowballStemmer para español.")
        return ("snowball", SnowballStemmer("spanish"))

    # Default: English WordNet
    return ("wordnet", WordNetLemmatizer())


def _lemmatize_token(token: str, lemmatizer_type: str, lemmatizer) -> str:
    """Apply the appropriate lemmatize call based on lemmatizer type."""
    if lemmatizer_type == "wordnet":
        return lemmatizer.lemmatize(token)
    if lemmatizer_type == "snowball":
        return lemmatizer.stem(token)
    # spaCy: process single token
    doc = lemmatizer(token)
    return doc[0].lemma_ if doc else token


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def process_text(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Run the NLP pipeline: normalize, tokenize, lemmatize and tag."""
    _ensure_nltk_resources()

    nlp_config        = config["nlp"]
    language          = nlp_config.get("language", "english")
    min_token_length  = nlp_config.get("min_token_length", 3)
    normalization_map = nlp_config.get("normalization_map", {})
    keywords          = [kw.lower() for kw in nlp_config.get("mental_health_keywords", [])]

    stop_words = set(stopwords.words(language)) | set(nlp_config.get("custom_stopwords", []))
    lem_type, lemmatizer = _build_lemmatizer(language)

    df = df.copy()
    df["tokens"] = df["text_data"].apply(
        lambda text: _tokenize(
            text, language, stop_words, lem_type, lemmatizer, normalization_map, min_token_length
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


def normalize_text(text: object, language: str = "english") -> str:
    """Lowercase and remove punctuation, preserving Spanish characters if needed."""
    if not isinstance(text, str):
        return ""
    lowered = text.lower()
    if language == "spanish":
        # Keep accented letters and ñ; only strip punctuation
        return _NON_ALNUM_ES_RE.sub("", lowered)
    # English: strip accents and non-ASCII
    normalized = unicodedata.normalize("NFKD", lowered).encode("ascii", "ignore").decode("utf-8")
    return _NON_ALNUM_RE.sub("", normalized)


def _tokenize(
    text: object,
    language: str,
    stop_words: set,
    lem_type: str,
    lemmatizer,
    normalization_map: dict,
    min_token_length: int,
) -> list[str]:
    tokens = []
    for token in normalize_text(text, language).split():
        if len(token) < min_token_length:
            continue
        lemma = _lemmatize_token(token, lem_type, lemmatizer)
        lemma = normalization_map.get(lemma, lemma)
        if lemma not in stop_words and len(lemma) >= min_token_length:
            tokens.append(lemma)
    return tokens


# Keep old name for backwards compatibility
def tokenize_and_lemmatize(text, stop_words, lemmatizer, normalization_map, min_token_length):
    return _tokenize(text, "english", stop_words, "wordnet", lemmatizer, normalization_map, min_token_length)


def _contains_keyword(text: object, keywords: list[str]) -> bool:
    if not isinstance(text, str) or not text:
        return False
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def token_frequencies(df: pd.DataFrame, token_column: str = "tokens") -> Counter:
    counter: Counter = Counter()
    for tokens in df[token_column]:
        counter.update(tokens)
    return counter
