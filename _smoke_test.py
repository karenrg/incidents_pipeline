import copy
import sys

import yaml

sys.path.insert(0, ".")

from src import configure_logging, set_global_seeds
from src.ingestion import load_and_validate
from src.preprocessing import preprocess
from src.nlp import process_text
from src.sentiment import run_sentiment
from src.analysis import run_analysis
from src.visualization import run_visualization

with open("config/params.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Smoke-test override: use the fast traditional_ml backend to exercise the
# full pipeline without downloading a transformer model.
config = copy.deepcopy(config)
config["sentiment"]["backend"] = "traditional_ml"

configure_logging()
set_global_seeds(config["random_state"])

df = load_and_validate(config)
print("After ingestion:", df.shape)

df = preprocess(df, config)
print("After preprocessing:", df.shape)

df = process_text(df, config)
print("After NLP:", df.shape)

df = run_sentiment(df, config)
print("After sentiment:", df.shape)
print(df[["sentiment_score", "sentiment_label"]].describe(include="all"))

df.to_parquet(config["data"]["processed_path"])
print("Saved processed dataset to", config["data"]["processed_path"])

metrics = run_analysis(df, config)
print("Metrics keys:", list(metrics.keys()))

report_path = run_visualization(df, metrics, config)
print("Report written to", report_path)

print("Pipeline completado.")
