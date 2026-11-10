from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "cicids2017-cleaned-and-preprocessed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DATASET_SLUG = "ericanacletoribeiro/cicids2017-cleaned-and-preprocessed"
EXPECTED_CSV_NAME = "cicids2017_cleaned.csv"
CHUNK_SIZE = 100_000
PCA_COMPONENTS = 14
KAISER_PLOT_COMPONENTS = 20
VARIANCE_THRESHOLD = 80.0
RANDOM_SEED = 42
