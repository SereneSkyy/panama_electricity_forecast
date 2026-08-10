from pathlib import Path

# Anchor all paths to the project root (parent of this src/ folder),
# not to whatever directory the script happened to be launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
MODELS_DIR = PROJECT_ROOT / "models"
