"""Centralized configuration for the running gait analysis app.

All filesystem paths are derived from the project root so the app behaves
the same regardless of the current working directory.
"""

from pathlib import Path

# Resolves to the directory that contains the `app/` package
# (this file is at app/config/settings.py -> parents[2] is project root).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# =========================
# Roboflow
# =========================
ROBOFLOW_API_KEY = "6JMmkQ0eKGad3RHGfpph"
KEYPOINT_MODEL_ID = "secondtry-bt4cj/3"
CONTACT_MODEL_ID = "contact-nocontact/2"

# =========================
# Filesystem
# =========================
DATA_DIR: Path = PROJECT_ROOT / "data"
SHOES_CSV: Path = DATA_DIR / "shoes.csv"
OUTPUT_ROOT: Path = PROJECT_ROOT / "analysis_results"

# =========================
# Pipeline tuning
# =========================
FRAME_STRIDE = 5
CONTACT_CONF_THRESHOLD = 0.5
