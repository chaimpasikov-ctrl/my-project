"""Centralized configuration for the running gait analysis app.

All filesystem paths are derived from the project root so the app behaves
the same regardless of the current working directory.
"""

import os
from pathlib import Path

# Resolves to the directory that contains the `app/` package
# (this file is at app/config/settings.py -> parents[2] is project root).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# =========================
# Roboflow
# =========================
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not ROBOFLOW_API_KEY:
    raise RuntimeError(
        "ROBOFLOW_API_KEY environment variable is not set. "
        "Create a .env file or export it before running."
    )
KEYPOINT_MODEL_ID = "secondtry-bt4cj/4"
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
# Sample every Nth frame for contact inference. Running stance phase is
# ~150-250 ms, so at 30 fps a stride of 2 gives ~3-4 samples per stance,
# enough to capture peak medial collapse near midstance.
FRAME_STRIDE = 2
CONTACT_CONF_THRESHOLD = 0.5

# =========================
# Rearfoot eversion angle thresholds (degrees)
# =========================
# Based on biomechanics literature for rear-view eversion angle.
# Positive = pronation/eversion, negative = supination/inversion.
EVERSION_NEUTRAL_THRESH_DEG = 4.0   # < this magnitude => neutral
EVERSION_MILD_THRESH_DEG    = 10.0  # neutral .. this => mild
EVERSION_MARKED_THRESH_DEG  = 15.0  # this and beyond => marked

# =========================
# DEPRECATED: legacy frontal-plane shank-collapse thresholds
# =========================
# Kept only so older session CSVs / scripts still import cleanly. The
# pipeline no longer uses these; the rearfoot eversion angle thresholds
# above replace them.
COLLAPSE_NEUTRAL_THRESH = 0.03  # DEPRECATED
COLLAPSE_MARKED_THRESH = 0.08   # DEPRECATED

# Minimum stance events per leg before we trust a per-leg classification.
MIN_STANCE_EVENTS_PER_LEG = 4
