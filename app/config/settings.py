"""Centralized configuration for the running gait analysis app.

All filesystem paths are derived from the project root so the app behaves
the same regardless of the current working directory.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Resolves to the directory that contains the `app/` package
# (this file is at app/config/settings.py -> parents[2] is project root).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

# =========================
# Roboflow
# =========================
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not ROBOFLOW_API_KEY:
    raise RuntimeError(
        "ROBOFLOW_API_KEY environment variable is not set. "
        "Create a .env file or export it before running."
    )
# Both the contact/no-contact classifier and the keypoint detector run as
# Roboflow Workflows (rather than direct model calls) so each can chain its
# base model with the workflow's post-processing logic.
ROBOFLOW_WORKSPACE_NAME = "chaims-workspace"
CONTACT_WORKFLOW_ID = "contact-nocontact-vcontact-nocontact-4-resnet50-t2-logic"
# rf-detr keypoint model; outputs 6 keypoints per runner detection, in order:
# left_knee, left_ankle, right_knee, right_ankle, left_heel, right_heel.
KEYPOINT_WORKFLOW_ID = "secondtry-vsecondtry-bt4cj-6-rfdetr-keypoint-preview-t1-logic"

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
