"""Rearfoot eversion angle analysis from rear-view treadmill video.

The metric is the **signed rearfoot eversion angle** (degrees) between the
shank vector (knee -> ankle) and the calcaneus vector (ankle -> heel),
computed per frame in ``app.pipelines.main_pipeline`` and aggregated here
per stance event and per leg.

Sign convention (set in the pipeline):
    +ve => eversion / pronation tendency
    -ve => inversion / supination tendency
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.config.settings import (
    EVERSION_MARKED_THRESH_DEG,
    EVERSION_MILD_THRESH_DEG,
    EVERSION_NEUTRAL_THRESH_DEG,
    MIN_STANCE_EVENTS_PER_LEG,
)
from app.reporting.pronation_report import save_demo_report


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
LABEL_NEUTRAL = "neutral_alignment"
LABEL_MILD_PRONATION = "mild_pronation"
LABEL_MODERATE_PRONATION = "moderate_pronation"
LABEL_MARKED_PRONATION = "marked_pronation"
LABEL_MILD_SUPINATION = "mild_supination"
LABEL_MODERATE_SUPINATION = "moderate_supination"
LABEL_MARKED_SUPINATION = "marked_supination"
LABEL_INSUFFICIENT = "insufficient_data"

ALL_LABELS = [
    LABEL_NEUTRAL,
    LABEL_MILD_PRONATION,
    LABEL_MODERATE_PRONATION,
    LABEL_MARKED_PRONATION,
    LABEL_MILD_SUPINATION,
    LABEL_MODERATE_SUPINATION,
    LABEL_MARKED_SUPINATION,
    LABEL_INSUFFICIENT,
]


def classify_eversion(
    angle_deg: Optional[float],
    neutral_thresh: float = EVERSION_NEUTRAL_THRESH_DEG,
    mild_thresh: float = EVERSION_MILD_THRESH_DEG,
    marked_thresh: float = EVERSION_MARKED_THRESH_DEG,
) -> str:
    """Map a signed rearfoot eversion angle (degrees) to a label.

    Magnitude bands:
        |angle| < neutral_thresh        => neutral_alignment
        neutral_thresh .. mild_thresh   => mild_*
        mild_thresh    .. marked_thresh => moderate_*
        >= marked_thresh                => marked_*
    """
    if angle_deg is None or not np.isfinite(angle_deg):
        return LABEL_INSUFFICIENT
    mag = abs(float(angle_deg))
    sign = 1 if angle_deg >= 0 else -1
    if mag < neutral_thresh:
        return LABEL_NEUTRAL
    if mag < mild_thresh:
        return LABEL_MILD_PRONATION if sign > 0 else LABEL_MILD_SUPINATION
    if mag < marked_thresh:
        return LABEL_MODERATE_PRONATION if sign > 0 else LABEL_MODERATE_SUPINATION
    return LABEL_MARKED_PRONATION if sign > 0 else LABEL_MARKED_SUPINATION


# ---------------------------------------------------------------------------
# Stance segmentation and signed eversion
# ---------------------------------------------------------------------------
def _detect_stance_leg(la_y: float, ra_y: float) -> str:
    # MVP heuristic; will be replaced by a per-leg contact classifier later.
    return "left" if la_y > ra_y else "right"


def _signed_eversion(row: pd.Series) -> float:
    """Pick the stance leg's eversion angle (already signed per leg by the pipeline)."""
    if row["stance_leg"] == "left":
        return float(row["left_eversion_deg"])
    return float(row["right_eversion_deg"])


def _segment_stance_events(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Group consecutive same-leg stance frames into discrete events."""
    if df.empty:
        return []

    df = df.sort_values("frame_idx").reset_index(drop=True)
    median_step = df["frame_idx"].diff().median()
    if not np.isfinite(median_step) or median_step <= 0:
        median_step = 1.0
    # A "new event" starts when the leg changes OR there's a large temporal gap
    # (e.g. the runner missed a contact-classification on a couple of frames).
    max_gap = max(3.0, median_step * 6.0)

    events: List[Dict[str, Any]] = []
    cur_leg: Optional[str] = None
    cur_rows: List[pd.Series] = []
    last_idx: Optional[float] = None

    for _, row in df.iterrows():
        leg = row["stance_leg"]
        gap = (row["frame_idx"] - last_idx) if last_idx is not None else 0.0
        if cur_rows and (leg != cur_leg or gap > max_gap):
            events.append(_event_from_rows(cur_leg, cur_rows))
            cur_rows = []
        cur_leg = leg
        cur_rows.append(row)
        last_idx = row["frame_idx"]

    if cur_rows:
        events.append(_event_from_rows(cur_leg, cur_rows))
    return events


def _event_from_rows(leg: str, rows: List[pd.Series]) -> Dict[str, Any]:
    vals = [float(r["eversion_deg"]) for r in rows]
    idxs = [int(r["frame_idx"]) for r in rows]
    n = len(vals)
    midstance = vals[n // 2] if n else float("nan")
    peak = max(vals, key=abs) if vals else float("nan")
    return {
        "leg": leg,
        "first_idx": idxs[0],
        "last_idx": idxs[-1],
        "n_frames": n,
        "midstance": midstance,
        "peak": peak,
    }


# ---------------------------------------------------------------------------
# Free metrics: cadence, ground-contact time, step width
# ---------------------------------------------------------------------------
def _free_metrics(
    df: pd.DataFrame,
    events: List[Dict[str, Any]],
    shank_scale_px: float,
    fps: float,
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "cadence_spm": None,
        "left_gct_ms": None,
        "right_gct_ms": None,
        "gct_asymmetry_pct": None,
        "step_width_norm": None,
    }
    if not events or not fps or fps <= 0:
        return out

    duration_s = (events[-1]["last_idx"] - events[0]["first_idx"]) / fps
    if duration_s > 0:
        out["cadence_spm"] = float(len(events) / duration_s * 60.0)

    def _gct(leg: str) -> Optional[float]:
        legs = [e for e in events if e["leg"] == leg and e["n_frames"] >= 2]
        if not legs:
            return None
        spans_ms = [(e["last_idx"] - e["first_idx"]) / fps * 1000.0 for e in legs]
        return float(np.median(spans_ms))

    out["left_gct_ms"] = _gct("left")
    out["right_gct_ms"] = _gct("right")
    if out["left_gct_ms"] and out["right_gct_ms"]:
        lg, rg = out["left_gct_ms"], out["right_gct_ms"]
        out["gct_asymmetry_pct"] = float(abs(lg - rg) / ((lg + rg) / 2.0) * 100.0)

    if shank_scale_px and shank_scale_px > 0:
        widths_px = (df["la_x"] - df["ra_x"]).abs()
        out["step_width_norm"] = float(widths_px.median() / shank_scale_px)

    return out


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------
# Beyond this magnitude (degrees) the rearfoot eversion angle is almost
# certainly a keypoint detection error rather than a real biomechanical
# signal.
EVERSION_OUTLIER_THRESH_DEG = 30.0


def _quality_check(
    df: pd.DataFrame,
    events: List[Dict[str, Any]],
    min_events_per_leg: int = MIN_STANCE_EVENTS_PER_LEG,
) -> List[str]:
    issues: List[str] = []
    n_left = sum(1 for e in events if e["leg"] == "left")
    n_right = sum(1 for e in events if e["leg"] == "right")

    if n_left < min_events_per_leg:
        issues.append("low_left_event_count")
    if n_right < min_events_per_leg:
        issues.append("low_right_event_count")
    if df["eversion_deg"].std() < 1e-3:
        issues.append("signal_flat_check_keypoints")
    if df["eversion_deg"].abs().max() > EVERSION_OUTLIER_THRESH_DEG:
        issues.append("extreme_outliers_present")

    if n_left and n_right:
        left_mid = float(np.median([e["midstance"] for e in events if e["leg"] == "left"]))
        right_mid = float(np.median([e["midstance"] for e in events if e["leg"] == "right"]))
        # Use the marked-pronation threshold (deg) as the asymmetry alarm.
        if abs(left_mid - right_mid) > EVERSION_MARKED_THRESH_DEG:
            issues.append("large_left_right_asymmetry")

    return issues


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
REQUIRED_COLS = [
    "frame_idx",
    "detected",
    "lk_x", "lk_y",
    "la_x", "la_y",
    "rk_x", "rk_y",
    "ra_x", "ra_y",
    "lh_x", "lh_y",
    "rh_x", "rh_y",
    "left_dx_px",
    "right_dx_px",
    "left_shank_px",
    "right_shank_px",
    "left_eversion_deg",
    "right_eversion_deg",
]


def _read_session_meta(csv_path: str) -> Dict[str, Any]:
    meta_path = Path(csv_path).parent / "session_meta.json"
    if not meta_path.is_file():
        return {}
    try:
        with meta_path.open() as f:
            return json.load(f)
    except Exception:
        return {}


def analyze_pronation(
    csv_path: str,
    output_csv: Optional[str] = None,
    neutral_thresh: float = EVERSION_NEUTRAL_THRESH_DEG,
    mild_thresh: float = EVERSION_MILD_THRESH_DEG,
    marked_thresh: float = EVERSION_MARKED_THRESH_DEG,
) -> Dict[str, Any]:
    """Analyze a predictions CSV and return a structured result dict.

    The metric is the per-stance signed **rearfoot eversion angle** in degrees
    (positive = pronation tendency, negative = supination tendency).
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        new_cols = {"lh_x", "lh_y", "rh_x", "rh_y", "left_eversion_deg", "right_eversion_deg"}
        if any(c in new_cols for c in missing):
            raise ValueError(
                "This session was generated with an older pipeline "
                "(pre-heel-keypoints). Re-run the pipeline on the source video. "
                f"Missing columns: {missing}"
            )
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["detected"] == 1].copy()
    if df.empty:
        raise ValueError("No detected frames found in CSV.")

    numeric_cols = [c for c in REQUIRED_COLS if c not in ("frame_idx", "detected")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols).copy()
    if df.empty:
        raise ValueError("No usable frames found after cleaning.")

    # ----- Anatomical scale: clip-wide median shank length ----------------
    # Kept for step-width normalization and as a sanity-check artifact.
    shank_samples = pd.concat([df["left_shank_px"], df["right_shank_px"]])
    shank_scale = float(shank_samples.median())
    if not np.isfinite(shank_scale) or shank_scale <= 0:
        raise ValueError("Could not compute shank-length scale.")

    # ----- Stance leg + signed eversion (per-leg sign already applied) ----
    df["stance_leg"] = df.apply(
        lambda r: _detect_stance_leg(r["la_y"], r["ra_y"]),
        axis=1,
    )
    df["eversion_deg"] = df.apply(_signed_eversion, axis=1)

    # ----- Per-stance-event aggregation -----------------------------------
    events = _segment_stance_events(df)

    def per_leg(leg: str) -> Dict[str, Any]:
        ev = [e for e in events if e["leg"] == leg]
        if not ev:
            return {
                "median_deg": None,
                "iqr_deg": None,
                "n_events": 0,
                "peak_deg": None,
                "classification": LABEL_INSUFFICIENT,
            }
        mids = np.array([e["midstance"] for e in ev])
        peaks = np.array([e["peak"] for e in ev])
        med = float(np.median(mids))
        iqr = float(np.subtract(*np.percentile(mids, [75, 25])))
        return {
            "median_deg": med,
            "iqr_deg": iqr,
            "n_events": len(ev),
            "peak_deg": float(np.median(peaks)),
            "classification": classify_eversion(
                med, neutral_thresh, mild_thresh, marked_thresh
            ),
        }

    left_stats = per_leg("left")
    right_stats = per_leg("right")

    if events:
        all_mids = np.array([e["midstance"] for e in events])
        overall_median = float(np.median(all_mids))
        overall_label = classify_eversion(
            overall_median, neutral_thresh, mild_thresh, marked_thresh
        )
    else:
        overall_median = None
        overall_label = LABEL_INSUFFICIENT

    overall = {
        "median_deg": overall_median,
        "std_deg": float(df["eversion_deg"].std()) if len(df) > 1 else 0.0,
        "min_deg": float(df["eversion_deg"].min()),
        "max_deg": float(df["eversion_deg"].max()),
        "classification": overall_label,
    }

    # ----- Free metrics + quality control ---------------------------------
    meta = _read_session_meta(csv_path)
    fps = float(meta.get("fps", 30.0))
    free = _free_metrics(df, events, shank_scale, fps)
    issues = _quality_check(df, events)

    if output_csv is not None:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved per-frame analysis to: {output_csv}")

    result: Dict[str, Any] = {
        "frames_analyzed": int(len(df)),
        "left_stance_frames": int((df["stance_leg"] == "left").sum()),
        "right_stance_frames": int((df["stance_leg"] == "right").sum()),
        "shank_scale_px": shank_scale,
        "fps": fps,
        "overall": overall,
        "left_stance_only": left_stats,
        "right_stance_only": right_stats,
        "free_metrics": free,
        "quality_issues": issues,
        "metric_definition": (
            "rearfoot eversion angle (degrees), signed positive for "
            "pronation tendency"
        ),
    }

    if output_csv is not None:
        report_base = output_csv.replace(".csv", "_demo_report")
    else:
        report_base = os.path.join(
            os.path.dirname(csv_path),
            "pronation_demo_report",
        )

    save_demo_report(result, report_base)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run rearfoot eversion angle analysis on a predictions CSV."
    )
    parser.add_argument("csv_path", help="Path to roboflow_predictions.csv")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional output path for per-frame analysis CSV.",
    )
    args = parser.parse_args()

    analyze_pronation(csv_path=args.csv_path, output_csv=args.output_csv)
