import os
import pandas as pd

from app.reporting.pronation_report import save_demo_report


def moving_average(series, window=5):
    """Centered moving average with edge handling."""
    return series.rolling(window=window, center=True, min_periods=1).mean()


def classify_pronation(mean_value, neutral_thresh=0.015, over_thresh=0.04):
    """
    Heuristic classification based on normalized ankle-vs-knee offset.

    Interpretation:
    - more negative => ankle is more inward relative to knee
    - near zero     => neutral
    - more positive => ankle is more outward relative to knee
    """
    if mean_value <= -over_thresh:
        return "overpronation_tendency"
    if mean_value < -neutral_thresh:
        return "mild_overpronation_tendency"
    if mean_value >= over_thresh:
        return "underpronation_tendency"
    if mean_value > neutral_thresh:
        return "mild_underpronation_tendency"
    return "neutral_tendency"


def analyze_pronation(
    csv_path,
    output_csv=None,
    smoothing_window=5,
    neutral_thresh=0.015,
    over_thresh=0.04,
):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = [
        "frame_idx",
        "detected",
        "lk_x", "lk_y",
        "la_x", "la_y",
        "rk_x", "rk_y",
        "ra_x", "ra_y",
        "left_dx_norm",
        "right_dx_norm",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only frames that passed contact filtering and had valid keypoint detections
    df = df[df["detected"] == 1].copy()
    if df.empty:
        raise ValueError("No detected frames found in CSV.")

    numeric_cols = [
        "lk_x", "lk_y",
        "la_x", "la_y",
        "rk_x", "rk_y",
        "ra_x", "ra_y",
        "left_dx_norm",
        "right_dx_norm",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols).copy()
    if df.empty:
        raise ValueError("No usable frames found after cleaning.")

    # ----------------------------
    # Stance detection heuristic
    # ----------------------------
    # The lower ankle in the image (larger y) is assumed to be the stance leg.
    def detect_stance_leg(row):
        if row["la_y"] > row["ra_y"]:
            return "left"
        return "right"

    df["stance_leg"] = df.apply(detect_stance_leg, axis=1)

    # ----------------------------
    # Use only stance leg signal
    # ----------------------------
    def get_stance_dx(row):
        if row["stance_leg"] == "left":
            return row["left_dx_norm"]
        return row["right_dx_norm"]

    df["stance_dx_norm"] = df.apply(get_stance_dx, axis=1)
    df["stance_dx_smooth"] = moving_average(df["stance_dx_norm"], window=smoothing_window)

    df["left_used"] = (df["stance_leg"] == "left").astype(int)
    df["right_used"] = (df["stance_leg"] == "right").astype(int)

    # Overall stance-based stats
    stance_mean = df["stance_dx_smooth"].mean()
    stance_std = df["stance_dx_smooth"].std()
    stance_min = df["stance_dx_smooth"].min()
    stance_max = df["stance_dx_smooth"].max()

    stance_label = classify_pronation(
        stance_mean,
        neutral_thresh=neutral_thresh,
        over_thresh=over_thresh,
    )

    left_count = int((df["stance_leg"] == "left").sum())
    right_count = int((df["stance_leg"] == "right").sum())

    # Left stance only
    left_df = df[df["stance_leg"] == "left"].copy()
    if not left_df.empty:
        left_mean = left_df["stance_dx_smooth"].mean()
        left_label = classify_pronation(
            left_mean,
            neutral_thresh=neutral_thresh,
            over_thresh=over_thresh,
        )
    else:
        left_mean = None
        left_label = "no_left_stance_frames"

    # Right stance only
    right_df = df[df["stance_leg"] == "right"].copy()
    if not right_df.empty:
        right_mean = right_df["stance_dx_smooth"].mean()
        right_label = classify_pronation(
            right_mean,
            neutral_thresh=neutral_thresh,
            over_thresh=over_thresh,
        )
    else:
        right_mean = None
        right_label = "no_right_stance_frames"

    # Save per-frame analysis
    if output_csv is not None:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved per-frame analysis to: {output_csv}")

    analysis_result = {
        "frames_analyzed": len(df),
        "left_stance_frames": left_count,
        "right_stance_frames": right_count,
        "overall": {
            "mean": float(stance_mean),
            "std": float(stance_std) if pd.notna(stance_std) else 0.0,
            "min": float(stance_min),
            "max": float(stance_max),
            "classification": stance_label,
        },
        "left_stance_only": {
            "mean": None if left_mean is None else float(left_mean),
            "classification": left_label,
        },
        "right_stance_only": {
            "mean": None if right_mean is None else float(right_mean),
            "classification": right_label,
        },
    }

    # Save human-readable demo report
    if output_csv is not None:
        report_base = output_csv.replace(".csv", "_demo_report")
    else:
        report_base = os.path.join(
            os.path.dirname(csv_path),
            "pronation_demo_report"
        )

    save_demo_report(analysis_result, report_base)

    return analysis_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run pronation analysis on a Roboflow predictions CSV."
    )
    parser.add_argument("csv_path", help="Path to roboflow_predictions.csv")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional output path for per-frame analysis CSV.",
    )
    args = parser.parse_args()

    analyze_pronation(
        csv_path=args.csv_path,
        output_csv=args.output_csv,
        smoothing_window=5,
        neutral_thresh=0.015,
        over_thresh=0.04,
    )