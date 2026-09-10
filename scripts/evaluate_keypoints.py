"""Standalone keypoint-accuracy evaluator (not part of the Streamlit app).

Takes the rendered prediction frames from a pipeline run
(``<session>/rendered_predictions/``), lets you click the ground-truth
location of each of the 6 keypoints on every frame, and reports MSE / RMSE /
mean Euclidean error of the model's predicted keypoints
(``<session>/roboflow_predictions.csv``) against what you clicked.

Usage:
    python scripts/evaluate_keypoints.py                 # latest session
    python scripts/evaluate_keypoints.py --session-dir analysis_results/myvideo
    python scripts/evaluate_keypoints.py --report-only    # re-print report only
    python scripts/evaluate_keypoints.py --redo           # re-annotate everything

Controls while annotating:
    left-click      place the next keypoint
    u               undo the last click
    space / enter   confirm the 6 points and move to the next frame
    s               skip this frame (excluded from the report)
    q               quit and save progress so far

Ground-truth clicks are saved to
``<session-dir>/keypoint_eval_annotations.json`` so you can stop partway
through and resume later, or re-run with --report-only to regenerate the
report without re-clicking.
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import OUTPUT_ROOT  # noqa: E402


KEYPOINT_NAMES = [
    "left_knee", "left_ankle", "right_knee", "right_ankle", "left_heel", "right_heel",
]
CSV_COLUMNS = [
    ("lk_x", "lk_y"), ("la_x", "la_y"), ("rk_x", "rk_y"),
    ("ra_x", "ra_y"), ("lh_x", "lh_y"), ("rh_x", "rh_y"),
]
# left_shank_px / right_shank_px, one per keypoint side, used to normalize
# pixel error by a scale reference so it's comparable across frames/videos.
SHANK_COLUMN_FOR_KEYPOINT = [
    "left_shank_px", "left_shank_px", "right_shank_px",
    "right_shank_px", "left_shank_px", "right_shank_px",
]
POINT_COLORS = [
    (0, 255, 0), (0, 200, 255), (255, 0, 0), (255, 150, 0), (255, 0, 255), (0, 255, 255),
]

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Session / data loading
# ---------------------------------------------------------------------------
def find_latest_session(output_root: Path) -> Path:
    sessions = [
        p for p in output_root.iterdir()
        if p.is_dir() and (p / "rendered_predictions").is_dir()
    ]
    if not sessions:
        raise FileNotFoundError(
            f"No pipeline session with rendered frames found under {output_root}. "
            "Run the Streamlit app (or the pipeline) at least once first."
        )
    return max(sessions, key=lambda p: p.stat().st_mtime)


def load_predictions(session_dir: Path) -> Dict[str, dict]:
    """Map rendered-frame filename -> {'points': [...], 'shank': {...}}."""
    csv_path = session_dir / "roboflow_predictions.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Predictions CSV not found: {csv_path}")

    predictions: Dict[str, dict] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            points: List[Optional[Point]] = []
            for x_col, y_col in CSV_COLUMNS:
                x_raw, y_raw = row.get(x_col, ""), row.get(y_col, "")
                if x_raw in ("", None) or y_raw in ("", None):
                    points.append(None)
                else:
                    points.append((float(x_raw), float(y_raw)))
            shank = {}
            for side in ("left_shank_px", "right_shank_px"):
                raw = row.get(side, "")
                shank[side] = float(raw) if raw not in ("", None) else None
            predictions[row["file"]] = {"points": points, "shank": shank}
    return predictions


def list_rendered_images(session_dir: Path) -> List[Path]:
    rendered_dir = session_dir / "rendered_predictions"
    return sorted(
        p for p in rendered_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )


# ---------------------------------------------------------------------------
# Annotation persistence
# ---------------------------------------------------------------------------
def load_annotations(path: Path) -> Dict[str, List[Optional[Point]]]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    return {
        fname: [tuple(pt) if pt is not None else None for pt in pts]
        for fname, pts in raw.items()
    }


def save_annotations(path: Path, annotations: Dict[str, List[Optional[Point]]]) -> None:
    serializable = {
        fname: [list(pt) if pt is not None else None for pt in pts]
        for fname, pts in annotations.items()
    }
    path.write_text(json.dumps(serializable, indent=2))


# ---------------------------------------------------------------------------
# Interactive click annotation
# ---------------------------------------------------------------------------
def annotate_image(image_path: Path) -> Tuple[Optional[List[Point]], bool]:
    """Show one frame and collect ground-truth clicks for the 6 keypoints.

    Returns (points, quit_requested). ``points`` is None if the frame was
    skipped.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[WARN] Could not read image: {image_path}")
        return None, False

    clicks: List[Point] = []
    window = "Keypoint ground-truth annotator"
    cv2.namedWindow(window)

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(KEYPOINT_NAMES):
            clicks.append((float(x), float(y)))

    cv2.setMouseCallback(window, on_mouse)

    try:
        while True:
            display = img.copy()
            for i, pt in enumerate(clicks):
                color = POINT_COLORS[i % len(POINT_COLORS)]
                cv2.circle(display, (int(pt[0]), int(pt[1])), 5, color, -1)
                cv2.putText(
                    display, KEYPOINT_NAMES[i], (int(pt[0]) + 8, int(pt[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                )

            if len(clicks) < len(KEYPOINT_NAMES):
                prompt = f"Click: {KEYPOINT_NAMES[len(clicks)]} ({len(clicks) + 1}/{len(KEYPOINT_NAMES)})"
            else:
                prompt = "All 6 placed - press SPACE/ENTER to confirm, u to undo"
            cv2.putText(display, prompt, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                display, "[u]ndo  [s]kip frame  [q]uit & save",
                (10, display.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                return None, True
            if key == ord("s"):
                return None, False
            if key == ord("u") and clicks:
                clicks.pop()
            if len(clicks) == len(KEYPOINT_NAMES) and key in (13, 32):
                return clicks, False
    finally:
        cv2.destroyWindow(window)


def run_annotation_session(
    images: List[Path],
    annotations: Dict[str, List[Optional[Point]]],
    annotations_path: Path,
    redo: bool,
) -> None:
    todo = [img for img in images if redo or img.name not in annotations]
    if not todo:
        print("Nothing left to annotate (use --redo to re-annotate everything).")
        return

    print(f"Annotating {len(todo)} frame(s). Controls: click 6 points in order, "
          f"SPACE/ENTER confirm, u undo, s skip, q quit & save.\n")

    for i, image_path in enumerate(todo, start=1):
        print(f"[{i}/{len(todo)}] {image_path.name}")
        points, quit_now = annotate_image(image_path)
        if points is not None:
            annotations[image_path.name] = points
            save_annotations(annotations_path, annotations)
        if quit_now:
            print("\nQuit requested - progress saved.")
            break

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def compute_report(
    annotations: Dict[str, List[Optional[Point]]],
    predictions: Dict[str, dict],
) -> dict:
    per_kp = {name: {"sq_errors": [], "eucl_errors": [], "norm_errors": []} for name in KEYPOINT_NAMES}

    for fname, gt_points in annotations.items():
        pred_entry = predictions.get(fname)
        if pred_entry is None:
            continue
        pred_points = pred_entry["points"]
        shank = pred_entry["shank"]

        for i, name in enumerate(KEYPOINT_NAMES):
            gt = gt_points[i] if i < len(gt_points) else None
            pred = pred_points[i] if i < len(pred_points) else None
            if gt is None or pred is None:
                continue

            dx = pred[0] - gt[0]
            dy = pred[1] - gt[1]
            sq_err = dx * dx + dy * dy
            eucl_err = sq_err ** 0.5
            per_kp[name]["sq_errors"].append(sq_err)
            per_kp[name]["eucl_errors"].append(eucl_err)

            shank_px = shank.get(SHANK_COLUMN_FOR_KEYPOINT[i])
            if shank_px:
                per_kp[name]["norm_errors"].append(eucl_err / shank_px)

    keypoint_stats = {}
    all_sq_errors: List[float] = []
    all_eucl_errors: List[float] = []

    for name, data in per_kp.items():
        sq_errors = data["sq_errors"]
        eucl_errors = data["eucl_errors"]
        norm_errors = data["norm_errors"]
        all_sq_errors.extend(sq_errors)
        all_eucl_errors.extend(eucl_errors)

        if not sq_errors:
            keypoint_stats[name] = {"n": 0}
            continue

        keypoint_stats[name] = {
            "n": len(sq_errors),
            "mse_px2": statistics.mean(sq_errors),
            "rmse_px": statistics.mean(sq_errors) ** 0.5,
            "mean_eucl_err_px": statistics.mean(eucl_errors),
            "median_eucl_err_px": statistics.median(eucl_errors),
            "max_eucl_err_px": max(eucl_errors),
            "mean_norm_err_pct": (statistics.mean(norm_errors) * 100) if norm_errors else None,
        }

    overall = {"n": len(all_sq_errors)}
    if all_sq_errors:
        overall.update({
            "mse_px2": statistics.mean(all_sq_errors),
            "rmse_px": statistics.mean(all_sq_errors) ** 0.5,
            "mean_eucl_err_px": statistics.mean(all_eucl_errors),
            "median_eucl_err_px": statistics.median(all_eucl_errors),
            "max_eucl_err_px": max(all_eucl_errors),
        })

    return {
        "n_frames_annotated": len(annotations),
        "n_frames_with_predictions": sum(1 for f in annotations if f in predictions),
        "overall": overall,
        "per_keypoint": keypoint_stats,
    }


def format_report(report: dict) -> str:
    lines = []
    lines.append("========== KEYPOINT MODEL EVALUATION ==========")
    lines.append("")
    lines.append(f"Frames annotated:            {report['n_frames_annotated']}")
    lines.append(f"Frames with model prediction: {report['n_frames_with_predictions']}")
    lines.append("")

    overall = report["overall"]
    lines.append("Overall (all keypoints, all frames)")
    if overall["n"] == 0:
        lines.append("- No comparable points yet. Annotate some frames first.")
    else:
        lines.append(f"- Samples:            {overall['n']}")
        lines.append(f"- MSE (px^2):         {overall['mse_px2']:.2f}")
        lines.append(f"- RMSE (px):          {overall['rmse_px']:.2f}")
        lines.append(f"- Mean error (px):    {overall['mean_eucl_err_px']:.2f}")
        lines.append(f"- Median error (px):  {overall['median_eucl_err_px']:.2f}")
        lines.append(f"- Max error (px):     {overall['max_eucl_err_px']:.2f}")
    lines.append("")

    lines.append("Per-keypoint breakdown")
    header = f"{'keypoint':<14}{'n':>5}{'MSE(px^2)':>12}{'RMSE(px)':>11}{'mean(px)':>11}{'max(px)':>10}{'norm(%shank)':>14}"
    lines.append(header)
    lines.append("-" * len(header))
    for name in KEYPOINT_NAMES:
        stats = report["per_keypoint"][name]
        if stats["n"] == 0:
            lines.append(f"{name:<14}{0:>5}{'-':>12}{'-':>11}{'-':>11}{'-':>10}{'-':>14}")
            continue
        norm_pct = stats["mean_norm_err_pct"]
        norm_str = f"{norm_pct:.1f}" if norm_pct is not None else "-"
        lines.append(
            f"{name:<14}{stats['n']:>5}{stats['mse_px2']:>12.2f}{stats['rmse_px']:>11.2f}"
            f"{stats['mean_eucl_err_px']:>11.2f}{stats['max_eucl_err_px']:>10.2f}{norm_str:>14}"
        )
    lines.append("")

    scored = [
        (name, s["mean_eucl_err_px"]) for name, s in report["per_keypoint"].items() if s["n"] > 0
    ]
    if scored:
        worst = max(scored, key=lambda pair: pair[1])
        best = min(scored, key=lambda pair: pair[1])
        lines.append(
            f"Best-tracked keypoint:  {best[0]} (mean error {best[1]:.2f} px)"
        )
        lines.append(
            f"Worst-tracked keypoint: {worst[0]} (mean error {worst[1]:.2f} px) - "
            "look here first if the model is underperforming."
        )
    lines.append("")
    lines.append(
        "Note: 'norm(%shank)' expresses error as a percentage of that frame's "
        "shank length, so it's comparable across frames/videos shot at "
        "different distances from the camera."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--session-dir", type=Path, default=None,
        help="Pipeline output session directory (default: most recent under analysis_results/)",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Skip the click UI; just recompute and print the report from saved annotations.",
    )
    parser.add_argument(
        "--redo", action="store_true",
        help="Re-annotate every frame, even ones already annotated.",
    )
    args = parser.parse_args()

    session_dir = args.session_dir or find_latest_session(OUTPUT_ROOT)
    session_dir = session_dir.resolve()
    print(f"Session: {session_dir}")

    predictions = load_predictions(session_dir)
    images = list_rendered_images(session_dir)
    if not images:
        print("No rendered frames found in this session.")
        return 1

    annotations_path = session_dir / "keypoint_eval_annotations.json"
    annotations = load_annotations(annotations_path)

    if not args.report_only:
        run_annotation_session(images, annotations, annotations_path, redo=args.redo)

    report = compute_report(annotations, predictions)
    text = format_report(report)
    print("\n" + text)

    report_path = session_dir / "keypoint_eval_report.txt"
    report_path.write_text(text)
    (session_dir / "keypoint_eval_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved report to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
