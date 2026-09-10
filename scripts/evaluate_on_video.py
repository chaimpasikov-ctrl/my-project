"""Standalone batch evaluator: multiple videos in → mean pixel error report out.

Runs the analysis pipeline on every video in a folder (or a single video),
opens a click-based annotator for ground-truth keypoints, and prints a
report of mean pixel error per keypoint, per video, and overall.

Also tracks contact classifier false positives: press 'n' to mark a
frame as "not a real contact frame" — those frames are excluded from
the keypoint accuracy report and counted as classifier errors instead.

Usage:
    # Process every video in scripts/test_videos/
    python scripts/evaluate_on_video.py

    # Process a specific folder
    python scripts/evaluate_on_video.py --videos-dir path/to/videos

    # Single video (backward compatible)
    python scripts/evaluate_on_video.py --video path/to/one.mp4

    # Report-only (recompute from saved clicks, no annotator)
    python scripts/evaluate_on_video.py --report-only

    # Re-annotate everything
    python scripts/evaluate_on_video.py --redo

    # Cap frames per video (default: 30)
    python scripts/evaluate_on_video.py --max-frames-per-video 20

Controls while annotating:
    left-click      place the next keypoint
    u               undo the last click
    space / enter   confirm the 6 points and move to the next frame
    n               mark frame as "not a real contact" (classifier error)
    s               skip this frame (excluded from any report)
    q               quit and save progress so far (moves to next video)
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup — allow importing from repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

# Reuse the existing evaluator's constants and IO helpers
from scripts.evaluate_keypoints import (  # noqa: E402
    KEYPOINT_NAMES,
    POINT_COLORS,
    SHANK_COLUMN_FOR_KEYPOINT,
    load_annotations,
    save_annotations,
    load_predictions,
    list_rendered_images,
)

from app.config.settings import OUTPUT_ROOT  # noqa: E402


# Video extensions we'll recognize in a folder
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

# Default videos folder — sits next to this script
DEFAULT_VIDEOS_DIR = Path(__file__).resolve().parent / "test_videos"


Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Pipeline invocation
# ---------------------------------------------------------------------------
def run_pipeline_on_video(video_path: Path) -> Path:
    """Run the analysis pipeline on the given video, return the session dir.

    Adjust the import + call below if your pipeline's entry function has a
    different name or signature.
    """
    try:
        from app.pipelines.main_pipeline import process_video
        session_dir = process_video(str(video_path))
    except ImportError:
        try:
            from app.pipelines.main_pipeline import run_pipeline
            session_dir = run_pipeline(str(video_path))
        except ImportError as e:
            raise RuntimeError(
                "Could not import a pipeline entry point from "
                "app.pipelines.main_pipeline. Edit run_pipeline_on_video() "
                "in this script to call your actual function."
            ) from e

    # process_video may return a Path, a str, a dict of results, or None.
    # In all "no clear path" cases, fall back to the newest session dir on
    # disk — we know the pipeline just wrote one because it just ran.
    if not isinstance(session_dir, (str, Path)):
        candidates = [
            p for p in OUTPUT_ROOT.iterdir()
            if p.is_dir() and (p / "rendered_predictions").is_dir()
        ]
        if not candidates:
            raise RuntimeError(
                "Pipeline finished but no session directory was found under "
                f"{OUTPUT_ROOT}. Cannot continue."
            )
        session_dir = max(candidates, key=lambda p: p.stat().st_mtime)

    return Path(session_dir)


# ---------------------------------------------------------------------------
# Click UI — this replaces the imported annotate_image so we can add 'n' key
# ---------------------------------------------------------------------------
def annotate_image_with_reject(
    image_path: Path,
) -> Tuple[Optional[List[Point]], bool, bool]:
    """Show one frame; collect ground-truth clicks or reject as non-contact.

    Returns (points, quit_requested, rejected_as_non_contact):
        - points is a list of 6 (x, y) tuples if the frame was annotated
        - points is None if the frame was skipped OR rejected OR quit
        - rejected_as_non_contact is True iff the user pressed 'n'
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[WARN] Could not read image: {image_path}")
        return None, False, False

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
                    display, KEYPOINT_NAMES[i],
                    (int(pt[0]) + 8, int(pt[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                )

            if len(clicks) < len(KEYPOINT_NAMES):
                prompt = (
                    f"Click: {KEYPOINT_NAMES[len(clicks)]} "
                    f"({len(clicks) + 1}/{len(KEYPOINT_NAMES)})"
                )
            else:
                prompt = "All 6 placed - press SPACE/ENTER to confirm, u to undo"
            cv2.putText(display, prompt, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(
                display,
                "[u]ndo  [n]ot a real contact  [s]kip  [q]uit & save",
                (10, display.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA,
            )

            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                return None, True, False
            if key == ord("s"):
                return None, False, False
            if key == ord("n"):
                return None, False, True
            if key == ord("u") and clicks:
                clicks.pop()
            if len(clicks) == len(KEYPOINT_NAMES) and key in (13, 32):
                return clicks, False, False
    finally:
        cv2.destroyWindow(window)


def run_annotation_session_with_reject(
    images: List[Path],
    annotations: Dict[str, List[Optional[Point]]],
    rejections: Dict[str, bool],
    annotations_path: Path,
    rejections_path: Path,
    redo: bool,
) -> None:
    """Annotate a batch of frames; support rejecting as non-contact."""
    def already_seen(name: str) -> bool:
        return name in annotations or name in rejections

    todo = [img for img in images if redo or not already_seen(img.name)]
    if not todo:
        print(
            "Nothing left to annotate (use --redo to re-annotate everything)."
        )
        return

    print(
        f"Annotating {len(todo)} frame(s). "
        f"Click 6 points, SPACE/ENTER confirm, "
        f"u undo, n not-a-contact, s skip, q quit & save.\n"
    )

    for i, image_path in enumerate(todo, start=1):
        print(f"[{i}/{len(todo)}] {image_path.name}")
        points, quit_now, rejected = annotate_image_with_reject(image_path)

        if rejected:
            rejections[image_path.name] = True
            # If we previously accepted this frame, drop the annotation.
            annotations.pop(image_path.name, None)
            save_annotations(annotations_path, annotations)
            _save_rejections(rejections_path, rejections)
            print("  -> flagged as not-a-real-contact (classifier false positive)")
        elif points is not None:
            annotations[image_path.name] = points
            # If we previously rejected, drop the rejection.
            rejections.pop(image_path.name, None)
            save_annotations(annotations_path, annotations)
            _save_rejections(rejections_path, rejections)

        if quit_now:
            print("\nQuit requested - progress saved.")
            break

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Rejections persistence — a simple JSON file next to annotations
# ---------------------------------------------------------------------------
def _load_rejections(path: Path) -> Dict[str, bool]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    return {k: bool(v) for k, v in raw.items() if v}


def _save_rejections(path: Path, rejections: Dict[str, bool]) -> None:
    path.write_text(json.dumps(rejections, indent=2))


# ---------------------------------------------------------------------------
# Video discovery
# ---------------------------------------------------------------------------
def collect_videos(
    videos_dir: Optional[Path], single_video: Optional[Path]
) -> List[Path]:
    if single_video is not None:
        if not single_video.is_file():
            raise FileNotFoundError(f"Video not found: {single_video}")
        return [single_video]

    d = videos_dir if videos_dir is not None else DEFAULT_VIDEOS_DIR
    if not d.is_dir():
        raise FileNotFoundError(
            f"Videos folder not found: {d}\n"
            "Create it and drop your test videos in, or use --video / --videos-dir."
        )

    videos = sorted(
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    if not videos:
        raise FileNotFoundError(
            f"No videos found in {d}. "
            f"Supported extensions: {sorted(VIDEO_EXTS)}"
        )
    return videos


# ---------------------------------------------------------------------------
# Training-set check
# ---------------------------------------------------------------------------
def check_training_set(video_paths: List[Path]) -> List[Path]:
    training_list = REPO_ROOT / "data" / "training_video_names.txt"
    if not training_list.is_file():
        print(
            "\nNOTE: Cannot automatically verify these videos weren't in "
            "training. Confirm you're using unseen videos for the numbers "
            "to be defensible.\n"
        )
        return video_paths

    trained_names = {
        line.strip().lower()
        for line in training_list.read_text().splitlines()
        if line.strip()
    }
    suspect = [v for v in video_paths if v.name.lower() in trained_names]
    if not suspect:
        return video_paths

    print("\nWARNING: These videos appear to be in the training set:")
    for v in suspect:
        print(f"  - {v.name}")
    answer = input(
        "Skip them and continue with the rest? [Y/n] "
    ).strip().lower()
    if answer in ("", "y", "yes"):
        return [v for v in video_paths if v not in suspect]
    else:
        print(
            "Continuing with ALL videos including training-set ones. "
            "Numbers will not be defensible."
        )
        return video_paths


# ---------------------------------------------------------------------------
# Session lookup for --report-only
# ---------------------------------------------------------------------------
def find_session_for_video(video_path: Path) -> Optional[Path]:
    if not OUTPUT_ROOT.is_dir():
        return None
    stem = video_path.stem
    candidates = [
        p for p in OUTPUT_ROOT.iterdir()
        if p.is_dir()
        and p.name.startswith(stem)
        and (p / "rendered_predictions").is_dir()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Report — mean pixel error only + classifier false-positive rate
# ---------------------------------------------------------------------------
def compute_mean_report(
    annotations: Dict[str, list],
    rejections: Dict[str, bool],
    predictions: Dict[str, dict],
    total_frames_reviewed: int,
) -> dict:
    """Compute per-keypoint mean pixel error + contact classifier stats."""
    per_kp: Dict[str, dict] = {
        name: {"errors": [], "norm_errors": []} for name in KEYPOINT_NAMES
    }

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
            eucl_err = (dx * dx + dy * dy) ** 0.5
            per_kp[name]["errors"].append(eucl_err)

            shank_px = shank.get(SHANK_COLUMN_FOR_KEYPOINT[i])
            if shank_px:
                per_kp[name]["norm_errors"].append(eucl_err / shank_px)

    keypoint_stats: Dict[str, dict] = {}
    all_errors: List[float] = []
    all_norm_errors: List[float] = []

    for name, data in per_kp.items():
        errors = data["errors"]
        norm_errors = data["norm_errors"]
        all_errors.extend(errors)
        all_norm_errors.extend(norm_errors)

        if not errors:
            keypoint_stats[name] = {"n": 0}
            continue

        keypoint_stats[name] = {
            "n": len(errors),
            "mean_px": statistics.mean(errors),
            "mean_norm_pct": (
                statistics.mean(norm_errors) * 100 if norm_errors else None
            ),
        }

    overall = {"n": len(all_errors)}
    if all_errors:
        overall["mean_px"] = statistics.mean(all_errors)
        overall["mean_norm_pct"] = (
            statistics.mean(all_norm_errors) * 100 if all_norm_errors else None
        )

    # Contact classifier false-positive stats
    # (all frames handed to the annotator were labelled "contact" by the
    # classifier; the user marked some of them as not-really-contact)
    n_rejected = sum(1 for v in rejections.values() if v)
    n_accepted = len(annotations)
    contact_reviewed = n_accepted + n_rejected
    fp_rate = (n_rejected / contact_reviewed) if contact_reviewed > 0 else None

    return {
        "n_frames_reviewed": total_frames_reviewed,
        "n_frames_annotated": n_accepted,
        "n_frames_rejected_non_contact": n_rejected,
        "n_frames_with_predictions": sum(
            1 for f in annotations if f in predictions
        ),
        "contact_classifier": {
            "n_reviewed": contact_reviewed,
            "n_false_positive": n_rejected,
            "false_positive_rate": fp_rate,
        },
        "overall": overall,
        "per_keypoint": keypoint_stats,
    }


def combine_reports(per_video_reports: List[Tuple[Path, dict]]) -> dict:
    """Aggregate per-video reports into one report across all videos."""
    per_kp: Dict[str, List[Tuple[float, int]]] = {
        name: [] for name in KEYPOINT_NAMES
    }
    per_kp_norm: Dict[str, List[Tuple[float, int]]] = {
        name: [] for name in KEYPOINT_NAMES
    }

    total_frames_reviewed = 0
    total_frames_annotated = 0
    total_frames_rejected = 0
    total_frames_with_predictions = 0

    for _, r in per_video_reports:
        total_frames_reviewed += r.get("n_frames_reviewed", 0)
        total_frames_annotated += r["n_frames_annotated"]
        total_frames_rejected += r.get("n_frames_rejected_non_contact", 0)
        total_frames_with_predictions += r["n_frames_with_predictions"]

        for name, stats in r["per_keypoint"].items():
            if stats["n"] == 0:
                continue
            per_kp[name].append((stats["mean_px"], stats["n"]))
            if stats.get("mean_norm_pct") is not None:
                per_kp_norm[name].append((stats["mean_norm_pct"], stats["n"]))

    combined_kp: Dict[str, dict] = {}
    for name in KEYPOINT_NAMES:
        entries = per_kp[name]
        norm_entries = per_kp_norm[name]

        if not entries:
            combined_kp[name] = {"n": 0}
            continue

        n = sum(e[1] for e in entries)
        weighted_mean = sum(e[0] * e[1] for e in entries) / n

        norm_mean = None
        if norm_entries:
            n_norm = sum(e[1] for e in norm_entries)
            norm_mean = sum(e[0] * e[1] for e in norm_entries) / n_norm

        combined_kp[name] = {
            "n": n,
            "mean_px": weighted_mean,
            "mean_norm_pct": norm_mean,
        }

    total_px = 0.0
    total_n = 0
    total_norm = 0.0
    total_n_norm = 0
    for name in KEYPOINT_NAMES:
        s = combined_kp[name]
        if s["n"] == 0:
            continue
        total_px += s["mean_px"] * s["n"]
        total_n += s["n"]
        if s["mean_norm_pct"] is not None:
            total_norm += s["mean_norm_pct"] * s["n"]
            total_n_norm += s["n"]

    overall = {"n": total_n}
    if total_n > 0:
        overall["mean_px"] = total_px / total_n
        overall["mean_norm_pct"] = (
            total_norm / total_n_norm if total_n_norm > 0 else None
        )

    contact_reviewed = total_frames_annotated + total_frames_rejected
    fp_rate = (
        total_frames_rejected / contact_reviewed
        if contact_reviewed > 0
        else None
    )

    return {
        "n_videos": len(per_video_reports),
        "n_frames_reviewed": total_frames_reviewed,
        "n_frames_annotated": total_frames_annotated,
        "n_frames_rejected_non_contact": total_frames_rejected,
        "n_frames_with_predictions": total_frames_with_predictions,
        "contact_classifier": {
            "n_reviewed": contact_reviewed,
            "n_false_positive": total_frames_rejected,
            "false_positive_rate": fp_rate,
        },
        "overall": overall,
        "per_keypoint": combined_kp,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------
def _format_kp_block(report: dict, indent: str = "  ") -> List[str]:
    lines = []
    header = (
        f"{indent}{'keypoint':<14}{'n':>5}{'mean (px)':>12}{'% shank':>12}"
    )
    lines.append(header)
    lines.append(indent + "-" * (len(header) - len(indent)))
    for name in KEYPOINT_NAMES:
        stats = report["per_keypoint"][name]
        if stats["n"] == 0:
            lines.append(f"{indent}{name:<14}{0:>5}{'-':>12}{'-':>12}")
            continue
        norm_str = (
            f"{stats['mean_norm_pct']:.2f}"
            if stats.get("mean_norm_pct") is not None
            else "-"
        )
        lines.append(
            f"{indent}{name:<14}{stats['n']:>5}"
            f"{stats['mean_px']:>12.2f}{norm_str:>12}"
        )
    return lines


def _format_classifier_block(report: dict, indent: str = "  ") -> List[str]:
    c = report.get("contact_classifier", {})
    n = c.get("n_reviewed", 0)
    fp = c.get("n_false_positive", 0)
    rate = c.get("false_positive_rate")
    lines = []
    if n == 0:
        lines.append(f"{indent}(No classifier data — no frames reviewed)")
        return lines
    lines.append(
        f"{indent}Contact frames reviewed:  {n}"
    )
    lines.append(
        f"{indent}Marked not-real-contact:  {fp} "
        f"(false-positive rate {rate * 100:.1f}%)"
        if rate is not None else f"{indent}Marked not-real-contact:  {fp}"
    )
    return lines


def format_per_video_report(report: dict, video_path: Path) -> str:
    lines = []
    lines.append(f"---- {video_path.name} ----")
    lines.append(
        f"Frames reviewed: {report.get('n_frames_reviewed', '?')}  "
        f"annotated: {report['n_frames_annotated']}  "
        f"rejected: {report.get('n_frames_rejected_non_contact', 0)}"
    )
    lines.append("")

    lines.append("Contact classifier")
    lines.extend(_format_classifier_block(report))
    lines.append("")

    overall = report["overall"]
    if overall["n"] == 0:
        lines.append("No comparable points for keypoint accuracy.")
        return "\n".join(lines)

    lines.append("Keypoint accuracy")
    lines.append(
        f"  Mean pixel error:         {overall['mean_px']:.2f} px"
    )
    if overall.get("mean_norm_pct") is not None:
        lines.append(
            f"  Shank-normalized error:   "
            f"{overall['mean_norm_pct']:.2f}% of shank"
        )
    lines.append("")
    lines.extend(_format_kp_block(report))
    return "\n".join(lines)


def format_combined_report(
    combined: dict,
    per_video_reports: List[Tuple[Path, dict]],
) -> str:
    lines = []
    lines.append("========== MODEL EVALUATION ==========")
    lines.append(f"Videos evaluated:         {combined['n_videos']}")
    lines.append(
        f"Total frames reviewed:    {combined.get('n_frames_reviewed', '?')}"
    )
    lines.append(f"Frames annotated:         {combined['n_frames_annotated']}")
    lines.append(
        f"Frames marked not-contact: "
        f"{combined.get('n_frames_rejected_non_contact', 0)}"
    )
    lines.append("")

    lines.append("Contact classifier (across all videos)")
    lines.extend(_format_classifier_block(combined))
    lines.append("")

    overall = combined["overall"]
    if overall["n"] == 0:
        lines.append("No comparable points across any video yet.")
    else:
        lines.append("Keypoint accuracy (across all videos)")
        lines.append(
            f"  Mean pixel error:         {overall['mean_px']:.2f} px"
        )
        if overall.get("mean_norm_pct") is not None:
            lines.append(
                f"  Shank-normalized error:   "
                f"{overall['mean_norm_pct']:.2f}% of shank"
            )
        lines.append("")

        lines.append("Per keypoint (across all videos)")
        lines.extend(_format_kp_block(combined))
        lines.append("")

        scored = [
            (name, s["mean_px"])
            for name, s in combined["per_keypoint"].items()
            if s["n"] > 0
        ]
        if scored:
            best = min(scored, key=lambda pair: pair[1])
            worst = max(scored, key=lambda pair: pair[1])
            lines.append(
                f"Best-tracked keypoint:   {best[0]} ({best[1]:.2f} px)"
            )
            lines.append(
                f"Worst-tracked keypoint:  {worst[0]} ({worst[1]:.2f} px)"
            )
        lines.append("")

    lines.append("========== PER-VIDEO BREAKDOWN ==========")
    lines.append("")
    for video_path, r in per_video_reports:
        lines.append(format_per_video_report(r, video_path))
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=None,
        help="Folder containing test videos. Default: scripts/test_videos/",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Single video to evaluate (overrides --videos-dir).",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=30,
        help="Max frames to review per video (default: 30). 0 = no cap.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip pipeline + click UI. Recompute reports from prior sessions.",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Re-annotate every frame, even ones already annotated.",
    )
    return parser.parse_args()


def evaluate_one_video(
    video_path: Path,
    args: argparse.Namespace,
) -> Optional[Tuple[Path, dict]]:
    print()
    print("=" * 60)
    print(f"Video: {video_path.name}")
    print("=" * 60)

    if args.report_only:
        session_dir = find_session_for_video(video_path)
        if session_dir is None:
            print(
                f"[skip] No prior session found for {video_path.name}. "
                "Run without --report-only first."
            )
            return None
        print(f"Reusing session: {session_dir}")
    else:
        print(f"Running pipeline on: {video_path}")
        try:
            session_dir = run_pipeline_on_video(video_path)
        except Exception as e:
            print(f"[skip] Pipeline failed on {video_path.name}: {e}")
            return None
        print(f"Session: {session_dir}")

    predictions = load_predictions(session_dir)
    images = list_rendered_images(session_dir)
    if not images:
        print(f"[skip] No rendered frames in {session_dir}")
        return None

    if args.max_frames_per_video and args.max_frames_per_video > 0:
        images = images[: args.max_frames_per_video]
        print(f"Reviewing up to {len(images)} frame(s).")

    annotations_path = session_dir / "keypoint_eval_annotations.json"
    rejections_path = session_dir / "keypoint_eval_rejections.json"
    annotations = load_annotations(annotations_path)
    rejections = _load_rejections(rejections_path)

    if not args.report_only:
        run_annotation_session_with_reject(
            images, annotations, rejections,
            annotations_path, rejections_path, redo=args.redo,
        )

    report = compute_mean_report(
        annotations, rejections, predictions,
        total_frames_reviewed=len(images),
    )

    per_video_txt = session_dir / "mean_pixel_error_report.txt"
    per_video_json = session_dir / "mean_pixel_error_report.json"
    per_video_txt.write_text(format_per_video_report(report, video_path))
    per_video_json.write_text(json.dumps(report, indent=2))

    return video_path, report


def main() -> int:
    args = parse_args()

    try:
        videos = collect_videos(args.videos_dir, args.video)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print(f"Found {len(videos)} video(s) to evaluate.")

    videos = check_training_set(videos)
    if not videos:
        print("No videos left after training-set filter. Nothing to do.")
        return 0

    per_video_reports: List[Tuple[Path, dict]] = []
    for video_path in videos:
        result = evaluate_one_video(video_path, args)
        if result is not None:
            per_video_reports.append(result)

    if not per_video_reports:
        print("\nNo successful video evaluations. No report to produce.")
        return 1

    combined = combine_reports(per_video_reports)
    text = format_combined_report(combined, per_video_reports)
    print("\n" + text)

    combined_dir = OUTPUT_ROOT / "batch_evaluation"
    combined_dir.mkdir(parents=True, exist_ok=True)
    (combined_dir / "mean_pixel_error_combined.txt").write_text(text)
    (combined_dir / "mean_pixel_error_combined.json").write_text(
        json.dumps(combined, indent=2)
    )
    print(
        f"\nSaved combined report to: "
        f"{combined_dir / 'mean_pixel_error_combined.txt'}"
    )

    # Presentation one-liner
    overall = combined["overall"]
    c = combined.get("contact_classifier", {})
    print()
    print("=" * 60)
    if overall["n"] > 0:
        norm = overall.get("mean_norm_pct")
        norm_part = (
            f" ({norm:.2f}% of shank length)" if norm is not None else ""
        )
        print(
            f"KEYPOINT MODEL: mean pixel error = {overall['mean_px']:.2f} px"
            f"{norm_part} across {combined['n_frames_annotated']} annotated "
            f"frames from {combined['n_videos']} unseen video(s)."
        )
    fp_rate = c.get("false_positive_rate")
    if fp_rate is not None:
        print(
            f"CONTACT CLASSIFIER: {c['n_false_positive']} of "
            f"{c['n_reviewed']} reviewed 'contact' frames were false "
            f"positives ({fp_rate * 100:.1f}%)."
        )
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())