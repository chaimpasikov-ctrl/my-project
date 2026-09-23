"""Standalone batch evaluator: stance-leg keypoint accuracy on unseen videos.

Runs the analysis pipeline on every video in a folder (or one video),
then for each rendered frame you:
  - press L if the LEFT leg is in stance (foot on ground)
  - press R if the RIGHT leg is in stance
  - press N if the classifier was wrong and no foot is really planted

After L/R you click 3 keypoints for that leg only: knee, ankle, heel.

This produces:
  - stance-leg keypoint accuracy (mean pixel error), broken down by keypoint
  - a contact classifier false-positive rate

Usage:
    # Process every video in scripts/test_videos/
    python scripts/evaluate_on_video.py

    # Specific folder / video
    python scripts/evaluate_on_video.py --videos-dir path/to/videos
    python scripts/evaluate_on_video.py --video path/to/one.mp4

    # Report-only (no pipeline, no clicks — just recompute from prior data)
    python scripts/evaluate_on_video.py --report-only

    # Re-do every frame
    python scripts/evaluate_on_video.py --redo

    # Cap frames per video (default: 30)
    python scripts/evaluate_on_video.py --max-frames-per-video 20

Controls while annotating each frame:
    L               left leg is stance leg    -> click 3 points
    R               right leg is stance leg   -> click 3 points
    N               not a real contact frame  -> counted as classifier error
    S               skip (excluded from report)
    Q               quit and save progress
While placing the 3 clicks:
    left-click      place next point (knee, then ankle, then heel)
    U               undo the last click
    space / enter   confirm the 3 points
    B               back — change your mind about which leg (or reject)
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
    load_predictions,
    list_rendered_images,
)

from app.config.settings import OUTPUT_ROOT  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
DEFAULT_VIDEOS_DIR = Path(__file__).resolve().parent / "test_videos"

# Per-leg keypoint order used inside this script.
# When the user picks LEFT, they click these three global-KEYPOINT_NAMES
# indices, in this order.
KEYPOINT_INDEX_BY_LEG = {
    "left":  [0, 1, 4],   # left_knee, left_ankle, left_heel
    "right": [2, 3, 5],   # right_knee, right_ankle, right_heel
}
LEG_STEP_LABELS = ["knee", "ankle", "heel"]

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Pipeline invocation
# ---------------------------------------------------------------------------
def run_pipeline_on_video(video_path: Path) -> Path:
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
                "app.pipelines.main_pipeline."
            ) from e

    if not isinstance(session_dir, (str, Path)):
        candidates = [
            p for p in OUTPUT_ROOT.iterdir()
            if p.is_dir() and (p / "rendered_predictions").is_dir()
        ]
        if not candidates:
            raise RuntimeError(
                "Pipeline finished but no session directory was found."
            )
        session_dir = max(candidates, key=lambda p: p.stat().st_mtime)

    return Path(session_dir)



def load_frame_records(path: Path) -> Dict[str, dict]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            return {}
        return raw
    except Exception:
        return {}


def save_frame_records(path: Path, records: Dict[str, dict]) -> None:
    path.write_text(json.dumps(records, indent=2))


# ---------------------------------------------------------------------------
# Click UI
# ---------------------------------------------------------------------------
def _put_header(img, top_text, bottom_text):
    cv2.putText(img, top_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(img, bottom_text, (10, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)


def annotate_one_frame(image_path: Path) -> Tuple[Optional[dict], bool]:
    """Annotate a single frame with the L/R/N flow.

    Returns (record, quit_requested):
        record is one of:
          {"kind": "annotated", "stance_leg": "left"|"right",
           "points": [[x,y],[x,y],[x,y]]}
          {"kind": "rejected"}
          {"kind": "skipped"}
          None  -> caller should skip saving (frame not touched, quit fired)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[WARN] Could not read image: {image_path}")
        return None, False

    window = "Evaluate stance leg"
    cv2.namedWindow(window)

    while True:
        # ---- Phase 1: choose leg (or reject/skip/quit) -----------------
        chosen_leg: Optional[str] = None
        while chosen_leg is None:
            display = img.copy()
            _put_header(
                display,
                "L=left leg  R=right leg  N=not-a-real-contact",
                "[s]kip  [q]uit & save",
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("l"):
                chosen_leg = "left"
            elif key == ord("r"):
                chosen_leg = "right"
            elif key == ord("n"):
                cv2.destroyWindow(window)
                return {"kind": "rejected"}, False
            elif key == ord("s"):
                cv2.destroyWindow(window)
                return {"kind": "skipped"}, False
            elif key == ord("q"):
                cv2.destroyWindow(window)
                return None, True

        # ---- Phase 2: click 3 points for the chosen leg ----------------
        clicks: List[Point] = []

        def on_mouse(event, x, y, flags, userdata):
            if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 3:
                clicks.append((float(x), float(y)))

        cv2.setMouseCallback(window, on_mouse)

        went_back = False
        while True:
            display = img.copy()
            for i, pt in enumerate(clicks):
                color = POINT_COLORS[
                    KEYPOINT_INDEX_BY_LEG[chosen_leg][i] % len(POINT_COLORS)
                ]
                cv2.circle(display, (int(pt[0]), int(pt[1])), 5, color, -1)
                cv2.putText(
                    display, LEG_STEP_LABELS[i],
                    (int(pt[0]) + 8, int(pt[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                )

            if len(clicks) < 3:
                top = (
                    f"[{chosen_leg.upper()}]  Click: "
                    f"{LEG_STEP_LABELS[len(clicks)]} ({len(clicks) + 1}/3)"
                )
            else:
                top = (
                    f"[{chosen_leg.upper()}]  3/3 placed - "
                    f"SPACE/ENTER to confirm"
                )
            _put_header(
                display, top,
                "[u]ndo  [b]ack (change leg)  [s]kip  [q]uit & save",
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                cv2.destroyWindow(window)
                return None, True
            if key == ord("s"):
                cv2.destroyWindow(window)
                return {"kind": "skipped"}, False
            if key == ord("b"):
                # go back to leg selection for this frame
                went_back = True
                break
            if key == ord("u") and clicks:
                clicks.pop()
            if len(clicks) == 3 and key in (13, 32):
                cv2.destroyWindow(window)
                return {
                    "kind": "annotated",
                    "stance_leg": chosen_leg,
                    "points": [list(p) for p in clicks],
                }, False

        if went_back:
            # loop back to phase 1
            continue


# ---------------------------------------------------------------------------
# Session-level annotator
# ---------------------------------------------------------------------------
def run_annotation_session(
    images: List[Path],
    records: Dict[str, dict],
    records_path: Path,
    redo: bool,
) -> None:
    def already_seen(name: str) -> bool:
        return name in records and records[name].get("kind") in (
            "annotated", "rejected"
        )

    todo = [img for img in images if redo or not already_seen(img.name)]
    if not todo:
        print(
            "Nothing left to annotate (use --redo to re-annotate everything)."
        )
        return

    print(
        f"Reviewing {len(todo)} frame(s). "
        f"Press L/R to pick the stance leg, N to reject as false contact, "
        f"S skip, Q quit & save.\n"
    )

    for i, image_path in enumerate(todo, start=1):
        print(f"[{i}/{len(todo)}] {image_path.name}")
        record, quit_now = annotate_one_frame(image_path)
        if record is not None:
            records[image_path.name] = record
            save_frame_records(records_path, records)
            kind = record["kind"]
            if kind == "annotated":
                print(
                    f"  -> {record['stance_leg']} leg, "
                    f"knee/ankle/heel placed"
                )
            elif kind == "rejected":
                print("  -> not-a-real-contact (classifier false positive)")
            elif kind == "skipped":
                print("  -> skipped")
        if quit_now:
            print("\nQuit requested - progress saved.")
            break

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Video discovery / training-set check / session lookup
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


def check_training_set(video_paths: List[Path]) -> List[Path]:
    training_list = REPO_ROOT / "data" / "training_video_names.txt"
    if not training_list.is_file():
        print(
            "\nNOTE: Cannot automatically verify these videos weren't in "
            "training. Confirm you're using unseen videos.\n"
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
    return video_paths


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
# Report — stance-leg keypoint accuracy + classifier FP rate
# ---------------------------------------------------------------------------
def compute_report(
    records: Dict[str, dict],
    predictions: Dict[str, dict],
    total_frames_reviewed: int,
) -> dict:
    # Only 3 keypoints per record. We report per-role: knee / ankle / heel.
    per_role: Dict[str, dict] = {
        role: {"errors": [], "norm_errors": []} for role in LEG_STEP_LABELS
    }
    # Also track per-leg counts so we can report imbalance.
    leg_counts = {"left": 0, "right": 0}

    n_annotated = 0
    n_rejected = 0

    for fname, rec in records.items():
        kind = rec.get("kind")
        if kind == "rejected":
            n_rejected += 1
            continue
        if kind != "annotated":
            continue

        n_annotated += 1
        stance_leg = rec["stance_leg"]
        gt_points = rec["points"]  # [knee, ankle, heel]
        leg_counts[stance_leg] += 1

        pred_entry = predictions.get(fname)
        if pred_entry is None:
            continue
        pred_points = pred_entry["points"]
        shank = pred_entry["shank"]

        keypoint_indices = KEYPOINT_INDEX_BY_LEG[stance_leg]

        for role_i, kp_i in enumerate(keypoint_indices):
            gt = tuple(gt_points[role_i])
            pred = (
                pred_points[kp_i]
                if kp_i < len(pred_points) else None
            )
            if pred is None:
                continue
            dx = pred[0] - gt[0]
            dy = pred[1] - gt[1]
            eucl_err = (dx * dx + dy * dy) ** 0.5

            role = LEG_STEP_LABELS[role_i]
            per_role[role]["errors"].append(eucl_err)

            shank_col = SHANK_COLUMN_FOR_KEYPOINT[kp_i]
            shank_px = shank.get(shank_col)
            if shank_px:
                per_role[role]["norm_errors"].append(eucl_err / shank_px)

    role_stats: Dict[str, dict] = {}
    all_errors: List[float] = []
    all_norm_errors: List[float] = []
    for role, data in per_role.items():
        errors = data["errors"]
        norm_errors = data["norm_errors"]
        all_errors.extend(errors)
        all_norm_errors.extend(norm_errors)
        if not errors:
            role_stats[role] = {"n": 0}
            continue
        role_stats[role] = {
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
            statistics.mean(all_norm_errors) * 100
            if all_norm_errors else None
        )

    contact_reviewed = n_annotated + n_rejected
    fp_rate = (
        n_rejected / contact_reviewed if contact_reviewed > 0 else None
    )

    return {
        "n_frames_reviewed": total_frames_reviewed,
        "n_frames_annotated": n_annotated,
        "n_frames_rejected_non_contact": n_rejected,
        "leg_counts": leg_counts,
        "contact_classifier": {
            "n_reviewed": contact_reviewed,
            "n_false_positive": n_rejected,
            "false_positive_rate": fp_rate,
        },
        "overall": overall,
        "per_role": role_stats,
    }


def combine_reports(per_video: List[Tuple[Path, dict]]) -> dict:
    per_role: Dict[str, List[Tuple[float, int]]] = {
        r: [] for r in LEG_STEP_LABELS
    }
    per_role_norm: Dict[str, List[Tuple[float, int]]] = {
        r: [] for r in LEG_STEP_LABELS
    }
    total_reviewed = 0
    total_annotated = 0
    total_rejected = 0
    leg_counts = {"left": 0, "right": 0}

    for _, r in per_video:
        total_reviewed += r.get("n_frames_reviewed", 0)
        total_annotated += r["n_frames_annotated"]
        total_rejected += r.get("n_frames_rejected_non_contact", 0)
        for leg in leg_counts:
            leg_counts[leg] += r.get("leg_counts", {}).get(leg, 0)

        for role, stats in r["per_role"].items():
            if stats["n"] == 0:
                continue
            per_role[role].append((stats["mean_px"], stats["n"]))
            if stats.get("mean_norm_pct") is not None:
                per_role_norm[role].append((stats["mean_norm_pct"], stats["n"]))

    combined_role: Dict[str, dict] = {}
    for role in LEG_STEP_LABELS:
        entries = per_role[role]
        norm_entries = per_role_norm[role]
        if not entries:
            combined_role[role] = {"n": 0}
            continue
        n = sum(e[1] for e in entries)
        weighted_mean = sum(e[0] * e[1] for e in entries) / n
        norm_mean = None
        if norm_entries:
            n_norm = sum(e[1] for e in norm_entries)
            norm_mean = sum(e[0] * e[1] for e in norm_entries) / n_norm
        combined_role[role] = {
            "n": n,
            "mean_px": weighted_mean,
            "mean_norm_pct": norm_mean,
        }

    total_px = 0.0
    total_n = 0
    total_norm = 0.0
    total_n_norm = 0
    for role in LEG_STEP_LABELS:
        s = combined_role[role]
        if s["n"] == 0:
            continue
        total_px += s["mean_px"] * s["n"]
        total_n += s["n"]
        if s.get("mean_norm_pct") is not None:
            total_norm += s["mean_norm_pct"] * s["n"]
            total_n_norm += s["n"]

    overall = {"n": total_n}
    if total_n > 0:
        overall["mean_px"] = total_px / total_n
        overall["mean_norm_pct"] = (
            total_norm / total_n_norm if total_n_norm > 0 else None
        )

    contact_reviewed = total_annotated + total_rejected
    fp_rate = (
        total_rejected / contact_reviewed
        if contact_reviewed > 0 else None
    )

    return {
        "n_videos": len(per_video),
        "n_frames_reviewed": total_reviewed,
        "n_frames_annotated": total_annotated,
        "n_frames_rejected_non_contact": total_rejected,
        "leg_counts": leg_counts,
        "contact_classifier": {
            "n_reviewed": contact_reviewed,
            "n_false_positive": total_rejected,
            "false_positive_rate": fp_rate,
        },
        "overall": overall,
        "per_role": combined_role,
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def _format_role_block(report: dict, indent: str = "  ") -> List[str]:
    lines = []
    header = (
        f"{indent}{'role':<10}{'n':>6}{'mean (px)':>12}{'% shank':>12}"
    )
    lines.append(header)
    lines.append(indent + "-" * (len(header) - len(indent)))
    for role in LEG_STEP_LABELS:
        stats = report["per_role"][role]
        if stats["n"] == 0:
            lines.append(f"{indent}{role:<10}{0:>6}{'-':>12}{'-':>12}")
            continue
        norm_str = (
            f"{stats['mean_norm_pct']:.2f}"
            if stats.get("mean_norm_pct") is not None
            else "-"
        )
        lines.append(
            f"{indent}{role:<10}{stats['n']:>6}"
            f"{stats['mean_px']:>12.2f}{norm_str:>12}"
        )
    return lines


def _format_classifier_block(report: dict, indent: str = "  ") -> List[str]:
    c = report.get("contact_classifier", {})
    n = c.get("n_reviewed", 0)
    fp = c.get("n_false_positive", 0)
    rate = c.get("false_positive_rate")
    if n == 0:
        return [f"{indent}(No classifier data — no frames reviewed)"]
    if rate is not None:
        return [
            f"{indent}Contact frames reviewed:  {n}",
            f"{indent}Marked not-real-contact:  {fp} "
            f"(false-positive rate {rate * 100:.1f}%)",
        ]
    return [
        f"{indent}Contact frames reviewed:  {n}",
        f"{indent}Marked not-real-contact:  {fp}",
    ]


def format_per_video_report(report: dict, video_path: Path) -> str:
    lines = []
    lines.append(f"---- {video_path.name} ----")
    lines.append(
        f"Frames reviewed: {report.get('n_frames_reviewed', '?')}  "
        f"annotated: {report['n_frames_annotated']}  "
        f"rejected: {report.get('n_frames_rejected_non_contact', 0)}"
    )
    leg = report.get("leg_counts", {})
    if leg:
        lines.append(
            f"Stance-leg split:  left={leg.get('left', 0)}, "
            f"right={leg.get('right', 0)}"
        )
    lines.append("")

    lines.append("Contact classifier")
    lines.extend(_format_classifier_block(report))
    lines.append("")

    overall = report["overall"]
    if overall["n"] == 0:
        lines.append("No comparable points for stance-leg keypoints.")
        return "\n".join(lines)

    lines.append("Stance-leg keypoint accuracy")
    lines.append(
        f"  Mean pixel error:         {overall['mean_px']:.2f} px"
    )
    if overall.get("mean_norm_pct") is not None:
        lines.append(
            f"  Shank-normalized error:   "
            f"{overall['mean_norm_pct']:.2f}% of shank"
        )
    lines.append("")
    lines.extend(_format_role_block(report))
    return "\n".join(lines)


def format_combined_report(
    combined: dict, per_video: List[Tuple[Path, dict]]
) -> str:
    lines = []
    lines.append("========== STANCE-LEG MODEL EVALUATION ==========")
    lines.append(f"Videos evaluated:         {combined['n_videos']}")
    lines.append(
        f"Total frames reviewed:    {combined.get('n_frames_reviewed', '?')}"
    )
    lines.append(f"Frames annotated:         {combined['n_frames_annotated']}")
    lines.append(
        f"Frames marked not-contact: "
        f"{combined.get('n_frames_rejected_non_contact', 0)}"
    )
    leg = combined.get("leg_counts", {})
    lines.append(
        f"Stance-leg split:         left={leg.get('left', 0)}, "
        f"right={leg.get('right', 0)}"
    )
    lines.append("")

    lines.append("Contact classifier (across all videos)")
    lines.extend(_format_classifier_block(combined))
    lines.append("")

    overall = combined["overall"]
    if overall["n"] == 0:
        lines.append("No stance-leg keypoint data across any video yet.")
    else:
        lines.append("Stance-leg keypoint accuracy (across all videos)")
        lines.append(
            f"  Mean pixel error:         {overall['mean_px']:.2f} px"
        )
        if overall.get("mean_norm_pct") is not None:
            lines.append(
                f"  Shank-normalized error:   "
                f"{overall['mean_norm_pct']:.2f}% of shank"
            )
        lines.append("")
        lines.append("Per role (across all videos)")
        lines.extend(_format_role_block(combined))
        lines.append("")

        scored = [
            (role, s["mean_px"])
            for role, s in combined["per_role"].items()
            if s["n"] > 0
        ]
        if scored:
            best = min(scored, key=lambda pair: pair[1])
            worst = max(scored, key=lambda pair: pair[1])
            lines.append(
                f"Best-tracked role:   {best[0]} ({best[1]:.2f} px)"
            )
            lines.append(
                f"Worst-tracked role:  {worst[0]} ({worst[1]:.2f} px)"
            )
        lines.append("")

    lines.append("========== PER-VIDEO BREAKDOWN ==========")
    lines.append("")
    for video_path, r in per_video:
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
        "--videos-dir", type=Path, default=None,
        help="Folder containing test videos. Default: scripts/test_videos/",
    )
    parser.add_argument(
        "--video", type=Path, default=None,
        help="Single video to evaluate (overrides --videos-dir).",
    )
    parser.add_argument(
        "--max-frames-per-video", type=int, default=30,
        help="Max frames to review per video (default: 30). 0 = no cap.",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Skip pipeline + click UI. Recompute reports from prior sessions.",
    )
    parser.add_argument(
        "--redo", action="store_true",
        help="Re-annotate every frame, even ones already annotated.",
    )
    return parser.parse_args()


def evaluate_one_video(
    video_path: Path, args: argparse.Namespace,
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

    records_path = session_dir / "stance_leg_eval_records.json"
    records = load_frame_records(records_path)

    if not args.report_only:
        run_annotation_session(images, records, records_path, redo=args.redo)

    report = compute_report(records, predictions, len(images))

    per_video_txt = session_dir / "stance_leg_eval_report.txt"
    per_video_json = session_dir / "stance_leg_eval_report.json"
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
    (combined_dir / "stance_leg_eval_combined.txt").write_text(text)
    (combined_dir / "stance_leg_eval_combined.json").write_text(
        json.dumps(combined, indent=2)
    )
    print(
        f"\nSaved combined report to: "
        f"{combined_dir / 'stance_leg_eval_combined.txt'}"
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
            f"{norm_part} across {combined['n_frames_annotated']} stance-leg "
            f"annotations from {combined['n_videos']} unseen video(s)."
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