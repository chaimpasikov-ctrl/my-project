"""
Batch evaluator for BOTH models:

1) Contact / no-contact model
   - Runs the normal project pipeline on every video in scripts/test_videos/
   - Reviews EVERY frame that the contact model predicted as CONTACT
   - Manual labels:
       L = true contact, LEFT stance leg
       R = true contact, RIGHT stance leg
       N = NOT actually contact -> contact-model false positive
       S = skip
       Q = quit/save

2) Keypoint model
   - On true-contact frames (L/R), checks whether keypoints were produced.
   - If keypoints exist:
       manually click knee -> ankle -> heel on the stance leg
       and compare manual GT with model prediction.
   - If the contact frame has no keypoint prediction:
       it is counted as a keypoint detection failure.

Outputs:
  - Contact false-positive percentage among predicted-contact frames
  - Keypoint detection success/failure rate on true-contact frames
  - Stance-leg knee/ankle/heel pixel error
  - Shank-normalized error
  - Per-video and combined reports

Default:
    python scripts/evaluate_both_models.py

Other examples:
    python scripts/evaluate_both_models.py --videos-dir scripts/test_videos
    python scripts/evaluate_both_models.py --video scripts/test_videos/test1.mp4
    python scripts/evaluate_both_models.py --report-only
    python scripts/evaluate_both_models.py --redo
    python scripts/evaluate_both_models.py --max-contact-frames-per-video 40

Important:
This script does NOT modify your models or training data.
It only runs the existing pipeline and writes evaluation files inside each
analysis-results session plus one combined report.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

# ---------------------------------------------------------------------------
# Repo imports
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.config.settings import OUTPUT_ROOT  # noqa: E402

from scripts.evaluate_keypoints import (  # noqa: E402
    KEYPOINT_NAMES,
    POINT_COLORS,
    SHANK_COLUMN_FOR_KEYPOINT,
    load_predictions,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
DEFAULT_VIDEOS_DIR = Path(__file__).resolve().parent / "test_videos"

# Global keypoint indices expected from your existing evaluator:
# 0 left_knee
# 1 left_ankle
# 2 right_knee
# 3 right_ankle
# 4 left_heel
# 5 right_heel
KEYPOINT_INDEX_BY_LEG = {
    "left": [0, 1, 4],
    "right": [2, 3, 5],
}
LEG_STEP_LABELS = ["knee", "ankle", "heel"]

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline_on_video(video_path: Path) -> Path:
    """Run your existing main pipeline and return its session directory."""
    try:
        from app.pipelines.main_pipeline import process_video
        session_dir = process_video(str(video_path))
    except ImportError:
        try:
            from app.pipelines.main_pipeline import run_pipeline
            session_dir = run_pipeline(str(video_path))
        except ImportError as e:
            raise RuntimeError(
                "Could not import process_video/run_pipeline from "
                "app.pipelines.main_pipeline"
            ) from e

    if isinstance(session_dir, (str, Path)):
        return Path(session_dir)

    # Fallback: newest output directory
    if not OUTPUT_ROOT.is_dir():
        raise RuntimeError("Pipeline finished but OUTPUT_ROOT does not exist.")

    candidates = [p for p in OUTPUT_ROOT.iterdir() if p.is_dir()]
    if not candidates:
        raise RuntimeError("Pipeline finished but no output session was found.")

    return max(candidates, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def collect_videos(
    videos_dir: Optional[Path],
    single_video: Optional[Path],
) -> List[Path]:
    if single_video is not None:
        if not single_video.is_file():
            raise FileNotFoundError(f"Video not found: {single_video}")
        return [single_video]

    folder = videos_dir if videos_dir is not None else DEFAULT_VIDEOS_DIR
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Video folder not found: {folder}\n"
            f"Create {DEFAULT_VIDEOS_DIR} or pass --videos-dir."
        )

    videos = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )

    if not videos:
        raise FileNotFoundError(f"No videos found in: {folder}")

    return videos


def check_training_set(video_paths: List[Path]) -> List[Path]:
    """Optional safeguard against accidentally evaluating training videos."""
    training_list = REPO_ROOT / "data" / "training_video_names.txt"
    if not training_list.is_file():
        return video_paths

    trained = {
        line.strip().lower()
        for line in training_list.read_text().splitlines()
        if line.strip()
    }

    suspect = [v for v in video_paths if v.name.lower() in trained]
    if not suspect:
        return video_paths

    print("\nWARNING: these videos appear in training_video_names.txt:")
    for p in suspect:
        print(f"  - {p.name}")

    answer = input("Skip them? [Y/n] ").strip().lower()
    if answer in ("", "y", "yes"):
        return [v for v in video_paths if v not in suspect]

    return video_paths


def find_session_for_video(video_path: Path) -> Optional[Path]:
    if not OUTPUT_ROOT.is_dir():
        return None

    stem = video_path.stem.lower()
    candidates = [
        p for p in OUTPUT_ROOT.iterdir()
        if p.is_dir() and p.name.lower().startswith(stem)
    ]

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Contact-frame discovery
# ---------------------------------------------------------------------------
def find_contact_frames_dir(session_dir: Path) -> Optional[Path]:
    """
    Find the folder containing frames classified as CONTACT.

    Tries common names first, then searches one level down.
    """
    candidates = [
        session_dir / "contact_frames",
        session_dir / "contacts",
        session_dir / "predicted_contact_frames",
        session_dir / "contact_predictions",
    ]

    for p in candidates:
        if p.is_dir():
            return p

    # Fallback: any child directory containing both "contact" and image files
    for p in session_dir.iterdir():
        if not p.is_dir():
            continue
        if "contact" not in p.name.lower():
            continue
        if any(
            f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            for f in p.iterdir()
        ):
            return p

    return None


def list_contact_frames(session_dir: Path) -> List[Path]:
    """
    IMPORTANT:
    These are the frames the contact classifier said were CONTACT.
    This is the denominator for the manual false-positive check.
    """
    contact_dir = find_contact_frames_dir(session_dir)
    if contact_dir is None:
        return []

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(
        p for p in contact_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_exts
    )


# ---------------------------------------------------------------------------
# Robust prediction matching
# ---------------------------------------------------------------------------
def _canonical_name(name: str) -> str:
    """
    Make filenames easier to match if one side has prefixes/suffixes/extensions.

    Example:
      frame_00123.jpg
      frame_00123_rendered.png
    """
    stem = Path(name).stem.lower()

    suffixes = [
        "_rendered",
        "_prediction",
        "_pred",
        "_pose",
        "_keypoints",
        "_annotated",
    ]
    for suffix in suffixes:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    return stem


def build_prediction_lookup(predictions: Dict[str, dict]) -> Dict[str, dict]:
    lookup: Dict[str, dict] = {}

    for name, entry in predictions.items():
        lookup[name] = entry
        lookup[Path(name).name] = entry
        lookup[Path(name).stem] = entry
        lookup[_canonical_name(name)] = entry

    return lookup


def prediction_for_frame(
    frame_path: Path,
    lookup: Dict[str, dict],
) -> Optional[dict]:
    keys = [
        frame_path.name,
        frame_path.stem,
        _canonical_name(frame_path.name),
    ]

    for key in keys:
        if key in lookup:
            return lookup[key]

    # Last fallback: compare canonical stems
    target = _canonical_name(frame_path.name)
    for key, entry in lookup.items():
        if _canonical_name(str(key)) == target:
            return entry

    return None


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _put_header(img, top_text: str, bottom_text: str) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 42), (0, 0, 0), -1)
    cv2.rectangle(
        img,
        (0, img.shape[0] - 32),
        (img.shape[1], img.shape[0]),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        img,
        top_text,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        bottom_text,
        (10, img.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_model_stance_keypoints(
    img,
    pred_entry: Optional[dict],
    stance_leg: str,
):
    """Overlay model points for the selected stance leg."""
    if pred_entry is None:
        return img

    points = pred_entry.get("points", [])
    kp_indices = KEYPOINT_INDEX_BY_LEG[stance_leg]

    for local_i, kp_i in enumerate(kp_indices):
        if kp_i >= len(points):
            continue

        pt = points[kp_i]
        if pt is None or len(pt) < 2:
            continue

        x, y = int(pt[0]), int(pt[1])
        color = POINT_COLORS[kp_i % len(POINT_COLORS)]
        cv2.circle(img, (x, y), 7, color, 2)
        cv2.putText(
            img,
            f"MODEL {LEG_STEP_LABELS[local_i]}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            color,
            1,
            cv2.LINE_AA,
        )

    return img


# ---------------------------------------------------------------------------
# Manual annotation
# ---------------------------------------------------------------------------
def annotate_contact_frame(
    image_path: Path,
    pred_entry: Optional[dict],
) -> Tuple[Optional[dict], bool]:
    """
    Manual evaluation of one CONTACT-PREDICTED frame.

    Phase 1:
      L / R / N

    Phase 2, only if true contact and keypoint prediction exists:
      click knee / ankle / heel
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[WARN] Cannot open image: {image_path}")
        return {"kind": "skipped", "reason": "image_read_failed"}, False

    window = "Evaluate contact + stance keypoints"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # ------------------------------------------------------------------
    # Phase 1: Is this really contact? Which stance leg?
    # ------------------------------------------------------------------
    chosen_leg: Optional[str] = None

    while chosen_leg is None:
        display = img.copy()

        pose_text = (
            "keypoints: AVAILABLE"
            if pred_entry is not None
            else "keypoints: MISSING"
        )

        _put_header(
            display,
            f"L=left stance  R=right stance  N=false contact   [{pose_text}]",
            "S=skip   Q=quit/save",
        )
        cv2.imshow(window, display)

        key = cv2.waitKey(20) & 0xFF

        if key == ord("l"):
            chosen_leg = "left"
        elif key == ord("r"):
            chosen_leg = "right"
        elif key == ord("n"):
            cv2.destroyWindow(window)
            return {
                "kind": "false_positive_contact",
                "keypoints_available": pred_entry is not None,
            }, False
        elif key == ord("s"):
            cv2.destroyWindow(window)
            return {"kind": "skipped"}, False
        elif key == ord("q"):
            cv2.destroyWindow(window)
            return None, True

    # ------------------------------------------------------------------
    # True contact, but keypoint model failed
    # ------------------------------------------------------------------
    if pred_entry is None:
        display = img.copy()
        _put_header(
            display,
            f"TRUE CONTACT: {chosen_leg.upper()} stance | KEYPOINT MODEL FAILED",
            "Press SPACE/ENTER to record failure   B=back   Q=quit",
        )
        cv2.imshow(window, display)

        while True:
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 32):
                cv2.destroyWindow(window)
                return {
                    "kind": "true_contact_pose_failed",
                    "stance_leg": chosen_leg,
                }, False

            if key == ord("b"):
                # restart annotation for same image
                cv2.destroyWindow(window)
                return annotate_contact_frame(image_path, pred_entry)

            if key == ord("q"):
                cv2.destroyWindow(window)
                return None, True

    # ------------------------------------------------------------------
    # True contact + pose exists: manually click stance-leg GT
    # ------------------------------------------------------------------
    clicks: List[Point] = []

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 3:
            clicks.append((float(x), float(y)))

    cv2.setMouseCallback(window, on_mouse)

    while True:
        display = img.copy()
        draw_model_stance_keypoints(display, pred_entry, chosen_leg)

        for i, pt in enumerate(clicks):
            kp_i = KEYPOINT_INDEX_BY_LEG[chosen_leg][i]
            color = POINT_COLORS[kp_i % len(POINT_COLORS)]

            cv2.circle(
                display,
                (int(pt[0]), int(pt[1])),
                5,
                color,
                -1,
            )
            cv2.putText(
                display,
                f"GT {LEG_STEP_LABELS[i]}",
                (int(pt[0]) + 8, int(pt[1]) + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

        if len(clicks) < 3:
            top = (
                f"{chosen_leg.upper()} STANCE | "
                f"Click GT {LEG_STEP_LABELS[len(clicks)]} "
                f"({len(clicks)+1}/3)"
            )
        else:
            top = (
                f"{chosen_leg.upper()} STANCE | "
                "GT complete - SPACE/ENTER to confirm"
            )

        _put_header(
            display,
            top,
            "U=undo   B=back/change leg   S=skip   Q=quit",
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
            cv2.destroyWindow(window)
            return annotate_contact_frame(image_path, pred_entry)

        if key == ord("u") and clicks:
            clicks.pop()

        if len(clicks) == 3 and key in (13, 32):
            cv2.destroyWindow(window)
            return {
                "kind": "true_contact_pose_evaluated",
                "stance_leg": chosen_leg,
                "points": [list(p) for p in clicks],
            }, False


def run_annotation_session(
    contact_frames: List[Path],
    prediction_lookup: Dict[str, dict],
    records: Dict[str, dict],
    records_path: Path,
    redo: bool,
) -> None:
    def already_done(name: str) -> bool:
        return name in records and records[name].get("kind") not in (None, "")

    todo = [
        p for p in contact_frames
        if redo or not already_done(p.name)
    ]

    if not todo:
        print("Nothing left to annotate.")
        return

    print(f"\nManual review: {len(todo)} predicted-contact frame(s)")
    print("L/R = true contact + stance leg")
    print("N   = false positive from contact model")
    print()

    for i, frame_path in enumerate(todo, start=1):
        pred_entry = prediction_for_frame(frame_path, prediction_lookup)

        print(
            f"[{i}/{len(todo)}] {frame_path.name} | "
            f"keypoints={'yes' if pred_entry is not None else 'NO'}"
        )

        record, quit_now = annotate_contact_frame(
            frame_path,
            pred_entry,
        )

        if record is not None:
            records[frame_path.name] = record
            save_json(records_path, records)

        if quit_now:
            print("Quit requested. Progress saved.")
            break

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_report(
    records: Dict[str, dict],
    predictions: Dict[str, dict],
    contact_frame_names: List[str],
) -> dict:
    prediction_lookup = build_prediction_lookup(predictions)

    contact_reviewed = 0
    contact_false_positive = 0
    true_contact = 0
    skipped = 0

    pose_success = 0
    pose_failed = 0

    leg_counts = {"left": 0, "right": 0}

    per_role = {
        role: {
            "errors_px": [],
            "errors_norm": [],
        }
        for role in LEG_STEP_LABELS
    }

    # Evaluate only frames from the current predicted-contact set.
    # This avoids stale records from older runs.
    valid_names = set(contact_frame_names)

    for fname, rec in records.items():
        if fname not in valid_names:
            continue

        kind = rec.get("kind")

        if kind == "skipped":
            skipped += 1
            continue

        if kind == "false_positive_contact":
            contact_reviewed += 1
            contact_false_positive += 1
            continue

        if kind == "true_contact_pose_failed":
            contact_reviewed += 1
            true_contact += 1
            pose_failed += 1

            leg = rec.get("stance_leg")
            if leg in leg_counts:
                leg_counts[leg] += 1
            continue

        if kind != "true_contact_pose_evaluated":
            continue

        contact_reviewed += 1
        true_contact += 1
        pose_success += 1

        stance_leg = rec["stance_leg"]
        leg_counts[stance_leg] += 1

        pred_entry = prediction_for_frame(
            Path(fname),
            prediction_lookup,
        )

        # Defensive fallback:
        # If record says it was evaluated but prediction is now unavailable,
        # count it as pose failure instead of silently ignoring it.
        if pred_entry is None:
            pose_success -= 1
            pose_failed += 1
            continue

        gt_points = rec["points"]
        pred_points = pred_entry.get("points", [])
        shank = pred_entry.get("shank", {})

        kp_indices = KEYPOINT_INDEX_BY_LEG[stance_leg]

        for role_i, kp_i in enumerate(kp_indices):
            if role_i >= len(gt_points):
                continue
            if kp_i >= len(pred_points):
                continue

            pred = pred_points[kp_i]
            if pred is None or len(pred) < 2:
                continue

            gt = gt_points[role_i]

            dx = float(pred[0]) - float(gt[0])
            dy = float(pred[1]) - float(gt[1])
            err = (dx * dx + dy * dy) ** 0.5

            role = LEG_STEP_LABELS[role_i]
            per_role[role]["errors_px"].append(err)

            # Keep compatibility with your current evaluate_keypoints.py
            try:
                shank_col = SHANK_COLUMN_FOR_KEYPOINT[kp_i]
                shank_px = shank.get(shank_col)

                if shank_px is not None and float(shank_px) > 0:
                    per_role[role]["errors_norm"].append(
                        err / float(shank_px)
                    )
            except Exception:
                pass

    # -------------------------------
    # Contact classifier stats
    # -------------------------------
    contact_fp_rate = (
        contact_false_positive / contact_reviewed
        if contact_reviewed > 0
        else None
    )

    contact_precision = (
        true_contact / contact_reviewed
        if contact_reviewed > 0
        else None
    )

    # -------------------------------
    # Pose availability stats
    # -------------------------------
    pose_total_true_contact = pose_success + pose_failed

    pose_success_rate = (
        pose_success / pose_total_true_contact
        if pose_total_true_contact > 0
        else None
    )

    # -------------------------------
    # Keypoint error stats
    # -------------------------------
    role_stats = {}
    all_px = []
    all_norm = []

    for role, values in per_role.items():
        px = values["errors_px"]
        norm = values["errors_norm"]

        all_px.extend(px)
        all_norm.extend(norm)

        if not px:
            role_stats[role] = {"n": 0}
            continue

        role_stats[role] = {
            "n": len(px),
            "mean_px": statistics.mean(px),
            "median_px": statistics.median(px),
            "mean_norm_pct": (
                statistics.mean(norm) * 100
                if norm else None
            ),
            "median_norm_pct": (
                statistics.median(norm) * 100
                if norm else None
            ),
        }

    overall = {"n_points": len(all_px)}

    if all_px:
        overall.update({
            "mean_px": statistics.mean(all_px),
            "median_px": statistics.median(all_px),
            "mean_norm_pct": (
                statistics.mean(all_norm) * 100
                if all_norm else None
            ),
            "median_norm_pct": (
                statistics.median(all_norm) * 100
                if all_norm else None
            ),
        })

    return {
        "predicted_contact_frames_available": len(contact_frame_names),
        "manual_contact_frames_reviewed": contact_reviewed,
        "manual_frames_skipped": skipped,

        "contact_classifier": {
            "n_predicted_contact_reviewed": contact_reviewed,
            "n_true_contact": true_contact,
            "n_false_positive": contact_false_positive,
            "false_positive_fraction_among_predicted_contacts": contact_fp_rate,
            "precision_among_reviewed_predicted_contacts": contact_precision,
        },

        "keypoint_detection": {
            "n_true_contact_frames": pose_total_true_contact,
            "n_pose_success": pose_success,
            "n_pose_failed": pose_failed,
            "pose_success_rate": pose_success_rate,
        },

        "stance_leg_counts": leg_counts,
        "keypoint_accuracy": {
            "overall": overall,
            "per_role": role_stats,
        },
    }


# ---------------------------------------------------------------------------
# Combined reports
# ---------------------------------------------------------------------------
def combine_reports(per_video: List[Tuple[Path, dict]]) -> dict:
    total_contact_reviewed = 0
    total_true_contact = 0
    total_fp = 0

    total_pose_success = 0
    total_pose_failed = 0

    total_contact_frames_available = 0
    total_skipped = 0

    leg_counts = {"left": 0, "right": 0}

    role_px: Dict[str, List[Tuple[float, int]]] = {
        role: [] for role in LEG_STEP_LABELS
    }
    role_median_px_values: Dict[str, List[Tuple[float, int]]] = {
        role: [] for role in LEG_STEP_LABELS
    }
    role_norm: Dict[str, List[Tuple[float, int]]] = {
        role: [] for role in LEG_STEP_LABELS
    }

    for _, report in per_video:
        total_contact_frames_available += report.get(
            "predicted_contact_frames_available", 0
        )
        total_skipped += report.get("manual_frames_skipped", 0)

        c = report["contact_classifier"]
        total_contact_reviewed += c["n_predicted_contact_reviewed"]
        total_true_contact += c["n_true_contact"]
        total_fp += c["n_false_positive"]

        kd = report["keypoint_detection"]
        total_pose_success += kd["n_pose_success"]
        total_pose_failed += kd["n_pose_failed"]

        for leg in leg_counts:
            leg_counts[leg] += report.get(
                "stance_leg_counts", {}
            ).get(leg, 0)

        for role, stats in report["keypoint_accuracy"]["per_role"].items():
            n = stats.get("n", 0)
            if not n:
                continue

            role_px[role].append((stats["mean_px"], n))

            if stats.get("mean_norm_pct") is not None:
                role_norm[role].append(
                    (stats["mean_norm_pct"], n)
                )

    contact_fp_rate = (
        total_fp / total_contact_reviewed
        if total_contact_reviewed > 0 else None
    )

    contact_precision = (
        total_true_contact / total_contact_reviewed
        if total_contact_reviewed > 0 else None
    )

    pose_total = total_pose_success + total_pose_failed
    pose_success_rate = (
        total_pose_success / pose_total
        if pose_total > 0 else None
    )

    combined_role = {}

    total_weighted_px = 0.0
    total_px_n = 0

    total_weighted_norm = 0.0
    total_norm_n = 0

    for role in LEG_STEP_LABELS:
        entries = role_px[role]
        norms = role_norm[role]

        if not entries:
            combined_role[role] = {"n": 0}
            continue

        n = sum(count for _, count in entries)
        mean_px = sum(v * count for v, count in entries) / n

        combined_role[role] = {
            "n": n,
            "mean_px": mean_px,
        }

        total_weighted_px += mean_px * n
        total_px_n += n

        if norms:
            nn = sum(count for _, count in norms)
            mean_norm = sum(v * count for v, count in norms) / nn

            combined_role[role]["mean_norm_pct"] = mean_norm

            total_weighted_norm += mean_norm * nn
            total_norm_n += nn

    overall = {"n_points": total_px_n}

    if total_px_n:
        overall["mean_px"] = total_weighted_px / total_px_n

    if total_norm_n:
        overall["mean_norm_pct"] = (
            total_weighted_norm / total_norm_n
        )

    return {
        "n_videos": len(per_video),
        "predicted_contact_frames_available": total_contact_frames_available,
        "manual_frames_skipped": total_skipped,

        "contact_classifier": {
            "n_predicted_contact_reviewed": total_contact_reviewed,
            "n_true_contact": total_true_contact,
            "n_false_positive": total_fp,
            "false_positive_fraction_among_predicted_contacts": contact_fp_rate,
            "precision_among_reviewed_predicted_contacts": contact_precision,
        },

        "keypoint_detection": {
            "n_true_contact_frames": pose_total,
            "n_pose_success": total_pose_success,
            "n_pose_failed": total_pose_failed,
            "pose_success_rate": pose_success_rate,
        },

        "stance_leg_counts": leg_counts,

        "keypoint_accuracy": {
            "overall": overall,
            "per_role": combined_role,
        },
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def pct(value: Optional[float]) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def format_report(report: dict, title: str) -> str:
    lines = []
    lines.append("=" * 68)
    lines.append(title)
    lines.append("=" * 68)

    c = report["contact_classifier"]
    kd = report["keypoint_detection"]
    kp = report["keypoint_accuracy"]
    leg = report["stance_leg_counts"]

    lines.append("")
    lines.append("CONTACT MODEL")
    lines.append(
        f"Predicted-contact frames reviewed: "
        f"{c['n_predicted_contact_reviewed']}"
    )
    lines.append(
        f"Actually contact:                  "
        f"{c['n_true_contact']}"
    )
    lines.append(
        f"False positives:                   "
        f"{c['n_false_positive']}"
    )
    lines.append(
        f"False-positive percentage:         "
        f"{pct(c['false_positive_fraction_among_predicted_contacts'])}"
    )
    lines.append(
        f"Precision on reviewed positives:   "
        f"{pct(c['precision_among_reviewed_predicted_contacts'])}"
    )

    lines.append("")
    lines.append("KEYPOINT DETECTION")
    lines.append(
        f"True-contact frames reviewed:      "
        f"{kd['n_true_contact_frames']}"
    )
    lines.append(
        f"Keypoint detection succeeded:      "
        f"{kd['n_pose_success']}"
    )
    lines.append(
        f"Keypoint detection failed:         "
        f"{kd['n_pose_failed']}"
    )
    lines.append(
        f"Detection success rate:            "
        f"{pct(kd['pose_success_rate'])}"
    )

    lines.append("")
    lines.append(
        f"Stance-leg split: left={leg.get('left', 0)}, "
        f"right={leg.get('right', 0)}"
    )

    lines.append("")
    lines.append("STANCE-LEG KEYPOINT ERROR")
    overall = kp["overall"]

    if overall.get("n_points", 0) == 0:
        lines.append("No comparable stance-leg keypoints.")
    else:
        lines.append(
            f"Points compared:                   "
            f"{overall['n_points']}"
        )
        lines.append(
            f"Mean pixel error:                  "
            f"{overall['mean_px']:.2f} px"
        )

        if overall.get("mean_norm_pct") is not None:
            lines.append(
                f"Mean normalized error:             "
                f"{overall['mean_norm_pct']:.2f}% of shank"
            )

        lines.append("")
        lines.append(
            f"{'Keypoint':<12}{'n':>7}{'mean px':>12}{'% shank':>12}"
        )
        lines.append("-" * 43)

        for role in LEG_STEP_LABELS:
            s = kp["per_role"][role]

            if s.get("n", 0) == 0:
                lines.append(
                    f"{role:<12}{0:>7}{'-':>12}{'-':>12}"
                )
                continue

            norm = (
                f"{s['mean_norm_pct']:.2f}"
                if s.get("mean_norm_pct") is not None
                else "-"
            )

            lines.append(
                f"{role:<12}"
                f"{s['n']:>7}"
                f"{s['mean_px']:>12.2f}"
                f"{norm:>12}"
            )

    return "\n".join(lines)


def format_combined(
    combined: dict,
    per_video: List[Tuple[Path, dict]],
) -> str:
    parts = [
        format_report(
            combined,
            f"COMBINED EVALUATION — {combined['n_videos']} VIDEO(S)"
        ),
        "",
        "",
        "PER-VIDEO RESULTS",
        "",
    ]

    for video_path, report in per_video:
        parts.append(format_report(report, video_path.name))
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Single-video evaluation
# ---------------------------------------------------------------------------
def evaluate_one_video(
    video_path: Path,
    args: argparse.Namespace,
) -> Optional[Tuple[Path, dict]]:
    print("\n" + "=" * 68)
    print(f"VIDEO: {video_path.name}")
    print("=" * 68)

    if args.report_only:
        session_dir = find_session_for_video(video_path)

        if session_dir is None:
            print("[skip] No existing session found.")
            return None

        print(f"Using existing session: {session_dir}")

    else:
        print("Running project pipeline...")
        try:
            session_dir = run_pipeline_on_video(video_path)
        except Exception as e:
            print(f"[skip] Pipeline failed: {e}")
            return None

        print(f"Session: {session_dir}")

    # -----------------------------------------------------------
    # 1) ALL predicted-contact frames
    # -----------------------------------------------------------
    contact_frames = list_contact_frames(session_dir)

    if not contact_frames:
        print(
            "[skip] Could not find any predicted-contact frame images.\n"
            "Expected a folder like session/contact_frames/."
        )
        return None

    # Optional cap
    if (
        args.max_contact_frames_per_video
        and args.max_contact_frames_per_video > 0
        and len(contact_frames) > args.max_contact_frames_per_video
    ):
        # Evenly sample instead of taking the first N.
        n = args.max_contact_frames_per_video

        if n == 1:
            indices = [len(contact_frames) // 2]
        else:
            indices = [
                round(i * (len(contact_frames) - 1) / (n - 1))
                for i in range(n)
            ]

        contact_frames = [contact_frames[i] for i in indices]

    print(
        f"Predicted-contact frames selected for review: "
        f"{len(contact_frames)}"
    )

    # -----------------------------------------------------------
    # 2) Keypoint predictions
    # -----------------------------------------------------------
    try:
        predictions = load_predictions(session_dir)
    except Exception as e:
        print(f"[WARN] Could not load keypoint predictions: {e}")
        predictions = {}

    prediction_lookup = build_prediction_lookup(predictions)

    n_pose_available = sum(
        prediction_for_frame(p, prediction_lookup) is not None
        for p in contact_frames
    )

    print(
        f"Frames with keypoint prediction: {n_pose_available}/"
        f"{len(contact_frames)}"
    )

    # -----------------------------------------------------------
    # 3) Manual evaluation
    # -----------------------------------------------------------
    records_path = session_dir / "both_models_eval_records.json"
    records = load_json(records_path)

    if not args.report_only:
        run_annotation_session(
            contact_frames=contact_frames,
            prediction_lookup=prediction_lookup,
            records=records,
            records_path=records_path,
            redo=args.redo,
        )

    # -----------------------------------------------------------
    # 4) Report
    # -----------------------------------------------------------
    report = compute_report(
        records=records,
        predictions=predictions,
        contact_frame_names=[p.name for p in contact_frames],
    )

    txt = format_report(report, video_path.name)

    (session_dir / "both_models_eval_report.txt").write_text(txt)
    (session_dir / "both_models_eval_report.json").write_text(
        json.dumps(report, indent=2)
    )

    print()
    print(txt)

    return video_path, report


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
        help=(
            "Folder of test videos. "
            "Default: scripts/test_videos/"
        ),
    )

    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Evaluate one video instead of the whole folder.",
    )

    parser.add_argument(
        "--max-contact-frames-per-video",
        type=int,
        default=30,
        help=(
            "Maximum predicted-contact frames to manually review per video. "
            "Frames are sampled EVENLY across the contact sequence. "
            "Use 0 for all frames. Default: 30."
        ),
    )

    parser.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "Do not rerun pipeline or annotation UI. "
            "Recompute reports from saved records."
        ),
    )

    parser.add_argument(
        "--redo",
        action="store_true",
        help="Redo manual annotations.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        videos = collect_videos(args.videos_dir, args.video)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1

    print(f"Found {len(videos)} video(s).")

    videos = check_training_set(videos)

    if not videos:
        print("No videos left to evaluate.")
        return 0

    per_video_reports: List[Tuple[Path, dict]] = []

    for video_path in videos:
        result = evaluate_one_video(video_path, args)

        if result is not None:
            per_video_reports.append(result)

    if not per_video_reports:
        print("\nNo successful evaluations.")
        return 1

    combined = combine_reports(per_video_reports)
    combined_text = format_combined(combined, per_video_reports)

    print("\n\n" + combined_text)

    out_dir = OUTPUT_ROOT / "batch_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / "both_models_eval_combined.txt"
    json_path = out_dir / "both_models_eval_combined.json"

    txt_path.write_text(combined_text)
    json_path.write_text(json.dumps(combined, indent=2))

    print("\nSaved:")
    print(f"  {txt_path}")
    print(f"  {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
