"""
Evaluate the EXISTING contact/no-contact model together with MediaPipe Pose.

Purpose
-------
This script mirrors scripts/evaluate_both_models.py as closely as practical,
but replaces the project's custom keypoint model with MediaPipe Pose.

The contact model is unchanged:
    1) Run ONLY the project's existing contact/no-contact workflow on each test video.
    2) Review the same frames predicted as CONTACT.
    3) L/R = true contact with left/right stance leg.
       N   = contact-model false positive.
       S   = skip.
       Q   = quit/save.

For TRUE-CONTACT frames:
    - MediaPipe predicts knee, ankle, and heel for the chosen stance leg.
    - You manually click knee -> ankle -> heel.
    - The script measures Euclidean error in pixels.
    - It also reports error normalized by manual shank length
      (manual knee-to-ankle distance), matching the project's evaluator.

This gives a direct baseline for the custom six-keypoint model:
    custom keypoint model vs. MediaPipe
using the SAME videos, SAME contact model, SAME stance-leg labels,
and SAME manual ground-truth procedure.

Outputs
-------
Per session:
    mediapipe_eval_records.json
    mediapipe_eval_report.json
    mediapipe_eval_report.txt

Combined:
    analysis_results/batch_evaluation/mediapipe_eval_combined.json
    analysis_results/batch_evaluation/mediapipe_eval_combined.txt

Examples
--------
    python scripts/evaluate_mediapipe_contact_only.py

    python scripts/evaluate_mediapipe_contact_only.py \
        --videos-dir scripts/test_videos

    python scripts/evaluate_mediapipe_contact_only.py \
        --video scripts/test_videos/test_vid_1.MOV

    python scripts/evaluate_mediapipe_contact_only.py --redo

Notes
-----
- This file does NOT modify either trained model.
- The project contact classifier is still used exactly as before.
- The project's RF-DETR keypoint workflow is NOT CALLED at all.
- MediaPipe's built-in LEFT/RIGHT_KNEE, LEFT/RIGHT_ANKLE,
  and LEFT/RIGHT_HEEL landmarks are used.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp


# ---------------------------------------------------------------------------
# Repo imports
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.config.settings import (  # noqa: E402
    CONTACT_CONF_THRESHOLD,
    CONTACT_WORKFLOW_ID,
    FRAME_STRIDE,
    OUTPUT_ROOT,
    ROBOFLOW_WORKSPACE_NAME,
)
from app.inference.contact_inference import infer_contact  # noqa: E402

# Reuse the contact-model / file-discovery helpers from your existing evaluator.
# This is deliberate: it keeps the contact side of the experiment identical.
from scripts.evaluate_both_models import (  # noqa: E402
    collect_videos,
    check_training_set,
    find_session_for_video,
    list_contact_frames,
    load_json,
    save_json,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_VIDEOS_DIR = Path(__file__).resolve().parent / "test_videos"
LEG_STEP_LABELS = ["knee", "ankle", "heel"]

Point = Tuple[float, float]

mp_pose = mp.solutions.pose

MEDIAPIPE_INDEX_BY_LEG = {
    "left": {
        "knee": mp_pose.PoseLandmark.LEFT_KNEE.value,
        "ankle": mp_pose.PoseLandmark.LEFT_ANKLE.value,
        "heel": mp_pose.PoseLandmark.LEFT_HEEL.value,
    },
    "right": {
        "knee": mp_pose.PoseLandmark.RIGHT_KNEE.value,
        "ankle": mp_pose.PoseLandmark.RIGHT_ANKLE.value,
        "heel": mp_pose.PoseLandmark.RIGHT_HEEL.value,
    },
}

ROLE_COLORS = {
    "knee": (0, 255, 255),
    "ankle": (0, 255, 0),
    "heel": (255, 0, 255),
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _put_header(img, top_text: str, bottom_text: str) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
    cv2.rectangle(
        img,
        (0, img.shape[0] - 34),
        (img.shape[1], img.shape[0]),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        img,
        top_text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        bottom_text,
        (10, img.shape[0] - 11),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _resize_for_screen(img, max_width=1500, max_height=900):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale >= 1.0:
        return img, 1.0

    resized = cv2.resize(
        img,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _draw_predicted_points(img, pred: Optional[dict]):
    if not pred:
        return img

    out = img.copy()

    points = pred.get("points", {})
    for role in LEG_STEP_LABELS:
        entry = points.get(role)
        if not entry:
            continue

        x = int(round(entry["x"]))
        y = int(round(entry["y"]))
        color = ROLE_COLORS[role]

        cv2.circle(out, (x, y), 8, color, -1)
        cv2.circle(out, (x, y), 11, (255, 255, 255), 2)

        cv2.putText(
            out,
            f"MP {role}",
            (x + 10, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
            cv2.LINE_AA,
        )

    # Connect knee -> ankle -> heel when all three exist.
    if all(points.get(role) for role in LEG_STEP_LABELS):
        p = {
            role: (
                int(round(points[role]["x"])),
                int(round(points[role]["y"])),
            )
            for role in LEG_STEP_LABELS
        }
        cv2.line(out, p["knee"], p["ankle"], (255, 255, 255), 2)
        cv2.line(out, p["ankle"], p["heel"], (255, 255, 255), 2)

    return out



# ---------------------------------------------------------------------------
# Contact-only project inference
# ---------------------------------------------------------------------------

def run_contact_only_on_video(video_path: Path) -> Path:
    """
    Run ONLY the project's existing contact/no-contact workflow.

    This deliberately does NOT call app.pipelines.main_pipeline.run_pipeline(),
    because that function also invokes the RF-DETR keypoint workflow.

    The output layout matches the normal project session closely enough for
    list_contact_frames() and the rest of this evaluator to work unchanged.
    """
    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    session_dir = Path(OUTPUT_ROOT) / video_path.stem
    sampled_frames_dir = session_dir / "sampled_frames"
    contact_frames_dir = session_dir / "contact_frames"

    session_dir.mkdir(parents=True, exist_ok=True)
    sampled_frames_dir.mkdir(parents=True, exist_ok=True)
    contact_frames_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale images from an older run of the same video. This is important:
    # otherwise list_contact_frames() could accidentally evaluate frames that the
    # current contact-model run did not predict as contact.
    for folder in (sampled_frames_dir, contact_frames_dir):
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                p.unlink()

    cap = cv2.VideoCapture(str(video_path))

    # Do not let OpenCV auto-rotate phone videos using metadata.
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frame_idx = 0
    sampled_count = 0
    contact_count = 0
    rows = []

    print(
        f"Processing CONTACT ONLY: {video_path} "
        f"(fps={fps:.2f}, frames={total_frames})"
    )
    print(
        f"Contact workflow: "
        f"{ROBOFLOW_WORKSPACE_NAME}/{CONTACT_WORKFLOW_ID}"
    )
    print(
        "RF-DETR keypoint workflow: NOT CALLED"
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % FRAME_STRIDE != 0:
                frame_idx += 1
                continue

            sampled_count += 1
            frame_name = f"frame_{frame_idx:06d}.jpg"
            sampled_path = sampled_frames_dir / frame_name

            if not cv2.imwrite(str(sampled_path), frame):
                raise RuntimeError(
                    f"Failed to save sampled frame: {sampled_path}"
                )

            try:
                contact_label, contact_conf, _ = infer_contact(
                    str(sampled_path),
                    workspace_name=ROBOFLOW_WORKSPACE_NAME,
                    workflow_id=CONTACT_WORKFLOW_ID,
                )
            except Exception as e:
                print(
                    f"[WARN] Contact inference failed on frame "
                    f"{frame_idx}: {e}"
                )
                rows.append({
                    "frame_idx": frame_idx,
                    "file": frame_name,
                    "contact_label": "",
                    "contact_conf": 0.0,
                    "kept": 0,
                })
                frame_idx += 1
                continue

            kept = int(
                contact_label == "contact"
                and contact_conf >= CONTACT_CONF_THRESHOLD
            )

            rows.append({
                "frame_idx": frame_idx,
                "file": frame_name,
                "contact_label": contact_label,
                "contact_conf": float(contact_conf),
                "kept": kept,
            })

            if kept:
                contact_count += 1
                contact_path = contact_frames_dir / frame_name

                if not cv2.imwrite(str(contact_path), frame):
                    raise RuntimeError(
                        f"Failed to save contact frame: {contact_path}"
                    )

            frame_idx += 1

    finally:
        cap.release()

    # Save a small contact-only record for reproducibility/debugging.
    csv_path = session_dir / "mediapipe_contact_predictions.csv"
    import csv

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx",
                "file",
                "contact_label",
                "contact_conf",
                "kept",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    meta_path = session_dir / "mediapipe_contact_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "video": str(video_path),
                "fps": fps,
                "total_frames": total_frames,
                "frame_stride": int(FRAME_STRIDE),
                "sampled_frames": sampled_count,
                "contact_frames": contact_count,
                "contact_conf_threshold": float(CONTACT_CONF_THRESHOLD),
                "workspace_name": ROBOFLOW_WORKSPACE_NAME,
                "contact_workflow_id": CONTACT_WORKFLOW_ID,
                "keypoint_workflow_called": False,
            },
            indent=2,
        )
    )

    print(f"Sampled frames:     {sampled_count}")
    print(f"Contact frames kept: {contact_count}")
    print(f"Contact-only CSV:    {csv_path}")
    print(f"Session:             {session_dir}")

    return session_dir


# ---------------------------------------------------------------------------
# MediaPipe inference
# ---------------------------------------------------------------------------

def create_mediapipe_pose(
    model_complexity: int,
    min_detection_confidence: float,
):
    """
    Static-image mode is intentional here.

    We evaluate the exact saved contact frames one-by-one, just as the custom
    keypoint evaluator evaluates individual predictions. Using tracking would
    make MediaPipe depend on frames that are not part of the sampled evaluation.
    """
    return mp_pose.Pose(
        static_image_mode=True,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
    )


def mediapipe_predict_stance_leg(
    image_bgr,
    pose,
    stance_leg: str,
    min_visibility: float,
) -> Optional[dict]:
    """
    Return knee/ankle/heel coordinates for the selected stance leg.

    Detection counts as failed when:
      - MediaPipe finds no pose, or
      - one of the three required landmarks has visibility below threshold.

    Coordinates are returned in ORIGINAL image pixels.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if not result.pose_landmarks:
        return None

    h, w = image_bgr.shape[:2]
    landmarks = result.pose_landmarks.landmark

    points = {}

    for role, idx in MEDIAPIPE_INDEX_BY_LEG[stance_leg].items():
        lm = landmarks[idx]

        visibility = float(getattr(lm, "visibility", 0.0))

        if visibility < min_visibility:
            return None

        x = float(lm.x * w)
        y = float(lm.y * h)

        # Guard against wildly invalid normalized outputs.
        if not (-0.10 * w <= x <= 1.10 * w):
            return None
        if not (-0.10 * h <= y <= 1.10 * h):
            return None

        points[role] = {
            "x": x,
            "y": y,
            "visibility": visibility,
        }

    return {
        "stance_leg": stance_leg,
        "points": points,
    }


# ---------------------------------------------------------------------------
# Annotation UI
# ---------------------------------------------------------------------------

def annotate_contact_frame(
    frame_path: Path,
    pose,
    min_visibility: float,
):
    """
    First choose whether the project contact model was correct.

    Keys:
      L = true contact; left leg is stance leg
      R = true contact; right leg is stance leg
      N = NOT actually contact
      S = skip
      Q = quit/save

    For true contact:
      manually click knee -> ankle -> heel
      Enter/Space accepts
      U removes last click
      Esc returns to contact classification
    """
    img = cv2.imread(str(frame_path))
    if img is None:
        print(f"[warning] Could not read {frame_path}")
        return {"kind": "skipped"}, False

    # -------------------------
    # Stage 1: contact review
    # -------------------------
    window = "MediaPipe evaluation"

    while True:
        display = img.copy()
        _put_header(
            display,
            "CONTACT REVIEW",
            "L=left stance | R=right stance | N=not contact | S=skip | Q=quit",
        )

        screen, _ = _resize_for_screen(display)
        cv2.imshow(window, screen)

        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), ord("Q")):
            cv2.destroyWindow(window)
            return None, True

        if key in (ord("s"), ord("S")):
            cv2.destroyWindow(window)
            return {"kind": "skipped"}, False

        if key in (ord("n"), ord("N")):
            cv2.destroyWindow(window)
            return {"kind": "false_positive_contact"}, False

        if key in (ord("l"), ord("L")):
            stance_leg = "left"
            break

        if key in (ord("r"), ord("R")):
            stance_leg = "right"
            break

    # -------------------------
    # Stage 2: MediaPipe
    # -------------------------
    pred = mediapipe_predict_stance_leg(
        img,
        pose,
        stance_leg=stance_leg,
        min_visibility=min_visibility,
    )

    if pred is None:
        # We still know the contact model was correct, but MediaPipe failed.
        failure_view = img.copy()
        _put_header(
            failure_view,
            f"TRUE CONTACT — {stance_leg.upper()} stance",
            "MediaPipe missing/low-visibility knee, ankle, or heel. "
            "Press any key.",
        )
        screen, _ = _resize_for_screen(failure_view)
        cv2.imshow(window, screen)
        cv2.waitKey(0)

        cv2.destroyWindow(window)
        return {
            "kind": "true_contact_pose_failed",
            "stance_leg": stance_leg,
        }, False

    # -------------------------
    # Stage 3: manual GT clicks
    # -------------------------
    clicks: List[Point] = []

    def mouse_callback(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(clicks) >= 3:
            return

        # x/y refer to screen-resized image. Convert back to original pixels.
        scale = param["scale"]
        clicks.append((float(x / scale), float(y / scale)))

    while True:
        display = _draw_predicted_points(img, pred)

        # Draw manual clicks.
        for i, pt in enumerate(clicks):
            role = LEG_STEP_LABELS[i]
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(display, (x, y), 7, (0, 0, 255), -1)
            cv2.putText(
                display,
                f"GT {role}",
                (x + 10, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        next_role = (
            LEG_STEP_LABELS[len(clicks)]
            if len(clicks) < 3
            else "DONE"
        )

        _put_header(
            display,
            f"{stance_leg.upper()} stance — click: {next_role}",
            "Manual GT order: KNEE -> ANKLE -> HEEL | "
            "U=undo | Enter/Space=accept | Esc=back",
        )

        screen, scale = _resize_for_screen(display)
        cv2.imshow(window, screen)
        cv2.setMouseCallback(
            window,
            mouse_callback,
            {"scale": scale},
        )

        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # Esc
            # Return to classifying this frame.
            cv2.destroyWindow(window)
            return annotate_contact_frame(
                frame_path,
                pose,
                min_visibility,
            )

        if key in (ord("u"), ord("U")) and clicks:
            clicks.pop()

        if len(clicks) == 3 and key in (13, 32):
            cv2.destroyWindow(window)
            return {
                "kind": "true_contact_pose_evaluated",
                "stance_leg": stance_leg,
                "points": [list(p) for p in clicks],
                "mediapipe_prediction": pred,
            }, False


def run_annotation_session(
    contact_frames: List[Path],
    records: Dict[str, dict],
    records_path: Path,
    pose,
    min_visibility: float,
    redo: bool,
) -> None:

    def already_done(name: str) -> bool:
        return (
            name in records
            and records[name].get("kind") not in (None, "")
        )

    todo = [
        p
        for p in contact_frames
        if redo or not already_done(p.name)
    ]

    if not todo:
        print("Nothing left to annotate.")
        return

    print(f"\nManual review: {len(todo)} predicted-contact frame(s)")
    print("L/R = true contact + stance leg")
    print("N   = false positive from contact model")
    print("S   = skip")
    print("Q   = quit/save")
    print()

    for i, frame_path in enumerate(todo, start=1):
        print(f"[{i}/{len(todo)}] {frame_path.name}")

        record, quit_now = annotate_contact_frame(
            frame_path,
            pose=pose,
            min_visibility=min_visibility,
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
    contact_frame_names: List[str],
) -> dict:

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
            "visibility": [],
        }
        for role in LEG_STEP_LABELS
    }

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

        gt = rec.get("points", [])
        pred = rec.get("mediapipe_prediction", {}).get("points", {})

        if len(gt) != 3:
            continue

        gt_points = {
            role: (float(gt[i][0]), float(gt[i][1]))
            for i, role in enumerate(LEG_STEP_LABELS)
        }

        # Same normalizer used conceptually in your custom evaluator:
        # stance-leg knee-to-ankle shank length.
        shank_len = _distance(
            gt_points["knee"],
            gt_points["ankle"],
        )

        for role in LEG_STEP_LABELS:
            p = pred.get(role)
            if not p:
                continue

            pred_xy = (float(p["x"]), float(p["y"]))
            err_px = _distance(pred_xy, gt_points[role])

            per_role[role]["errors_px"].append(err_px)
            per_role[role]["visibility"].append(
                float(p.get("visibility", 0.0))
            )

            if shank_len > 1e-6:
                per_role[role]["errors_norm"].append(
                    100.0 * err_px / shank_len
                )

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

    pose_total = pose_success + pose_failed
    pose_success_rate = (
        pose_success / pose_total
        if pose_total > 0
        else None
    )

    role_stats = {}
    all_px = []
    all_norm = []

    for role in LEG_STEP_LABELS:
        px = per_role[role]["errors_px"]
        norm = per_role[role]["errors_norm"]
        vis = per_role[role]["visibility"]

        stats = {"n": len(px)}

        if px:
            stats["mean_px"] = sum(px) / len(px)
            stats["median_px"] = sorted(px)[len(px) // 2]

            all_px.extend(px)

        if norm:
            stats["mean_norm_pct"] = sum(norm) / len(norm)
            stats["median_norm_pct"] = sorted(norm)[len(norm) // 2]
            all_norm.extend(norm)

        if vis:
            stats["mean_visibility"] = sum(vis) / len(vis)

        role_stats[role] = stats

    overall = {"n_points": len(all_px)}

    if all_px:
        overall["mean_px"] = sum(all_px) / len(all_px)

    if all_norm:
        overall["mean_norm_pct"] = sum(all_norm) / len(all_norm)

    return {
        "predicted_contact_frames_available": len(contact_frame_names),
        "manual_frames_skipped": skipped,

        "contact_classifier": {
            "n_predicted_contact_reviewed": contact_reviewed,
            "n_true_contact": true_contact,
            "n_false_positive": contact_false_positive,
            "false_positive_fraction_among_predicted_contacts":
                contact_fp_rate,
            "precision_among_reviewed_predicted_contacts":
                contact_precision,
        },

        "mediapipe_detection": {
            "n_true_contact_frames": pose_total,
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


def _weighted_mean(entries):
    total_n = sum(n for value, n in entries)
    if total_n == 0:
        return None
    return sum(value * n for value, n in entries) / total_n


def combine_reports(per_video_reports):
    total_available = 0
    total_skipped = 0

    total_reviewed = 0
    total_true_contact = 0
    total_fp = 0

    total_pose_success = 0
    total_pose_failed = 0

    leg_counts = {"left": 0, "right": 0}

    role_px = {r: [] for r in LEG_STEP_LABELS}
    role_norm = {r: [] for r in LEG_STEP_LABELS}
    role_vis = {r: [] for r in LEG_STEP_LABELS}

    for _, report in per_video_reports:
        total_available += report.get(
            "predicted_contact_frames_available", 0
        )
        total_skipped += report.get("manual_frames_skipped", 0)

        c = report["contact_classifier"]
        total_reviewed += c["n_predicted_contact_reviewed"]
        total_true_contact += c["n_true_contact"]
        total_fp += c["n_false_positive"]

        md = report["mediapipe_detection"]
        total_pose_success += md["n_pose_success"]
        total_pose_failed += md["n_pose_failed"]

        for leg in ("left", "right"):
            leg_counts[leg] += report["stance_leg_counts"].get(leg, 0)

        for role in LEG_STEP_LABELS:
            s = report["keypoint_accuracy"]["per_role"][role]
            n = s.get("n", 0)

            if n <= 0:
                continue

            if s.get("mean_px") is not None:
                role_px[role].append((s["mean_px"], n))

            if s.get("mean_norm_pct") is not None:
                role_norm[role].append((s["mean_norm_pct"], n))

            if s.get("mean_visibility") is not None:
                role_vis[role].append((s["mean_visibility"], n))

    contact_fp_rate = (
        total_fp / total_reviewed
        if total_reviewed > 0
        else None
    )

    contact_precision = (
        total_true_contact / total_reviewed
        if total_reviewed > 0
        else None
    )

    pose_total = total_pose_success + total_pose_failed
    pose_success_rate = (
        total_pose_success / pose_total
        if pose_total > 0
        else None
    )

    combined_roles = {}
    total_weighted_px = 0.0
    total_px_n = 0

    total_weighted_norm = 0.0
    total_norm_n = 0

    for role in LEG_STEP_LABELS:
        n = sum(count for _, count in role_px[role])

        stats = {"n": n}

        if role_px[role]:
            mean_px = _weighted_mean(role_px[role])
            stats["mean_px"] = mean_px

            total_weighted_px += mean_px * n
            total_px_n += n

        if role_norm[role]:
            mean_norm = _weighted_mean(role_norm[role])
            stats["mean_norm_pct"] = mean_norm

            nn = sum(count for _, count in role_norm[role])
            total_weighted_norm += mean_norm * nn
            total_norm_n += nn

        if role_vis[role]:
            stats["mean_visibility"] = _weighted_mean(role_vis[role])

        combined_roles[role] = stats

    overall = {"n_points": total_px_n}

    if total_px_n:
        overall["mean_px"] = total_weighted_px / total_px_n

    if total_norm_n:
        overall["mean_norm_pct"] = (
            total_weighted_norm / total_norm_n
        )

    return {
        "n_videos": len(per_video_reports),
        "predicted_contact_frames_available": total_available,
        "manual_frames_skipped": total_skipped,

        "contact_classifier": {
            "n_predicted_contact_reviewed": total_reviewed,
            "n_true_contact": total_true_contact,
            "n_false_positive": total_fp,
            "false_positive_fraction_among_predicted_contacts":
                contact_fp_rate,
            "precision_among_reviewed_predicted_contacts":
                contact_precision,
        },

        "mediapipe_detection": {
            "n_true_contact_frames": pose_total,
            "n_pose_success": total_pose_success,
            "n_pose_failed": total_pose_failed,
            "pose_success_rate": pose_success_rate,
        },

        "stance_leg_counts": leg_counts,

        "keypoint_accuracy": {
            "overall": overall,
            "per_role": combined_roles,
        },
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def pct_fraction(value: Optional[float]) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def format_report(report: dict, title: str) -> str:
    lines = []

    lines.append("=" * 72)
    lines.append(title)
    lines.append("=" * 72)

    c = report["contact_classifier"]
    md = report["mediapipe_detection"]
    kp = report["keypoint_accuracy"]
    leg = report["stance_leg_counts"]

    lines.append("")
    lines.append("CONTACT MODEL — SAME PROJECT MODEL")
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
        f"{pct_fraction(c['false_positive_fraction_among_predicted_contacts'])}"
    )
    lines.append(
        f"Precision on reviewed positives:   "
        f"{pct_fraction(c['precision_among_reviewed_predicted_contacts'])}"
    )

    lines.append("")
    lines.append("MEDIAPIPE KEYPOINT DETECTION")
    lines.append(
        f"True-contact frames reviewed:      "
        f"{md['n_true_contact_frames']}"
    )
    lines.append(
        f"MediaPipe detection succeeded:     "
        f"{md['n_pose_success']}"
    )
    lines.append(
        f"MediaPipe detection failed:        "
        f"{md['n_pose_failed']}"
    )
    lines.append(
        f"Detection success rate:            "
        f"{pct_fraction(md['pose_success_rate'])}"
    )

    lines.append("")
    lines.append(
        f"Stance-leg split: left={leg.get('left', 0)}, "
        f"right={leg.get('right', 0)}"
    )

    lines.append("")
    lines.append("STANCE-LEG KEYPOINT ERROR — MEDIAPIPE")
    overall = kp["overall"]

    lines.append(
        f"Total evaluated keypoints:         "
        f"{overall.get('n_points', 0)}"
    )

    if overall.get("mean_px") is not None:
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
        f"{'Keypoint':<12}"
        f"{'n':>7}"
        f"{'mean px':>12}"
        f"{'% shank':>12}"
        f"{'visibility':>14}"
    )
    lines.append("-" * 57)

    for role in LEG_STEP_LABELS:
        s = kp["per_role"][role]

        if s.get("n", 0) == 0:
            lines.append(
                f"{role:<12}{0:>7}{'-':>12}{'-':>12}{'-':>14}"
            )
            continue

        mean_px = (
            f"{s['mean_px']:.2f}"
            if s.get("mean_px") is not None else "-"
        )
        norm = (
            f"{s['mean_norm_pct']:.2f}"
            if s.get("mean_norm_pct") is not None else "-"
        )
        vis = (
            f"{s['mean_visibility']:.3f}"
            if s.get("mean_visibility") is not None else "-"
        )

        lines.append(
            f"{role:<12}"
            f"{s['n']:>7}"
            f"{mean_px:>12}"
            f"{norm:>12}"
            f"{vis:>14}"
        )

    return "\n".join(lines)


def format_combined(combined, per_video_reports):
    parts = [
        format_report(
            combined,
            f"MEDIAPIPE COMBINED EVALUATION — "
            f"{combined['n_videos']} VIDEO(S)",
        ),
        "",
        "",
        "PER-VIDEO RESULTS",
        "",
    ]

    for video_path, report in per_video_reports:
        parts.append(format_report(report, video_path.name))
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Single-video evaluation
# ---------------------------------------------------------------------------

def evaluate_one_video(video_path: Path, args, pose):
    print("\n" + "=" * 72)
    print(f"VIDEO: {video_path.name}")
    print("=" * 72)

    if args.report_only:
        session_dir = find_session_for_video(video_path)

        if session_dir is None:
            print("[skip] No existing session found.")
            return None

        print(f"Using existing session: {session_dir}")

    else:
        print("Running PROJECT CONTACT MODEL ONLY...")
        print(
            "RF-DETR keypoint inference is completely skipped. "
            "MediaPipe is the only keypoint detector used in this run."
        )

        try:
            session_dir = run_contact_only_on_video(video_path)
        except Exception as e:
            print(f"[skip] Contact-only pipeline failed: {e}")
            return None

        print(f"Session: {session_dir}")

    contact_frames = list_contact_frames(session_dir)

    if not contact_frames:
        print(
            "[skip] Could not find predicted-contact frame images.\n"
            "Expected a folder like session/contact_frames/."
        )
        return None

    # Match the original evaluator: evenly sample when capped.
    if (
        args.max_contact_frames_per_video
        and args.max_contact_frames_per_video > 0
        and len(contact_frames) > args.max_contact_frames_per_video
    ):
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

    records_path = session_dir / "mediapipe_eval_records.json"
    records = load_json(records_path)

    if not args.report_only:
        run_annotation_session(
            contact_frames=contact_frames,
            records=records,
            records_path=records_path,
            pose=pose,
            min_visibility=args.min_visibility,
            redo=args.redo,
        )

    report = compute_report(
        records=records,
        contact_frame_names=[p.name for p in contact_frames],
    )

    txt = format_report(report, video_path.name)

    (session_dir / "mediapipe_eval_report.txt").write_text(txt)
    (session_dir / "mediapipe_eval_report.json").write_text(
        json.dumps(report, indent=2)
    )

    print()
    print(txt)

    return video_path, report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
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
            "Frames are sampled evenly. Use 0 for all. Default: 30."
        ),
    )

    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.50,
        help=(
            "Minimum MediaPipe visibility required for knee, ankle, AND heel. "
            "If any required point is below this, the frame counts as a "
            "MediaPipe keypoint-detection failure. Default: 0.50."
        ),
    )

    parser.add_argument(
        "--model-complexity",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help=(
            "MediaPipe Pose model complexity. "
            "2 gives MediaPipe its strongest legacy Pose model. Default: 2."
        ),
    )

    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.30,
        help=(
            "MediaPipe minimum pose detection confidence. Default: 0.30."
        ),
    )

    parser.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "Do not rerun pipeline or annotation UI. "
            "Recompute reports from saved MediaPipe records."
        ),
    )

    parser.add_argument(
        "--redo",
        action="store_true",
        help="Redo MediaPipe manual annotations.",
    )

    return parser.parse_args()


def main():
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

    pose = create_mediapipe_pose(
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
    )

    per_video_reports = []

    try:
        for video_path in videos:
            result = evaluate_one_video(
                video_path,
                args=args,
                pose=pose,
            )

            if result is not None:
                per_video_reports.append(result)
    finally:
        pose.close()

    if not per_video_reports:
        print("\nNo successful evaluations.")
        return 1

    combined = combine_reports(per_video_reports)
    combined_text = format_combined(
        combined,
        per_video_reports,
    )

    print("\n\n" + combined_text)

    out_dir = OUTPUT_ROOT / "batch_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / "mediapipe_eval_combined.txt"
    json_path = out_dir / "mediapipe_eval_combined.json"

    txt_path.write_text(combined_text)
    json_path.write_text(json.dumps(combined, indent=2))

    print("\nSaved:")
    print(f"  {txt_path}" )
    print(f"  {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
