"""End-to-end pipeline: video -> contact filtering -> keypoints -> pronation report."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from app.analysis.analyze_pronation import analyze_pronation
from app.config.settings import (
    CONTACT_CONF_THRESHOLD,
    CONTACT_MODEL_ID,
    FRAME_STRIDE,
    KEYPOINT_MODEL_ID,
    OUTPUT_ROOT,
)
from app.inference.contact_inference import infer_contact
from app.inference.roboflow_client import extract_runner_prediction, infer_image


# Indices of the four keypoints in the Roboflow keypoint model output.
LEFT_KNEE = 0
LEFT_ANKLE = 1
RIGHT_KNEE = 2
RIGHT_ANKLE = 3


def _save_frame(frame: np.ndarray, path: Path) -> None:
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Failed to save image to {path}")


def parse_keypoints(prediction: Dict[str, Any]) -> Optional[List[Tuple[float, float]]]:
    keypoints = prediction.get("keypoints", [])
    if len(keypoints) < 4:
        return None
    return [(float(kp["x"]), float(kp["y"])) for kp in keypoints[:4]]


def compute_pronation_proxy(
    keypoints: List[Tuple[float, float]],
    bbox_width: float,
) -> Dict[str, float]:
    lk_x, _ = keypoints[LEFT_KNEE]
    la_x, _ = keypoints[LEFT_ANKLE]
    rk_x, _ = keypoints[RIGHT_KNEE]
    ra_x, _ = keypoints[RIGHT_ANKLE]

    left_dx = la_x - lk_x
    right_dx = ra_x - rk_x

    norm = max(bbox_width, 1.0)
    return {
        "left_dx": left_dx,
        "right_dx": right_dx,
        "left_dx_norm": left_dx / norm,
        "right_dx_norm": right_dx / norm,
    }


def draw_keypoint_prediction(
    image: np.ndarray,
    prediction: Dict[str, Any],
    contact_label: str,
    contact_conf: float,
    keypoint_conf: float,
) -> np.ndarray:
    vis = image.copy()

    x = int(prediction["x"] - prediction["width"] / 2)
    y = int(prediction["y"] - prediction["height"] / 2)
    w = int(prediction["width"])
    h = int(prediction["height"])

    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

    header = f"contact={contact_label} ({contact_conf:.2f}) | keypoints={keypoint_conf:.2f}"
    cv2.putText(
        vis,
        header,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    keypoints = prediction.get("keypoints", [])
    for i, kp in enumerate(keypoints):
        kx = int(kp["x"])
        ky = int(kp["y"])
        cv2.circle(vis, (kx, ky), 5, (0, 255, 255), -1)
        cv2.putText(
            vis,
            str(i),
            (kx + 6, ky - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    if len(keypoints) >= 4:
        lk = (int(keypoints[LEFT_KNEE]["x"]), int(keypoints[LEFT_KNEE]["y"]))
        la = (int(keypoints[LEFT_ANKLE]["x"]), int(keypoints[LEFT_ANKLE]["y"]))
        rk = (int(keypoints[RIGHT_KNEE]["x"]), int(keypoints[RIGHT_KNEE]["y"]))
        ra = (int(keypoints[RIGHT_ANKLE]["x"]), int(keypoints[RIGHT_ANKLE]["y"]))

        cv2.line(vis, lk, la, (255, 0, 0), 2)
        cv2.line(vis, rk, ra, (0, 0, 255), 2)

        # Vertical reference lines from each knee help visualize lateral offset.
        cv2.line(vis, lk, (lk[0], la[1]), (0, 255, 0), 1)
        cv2.line(vis, rk, (rk[0], ra[1]), (0, 255, 0), 1)

    return vis


def run_pipeline(
    video_path: Union[str, Path],
    output_root: Union[str, Path] = OUTPUT_ROOT,
    frame_stride: int = FRAME_STRIDE,
    contact_model_id: str = CONTACT_MODEL_ID,
    keypoint_model_id: str = KEYPOINT_MODEL_ID,
    contact_conf_threshold: float = CONTACT_CONF_THRESHOLD,
) -> Optional[Dict[str, Any]]:
    """Process a rear-view running video and return the pronation analysis dict.

    Returns ``None`` if no usable contact frames were found.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_root = Path(output_root)
    session_dir = output_root / video_path.stem

    sampled_frames_dir = session_dir / "sampled_frames"
    contact_frames_dir = session_dir / "contact_frames"
    rendered_dir = session_dir / "rendered_predictions"

    csv_path = session_dir / "roboflow_predictions.csv"
    analysis_csv_path = session_dir / "pronation_analysis.csv"

    for directory in (session_dir, sampled_frames_dir, contact_frames_dir, rendered_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    rows: List[List[Any]] = []
    frame_idx = 0
    sampled_count = 0
    contact_count = 0
    analyzed_count = 0

    print(f"Processing video: {video_path}")
    print(f"Contact model: {contact_model_id}")
    print(f"Keypoint model: {keypoint_model_id}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            sampled_count += 1
            frame_name = f"frame_{frame_idx:06d}.jpg"
            sampled_frame_path = sampled_frames_dir / frame_name
            _save_frame(frame, sampled_frame_path)

            # ----------------------------
            # Phase 1: Contact filtering
            # ----------------------------
            try:
                contact_label, contact_conf, _ = infer_contact(
                    str(sampled_frame_path),
                    model_id=contact_model_id,
                )
            except Exception as e:
                print(f"[WARN] Contact inference failed on frame {frame_idx}: {e}")
                rows.append([
                    frame_idx, frame_name, 0, "", "", "", "", "", "", "", "", "", "", ""
                ])
                frame_idx += 1
                continue

            if contact_label != "contact" or contact_conf < contact_conf_threshold:
                rows.append([
                    frame_idx, frame_name, 0, contact_label, contact_conf,
                    "", "", "", "", "", "", "", "", ""
                ])
                frame_idx += 1
                continue

            contact_count += 1
            contact_frame_path = contact_frames_dir / frame_name
            _save_frame(frame, contact_frame_path)

            # ----------------------------
            # Phase 2: Keypoint inference
            # ----------------------------
            try:
                result = infer_image(str(contact_frame_path), model_id=keypoint_model_id)
                pred = extract_runner_prediction(result)
            except Exception as e:
                print(f"[WARN] Keypoint inference failed on frame {frame_idx}: {e}")
                rows.append([
                    frame_idx, frame_name, 0, contact_label, contact_conf,
                    "", "", "", "", "", "", "", "", ""
                ])
                frame_idx += 1
                continue

            if pred is None:
                rows.append([
                    frame_idx, frame_name, 0, contact_label, contact_conf,
                    "", "", "", "", "", "", "", "", ""
                ])
                frame_idx += 1
                continue

            keypoints = parse_keypoints(pred)
            if keypoints is None:
                rows.append([
                    frame_idx, frame_name, 0, contact_label, contact_conf,
                    "", "", "", "", "", "", "", "", ""
                ])
                frame_idx += 1
                continue

            analyzed_count += 1

            (lk_x, lk_y) = keypoints[LEFT_KNEE]
            (la_x, la_y) = keypoints[LEFT_ANKLE]
            (rk_x, rk_y) = keypoints[RIGHT_KNEE]
            (ra_x, ra_y) = keypoints[RIGHT_ANKLE]

            bbox_width = float(pred.get("width", 1.0))
            keypoint_conf = float(pred.get("confidence", 0.0))
            metrics = compute_pronation_proxy(keypoints, bbox_width)

            rendered = draw_keypoint_prediction(
                frame,
                pred,
                contact_label=contact_label,
                contact_conf=contact_conf,
                keypoint_conf=keypoint_conf,
            )
            _save_frame(rendered, rendered_dir / frame_name)

            rows.append([
                frame_idx,
                frame_name,
                1,
                contact_label,
                contact_conf,
                lk_x, lk_y,
                la_x, la_y,
                rk_x, rk_y,
                ra_x, ra_y,
                metrics["left_dx_norm"],
                metrics["right_dx_norm"],
            ])

            frame_idx += 1
    finally:
        cap.release()

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_idx",
            "file",
            "detected",
            "contact_label",
            "contact_conf",
            "lk_x", "lk_y",
            "la_x", "la_y",
            "rk_x", "rk_y",
            "ra_x", "ra_y",
            "left_dx_norm",
            "right_dx_norm",
        ])
        writer.writerows(rows)

    print(f"Sampled frames: {sampled_count}")
    print(f"Contact frames kept: {contact_count}")
    print(f"Frames with keypoints analyzed: {analyzed_count}")
    print(f"Rendered frames: {rendered_dir}")
    print(f"CSV: {csv_path}")

    if analyzed_count == 0:
        print("No contact frames with valid keypoints were found. Skipping pronation analysis.")
        return None

    print("\n--- Running pronation analysis ---")
    result = analyze_pronation(
        csv_path=str(csv_path),
        output_csv=str(analysis_csv_path),
        smoothing_window=5,
        neutral_thresh=0.015,
        over_thresh=0.04,
    )
    print(f"Analysis CSV: {analysis_csv_path}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the gait analysis pipeline on a video.")
    parser.add_argument("video", type=str, help="Path to the input video file.")
    args = parser.parse_args()

    run_pipeline(video_path=args.video)
