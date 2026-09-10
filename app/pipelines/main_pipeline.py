"""End-to-end pipeline: video -> contact filtering -> keypoints -> pronation report."""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from app.analysis.analyze_pronation import analyze_pronation
from app.config.settings import (
    CONTACT_CONF_THRESHOLD,
    CONTACT_WORKFLOW_ID,
    FRAME_STRIDE,
    KEYPOINT_WORKFLOW_ID,
    OUTPUT_ROOT,
    ROBOFLOW_WORKSPACE_NAME,
)
from app.inference.contact_inference import infer_contact
from app.inference.roboflow_client import extract_runner_prediction, infer_workflow


# Indices of the six keypoints in the rf-detr keypoint workflow output.
LEFT_KNEE = 0
LEFT_ANKLE = 1
RIGHT_KNEE = 2
RIGHT_ANKLE = 3
LEFT_HEEL = 4
RIGHT_HEEL = 5


def _save_frame(frame: np.ndarray, path: Path) -> None:
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Failed to save image to {path}")


def parse_keypoints(prediction: Dict[str, Any]) -> Optional[List[Tuple[float, float]]]:
    keypoints = prediction.get("keypoints", [])
    if len(keypoints) < 6:
        return None
    return [(float(kp["x"]), float(kp["y"])) for kp in keypoints[:6]]


def compute_eversion_angle_deg(
    knee: Tuple[float, float],
    ankle: Tuple[float, float],
    heel: Tuple[float, float],
    side: str,
) -> float:
    """
    Rearfoot eversion angle in degrees, signed per leg.

    Geometry (rear view, runner facing away from camera, image x grows to the
    runner's left):
        - Shank vector points from knee DOWN to ankle.
        - Calcaneus vector points from ankle DOWN to heel.
        - Eversion angle is the signed angle between these two vectors in the
          image (frontal) plane.

    Sign convention (positive = pronation, negative = supination):
        - Right leg: heel displaced to the runner's right (image x > ankle x)
          relative to the shank line is pronation/eversion => positive.
        - Left  leg: heel displaced to the runner's left  (image x < ankle x)
          relative to the shank line is pronation/eversion => positive.

    Args:
        knee, ankle, heel: (x, y) tuples in pixels (image coordinates).
        side: "left" or "right".

    Returns:
        Angle in degrees, signed by the convention above.
        Returns 0.0 if any vector has degenerate length.
    """
    kx, ky = knee
    ax, ay = ankle
    hx, hy = heel

    # Shank vector: knee -> ankle (pointing downward in image)
    sx = ax - kx
    sy = ay - ky
    # Calcaneus vector: ankle -> heel
    cx = hx - ax
    cy = hy - ay

    s_len = math.hypot(sx, sy)
    c_len = math.hypot(cx, cy)
    if s_len < 1e-6 or c_len < 1e-6:
        return 0.0

    # Signed angle from shank to calcaneus, via 2D cross + dot.
    # cross > 0 means calcaneus rotates counter-clockwise from shank in image coords.
    cross = sx * cy - sy * cx
    dot = sx * cx + sy * cy
    angle_rad = math.atan2(cross, dot)
    angle_deg = math.degrees(angle_rad)

    # In image coordinates y grows DOWN. For a runner viewed from behind:
    #   - On the RIGHT leg, "pronation" = heel laterally displaced =
    #     heel is to the runner's right of the shank line.
    #     With image x growing to the runner's left, that means
    #     heel.x < (extension of shank).x, which gives a NEGATIVE cross
    #     under our 2D convention. So we negate to make pronation positive.
    #   - On the LEFT leg, "pronation" = heel laterally displaced =
    #     heel is to the runner's left of the shank line, which gives a
    #     POSITIVE cross. Sign is already correct, no flip needed.
    if side == "right":
        angle_deg = -angle_deg

    return angle_deg


def compute_kinematic_metrics(
    keypoints: List[Tuple[float, float]],
) -> Dict[str, float]:
    """Per-frame raw geometry from the 6-keypoint model.

    Returns horizontal ankle-knee offsets, shank lengths (used for an
    anatomical pixel scale in ``analyze_pronation``), heel pixel positions,
    and per-leg signed rearfoot eversion angle in degrees.
    """
    lk_x, lk_y = keypoints[LEFT_KNEE]
    la_x, la_y = keypoints[LEFT_ANKLE]
    rk_x, rk_y = keypoints[RIGHT_KNEE]
    ra_x, ra_y = keypoints[RIGHT_ANKLE]
    lh_x, lh_y = keypoints[LEFT_HEEL]
    rh_x, rh_y = keypoints[RIGHT_HEEL]

    left_eversion_deg = compute_eversion_angle_deg(
        knee=(lk_x, lk_y), ankle=(la_x, la_y), heel=(lh_x, lh_y), side="left"
    )
    right_eversion_deg = compute_eversion_angle_deg(
        knee=(rk_x, rk_y), ankle=(ra_x, ra_y), heel=(rh_x, rh_y), side="right"
    )

    left_shank_px = math.hypot(la_x - lk_x, la_y - lk_y)
    right_shank_px = math.hypot(ra_x - rk_x, ra_y - rk_y)

    return {
        "left_dx_px": la_x - lk_x,
        "right_dx_px": ra_x - rk_x,
        "left_shank_px": left_shank_px,
        "right_shank_px": right_shank_px,
        "left_heel_x": lh_x,
        "left_heel_y": lh_y,
        "right_heel_x": rh_x,
        "right_heel_y": rh_y,
        "left_eversion_deg": left_eversion_deg,
        "right_eversion_deg": right_eversion_deg,
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

        # Shank vectors (knee -> ankle).
        cv2.line(vis, lk, la, (255, 0, 0), 2)
        cv2.line(vis, rk, ra, (0, 0, 255), 2)

        # Vertical reference lines from each knee help visualize lateral offset.
        cv2.line(vis, lk, (lk[0], la[1]), (0, 255, 0), 1)
        cv2.line(vis, rk, (rk[0], ra[1]), (0, 255, 0), 1)

    if len(keypoints) >= 6:
        la = (int(keypoints[LEFT_ANKLE]["x"]), int(keypoints[LEFT_ANKLE]["y"]))
        ra = (int(keypoints[RIGHT_ANKLE]["x"]), int(keypoints[RIGHT_ANKLE]["y"]))
        lh = (int(keypoints[LEFT_HEEL]["x"]), int(keypoints[LEFT_HEEL]["y"]))
        rh = (int(keypoints[RIGHT_HEEL]["x"]), int(keypoints[RIGHT_HEEL]["y"]))

        # Calcaneus vectors (ankle -> heel), drawn in magenta so they are
        # visually distinct from the blue/red shank lines above.
        cv2.line(vis, la, lh, (255, 0, 255), 2)
        cv2.line(vis, ra, rh, (255, 0, 255), 2)

    return vis


def run_pipeline(
    video_path: Union[str, Path],
    output_root: Union[str, Path] = OUTPUT_ROOT,
    frame_stride: int = FRAME_STRIDE,
    workspace_name: str = ROBOFLOW_WORKSPACE_NAME,
    contact_workflow_id: str = CONTACT_WORKFLOW_ID,
    keypoint_workflow_id: str = KEYPOINT_WORKFLOW_ID,
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

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    rows: List[List[Any]] = []
    frame_idx = 0
    sampled_count = 0
    contact_count = 0
    analyzed_count = 0

    print(f"Processing video: {video_path} (fps={fps:.2f}, frames={total_frames})")
    print(f"Contact workflow: {workspace_name}/{contact_workflow_id}")
    print(f"Keypoint workflow: {workspace_name}/{keypoint_workflow_id}")

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
                    workspace_name=workspace_name,
                    workflow_id=contact_workflow_id,
                )
            except Exception as e:
                print(f"[WARN] Contact inference failed on frame {frame_idx}: {e}")
                # 23 columns total (5 fixed + 18 empties), see header below.
                rows.append([
                    frame_idx, frame_name, 0, "", "",
                    "", "", "", "", "", "", "", "",
                    "", "", "", "",
                    "", "", "", "",
                    "", "",
                ])
                frame_idx += 1
                continue

            def empty_row():
                # Trailing empty fields match the CSV column count below:
                #   lk_x, lk_y, la_x, la_y, rk_x, rk_y, ra_x, ra_y,        (8)
                #   lh_x, lh_y, rh_x, rh_y,                                (4)
                #   left_dx_px, right_dx_px, left_shank_px, right_shank_px,(4)
                #   left_eversion_deg, right_eversion_deg                  (2)
                # = 18 placeholders.
                return [
                    frame_idx, frame_name, 0, contact_label, contact_conf,
                    "", "", "", "", "", "", "", "",
                    "", "", "", "",
                    "", "", "", "",
                    "", "",
                ]

            if contact_label != "contact" or contact_conf < contact_conf_threshold:
                rows.append(empty_row())
                frame_idx += 1
                continue

            contact_count += 1
            contact_frame_path = contact_frames_dir / frame_name
            _save_frame(frame, contact_frame_path)

            # ----------------------------
            # Phase 2: Keypoint inference
            # ----------------------------
            try:
                result = infer_workflow(
                    str(contact_frame_path),
                    workspace_name=workspace_name,
                    workflow_id=keypoint_workflow_id,
                )
                pred = extract_runner_prediction(result)
            except Exception as e:
                print(f"[WARN] Keypoint inference failed on frame {frame_idx}: {e}")
                rows.append(empty_row())
                frame_idx += 1
                continue

            if pred is None:
                rows.append(empty_row())
                frame_idx += 1
                continue

            keypoints = parse_keypoints(pred)
            if keypoints is None:
                rows.append(empty_row())
                frame_idx += 1
                continue

            analyzed_count += 1

            (lk_x, lk_y) = keypoints[LEFT_KNEE]
            (la_x, la_y) = keypoints[LEFT_ANKLE]
            (rk_x, rk_y) = keypoints[RIGHT_KNEE]
            (ra_x, ra_y) = keypoints[RIGHT_ANKLE]
            (lh_x, lh_y) = keypoints[LEFT_HEEL]
            (rh_x, rh_y) = keypoints[RIGHT_HEEL]

            keypoint_conf = float(pred.get("confidence", 0.0))
            metrics = compute_kinematic_metrics(keypoints)

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
                lh_x, lh_y,
                rh_x, rh_y,
                metrics["left_dx_px"],
                metrics["right_dx_px"],
                metrics["left_shank_px"],
                metrics["right_shank_px"],
                metrics["left_eversion_deg"],
                metrics["right_eversion_deg"],
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
            "lh_x", "lh_y",
            "rh_x", "rh_y",
            "left_dx_px",
            "right_dx_px",
            "left_shank_px",
            "right_shank_px",
            "left_eversion_deg",
            "right_eversion_deg",
        ])
        writer.writerows(rows)

    # Sidecar metadata so analyze_pronation can compute time-based metrics
    # (cadence, ground-contact time) without re-opening the video.
    meta_path = session_dir / "session_meta.json"
    with meta_path.open("w") as f:
        json.dump(
            {
                "fps": fps,
                "total_frames": total_frames,
                "frame_stride": int(frame_stride),
                "sampled_frames": sampled_count,
                "contact_frames": contact_count,
                "analyzed_frames": analyzed_count,
                "video_stem": video_path.stem,
            },
            f,
            indent=2,
        )

    print(f"Sampled frames: {sampled_count}")
    print(f"Contact frames kept: {contact_count}")
    print(f"Frames with keypoints analyzed: {analyzed_count}")
    print(f"Rendered frames: {rendered_dir}")
    print(f"CSV: {csv_path}")
    print(f"Session meta: {meta_path}")

    if analyzed_count == 0:
        print("No contact frames with valid keypoints were found. Skipping analysis.")
        return None

    print("\n--- Running rearfoot eversion angle analysis ---")
    result = analyze_pronation(
        csv_path=str(csv_path),
        output_csv=str(analysis_csv_path),
    )
    print(f"Analysis CSV: {analysis_csv_path}")
    return result


def _self_test_eversion_sign() -> None:
    """Sanity-check the sign convention of ``compute_eversion_angle_deg``.

    Image coordinates: x grows RIGHT, y grows DOWN (standard OpenCV/Roboflow).
    For a rear-view camera placed directly behind the runner, the runner's
    right side appears on the RIGHT side of the image, so image x grows in
    the direction of the runner's RIGHT. (This is the empirically validated
    convention; the docstring at the top of ``compute_eversion_angle_deg``
    talks about image x growing to the runner's "left" but the final
    sign-flip implemented there is correct only under the standard
    rear-view convention used here.)

    Expected sign convention (positive = pronation):
        - Aligned heel (directly below ankle) => ~0 degrees on both legs.
        - Right leg, heel displaced to runner's RIGHT (heel.x > ankle.x)
            => POSITIVE angle.
        - Left  leg, heel displaced to runner's LEFT  (heel.x < ankle.x)
            => POSITIVE angle.
        - Same physical eversion magnitude on both legs => same sign and
          approximately equal magnitude.
        - Inversion (heel toward the midline) => NEGATIVE on both legs.
    """
    tol = 1e-6
    knee = (100.0, 100.0)
    ankle = (100.0, 200.0)

    # 1. Perfectly aligned (heel directly below ankle).
    aligned_heel = (100.0, 250.0)
    left_aligned = compute_eversion_angle_deg(knee, ankle, aligned_heel, side="left")
    right_aligned = compute_eversion_angle_deg(knee, ankle, aligned_heel, side="right")
    assert abs(left_aligned) < tol, f"aligned left expected ~0, got {left_aligned}"
    assert abs(right_aligned) < tol, f"aligned right expected ~0, got {right_aligned}"

    # 2. Right leg pronation: heel displaced to runner's RIGHT
    #    (image x > ankle x because image x grows toward runner's right).
    right_pronation_heel = (120.0, 250.0)
    right_pronation = compute_eversion_angle_deg(
        knee, ankle, right_pronation_heel, side="right"
    )
    assert right_pronation > 0.0, (
        f"right pronation should be POSITIVE, got {right_pronation}"
    )

    # 3. Left leg pronation: heel displaced to runner's LEFT
    #    (image x < ankle x).
    left_pronation_heel = (80.0, 250.0)
    left_pronation = compute_eversion_angle_deg(
        knee, ankle, left_pronation_heel, side="left"
    )
    assert left_pronation > 0.0, (
        f"left pronation should be POSITIVE, got {left_pronation}"
    )

    # 4. Same physical eversion magnitude on both legs => same sign and
    #    approximately equal magnitude. 20 px lateral on a 50 px calcaneus
    #    works out to the same geometry mirrored.
    assert abs(left_pronation - right_pronation) < 1e-6, (
        f"left/right pronation magnitudes should match: "
        f"left={left_pronation}, right={right_pronation}"
    )

    # 5. Inversion / supination should be negative on both legs.
    right_supination_heel = (80.0, 250.0)  # heel toward midline
    right_supination = compute_eversion_angle_deg(
        knee, ankle, right_supination_heel, side="right"
    )
    assert right_supination < 0.0, (
        f"right supination should be NEGATIVE, got {right_supination}"
    )
    left_supination_heel = (120.0, 250.0)  # heel toward midline
    left_supination = compute_eversion_angle_deg(
        knee, ankle, left_supination_heel, side="left"
    )
    assert left_supination < 0.0, (
        f"left supination should be NEGATIVE, got {left_supination}"
    )

    # 6. Degenerate vectors return 0.
    degen = compute_eversion_angle_deg((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), side="left")
    assert degen == 0.0, f"degenerate inputs should return 0, got {degen}"

    print("[eversion sign self-test] PASS")
    print(f"  aligned   L/R : {left_aligned:+.3f}, {right_aligned:+.3f} deg")
    print(f"  pronation L/R : {left_pronation:+.3f}, {right_pronation:+.3f} deg")
    print(f"  supination L/R: {left_supination:+.3f}, {right_supination:+.3f} deg")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the gait analysis pipeline on a video.")
    parser.add_argument("video", type=str, nargs="?", help="Path to the input video file.")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the eversion-angle sign-convention self-test and exit.",
    )
    args = parser.parse_args()

    if args.self_test or not args.video:
        _self_test_eversion_sign()
        if not args.video:
            raise SystemExit(0)

    run_pipeline(video_path=args.video)
