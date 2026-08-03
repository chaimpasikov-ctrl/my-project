"""Synthetic end-to-end test for the new 6-keypoint pipeline.

Generates a fake roboflow_predictions.csv that mimics a runner with a
moderate pronation tendency on both legs, runs ``analyze_pronation`` on
it, and prints the key outputs we want to confirm.
"""

import csv
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path


def main() -> int:
    # Add repo root to sys.path so `app` package is importable.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from app.pipelines.main_pipeline import (
        compute_eversion_angle_deg,
        compute_kinematic_metrics,
        _self_test_eversion_sign,
    )
    from app.analysis.analyze_pronation import analyze_pronation

    print("--- step 1: re-run sign-convention self-test ---")
    _self_test_eversion_sign()
    print()

    tmpdir = Path(tempfile.mkdtemp(prefix="gait_e2e_"))
    try:
        csv_path = tmpdir / "roboflow_predictions.csv"
        meta_path = tmpdir / "session_meta.json"
        analysis_csv = tmpdir / "pronation_analysis.csv"

        fps = 30.0
        with meta_path.open("w") as f:
            json.dump({"fps": fps, "frame_stride": 2, "video_stem": "synthetic"}, f)

        # Geometry:
        #   Knee directly above ankle. Heel below ankle, displaced laterally.
        #   With our sign convention (image x grows toward runner's right):
        #     right leg pronation => heel.x > ankle.x
        #     left  leg pronation => heel.x < ankle.x
        # We simulate ~7-8 deg eversion on both legs (moderate pronation).
        knee_y, ankle_y, heel_y = 100.0, 200.0, 250.0
        l_knee_x, l_ankle_x = 100.0, 100.0
        r_knee_x, r_ankle_x = 200.0, 200.0
        l_heel_lateral = -7.0   # left heel to runner's left
        r_heel_lateral = +7.0   # right heel to runner's right

        rows = []
        # Alternate stance leg per frame to simulate left/right contact events.
        # The pipeline normally has multiple consecutive frames per stance, so
        # we group them in bursts of 3.
        FRAME_STRIDE = 2
        n_strides = 60  # 60 strides => ~30 left + ~30 right stance events
        burst = 3
        frame_idx = 0
        for s in range(n_strides):
            leg = "right" if s % 2 == 0 else "left"
            # The "stance leg" heuristic in analyze_pronation picks the leg
            # whose ankle is LOWER on screen (la_y > ra_y => left). To force
            # a specific stance leg, we offset that ankle by +5 px.
            for k in range(burst):
                la_y = ankle_y
                ra_y = ankle_y
                if leg == "left":
                    la_y += 5.0
                else:
                    ra_y += 5.0
                kps = [
                    (l_knee_x, knee_y),
                    (l_ankle_x, la_y),
                    (r_knee_x, knee_y),
                    (r_ankle_x, ra_y),
                    (l_ankle_x + l_heel_lateral, heel_y),
                    (r_ankle_x + r_heel_lateral, heel_y),
                ]
                m = compute_kinematic_metrics(kps)
                rows.append({
                    "frame_idx": frame_idx,
                    "file": f"frame_{frame_idx:06d}.jpg",
                    "detected": 1,
                    "contact_label": "contact",
                    "contact_conf": 0.95,
                    "lk_x": kps[0][0], "lk_y": kps[0][1],
                    "la_x": kps[1][0], "la_y": kps[1][1],
                    "rk_x": kps[2][0], "rk_y": kps[2][1],
                    "ra_x": kps[3][0], "ra_y": kps[3][1],
                    "lh_x": kps[4][0], "lh_y": kps[4][1],
                    "rh_x": kps[5][0], "rh_y": kps[5][1],
                    "left_dx_px": m["left_dx_px"],
                    "right_dx_px": m["right_dx_px"],
                    "left_shank_px": m["left_shank_px"],
                    "right_shank_px": m["right_shank_px"],
                    "left_eversion_deg": m["left_eversion_deg"],
                    "right_eversion_deg": m["right_eversion_deg"],
                })
                frame_idx += FRAME_STRIDE

        cols = [
            "frame_idx", "file", "detected", "contact_label", "contact_conf",
            "lk_x", "lk_y", "la_x", "la_y", "rk_x", "rk_y", "ra_x", "ra_y",
            "lh_x", "lh_y", "rh_x", "rh_y",
            "left_dx_px", "right_dx_px", "left_shank_px", "right_shank_px",
            "left_eversion_deg", "right_eversion_deg",
        ]
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow([r[c] for c in cols])

        print("--- step 2: synthetic CSV header ---")
        print(",".join(cols))
        print()
        print("--- step 3: one sample data row ---")
        sample = rows[0]
        print(",".join(f"{sample[c]}" for c in cols))
        print()

        print("--- step 4: run analyze_pronation on the synthetic CSV ---")
        result = analyze_pronation(
            csv_path=str(csv_path), output_csv=str(analysis_csv)
        )

        print()
        print("--- step 5: result summary ---")
        print(json.dumps({
            "frames_analyzed": result["frames_analyzed"],
            "left_stance_frames": result["left_stance_frames"],
            "right_stance_frames": result["right_stance_frames"],
            "overall": result["overall"],
            "left_stance_only": result["left_stance_only"],
            "right_stance_only": result["right_stance_only"],
            "free_metrics": result["free_metrics"],
            "quality_issues": result["quality_issues"],
            "metric_definition": result["metric_definition"],
        }, indent=2))

        # ---------------------------------------------------------------
        # Now build a second synthetic CSV that injects extreme outliers
        # (heel ~80 deg away from the shank) to confirm the new QC gate
        # fires on the 30-deg threshold.
        # ---------------------------------------------------------------
        print()
        print("--- step 6: QC test with injected 80-deg outliers ---")
        out_csv = tmpdir / "with_outliers.csv"
        out_analysis = tmpdir / "with_outliers_analysis.csv"
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for i, r in enumerate(rows):
                if i in (5, 17, 29):  # spike a few frames
                    r = dict(r)
                    # Push heels far medial/lateral to force big angles.
                    r["lh_x"] = r["la_x"] - 200.0
                    r["rh_x"] = r["ra_x"] + 200.0
                    m = compute_kinematic_metrics([
                        (r["lk_x"], r["lk_y"]),
                        (r["la_x"], r["la_y"]),
                        (r["rk_x"], r["rk_y"]),
                        (r["ra_x"], r["ra_y"]),
                        (r["lh_x"], r["lh_y"]),
                        (r["rh_x"], r["rh_y"]),
                    ])
                    r["left_eversion_deg"] = m["left_eversion_deg"]
                    r["right_eversion_deg"] = m["right_eversion_deg"]
                w.writerow([r[c] for c in cols])
        # session_meta.json already lives in this tmpdir from step 2.
        result_qc = analyze_pronation(
            csv_path=str(out_csv), output_csv=str(out_analysis)
        )
        print("max |eversion| (deg):",
              max(abs(result_qc["overall"]["min_deg"]),
                  abs(result_qc["overall"]["max_deg"])))
        print("quality_issues:", result_qc["quality_issues"])
        assert "extreme_outliers_present" in result_qc["quality_issues"], (
            "QC gate did NOT fire on injected 80-deg outliers"
        )
        print("QC gate fired correctly on > 30 deg outliers.")

        # ---------------------------------------------------------------
        # Confirm the legacy-CSV error path.
        # ---------------------------------------------------------------
        print()
        print("--- step 7: legacy CSV error path ---")
        legacy_csv = tmpdir / "legacy.csv"
        legacy_cols = [
            "frame_idx", "file", "detected", "contact_label", "contact_conf",
            "lk_x", "lk_y", "la_x", "la_y", "rk_x", "rk_y", "ra_x", "ra_y",
            "left_dx_px", "right_dx_px", "left_shank_px", "right_shank_px",
        ]
        with legacy_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(legacy_cols)
            w.writerow([0, "f.jpg", 1, "contact", 0.9] + [0.0] * 8 + [0.0] * 4)
        try:
            analyze_pronation(csv_path=str(legacy_csv), output_csv=str(tmpdir / "legacy_out.csv"))
        except ValueError as e:
            print(f"Legacy CSV correctly rejected with: {e}")
        else:
            raise RuntimeError("Legacy CSV should have been rejected!")

        return 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
