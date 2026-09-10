"""Streamlit front-end for the running gait analysis pipeline."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.config.settings import OUTPUT_ROOT
from app.pipelines.main_pipeline import run_pipeline
from app.reporting.pronation_report import pick_rendered_images, save_demo_report


st.set_page_config(page_title="Running Gait Analysis", layout="centered")

st.title("Running Gait Analysis")
st.write("Upload a rear-view treadmill running video to get your gait analysis.")
st.caption(
    "The gait metric is the rearfoot eversion angle (pronation tendency), "
    "measured from a single rear-view camera. Single-camera 2D screening, "
    "not a clinical diagnosis."
)

# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# -----------------------------------------------------------------------------
# Video upload
# -----------------------------------------------------------------------------
uploaded_video = st.file_uploader(
    "Upload running video",
    type=["mp4", "mov", "avi"],
    key=f"video_uploader_{st.session_state.uploader_key}",
)

# -----------------------------------------------------------------------------
# Run analysis
# -----------------------------------------------------------------------------
if st.button("Run Analysis"):
    if uploaded_video is None:
        st.error("Please upload a video first.")
    else:
        suffix = Path(uploaded_video.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_video.read())
            video_path = Path(tmp.name)

        with st.spinner("Running analysis... this can take a minute."):
            try:
                analysis_result = run_pipeline(video_path=video_path)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                analysis_result = None

        if analysis_result is None:
            st.session_state.analysis = None
            st.warning(
                "No usable contact frames were detected. "
                "Try a clearer rear-view video."
            )
        else:
            pronation_label = analysis_result["overall"]["classification"]

            session_dir = OUTPUT_ROOT / video_path.stem
            report_base = session_dir / "pronation_analysis_demo_report"
            try:
                save_demo_report(analysis_result, str(report_base))
            except Exception as e:
                st.warning(f"Could not save report: {e}")

            image_paths = pick_rendered_images(str(report_base), n=4)
            st.session_state.analysis = {
                "pronation_label": pronation_label,
                "overall": analysis_result["overall"],
                "metric_definition": analysis_result.get("metric_definition", ""),
                "image_paths": image_paths,
            }

# -----------------------------------------------------------------------------
# Render results
# -----------------------------------------------------------------------------
if st.session_state.analysis is not None:
    result = st.session_state.analysis

    st.success("Analysis complete!")
    st.write(f"Detected alignment pattern: **{result['pronation_label']}**")
    if result.get("metric_definition"):
        st.caption(result["metric_definition"])

    # --- Pronation analysis ---
    overall = result["overall"]

    st.subheader("Rearfoot eversion angle (pronation tendency)")
    st.caption(
        "Positive = pronation (eversion). Negative = supination (inversion). "
        "Median across detected stance events."
    )
    overall_med = overall.get("median_deg")
    st.metric(
        "Overall (median midstance)",
        f"{overall_med:+.1f} deg" if overall_med is not None else "N/A",
    )

    # --- Rendered frames ---
    image_paths = result.get("image_paths") or []
    if image_paths:
        st.subheader("Example Analyzed Frames")
        cols = st.columns(len(image_paths))
        for col, path in zip(cols, image_paths):
            col.image(path, width=150)

    st.divider()
    if st.button("Start Over"):
        st.session_state.analysis = None
        st.session_state.uploader_key += 1
        st.rerun()
