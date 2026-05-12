"""Streamlit front-end for the running gait analysis pipeline."""

import tempfile
from pathlib import Path

import streamlit as st

from app.analysis.shoe_recommender import recommend_shoes
from app.config.settings import OUTPUT_ROOT
from app.pipelines.main_pipeline import run_pipeline
from app.reporting.pronation_report import save_demo_report


st.set_page_config(page_title="Running Gait Analysis", layout="centered")

st.title("Running Gait Analysis & Shoe Recommendation")
st.write(
    "Upload a rear-view running video, answer a few questions, "
    "and get a personalized shoe recommendation."
)

# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
# `analysis` holds the most recent result so it survives the reruns Streamlit
# triggers on every widget interaction (notably the download button).
# `uploader_key` is bumped on "Start Over" to force-reset the file uploader.
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
uploaded_video = st.file_uploader(
    "Upload running video",
    type=["mp4", "mov", "avi"],
    key=f"video_uploader_{st.session_state.uploader_key}",
)

st.subheader("Runner Profile")
surface = st.selectbox("Running surface", ["road", "treadmill", "trail"])
distance = st.selectbox(
    "Weekly running distance",
    ["0-5 km", "5-10 km", "10-20 km", "20+ km"],
)
comfort = st.selectbox("Preferred feel", ["soft", "balanced", "firm"])

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
            category, shoes = recommend_shoes(pronation_label, surface=surface)

            # Regenerate the PDF so the downloadable report includes the
            # shoe recommendation alongside the pronation analysis.
            session_dir = OUTPUT_ROOT / video_path.stem
            report_base = session_dir / "pronation_analysis_demo_report"
            try:
                save_demo_report(
                    analysis_result,
                    str(report_base),
                    shoe_recommendation={
                        "category": category,
                        "surface": surface,
                        "shoes": shoes,
                    },
                )
            except Exception as e:
                st.warning(f"Could not embed shoe info in the PDF: {e}")

            pdf_path = report_base.with_suffix(".pdf")
            st.session_state.analysis = {
                "pronation_label": pronation_label,
                "pdf_path": str(pdf_path) if pdf_path.is_file() else None,
                "pdf_name": pdf_path.name,
                "category": category,
                "surface": surface,
                "shoes": shoes,
            }

# -----------------------------------------------------------------------------
# Render results from session state so they survive download-button reruns
# -----------------------------------------------------------------------------
if st.session_state.analysis is not None:
    result = st.session_state.analysis

    st.success("Analysis complete!")
    st.write(f"Detected pronation pattern: **{result['pronation_label']}**")

    if result.get("pdf_path"):
        st.subheader("Download Report")
        with open(result["pdf_path"], "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=result["pdf_name"],
            mime="application/pdf",
        )
    else:
        st.warning("No PDF report was generated.")

    st.subheader("Shoe Recommendation")
    st.write(f"Recommended category: **{result['category']}**")

    if result["shoes"]:
        st.write("Suggested models:")
        for shoe in result["shoes"]:
            st.write(f"- {shoe['brand']} {shoe['name']}")
    else:
        st.write("No matching shoes found.")

    st.divider()
    if st.button("Start Over"):
        st.session_state.analysis = None
        st.session_state.uploader_key += 1
        st.rerun()
