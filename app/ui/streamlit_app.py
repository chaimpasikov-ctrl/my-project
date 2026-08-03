"""Streamlit front-end for the running gait analysis pipeline."""

import json
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
    "Upload a rear-view treadmill running video, answer a few questions, "
    "and get a personalized shoe recommendation."
)
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
# Runner profile (questionnaire)
# -----------------------------------------------------------------------------
# These inputs do not yet drive the analysis numerically (the multi-factor
# shoe scorer is a "next" task). They ARE persisted into the report so we
# can iterate on the recommender without re-asking users.
st.subheader("Runner Profile")

col_a, col_b = st.columns(2)
with col_a:
    weight_kg = st.number_input(
        "Body weight (kg)", min_value=30, max_value=180, value=70, step=1,
        help="Drives cushion stack and shoe durability.",
    )
    age = st.number_input("Age", min_value=10, max_value=90, value=35, step=1)
    sex = st.selectbox("Sex", ["prefer not to say", "female", "male"])
with col_b:
    height_cm = st.number_input(
        "Height (cm)", min_value=120, max_value=220, value=175, step=1,
    )
    arch = st.selectbox(
        "Foot arch (if known)",
        ["unknown", "high", "normal", "low / flat"],
    )

st.markdown("**Training**")
col_c, col_d = st.columns(2)
with col_c:
    surface = st.selectbox("Primary running surface", ["road", "treadmill", "trail"])
    weekly_km = st.selectbox(
        "Weekly running distance",
        ["0-5 km", "5-10 km", "10-20 km", "20-40 km", "40+ km"],
    )
with col_d:
    goal = st.selectbox(
        "Primary use case",
        ["daily training", "easy / recovery", "long runs", "tempo / speedwork", "racing"],
    )
    comfort = st.selectbox("Preferred feel", ["balanced", "soft", "firm"])

st.markdown("**Form & gear**")
col_e, col_f = st.columns(2)
with col_e:
    foot_strike = st.selectbox(
        "How do you typically land?",
        ["I don't know", "heel", "midfoot", "forefoot"],
        help="Drives drop and stack-height recommendations.",
    )
    preferred_drop = st.selectbox(
        "Preferred / current heel-toe drop",
        ["no preference", "low (0-4 mm)", "mid (5-8 mm)", "high (9-12 mm)"],
        help="Transitions between drops should be gradual.",
    )
with col_f:
    current_shoes = st.text_input("Current main running shoe (brand + model)", value="")

st.markdown("**Injury history** (select all that apply - strongest predictor of shoe needs)")
injuries = st.multiselect(
    "Past running injuries",
    [
        "none",
        "plantar fasciitis",
        "achilles tendinopathy",
        "shin splints / MTSS",
        "tibial stress reaction",
        "knee pain (PFP)",
        "IT band syndrome",
        "calf strain",
        "posterior tibial tendinopathy",
        "ankle sprain history",
    ],
    default=[],
)


def collect_profile():
    return {
        "weight_kg": int(weight_kg),
        "height_cm": int(height_cm),
        "age": int(age),
        "sex": sex,
        "arch": arch,
        "surface": surface,
        "weekly_km": weekly_km,
        "goal": goal,
        "comfort": comfort,
        "foot_strike": foot_strike,
        "preferred_drop": preferred_drop,
        "current_shoes": current_shoes.strip(),
        "injuries": injuries,
    }


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
            profile = collect_profile()
            pronation_label = analysis_result["overall"]["classification"]
            category, shoes = recommend_shoes(pronation_label, surface=profile["surface"])

            # Persist the profile alongside the analysis for future iterations.
            session_dir = OUTPUT_ROOT / video_path.stem
            try:
                with (session_dir / "runner_profile.json").open("w") as f:
                    json.dump(profile, f, indent=2)
            except Exception as e:
                st.warning(f"Could not save runner profile: {e}")

            # Regenerate the PDF so the downloadable report includes both the
            # shoe recommendation and a snapshot of the runner profile.
            report_base = session_dir / "pronation_analysis_demo_report"
            try:
                save_demo_report(
                    analysis_result,
                    str(report_base),
                    shoe_recommendation={
                        "category": category,
                        "surface": profile["surface"],
                        "shoes": shoes,
                    },
                )
            except Exception as e:
                st.warning(f"Could not embed shoe info in the PDF: {e}")

            pdf_path = report_base.with_suffix(".pdf")
            st.session_state.analysis = {
                "pronation_label": pronation_label,
                "overall": analysis_result["overall"],
                "left_stance_only": analysis_result["left_stance_only"],
                "right_stance_only": analysis_result["right_stance_only"],
                "free_metrics": analysis_result.get("free_metrics", {}),
                "quality_issues": analysis_result.get("quality_issues", []),
                "metric_definition": analysis_result.get("metric_definition", ""),
                "pdf_path": str(pdf_path) if pdf_path.is_file() else None,
                "pdf_name": pdf_path.name,
                "category": category,
                "surface": profile["surface"],
                "shoes": shoes,
                "profile": profile,
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

    # --- Key numbers ---
    overall = result["overall"]
    left = result["left_stance_only"]
    right = result["right_stance_only"]
    free = result.get("free_metrics", {})

    st.subheader("Rearfoot eversion angle (pronation tendency)")
    st.caption(
        "Positive = pronation (eversion). Negative = supination (inversion). "
        "Per-leg values are the median across detected stance events."
    )

    def _fmt_leg(side: dict) -> str:
        v = side.get("median_deg")
        if v is None:
            return "N/A"
        label = (side.get("classification") or "").replace("_", " ")
        return f"{v:+.1f} deg ({label})" if label else f"{v:+.1f} deg"

    c1, c2, c3 = st.columns(3)
    overall_med = overall.get("median_deg")
    c1.metric(
        "Overall (median midstance)",
        f"{overall_med:+.1f} deg" if overall_med is not None else "N/A",
    )
    c2.metric(
        "Left leg",
        _fmt_leg(left),
        help=f"{left.get('n_events', 0)} stance events",
    )
    c3.metric(
        "Right leg",
        _fmt_leg(right),
        help=f"{right.get('n_events', 0)} stance events",
    )

    # --- Free metrics ---
    st.subheader("Free metrics (no extra models needed)")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Cadence (spm)",
              f"{free['cadence_spm']:.0f}" if free.get("cadence_spm") else "N/A")
    g2.metric("Left GCT (ms)",
              f"{free['left_gct_ms']:.0f}" if free.get("left_gct_ms") else "N/A")
    g3.metric("Right GCT (ms)",
              f"{free['right_gct_ms']:.0f}" if free.get("right_gct_ms") else "N/A")
    g4.metric("GCT asymmetry",
              f"{free['gct_asymmetry_pct']:.1f}%" if free.get("gct_asymmetry_pct") else "N/A")
    if free.get("step_width_norm") is not None:
        st.caption(f"Step width (normalized by shank length): {free['step_width_norm']:.3f}")

    # --- Quality checks ---
    issues = result.get("quality_issues", []) or []
    if issues:
        st.warning("Quality flags raised - interpret results with caution:")
        for code in issues:
            st.write(f"- `{code}`")
    else:
        st.info("All automated quality checks passed.")

    # --- Report download ---
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

    # --- Shoe recommendation ---
    st.subheader("Shoe Recommendation")
    st.write(f"Recommended category: **{result['category']}**")
    if result["shoes"]:
        st.write("Suggested models:")
        for shoe in result["shoes"]:
            st.write(f"- {shoe['brand']} {shoe['name']}")
    else:
        st.write("No matching shoes found.")
    st.caption(
        "MVP: this single-axis mapping uses only the rearfoot-eversion "
        "label. The multi-factor scorer (weight, mileage, foot strike, "
        "injury history, drop, comfort) is the next planned upgrade."
    )

    st.divider()
    if st.button("Start Over"):
        st.session_state.analysis = None
        st.session_state.uploader_key += 1
        st.rerun()
