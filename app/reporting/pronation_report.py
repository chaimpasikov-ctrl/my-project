import os
import json
from textwrap import wrap

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


# ---------------------------------------------------------------------------
# Label explanations
# ---------------------------------------------------------------------------
# Labels now describe the **rearfoot eversion angle (pronation tendency)**
# directly. With knee + ankle + heel keypoints the metric is no longer just
# a tibial-alignment proxy: it is the standard rear-view eversion angle in
# the frontal plane. It is still a single-camera 2D measurement, so it is a
# screening tool rather than a clinical diagnosis.

EXPLANATIONS = {
    "marked_pronation": {
        "title": "Marked Pronation",
        "meaning": (
            "Your heel rolls strongly inward during stance, producing a "
            "large rearfoot eversion angle relative to the shank."
        ),
        "impact": (
            "Suggests a strong overpronation tendency and increased loading "
            "on the medial side of the foot, ankle and lower leg."
        ),
        "shoe_recommendation": "Stability running shoes",
        "shoe_reason": (
            "Medial support may help limit excessive eversion, especially "
            "for runners with medial-side injury history."
        ),
    },
    "moderate_pronation": {
        "title": "Moderate Pronation",
        "meaning": (
            "Your heel rolls noticeably inward during stance, suggesting a "
            "moderate pronation tendency."
        ),
        "impact": (
            "Often well-tolerated but worth considering supportive footwear "
            "as mileage increases."
        ),
        "shoe_recommendation": "Stability running shoes",
        "shoe_reason": (
            "Light-to-moderate guidance can reduce repetitive medial load."
        ),
    },
    "mild_pronation": {
        "title": "Mild Pronation",
        "meaning": (
            "Your heel rolls slightly inward during stance, a small "
            "pronation tendency that is within normal variation."
        ),
        "impact": (
            "Usually unremarkable; modest guidance may improve comfort for "
            "high-mileage runners."
        ),
        "shoe_recommendation": "Supportive neutral / light-stability shoes",
        "shoe_reason": (
            "Light guidance can help without feeling overly corrective."
        ),
    },
    "neutral_alignment": {
        "title": "Neutral Alignment",
        "meaning": (
            "Your heel stays roughly in line with the shank during stance "
            "(small rearfoot eversion angle)."
        ),
        "impact": "Suggests balanced frontal-plane loading.",
        "shoe_recommendation": "Neutral running shoes",
        "shoe_reason": (
            "Neutral shoes usually provide enough cushioning without extra "
            "corrective support."
        ),
    },
    "mild_supination": {
        "title": "Mild Supination",
        "meaning": (
            "Your heel rolls slightly outward during stance, a small "
            "supination tendency."
        ),
        "impact": (
            "May modestly reduce natural shock absorption; usually well "
            "tolerated."
        ),
        "shoe_recommendation": "Neutral running shoes",
        "shoe_reason": (
            "Avoid medial posts; a neutral shoe lets the foot move freely."
        ),
    },
    "moderate_supination": {
        "title": "Moderate Supination",
        "meaning": (
            "Your heel rolls noticeably outward during stance, suggesting "
            "a moderate supination tendency."
        ),
        "impact": (
            "Reduced inward motion can mean higher impact loading through "
            "the lateral foot and lower leg."
        ),
        "shoe_recommendation": "Cushioned, flexible running shoes",
        "shoe_reason": (
            "Extra cushioning helps when natural pronation is limited."
        ),
    },
    "marked_supination": {
        "title": "Marked Supination",
        "meaning": (
            "Your heel rolls strongly outward during stance, producing a "
            "large rearfoot inversion angle relative to the shank."
        ),
        "impact": (
            "Suggests a strong supination tendency and higher impact "
            "loading on the lateral side."
        ),
        "shoe_recommendation": "Highly cushioned, flexible running shoes",
        "shoe_reason": (
            "Cushioning and flexibility can help reduce impact loading."
        ),
    },
    "insufficient_data": {
        "title": "Insufficient Data",
        "meaning": (
            "Not enough usable stance frames to make a confident assessment."
        ),
        "impact": "Interpretation is unreliable for this segment.",
        "shoe_recommendation": "No recommendation",
        "shoe_reason": "Please re-record with a clearer rear-view video.",
    },
}


def explain_pronation(label):
    return EXPLANATIONS.get(label, {
        "title": "Unknown Pattern",
        "meaning": "The system could not confidently classify the detected pattern.",
        "impact": "Interpretation is uncertain.",
        "shoe_recommendation": "No recommendation",
        "shoe_reason": "More data or better detections are needed.",
    })


# ---------------------------------------------------------------------------
# Quality-issue messages
# ---------------------------------------------------------------------------
QC_MESSAGES = {
    "low_left_event_count": (
        "Few left-leg stance events detected. Left-side interpretation is "
        "low-confidence; try a longer clip."
    ),
    "low_right_event_count": (
        "Few right-leg stance events detected. Right-side interpretation is "
        "low-confidence; try a longer clip."
    ),
    "signal_flat_check_keypoints": (
        "The eversion-angle signal is unusually flat. This often means "
        "keypoints snapped to clothing or the runner did not move enough "
        "on screen."
    ),
    "extreme_outliers_present": (
        "A few frames have extreme eversion angles (> 30 deg). Likely "
        "keypoint detection errors; the per-stance median should still be "
        "reliable."
    ),
    "large_left_right_asymmetry": (
        "Left and right legs differ noticeably. This can be a true gait "
        "asymmetry or a camera-angle artifact - inspect the rendered frames."
    ),
}


def _qc_text(issues):
    if not issues:
        return ["All quality checks passed."]
    out = []
    for code in issues:
        out.append(f"- {QC_MESSAGES.get(code, code)}")
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt(value, spec=".1f", fallback="N/A"):
    if value is None:
        return fallback
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return fallback


def _fmt_deg(value, fallback="N/A"):
    """Format an angle in degrees with one decimal place and a degree suffix."""
    if value is None:
        return fallback
    try:
        return f"{float(value):+.1f} deg"
    except (TypeError, ValueError):
        return fallback


def build_demo_report(analysis_result):
    overall = analysis_result["overall"]
    left = analysis_result["left_stance_only"]
    right = analysis_result["right_stance_only"]

    overall_exp = explain_pronation(overall["classification"])
    left_exp = explain_pronation(left["classification"])
    right_exp = explain_pronation(right["classification"])

    stronger_side = "balanced"
    left_med = left.get("median_deg")
    right_med = right.get("median_deg")
    if left_med is not None and right_med is not None:
        if abs(left_med) > abs(right_med) + 1e-6:
            stronger_side = "left"
        elif abs(right_med) > abs(left_med) + 1e-6:
            stronger_side = "right"

    report = {
        "summary": {
            "frames_analyzed": analysis_result["frames_analyzed"],
            "left_stance_frames": analysis_result["left_stance_frames"],
            "right_stance_frames": analysis_result["right_stance_frames"],
            "stronger_side": stronger_side,
            "shank_scale_px": analysis_result.get("shank_scale_px"),
            "fps": analysis_result.get("fps"),
        },
        "overall": {**overall, **overall_exp},
        "left_leg": {**left, **left_exp},
        "right_leg": {**right, **right_exp},
        "free_metrics": analysis_result.get("free_metrics", {}),
        "quality_issues": analysis_result.get("quality_issues", []),
        "metric_definition": analysis_result.get("metric_definition", ""),
    }

    return report


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------
def report_to_text(report):
    summary = report["summary"]
    overall = report["overall"]
    left = report["left_leg"]
    right = report["right_leg"]
    free = report.get("free_metrics", {})
    issues = report.get("quality_issues", [])

    lines = []
    lines.append("========== RUNNING GAIT REPORT ==========")
    lines.append("")
    lines.append("Summary")
    lines.append(f"- Frames analyzed: {summary['frames_analyzed']}")
    lines.append(f"- Left stance frames: {summary['left_stance_frames']}")
    lines.append(f"- Right stance frames: {summary['right_stance_frames']}")
    lines.append(f"- Stronger (more everted) side: {summary['stronger_side']}")
    lines.append("")

    lines.append("Overall Assessment")
    lines.append(f"- Classification: {overall['title']}")
    lines.append(f"- Median rearfoot eversion angle: {_fmt_deg(overall.get('median_deg'))}")
    lines.append(f"- Meaning: {overall['meaning']}")
    lines.append(f"- Impact: {overall['impact']}")
    lines.append(f"- Recommended shoe type: {overall['shoe_recommendation']}")
    lines.append(f"- Why: {overall['shoe_reason']}")
    lines.append("")

    lines.append("Left Leg")
    lines.append(f"- Classification: {left['title']}")
    lines.append(f"- Median rearfoot eversion angle: {_fmt_deg(left.get('median_deg'))}")
    lines.append(f"- Stance events: {left.get('n_events', 0)}")
    lines.append(f"- IQR (deg): {_fmt(left.get('iqr_deg'))}")
    lines.append(f"- Recommended shoe type: {left['shoe_recommendation']}")
    lines.append("")

    lines.append("Right Leg")
    lines.append(f"- Classification: {right['title']}")
    lines.append(f"- Median rearfoot eversion angle: {_fmt_deg(right.get('median_deg'))}")
    lines.append(f"- Stance events: {right.get('n_events', 0)}")
    lines.append(f"- IQR (deg): {_fmt(right.get('iqr_deg'))}")
    lines.append(f"- Recommended shoe type: {right['shoe_recommendation']}")
    lines.append("")

    lines.append("Free Metrics")
    lines.append(f"- Cadence (spm): {_fmt(free.get('cadence_spm'), '.0f')}")
    lines.append(f"- Left ground-contact time (ms): {_fmt(free.get('left_gct_ms'), '.0f')}")
    lines.append(f"- Right ground-contact time (ms): {_fmt(free.get('right_gct_ms'), '.0f')}")
    lines.append(f"- GCT asymmetry (%): {_fmt(free.get('gct_asymmetry_pct'), '.1f')}")
    lines.append(f"- Step width (normalized): {_fmt(free.get('step_width_norm'), '.3f')}")
    lines.append("")

    lines.append("Quality Checks")
    for line in _qc_text(issues):
        lines.append(line)
    lines.append("")

    rec = report.get("shoe_recommendation")
    if rec:
        lines.append("Shoe Recommendation")
        lines.append(f"- Recommended category: {rec.get('category', 'neutral')}")
        if rec.get("surface"):
            lines.append(f"- Surface preference: {rec['surface']}")
        shoes = rec.get("shoes") or []
        if shoes:
            lines.append("- Suggested models:")
            for shoe in shoes:
                brand = shoe.get("brand", "")
                name = shoe.get("name", "")
                lines.append(f"    * {brand} {name}".rstrip())
        else:
            lines.append("- No matching shoes found in the database.")
        lines.append("")

    lines.append("Note")
    lines.append(
        "This is a computer-vision estimate of the rearfoot eversion angle "
        "(pronation tendency) from a single rear camera. It is a real "
        "biomechanical measurement, but single-camera 2D has limitations - "
        "this is a screening tool, not a clinical diagnosis."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------
def pick_rendered_image(output_base_path):
    """Pick one rendered frame to include in the PDF."""
    session_dir = os.path.dirname(output_base_path)
    rendered_dir = os.path.join(session_dir, "rendered_predictions")
    if not os.path.isdir(rendered_dir):
        return None
    candidates = sorted(
        os.path.join(rendered_dir, f)
        for f in os.listdir(rendered_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not candidates:
        return None
    return candidates[len(candidates) // 2]


def draw_wrapped_text(pdf, text, x, y, max_chars=90, line_height=14, font="Helvetica", size=11):
    pdf.setFont(font, size)
    for line in wrap(text, width=max_chars):
        pdf.drawString(x, y, line)
        y -= line_height
    return y


def save_pdf_report(report, output_base_path, shoe_recommendation=None):
    pdf_path = output_base_path + ".pdf"
    os.makedirs(os.path.dirname(output_base_path), exist_ok=True)

    page_width, page_height = A4
    left_margin = 50
    right_margin = 50
    usable_width = page_width - left_margin - right_margin

    pdf = canvas.Canvas(pdf_path, pagesize=A4)

    summary = report["summary"]
    overall = report["overall"]
    left = report["left_leg"]
    right = report["right_leg"]
    free = report.get("free_metrics", {})
    issues = report.get("quality_issues", [])

    # ----- Page 1: summary + overall + per-leg -----
    y = page_height - 60

    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(left_margin, y, "Running Gait Report")
    y -= 28

    pdf.setFont("Helvetica-Oblique", 10)
    y = draw_wrapped_text(
        pdf,
        "Rearfoot eversion angle (pronation tendency) from a rear-view video. "
        "Single-camera 2D screening, not a clinical diagnosis.",
        left_margin, y, max_chars=100, line_height=12, font="Helvetica-Oblique", size=10,
    )
    y -= 8

    pdf.setFont("Helvetica", 11)
    pdf.drawString(left_margin, y, f"Frames analyzed: {summary['frames_analyzed']}")
    y -= 16
    pdf.drawString(left_margin, y, f"Left stance frames: {summary['left_stance_frames']}")
    y -= 16
    pdf.drawString(left_margin, y, f"Right stance frames: {summary['right_stance_frames']}")
    y -= 16
    pdf.drawString(left_margin, y, f"Stronger (more everted) side: {summary['stronger_side']}")
    y -= 24

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(left_margin, y, "Overall Assessment")
    y -= 20

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(left_margin, y, f"Classification: {overall['title']}")
    y -= 18

    pdf.setFont("Helvetica", 11)
    pdf.drawString(
        left_margin, y,
        f"Median rearfoot eversion angle: {_fmt_deg(overall.get('median_deg'))}",
    )
    y -= 20

    y = draw_wrapped_text(pdf, f"Meaning: {overall['meaning']}", left_margin, y, max_chars=90)
    y -= 6
    y = draw_wrapped_text(pdf, f"Impact: {overall['impact']}", left_margin, y, max_chars=90)
    y -= 6
    y = draw_wrapped_text(pdf, f"Recommended shoe type: {overall['shoe_recommendation']}",
                          left_margin, y, max_chars=90)
    y -= 6
    y = draw_wrapped_text(pdf, f"Why: {overall['shoe_reason']}", left_margin, y, max_chars=90)

    y -= 18
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(left_margin, y, "Leg-by-Leg Breakdown")
    y -= 20

    for side_name, side in (("Left Leg", left), ("Right Leg", right)):
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(left_margin, y, side_name)
        y -= 16
        pdf.setFont("Helvetica", 11)
        pdf.drawString(left_margin, y, f"Classification: {side['title']}")
        y -= 14
        pdf.drawString(
            left_margin, y,
            f"Median eversion angle: {_fmt_deg(side.get('median_deg'))}  "
            f"(events: {side.get('n_events', 0)}, IQR: {_fmt(side.get('iqr_deg'))} deg)",
        )
        y -= 14
        y = draw_wrapped_text(pdf, f"Meaning: {side['meaning']}", left_margin, y, max_chars=90)
        y -= 8

    # ----- Page 2: free metrics + quality + (image) -----
    pdf.showPage()
    y = page_height - 60

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(left_margin, y, "Free Metrics & Quality")
    y -= 28

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(left_margin, y, "Free metrics")
    y -= 18
    pdf.setFont("Helvetica", 11)
    pdf.drawString(left_margin, y, f"Cadence (spm): {_fmt(free.get('cadence_spm'), '.0f')}")
    y -= 14
    pdf.drawString(left_margin, y,
                   f"Left GCT (ms): {_fmt(free.get('left_gct_ms'), '.0f')}    "
                   f"Right GCT (ms): {_fmt(free.get('right_gct_ms'), '.0f')}")
    y -= 14
    pdf.drawString(left_margin, y,
                   f"GCT asymmetry: {_fmt(free.get('gct_asymmetry_pct'), '.1f')} %")
    y -= 14
    pdf.drawString(left_margin, y,
                   f"Step width (normalized): {_fmt(free.get('step_width_norm'), '.3f')}")
    y -= 24

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(left_margin, y, "Quality checks")
    y -= 18
    for line in _qc_text(issues):
        y = draw_wrapped_text(pdf, line, left_margin, y, max_chars=90)
        y -= 4

    image_path = pick_rendered_image(output_base_path)
    if image_path:
        pdf.showPage()
        y = page_height - 60
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(left_margin, y, "Example Analyzed Frame")
        y -= 30

        try:
            img = ImageReader(image_path)
            img_width, img_height = img.getSize()
            max_w = usable_width
            max_h = page_height - 180
            scale = min(max_w / img_width, max_h / img_height)
            draw_w = img_width * scale
            draw_h = img_height * scale
            x = (page_width - draw_w) / 2
            pdf.drawImage(
                img, x, y - draw_h, width=draw_w, height=draw_h,
                preserveAspectRatio=True, mask="auto",
            )
            pdf.setFont("Helvetica", 10)
            pdf.drawString(left_margin, 40,
                           f"Rendered frame: {os.path.basename(image_path)}")
        except Exception as e:
            pdf.setFont("Helvetica", 11)
            pdf.drawString(left_margin, y, f"Could not render image: {e}")

    # ----- Final page: shoe recommendation -----
    if shoe_recommendation:
        pdf.showPage()
        y = page_height - 60
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(left_margin, y, "Shoe Recommendation")
        y -= 30

        category = shoe_recommendation.get("category", "neutral")
        surface = shoe_recommendation.get("surface")
        shoes = shoe_recommendation.get("shoes") or []

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(left_margin, y, f"Recommended category: {category}")
        y -= 20

        if surface:
            pdf.setFont("Helvetica", 11)
            pdf.drawString(left_margin, y, f"Surface preference: {surface}")
            y -= 20

        reasoning = report["overall"].get("shoe_reason", "")
        if reasoning:
            y = draw_wrapped_text(pdf, f"Why this category: {reasoning}",
                                  left_margin, y, max_chars=90)
            y -= 10

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(left_margin, y, "Suggested models:")
        y -= 20

        pdf.setFont("Helvetica", 11)
        if shoes:
            for shoe in shoes:
                brand = shoe.get("brand", "")
                name = shoe.get("name", "")
                pdf.drawString(left_margin, y, f"- {brand} {name}".rstrip())
                y -= 16
        else:
            pdf.drawString(
                left_margin, y,
                "No matching shoes found in our database for this profile.",
            )
            y -= 16

        y -= 14
        draw_wrapped_text(
            pdf,
            "Note: shoe recommendations are MVP-level guidance based on the "
            "rearfoot eversion angle and your selected surface. They are "
            "not a substitute for a fitting at a specialty running store.",
            left_margin, y, max_chars=90,
            line_height=14, font="Helvetica-Oblique", size=10,
        )

    pdf.save()
    print(f"Saved PDF report to: {pdf_path}")
    return pdf_path


def save_demo_report(analysis_result, output_base_path, shoe_recommendation=None):
    """Save JSON, TXT and PDF versions of the report."""
    os.makedirs(os.path.dirname(output_base_path), exist_ok=True)

    report = build_demo_report(analysis_result)
    if shoe_recommendation:
        report["shoe_recommendation"] = shoe_recommendation

    text_report = report_to_text(report)

    json_path = output_base_path + ".json"
    txt_path = output_base_path + ".txt"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=4)
    with open(txt_path, "w") as f:
        f.write(text_report)

    print(f"Saved JSON report to: {json_path}")
    print(f"Saved text report to: {txt_path}")

    pdf_path = save_pdf_report(report, output_base_path, shoe_recommendation=shoe_recommendation)
    return report, json_path, txt_path, pdf_path
