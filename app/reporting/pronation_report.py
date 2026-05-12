import os
import json
from textwrap import wrap

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def explain_pronation(label):
    explanations = {
        "overpronation_tendency": {
            "title": "Overpronation Tendency",
            "meaning": "The ankle consistently moves inward relative to the knee during stance.",
            "impact": "This may increase load on the inner side of the foot and lower leg.",
            "shoe_recommendation": "Stability or motion-control running shoes",
            "shoe_reason": "These shoes provide additional medial support and help limit excessive inward collapse."
        },
        "mild_overpronation_tendency": {
            "title": "Mild Overpronation Tendency",
            "meaning": "The ankle shows a slight inward movement relative to the knee.",
            "impact": "This is usually mild, but additional support may improve alignment and comfort.",
            "shoe_recommendation": "Light stability shoes",
            "shoe_reason": "A moderate level of support may help without feeling overly corrective."
        },
        "neutral_tendency": {
            "title": "Neutral Alignment",
            "meaning": "The ankle stays mostly aligned under the knee during stance.",
            "impact": "This suggests balanced movement and more even load distribution.",
            "shoe_recommendation": "Neutral running shoes",
            "shoe_reason": "Neutral shoes usually provide enough cushioning without extra corrective support."
        },
        "mild_underpronation_tendency": {
            "title": "Mild Underpronation Tendency",
            "meaning": "The ankle tends to stay slightly outward relative to the knee.",
            "impact": "This may reduce natural shock absorption during contact.",
            "shoe_recommendation": "Cushioned running shoes",
            "shoe_reason": "Extra cushioning can help absorb impact when inward motion is limited."
        },
        "underpronation_tendency": {
            "title": "Underpronation Tendency",
            "meaning": "The ankle consistently stays outward, indicating reduced inward motion during stance.",
            "impact": "This can increase impact forces and reduce shock absorption.",
            "shoe_recommendation": "Highly cushioned, flexible running shoes",
            "shoe_reason": "These shoes can improve comfort and help reduce impact loading."
        },
        "no_left_stance_frames": {
            "title": "No Left Stance Frames",
            "meaning": "The system could not confidently evaluate left-leg stance frames.",
            "impact": "Left-side interpretation is limited.",
            "shoe_recommendation": "No recommendation",
            "shoe_reason": "More usable frames are needed."
        },
        "no_right_stance_frames": {
            "title": "No Right Stance Frames",
            "meaning": "The system could not confidently evaluate right-leg stance frames.",
            "impact": "Right-side interpretation is limited.",
            "shoe_recommendation": "No recommendation",
            "shoe_reason": "More usable frames are needed."
        },
    }

    return explanations.get(label, {
        "title": "Unknown Pattern",
        "meaning": "The system could not confidently classify the detected pattern.",
        "impact": "Interpretation is uncertain.",
        "shoe_recommendation": "No recommendation",
        "shoe_reason": "More data or better detections are needed."
    })


def build_demo_report(analysis_result):
    overall = analysis_result["overall"]
    left = analysis_result["left_stance_only"]
    right = analysis_result["right_stance_only"]

    overall_exp = explain_pronation(overall["classification"])
    left_exp = explain_pronation(left["classification"])
    right_exp = explain_pronation(right["classification"])

    stronger_side = "balanced"
    left_mean = left.get("mean")
    right_mean = right.get("mean")

    if left_mean is not None and right_mean is not None:
        if abs(left_mean) > abs(right_mean) + 1e-6:
            stronger_side = "left"
        elif abs(right_mean) > abs(left_mean) + 1e-6:
            stronger_side = "right"

    report = {
        "summary": {
            "frames_analyzed": analysis_result["frames_analyzed"],
            "left_stance_frames": analysis_result["left_stance_frames"],
            "right_stance_frames": analysis_result["right_stance_frames"],
            "stronger_side": stronger_side,
        },
        "overall": {
            **overall,
            **overall_exp,
        },
        "left_leg": {
            **left,
            **left_exp,
        },
        "right_leg": {
            **right,
            **right_exp,
        }
    }

    return report


def report_to_text(report):
    summary = report["summary"]
    overall = report["overall"]
    left = report["left_leg"]
    right = report["right_leg"]

    lines = []
    lines.append("========== RUNNING PRONATION REPORT ==========")
    lines.append("")
    lines.append("Summary")
    lines.append(f"- Frames analyzed: {summary['frames_analyzed']}")
    lines.append(f"- Left stance frames: {summary['left_stance_frames']}")
    lines.append(f"- Right stance frames: {summary['right_stance_frames']}")
    lines.append(f"- Stronger side: {summary['stronger_side']}")
    lines.append("")
    lines.append("Overall Assessment")
    lines.append(f"- Classification: {overall['title']}")
    lines.append(f"- Mean score: {overall['mean']:.4f}")
    lines.append(f"- Meaning: {overall['meaning']}")
    lines.append(f"- Impact: {overall['impact']}")
    lines.append(f"- Recommended shoe type: {overall['shoe_recommendation']}")
    lines.append(f"- Why: {overall['shoe_reason']}")
    lines.append("")
    lines.append("Left Leg")
    lines.append(f"- Classification: {left['title']}")
    lines.append(f"- Mean score: {left['mean'] if left['mean'] is not None else 'N/A'}")
    lines.append(f"- Meaning: {left['meaning']}")
    lines.append(f"- Recommended shoe type: {left['shoe_recommendation']}")
    lines.append("")
    lines.append("Right Leg")
    lines.append(f"- Classification: {right['title']}")
    lines.append(f"- Mean score: {right['mean'] if right['mean'] is not None else 'N/A'}")
    lines.append(f"- Meaning: {right['meaning']}")
    lines.append(f"- Recommended shoe type: {right['shoe_recommendation']}")
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
    lines.append("This is an MVP computer-vision estimate based on knee-ankle alignment during stance, not a medical diagnosis.")

    return "\n".join(lines)


def pick_rendered_image(output_base_path):
    """
    Tries to find one rendered frame to include in the PDF.
    Assumes output_base_path is something like:
    analysis_results/video_1/pronation_analysis_demo_report
    """
    session_dir = os.path.dirname(output_base_path)
    rendered_dir = os.path.join(session_dir, "rendered_predictions")

    if not os.path.isdir(rendered_dir):
        return None

    candidates = sorted(
        [
            os.path.join(rendered_dir, f)
            for f in os.listdir(rendered_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )

    if not candidates:
        return None

    # Pick middle image so it's less likely to be the very first weird frame
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

    # Page 1
    y = page_height - 60

    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(left_margin, y, "Running Pronation Report")
    y -= 30

    pdf.setFont("Helvetica", 11)
    pdf.drawString(left_margin, y, f"Frames analyzed: {summary['frames_analyzed']}")
    y -= 16
    pdf.drawString(left_margin, y, f"Left stance frames: {summary['left_stance_frames']}")
    y -= 16
    pdf.drawString(left_margin, y, f"Right stance frames: {summary['right_stance_frames']}")
    y -= 16
    pdf.drawString(left_margin, y, f"Stronger side: {summary['stronger_side']}")
    y -= 30

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(left_margin, y, "Overall Assessment")
    y -= 20

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(left_margin, y, f"Classification: {overall['title']}")
    y -= 18

    pdf.setFont("Helvetica", 11)
    pdf.drawString(left_margin, y, f"Mean score: {overall['mean']:.4f}")
    y -= 20

    y = draw_wrapped_text(pdf, f"Meaning: {overall['meaning']}", left_margin, y, max_chars=90)
    y -= 6
    y = draw_wrapped_text(pdf, f"Impact: {overall['impact']}", left_margin, y, max_chars=90)
    y -= 6
    y = draw_wrapped_text(pdf, f"Recommended shoe type: {overall['shoe_recommendation']}", left_margin, y, max_chars=90)
    y -= 6
    y = draw_wrapped_text(pdf, f"Why: {overall['shoe_reason']}", left_margin, y, max_chars=90)

    y -= 24
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(left_margin, y, "Leg-by-Leg Breakdown")
    y -= 20

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(left_margin, y, "Left Leg")
    y -= 16
    pdf.setFont("Helvetica", 11)
    left_mean_str = "N/A" if left["mean"] is None else f"{left['mean']:.4f}"
    pdf.drawString(left_margin, y, f"Classification: {left['title']}")
    y -= 16
    pdf.drawString(left_margin, y, f"Mean score: {left_mean_str}")
    y -= 16
    y = draw_wrapped_text(pdf, f"Meaning: {left['meaning']}", left_margin, y, max_chars=90)

    y -= 14
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(left_margin, y, "Right Leg")
    y -= 16
    pdf.setFont("Helvetica", 11)
    right_mean_str = "N/A" if right["mean"] is None else f"{right['mean']:.4f}"
    pdf.drawString(left_margin, y, f"Classification: {right['title']}")
    y -= 16
    pdf.drawString(left_margin, y, f"Mean score: {right_mean_str}")
    y -= 16
    y = draw_wrapped_text(pdf, f"Meaning: {right['meaning']}", left_margin, y, max_chars=90)

    y -= 24
    y = draw_wrapped_text(
        pdf,
        "Note: This is an MVP computer-vision estimate based on knee-ankle alignment during stance, not a medical diagnosis.",
        left_margin,
        y,
        max_chars=90,
        line_height=14,
        font="Helvetica-Oblique",
        size=10,
    )

    # Page 2 with image if available
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
            pdf.drawImage(img, x, y - draw_h, width=draw_w, height=draw_h, preserveAspectRatio=True, mask='auto')

            pdf.setFont("Helvetica", 10)
            pdf.drawString(left_margin, 40, f"Rendered frame: {os.path.basename(image_path)}")
        except Exception as e:
            pdf.setFont("Helvetica", 11)
            pdf.drawString(left_margin, y, f"Could not render image: {e}")

    # Final page: shoe recommendation (only if provided).
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

        # Reuse the per-classification reasoning we already have on the report.
        reasoning = report["overall"].get("shoe_reason", "")
        if reasoning:
            y = draw_wrapped_text(
                pdf,
                f"Why this category: {reasoning}",
                left_margin,
                y,
                max_chars=90,
            )
            y -= 10

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(left_margin, y, "Suggested models:")
        y -= 20

        pdf.setFont("Helvetica", 11)
        if shoes:
            for shoe in shoes:
                brand = shoe.get("brand", "")
                name = shoe.get("name", "")
                line = f"- {brand} {name}".rstrip()
                pdf.drawString(left_margin, y, line)
                y -= 16
        else:
            pdf.drawString(
                left_margin,
                y,
                "No matching shoes found in our database for this profile.",
            )
            y -= 16

        y -= 14
        draw_wrapped_text(
            pdf,
            "Note: shoe recommendations are MVP-level guidance based on the "
            "detected pronation pattern and your selected surface. They are "
            "not a substitute for a fitting at a specialty running store.",
            left_margin,
            y,
            max_chars=90,
            line_height=14,
            font="Helvetica-Oblique",
            size=10,
        )

    pdf.save()
    print(f"Saved PDF report to: {pdf_path}")
    return pdf_path


def save_demo_report(analysis_result, output_base_path, shoe_recommendation=None):
    """
    Saves:
    - JSON report
    - TXT report
    - PDF report

    When ``shoe_recommendation`` is provided, it is embedded into the report
    so the PDF/JSON/TXT all include the suggested shoe category and models.
    """
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

    pdf_path = save_pdf_report(
        report,
        output_base_path,
        shoe_recommendation=shoe_recommendation,
    )

    return report, json_path, txt_path, pdf_path