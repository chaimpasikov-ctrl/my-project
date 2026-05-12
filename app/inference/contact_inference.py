from typing import Optional, Tuple

from app.inference.roboflow_client import infer_image, extract_best_prediction


def infer_contact(
    image_path: str,
    model_id: str,
) -> Tuple[Optional[str], float, dict]:
    result = infer_image(image_path, model_id=model_id)
    pred = extract_best_prediction(result)

    if pred is None:
        return None, 0.0, result

    label = pred.get("class")
    confidence = float(pred.get("confidence", 0.0))
    return label, confidence, result