from typing import Any, Dict, List, Optional

from inference_sdk import InferenceHTTPClient

from app.config.settings import ROBOFLOW_API_KEY


# ----------------------------
# Create a single global client
# ----------------------------
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY,
)


# ----------------------------
# Run inference on an image
# ----------------------------
def infer_image(image_path: str, model_id: str) -> Dict[str, Any]:
    """
    Runs Roboflow inference on an image.

    Args:
        image_path: Path to image file
        model_id: Roboflow model ID (e.g. "secondtry-bt4cj/3")

    Returns:
        Raw inference result dictionary
    """
    return CLIENT.infer(image_path, model_id=model_id)


# ----------------------------
# Extract best prediction
# ----------------------------
def extract_best_prediction(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Returns the prediction with highest confidence.

    Works for:
    - classification (contact model)
    - detection/keypoint model
    """
    predictions: List[Dict[str, Any]] = result.get("predictions", [])

    if not predictions:
        return None

    return max(predictions, key=lambda p: float(p.get("confidence", 0.0)))


# ----------------------------
# Extract runner prediction (alias)
# ----------------------------
def extract_runner_prediction(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Alias for keypoint model — keeps naming clear in pipeline.
    """
    return extract_best_prediction(result)