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
# Run a Roboflow workflow on an image
# ----------------------------
def infer_workflow(image_path: str, workspace_name: str, workflow_id: str) -> Dict[str, Any]:
    """
    Runs a Roboflow Workflow on an image.

    Args:
        image_path: Path to image file
        workspace_name: Roboflow workspace containing the workflow
        workflow_id: Roboflow workflow ID

    Returns:
        The workflow's prediction block, in ``{"predictions": [...], ...}``
        shape, so callers like ``extract_best_prediction`` don't need to
        care which workflow produced it.
    """
    result = CLIENT.run_workflow(
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        images={"image": image_path},
        use_cache=True,
    )
    return result[0]["predictions"]


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