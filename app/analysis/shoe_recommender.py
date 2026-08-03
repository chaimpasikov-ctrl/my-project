"""Simple rule-based shoe recommender driven by the pronation label."""

from typing import Dict, List, Optional, Tuple

import pandas as pd

from app.config.settings import SHOES_CSV


LABEL_TO_CATEGORY = {
    "neutral_alignment": "neutral",
    "mild_pronation": "supportive_neutral",
    "moderate_pronation": "stability",
    # NOTE: motion-control reserved for severe + medial injury history
    # (see future multi-factor recommender).
    "marked_pronation": "stability",
    "mild_supination": "neutral",
    "moderate_supination": "cushioned",
    "marked_supination": "cushioned",
    "insufficient_data": "neutral",
}


def get_shoe_category(label: str) -> str:
    """Map a rearfoot-eversion label to a high-level shoe category.

    Note: this is the MVP, single-axis mapping. The roadmap calls for a
    multi-factor scorer (body weight, mileage, foot strike, injury history,
    comfort preference, drop), where this label is just one input among many.
    """
    if label in LABEL_TO_CATEGORY:
        return LABEL_TO_CATEGORY[label]
    # Backward compatibility with previous label schemes.
    if "overpronation" in label or "medial_collapse" in label:
        return "stability"
    if "underpronation" in label or "lateral_offset" in label or "supination" in label:
        return "cushioned"
    return "neutral"


def load_shoe_database() -> pd.DataFrame:
    if not SHOES_CSV.is_file():
        raise FileNotFoundError(f"Shoe database not found at: {SHOES_CSV}")
    return pd.read_csv(SHOES_CSV)


def filter_shoes(
    df: pd.DataFrame,
    category: str,
    surface: Optional[str] = None,
    brand: Optional[str] = None,
) -> pd.DataFrame:
    result = df[df["category"] == category]

    if surface:
        # "road" and "mixed" are considered safe defaults for most surfaces
        result = result[result["surface"].isin([surface, "road", "mixed"])]

    if brand:
        result = result[result["brand"] == brand]

    return result


def recommend_shoes(
    pronation_label: str,
    surface: Optional[str] = None,
    brand: Optional[str] = None,
    max_results: int = 3,
) -> Tuple[str, List[Dict[str, str]]]:
    df = load_shoe_database()
    category = get_shoe_category(pronation_label)
    filtered = filter_shoes(df, category, surface, brand)

    if filtered.empty:
        return category, []

    top = filtered.head(max_results)
    return category, top[["name", "brand"]].to_dict(orient="records")
