"""Shoe recommender driven by rearfoot eversion angle + body weight.

Support (motion-control) category comes solely from the eversion
classification label; cushioning level comes solely from body weight. Sex is
used only to softly prefer shoes with a matching sex-specific last - it never
factors into the biomechanical (support/cushioning) decision.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from app.config.settings import SHOES_CSV


LABEL_TO_SUPPORT = {
    "neutral_alignment": "neutral",
    "mild_pronation": "supportive_neutral",
    "moderate_pronation": "stability",
    "marked_pronation": "stability",
    "mild_supination": "neutral",
    "moderate_supination": "cushioned",
    "marked_supination": "cushioned",
    "insufficient_data": "neutral",
}

# Ordered low -> high so "adjacent" can be computed as an index distance of 1.
CUSHIONING_ORDER = ["low", "medium", "medium-high", "high"]


def support_category_from_label(label: str) -> str:
    """Map a rearfoot-eversion classification label to a support category."""
    return LABEL_TO_SUPPORT.get(label, "neutral")


def cushioning_from_weight(weight_kg: Optional[float]) -> str:
    """Higher body weight -> more cushioning to reduce impulsive load.

    Thresholds are rough MVP heuristics, not clinical claims.
    """
    if weight_kg is None:
        return "medium"
    if weight_kg < 60:
        return "low"       # lighter runners tolerate firmer/lower-cushion shoes
    if weight_kg < 80:
        return "medium"
    return "high"          # heavier runners benefit from more cushioning


def adjacent_cushioning(shoe_cushion: str, target_cushion: str) -> bool:
    if shoe_cushion not in CUSHIONING_ORDER or target_cushion not in CUSHIONING_ORDER:
        return False
    idx_a = CUSHIONING_ORDER.index(shoe_cushion)
    idx_b = CUSHIONING_ORDER.index(target_cushion)
    return abs(idx_a - idx_b) == 1


def score_shoe(
    shoe: Dict[str, Any],
    support_category: str,
    target_cushion: str,
    sex: Optional[str],
) -> int:
    score = 0

    # Support match - the biggest factor.
    if shoe["category"] == support_category:
        score += 3
    elif support_category == "supportive_neutral" and shoe["category"] in ("neutral", "stability"):
        score += 1  # partial credit for adjacent categories

    # Cushioning match.
    if shoe["cushioning"] == target_cushion:
        score += 2
    elif adjacent_cushioning(shoe["cushioning"], target_cushion):
        score += 1

    # Sex-specific fit filter (fit-only; never affects the biomechanical decision).
    if sex in ("male", "female") and "sex_fit" in shoe and shoe["sex_fit"] not in (None, "", "unisex"):
        if shoe["sex_fit"] != sex:
            score -= 1  # softly deprioritize, don't hard-exclude

    return score


def load_shoe_database() -> pd.DataFrame:
    if not SHOES_CSV.is_file():
        raise FileNotFoundError(f"Shoe database not found at: {SHOES_CSV}")
    return pd.read_csv(SHOES_CSV)


def _shoe_summary(shoe: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": shoe.get("name"),
        "brand": shoe.get("brand"),
        "category": shoe.get("category"),
        "cushioning": shoe.get("cushioning"),
    }


def _build_reasoning(
    pronation_label: str,
    weight_kg: Optional[float],
    support_category: str,
    target_cushion: str,
    sex: Optional[str],
    primary_matches_sex_fit: bool,
) -> str:
    label_text = (pronation_label or "insufficient_data").replace("_", " ")
    support_text = support_category.replace("_", "-")

    if weight_kg is not None:
        weight_clause = f"body weight ({weight_kg:.0f} kg)"
    else:
        weight_clause = "an unspecified body weight (defaulted to medium cushioning)"

    reasoning = (
        f"Based on your eversion angle ({label_text}) and {weight_clause}, "
        f"we recommend a {support_text} shoe with {target_cushion} cushioning."
    )

    if sex in ("male", "female") and primary_matches_sex_fit:
        reasoning += f" This model also comes in a {sex}-specific fit."

    return reasoning


def recommend_shoes(
    pronation_label: str,
    weight_kg: Optional[float] = None,
    sex: Optional[str] = None,
    max_alternatives: int = 2,
) -> Dict[str, Any]:
    """Recommend a shoe from the eversion label (support) and weight (cushioning).

    Sex is used only to softly prefer a matching sex-specific last; it never
    changes the support_category or cushioning_level decision.
    """
    df = load_shoe_database()
    support_category = support_category_from_label(pronation_label)
    target_cushion = cushioning_from_weight(weight_kg)

    shoes = df.to_dict(orient="records")
    scored = [
        (score_shoe(shoe, support_category, target_cushion, sex), shoe)
        for shoe in shoes
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    if not scored:
        return {
            "support_category": support_category,
            "cushioning_level": target_cushion,
            "primary": None,
            "alternatives": [],
            "reasoning": _build_reasoning(
                pronation_label, weight_kg, support_category, target_cushion, sex, False
            ),
        }

    top_score, primary = scored[0]
    primary_sex_fit = str(primary.get("sex_fit", "")).strip().lower() == (sex or "")

    alternatives = [
        shoe
        for score, shoe in scored[1:]
        if top_score - score <= 1
    ][:max_alternatives]

    return {
        "support_category": support_category,
        "cushioning_level": target_cushion,
        "primary": _shoe_summary(primary),
        "alternatives": [_shoe_summary(shoe) for shoe in alternatives],
        "reasoning": _build_reasoning(
            pronation_label, weight_kg, support_category, target_cushion, sex, primary_sex_fit
        ),
    }
