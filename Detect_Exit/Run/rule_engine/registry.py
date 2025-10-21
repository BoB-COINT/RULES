"""Evaluator registry for STE rules."""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

from .evaluators import (
    evaluate_mint_and_dump,
    evaluate_multi_drop,
    evaluate_single_drop,
    evaluate_total_lp_pull,
)
from .models import EventRecord, FeatureWindow

Evaluator = Callable[
    [
        str,
        Sequence[FeatureWindow],
        Dict[str, object],
        Dict[str, object],
        Dict[str, object],
        Dict[str, List[EventRecord]],
    ],
    Tuple[Dict[str, object], List[Dict[str, object]]],
]


RULE_ID_REGISTRY: Dict[str, Evaluator] = {
    "STE0201": evaluate_total_lp_pull,
    "STE0201.2": evaluate_multi_drop,
    "STE0302": evaluate_mint_and_dump,
}

DETECTION_TYPE_REGISTRY: Dict[str, Evaluator] = {
    "single_step": evaluate_single_drop,
    "multi_step": evaluate_multi_drop,
    "total_lp_pull": evaluate_total_lp_pull,
    "mint_and_dump": evaluate_mint_and_dump,
}


def resolve_evaluator(rule: Dict[str, object]) -> Evaluator:
    rule_id = str(rule.get("rule_id", "")).strip()
    detection_type = str(rule.get("detection_type", "")).strip().lower()

    if rule_id and rule_id in RULE_ID_REGISTRY:
        return RULE_ID_REGISTRY[rule_id]
    if detection_type and detection_type in DETECTION_TYPE_REGISTRY:
        return DETECTION_TYPE_REGISTRY[detection_type]
    return evaluate_single_drop
