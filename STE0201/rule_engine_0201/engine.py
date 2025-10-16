"""Core evaluation entrypoints."""

from __future__ import annotations

from typing import Dict, List

from .models import EventRecord, FeatureWindow, SlowFeatureWindow
from .registry import resolve_evaluator
from .evaluators import evaluate_slow_drain


def evaluate(
    features_by_token: Dict[str, List[FeatureWindow]],
    rule: Dict[str, object],
    events: Dict[str, List[EventRecord]],
):
    params = dict(rule.get("parameters", {}))
    scoring = dict(rule.get("scoring", {}))
    evidence_cfg = dict(rule.get("evidence", {}))

    tokens_state: Dict[str, Dict[str, object]] = {}
    all_detections: List[Dict[str, object]] = []

    evaluator = resolve_evaluator(rule)

    for token_idx, windows in features_by_token.items():
        token_state, detections = evaluator(
            token_idx,
            windows,
            params,
            scoring,
            evidence_cfg,
            events,
        )
        tokens_state[token_idx] = token_state
        all_detections.extend(detections)

    return tokens_state, all_detections


def evaluate_slow(
    features_by_token: Dict[str, List[SlowFeatureWindow]],
    rule: Dict[str, object],
):
    params = dict(rule.get("parameters", {}))
    scoring = dict(rule.get("scoring", {}))
    evidence_cfg = dict(rule.get("evidence", {}))

    tokens_state: Dict[str, Dict[str, object]] = {}
    all_detections: List[Dict[str, object]] = []

    for token_idx, windows in features_by_token.items():
        token_state, detections = evaluate_slow_drain(
            token_idx,
            windows,
            params,
            scoring,
            evidence_cfg,
        )
        tokens_state[token_idx] = token_state
        all_detections.extend(detections)

    return tokens_state, all_detections
