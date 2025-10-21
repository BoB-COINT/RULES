#!/usr/bin/env python3
"""CLI to evaluate STE0201 rule against feature windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from rule_engine import evaluate, load_events, load_features, load_rule


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Apply the STE0201 rule to feature windows")
    parser.add_argument("--features", required=True, help="Path to feature CSV")
    parser.add_argument("--rule", required=True, help="Path to YAML rule file")
    parser.add_argument("--output-json", help="Path to write detection JSON if triggered")
    parser.add_argument("--token-events", help="Optional Token_onData CSV for evidence")

    args = parser.parse_args(argv)

    features_path = Path(args.features)
    rule_path = Path(args.rule)
    output_path = Path(args.output_json) if args.output_json else None
    events_path = Path(args.token_events) if args.token_events else None

    features_by_token = load_features(features_path)
    if not features_by_token:
        print("No feature windows to evaluate.")
        return

    rule = load_rule(rule_path)
    events = load_events(events_path)
    tokens_state, detections = evaluate(features_by_token, rule, events)

    scoring = dict(rule.get("scoring", {}))
    detection_threshold = float(scoring.get("detection_threshold", 0))

    for token_idx, token_state in tokens_state.items():
        score = token_state.get("score", 0)
        severity = token_state.get("severity")
        severity_str = f", severity={severity}" if severity is not None else ""
        print(
            f"Token {token_idx}: score={score} threshold={detection_threshold}{severity_str}"
        )

    if detections and output_path:
        payload = {
            "rule_id": rule.get("rule_id"),
            "name": rule.get("name"),
            "detections": detections,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False)
        print(f"Detections written to {output_path}")
    elif detections:
        print("Detections identified but no --output-json path provided.")


if __name__ == "__main__":  # pragma: no cover
    main()
