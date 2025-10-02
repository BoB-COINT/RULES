#!/usr/bin/env python3
"""CLI to evaluate STE0202 slow-rug rule against aggregated windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from rule_engine import evaluate_slow, load_rule, load_slow_features


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Apply the STE0202 rule to slow-drain feature windows")
    parser.add_argument("--features", required=True, help="Path to slow feature CSV")
    parser.add_argument("--rule", required=True, help="Path to YAML rule file")
    parser.add_argument("--output-json", help="Optional path to write detection output")

    args = parser.parse_args(argv)

    features_path = Path(args.features)
    rule_path = Path(args.rule)
    output_path = Path(args.output_json) if args.output_json else None

    features_by_token = load_slow_features(features_path)
    if not features_by_token:
        print("No slow features to evaluate.")
        return

    rule = load_rule(rule_path)
    tokens_state, detections = evaluate_slow(features_by_token, rule)

    scoring = dict(rule.get("scoring", {}))
    detection_threshold = float(scoring.get("detection_threshold", 0))

    for token_idx, token_state in tokens_state.items():
        score = token_state.get("score", 0)
        print(f"Token {token_idx}: score={score} threshold={detection_threshold}")

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
