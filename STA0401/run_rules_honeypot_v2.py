#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import yaml # type: ignore


def load_features(path: Path) -> pd.DataFrame:
    """Load feature CSV file and return as DataFrame"""
    df = pd.read_csv(path)
    print(f"[INFO] Features loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[DEBUG] Columns: {list(df.columns)}")
    print(f"[DEBUG] Sample data:\n{df.head()}")
    return df


def load_rule(path: Path) -> dict:
    """Load YAML rule configuration"""
    with open(path, "r", encoding="utf-8") as f:
        rule = yaml.safe_load(f)
    print(f"[INFO] Rule loaded: {rule.get('name')} (ID: {rule.get('rule_id')})")
    print(f"[DEBUG] Rule parameters: {rule.get('parameters')}")
    print(f"[DEBUG] Rule scoring: {rule.get('scoring')}")
    return rule


def evaluate_rule_sta0401(df: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """Evaluate Honeypot rule (STA0401) on feature dataframe."""

    p = rule["parameters"]
    s = rule["scoring"]

    # 각 조건 충족 여부 계산
    cond_sell_fail = df["consecutive_sell_fail_windows"] >= p["min_consecutive_sell_fail_windows"]
    cond_buy_cnt = df["total_buy_cnt"] >= p["min_total_buy_cnt"]
    cond_imbalance = df["imbalance_rate"] >= p["min_imbalance_rate"]
    cond_approve_ratio = df["approval_to_sell_ratio"] >= p["min_approval_to_sell_ratio"]
    cond_failed_sell_frac = df["failed_sell_proxy_frac"] >= p["min_failed_sell_proxy_frac"]
    cond_sell_concentration = df["max_sell_share"] >= p["min_max_sell_share"]
    cond_privileged = df["privileged_event_flag"] == p["privileged_event_flag_value"]
    cond_router_only = df["router_only_sell_proxy"] == p["router_only_sell_proxy_value"]

    # 디버깅: 각 조건별 충족 건수 출력
    print("\n[DEBUG] Condition satisfaction counts:")
    print(f"  - consecutive_sell_fail_windows >= {p['min_consecutive_sell_fail_windows']}: {cond_sell_fail.sum()}")
    print(f"  - total_buy_cnt >= {p['min_total_buy_cnt']}: {cond_buy_cnt.sum()}")
    print(f"  - imbalance_rate >= {p['min_imbalance_rate']}: {cond_imbalance.sum()}")
    print(f"  - approval_to_sell_ratio >= {p['min_approval_to_sell_ratio']}: {cond_approve_ratio.sum()}")
    print(f"  - failed_sell_proxy_frac >= {p['min_failed_sell_proxy_frac']}: {cond_failed_sell_frac.sum()}")
    print(f"  - max_sell_share >= {p['min_max_sell_share']}: {cond_sell_concentration.sum()}")
    print(f"  - privileged_event_flag == {p['privileged_event_flag_value']}: {cond_privileged.sum()}")
    print(f"  - router_only_sell_proxy == {p['router_only_sell_proxy_value']}: {cond_router_only.sum()}")

    # 실제 feature 값 분포 확인
    print("\n[DEBUG] Feature value distributions:")
    for col in ["consecutive_sell_fail_windows", "total_buy_cnt", "imbalance_rate", 
                "approval_to_sell_ratio", "failed_sell_proxy_frac", "max_sell_share"]:
        if col in df.columns:
            print(f"  - {col}: min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}")

    # 모든 조건 중 주요 조건 충족 시 탐지로 간주
    df["score"] = s["base_score"]

    # 점수 가중치 적용
    bonus_conditions = [
        cond_sell_fail,
        cond_buy_cnt,
        cond_imbalance,
        cond_approve_ratio,
        cond_failed_sell_frac,
        cond_sell_concentration,
        cond_privileged,
        cond_router_only,
    ]

    df["bonus_hits"] = sum(bonus_conditions)
    df["score"] += df["bonus_hits"] * (s["bonus_weight"] / len(bonus_conditions))

    # 디버깅: 점수 분포 확인
    print(f"\n[DEBUG] Score distribution:")
    print(f"  - Min score: {df['score'].min():.2f}")
    print(f"  - Max score: {df['score'].max():.2f}")
    print(f"  - Mean score: {df['score'].mean():.2f}")
    print(f"  - Detection threshold: {s['detection_threshold']}")
    print(f"  - Bonus hits distribution:\n{df['bonus_hits'].value_counts().sort_index()}")

    # 최종 탐지 여부
    df["detected"] = df["score"] >= s["detection_threshold"]
    df["severity"] = df["score"].apply(
        lambda x: "high" if x >= 90 else ("medium" if x >= 80 else "low")
    )

    print(f"\n[INFO] Detection evaluation complete. {df['detected'].sum()} tokens flagged.")
    
    # 점수가 높은 상위 토큰 출력
    if len(df) > 0:
        print("\n[DEBUG] Top 5 tokens by score:")
        top_tokens = df.nlargest(5, 'score')[['token_address', 'score', 'bonus_hits', 'detected']] if 'token_address' in df.columns else df.nlargest(5, 'score')[['score', 'bonus_hits', 'detected']]
        print(top_tokens)
    
    return df


def save_detections(df: pd.DataFrame, rule: dict, output_path: Path):
    detections = df[df["detected"]].copy()

    payload = {
        "rule_id": rule.get("rule_id"),
        "name": rule.get("name"),
        "detections": detections.to_dict(orient="records"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)

    print(f"[INFO] Detections written to {output_path} ({len(detections)} records)")


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Apply honeypot detection rule (STA0401)")
    parser.add_argument("--features", required=True, help="Path to features CSV file")
    parser.add_argument("--rule", required=True, help="Path to honeypot YAML rule file")
    parser.add_argument("--output-json", required=True, help="Output path for detection JSON")

    args = parser.parse_args(argv)

    features_path = Path(args.features)
    rule_path = Path(args.rule)
    output_path = Path(args.output_json)

    df = load_features(features_path)
    rule = load_rule(rule_path)

    df_result = evaluate_rule_sta0401(df, rule)
    save_detections(df_result, rule, output_path)


if __name__ == "__main__":
    main()