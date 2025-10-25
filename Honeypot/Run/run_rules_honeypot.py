#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional, Sequence, Dict, List

import pandas as pd
import yaml  # type: ignore


# ------------------------------
# IO
# ------------------------------
def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[INFO] Features loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[DEBUG] Columns: {list(df.columns)}")
    print(f"[DEBUG] Sample data:\n{df.head()}")
    return df


def load_rule(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        rule = yaml.safe_load(f)
    print(f"[INFO] Rule loaded: {rule.get('name')} (ID: {rule.get('rule_id')})")
    print(f"[DEBUG] Rule parameters: {rule.get('parameters')}")
    print(f"[DEBUG] Rule scoring: {rule.get('scoring')}")
    return rule


# ------------------------------
# Helpers
# ------------------------------
NUM_COL_DEFAULTS: Dict[str, float] = {
    # 토큰 레벨 기본 컬럼들 (없으면 0으로 생성)
    "consecutive_sell_fail_windows": 0,
    "total_buy_cnt": 0,
    "total_sell_cnt": 0,
    "total_approval_cnt": 0,
    "imbalance_rate": 0.0,
    "approval_to_sell_ratio": 0.0,
    "failed_sell_proxy_frac": 0.0,
    "max_sell_share": 0.0,
    "privileged_event_flag": 0,
    "router_only_sell_proxy": 0,
    "total_windows": 0,
    "windows_with_activity": 0,
    "total_burn_events": 0,
    "total_mint_events": 0,
    "avg_burn_frac": 0.0,
    "avg_reserve_drop": 0.0,
}

PARAM_DEFAULTS: Dict[str, float] = {
    # 파라미터 누락 시 기본값
    "min_consecutive_sell_fail_windows": 3,
    "min_total_buy_cnt": 50,
    "min_imbalance_rate": 0.85,
    "min_approval_to_sell_ratio": 8.0,
    "min_failed_sell_proxy_frac": 0.50,
    "min_max_sell_share": 0.70,
    "min_windows_with_activity": 5,
    "privileged_event_flag_value": 1,
    "router_only_sell_proxy_value": 1,
}

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """필요 컬럼이 없으면 0/0.0으로 채워 넣어 내성을 높인다."""
    missing = []
    for col, default in NUM_COL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
            missing.append(col)
    if missing:
        print(f"[WARN] Missing columns were created with defaults: {missing}")
    return df


def pget(p: dict, key: str) -> float:
    """파라미터 읽기(없으면 안전한 기본값 사용)"""
    if key in p:
        return p[key]
    # 토큰 레벨 ↔ 윈도우 레벨 혼용 대비(키 이름 변형 지원)
    alias = {
        "min_total_buy_cnt": "token_level_min_total_buy_cnt",
        "token_level_min_total_buy_cnt": "min_total_buy_cnt",
    }
    if key in alias and alias[key] in p:
        return p[alias[key]]
    return PARAM_DEFAULTS.get(key)


def safe_cond_ge(df: pd.DataFrame, col: str, thr: float) -> pd.Series:
    """df[col] >= thr (col 없으면 False 시리즈)"""
    if col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df[col] >= thr


def safe_cond_eq(df: pd.DataFrame, col: str, val: float) -> pd.Series:
    """df[col] == val (col 없으면 False 시리즈)"""
    if col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df[col] == val


# ------------------------------
# Core evaluation
# ------------------------------
def evaluate_rule_sta0401(df: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """Evaluate Honeypot rule (STA0401) on feature dataframe (token-level)."""

    df = ensure_columns(df)

    p = rule.get("parameters", {}) or {}
    s = rule.get("scoring", {}) or {}
    v = rule.get("validation", {}) or {}

    # 임계치(부재 시 기본값 채움)
    thr_sell_fail = pget(p, "min_consecutive_sell_fail_windows")
    thr_total_buy = pget(p, "min_total_buy_cnt")
    thr_imbalance = pget(p, "min_imbalance_rate")
    thr_approve_ratio = pget(p, "min_approval_to_sell_ratio")
    thr_failed_frac = pget(p, "min_failed_sell_proxy_frac")
    thr_max_sell_share = pget(p, "min_max_sell_share")
    thr_windows_with_activity = pget(p, "min_windows_with_activity")
    val_privileged = pget(p, "privileged_event_flag_value")
    val_router_only = pget(p, "router_only_sell_proxy_value")

    # 조건 계산(없는 컬럼이면 False 처리)
    conds: List[pd.Series] = []

    cond_sell_fail = safe_cond_ge(df, "consecutive_sell_fail_windows", thr_sell_fail)
    conds.append(cond_sell_fail)

    cond_buy_cnt = safe_cond_ge(df, "total_buy_cnt", thr_total_buy)
    conds.append(cond_buy_cnt)

    cond_imbalance = safe_cond_ge(df, "imbalance_rate", thr_imbalance)
    conds.append(cond_imbalance)

    cond_approve_ratio = safe_cond_ge(df, "approval_to_sell_ratio", thr_approve_ratio)
    conds.append(cond_approve_ratio)

    cond_failed_sell_frac = safe_cond_ge(df, "failed_sell_proxy_frac", thr_failed_frac)
    conds.append(cond_failed_sell_frac)

    cond_sell_concentration = safe_cond_ge(df, "max_sell_share", thr_max_sell_share)
    conds.append(cond_sell_concentration)

    cond_privileged = safe_cond_eq(df, "privileged_event_flag", val_privileged)
    conds.append(cond_privileged)

    cond_router_only = safe_cond_eq(df, "router_only_sell_proxy", val_router_only)
    conds.append(cond_router_only)

    print("\n[DEBUG] Condition satisfaction counts:")
    print(f"  - consecutive_sell_fail_windows >= {thr_sell_fail}: {int(cond_sell_fail.sum())}")
    print(f"  - total_buy_cnt >= {thr_total_buy}: {int(cond_buy_cnt.sum())}")
    print(f"  - imbalance_rate >= {thr_imbalance}: {int(cond_imbalance.sum())}")
    print(f"  - approval_to_sell_ratio >= {thr_approve_ratio}: {int(cond_approve_ratio.sum())}")
    print(f"  - failed_sell_proxy_frac >= {thr_failed_frac}: {int(cond_failed_sell_frac.sum())}")
    print(f"  - max_sell_share >= {thr_max_sell_share}: {int(cond_sell_concentration.sum())}")
    print(f"  - privileged_event_flag == {val_privileged}: {int(cond_privileged.sum())}")
    print(f"  - router_only_sell_proxy == {val_router_only}: {int(cond_router_only.sum())}")

    # 분포 로그(있을 때만)
    print("\n[DEBUG] Feature value distributions:")
    for col in ["consecutive_sell_fail_windows", "total_buy_cnt", "imbalance_rate",
                "approval_to_sell_ratio", "failed_sell_proxy_frac", "max_sell_share"]:
        if col in df.columns:
            print(f"  - {col}: min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}")

    # 점수화: base_score + bonus_weight/조건수 * 충족조건수
    base_score = float(s.get("base_score", 70.0))
    detection_threshold = float(s.get("detection_threshold", 75.0))
    bonus_weight = float(s.get("bonus_weight", 30.0))  # 없으면 기본 30

    # conds는 Series들의 리스트. True=>1, False=>0
    bonus_hits = sum(c.astype(int) for c in conds)
    # 조건 개수(0으로 나누는 일 없도록 보호)
    n_conds = max(1, len(conds))

    df["bonus_hits"] = bonus_hits
    df["score"] = base_score + (bonus_weight / n_conds) * df["bonus_hits"]

    print(f"\n[DEBUG] Score distribution:")
    print(f"  - Min score: {df['score'].min():.2f}")
    print(f"  - Max score: {df['score'].max():.2f}")
    print(f"  - Mean score: {df['score'].mean():.2f}")
    print(f"  - Detection threshold: {detection_threshold}")
    print(f"  - Bonus hits distribution:\n{df['bonus_hits'].value_counts().sort_index()}")

    df["detected"] = df["score"] >= detection_threshold
    df["severity"] = df["score"].apply(lambda x: "high" if x >= 90 else ("medium" if x >= 80 else "low"))

    # Validation 섹션(옵션)
    min_buy_count = int(v.get("min_buy_count", 0))
    if min_buy_count > 0 and "total_buy_cnt" in df.columns and "total_sell_cnt" in df.columns:
        invalid = (df["total_buy_cnt"] < min_buy_count) & (df["total_sell_cnt"] > 0)
        if invalid.any():
            excluded_tokens = df.loc[invalid, "token_addr_idx"].tolist() if "token_addr_idx" in df.columns else []
            print(f"\n[INFO] ⚠️  Validation filter applied: Excluding {int(invalid.sum())} tokens (buy_count < {min_buy_count} & sell_cnt > 0)")
            if excluded_tokens:
                print(f"        Excluded Token IDs: {excluded_tokens}")
            df.loc[invalid, "detected"] = False
            df.loc[invalid, "severity"] = "excluded"

    print(f"\n[INFO] Detection evaluation complete. {int(df['detected'].sum())} tokens flagged.")

    if len(df) > 0:
        print("\n[DEBUG] Top 5 tokens by score:")
        cols = ["score", "bonus_hits", "detected", "severity"]
        if "token_addr_idx" in df.columns:
            cols = ["token_addr_idx"] + cols
        print(df.nlargest(5, "score")[cols].to_string(index=False))

    return df


# ------------------------------
# Save
# ------------------------------
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


# ------------------------------
# Main
# ------------------------------
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
