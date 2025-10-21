#!/usr/bin/env python3
"""
generate_features_honeypot_pair_integrated_v6.py
───────────────────────────────────────────────
✅ v3의 모든 honeypot feature + v5의 LP 계산 구조 완전 통합 버전
"""

from __future__ import annotations
import csv, json, re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
import sys

getcontext().prec = 28

# ============================================================
# 라우터 주소 목록
# ============================================================
KNOWN_ROUTER_ADDRS = {
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Uniswap V2
    "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",  # Uniswap V3
    "0x1111111254fb6c44bac0bed2854e76f90643097d",  # 1inch
    "0x10ed43c718714eb63d5aa57b78b54704e256024e",  # PancakeSwap
    "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f"   # SushiSwap
}

# ============================================================
# 데이터 구조 정의
# ============================================================
@dataclass
class TokenEvent:
    token_addr_idx: str
    timestamp: datetime
    tx_hash: str
    block_number: int
    evt_type: str
    tx_from: str
    tx_to: str
    value: Optional[Decimal]
    success: bool


@dataclass
class PairEvent:
    token_addr_idx: str
    timestamp: datetime
    tx_hash: str
    evt_type: str
    reserve0: Optional[Decimal]
    reserve1: Optional[Decimal]
    total_supply: Optional[Decimal]


@dataclass
class FeatureRow:
    win_id: int
    token_addr_idx: str
    win_start_ts: str
    win_tx_count: int
    is_liquidity_event_tx: int
    buy_cnt: int
    sell_cnt: int
    buy_vol_sum: Optional[str]
    sell_vol_sum: Optional[str]
    imbalance_rate: Optional[str]
    approval_cnt: int
    unique_approvers: int
    approve_to_known_router_cnt: int
    approval_to_sell_ratio: Optional[str]
    unique_sellers: int
    max_sell_share_per_addr: Optional[str]
    failed_sell_proxy_cnt: int
    failed_sell_proxy_frac: Optional[str]
    suspected_privileged_event_flag: int
    router_only_sell_proxy: int
    total_sell_tx: int
    burn_frac: Optional[str]
    reserve_drop_frac: Optional[str]
    burn_events: int
    mint_events: int
    sync_events: int


# ============================================================
# 유틸 함수
# ============================================================
def parse_iso(value: str) -> datetime:
    if not value:
        raise ValueError("Empty timestamp")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_decimal(value: Optional[str]) -> Optional[Decimal]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def extract_value_from_evtlog(log_str: str) -> Optional[Decimal]:
    if not isinstance(log_str, str):
        return None
    try:
        data = json.loads(log_str.replace("'", '"'))
        for k in ["value", "amount", "amount0In", "amount1Out", "amount1In", "amount0Out"]:
            if k in data:
                val = data[k]
                if isinstance(val, (int, float, str)):
                    return Decimal(str(val))
    except Exception:
        pass
    nums = re.findall(r"\d+", log_str)
    if nums:
        return Decimal(nums[-1])
    return None


def floor_to_window(dt: datetime, window_seconds: int) -> datetime:
    epoch = int(dt.timestamp())
    floored = epoch - (epoch % window_seconds)
    return datetime.fromtimestamp(floored, tz=timezone.utc)


def decimal_to_str(val: Optional[Decimal]) -> Optional[str]:
    if val is None:
        return None
    return format(val.normalize(), "f")


# ============================================================
# 데이터 로더
# ============================================================
def load_token_events(path: Path) -> Dict[str, List[TokenEvent]]:
    event_map: Dict[str, List[TokenEvent]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                evt_type = (row.get("evt_type") or "").lower().strip()
                token_idx = row.get("token_addr_idx") or ""
                val = None
                if "value" in row and row["value"]:
                    val = parse_decimal(row["value"])
                elif "evt_log" in row and row["evt_log"]:
                    val = extract_value_from_evtlog(row["evt_log"])
                evt = TokenEvent(
                    token_addr_idx=token_idx,
                    timestamp=parse_iso(row.get("timestamp")),
                    tx_hash=row.get("tx_hash", ""),
                    block_number=int(row.get("block_number", 0)),
                    evt_type=evt_type,
                    tx_from=(row.get("tx_from") or "").lower(),
                    tx_to=(row.get("tx_to") or "").lower(),
                    value=val,
                    success=True
                )
                event_map.setdefault(token_idx, []).append(evt)
            except Exception:
                continue
    return event_map


def load_pair_events(path: Path) -> Dict[str, List[PairEvent]]:
    pair_map: Dict[str, List[PairEvent]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                evt_type = (row.get("evt_type") or "").lower().strip()
                token_idx = row.get("token_addr_idx") or ""
                tx_hash = row.get("tx_hash") or ""
                evt = PairEvent(
                    token_addr_idx=token_idx,
                    timestamp=parse_iso(row.get("timestamp")),
                    tx_hash=tx_hash,
                    evt_type=evt_type,
                    reserve0=parse_decimal(row.get("reserve0")),
                    reserve1=parse_decimal(row.get("reserve1")),
                    total_supply=parse_decimal(row.get("lp_total_supply")),
                )
                pair_map.setdefault(token_idx, []).append(evt)
            except Exception:
                continue
    return pair_map


# ============================================================
# Feature 계산
# ============================================================
def generate_token_features(token_evts: Sequence[TokenEvent],
                            pair_evts: Sequence[PairEvent],
                            window_seconds: int,
                            win_id_start: int) -> Tuple[List[FeatureRow], int]:

    if not token_evts and not pair_evts:
        return [], win_id_start

    # timestamp 통합
    all_ts = []
    if token_evts:
        all_ts.extend([e.timestamp for e in token_evts])
    if pair_evts:
        all_ts.extend([p.timestamp for p in pair_evts])
    all_ts.sort()
    first_win = floor_to_window(all_ts[0], window_seconds)
    last_win = floor_to_window(all_ts[-1], window_seconds)
    delta = timedelta(seconds=window_seconds)

    result: List[FeatureRow] = []
    win_id = win_id_start
    token_evts = sorted(token_evts, key=lambda e: e.timestamp)
    pair_evts = sorted(pair_evts, key=lambda p: p.timestamp)
    i, n = 0, len(token_evts)

    while first_win <= last_win:
        win_end = first_win + delta
        tx_hashes: Set[str] = set()
        buy_cnt = sell_cnt = approval_cnt = approve_router = 0
        buy_vol = sell_vol = Decimal(0)
        approvers: Set[str] = set()
        sellers: Dict[str, int] = {}
        total_sell_tx = failed_sell = 0
        privileged_flag = liquidity_flag = router_only_flag = 0

        while i < n and token_evts[i].timestamp < win_end:
            e = token_evts[i]
            tx_hashes.add(e.tx_hash)
            evt = e.evt_type
            from_router = e.tx_from in KNOWN_ROUTER_ADDRS
            to_router = e.tx_to in KNOWN_ROUTER_ADDRS
            if evt == "approval":
                approval_cnt += 1
                approvers.add(e.tx_from)
                if to_router:
                    approve_router += 1
            elif evt == "transfer":
                if from_router:
                    buy_cnt += 1
                    if e.value: buy_vol += e.value
                elif to_router:
                    sell_cnt += 1
                    if e.value: sell_vol += e.value
                    total_sell_tx += 1
                    sellers[e.tx_from] = sellers.get(e.tx_from, 0) + 1
                    if not e.success:
                        failed_sell += 1
            if any(k in evt for k in ["fee", "privileged", "tax"]):
                privileged_flag = 1
            i += 1

        # router-only 매도자 감지
        if len(sellers) == 1:
            addr = next(iter(sellers))
            if addr in KNOWN_ROUTER_ADDRS:
                router_only_flag = 1

        # pair_evt 윈도우 내
        window_pair = [p for p in pair_evts if first_win <= p.timestamp < win_end]
        burn_events = len([p for p in window_pair if p.evt_type == "burn"])
        mint_events = len([p for p in window_pair if p.evt_type == "mint"])
        sync_events = len([p for p in window_pair if p.evt_type == "sync"])
        liquidity_flag = 1 if (burn_events + mint_events + sync_events) > 0 else 0

        burn_frac = reserve_drop = None
        try:
            lp_vals = [p.total_supply for p in window_pair if p.total_supply is not None]
            res_vals = [p.reserve0 or p.reserve1 for p in window_pair if (p.reserve0 or p.reserve1)]
            if len(lp_vals) >= 2:
                start_lp, end_lp = lp_vals[0], lp_vals[-1]
                burn_frac = max(Decimal(0), (start_lp - end_lp) / start_lp)
            if len(res_vals) >= 2:
                start_r, end_r = res_vals[0], res_vals[-1]
                reserve_drop = max(Decimal(0), Decimal(1) - (end_r / start_r))
        except Exception:
            pass

        imbalance = None
        if buy_cnt + sell_cnt > 0:
            imbalance = Decimal((buy_cnt - sell_cnt) / (buy_cnt + sell_cnt))

        approval_to_sell = Decimal(approve_router) / Decimal(max(sell_cnt, 1)) if sell_cnt >= 0 else None
        failed_frac = Decimal(failed_sell) / Decimal(max(total_sell_tx, 1)) if total_sell_tx > 0 else None
        max_sell_share = None
        if sellers:
            total_sells = sum(sellers.values())
            max_sell_share = Decimal(max(sellers.values())) / Decimal(total_sells)

        result.append(FeatureRow(
            win_id=win_id,
            token_addr_idx=token_evts[0].token_addr_idx if token_evts else pair_evts[0].token_addr_idx,
            win_start_ts=first_win.isoformat(),
            win_tx_count=len(tx_hashes),
            is_liquidity_event_tx=liquidity_flag,
            buy_cnt=buy_cnt,
            sell_cnt=sell_cnt,
            buy_vol_sum=decimal_to_str(buy_vol),
            sell_vol_sum=decimal_to_str(sell_vol),
            imbalance_rate=decimal_to_str(imbalance),
            approval_cnt=approval_cnt,
            unique_approvers=len(approvers),
            approve_to_known_router_cnt=approve_router,
            approval_to_sell_ratio=decimal_to_str(approval_to_sell),
            unique_sellers=len(sellers),
            max_sell_share_per_addr=decimal_to_str(max_sell_share),
            failed_sell_proxy_cnt=failed_sell,
            failed_sell_proxy_frac=decimal_to_str(failed_frac),
            suspected_privileged_event_flag=privileged_flag,
            router_only_sell_proxy=router_only_flag,
            total_sell_tx=total_sell_tx,
            burn_frac=decimal_to_str(burn_frac),
            reserve_drop_frac=decimal_to_str(reserve_drop),
            burn_events=burn_events,
            mint_events=mint_events,
            sync_events=sync_events,
        ))

        win_id += 1
        first_win += delta

    print(f"✅ {token_evts[0].token_addr_idx if token_evts else pair_evts[0].token_addr_idx}: {len(result)} windows processed")
    return result, win_id


# ============================================================
# 실행 메인
# ============================================================
if __name__ == "__main__":
    BASE_DIR = Path(r"C:\Users\ljh24\Desktop\COINT\rule")
    TOKEN_EVENTS_PATH = BASE_DIR / "token_evt.csv"
    PAIR_EVENTS_PATH = BASE_DIR / "pair_evt.csv"
    OUTPUT_PATH = BASE_DIR / "features_pair_integrated_v6.csv"
    WINDOW_SECONDS = 5

    print("🚀 Honeypot Feature Extraction (v6: full integration) Started")
    print("----------------------------------------------------")

    token_events = load_token_events(TOKEN_EVENTS_PATH)
    pair_events = load_pair_events(PAIR_EVENTS_PATH)

    features, win_id = [], 0
    all_token_ids = set(token_events.keys()) | set(pair_events.keys())

    for idx in sorted(all_token_ids):
        t_evts = token_events.get(idx, [])
        p_evts = pair_events.get(idx, [])
        f, win_id = generate_token_features(t_evts, p_evts, WINDOW_SECONDS, win_id)
        features.extend(f)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=FeatureRow.__dataclass_fields__.keys())
        writer.writeheader()
        for r in features:
            writer.writerow({k: getattr(r, k) for k in FeatureRow.__dataclass_fields__.keys()})

    print(f"\n✅ Feature extraction completed successfully → {OUTPUT_PATH}")