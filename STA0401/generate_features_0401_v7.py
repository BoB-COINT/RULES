#!/usr/bin/env python3
"""
generate_features_honeypot_pair_integrated_v7.py
───────────────────────────────────────────────
✅ Swap 이벤트 기반 Buy/Sell 탐지 (정확한 방식)
✅ Token Transfer는 Approval 및 실패 거래 감지에만 사용
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
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",
    "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",
    "0x1111111254fb6c44bac0bed2854e76f90643097d",
    "0x10ed43c718714eb63d5aa57b78b54704e256024e",
    "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f",
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
    amount0_in: Optional[Decimal]
    amount1_in: Optional[Decimal]
    amount0_out: Optional[Decimal]
    amount1_out: Optional[Decimal]
    sender: Optional[str]
    to: Optional[str]


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
    swap_events: int


@dataclass
class TokenLevelFeature:
    token_addr_idx: str
    consecutive_sell_fail_windows: int
    total_buy_cnt: int
    total_sell_cnt: int
    total_approval_cnt: int
    imbalance_rate: float
    approval_to_sell_ratio: float
    failed_sell_proxy_frac: float
    max_sell_share: float
    privileged_event_flag: int
    router_only_sell_proxy: int
    total_windows: int
    windows_with_activity: int
    total_burn_events: int
    total_mint_events: int
    avg_burn_frac: float
    avg_reserve_drop: float


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
                
                tx_from = (row.get("tx_from") or "").lower().strip()
                tx_to = (row.get("tx_to") or "").lower().strip()
                
                evt = TokenEvent(
                    token_addr_idx=token_idx,
                    timestamp=parse_iso(row.get("timestamp")),
                    tx_hash=row.get("tx_hash", ""),
                    block_number=int(row.get("block_number", 0)),
                    evt_type=evt_type,
                    tx_from=tx_from,
                    tx_to=tx_to,
                    value=val,
                    success=True
                )
                event_map.setdefault(token_idx, []).append(evt)
            except Exception:
                continue
    
    return event_map


def load_pair_events(path: Path) -> Dict[str, List[PairEvent]]:
    pair_map: Dict[str, List[PairEvent]] = {}
    swap_count = 0
    
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                evt_type = (row.get("evt_type") or "").lower().strip()
                token_idx = row.get("token_addr_idx") or ""
                
                # evt_log에서 Swap 정보 파싱
                amount0_in = None
                amount1_in = None
                amount0_out = None
                amount1_out = None
                sender = None
                to = None
                
                evt_log_str = row.get("evt_log", "")
                if evt_log_str:
                    try:
                        # JSON 파싱 (문자열을 dict로 변환)
                        evt_log = json.loads(evt_log_str.replace("'", '"'))
                        amount0_in = parse_decimal(str(evt_log.get("amount0In", "")))
                        amount1_in = parse_decimal(str(evt_log.get("amount1In", "")))
                        amount0_out = parse_decimal(str(evt_log.get("amount0Out", "")))
                        amount1_out = parse_decimal(str(evt_log.get("amount1Out", "")))
                        sender = str(evt_log.get("sender", "")).lower().strip()
                        to = str(evt_log.get("to", "")).lower().strip()
                    except:
                        pass
                
                if evt_type == "swap":
                    swap_count += 1
                
                evt = PairEvent(
                    token_addr_idx=token_idx,
                    timestamp=parse_iso(row.get("timestamp")),
                    tx_hash=row.get("tx_hash") or "",
                    evt_type=evt_type,
                    reserve0=parse_decimal(row.get("reserve0")),
                    reserve1=parse_decimal(row.get("reserve1")),
                    total_supply=parse_decimal(row.get("lp_total_supply")),
                    amount0_in=amount0_in,
                    amount1_in=amount1_in,
                    amount0_out=amount0_out,
                    amount1_out=amount1_out,
                    sender=sender,
                    to=to,
                )
                pair_map.setdefault(token_idx, []).append(evt)
            except Exception:
                continue
    
    print(f"[DEBUG] Loaded {swap_count} Swap events from pair_evt.csv")
    return pair_map


# ============================================================
# Feature 계산 (Window-level) - Swap 기반
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
    i_token, n_token = 0, len(token_evts)
    i_pair, n_pair = 0, len(pair_evts)

    while first_win <= last_win:
        win_end = first_win + delta
        tx_hashes: Set[str] = set()
        
        # Token events 처리 (Approval 및 Transfer)
        approval_cnt = approve_router = 0
        approvers: Set[str] = set()
        sellers: Dict[str, int] = {}
        total_sell_tx = failed_sell = 0
        privileged_flag = 0
        
        while i_token < n_token and token_evts[i_token].timestamp < win_end:
            e = token_evts[i_token]
            tx_hashes.add(e.tx_hash)
            evt = e.evt_type
            
            if evt == "approval":
                approval_cnt += 1
                approvers.add(e.tx_from)
                if e.tx_to in KNOWN_ROUTER_ADDRS:
                    approve_router += 1
            elif evt == "transfer":
                # Transfer를 통한 매도 시도 감지 (라우터로 전송)
                if e.tx_to in KNOWN_ROUTER_ADDRS:
                    total_sell_tx += 1
                    sellers[e.tx_from] = sellers.get(e.tx_from, 0) + 1
                    if not e.success:
                        failed_sell += 1
            
            if any(k in evt for k in ["fee", "privileged", "tax"]):
                privileged_flag = 1
            
            i_token += 1
        
        # Pair events 처리 (Swap으로 실제 Buy/Sell 판별)
        buy_cnt = sell_cnt = 0
        buy_vol = sell_vol = Decimal(0)
        burn_events = mint_events = sync_events = swap_events = 0
        liquidity_flag = router_only_flag = 0
        
        window_pair = []
        while i_pair < n_pair and pair_evts[i_pair].timestamp < win_end:
            p = pair_evts[i_pair]
            window_pair.append(p)
            tx_hashes.add(p.tx_hash)
            
            if p.evt_type == "swap":
                swap_events += 1
                
                # Swap 방향 판별 (하나의 swap은 Buy 또는 Sell 중 하나)
                # token0 = 우리가 관심있는 토큰, token1 = 페어 토큰 (보통 ETH/USDT)
                # amount1In > 0 && amount0Out > 0 → ETH 넣고 토큰 받음 = BUY
                # amount0In > 0 && amount1Out > 0 → 토큰 넣고 ETH 받음 = SELL
                
                is_buy = False
                is_sell = False
                
                # 토큰이 amount0인 경우
                if p.amount0_out and p.amount0_out > 0 and p.amount1_in and p.amount1_in > 0:
                    # amount0이 나가고 amount1이 들어옴 = BUY
                    is_buy = True
                    buy_vol += p.amount0_out
                elif p.amount0_in and p.amount0_in > 0 and p.amount1_out and p.amount1_out > 0:
                    # amount0이 들어가고 amount1이 나감 = SELL
                    is_sell = True
                    sell_vol += p.amount0_in
                # 토큰이 amount1인 경우
                elif p.amount1_out and p.amount1_out > 0 and p.amount0_in and p.amount0_in > 0:
                    # amount1이 나가고 amount0이 들어옴 = BUY
                    is_buy = True
                    buy_vol += p.amount1_out
                elif p.amount1_in and p.amount1_in > 0 and p.amount0_out and p.amount0_out > 0:
                    # amount1이 들어가고 amount0이 나감 = SELL
                    is_sell = True
                    sell_vol += p.amount1_in
                
                if is_buy:
                    buy_cnt += 1
                elif is_sell:
                    sell_cnt += 1
            
            elif p.evt_type == "burn":
                burn_events += 1
            elif p.evt_type == "mint":
                mint_events += 1
            elif p.evt_type == "sync":
                sync_events += 1
            
            i_pair += 1
        
        liquidity_flag = 1 if (burn_events + mint_events + sync_events) > 0 else 0
        
        # Router-only 매도자 감지
        if len(sellers) == 1:
            addr = next(iter(sellers))
            if addr in KNOWN_ROUTER_ADDRS:
                router_only_flag = 1
        
        # LP 관련 계산
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
        
        # Imbalance 계산
        imbalance = None
        if buy_cnt + sell_cnt > 0:
            imbalance = Decimal((buy_cnt - sell_cnt) / (buy_cnt + sell_cnt))
        
        # Approval to sell ratio
        approval_to_sell = Decimal(approve_router) / Decimal(max(sell_cnt, 1)) if sell_cnt > 0 else None
        
        # Failed sell fraction
        failed_frac = Decimal(failed_sell) / Decimal(max(total_sell_tx, 1)) if total_sell_tx > 0 else None
        
        # Max sell share
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
            swap_events=swap_events,
        ))
        
        win_id += 1
        first_win += delta
    
    return result, win_id


# ============================================================
# Token-level Aggregation
# ============================================================
def aggregate_to_token_level(window_features: List[FeatureRow]) -> List[TokenLevelFeature]:
    """Window-level features를 token-level로 집계"""
    
    token_groups: Dict[str, List[FeatureRow]] = {}
    for row in window_features:
        token_groups.setdefault(row.token_addr_idx, []).append(row)
    
    token_level_features = []
    
    for token_idx, windows in token_groups.items():
        windows.sort(key=lambda w: w.win_start_ts)
        
        # consecutive_sell_fail_windows 계산
        consecutive_count = 0
        max_consecutive = 0
        for w in windows:
            is_sell_fail = (w.approval_cnt > 0) and (w.sell_cnt == 0) and (w.buy_cnt > 0)
            if is_sell_fail:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 0
        
        # 기본 집계
        total_buy_cnt = sum(w.buy_cnt for w in windows)
        total_sell_cnt = sum(w.sell_cnt for w in windows)
        total_approval_cnt = sum(w.approval_cnt for w in windows)
        total_sell_tx = sum(w.total_sell_tx for w in windows)
        
        # Imbalance rate (평균)
        valid_imbalance = [float(w.imbalance_rate) for w in windows if w.imbalance_rate is not None]
        imbalance_rate = sum(valid_imbalance) / len(valid_imbalance) if valid_imbalance else 0.0
        
        # Approval to sell ratio
        approve_to_router = sum(w.approve_to_known_router_cnt for w in windows)
        approval_to_sell_ratio = approve_to_router / total_sell_cnt if total_sell_cnt > 0 else 99.0
        
        # Failed sell proxy fraction
        failed_sell_proxy_cnt = sum(w.failed_sell_proxy_cnt for w in windows)
        failed_sell_proxy_frac = failed_sell_proxy_cnt / total_sell_tx if total_sell_tx > 0 else 0.0
        
        # Max sell share
        valid_max_sell = [float(w.max_sell_share_per_addr) for w in windows if w.max_sell_share_per_addr is not None]
        max_sell_share = max(valid_max_sell) if valid_max_sell else 0.0
        
        # Flags
        privileged_event_flag = 1 if any(w.suspected_privileged_event_flag for w in windows) else 0
        router_only_sell_proxy = 1 if any(w.router_only_sell_proxy for w in windows) else 0
        
        # Window 통계
        total_windows = len(windows)
        windows_with_activity = sum(1 for w in windows if w.win_tx_count > 0)
        
        # LP 관련
        total_burn_events = sum(w.burn_events for w in windows)
        total_mint_events = sum(w.mint_events for w in windows)
        
        valid_burn = [float(w.burn_frac) for w in windows if w.burn_frac is not None]
        avg_burn_frac = sum(valid_burn) / len(valid_burn) if valid_burn else 0.0
        
        valid_reserve = [float(w.reserve_drop_frac) for w in windows if w.reserve_drop_frac is not None]
        avg_reserve_drop = sum(valid_reserve) / len(valid_reserve) if valid_reserve else 0.0
        
        token_level_features.append(TokenLevelFeature(
            token_addr_idx=token_idx,
            consecutive_sell_fail_windows=max_consecutive,
            total_buy_cnt=total_buy_cnt,
            total_sell_cnt=total_sell_cnt,
            total_approval_cnt=total_approval_cnt,
            imbalance_rate=imbalance_rate,
            approval_to_sell_ratio=approval_to_sell_ratio,
            failed_sell_proxy_frac=failed_sell_proxy_frac,
            max_sell_share=max_sell_share,
            privileged_event_flag=privileged_event_flag,
            router_only_sell_proxy=router_only_sell_proxy,
            total_windows=total_windows,
            windows_with_activity=windows_with_activity,
            total_burn_events=total_burn_events,
            total_mint_events=total_mint_events,
            avg_burn_frac=avg_burn_frac,
            avg_reserve_drop=avg_reserve_drop,
        ))
    
    return token_level_features


# ============================================================
# 실행 메인
# ============================================================
if __name__ == "__main__":
    BASE_DIR = Path(r"D:\BoB_14기\COINT\Rules\STA0401")
    TOKEN_EVENTS_PATH = BASE_DIR / "token_evt.csv"
    PAIR_EVENTS_PATH = BASE_DIR / "pair_evt.csv"
    WINDOW_OUTPUT_PATH = BASE_DIR / "features.csv"
    TOKEN_OUTPUT_PATH = BASE_DIR / "token_features.csv"
    WINDOW_SECONDS = 5

    print("🚀 Honeypot Feature Extraction (v7: Swap-based) Started")
    print("="*60)

    token_events = load_token_events(TOKEN_EVENTS_PATH)
    pair_events = load_pair_events(PAIR_EVENTS_PATH)

    window_features, win_id = [], 0
    all_token_ids = set(token_events.keys()) | set(pair_events.keys())

    print(f"\n[INFO] Processing {len(all_token_ids)} tokens...")
    for idx in sorted(all_token_ids):
        t_evts = token_events.get(idx, [])
        p_evts = pair_events.get(idx, [])
        f, win_id = generate_token_features(t_evts, p_evts, WINDOW_SECONDS, win_id)
        window_features.extend(f)
        
        if idx == sorted(all_token_ids)[0]:
            buy_total = sum(w.buy_cnt for w in f)
            sell_total = sum(w.sell_cnt for w in f)
            swap_total = sum(w.swap_events for w in f)
            print(f"  [DEBUG] Token {idx} sample: {buy_total} buys, {sell_total} sells, {swap_total} swaps")

    # Window-level features 저장
    print(f"\n[INFO] Saving window-level features...")
    with open(WINDOW_OUTPUT_PATH, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=FeatureRow.__dataclass_fields__.keys())
        writer.writeheader()
        for r in window_features:
            writer.writerow({k: getattr(r, k) for k in FeatureRow.__dataclass_fields__.keys()})
    print(f"  ✅ Window-level features saved → {WINDOW_OUTPUT_PATH}")

    # Token-level features 집계 및 저장
    print(f"\n[INFO] Aggregating to token-level features...")
    token_level_features = aggregate_to_token_level(window_features)
    
    with open(TOKEN_OUTPUT_PATH, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=TokenLevelFeature.__dataclass_fields__.keys())
        writer.writeheader()
        for t in token_level_features:
            writer.writerow({k: getattr(t, k) for k in TokenLevelFeature.__dataclass_fields__.keys()})
    print(f"  ✅ Token-level features saved → {TOKEN_OUTPUT_PATH}")

    print(f"\n{'='*60}")
    print(f"✅ Feature extraction completed successfully!")
    print(f"   - Window-level: {len(window_features)} records")
    print(f"   - Token-level: {len(token_level_features)} records")
    
    if token_level_features:
        print(f"\n[DEBUG] Token-level statistics:")
        print(f"  - Total buy count range: {min(t.total_buy_cnt for t in token_level_features)} ~ {max(t.total_buy_cnt for t in token_level_features)}")
        print(f"  - Total sell count range: {min(t.total_sell_cnt for t in token_level_features)} ~ {max(t.total_sell_cnt for t in token_level_features)}")
    print(f"{'='*60}")