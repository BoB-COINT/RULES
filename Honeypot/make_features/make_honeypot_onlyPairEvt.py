#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import csv, json, re, sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

getcontext().prec = 28

# -------------------- 상수 --------------------
ZERO_ADDR = "0x0000000000000000000000000000000000000000"
WINDOW_SECONDS = 5
DEBUG = True

# -------------------- 데이터 구조 --------------------
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
    pair_addr: Optional[str]
    token0: Optional[str] = None
    token1: Optional[str] = None
    target_token_addr: Optional[str] = None

@dataclass
class WindowFeature:
    buy_cnt: int
    sell_cnt: int
    owner_sell_cnt: int
    owner_sell_vol: Decimal
    non_owner_sell_cnt: int
    buy_vol: Decimal
    sell_vol: Decimal
    unique_sellers: int
    unique_buyers: int
    unique_owner_sellers: int
    burn_events: int
    mint_events: int
    sync_events: int
    swap_events: int
    sell_block_flag: int

@dataclass
class TokenFeature:
    token_addr_idx: str
    total_buy_cnt: int
    total_sell_cnt: int
    total_owner_sell_cnt: int
    total_non_owner_sell_cnt: int
    owner_sell_ratio: float
    imbalance_rate: float
    total_windows: int
    windows_with_activity: int
    total_burn_events: int
    total_mint_events: int
    s_owner_count: int
    total_sell_vol: float
    total_buy_vol: float
    total_owner_sell_vol: float
    owner_sell_vol_ratio: float
    liquidity_event_mask: int
    max_sell_share: float
    unique_sellers: int
    unique_buyers: int
    consecutive_sell_block_windows: int
    total_sell_block_windows: int
    sell_block_rate: float

# -------------------- 유틸 --------------------
def parse_iso(v: str) -> datetime:
    if not v: raise ValueError("Empty timestamp")
    if v.endswith("Z"): v = v[:-1] + "+00:00"
    dt = datetime.fromisoformat(v)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def parse_decimal(v: Optional[str]) -> Optional[Decimal]:
    if v is None or str(v).strip()=="":
        return None
    try: return Decimal(str(v))
    except InvalidOperation: return None

def try_json(s: Optional[str]) -> Optional[dict]:
    if not s or not isinstance(s, str): return None
    try: return json.loads(s.replace("'", '"'))
    except Exception: return None

def floor_to_window(dt: datetime, window_seconds: int) -> datetime:
    epoch = int(dt.timestamp())
    return datetime.fromtimestamp(epoch - (epoch % window_seconds), tz=timezone.utc)

def longest_consecutive_ones(flags: List[int]) -> int:
    best = cur = 0
    for f in flags:
        if f:
            cur += 1
            if cur > best: best = cur
        else:
            cur = 0
    return best

# -------------------- S_owner --------------------
def identify_s_owner(pair_events: List[PairEvent]) -> Set[str]:
    s: Set[str] = set()
    for evt in pair_events:
        if evt.evt_type.lower() in ("mint", "burn"):
            if evt.sender and evt.sender != ZERO_ADDR:
                s.add(evt.sender.lower())
            if evt.to and evt.to != ZERO_ADDR:
                s.add(evt.to.lower())
    return s

# -------------------- 로더 --------------------
def load_pair_events(path: Path) -> Dict[str, List[PairEvent]]:
    m: Dict[str, List[PairEvent]] = {}
    if not path.exists():
        raise FileNotFoundError(f"pair_evt.csv not found: {path}")
    
    errs = 0
    temp: Dict[str, List[PairEvent]] = {}
    
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                token_idx = (row.get("token_addr_idx") or "").strip()
                evt_type = (row.get("evt_type") or "").strip()
                
                t0 = (row.get("token0") or "").strip()
                t1 = (row.get("token1") or "").strip()
                
                # evt_log에서 Swap 정보 추출
                evt_log = row.get("evt_log", "")
                evt_data = try_json(evt_log) if evt_log else {}
                
                # amount 값 추출 (evt_log 우선, 없으면 컬럼값)
                if evt_data and evt_type.lower() == "swap":
                    amount0_in = parse_decimal(str(evt_data.get("amount0In", "")))
                    amount1_in = parse_decimal(str(evt_data.get("amount1In", "")))
                    amount0_out = parse_decimal(str(evt_data.get("amount0Out", "")))
                    amount1_out = parse_decimal(str(evt_data.get("amount1Out", "")))
                else:
                    amount0_in = parse_decimal(row.get("amount0In"))
                    amount1_in = parse_decimal(row.get("amount1In"))
                    amount0_out = parse_decimal(row.get("amount0Out"))
                    amount1_out = parse_decimal(row.get("amount1Out"))
                
                # reserve 추출 (evt_log 우선)
                if evt_data and evt_type.lower() == "sync":
                    reserve0 = parse_decimal(str(evt_data.get("reserve0", "")))
                    reserve1 = parse_decimal(str(evt_data.get("reserve1", "")))
                else:
                    reserve0 = parse_decimal(row.get("reserve0"))
                    reserve1 = parse_decimal(row.get("reserve1"))
                
                pe = PairEvent(
                    token_addr_idx=token_idx,
                    timestamp=parse_iso(row.get("timestamp")),
                    tx_hash=row.get("tx_hash",""),
                    evt_type=evt_type,
                    reserve0=reserve0,
                    reserve1=reserve1,
                    total_supply=parse_decimal(row.get("total_supply") or row.get("lp_total_supply")),
                    amount0_in=amount0_in,
                    amount1_in=amount1_in,
                    amount0_out=amount0_out,
                    amount1_out=amount1_out,
                    sender=(row.get("tx_from") or row.get("sender") or "").lower().strip() or None,
                    to=(row.get("tx_to") or row.get("to") or "").lower().strip() or None,
                    pair_addr=None,
                    token0=t0.lower() if t0 else None,
                    token1=t1.lower() if t1 else None,
                )
                
                temp.setdefault(token_idx, []).append(pe)
                            
            except Exception as e:
                if DEBUG:
                    print(f"[WARN] Parse error: {e}")
                errs += 1
                continue
    
    if errs:
        print(f"[INFO] pair_evt.csv parse errors: {errs}")
    
    # 각 token_idx에 대해 실제 토큰 주소 결정
    for token_idx, evts in temp.items():
        target_addr = None
        for e in evts:
            if e.token0 and e.token1:
                if token_idx.lower().startswith("0x"):
                    target_addr = token_idx.lower()
                else:
                    if e.reserve0 and e.reserve1:
                        if e.reserve0 > e.reserve1 * 100:
                            target_addr = e.token0
                        elif e.reserve1 > e.reserve0 * 100:
                            target_addr = e.token1
                        else:
                            target_addr = e.token0
                    else:
                        target_addr = e.token0
                break
        
        for e in evts:
            e.target_token_addr = target_addr
        
        m[token_idx] = evts
        
        if DEBUG and target_addr:
            is_token0 = (target_addr == evts[0].token0) if evts[0].token0 else None
            print(f"[INFO] token_idx={token_idx} -> {target_addr} (is_token0={is_token0})")
    
    return m

# -------------------- Swap 방향 판단 --------------------
def determine_swap_direction(evt: PairEvent) -> Tuple[str, Optional[Decimal]]:
    """Swap 이벤트에서 매수/매도 판단"""
    
    if evt.evt_type.lower() != "swap":
        return ("unknown", None)
    
    if not evt.target_token_addr or not evt.token0 or not evt.token1:
        return ("unknown", None)
    
    a0_in = evt.amount0_in or Decimal(0)
    a1_in = evt.amount1_in or Decimal(0)
    a0_out = evt.amount0_out or Decimal(0)
    a1_out = evt.amount1_out or Decimal(0)
    
    target = evt.target_token_addr.lower()
    
    # target이 token0인 경우
    if target == evt.token0.lower():
        if a0_out > 0:
            return ("buy", a0_out)
        elif a0_in > 0:
            return ("sell", a0_in)
    
    # target이 token1인 경우
    elif target == evt.token1.lower():
        if a1_out > 0:
            return ("buy", a1_out)
        elif a1_in > 0:
            return ("sell", a1_in)
    
    return ("unknown", None)

# -------------------- 윈도우 집계 --------------------
def generate_window_features(
    pair_evts: List[PairEvent],
    window_seconds: int,
    s_owner: Set[str],
    token_id_for_log: str = "?"
) -> List[WindowFeature]:
    
    if not pair_evts:
        return []
    
    pair_evts_sorted = sorted(pair_evts, key=lambda e: e.timestamp)
    
    first_ts = pair_evts_sorted[0].timestamp
    last_ts = pair_evts_sorted[-1].timestamp
    cur = floor_to_window(first_ts, window_seconds)
    last_win = floor_to_window(last_ts, window_seconds)
    delta = timedelta(seconds=window_seconds)
    
    result: List[WindowFeature] = []
    
    while cur <= last_win:
        nxt = cur + delta
        win_events = [e for e in pair_evts_sorted if cur <= e.timestamp < nxt]
        
        if not win_events:
            cur = nxt
            continue
        
        buy_cnt = sell_cnt = owner_sell_cnt = 0
        non_owner_sell_cnt = 0
        owner_sell_vol = Decimal(0)
        buy_vol = Decimal(0)
        sell_vol = Decimal(0)
        sellers: Set[str] = set()
        buyers: Set[str] = set()
        owner_sellers: Set[str] = set()
        
        burn_events = 0
        mint_events = 0
        sync_events = 0
        swap_events = 0
        
        for e in win_events:
            evt_lower = e.evt_type.lower()
            
            if evt_lower == "swap":
                swap_events += 1
                direction, volume = determine_swap_direction(e)
                
                if direction == "buy":
                    buy_cnt += 1
                    if volume:
                        buy_vol += volume
                    if e.to:
                        buyers.add(e.to.lower())
                        
                elif direction == "sell":
                    sell_cnt += 1
                    if volume:
                        sell_vol += volume
                    
                    seller = (e.sender or "").lower()
                    if seller:
                        sellers.add(seller)
                        
                        if seller in s_owner:
                            owner_sell_cnt += 1
                            if volume:
                                owner_sell_vol += volume
                            owner_sellers.add(seller)
                        else:
                            non_owner_sell_cnt += 1
            
            elif evt_lower == "burn":
                burn_events += 1
            elif evt_lower == "mint":
                mint_events += 1
            elif evt_lower == "sync":
                sync_events += 1
        
        sell_block_flag = 1 if (
            buy_cnt > 0 and
            owner_sell_cnt > 0 and
            non_owner_sell_cnt == 0 and
            len(buyers) >= 2
        ) else 0
        
        result.append(WindowFeature(
            buy_cnt=buy_cnt,
            sell_cnt=sell_cnt,
            owner_sell_cnt=owner_sell_cnt,
            owner_sell_vol=owner_sell_vol,
            non_owner_sell_cnt=non_owner_sell_cnt,
            buy_vol=buy_vol,
            sell_vol=sell_vol,
            unique_sellers=len(sellers),
            unique_buyers=len(buyers),
            unique_owner_sellers=len(owner_sellers),
            burn_events=burn_events,
            mint_events=mint_events,
            sync_events=sync_events,
            swap_events=swap_events,
            sell_block_flag=sell_block_flag,
        ))
        cur = nxt
    
    if DEBUG:
        tb = sum(w.buy_cnt for w in result)
        ts = sum(w.sell_cnt for w in result)
        tsb = sum(w.sell_block_flag for w in result)
        print(f"[DBG] token={token_id_for_log} buy={tb} sell={ts} sell_block_windows={tsb}")
    
    return result

# -------------------- 토큰 집계 --------------------
def aggregate_to_token_feature(
    token_idx: str,
    windows: List[WindowFeature],
    s_owner: Set[str],
    pair_evts: List[PairEvent],
) -> TokenFeature:
    
    total_buy_cnt = sum(w.buy_cnt for w in windows)
    total_sell_cnt = sum(w.sell_cnt for w in windows)
    total_owner_sell_cnt = sum(w.owner_sell_cnt for w in windows)
    total_non_owner_sell_cnt = sum(w.non_owner_sell_cnt for w in windows)
    
    owner_sell_ratio = (total_owner_sell_cnt / total_sell_cnt) if total_sell_cnt > 0 else 0.0
    
    imb_vals = []
    for w in windows:
        d = w.buy_cnt + w.sell_cnt
        if d > 0:
            imb_vals.append((w.buy_cnt - w.sell_cnt) / d)
    imbalance_rate = (sum(imb_vals) / len(imb_vals)) if imb_vals else 0.0
    
    total_windows = len(windows)
    windows_with_activity = sum(1 for w in windows if (w.buy_cnt + w.sell_cnt) > 0)
    
    total_burn_events = sum(w.burn_events for w in windows)
    total_mint_events = sum(w.mint_events for w in windows)
    
    total_sell_vol_dec = sum((w.sell_vol for w in windows), start=Decimal(0))
    total_buy_vol_dec = sum((w.buy_vol for w in windows), start=Decimal(0))
    total_owner_sell_vol_dec = sum((w.owner_sell_vol for w in windows), start=Decimal(0))
    
    total_sell_vol = float(total_sell_vol_dec)
    total_buy_vol = float(total_buy_vol_dec)
    total_owner_sell_vol = float(total_owner_sell_vol_dec)
    
    owner_sell_vol_ratio = float(
        (total_owner_sell_vol_dec / total_sell_vol_dec) if total_sell_vol_dec > 0 else 0
    )
    
    liquidity_event_mask = 0
    if total_mint_events > 0:
        liquidity_event_mask |= 1
    if total_burn_events > 0:
        liquidity_event_mask |= 2
    if sum(w.sync_events for w in windows) > 0:
        liquidity_event_mask |= 4
    
    seller_cnt: Dict[str, int] = {}
    for e in pair_evts:
        if e.evt_type.lower() == "swap" and e.sender:
            direction, _ = determine_swap_direction(e)
            if direction == "sell":
                seller = e.sender.lower()
                seller_cnt[seller] = seller_cnt.get(seller, 0) + 1
    
    if total_sell_cnt > 0 and seller_cnt:
        max_sell_share = max(seller_cnt.values()) / total_sell_cnt
    else:
        max_sell_share = 0.0
    
    all_sellers = set()
    all_buyers = set()
    for e in pair_evts:
        if e.evt_type.lower() == "swap":
            direction, _ = determine_swap_direction(e)
            if direction == "sell" and e.sender:
                all_sellers.add(e.sender.lower())
            elif direction == "buy" and e.to:
                all_buyers.add(e.to.lower())
    
    sell_block_flags = [w.sell_block_flag for w in windows]
    consecutive_sell_block_windows = longest_consecutive_ones(sell_block_flags)
    total_sell_block_windows = sum(sell_block_flags)
    sell_block_rate = total_sell_block_windows / total_windows if total_windows > 0 else 0.0
    
    return TokenFeature(
        token_addr_idx=token_idx,
        total_buy_cnt=total_buy_cnt,
        total_sell_cnt=total_sell_cnt,
        total_owner_sell_cnt=total_owner_sell_cnt,
        total_non_owner_sell_cnt=total_non_owner_sell_cnt,
        owner_sell_ratio=owner_sell_ratio,
        imbalance_rate=imbalance_rate,
        total_windows=total_windows,
        windows_with_activity=windows_with_activity,
        total_burn_events=total_burn_events,
        total_mint_events=total_mint_events,
        s_owner_count=len(s_owner),
        total_sell_vol=total_sell_vol,
        total_buy_vol=total_buy_vol,
        total_owner_sell_vol=total_owner_sell_vol,
        owner_sell_vol_ratio=owner_sell_vol_ratio,
        liquidity_event_mask=liquidity_event_mask,
        max_sell_share=max_sell_share,
        unique_sellers=len(all_sellers),
        unique_buyers=len(all_buyers),
        consecutive_sell_block_windows=consecutive_sell_block_windows,
        total_sell_block_windows=total_sell_block_windows,
        sell_block_rate=sell_block_rate,
    )

# -------------------- 메인 --------------------
def main():
    BASE = Path(".")
    PAIR_EVENTS_PATH = BASE / "pair_evt.csv"
    OUTPUT_PATH = BASE / "features_pair_only_v5.csv"
    
    print("=" * 60)
    print("🚀 Honeypot Feature Extraction (Pair-Only v5 FINAL)")
    print("=" * 60)
    
    print("\n[1/4] Loading data...")
    pair_events = load_pair_events(PAIR_EVENTS_PATH)
    
    all_token_ids = sorted(pair_events.keys())
    print(f"  ✅ Loaded {len(all_token_ids)} tokens")
    
    print("\n[2/4] Generating features...")
    feats: List[TokenFeature] = []
    
    for i, tid in enumerate(all_token_ids, 1):
        p_evts = pair_events.get(tid, [])
        
        s_owner = identify_s_owner(p_evts)
        windows = generate_window_features(p_evts, WINDOW_SECONDS, s_owner, token_id_for_log=tid)
        
        if windows:
            feats.append(aggregate_to_token_feature(tid, windows, s_owner, p_evts))
        
        if i % 10 == 0 or i == len(all_token_ids):
            print(f"  Progress: {i}/{len(all_token_ids)} tokens processed", end="\r")
    
    print(f"\n  ✅ Generated features for {len(feats)} tokens")
    
    print("\n[3/4] Saving features...")
    fieldnames = [
        'token_addr_idx', 'total_buy_cnt', 'total_sell_cnt',
        'total_owner_sell_cnt', 'total_non_owner_sell_cnt',
        'owner_sell_ratio', 'imbalance_rate',
        'total_windows', 'windows_with_activity',
        'total_burn_events', 'total_mint_events',
        's_owner_count', 'total_sell_vol', 'total_buy_vol',
        'total_owner_sell_vol', 'owner_sell_vol_ratio',
        'liquidity_event_mask', 'max_sell_share',
        'unique_sellers', 'unique_buyers',
        'consecutive_sell_block_windows',
        'total_sell_block_windows', 'sell_block_rate',
    ]
    
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for ftr in feats:
            w.writerow({k: getattr(ftr, k) for k in fieldnames})
    
    print(f"  ✅ Saved → {OUTPUT_PATH}")
    
    print("\n[4/4] Statistics:")
    print(f"  - Total tokens: {len(feats)}")
    if feats:
        bc = [t.total_buy_cnt for t in feats]
        sc = [t.total_sell_cnt for t in feats]
        orat = [t.owner_sell_ratio for t in feats]
        sown = [t.s_owner_count for t in feats]
        mss = [t.max_sell_share for t in feats]
        csb = [t.consecutive_sell_block_windows for t in feats]
        sbr = [t.sell_block_rate for t in feats]
        
        print(f"  - Buy count range: {min(bc)} ~ {max(bc)}")
        print(f"  - Sell count range: {min(sc)} ~ {max(sc)}")
        print(f"  - Owner sell ratio: {min(orat):.2f} ~ {max(orat):.2f}")
        print(f"  - Avg S_owner count: {sum(sown)/len(sown):.1f}")
        print(f"  - Max of max_sell_share: {max(mss):.2f}")
        print(f"  - Max consecutive sell-block windows: {max(csb)}")
        print(f"  - Avg sell_block_rate: {sum(sbr)/len(sbr):.3f}")
    
    print("\n" + "=" * 60)
    print("✅ Feature extraction completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
