#!/usr/bin/env python3
"""
generate_features_0401_v8.py  (robust, pair-aware + fallback)
─────────────────────────────────────────────────────────
허니팟 토큰 탐지를 위한 Feature 추출 스크립트
- Transfer 기반 Buy/Sell (EOA ↔ Pair)
- PairCreated.evt_log 에서 pairaddr 주입 (로더에서 확정)
- pair_addr 비거나 부족 시 Transfer 패턴으로 페어 후보 추정 (폴백)
- Approval 라우터 집계: evt_log.spender 파싱
- S_owner: 초기 민팅 수령자 + LP 민터/버너(sender/to)
- 디버그 요약: 토큰별 페어 후보 및 매칭된 Transfer 건수 출력
"""

from __future__ import annotations
import csv, json, re, sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Dict, List, Optional, Set

getcontext().prec = 28

# -------------------- 상수 --------------------
KNOWN_ROUTER_ADDRS = {
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Uniswap V2
    "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",  # Uniswap V3
    "0x1111111254fb6c44bac0bed2854e76f90643097d",  # 1inch
    "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f",  # Sushi
}
ZERO_ADDR = "0x0000000000000000000000000000000000000000"
WINDOW_SECONDS = 5
DEBUG = True  # 필요시 False

# -------------------- 데이터 구조 --------------------
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
    spender: Optional[str] = None  # Approval용

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
    pair_addr: Optional[str]  # ★ 로더에서 주입

@dataclass
class WindowFeature:
    buy_cnt: int
    sell_cnt: int
    owner_sell_cnt: int
    owner_sell_vol: Decimal
    non_owner_sell_cnt: int
    buy_vol: Decimal
    sell_vol: Decimal
    approval_cnt: int
    unique_approvers: int
    approve_to_known_router_cnt: int
    unique_sellers: int
    unique_owner_sellers: int
    max_sell_share: Optional[Decimal]
    privileged_event_flag: int
    router_only_sell_proxy: int
    burn_events: int
    mint_events: int
    sync_events: int
    swap_events: int

@dataclass
class TokenFeature:
    token_addr_idx: str
    consecutive_sell_fail_windows: int
    total_buy_cnt: int
    total_sell_cnt: int
    total_owner_sell_cnt: int
    total_non_owner_sell_cnt: int
    owner_sell_ratio: float
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
    s_owner_count: int
    total_sell_vol: float
    total_owner_sell_vol: float
    owner_sell_vol_ratio: float
    router_approval_rate: float

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

def extract_value_from_evtlog(log_str: str) -> Optional[Decimal]:
    data = try_json(log_str)
    if isinstance(data, dict):
        for k in ["value","amount","amount0In","amount1Out","amount1In","amount0Out"]:
            if k in data:
                try: return Decimal(str(data[k]))
                except Exception: pass
    m = re.findall(r"\d+", log_str or "")
    if m:
        try: return Decimal(m[-1])
        except Exception: return None
    return None

def parse_spender_from_evtlog(log_str: Optional[str]) -> Optional[str]:
    data = try_json(log_str)
    if isinstance(data, dict):
        sp = (data.get("spender") or data.get("to") or "").lower().strip()
        return sp or None
    return None

def floor_to_window(dt: datetime, window_seconds: int) -> datetime:
    epoch = int(dt.timestamp())
    return datetime.fromtimestamp(epoch - (epoch % window_seconds), tz=timezone.utc)

# -------------------- S_owner --------------------
def identify_s_owner(token_events: List[TokenEvent], pair_events: List[PairEvent]) -> Set[str]:
    s = set()
    token_sorted = sorted(token_events, key=lambda e:(e.block_number,e.timestamp))
    first_swap_ts = min((e.timestamp for e in pair_events if e.evt_type=="swap"), default=None)

    for evt in token_sorted:
        if evt.evt_type=="transfer" and evt.tx_from==ZERO_ADDR and evt.tx_to:
            if first_swap_ts and evt.timestamp>first_swap_ts: break
            s.add(evt.tx_to.lower()); 
            if len(s)>=10: break

    for evt in pair_events:
        if evt.evt_type in ("mint","burn"):
            if evt.sender: s.add(evt.sender.lower())
            if evt.to:     s.add(evt.to.lower())
    return s

# -------------------- 로더 --------------------
def load_token_events(path: Path) -> Dict[str, List[TokenEvent]]:
    m: Dict[str, List[TokenEvent]] = {}
    if not path.exists():
        print(f"[WARN] token_evt.csv not found: {path}")
        return m

    errs = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                evt_type = (row.get("evt_type") or "").lower().strip()
                token_idx = (row.get("token_addr_idx") or "").strip()

                raw_log = row.get("evt_log")
                data = try_json(raw_log) if raw_log else None

                # 기본값: 트랜잭션 레벨 주소(폴백용)
                tx_from = (row.get("tx_from") or "").lower().strip()
                tx_to   = (row.get("tx_to")   or "").lower().strip()
                val = None
                spender = None

                if evt_type == "transfer" and isinstance(data, dict):
                    # ★ 이벤트 레벨 주소/값으로 치환
                    tx_from = (data.get("from") or "").lower().strip() or tx_from
                    tx_to   = (data.get("to")   or "").lower().strip() or tx_to
                    if "value" in data:
                        try:
                            val = Decimal(str(data["value"]))
                        except Exception:
                            val = None
                    else:
                        # 드물게 다른 키를 쓰는 소스가 있으면 기존 헬퍼 재사용
                        val = extract_value_from_evtlog(raw_log)
                elif evt_type == "approval" and isinstance(data, dict):
                    # ★ Approval 주체/대상 보정
                    # 표준: event Approval(address indexed owner, address indexed spender, uint value)
                    owner = (data.get("owner") or data.get("from") or "").lower().strip()
                    sp    = (data.get("spender") or data.get("to") or "").lower().strip()
                    if owner:
                        tx_from = owner
                    if sp:
                        spender = sp
                    # value 파싱(있으면)
                    if "value" in data:
                        try:
                            val = Decimal(str(data["value"]))
                        except Exception:
                            val = None
                    else:
                        val = extract_value_from_evtlog(raw_log)
                else:
                    # 기타 이벤트: 값만 시도
                    if raw_log:
                        val = extract_value_from_evtlog(raw_log)
                    if val is None and row.get("value"):
                        val = parse_decimal(row.get("value"))

                evt = TokenEvent(
                    token_addr_idx=token_idx,
                    timestamp=parse_iso(row.get("timestamp")),
                    tx_hash=row.get("tx_hash",""),
                    block_number=int(row.get("block_number",0)),
                    evt_type=evt_type,
                    tx_from=tx_from,
                    tx_to=tx_to,
                    value=val,
                    success=True,
                    spender=spender
                )
                m.setdefault(token_idx, []).append(evt)

            except Exception:
                errs += 1
                continue

    if errs:
        print(f"[INFO] token_evt.csv parse errors: {errs}")
    return m

def load_pair_events(path: Path) -> Dict[str, List[PairEvent]]:
    """
    - PairCreated.evt_log의 'pairaddr'를 토큰별로 수집
    - 2차 패스에서 같은 token의 모든 이벤트에 pair_addr 주입
    """
    m: Dict[str, List[PairEvent]] = {}
    if not path.exists():
        raise FileNotFoundError(f"pair_evt.csv not found: {path}")
    errs=0
    temp: Dict[str, List[PairEvent]] = {}
    pairaddr_by_token: Dict[str, List[str]] = {}

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                token_idx = (row.get("token_addr_idx") or "").strip()
                evt_type = (row.get("evt_type") or "").lower().strip()
                reserve0 = parse_decimal(row.get("reserve0"))
                reserve1 = parse_decimal(row.get("reserve1"))
                total_supply = parse_decimal(row.get("total_supply") or row.get("lp_total_supply"))

                pe = PairEvent(
                    token_addr_idx=token_idx,
                    timestamp=parse_iso(row.get("timestamp")),
                    tx_hash=row.get("tx_hash",""),
                    evt_type=evt_type,
                    reserve0=reserve0,
                    reserve1=reserve1,
                    total_supply=total_supply,
                    amount0_in=parse_decimal(row.get("amount0In")),
                    amount1_in=parse_decimal(row.get("amount1In")),
                    amount0_out=parse_decimal(row.get("amount0Out")),
                    amount1_out=parse_decimal(row.get("amount1Out")),
                    sender=(row.get("tx_from") or "").lower().strip() or None,
                    to=(row.get("tx_to") or "").lower().strip() or None,
                    pair_addr=None,
                )
                temp.setdefault(token_idx, []).append(pe)

                if evt_type=="paircreated":
                    data = try_json(row.get("evt_log"))
                    if isinstance(data, dict):
                        p = (data.get("pairaddr") or "").lower().strip()
                        if p:
                            pairaddr_by_token.setdefault(token_idx, [])
                            if p not in pairaddr_by_token[token_idx]:
                                pairaddr_by_token[token_idx].append(p)
            except Exception:
                errs+=1
                continue
    if errs: print(f"[INFO] pair_evt.csv parse errors: {errs}")

    # 주입
    for token_idx, evts in temp.items():
        plist = pairaddr_by_token.get(token_idx, [])
        rep = plist[0] if plist else None
        for e in evts:
            e.pair_addr = rep
        m[token_idx] = evts
    return m

# -------------------- 폴백: Transfer로 페어 후보 추정 --------------------
def infer_pair_addrs_from_transfers(token_evts: List[TokenEvent]) -> Set[str]:
    """
    아이디어: 페어는 buy(=pair→EOA), sell(=EOA→pair) 양쪽에서 '상대'로 반복 등장.
    - from 측 빈도, to 측 빈도 둘 다 높은 주소의 교집합을 후보로 채택
    - 지나치게 많으면 상위 빈도만 사용
    """
    from_cnt: Dict[str,int] = {}
    to_cnt: Dict[str,int]   = {}

    for e in token_evts:
        if e.evt_type!="transfer": 
            continue
        if e.tx_from: from_cnt[e.tx_from] = from_cnt.get(e.tx_from,0)+1
        if e.tx_to:   to_cnt[e.tx_to]     = to_cnt.get(e.tx_to,0)+1

    cands = set(from_cnt.keys()) & set(to_cnt.keys())
    if len(cands) > 12:
        def topk(d, k=12):
            return {k_ for k_, v in sorted(d.items(), key=lambda x:-x[1])[:k]}
        cands = topk(from_cnt) & topk(to_cnt)
    # heuristics: 라우터/EOA도 섞일 수 있으므로, 이후 매칭에서 자연 필터됨
    return {c for c in cands if c != ZERO_ADDR}

# -------------------- 윈도우 집계 --------------------
def generate_window_features(
    token_evts: List[TokenEvent],
    pair_evts: List[PairEvent],
    window_seconds: int,
    s_owner: Set[str],
    token_id_for_log: str = "?"
) -> List[WindowFeature]:

    # 1) 로더에서 주입된 pair_addr set
    pair_addr_set: Set[str] = { (p.pair_addr or "") for p in pair_evts if p.pair_addr }
    pair_addr_set = {x.lower() for x in pair_addr_set if x}

    # 2) 폴백: 비거나 너무 적으면 Transfer 기반 추정 병합
    if len(pair_addr_set) == 0:
        inferred = infer_pair_addrs_from_transfers(token_evts)
        pair_addr_set |= inferred

    if DEBUG:
        # 디버그: 토큰별 요약(처음 한 번만)
        total_transfers = sum(1 for e in token_evts if e.evt_type=="transfer")
        print(f"[DBG] token={token_id_for_log} pair_addrs={list(sorted(pair_addr_set))} transfers={total_transfers}")

    # 이벤트 타임라인 구성
    all_events = [(e.timestamp,"token",e) for e in token_evts] + \
                 [(e.timestamp,"pair", e) for e in pair_evts]
    if not all_events:
        return []
    all_events.sort(key=lambda x: x[0])

    first_ts = all_events[0][0]
    last_ts  = all_events[-1][0]
    cur = floor_to_window(first_ts, window_seconds)
    last_win = floor_to_window(last_ts, window_seconds)
    delta = timedelta(seconds=window_seconds)

    result: List[WindowFeature] = []

    while cur <= last_win:
        nxt = cur + delta
        win_token = [e for ts,typ,e in all_events if typ=="token" and cur<=ts<nxt]
        win_pair  = [e for ts,typ,e in all_events if typ=="pair"  and cur<=ts<nxt]

        if not win_token and not win_pair:
            cur = nxt; continue

        buy_cnt = sell_cnt = owner_sell_cnt = 0
        non_owner_sell_cnt = 0
        owner_sell_vol = Decimal(0)
        buy_vol = Decimal(0)
        sell_vol = Decimal(0)
        sellers: Dict[str,int] = {}
        owner_sellers: Set[str] = set()

        # Transfer 기반 집계
        for e in win_token:
            if e.evt_type != "transfer": continue
            frm = e.tx_from; to = e.tx_to
            # buy: pair -> EOA
            if frm in pair_addr_set and to and to not in pair_addr_set:
                buy_cnt += 1
                if e.value is not None: buy_vol += e.value
            # sell: EOA -> pair
            elif to in pair_addr_set and frm and frm not in pair_addr_set:
                sell_cnt += 1
                if e.value is not None: sell_vol += e.value
                if frm in s_owner:
                    owner_sell_cnt += 1
                    if e.value is not None: owner_sell_vol += e.value
                    owner_sellers.add(frm)
                else:
                    non_owner_sell_cnt += 1
                sellers[frm] = sellers.get(frm,0)+1

        approval_cnt = 0
        approvers: Set[str] = set()
        approve_router = 0
        for e in win_token:
            if e.evt_type=="approval":
                approval_cnt += 1
                approvers.add(e.tx_from)
                sp = (e.spender or "").lower() if e.spender else None
                if sp and sp in KNOWN_ROUTER_ADDRS:
                    approve_router += 1

        max_sell_share = None
        if sellers:
            total_sells = sum(sellers.values())
            max_sell_share = Decimal(max(sellers.values()))/Decimal(total_sells)

        privileged_flag = 1 if (buy_cnt>0 and sell_cnt==0 and approval_cnt>0) else 0
        router_only_flag = 1 if (sell_cnt==0 and approval_cnt>0 and approve_router==approval_cnt) else 0

        burn_events = sum(1 for e in win_pair if e.evt_type=="burn")
        mint_events = sum(1 for e in win_pair if e.evt_type=="mint")
        sync_events = sum(1 for e in win_pair if e.evt_type=="sync")
        swap_events = sum(1 for e in win_pair if e.evt_type=="swap")

        result.append(WindowFeature(
            buy_cnt=buy_cnt, sell_cnt=sell_cnt,
            owner_sell_cnt=owner_sell_cnt, owner_sell_vol=owner_sell_vol,
            non_owner_sell_cnt=non_owner_sell_cnt,
            buy_vol=buy_vol, sell_vol=sell_vol,
            approval_cnt=approval_cnt, unique_approvers=len(approvers),
            approve_to_known_router_cnt=approve_router,
            unique_sellers=len(sellers), unique_owner_sellers=len(owner_sellers),
            max_sell_share=max_sell_share,
            privileged_event_flag=privileged_flag,
            router_only_sell_proxy=router_only_flag,
            burn_events=burn_events, mint_events=mint_events,
            sync_events=sync_events, swap_events=swap_events
        ))

        cur = nxt

    if DEBUG:
        # 윈도우 합계도 찍어 보자(디버그)
        tb = sum(w.buy_cnt for w in result)
        ts = sum(w.sell_cnt for w in result)
        print(f"[DBG] token={token_id_for_log} buy_sum={tb} sell_sum={ts}")
    return result

# -------------------- 토큰 집계 --------------------
def aggregate_to_token_feature(token_idx: str, windows: List[WindowFeature], s_owner: Set[str]) -> TokenFeature:
    consec = cur = 0
    for w in windows:
        is_fail = (w.approval_cnt>0) and (w.non_owner_sell_cnt==0) and (w.buy_cnt>0)
        if is_fail: cur+=1; consec=max(consec,cur)
        else: cur=0

    total_buy_cnt = sum(w.buy_cnt for w in windows)
    total_sell_cnt = sum(w.sell_cnt for w in windows)
    total_owner_sell_cnt = sum(w.owner_sell_cnt for w in windows)
    total_non_owner_sell_cnt = sum(w.non_owner_sell_cnt for w in windows)
    total_approval_cnt = sum(w.approval_cnt for w in windows)
    owner_sell_ratio = (total_owner_sell_cnt/total_sell_cnt) if total_sell_cnt>0 else 0.0

    imb_vals=[]
    for w in windows:
        d = w.buy_cnt + w.sell_cnt
        if d>0: imb_vals.append((w.buy_cnt - w.sell_cnt)/d)
    imbalance_rate = (sum(imb_vals)/len(imb_vals)) if imb_vals else 0.0

    approval_to_sell_ratio = (total_approval_cnt/total_non_owner_sell_cnt) if total_non_owner_sell_cnt>0 else 99.0
    failed_sell_proxy_frac = 0.0

    max_sell_values = [float(w.max_sell_share) for w in windows if w.max_sell_share is not None]
    max_sell_share = max(max_sell_values) if max_sell_values else 0.0

    privileged_event_flag = 1 if any(w.privileged_event_flag for w in windows) else 0
    router_only_sell_proxy = 1 if any(w.router_only_sell_proxy for w in windows) else 0

    total_windows = len(windows)
    windows_with_activity = sum(1 for w in windows if (w.buy_cnt+w.sell_cnt+w.approval_cnt)>0)
    total_burn_events = sum(w.burn_events for w in windows)
    total_mint_events = sum(w.mint_events for w in windows)
    avg_burn_frac = 0.0
    avg_reserve_drop = 0.0

    total_sell_vol_dec = sum((w.sell_vol for w in windows), start=Decimal(0))
    total_owner_sell_vol_dec = sum((w.owner_sell_vol for w in windows), start=Decimal(0))
    total_sell_vol = float(total_sell_vol_dec)
    total_owner_sell_vol = float(total_owner_sell_vol_dec)
    owner_sell_vol_ratio = float((total_owner_sell_vol_dec/total_sell_vol_dec) if total_sell_vol_dec>0 else 0)

    approve_to_router = sum(w.approve_to_known_router_cnt for w in windows)
    router_approval_rate = (approve_to_router/total_approval_cnt) if total_approval_cnt>0 else 0.0

    return TokenFeature(
        token_addr_idx=token_idx,
        consecutive_sell_fail_windows=consec,
        total_buy_cnt=total_buy_cnt,
        total_sell_cnt=total_sell_cnt,
        total_owner_sell_cnt=total_owner_sell_cnt,
        total_non_owner_sell_cnt=total_non_owner_sell_cnt,
        owner_sell_ratio=owner_sell_ratio,
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
        s_owner_count=len(s_owner),
        total_sell_vol=total_sell_vol,
        total_owner_sell_vol=total_owner_sell_vol,
        owner_sell_vol_ratio=owner_sell_vol_ratio,
        router_approval_rate=router_approval_rate
    )

# -------------------- 메인 --------------------
def main():
    BASE = Path(".")
    TOKEN_EVENTS_PATH = BASE / "token_evt.csv"
    PAIR_EVENTS_PATH  = BASE / "pair_evt.csv"
    OUTPUT_PATH       = BASE / "features.csv"

    print("="*60)
    print("🚀 Honeypot Feature Extraction (robust pair-aware) Started")
    print("="*60)

    print("\n[1/4] Loading data...")
    token_events = load_token_events(TOKEN_EVENTS_PATH)
    pair_events  = load_pair_events(PAIR_EVENTS_PATH)

    all_token_ids = sorted(set(token_events.keys()) | set(pair_events.keys()))
    print(f"  ✅ Loaded {len(all_token_ids)} tokens")

    print("\n[2/4] Generating features...")
    feats: List[TokenFeature] = []

    for i, tid in enumerate(all_token_ids, 1):
        t_evts = token_events.get(tid, [])
        p_evts = pair_events.get(tid, [])

        s_owner = identify_s_owner(t_evts, p_evts)
        windows = generate_window_features(t_evts, p_evts, WINDOW_SECONDS, s_owner, token_id_for_log=tid)
        if windows:
            feats.append(aggregate_to_token_feature(tid, windows, s_owner))

        if i % 10 == 0 or i == len(all_token_ids):
            print(f"  Progress: {i}/{len(all_token_ids)} tokens processed", end="\r")

    print(f"\n  ✅ Generated features for {len(feats)} tokens")

    print("\n[3/4] Saving features...")
    fieldnames = [
        'token_addr_idx','consecutive_sell_fail_windows','total_buy_cnt','total_sell_cnt',
        'total_owner_sell_cnt','total_non_owner_sell_cnt','owner_sell_ratio','total_approval_cnt',
        'imbalance_rate','approval_to_sell_ratio','failed_sell_proxy_frac','max_sell_share',
        'privileged_event_flag','router_only_sell_proxy','total_windows','windows_with_activity',
        'total_burn_events','total_mint_events','avg_burn_frac','avg_reserve_drop','s_owner_count',
        'total_sell_vol','total_owner_sell_vol','owner_sell_vol_ratio','router_approval_rate'
    ]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames); w.writeheader()
        for ftr in feats: w.writerow({k:getattr(ftr,k) for k in fieldnames})
    print(f"  ✅ Saved → {OUTPUT_PATH}")

    print("\n[4/4] Statistics:")
    print(f"  - Total tokens: {len(feats)}")
    if feats:
        bc = [t.total_buy_cnt for t in feats]; sc = [t.total_sell_cnt for t in feats]
        orat = [t.owner_sell_ratio for t in feats]; sown = [t.s_owner_count for t in feats]
        print(f"  - Buy count range: {min(bc)} ~ {max(bc)}")
        print(f"  - Sell count range: {min(sc)} ~ {max(sc)}")
        print(f"  - Owner sell ratio: {min(orat):.2f} ~ {max(orat):.2f}")
        print(f"  - Avg S_owner count: {sum(sown)/len(sown):.1f}")

    print("\n" + "="*60)
    print("✅ Feature extraction (robust pair-aware) completed successfully!")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)
