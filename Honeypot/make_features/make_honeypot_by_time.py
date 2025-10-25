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
KNOWN_ROUTER_ADDRS = {
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Uniswap V2
    "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",  # Uniswap V3
    "0x1111111254fb6c44bac0bed2854e76f90643097d",  # 1inch
    "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f",  # Sushi
}
ZERO_ADDR = "0x0000000000000000000000000000000000000000"
WINDOW_SECONDS = 5     # 5초 창
DEBUG = False

# -------------------- 데이터 구조 --------------------
@dataclass
class TokenEvent:
    token_addr_idx: str
    timestamp: datetime
    tx_hash: str
    block_number: int
    evt_type: str              # "transfer" | "approval" | ...
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
    evt_type: str              # "mint" | "burn" | "sync" | "swap" | "paircreated"
    reserve0: Optional[Decimal]
    reserve1: Optional[Decimal]
    total_supply: Optional[Decimal]
    amount0_in: Optional[Decimal]
    amount1_in: Optional[Decimal]
    amount0_out: Optional[Decimal]
    amount1_out: Optional[Decimal]
    sender: Optional[str]
    to: Optional[str]
    pair_addr: Optional[str]   # 로더에서 주입

# -------------------- 유틸 --------------------
def parse_iso(v: str) -> datetime:
    if not v: raise ValueError("Empty timestamp")
    v = v.strip()
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

def floor_to_window(dt: datetime, window_seconds: int) -> datetime:
    epoch = int(dt.timestamp())
    return datetime.fromtimestamp(epoch - (epoch % window_seconds), tz=timezone.utc)

# -------------------- S_owner 추정 --------------------
def identify_s_owner(token_events: List[TokenEvent], pair_events: List[PairEvent]) -> Set[str]:
    s: Set[str] = set()
    token_sorted = sorted(token_events, key=lambda e: (e.block_number, e.timestamp))
    first_swap_ts = min((e.timestamp for e in pair_events if e.evt_type == "swap"), default=None)
    # 최초 민팅 수령자 (swap 이전)
    for evt in token_sorted:
        if evt.evt_type == "transfer" and evt.tx_from == ZERO_ADDR and evt.tx_to:
            if first_swap_ts and evt.timestamp > first_swap_ts: break
            s.add(evt.tx_to.lower())
            if len(s) >= 10: break
    # LP 민팅/소각 참여자
    for evt in pair_events:
        if evt.evt_type in ("mint", "burn"):
            if evt.sender and evt.sender != ZERO_ADDR: s.add(evt.sender.lower())
            if evt.to and evt.to != ZERO_ADDR: s.add(evt.to.lower())
    # 초기 Approval 보낸 주소(라우터/0 제외)
    approvals = [e for e in token_sorted if e.evt_type == "approval" and e.tx_from]
    if approvals:
        approvals.sort(key=lambda e: (e.block_number, e.timestamp))
        first_approver = approvals[0].tx_from.lower()
        if first_approver not in KNOWN_ROUTER_ADDRS and first_approver != ZERO_ADDR:
            s.add(first_approver)
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
                tx_from = (row.get("tx_from") or "").lower().strip()
                tx_to   = (row.get("tx_to")   or "").lower().strip()
                val = None
                spender = None
                if evt_type == "transfer" and isinstance(data, dict):
                    tx_from = (data.get("from") or "").lower().strip() or tx_from
                    tx_to   = (data.get("to")   or "").lower().strip() or tx_to
                    if "value" in data:
                        try: val = Decimal(str(data["value"]))
                        except Exception: val = None
                    else:
                        val = extract_value_from_evtlog(raw_log)
                elif evt_type == "approval" and isinstance(data, dict):
                    owner = (data.get("owner") or data.get("from") or "").lower().strip()
                    sp    = (data.get("spender") or data.get("to") or "").lower().strip()
                    if owner: tx_from = owner
                    if sp: spender = sp
                    if "value" in data:
                        try: val = Decimal(str(data["value"]))
                        except Exception: val = None
                    else:
                        val = extract_value_from_evtlog(raw_log)
                else:
                    if raw_log: val = extract_value_from_evtlog(raw_log)
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
                    success=True,  # 실패 정보 미제공 시 True
                    spender=spender
                )
                m.setdefault(token_idx, []).append(evt)
            except Exception:
                errs += 1
                continue
    if errs: print(f"[INFO] token_evt.csv parse errors: {errs}")
    return m

def load_pair_events(path: Path) -> Dict[str, List[PairEvent]]:
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
                pe = PairEvent(
                    token_addr_idx=token_idx,
                    timestamp=parse_iso(row.get("timestamp")),
                    tx_hash=row.get("tx_hash",""),
                    evt_type=evt_type,
                    reserve0=parse_decimal(row.get("reserve0")),
                    reserve1=parse_decimal(row.get("reserve1")),
                    total_supply=parse_decimal(row.get("total_supply") or row.get("lp_total_supply")),
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
    for token_idx, evts in temp.items():
        plist = pairaddr_by_token.get(token_idx, [])
        rep = plist[0] if plist else None
        for e in evts:
            e.pair_addr = rep
        m[token_idx] = evts
    return m

# -------------------- 페어 주소 추정 --------------------
def infer_pair_addrs_from_transfers(token_evts: List[TokenEvent]) -> Set[str]:
    from_cnt: Dict[str,int] = {}
    to_cnt: Dict[str,int]   = {}
    for e in token_evts:
        if e.evt_type!="transfer": continue
        if e.tx_from: from_cnt[e.tx_from] = from_cnt.get(e.tx_from,0)+1
        if e.tx_to:   to_cnt[e.tx_to]     = to_cnt.get(e.tx_to,0)+1
    cands = set(from_cnt.keys()) & set(to_cnt.keys())
    if len(cands) > 12:
        def topk(d, k=12):
            return {k_ for k_, v in sorted(d.items(), key=lambda x:-x[1])[:k]}
        cands = topk(from_cnt) & topk(to_cnt)
    return {c for c in cands if c != ZERO_ADDR}

def build_pair_addr_set(pair_evts: List[PairEvent], token_evts: List[TokenEvent]) -> Set[str]:
    pair_addr_set: Set[str] = { (p.pair_addr or "") for p in pair_evts if p.pair_addr }
    pair_addr_set = {x.lower() for x in pair_addr_set if x}
    if len(pair_addr_set) == 0:
        pair_addr_set |= infer_pair_addrs_from_transfers(token_evts)
    return pair_addr_set

# -------------------- 시계열(윈도우) 생성 --------------------
def generate_timeseries_rows_for_token(
    token_idx: str,
    token_evts: List[TokenEvent],
    pair_evts: List[PairEvent],
    window_seconds: int,
) -> List[Dict[str, object]]:
    """
    출력: 윈도우별 행(dict). 각 행은 아래 컬럼 보유:
      win_id, win_start_ts, token_addr_idx  +  (요청하신 토큰-단위 컬럼들의 '시점까지 누적/파생치')
    """
    # S_owner, 페어 주소 집합
    s_owner = identify_s_owner(token_evts, pair_evts)
    pair_addr_set = build_pair_addr_set(pair_evts, token_evts)

    # 타임라인 결정
    all_events = [(e.timestamp,"token",e) for e in token_evts] + [(e.timestamp,"pair",e) for e in pair_evts]
    if not all_events:
        return []
    all_events.sort(key=lambda x: (x[0], x[1]))
    t0 = floor_to_window(all_events[0][0], window_seconds)
    tN = floor_to_window(all_events[-1][0], window_seconds)
    step = timedelta(seconds=window_seconds)

    # 누적 상태(토큰-단위 의미 유지)
    total_buy_cnt = 0
    total_sell_cnt = 0
    total_owner_sell_cnt = 0
    total_non_owner_sell_cnt = 0
    total_approval_cnt = 0
    windows_with_activity = 0
    total_burn_events = 0
    total_mint_events = 0
    s_owner_count = len(s_owner)
    total_sell_vol_dec = Decimal(0)
    total_owner_sell_vol_dec = Decimal(0)
    approval_to_router = 0
    liquidity_event_mask = 0          # bit0: mint, bit1: burn, bit2: sync
    consecutive_fail_run = 0
    failed_sell_cnt = 0

    # 매도자 분포(점유율 계산용)
    seller_cnt: Dict[str, int] = {}

    # 불균형 러닝 평균용
    imb_sum = Decimal(0)
    imb_seen = 0

    # 윈도우 루프를 위해 이벤트 인덱스 준비
    tok_sorted = sorted(token_evts, key=lambda e: e.timestamp)
    pair_sorted = sorted(pair_evts, key=lambda e: e.timestamp)
    it_tok = 0
    it_pair = 0

    rows: List[Dict[str, object]] = []
    win_id = 0
    cur = t0
    while cur <= tN:
        nxt = cur + step

        # 창내 카운트
        buy_cnt = sell_cnt = owner_sell_cnt = non_owner_sell_cnt = 0
        approval_cnt = 0
        buy_vol = Decimal(0)
        sell_vol = Decimal(0)
        approve_router_win = 0
        burn_events = mint_events = sync_events = 0

        # token events
        while it_tok < len(tok_sorted) and tok_sorted[it_tok].timestamp < nxt:
            e = tok_sorted[it_tok]
            if e.timestamp >= cur:
                if e.evt_type == "transfer":
                    frm = e.tx_from; to = e.tx_to
                    if frm in pair_addr_set and to and to not in pair_addr_set:
                        buy_cnt += 1
                        if e.value is not None: buy_vol += e.value
                    elif to in pair_addr_set and frm and frm not in pair_addr_set:
                        sell_cnt += 1
                        if e.value is not None: sell_vol += e.value
                        # owner / non-owner
                        if frm in s_owner:
                            owner_sell_cnt += 1
                            if e.value is not None:
                                total_owner_sell_vol_dec += e.value
                            seller_cnt[frm] = seller_cnt.get(frm, 0) + 1
                        else:
                            non_owner_sell_cnt += 1
                            seller_cnt[frm] = seller_cnt.get(frm, 0) + 1
                elif e.evt_type == "approval":
                    approval_cnt += 1
                    sp = (e.spender or "").lower() if e.spender else None
                    if sp and sp in KNOWN_ROUTER_ADDRS:
                        approve_router_win += 1
            it_tok += 1

        # pair events
        while it_pair < len(pair_sorted) and pair_sorted[it_pair].timestamp < nxt:
            pe = pair_sorted[it_pair]
            if pe.timestamp >= cur:
                if pe.evt_type == "mint":
                    mint_events += 1
                    liquidity_event_mask |= 1
                elif pe.evt_type == "burn":
                    burn_events += 1
                    liquidity_event_mask |= 2
                elif pe.evt_type == "sync":
                    sync_events += 1
                    liquidity_event_mask |= 4
            it_pair += 1

        # 창별 활동 판단
        active = (buy_cnt + sell_cnt + approval_cnt + burn_events + mint_events + sync_events) > 0
        if active:
            windows_with_activity += 1

        # 누적 갱신
        total_buy_cnt += buy_cnt
        total_sell_cnt += sell_cnt
        total_owner_sell_cnt += owner_sell_cnt
        total_non_owner_sell_cnt += non_owner_sell_cnt
        total_approval_cnt += approval_cnt
        total_burn_events += burn_events
        total_mint_events += mint_events
        total_sell_vol_dec += sell_vol

        # 실패 플래그(창 단위) & 누적 실패 카운트/연속 길이
        fail_flag = (buy_cnt > 0) and (non_owner_sell_cnt == 0) and (approval_cnt > 0)
        if fail_flag:
            consecutive_fail_run += 1
            failed_sell_cnt += 1
        else:
            consecutive_fail_run = 0

        approval_to_router += approve_router_win

        # 파생 비율/평균
        owner_sell_ratio = (total_owner_sell_cnt / total_sell_cnt) if total_sell_cnt > 0 else 0.0
        owner_sell_vol_ratio = float((total_owner_sell_vol_dec / total_sell_vol_dec) if total_sell_vol_dec > 0 else 0)
        approval_to_sell_ratio = (total_approval_cnt / total_non_owner_sell_cnt) if total_non_owner_sell_cnt > 0 else 99.0
        router_approval_rate = (approval_to_router / total_approval_cnt) if total_approval_cnt > 0 else 0.0

        # 불균형(창별) → 러닝 평균
        if (buy_vol + sell_vol) > 0:
            imb_win = (buy_vol - sell_vol) / (buy_vol + sell_vol)
            imb_sum += imb_win
            imb_seen += 1
        imbalance_rate = float(imb_sum / imb_seen) if imb_seen > 0 else 0.0

        # max_sell_share (지금까지의 매도 횟수 점유율 최대)
        if total_sell_cnt > 0 and seller_cnt:
            max_sell_share = max(seller_cnt.values()) / total_sell_cnt
        else:
            max_sell_share = 0.0

        # 행 기록 (요청 컬럼명 유지 + 시계열 키 2개 추가)
        rows.append({
            "win_id": win_id,
            "win_start_ts": cur.isoformat().replace("+00:00", "Z"),
            "token_addr_idx": token_idx,

            "consecutive_sell_fail_windows": consecutive_fail_run,
            "windows_with_activity": windows_with_activity,
            "total_windows": (win_id + 1),                  # 지금까지 생성된 윈도 수
            "failed_sell_cnt": failed_sell_cnt,
            "total_non_owner_sell_cnt": total_non_owner_sell_cnt,
            "liquidity_event_mask": liquidity_event_mask,
            "approval_to_sell_ratio": float(approval_to_sell_ratio),
            "total_approval_cnt": total_approval_cnt,
            "total_sell_cnt": total_sell_cnt,
            "router_approval_rate": float(router_approval_rate),
            "imbalance_rate": float(imbalance_rate),
            "total_buy_cnt": total_buy_cnt,
            "max_sell_share": float(max_sell_share),
            "total_owner_sell_cnt": total_owner_sell_cnt,
            "s_owner_count": s_owner_count,
            "owner_sell_ratio": float(owner_sell_ratio),
            "owner_sell_vol_ratio": float(owner_sell_vol_ratio),
            "total_sell_vol": float(total_sell_vol_dec),
            "total_owner_sell_vol": float(total_owner_sell_vol_dec),
        })

        win_id += 1
        cur = nxt

    return rows

# -------------------- 메인 --------------------
def main():
    BASE = Path(".")
    TOKEN_EVENTS_PATH = BASE / "token_evt.csv"   # Transfer/Approval 등 (token 컨트랙트 로그)
    PAIR_EVENTS_PATH  = BASE / "pair_evt.csv"    # Mint/Burn/Sync/Swap 등 (pair 로그)
    OUTPUT_PATH       = BASE / "features_timeseries.csv"

    print("="*60)
    print("🚀 Honeypot Timeseries Feature (v11 semantics, windowed) Started")
    print("="*60)

    print("\n[1/3] Loading data...")
    token_events = load_token_events(TOKEN_EVENTS_PATH)
    pair_events  = load_pair_events(PAIR_EVENTS_PATH)
    token_ids = sorted(set(token_events.keys()) | set(pair_events.keys()))
    print(f"  ✅ Loaded {len(token_ids)} tokens")

    print("\n[2/3] Building windowed timeseries...")
    all_rows: List[Dict[str, object]] = []
    for i, tid in enumerate(token_ids, 1):
        t_evts = token_events.get(tid, [])
        p_evts = pair_events.get(tid, [])
        rows = generate_timeseries_rows_for_token(tid, t_evts, p_evts, WINDOW_SECONDS)
        all_rows.extend(rows)
        if i % 10 == 0 or i == len(token_ids):
            print(f"  Progress: {i}/{len(token_ids)} tokens processed", end="\r")

    print(f"\n  ✅ Generated {len(all_rows)} window rows")

    print("\n[3/3] Saving timeseries...")
    fieldnames = [
        # 시계열 키
        "win_id","win_start_ts",
        # 유지 요청 컬럼 (토큰-단위 의미를 창 시점까지의 누적으로 표현)
        "token_addr_idx",
        "consecutive_sell_fail_windows",
        "windows_with_activity",
        "total_windows",
        "failed_sell_cnt",
        "total_non_owner_sell_cnt",
        "liquidity_event_mask",
        "approval_to_sell_ratio",
        "total_approval_cnt",
        "total_sell_cnt",
        "router_approval_rate",
        "imbalance_rate",
        "total_buy_cnt",
        "max_sell_share",
        "total_owner_sell_cnt",
        "s_owner_count",
        "owner_sell_ratio",
        "owner_sell_vol_ratio",
        "total_sell_vol",
        "total_owner_sell_vol",
    ]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"  ✅ Saved → {OUTPUT_PATH}")
    print("\n" + "="*60)
    print("✅ Timeseries generation completed successfully!")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)
