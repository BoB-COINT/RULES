#!/usr/bin/env python3
"""Generate mint-and-dump detection features from token, pair, and holder data."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

getcontext().prec = 28

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
DEFAULT_WINDOW_SECONDS = 60
DEFAULT_MINT_LOOKBACK_SECONDS = 1800


@dataclass
class TokenInfo:
    token_addr_idx: str
    token_addr: Optional[str]
    pair_addr: Optional[str]


@dataclass
class HolderStats:
    top1_share: Optional[Decimal]
    pair_share: Optional[Decimal]
    top20_share: Optional[Decimal]
    max_share: Optional[Decimal]
    entropy: Optional[Decimal]
    total_supply: Optional[Decimal]


@dataclass
class MintEvent:
    timestamp: datetime
    amount: Decimal
    to_address: Optional[str]
    to_pair: bool


@dataclass
class PairEvent:
    timestamp: datetime
    evt_type: str
    base_in: Decimal = Decimal("0")
    base_out: Decimal = Decimal("0")
    quote_in: Decimal = Decimal("0")
    quote_out: Decimal = Decimal("0")
    reserve_base: Optional[Decimal] = None
    reserve_quote: Optional[Decimal] = None


@dataclass
class TokenState:
    reserve_base: Optional[Decimal] = None
    reserve_quote: Optional[Decimal] = None
    cumulative_mint_to_pair: Decimal = Decimal("0")
    last_mint_to_pair_ts: Optional[datetime] = None
    recent_mints: Deque[Tuple[datetime, Decimal]] = field(default_factory=deque)


@dataclass
class WindowAccumulator:
    mint_event_count: int = 0
    mint_total_amount: Decimal = Decimal("0")
    mint_to_pair_amount: Decimal = Decimal("0")
    dump_swap_count: int = 0
    dump_sell_volume: Decimal = Decimal("0")
    dump_quote_out_volume: Decimal = Decimal("0")
    dump_sell_ratio_sum: Decimal = Decimal("0")
    dump_sell_ratio_count: int = 0
    dump_sell_ratio_max: Decimal = Decimal("0")
    dump_sell_abs_max: Decimal = Decimal("0")
    dump_first_ts: Optional[datetime] = None
    dump_last_ts: Optional[datetime] = None

    def has_activity(self) -> bool:
        return (
            self.mint_event_count > 0
            or self.mint_to_pair_amount != 0
            or self.dump_swap_count > 0
        )


FIELDNAMES = [
    "win_id",
    "token_addr_idx",
    "win_start_ts",
    "holder_top1_supply_pct",
    "holder_pair_supply_pct",
    "holder_top20_supply_pct",
    "holder_single_max_pct",
    "holder_entropy",
    "holder_snapshot_lag_sec",
    "total_supply_estimate",
    "mint_event_count",
    "mint_total_amount",
    "mint_to_pair_amount",
    "mint_to_pair_supply_ratio",
    "mint_cum_to_pair_amount",
    "mint_last_to_pair_ts",
    "dump_swap_count",
    "dump_sell_volume",
    "dump_sell_tot_supply_ratio",
    "dump_sell_to_reserve_max_ratio",
    "dump_sell_to_reserve_avg_ratio",
    "dump_sell_abs_max",
    "dump_quote_out_volume",
    "dump_first_ts",
    "dump_last_ts",
    "dump_window_span_seconds",
    "mint_to_dump_latency_sec",
    "dump_to_mint_ratio",
]


def normalize_column_name(name: Optional[str]) -> str:
    if name is None:
        return ""
    return name.strip().lstrip("\ufeff")


def to_iso_z(value: Optional[datetime]) -> str:
    if value is None:
        return ""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_timestamp(raw: str) -> datetime:
    value = raw.strip()
    if not value:
        raise ValueError("timestamp is required")
    value = value.replace(" ", "T")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def floor_to_window(ts: datetime, window_seconds: int) -> datetime:
    epoch = int(ts.timestamp())
    floored = (epoch // window_seconds) * window_seconds
    return datetime.fromtimestamp(floored, tz=timezone.utc)


def parse_decimal(raw: Optional[str]) -> Optional[Decimal]:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid decimal value: {raw}") from exc


def parse_decimal_any(value: Optional[object]) -> Optional[Decimal]:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        if value == 0:
            return Decimal(0)
        return Decimal(str(value))
    if isinstance(value, str):
        return parse_decimal(value)
    return parse_decimal(str(value))


def decimal_to_str(value: Optional[Decimal]) -> str:
    if value is None:
        return ""
    normalized = value.normalize()
    s = format(normalized, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def float_to_str(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def normalize_address(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return value.lower()


def parse_event_log(raw: Optional[str]) -> Dict[str, object]:
    if raw is None:
        return {}
    text = raw.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return {}
        return parsed if isinstance(parsed, dict) else {}


def load_token_information(path: Path) -> Dict[str, TokenInfo]:
    result: Dict[str, TokenInfo] = {}
    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is not None:
            reader.fieldnames = [normalize_column_name(name) for name in reader.fieldnames]
        for row in reader:
            idx = (row.get("token_addr_idx") or "").strip()
            if not idx:
                raise ValueError("token_addr_idx is required in token_information.csv")
            token_addr = normalize_address(row.get("token_addr"))
            pair_addr = normalize_address(row.get("pair_addr"))
            result[idx] = TokenInfo(
                token_addr_idx=idx,
                token_addr=token_addr,
                pair_addr=pair_addr,
            )
    return result


def load_holders(path: Path) -> Dict[str, List[Tuple[Decimal, Decimal, str]]]:
    holders: Dict[str, List[Tuple[Decimal, Decimal, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            token_addr = normalize_address(row.get("token_addr"))
            if token_addr is None:
                continue
            balance = parse_decimal(row.get("balance")) or Decimal(0)
            share_raw = parse_decimal(row.get("rel_to_total")) or Decimal(0)
            share = share_raw
            if share > 1:
                share = share / Decimal(100)
            holders[token_addr].append(
                (balance, share, normalize_address(row.get("holder_addr")) or "")
            )
    return holders


def compute_holder_stats(
    entries: List[Tuple[Decimal, Decimal, str]],
    pair_addr: Optional[str],
) -> HolderStats:
    if not entries:
        return HolderStats(None, None, None, None, None, None)
    shares = [share for _, share, _ in entries if share is not None]
    if not shares:
        return HolderStats(None, None, None, None, None, None)
    sorted_shares = sorted(shares, reverse=True)
    top1 = sorted_shares[0] if sorted_shares else None
    top20_sum = sum(sorted_shares[:20]) if sorted_shares else None
    entropy_sum = Decimal("0")
    for share in shares:
        if share is None or share <= 0:
            continue
        entropy_sum -= share * Decimal(math.log(float(share)))
    entropy = entropy_sum if entropy_sum > 0 else None
    total_supply: Optional[Decimal] = None
    for balance, share, _ in entries:
        if share and share > 0:
            total_supply = balance / share
            break
    pair_share = Decimal(0) if pair_addr else None
    if pair_addr:
        for _, share, holder_addr in entries:
            if holder_addr == pair_addr:
                pair_share = share
                break
    max_share = top1
    return HolderStats(
        top1_share=top1,
        pair_share=pair_share,
        top20_share=top20_sum,
        max_share=max_share,
        entropy=entropy,
        total_supply=total_supply,
    )


def load_mint_events(path: Path, token_info: Dict[str, TokenInfo]) -> Dict[str, List[MintEvent]]:
    events: Dict[str, List[MintEvent]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is not None:
            reader.fieldnames = [normalize_column_name(name) for name in reader.fieldnames]
        for row in reader:
            token_idx = (row.get("token_addr_idx") or "").strip()
            if not token_idx:
                continue
            info = token_info.get(token_idx)
            if not info:
                continue
            timestamp_raw = row.get("timestamp")
            if not timestamp_raw:
                continue
            ts = parse_timestamp(timestamp_raw)
            evt_type = (row.get("evt_type") or "").strip().lower()
            if evt_type not in {"transfer", "mint"}:
                continue
            evt_log = parse_event_log(row.get("evt_log"))
            from_addr = normalize_address(evt_log.get("from") if isinstance(evt_log, dict) else None)
            amount = parse_decimal_any(evt_log.get("value") if isinstance(evt_log, dict) else None)
            if amount is None or amount <= 0:
                continue
            to_addr = normalize_address(evt_log.get("to") if isinstance(evt_log, dict) else None) or normalize_address(row.get("tx_to"))
            to_pair = bool(to_addr and info.pair_addr and to_addr == info.pair_addr)
            is_direct_mint = evt_type == "mint" or (evt_type == "transfer" and from_addr == ZERO_ADDRESS)
            if not is_direct_mint:
                continue

            events[token_idx].append(
                MintEvent(
                    timestamp=ts,
                    amount=amount,
                    to_address=to_addr,
                    to_pair=to_pair,
                )
            )
    for token_idx in events:
        events[token_idx].sort(key=lambda evt: evt.timestamp)
    return events


def load_pair_events(path: Path, token_info: Dict[str, TokenInfo]) -> Dict[str, List[PairEvent]]:
    events: Dict[str, List[PairEvent]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is not None:
            reader.fieldnames = [normalize_column_name(name) for name in reader.fieldnames]
        for row in reader:
            token_idx = (row.get("token_addr_idx") or "").strip()
            if not token_idx:
                continue
            info = token_info.get(token_idx)
            if not info:
                continue
            timestamp_raw = row.get("timestamp")
            if not timestamp_raw:
                continue
            ts = parse_timestamp(timestamp_raw)
            evt_type = (row.get("evt_type") or "").strip().lower()
            if not evt_type:
                continue
            token0 = normalize_address(row.get("token0"))
            token1 = normalize_address(row.get("token1"))
            base_is_token0 = True
            token_addr = info.token_addr
            if token_addr:
                if token0 == token_addr:
                    base_is_token0 = True
                elif token1 == token_addr:
                    base_is_token0 = False
            reserve0 = parse_decimal(row.get("reserve0"))
            reserve1 = parse_decimal(row.get("reserve1"))
            reserve_base = reserve0 if base_is_token0 else reserve1
            reserve_quote = reserve1 if base_is_token0 else reserve0
            evt_log = parse_event_log(row.get("evt_log"))
            if evt_type == "swap":
                amount0_in = parse_decimal_any(evt_log.get("amount0In")) or Decimal("0")
                amount0_out = parse_decimal_any(evt_log.get("amount0Out")) or Decimal("0")
                amount1_in = parse_decimal_any(evt_log.get("amount1In")) or Decimal("0")
                amount1_out = parse_decimal_any(evt_log.get("amount1Out")) or Decimal("0")
                base_in = amount0_in if base_is_token0 else amount1_in
                base_out = amount0_out if base_is_token0 else amount1_out
                quote_in = amount1_in if base_is_token0 else amount0_in
                quote_out = amount1_out if base_is_token0 else amount0_out
                events[token_idx].append(
                    PairEvent(
                        timestamp=ts,
                        evt_type="swap",
                        base_in=base_in,
                        base_out=base_out,
                        quote_in=quote_in,
                        quote_out=quote_out,
                        reserve_base=reserve_base,
                        reserve_quote=reserve_quote,
                    )
                )
            elif evt_type in {"sync", "mint", "burn"}:
                events[token_idx].append(
                    PairEvent(
                        timestamp=ts,
                        evt_type=evt_type,
                        reserve_base=reserve_base,
                        reserve_quote=reserve_quote,
                    )
                )
    for token_idx in events:
        events[token_idx].sort(key=lambda evt: evt.timestamp)
    return events


def trim_recent_mints(state: TokenState, current_time: datetime, lookback_seconds: int) -> None:
    cutoff = current_time - timedelta(seconds=lookback_seconds)
    while state.recent_mints and state.recent_mints[0][0] < cutoff:
        state.recent_mints.popleft()


def register_mint(event: MintEvent, state: TokenState, acc: WindowAccumulator, lookback_seconds: int) -> None:
    acc.mint_event_count += 1
    acc.mint_total_amount += event.amount
    if event.to_pair:
        acc.mint_to_pair_amount += event.amount
        state.cumulative_mint_to_pair += event.amount
        state.last_mint_to_pair_ts = event.timestamp
        state.recent_mints.append((event.timestamp, event.amount))
        trim_recent_mints(state, event.timestamp, lookback_seconds)


def register_pair_event(event: PairEvent, state: TokenState, acc: WindowAccumulator, lookback_seconds: int) -> None:
    if event.evt_type == "swap":
        base_in = event.base_in
        base_out = event.base_out
        quote_out = event.quote_out
        reserve_after = event.reserve_base
        reserve_before = None
        if reserve_after is not None:
            reserve_before = reserve_after - base_in + base_out
        elif state.reserve_base is not None:
            reserve_before = state.reserve_base
            reserve_after = state.reserve_base + base_in - base_out
        if base_in > 0 and quote_out > 0:
            acc.dump_swap_count += 1
            acc.dump_sell_volume += base_in
            acc.dump_quote_out_volume += quote_out
            if reserve_before is not None and reserve_before > 0:
                ratio = base_in / reserve_before
                acc.dump_sell_ratio_sum += ratio
                acc.dump_sell_ratio_count += 1
                if ratio > acc.dump_sell_ratio_max:
                    acc.dump_sell_ratio_max = ratio
            if base_in > acc.dump_sell_abs_max:
                acc.dump_sell_abs_max = base_in
            if acc.dump_first_ts is None or event.timestamp < acc.dump_first_ts:
                acc.dump_first_ts = event.timestamp
            if acc.dump_last_ts is None or event.timestamp > acc.dump_last_ts:
                acc.dump_last_ts = event.timestamp
        state.reserve_base = reserve_after
        if event.reserve_quote is not None:
            state.reserve_quote = event.reserve_quote
    else:
        reserve_before = state.reserve_base
        reserve_after = event.reserve_base
        if event.evt_type in {"sync", "mint"} and reserve_after is not None and reserve_before is not None:
            delta_base = reserve_after - reserve_before
            if delta_base > 0:
                acc.mint_event_count += 1
                acc.mint_to_pair_amount += delta_base
                state.cumulative_mint_to_pair += delta_base
                state.recent_mints.append((event.timestamp, delta_base))
        if event.reserve_base is not None:
            state.reserve_base = event.reserve_base
        if event.reserve_quote is not None:
            state.reserve_quote = event.reserve_quote
    if state.recent_mints:
        trim_recent_mints(state, event.timestamp, lookback_seconds)


def compute_recent_mint_total(state: TokenState, reference_time: datetime, lookback_seconds: int) -> Decimal:
    if not state.recent_mints:
        return Decimal("0")
    cutoff = reference_time - timedelta(seconds=lookback_seconds)
    total = Decimal("0")
    for ts, amount in state.recent_mints:
        if ts < cutoff or ts > reference_time:
            continue
        total += amount
    return total


def latest_mint_before(state: TokenState, reference_time: datetime) -> Optional[datetime]:
    for ts, _ in reversed(state.recent_mints):
        if ts <= reference_time:
            return ts
    return None


def finalize_window(
    token_idx: str,
    win_start: datetime,
    acc: WindowAccumulator,
    state: TokenState,
    holder_stats: HolderStats,
    snapshot_ts: Optional[datetime],
    lookback_seconds: int,
) -> Optional[Dict[str, object]]:
    if not acc.has_activity():
        return None
    total_supply = holder_stats.total_supply
    mint_to_pair_ratio: Optional[Decimal] = None
    if total_supply and total_supply > 0:
        mint_to_pair_ratio = acc.mint_to_pair_amount / total_supply
    elif acc.mint_to_pair_amount == 0:
        mint_to_pair_ratio = Decimal(0)
    dump_supply_ratio: Optional[Decimal] = None
    if total_supply and total_supply > 0:
        dump_supply_ratio = acc.dump_sell_volume / total_supply
    elif acc.dump_sell_volume == 0:
        dump_supply_ratio = Decimal(0)
    avg_ratio: Optional[Decimal] = None
    if acc.dump_sell_ratio_count > 0:
        avg_ratio = acc.dump_sell_ratio_sum / Decimal(acc.dump_sell_ratio_count)
    dump_span_seconds: Optional[float] = None
    if acc.dump_first_ts and acc.dump_last_ts:
        dump_span_seconds = (acc.dump_last_ts - acc.dump_first_ts).total_seconds()
    mint_to_dump_latency: Optional[float] = None
    dump_to_mint_ratio: Optional[Decimal] = None
    if acc.dump_first_ts:
        recent_mint_total = compute_recent_mint_total(state, acc.dump_first_ts, lookback_seconds)
        if recent_mint_total > 0:
            dump_to_mint_ratio = acc.dump_sell_volume / recent_mint_total
        last_mint_ts = latest_mint_before(state, acc.dump_first_ts)
        if last_mint_ts is not None:
            mint_to_dump_latency = (acc.dump_first_ts - last_mint_ts).total_seconds()
    holder_top1 = holder_stats.top1_share
    holder_pair = holder_stats.pair_share
    holder_top20 = holder_stats.top20_share
    holder_max = holder_stats.max_share
    holder_entropy = holder_stats.entropy
    snapshot_lag: Optional[float] = None
    if snapshot_ts is not None:
        snapshot_lag = abs((snapshot_ts - win_start).total_seconds())
    row: Dict[str, object] = {
        "token_addr_idx": token_idx,
        "win_start_ts": to_iso_z(win_start),
        "holder_top1_supply_pct": decimal_to_str(holder_top1),
        "holder_pair_supply_pct": decimal_to_str(holder_pair),
        "holder_top20_supply_pct": decimal_to_str(holder_top20),
        "holder_single_max_pct": decimal_to_str(holder_max),
        "holder_entropy": decimal_to_str(holder_entropy),
        "holder_snapshot_lag_sec": float_to_str(snapshot_lag),
        "total_supply_estimate": decimal_to_str(total_supply),
        "mint_event_count": acc.mint_event_count,
        "mint_total_amount": decimal_to_str(acc.mint_total_amount),
        "mint_to_pair_amount": decimal_to_str(acc.mint_to_pair_amount),
        "mint_to_pair_supply_ratio": decimal_to_str(mint_to_pair_ratio),
        "mint_cum_to_pair_amount": decimal_to_str(state.cumulative_mint_to_pair),
        "mint_last_to_pair_ts": to_iso_z(state.last_mint_to_pair_ts),
        "dump_swap_count": acc.dump_swap_count,
        "dump_sell_volume": decimal_to_str(acc.dump_sell_volume),
        "dump_sell_tot_supply_ratio": decimal_to_str(dump_supply_ratio),
        "dump_sell_to_reserve_max_ratio": decimal_to_str(acc.dump_sell_ratio_max if acc.dump_sell_ratio_count else None),
        "dump_sell_to_reserve_avg_ratio": decimal_to_str(avg_ratio),
        "dump_sell_abs_max": decimal_to_str(acc.dump_sell_abs_max),
        "dump_quote_out_volume": decimal_to_str(acc.dump_quote_out_volume),
        "dump_first_ts": to_iso_z(acc.dump_first_ts),
        "dump_last_ts": to_iso_z(acc.dump_last_ts),
        "dump_window_span_seconds": float_to_str(dump_span_seconds),
        "mint_to_dump_latency_sec": float_to_str(mint_to_dump_latency),
        "dump_to_mint_ratio": decimal_to_str(dump_to_mint_ratio),
    }
    return row


def generate_token_features(
    token_idx: str,
    holder_stats: HolderStats,
    mint_events: Sequence[MintEvent],
    pair_events: Sequence[PairEvent],
    window_seconds: int,
    lookback_seconds: int,
    snapshot_ts: Optional[datetime],
    start_win_id: int,
) -> Tuple[List[Dict[str, object]], int]:
    combined: List[Tuple[datetime, str, object]] = []
    for mint in mint_events:
        combined.append((mint.timestamp, "mint", mint))
    for pevt in pair_events:
        combined.append((pevt.timestamp, "pair", pevt))
    if not combined:
        return [], start_win_id
    combined.sort(key=lambda item: (item[0], 0 if item[1] == "mint" else 1))
    window_start = floor_to_window(combined[0][0], window_seconds)
    window_delta = timedelta(seconds=window_seconds)
    window_end = window_start + window_delta
    acc = WindowAccumulator()
    state = TokenState()
    features: List[Dict[str, object]] = []
    next_win_id = start_win_id
    for timestamp, kind, payload in combined:
        while timestamp >= window_end:
            row = finalize_window(
                token_idx,
                window_start,
                acc,
                state,
                holder_stats,
                snapshot_ts,
                lookback_seconds,
            )
            if row is not None:
                row["win_id"] = next_win_id
                next_win_id += 1
                features.append(row)
            acc = WindowAccumulator()
            window_start = window_end
            window_end = window_start + window_delta
        if kind == "mint":
            register_mint(payload, state, acc, lookback_seconds)
        else:
            register_pair_event(payload, state, acc, lookback_seconds)
    row = finalize_window(
        token_idx,
        window_start,
        acc,
        state,
        holder_stats,
        snapshot_ts,
        lookback_seconds,
    )
    if row is not None:
        row["win_id"] = next_win_id
        next_win_id += 1
        features.append(row)
    return features, next_win_id


def write_features_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            output = {}
            for field in FIELDNAMES:
                value = row.get(field, "")
                output[field] = value if isinstance(value, (int, float, str)) else value
            writer.writerow(output)


def parse_snapshot_timestamp(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return parse_timestamp(stripped)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate mint-and-dump detection features")
    parser.add_argument("--token-info", required=True, help="Path to token_information.csv")
    parser.add_argument("--token-events", required=True, help="Path to token_evt.csv")
    parser.add_argument("--pair-events", required=True, help="Path to pair_evt.csv")
    parser.add_argument("--holders", required=True, help="Path to holders.csv")
    parser.add_argument("--output", required=True, help="Path to write features CSV")
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=DEFAULT_WINDOW_SECONDS,
        help="Aggregation window size in seconds (default: 60)",
    )
    parser.add_argument(
        "--mint-lookback-seconds",
        type=int,
        default=DEFAULT_MINT_LOOKBACK_SECONDS,
        help="Lookback window for mint aggregation when comparing against dumps (default: 1800)",
    )
    parser.add_argument(
        "--holders-snapshot-ts",
        default=None,
        help="Optional ISO timestamp representing when holders.csv snapshot was captured",
    )

    args = parser.parse_args(argv)

    token_info_map = load_token_information(Path(args.token_info))
    holder_entries_map = load_holders(Path(args.holders))
    mint_events_map = load_mint_events(Path(args.token_events), token_info_map)
    pair_events_map = load_pair_events(Path(args.pair_events), token_info_map)

    snapshot_ts = parse_snapshot_timestamp(args.holders_snapshot_ts)

    all_tokens = sorted(set(mint_events_map.keys()) | set(pair_events_map.keys()))
    features: List[Dict[str, object]] = []
    next_win_id = 0

    for token_idx in all_tokens:
        info = token_info_map.get(token_idx)
        if not info:
            continue
        holder_entries = holder_entries_map.get(info.token_addr or "", [])
        holder_stats = compute_holder_stats(holder_entries, info.pair_addr)
        token_features, next_win_id = generate_token_features(
            token_idx,
            holder_stats,
            mint_events_map.get(token_idx, []),
            pair_events_map.get(token_idx, []),
            args.window_seconds,
            args.mint_lookback_seconds,
            snapshot_ts,
            next_win_id,
        )
        features.extend(token_features)

    write_features_csv(Path(args.output), features)


if __name__ == "__main__":
    main()
