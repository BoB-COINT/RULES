#!/usr/bin/env python3
"""Generate 5-second window features from token information and on-chain event data."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

getcontext().prec = 28

WINDOW_SECONDS_DEFAULT = 5


@dataclass
class TokenInfo:
    token_addr_idx: str
    token_addr: Optional[str]
    pair_addr: Optional[str]


@dataclass
class TokenEvent:
    token_addr_idx: str
    timestamp: datetime
    tx_hash: Optional[str]
    block_number: Optional[int]
    evt_idx: Optional[int]
    evt_type: Optional[str]
    tx_from: Optional[str]
    tx_to: Optional[str]
    reserve_in_pair: Optional[Decimal]
    total_supply_in_pair: Optional[Decimal]
    mint_base_amount: Optional[Decimal] = None
    mint_quote_amount: Optional[Decimal] = None
    burn_base_amount: Optional[Decimal] = None
    burn_quote_amount: Optional[Decimal] = None
    swap_base_in: Optional[Decimal] = None
    swap_base_out: Optional[Decimal] = None
    swap_quote_in: Optional[Decimal] = None
    swap_quote_out: Optional[Decimal] = None


@dataclass
class FeatureRow:
    win_id: int
    token_addr_idx: str
    win_start_ts: str
    win_start_block: Optional[int]
    win_tx_count: int
    win_blocks: Optional[int]
    lp_start: Optional[str]
    lp_end: Optional[str]
    lp_drop_amount: Optional[str]
    burn_frac: Optional[str]
    reserve_token_start: Optional[str]
    reserve_token_end: Optional[str]
    reserve_token_drop_frac: Optional[str]
    lp_mint_amount: Optional[str]
    lp_burn_amount: Optional[str]
    mint_events: int
    burn_events: int
    swap_events: int
    burn_to_mint_ratio: Optional[str]
    time_since_last_mint_sec: Optional[str]
    lp_peak_drop_frac: Optional[str]
    swap_base_sell_volume: Optional[str]
    swap_base_buy_volume: Optional[str]
    cum_base_minted: Optional[str]
    cum_base_burned: Optional[str]
    cum_quote_minted: Optional[str]
    cum_quote_burned: Optional[str]
    lp_start_peak_frac: Optional[str]


def parse_iso_timestamp(value: str) -> datetime:
    value = value.strip()
    if not value:
        raise ValueError("Empty timestamp value")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return int(value)


def parse_decimal(value: Optional[str]) -> Optional[Decimal]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid decimal value: {value}") from exc


def parse_decimal_any(value: Optional[object]) -> Optional[Decimal]:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        if value == 0:
            return Decimal(0)
        return Decimal(str(value))
    if isinstance(value, str):
        return parse_decimal(value)
    return parse_decimal(str(value))


def floor_to_window(dt: datetime, window_seconds: int) -> datetime:
    epoch_seconds = dt.timestamp()
    window = int(window_seconds)
    floored = int(epoch_seconds // window) * window
    return datetime.fromtimestamp(floored, tz=timezone.utc)


def decimal_to_str(value: Optional[Decimal]) -> Optional[str]:
    if value is None:
        return None
    return format(value.normalize(), 'f')


def normalize_column_name(name: Optional[str]) -> str:
    """Return a sanitized column name free of BOM/whitespace artifacts."""

    if name is None:
        return ""
    return name.strip().lstrip("\ufeff")


def load_token_information(path: Path) -> Tuple[Dict[str, TokenInfo], Sequence[str], Set[str]]:
    token_map: Dict[str, TokenInfo] = {}
    headers: Sequence[str] = ()
    used_columns: Set[str] = set()

    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is not None:
            reader.fieldnames = [normalize_column_name(name) for name in reader.fieldnames]
        headers = tuple(reader.fieldnames or ())
        for row in reader:
            idx_raw = (row.get("token_addr_idx") or "").strip()
            if not idx_raw:
                raise ValueError("token_addr_idx is required in Token_Information")
            used_columns.add("token_addr_idx")

            token_addr_raw = row.get("token_addr")
            token_addr = (token_addr_raw or "").strip() or None
            if token_addr_raw is not None:
                used_columns.add("token_addr")

            pair_addr_raw = row.get("pair_addr")
            pair_addr = (pair_addr_raw or "").strip() or None
            if pair_addr_raw is not None:
                used_columns.add("pair_addr")

            token_map[idx_raw] = TokenInfo(
                token_addr_idx=idx_raw,
                token_addr=token_addr,
                pair_addr=pair_addr,
            )

    return token_map, headers, used_columns


def load_token_events(
    path: Path,
    token_info_map: Optional[Dict[str, TokenInfo]] = None,
) -> Tuple[Dict[str, List[TokenEvent]], Sequence[str], Set[str]]:
    token_events: Dict[str, List[TokenEvent]] = {}
    headers: Sequence[str] = ()
    used_columns: Set[str] = set()

    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is not None:
            reader.fieldnames = [normalize_column_name(name) for name in reader.fieldnames]
        headers = tuple(reader.fieldnames or ())
        for row in reader:
            if not any((value or "").strip() for value in row.values()):
                continue
            idx_raw = (row.get("token_addr_idx") or "").strip()
            if not idx_raw:
                raise ValueError("token_addr_idx is required in Token_onData")
            used_columns.add("token_addr_idx")

            timestamp_raw = row.get("timestamp")
            if timestamp_raw is None:
                raise ValueError("timestamp is required in Token_onData")
            timestamp = parse_iso_timestamp(timestamp_raw)
            used_columns.add("timestamp")

            tx_hash_raw = row.get("tx_hash")
            tx_hash = (tx_hash_raw or "").strip() or None
            if tx_hash_raw is not None:
                used_columns.add("tx_hash")

            block_number_raw = row.get("block_number")
            block_number = parse_int(block_number_raw)
            if block_number_raw is not None:
                used_columns.add("block_number")

            evt_idx_raw = row.get("evt_idx")
            evt_idx = parse_int(evt_idx_raw)
            if evt_idx_raw is not None:
                used_columns.add("evt_idx")

            evt_type_raw = row.get("evt_type")
            evt_type = (evt_type_raw or "").strip() or None
            if evt_type_raw is not None:
                used_columns.add("evt_type")

            tx_from_raw = row.get("tx_from")
            tx_from = (tx_from_raw or "").strip() or None
            if tx_from_raw is not None:
                used_columns.add("tx_from")

            tx_to_raw = row.get("tx_to")
            tx_to = (tx_to_raw or "").strip() or None
            if tx_to_raw is not None:
                used_columns.add("tx_to")

            reserve_in_pair: Optional[Decimal] = None
            reserve_column = "currrent_reserve_in_pair"
            reserve_raw = row.get(reserve_column)
            if reserve_raw is None:
                reserve_column = "current_reserve_in_pair"
                reserve_raw = row.get(reserve_column)
            if reserve_raw is not None:
                reserve_in_pair = parse_decimal(reserve_raw)
                used_columns.add(reserve_column)

            total_supply_column = "current_total_supply_in_pair"
            total_supply_raw = row.get(total_supply_column)
            if total_supply_raw is None:
                total_supply_column = "lp_total_supply"
                total_supply_raw = row.get(total_supply_column)
            total_supply_in_pair = parse_decimal(total_supply_raw)
            if total_supply_raw is not None:
                used_columns.add(total_supply_column)

            token0 = (row.get("token0") or "").strip()
            token1 = (row.get("token1") or "").strip()
            if token0:
                used_columns.add("token0")
            if token1:
                used_columns.add("token1")

            token0_lower = token0.lower()
            token1_lower = token1.lower()

            token_info = token_info_map.get(idx_raw) if token_info_map else None
            token_addr = (token_info.token_addr or "").strip().lower() if token_info else ""

            if reserve_in_pair is None:
                reserve0_raw = row.get("reserve0")
                reserve1_raw = row.get("reserve1")
                if reserve0_raw is not None:
                    used_columns.add("reserve0")
                if reserve1_raw is not None:
                    used_columns.add("reserve1")

                reserve0 = parse_decimal(reserve0_raw)
                reserve1 = parse_decimal(reserve1_raw)

                token0_norm = token0_lower
                token1_norm = token1_lower

                if token_addr and token_addr == token0_norm:
                    reserve_in_pair = reserve0
                elif token_addr and token_addr == token1_norm:
                    reserve_in_pair = reserve1
                elif reserve1 is not None:
                    reserve_in_pair = reserve1
                else:
                    reserve_in_pair = reserve0

            if reserve_in_pair is None and (row.get("reserve0") is not None or row.get("reserve1") is not None):
                raise ValueError(
                    "Unable to determine reserve_in_pair from reserve0/reserve1 columns for token_addr_idx "
                    f"{idx_raw}."
                )

            is_base_token0 = False
            is_base_token1 = False
            if token_addr:
                if token_addr == token0_lower:
                    is_base_token0 = True
                elif token_addr == token1_lower:
                    is_base_token1 = True
            if not is_base_token0 and not is_base_token1:
                is_base_token0 = True

            evt_log_raw = row.get("evt_log")
            evt_payload: Dict[str, object] = {}
            if evt_log_raw is not None:
                used_columns.add("evt_log")
                evt_log_text = evt_log_raw.strip()
                if evt_log_text:
                    try:
                        evt_payload = json.loads(evt_log_text)
                    except json.JSONDecodeError:
                        try:
                            evt_payload = ast.literal_eval(evt_log_text)
                        except (ValueError, SyntaxError):
                            evt_payload = {}
                if not isinstance(evt_payload, dict):
                    evt_payload = {}

            mint_base_amount: Optional[Decimal] = None
            mint_quote_amount: Optional[Decimal] = None
            burn_base_amount: Optional[Decimal] = None
            burn_quote_amount: Optional[Decimal] = None
            swap_base_in: Optional[Decimal] = None
            swap_base_out: Optional[Decimal] = None
            swap_quote_in: Optional[Decimal] = None
            swap_quote_out: Optional[Decimal] = None

            evt_type_lower = (evt_type or "").lower()
            if evt_payload:
                if evt_type_lower == "mint":
                    amount0 = parse_decimal_any(evt_payload.get("amount0"))
                    amount1 = parse_decimal_any(evt_payload.get("amount1"))
                    if is_base_token0:
                        mint_base_amount = amount0
                        mint_quote_amount = amount1
                    else:
                        mint_base_amount = amount1
                        mint_quote_amount = amount0
                elif evt_type_lower == "burn":
                    amount0 = parse_decimal_any(evt_payload.get("amount0"))
                    amount1 = parse_decimal_any(evt_payload.get("amount1"))
                    if is_base_token0:
                        burn_base_amount = amount0
                        burn_quote_amount = amount1
                    else:
                        burn_base_amount = amount1
                        burn_quote_amount = amount0
                elif evt_type_lower == "swap":
                    amount0_in = parse_decimal_any(evt_payload.get("amount0In"))
                    amount0_out = parse_decimal_any(evt_payload.get("amount0Out"))
                    amount1_in = parse_decimal_any(evt_payload.get("amount1In"))
                    amount1_out = parse_decimal_any(evt_payload.get("amount1Out"))
                    if is_base_token0:
                        swap_base_in = amount0_in
                        swap_base_out = amount0_out
                        swap_quote_in = amount1_in
                        swap_quote_out = amount1_out
                    else:
                        swap_base_in = amount1_in
                        swap_base_out = amount1_out
                        swap_quote_in = amount0_in
                        swap_quote_out = amount0_out

            event = TokenEvent(
                token_addr_idx=idx_raw,
                timestamp=timestamp,
                tx_hash=tx_hash,
                block_number=block_number,
                evt_idx=evt_idx,
                evt_type=evt_type,
                tx_from=tx_from,
                tx_to=tx_to,
                reserve_in_pair=reserve_in_pair,
                total_supply_in_pair=total_supply_in_pair,
                mint_base_amount=mint_base_amount,
                mint_quote_amount=mint_quote_amount,
                burn_base_amount=burn_base_amount,
                burn_quote_amount=burn_quote_amount,
                swap_base_in=swap_base_in,
                swap_base_out=swap_base_out,
                swap_quote_in=swap_quote_in,
                swap_quote_out=swap_quote_out,
            )
            token_events.setdefault(idx_raw, []).append(event)

    return token_events, headers, used_columns


def sort_events(events: Iterable[TokenEvent]) -> List[TokenEvent]:
    return sorted(
        events,
        key=lambda e: (
            e.timestamp,
            e.block_number if e.block_number is not None else float("inf"),
            e.evt_idx if e.evt_idx is not None else 0,
        ),
    )


def to_iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def generate_token_features(
    events: Sequence[TokenEvent],
    window_seconds: int,
    win_id_start: int,
) -> Tuple[List[FeatureRow], int]:
    if not events:
        return [], win_id_start

    ordered_events = sort_events(events)
    first_window_start = floor_to_window(ordered_events[0].timestamp, window_seconds)
    last_window_start = floor_to_window(ordered_events[-1].timestamp, window_seconds)
    window_delta = timedelta(seconds=window_seconds)

    features: List[FeatureRow] = []
    win_id = win_id_start

    last_lp: Optional[Decimal] = None
    last_reserve: Optional[Decimal] = None
    last_block: Optional[int] = None
    last_mint_ts: Optional[datetime] = None
    lp_peak: Optional[Decimal] = None

    cumulative_base_minted = Decimal(0)
    cumulative_base_burned = Decimal(0)
    cumulative_quote_minted = Decimal(0)
    cumulative_quote_burned = Decimal(0)

    event_idx = 0
    total_events = len(ordered_events)
    window_start = first_window_start

    while window_start <= last_window_start:
        window_end = window_start + window_delta

        start_lp = last_lp
        start_reserve = last_reserve
        start_block = last_block

        unique_txs: Set[str] = set()
        window_mint_events = 0
        window_burn_events = 0
        window_swap_events = 0
        window_mint_lp = Decimal(0)
        window_burn_lp = Decimal(0)
        window_base_sell_volume = Decimal(0)
        window_base_buy_volume = Decimal(0)

        time_since_last_mint_seconds: Optional[Decimal] = None
        if last_mint_ts is not None:
            delta_sec = Decimal(int((window_start - last_mint_ts).total_seconds()))
            if delta_sec < 0:
                delta_sec = Decimal(0)
            time_since_last_mint_seconds = delta_sec

        while event_idx < total_events and ordered_events[event_idx].timestamp < window_end:
            evt = ordered_events[event_idx]

            if start_lp is None:
                start_lp = evt.total_supply_in_pair
            if start_reserve is None:
                start_reserve = evt.reserve_in_pair
            if start_block is None:
                start_block = evt.block_number

            if evt.tx_hash:
                unique_txs.add(evt.tx_hash)

            evt_type = (evt.evt_type or "").lower()
            if evt_type == "mint":
                window_mint_events += 1
                last_mint_ts = evt.timestamp
            elif evt_type == "burn":
                window_burn_events += 1
            elif evt_type == "swap":
                window_swap_events += 1

            prev_lp = last_lp
            new_lp = evt.total_supply_in_pair
            if new_lp is not None:
                if prev_lp is not None:
                    delta_lp = new_lp - prev_lp
                    if delta_lp > 0:
                        window_mint_lp += delta_lp
                    elif delta_lp < 0:
                        window_burn_lp += -delta_lp
                last_lp = new_lp
                if lp_peak is None or new_lp > lp_peak:
                    lp_peak = new_lp
            if evt.reserve_in_pair is not None:
                last_reserve = evt.reserve_in_pair
            if evt.block_number is not None:
                last_block = evt.block_number

            if evt.mint_base_amount is not None:
                cumulative_base_minted += evt.mint_base_amount
            if evt.mint_quote_amount is not None:
                cumulative_quote_minted += evt.mint_quote_amount
            if evt.burn_base_amount is not None:
                cumulative_base_burned += evt.burn_base_amount
            if evt.burn_quote_amount is not None:
                cumulative_quote_burned += evt.burn_quote_amount
            if evt.swap_base_in is not None:
                window_base_sell_volume += evt.swap_base_in
            if evt.swap_base_out is not None:
                window_base_buy_volume += evt.swap_base_out

            event_idx += 1

        if window_mint_events > 0:
            time_since_last_mint_seconds = Decimal(0)

        end_lp = last_lp if last_lp is not None else start_lp
        end_reserve = last_reserve if last_reserve is not None else start_reserve
        current_block = last_block if last_block is not None else start_block

        if start_lp is None:
            lp_drop_amount_value = None
            burn_frac_value = None
        else:
            start_lp_nonnull = start_lp
            end_lp_nonnull = end_lp if end_lp is not None else start_lp_nonnull
            decrease = start_lp_nonnull - end_lp_nonnull
            if decrease < 0:
                decrease = Decimal(0)
            lp_drop_amount_value = decrease
            burn_frac_value = None
            if start_lp_nonnull == 0:
                burn_frac_value = Decimal(0)
            else:
                burn_frac = (start_lp_nonnull - end_lp_nonnull) / start_lp_nonnull
                if burn_frac < 0:
                    burn_frac = Decimal(0)
                burn_frac_value = burn_frac

        if start_reserve is None:
            reserve_drop_frac_value = None
        else:
            start_reserve_nonnull = start_reserve
            end_reserve_nonnull = end_reserve if end_reserve is not None else start_reserve_nonnull
            if start_reserve_nonnull == 0:
                reserve_drop_frac_value = Decimal(0)
            else:
                drop = Decimal(1) - (end_reserve_nonnull / start_reserve_nonnull)
                if drop < 0:
                    drop = Decimal(0)
                reserve_drop_frac_value = drop

        win_start_block_value = start_block if start_block is not None else current_block
        if current_block is None or win_start_block_value is None:
            win_blocks_value = None
        else:
            win_blocks_value = current_block - win_start_block_value

        lp_mint_amount_value: Optional[Decimal] = None
        if window_mint_lp > 0:
            lp_mint_amount_value = window_mint_lp

        lp_burn_amount_value: Optional[Decimal] = None
        if window_burn_lp > 0:
            lp_burn_amount_value = window_burn_lp

        burn_to_mint_ratio_value: Optional[Decimal] = None
        if lp_burn_amount_value is not None and lp_burn_amount_value > 0:
            if lp_mint_amount_value is not None and lp_mint_amount_value > 0:
                burn_to_mint_ratio_value = lp_burn_amount_value / lp_mint_amount_value
            else:
                burn_to_mint_ratio_value = None

        lp_peak_drop_value: Optional[Decimal] = None
        if lp_peak is not None and lp_peak > 0 and end_lp is not None:
            peak_drop = (lp_peak - end_lp) / lp_peak
            if peak_drop < 0:
                peak_drop = Decimal(0)
            lp_peak_drop_value = peak_drop

        lp_start_peak_frac_value: Optional[Decimal] = None
        if lp_peak is not None and lp_peak > 0 and start_lp is not None:
            lp_start_peak_frac_value = start_lp / lp_peak

        swap_base_sell_value: Optional[Decimal] = None
        if window_base_sell_volume > 0:
            swap_base_sell_value = window_base_sell_volume

        swap_base_buy_value: Optional[Decimal] = None
        if window_base_buy_volume > 0:
            swap_base_buy_value = window_base_buy_volume

        features.append(
            FeatureRow(
                win_id=win_id,
                token_addr_idx=events[0].token_addr_idx,
                win_start_ts=to_iso_z(window_start),
                win_start_block=win_start_block_value,
                win_tx_count=len(unique_txs),
                win_blocks=win_blocks_value,
                lp_start=decimal_to_str(start_lp),
                lp_end=decimal_to_str(end_lp),
                lp_drop_amount=decimal_to_str(lp_drop_amount_value),
                burn_frac=decimal_to_str(burn_frac_value),
                reserve_token_start=decimal_to_str(start_reserve),
                reserve_token_end=decimal_to_str(end_reserve),
                reserve_token_drop_frac=decimal_to_str(reserve_drop_frac_value),
                lp_mint_amount=decimal_to_str(lp_mint_amount_value),
                lp_burn_amount=decimal_to_str(lp_burn_amount_value),
                mint_events=window_mint_events,
                burn_events=window_burn_events,
                swap_events=window_swap_events,
                burn_to_mint_ratio=decimal_to_str(burn_to_mint_ratio_value),
                time_since_last_mint_sec=decimal_to_str(time_since_last_mint_seconds),
                lp_peak_drop_frac=decimal_to_str(lp_peak_drop_value),
                lp_start_peak_frac=decimal_to_str(lp_start_peak_frac_value),
                swap_base_sell_volume=decimal_to_str(swap_base_sell_value),
                swap_base_buy_volume=decimal_to_str(swap_base_buy_value),
                cum_base_minted=decimal_to_str(cumulative_base_minted),
                cum_base_burned=decimal_to_str(cumulative_base_burned),
                cum_quote_minted=decimal_to_str(cumulative_quote_minted),
                cum_quote_burned=decimal_to_str(cumulative_quote_burned),
            )
        )

        win_id += 1
        window_start += window_delta

    return features, win_id


def generate_features(
    token_events: Dict[str, List[TokenEvent]],
    window_seconds: int,
) -> List[FeatureRow]:
    features: List[FeatureRow] = []
    win_id = 0
    for token_idx in sorted(token_events.keys()):
        token_features, win_id = generate_token_features(token_events[token_idx], window_seconds, win_id)
        features.extend(token_features)

    return features


def write_features_csv(path: Path, features: Sequence[FeatureRow]) -> None:
    fieldnames = [
        "win_id",
        "token_addr_idx",
        "win_start_ts",
        "win_start_block",
        "win_tx_count",
        "win_blocks",
        "lp_start",
        "lp_end",
        "lp_drop_amount",
        "burn_frac",
        "reserve_token_start",
        "reserve_token_end",
        "reserve_token_drop_frac",
        "lp_mint_amount",
        "lp_burn_amount",
        "mint_events",
        "burn_events",
        "swap_events",
        "burn_to_mint_ratio",
        "time_since_last_mint_sec",
        "lp_peak_drop_frac",
        "lp_start_peak_frac",
        "swap_base_sell_volume",
        "swap_base_buy_volume",
        "cum_base_minted",
        "cum_base_burned",
        "cum_quote_minted",
        "cum_quote_burned",
    ]

    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in features:
            writer.writerow({
                "win_id": row.win_id,
                "token_addr_idx": row.token_addr_idx,
                "win_start_ts": row.win_start_ts,
                "win_start_block": row.win_start_block if row.win_start_block is not None else "",
                "win_tx_count": row.win_tx_count,
                "win_blocks": row.win_blocks if row.win_blocks is not None else "",
                "lp_start": row.lp_start or "",
                "lp_end": row.lp_end or "",
                "lp_drop_amount": row.lp_drop_amount or "",
                "burn_frac": row.burn_frac or "",
                "reserve_token_start": row.reserve_token_start or "",
                "reserve_token_end": row.reserve_token_end or "",
                "reserve_token_drop_frac": row.reserve_token_drop_frac or "",
                "lp_mint_amount": row.lp_mint_amount or "",
                "lp_burn_amount": row.lp_burn_amount or "",
                "mint_events": row.mint_events,
                "burn_events": row.burn_events,
                "swap_events": row.swap_events,
                "burn_to_mint_ratio": row.burn_to_mint_ratio or "",
                "time_since_last_mint_sec": row.time_since_last_mint_sec or "",
                "lp_peak_drop_frac": row.lp_peak_drop_frac or "",
                "lp_start_peak_frac": row.lp_start_peak_frac or "",
                "swap_base_sell_volume": row.swap_base_sell_volume or "",
                "swap_base_buy_volume": row.swap_base_buy_volume or "",
                "cum_base_minted": row.cum_base_minted or "",
                "cum_base_burned": row.cum_base_burned or "",
                "cum_quote_minted": row.cum_quote_minted or "",
                "cum_quote_burned": row.cum_quote_burned or "",
            })


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate 5-second window features for tokens")
    parser.add_argument("--token-info", required=True, help="Path to Token_Information CSV")
    parser.add_argument("--token-events", required=True, help="Path to Token_onData CSV")
    parser.add_argument("--output", required=True, help="Path to write Feature CSV")
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=WINDOW_SECONDS_DEFAULT,
        help="Size of rolling window in seconds (default: 5)",
    )

    args = parser.parse_args(argv)

    token_info_path = Path(args.token_info)
    token_events_path = Path(args.token_events)
    output_path = Path(args.output)

    token_info_map, info_headers, info_used_columns = load_token_information(token_info_path)
    token_events_map, event_headers, event_used_columns = load_token_events(token_events_path, token_info_map)

    features = generate_features(token_events_map, args.window_seconds)
    write_features_csv(output_path, features)

    unused_info_columns = sorted(set(info_headers) - info_used_columns)
    unused_event_columns = sorted(set(event_headers) - event_used_columns)

    if unused_info_columns:
        print("Unused Token_Information columns:", ", ".join(unused_info_columns))
    else:
        print("All Token_Information columns were used in processing.")

    if unused_event_columns:
        print("Unused Token_onData columns:", ", ".join(unused_event_columns))
    else:
        print("All Token_onData columns were used in processing.")


if __name__ == "__main__":
    main()
