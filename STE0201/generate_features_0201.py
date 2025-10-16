#!/usr/bin/env python3
"""Generate 5-second window features from token information and on-chain event data."""

from __future__ import annotations

import argparse
import csv
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
    lp_dec: Optional[str]
    burn_frac: Optional[str]
    reserve_start: Optional[str]
    reserve_end: Optional[str]
    reserve_drop_frac: Optional[str]
    lp_increase: Optional[str]
    lp_burn_amount: Optional[str]
    mint_events: int
    burn_events: int
    swap_events: int
    burn_to_mint_ratio: Optional[str]
    time_since_last_mint_sec: Optional[str]
    lp_peak_drop_frac: Optional[str]


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

            if reserve_in_pair is None:
                reserve0_raw = row.get("reserve0")
                reserve1_raw = row.get("reserve1")
                token0 = (row.get("token0") or "").strip()
                token1 = (row.get("token1") or "").strip()

                if reserve0_raw is not None:
                    used_columns.add("reserve0")
                if reserve1_raw is not None:
                    used_columns.add("reserve1")
                if token0:
                    used_columns.add("token0")
                if token1:
                    used_columns.add("token1")

                reserve0 = parse_decimal(reserve0_raw)
                reserve1 = parse_decimal(reserve1_raw)

                token_info = token_info_map.get(idx_raw) if token_info_map else None
                token_addr = (token_info.token_addr or "").strip().lower() if token_info else ""
                token0_norm = token0.lower() if token0 else ""
                token1_norm = token1.lower() if token1 else ""

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

            event_idx += 1

        if window_mint_events > 0:
            time_since_last_mint_seconds = Decimal(0)

        end_lp = last_lp if last_lp is not None else start_lp
        end_reserve = last_reserve if last_reserve is not None else start_reserve
        current_block = last_block if last_block is not None else start_block

        if start_lp is None:
            lp_dec_value = None
            burn_frac_value = None
        else:
            start_lp_nonnull = start_lp
            end_lp_nonnull = end_lp if end_lp is not None else start_lp_nonnull
            decrease = start_lp_nonnull - end_lp_nonnull
            if decrease < 0:
                decrease = Decimal(0)
            lp_dec_value = decrease
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

        lp_increase_value: Optional[Decimal] = None
        if window_mint_lp > 0:
            lp_increase_value = window_mint_lp

        lp_burn_amount_value: Optional[Decimal] = None
        if window_burn_lp > 0:
            lp_burn_amount_value = window_burn_lp

        burn_to_mint_ratio_value: Optional[Decimal] = None
        if lp_burn_amount_value is not None and lp_burn_amount_value > 0:
            if lp_increase_value is not None and lp_increase_value > 0:
                burn_to_mint_ratio_value = lp_burn_amount_value / lp_increase_value
            else:
                burn_to_mint_ratio_value = None

        lp_peak_drop_value: Optional[Decimal] = None
        if lp_peak is not None and lp_peak > 0 and end_lp is not None:
            peak_drop = (lp_peak - end_lp) / lp_peak
            if peak_drop < 0:
                peak_drop = Decimal(0)
            lp_peak_drop_value = peak_drop

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
                lp_dec=decimal_to_str(lp_dec_value),
                burn_frac=decimal_to_str(burn_frac_value),
                reserve_start=decimal_to_str(start_reserve),
                reserve_end=decimal_to_str(end_reserve),
                reserve_drop_frac=decimal_to_str(reserve_drop_frac_value),
                lp_increase=decimal_to_str(lp_increase_value),
                lp_burn_amount=decimal_to_str(lp_burn_amount_value),
                mint_events=window_mint_events,
                burn_events=window_burn_events,
                swap_events=window_swap_events,
                burn_to_mint_ratio=decimal_to_str(burn_to_mint_ratio_value),
                time_since_last_mint_sec=decimal_to_str(time_since_last_mint_seconds),
                lp_peak_drop_frac=decimal_to_str(lp_peak_drop_value),
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
        "lp_dec",
        "burn_frac",
        "reserve_start",
        "reserve_end",
        "reserve_drop_frac",
        "lp_increase",
        "lp_burn_amount",
        "mint_events",
        "burn_events",
        "swap_events",
        "burn_to_mint_ratio",
        "time_since_last_mint_sec",
        "lp_peak_drop_frac",
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
                "lp_dec": row.lp_dec or "",
                "burn_frac": row.burn_frac or "",
                "reserve_start": row.reserve_start or "",
                "reserve_end": row.reserve_end or "",
                "reserve_drop_frac": row.reserve_drop_frac or "",
                "lp_increase": row.lp_increase or "",
                "lp_burn_amount": row.lp_burn_amount or "",
                "mint_events": row.mint_events,
                "burn_events": row.burn_events,
                "swap_events": row.swap_events,
                "burn_to_mint_ratio": row.burn_to_mint_ratio or "",
                "time_since_last_mint_sec": row.time_since_last_mint_sec or "",
                "lp_peak_drop_frac": row.lp_peak_drop_frac or "",
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
