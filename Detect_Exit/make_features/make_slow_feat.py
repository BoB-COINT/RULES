#!/usr/bin/env python3
"""Generate slow-rug detection features from token event data."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

getcontext().prec = 28


@dataclass
class Event:
    timestamp: datetime
    evt_type: str
    lp_total_supply: Optional[Decimal]
    reserve0: Optional[Decimal]
    reserve1: Optional[Decimal]


@dataclass
class TokenState:
    lp: Optional[Decimal] = None
    reserve0: Optional[Decimal] = None
    reserve1: Optional[Decimal] = None
    lp_peak: Decimal = Decimal("0")
    reserve1_peak: Decimal = Decimal("0")
    last_mint_ts: Optional[datetime] = None
    consecutive_drop: int = 0


@dataclass
class WindowStats:
    window_start: datetime
    event_count: int = 0
    burn_events: int = 0
    mint_events: int = 0
    swap_events: int = 0
    start_lp: Optional[Decimal] = None
    end_lp: Optional[Decimal] = None
    start_reserve1: Optional[Decimal] = None
    end_reserve1: Optional[Decimal] = None
    price_ratio_start: Optional[Decimal] = None
    price_ratio_end: Optional[Decimal] = None
    lp_minted_amount: Decimal = Decimal("0")
    lp_burned_amount: Decimal = Decimal("0")

# Tunable parameters for slow-drain aggregation (aligned with STE0202 rule defaults)
WINDOW_SECONDS = 600  # seconds per aggregation window
DROP_THRESHOLD = Decimal("0.02")  # per-window LP drop to count consecutive decline
MAX_SLOW_DROP_FRAC = Decimal("0.50")  # treat larger drops as rapid events, not slow drain


FIELDNAMES = [
    "win_id",
    "token_addr_idx",
    "win_start_ts",
    "event_count",
    "burn_events",
    "mint_events",
    "swap_events",
    "lp_start",
    "lp_end",
    "lp_drop_frac",
    "lp_cum_drawdown",
    "burn_to_mint_ratio",
    "lp_burn_amount",
    "lp_mint_amount",
    "time_since_last_mint_sec",
    "consecutive_drop_windows",
    "reserve_token_start",
    "reserve_token_end",
    "reserve_token_drop_frac",
    "price_ratio_start",
    "price_ratio_end",
    "price_ratio_change",
    "lp_tx_ratio",
    "swap_activity_ratio",
]


def parse_timestamp(value: str) -> datetime:
    value = value.strip()
    if not value:
        raise ValueError("timestamp is required")
    # normalise to ISO 8601
    value = value.replace(" ", "T")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    return dt.astimezone(timezone.utc)


def parse_decimal(value: str) -> Optional[Decimal]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid decimal value: {value}") from exc


def decimal_to_str(value: Optional[Decimal]) -> str:
    if value is None:
        return ""
    s = format(value, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def float_to_str(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def to_ratio(numerator: Optional[Decimal], denominator: Optional[Decimal]) -> Optional[Decimal]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def floor_to_window(ts: datetime, window_seconds: int) -> datetime:
    epoch_seconds = int(ts.timestamp())
    floored = (epoch_seconds // window_seconds) * window_seconds
    return datetime.fromtimestamp(floored, tz=timezone.utc)


def load_token_information(path: Path) -> Set[str]:
    """Load the set of token indices defined in token_information.csv."""

    token_indices: Set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            idx_raw = (row.get("token_addr_idx") or "").strip()
            if not idx_raw:
                continue
            token_indices.add(idx_raw)
    return token_indices


def read_events(path: Path, allowed_tokens: Optional[Set[str]] = None) -> Dict[str, List[Event]]:
    events_by_token: Dict[str, List[Event]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            token_idx = (row.get("token_addr_idx") or "0").strip()
            if allowed_tokens is not None and token_idx not in allowed_tokens:
                continue
            ts = parse_timestamp(row["timestamp"])
            evt_type = (row.get("evt_type") or "").strip()
            lp_total_supply = parse_decimal(row.get("lp_total_supply", ""))
            reserve0 = parse_decimal(row.get("reserve0", ""))
            reserve1 = parse_decimal(row.get("reserve1", ""))
            events_by_token[token_idx].append(
                Event(
                    timestamp=ts,
                    evt_type=evt_type,
                    lp_total_supply=lp_total_supply,
                    reserve0=reserve0,
                    reserve1=reserve1,
                )
            )
    for events in events_by_token.values():
        events.sort(key=lambda e: e.timestamp)
    return dict(events_by_token)


def init_window(start: datetime, state: TokenState) -> WindowStats:
    price_ratio = to_ratio(state.reserve0, state.reserve1)
    return WindowStats(
        window_start=start,
        start_lp=state.lp,
        end_lp=state.lp,
        start_reserve1=state.reserve1,
        end_reserve1=state.reserve1,
        price_ratio_start=price_ratio,
        price_ratio_end=price_ratio,
    )


def finalize_window(
    token_idx: str,
    window_id: int,
    stats: WindowStats,
    state: TokenState,
) -> Optional[Dict[str, str]]:
    if stats.event_count == 0 and stats.start_lp is None and stats.end_lp is None:
        return None

    start_lp = stats.start_lp if stats.start_lp is not None else stats.end_lp
    end_lp = stats.end_lp if stats.end_lp is not None else start_lp

    if start_lp is None and end_lp is None:
        return None

    # Ensure drawdown state is up-to-date
    if end_lp is not None and end_lp > state.lp_peak:
        state.lp_peak = end_lp

    lp_drop_frac = None
    if start_lp is not None and end_lp is not None:
        if start_lp > 0:
            lp_drop_frac = (start_lp - end_lp) / start_lp
        else:
            lp_drop_frac = Decimal("0")

    reserve_start = stats.start_reserve1 if stats.start_reserve1 is not None else stats.end_reserve1
    reserve_end = stats.end_reserve1 if stats.end_reserve1 is not None else reserve_start

    reserve_drop_frac = None
    if reserve_start is not None and reserve_end is not None:
        if reserve_start > 0:
            reserve_drop_frac = (reserve_start - reserve_end) / reserve_start
        else:
            reserve_drop_frac = Decimal("0")

    price_start = stats.price_ratio_start if stats.price_ratio_start is not None else stats.price_ratio_end
    price_end = stats.price_ratio_end if stats.price_ratio_end is not None else price_start
    price_change = None
    if price_start is not None and price_end is not None:
        price_change = price_end - price_start

    lp_cum_drawdown = None
    if state.lp_peak > 0 and end_lp is not None:
        lp_cum_drawdown = (state.lp_peak - end_lp) / state.lp_peak

    # Update consecutive drop counter for future windows
    if lp_drop_frac is not None and lp_drop_frac >= DROP_THRESHOLD:
        if MAX_SLOW_DROP_FRAC is None or lp_drop_frac <= MAX_SLOW_DROP_FRAC:
            state.consecutive_drop += 1
        else:
            state.consecutive_drop = 0
    else:
        state.consecutive_drop = 0

    burn_amount = stats.lp_burned_amount if stats.lp_burned_amount > 0 else None
    mint_amount = stats.lp_minted_amount if stats.lp_minted_amount > 0 else None
    burn_to_mint_ratio: Optional[Decimal] = None
    if burn_amount is not None and mint_amount is not None:
        if mint_amount > 0:
            burn_to_mint_ratio = burn_amount / mint_amount
    elif burn_amount is not None and mint_amount is None:
        # Represent effectively infinite ratio when no mint volume occurred.
        burn_to_mint_ratio = None

    window_end = stats.window_start + timedelta(seconds=WINDOW_SECONDS)
    time_since_last_mint = None
    if state.last_mint_ts is not None:
        delta = (window_end - state.last_mint_ts).total_seconds()
        if delta >= 0:
            time_since_last_mint = delta

    swap_activity_ratio = None
    lp_tx_ratio = None
    if stats.event_count > 0:
        swap_activity_ratio = Decimal(stats.swap_events) / Decimal(stats.event_count)
        lp_tx_ratio = Decimal(stats.burn_events + stats.mint_events) / Decimal(stats.event_count)

    return {
        "win_id": str(window_id),
        "token_addr_idx": token_idx,
        "win_start_ts": stats.window_start.isoformat().replace("+00:00", "Z"),
        "event_count": str(stats.event_count),
        "burn_events": str(stats.burn_events),
        "mint_events": str(stats.mint_events),
        "swap_events": str(stats.swap_events),
        "lp_start": decimal_to_str(start_lp),
        "lp_end": decimal_to_str(end_lp),
        "lp_drop_frac": decimal_to_str(lp_drop_frac),
        "lp_cum_drawdown": decimal_to_str(lp_cum_drawdown),
        "burn_to_mint_ratio": decimal_to_str(burn_to_mint_ratio),
        "lp_burn_amount": decimal_to_str(burn_amount),
        "lp_mint_amount": decimal_to_str(mint_amount),
        "time_since_last_mint_sec": float_to_str(time_since_last_mint),
        "consecutive_drop_windows": str(state.consecutive_drop),
        "reserve_token_start": decimal_to_str(reserve_start),
        "reserve_token_end": decimal_to_str(reserve_end),
        "reserve_token_drop_frac": decimal_to_str(reserve_drop_frac),
        "price_ratio_start": decimal_to_str(price_start),
        "price_ratio_end": decimal_to_str(price_end),
        "price_ratio_change": decimal_to_str(price_change),
        "lp_tx_ratio": decimal_to_str(lp_tx_ratio),
        "swap_activity_ratio": decimal_to_str(swap_activity_ratio),
    }


def generate_token_windows(
    token_idx: str,
    events: List[Event],
) -> List[Dict[str, str]]:
    if not events:
        return []

    state = TokenState()
    features: List[Dict[str, str]] = []

    first_ts = events[0].timestamp
    current_window_start = floor_to_window(first_ts, WINDOW_SECONDS)
    stats = init_window(current_window_start, state)
    window_id = 0

    for event in events:
        # Advance windows if the event falls outside the current window
        while event.timestamp >= current_window_start + timedelta(seconds=WINDOW_SECONDS):
            row = finalize_window(token_idx, window_id, stats, state)
            if row:
                features.append(row)
                window_id += 1
            current_window_start += timedelta(seconds=WINDOW_SECONDS)
            stats = init_window(current_window_start, state)

        prev_lp = state.lp
        prev_reserve0 = state.reserve0
        prev_reserve1 = state.reserve1

        new_lp = event.lp_total_supply if event.lp_total_supply is not None else prev_lp
        new_reserve0 = event.reserve0 if event.reserve0 is not None else prev_reserve0
        new_reserve1 = event.reserve1 if event.reserve1 is not None else prev_reserve1

        if stats.start_lp is None:
            stats.start_lp = prev_lp if prev_lp is not None else new_lp
        stats.end_lp = new_lp

        if prev_lp is not None and new_lp is not None:
            delta_lp = new_lp - prev_lp
            if delta_lp > 0:
                stats.lp_minted_amount += delta_lp
            elif delta_lp < 0:
                stats.lp_burned_amount += -delta_lp

        if stats.start_reserve1 is None:
            stats.start_reserve1 = prev_reserve1 if prev_reserve1 is not None else new_reserve1
        stats.end_reserve1 = new_reserve1

        if stats.price_ratio_start is None:
            stats.price_ratio_start = to_ratio(prev_reserve0, prev_reserve1) if prev_reserve0 is not None else to_ratio(new_reserve0, new_reserve1)
        stats.price_ratio_end = to_ratio(new_reserve0, new_reserve1)

        stats.event_count += 1
        if event.evt_type.lower() == "burn":
            stats.burn_events += 1
        elif event.evt_type.lower() == "mint":
            stats.mint_events += 1
            state.last_mint_ts = event.timestamp
        elif event.evt_type.lower() == "swap":
            stats.swap_events += 1

        # Update running state
        state.lp = new_lp
        state.reserve0 = new_reserve0
        state.reserve1 = new_reserve1
        if new_lp is not None and new_lp > state.lp_peak:
            state.lp_peak = new_lp
        if new_reserve1 is not None and new_reserve1 > state.reserve1_peak:
            state.reserve1_peak = new_reserve1

    # Final window
    row = finalize_window(token_idx, window_id, stats, state)
    if row:
        features.append(row)

    return features


def write_features(path: str, rows: Iterable[Dict[str, str]]) -> None:
    rows = list(rows)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate slow rug pull features from token events")
    parser.add_argument("--token-info", required=True, help="Path to token_information.csv")
    parser.add_argument("--pair-events", required=True, help="Path to pair_evt.csv")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args(argv)

    token_info_path = Path(args.token_info)
    pair_events_path = Path(args.pair_events)
    output_path = Path(args.output)

    token_indices = load_token_information(token_info_path)
    events_by_token = read_events(pair_events_path, allowed_tokens=token_indices)

    all_rows: List[Dict[str, str]] = []
    for token_idx, events in events_by_token.items():
        token_rows = generate_token_windows(token_idx, events)
        all_rows.extend(token_rows)

    write_features(str(output_path), all_rows)
    print(f"Generated {len(all_rows)} windows to {output_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
