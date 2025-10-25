#!/usr/bin/env python3
"""Merge multiple feature sets into a unified 5-second rolling window representation."""

from __future__ import annotations

import argparse
import csv
from bisect import bisect_left, bisect_right
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import make_hard_feat
import make_mintndump_feat
import make_slow_feat

getcontext().prec = 28


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def parse_iso_timestamp(value: str) -> datetime:
    """Parse an ISO 8601 timestamp (with optional trailing 'Z') into UTC datetime."""

    text = value.strip()
    if not text:
        raise ValueError("Empty timestamp value")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_iso_z(dt: datetime) -> str:
    """Return an ISO 8601 string with trailing Z."""

    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def decimal_to_str(value: Optional[Decimal]) -> str:
    if value is None:
        return ""
    try:
        normalized = value.normalize()
    except InvalidOperation:
        normalized = value
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def float_to_str(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def ratio(numerator: Optional[Decimal], denominator: Optional[Decimal]) -> Optional[Decimal]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


# --------------------------------------------------------------------------- #
# Mint feature rolling aggregation
# --------------------------------------------------------------------------- #


@dataclass
class MintSellEvent:
    timestamp: datetime
    base_in: Decimal
    quote_out: Decimal
    ratio: Optional[Decimal]


def prepare_mint_sell_events(events: Sequence[make_mintndump_feat.PairEvent]) -> List[MintSellEvent]:
    """Pre-compute sell-side swap metrics needed for rolling aggregation."""

    sell_events: List[MintSellEvent] = []
    state = make_mintndump_feat.TokenState()

    for evt in events:
        if evt.evt_type == "swap":
            base_in = evt.base_in
            base_out = evt.base_out
            quote_out = evt.quote_out
            reserve_after = evt.reserve_base
            reserve_before: Optional[Decimal] = None

            if reserve_after is not None:
                reserve_before = reserve_after - base_in + base_out
            elif state.reserve_base is not None:
                reserve_before = state.reserve_base
                reserve_after = state.reserve_base + base_in - base_out

            if base_in > 0 and quote_out > 0:
                sell_ratio: Optional[Decimal] = None
                if reserve_before is not None and reserve_before > 0:
                    sell_ratio = base_in / reserve_before
                sell_events.append(
                    MintSellEvent(
                        timestamp=evt.timestamp,
                        base_in=base_in,
                        quote_out=quote_out,
                        ratio=sell_ratio,
                    )
                )

            state.reserve_base = reserve_after
            if evt.reserve_quote is not None:
                state.reserve_quote = evt.reserve_quote
        else:
            if evt.reserve_base is not None:
                state.reserve_base = evt.reserve_base
            if evt.reserve_quote is not None:
                state.reserve_quote = evt.reserve_quote

    return sell_events


def compute_mint_metrics_for_token(
    timepoints: Sequence[datetime],
    events: Sequence[make_mintndump_feat.PairEvent],
) -> Dict[datetime, Dict[str, Optional[Decimal]]]:
    """Compute trailing 60-second mint/dump metrics for each timepoint."""

    if not timepoints:
        return {}

    sell_events_list = list(prepare_mint_sell_events(events))

    window_metrics: Dict[datetime, Dict[str, Optional[Decimal]]] = {}
    queue: Deque[MintSellEvent] = deque()
    idx = 0
    total_base = Decimal("0")
    total_quote = Decimal("0")
    ratio_sum = Decimal("0")
    ratio_count = 0

    for current_ts in timepoints:
        window_start = current_ts - timedelta(seconds=60)

        # Remove events that fall outside the trailing window
        while queue and queue[0].timestamp < window_start:
            ev = queue.popleft()
            total_base -= ev.base_in
            total_quote -= ev.quote_out
            if ev.ratio is not None:
                ratio_sum -= ev.ratio
                ratio_count -= 1

        # Add newly available events up to the current timepoint
        while idx < len(sell_events_list) and sell_events_list[idx].timestamp <= current_ts:
            ev = sell_events_list[idx]
            queue.append(ev)
            total_base += ev.base_in
            total_quote += ev.quote_out
            if ev.ratio is not None:
                ratio_sum += ev.ratio
                ratio_count += 1
            idx += 1

        sell_swap_count = len(queue)
        ratio_avg: Optional[Decimal] = None
        ratio_max: Optional[Decimal] = None
        if sell_swap_count > 0:
            if ratio_count > 0:
                ratio_avg = ratio_sum / Decimal(ratio_count)
            ratio_max = max((ev.ratio for ev in queue if ev.ratio is not None), default=None)
        sell_abs_max = max((ev.base_in for ev in queue), default=Decimal("0"))

        window_metrics[current_ts] = {
            "sell_swap_count": Decimal(sell_swap_count),
            "sell_base_volume": total_base if sell_swap_count > 0 else Decimal("0"),
            "sell_quote_volume": total_quote if sell_swap_count > 0 else Decimal("0"),
            "sell_to_reserve_avg_ratio": ratio_avg,
            "sell_to_reserve_max_ratio": ratio_max,
            "sell_base_abs_max": sell_abs_max if sell_swap_count > 0 else Decimal("0"),
        }

    return window_metrics


# --------------------------------------------------------------------------- #
# Slow-drain rolling aggregation
# --------------------------------------------------------------------------- #


@dataclass
class SlowEventSnapshot:
    timestamp: datetime
    evt_type: str
    lp_before: Optional[Decimal]
    lp_after: Optional[Decimal]
    reserve0_after: Optional[Decimal]
    reserve1_after: Optional[Decimal]
    mint_amount: Decimal
    burn_amount: Decimal


def prepare_slow_snapshots(
    events: Sequence[make_slow_feat.Event],
) -> Tuple[List[SlowEventSnapshot], List[datetime]]:
    """Augment slow-rug events with before/after state and mint/burn deltas."""

    snapshots: List[SlowEventSnapshot] = []
    mint_timestamps: List[datetime] = []

    prev_lp: Optional[Decimal] = None
    prev_reserve0: Optional[Decimal] = None
    prev_reserve1: Optional[Decimal] = None

    for evt in events:
        lp_before = prev_lp
        reserve0_before = prev_reserve0
        reserve1_before = prev_reserve1

        lp_after = evt.lp_total_supply if evt.lp_total_supply is not None else prev_lp
        reserve0_after = evt.reserve0 if evt.reserve0 is not None else prev_reserve0
        reserve1_after = evt.reserve1 if evt.reserve1 is not None else prev_reserve1

        mint_amount = Decimal("0")
        burn_amount = Decimal("0")
        if prev_lp is not None and lp_after is not None:
            delta = lp_after - prev_lp
            if delta > 0:
                mint_amount = delta
            elif delta < 0:
                burn_amount = -delta

        snapshot = SlowEventSnapshot(
            timestamp=evt.timestamp,
            evt_type=evt.evt_type.lower(),
            lp_before=lp_before,
            lp_after=lp_after,
            reserve0_after=reserve0_after,
            reserve1_after=reserve1_after,
            mint_amount=mint_amount,
            burn_amount=burn_amount,
        )
        snapshots.append(snapshot)

        if snapshot.evt_type == "mint":
            mint_timestamps.append(evt.timestamp)

        prev_lp = lp_after
        prev_reserve0 = reserve0_after
        prev_reserve1 = reserve1_after

    return snapshots, mint_timestamps


def state_at_time(
    snapshots: Sequence[SlowEventSnapshot],
    timestamps: Sequence[datetime],
    timepoint: datetime,
) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
    """Return the LP and reserve state immediately after the latest event <= timepoint."""

    if not snapshots:
        return None, None, None
    idx = bisect_right(timestamps, timepoint) - 1
    if idx < 0:
        return None, None, None
    snap = snapshots[idx]
    return snap.lp_after, snap.reserve0_after, snap.reserve1_after


def compute_prefix_sums(
    snapshots: Sequence[SlowEventSnapshot],
) -> Dict[str, List[Decimal]]:
    """Build prefix sums and counts for fast range queries."""

    prefix: Dict[str, List[Decimal]] = {
        "events": [Decimal("0")],
        "mint_count": [Decimal("0")],
        "burn_count": [Decimal("0")],
        "swap_count": [Decimal("0")],
        "mint_amount": [Decimal("0")],
        "burn_amount": [Decimal("0")],
    }

    for snap in snapshots:
        prefix["events"].append(prefix["events"][-1] + Decimal("1"))
        prefix["mint_count"].append(
            prefix["mint_count"][-1] + (Decimal("1") if snap.evt_type == "mint" else Decimal("0"))
        )
        prefix["burn_count"].append(
            prefix["burn_count"][-1] + (Decimal("1") if snap.evt_type == "burn" else Decimal("0"))
        )
        prefix["swap_count"].append(
            prefix["swap_count"][-1] + (Decimal("1") if snap.evt_type == "swap" else Decimal("0"))
        )
        prefix["mint_amount"].append(prefix["mint_amount"][-1] + snap.mint_amount)
        prefix["burn_amount"].append(prefix["burn_amount"][-1] + snap.burn_amount)

    return prefix


def range_sum(prefix: List[Decimal], start: int, end: int) -> Decimal:
    return prefix[end] - prefix[start]


def compute_slow_metrics_for_token(
    timepoints: Sequence[datetime],
    events: Sequence[make_slow_feat.Event],
) -> Dict[datetime, Dict[str, Optional[Decimal]]]:
    """Compute trailing 600-second slow-drain metrics for each timepoint."""

    if not timepoints:
        return {}

    snapshots, mint_timestamps = prepare_slow_snapshots(events)
    timestamps = [snap.timestamp for snap in snapshots]
    prefix = compute_prefix_sums(snapshots)

    result: Dict[datetime, Dict[str, Optional[Decimal]]] = {}
    lp_peak = Decimal("0")
    consecutive_drop = 0

    for current_ts in timepoints:
        window_start = current_ts - timedelta(seconds=600)

        left = bisect_left(timestamps, window_start)
        right = bisect_right(timestamps, current_ts)

        event_count = int(range_sum(prefix["events"], left, right))
        burn_events = int(range_sum(prefix["burn_count"], left, right))
        mint_events = int(range_sum(prefix["mint_count"], left, right))
        swap_events = int(range_sum(prefix["swap_count"], left, right))
        burn_amount = range_sum(prefix["burn_amount"], left, right)
        mint_amount = range_sum(prefix["mint_amount"], left, right)

        start_lp, start_reserve0, start_reserve1 = state_at_time(snapshots, timestamps, window_start)
        end_lp, end_reserve0, end_reserve1 = state_at_time(snapshots, timestamps, current_ts)

        if end_lp is not None and end_lp > lp_peak:
            lp_peak = end_lp

        lp_drop_frac: Optional[Decimal] = None
        if start_lp is not None and end_lp is not None:
            if start_lp > 0:
                lp_drop_frac = (start_lp - end_lp) / start_lp
            else:
                lp_drop_frac = Decimal("0")

        reserve_start = start_reserve1 if start_reserve1 is not None else end_reserve1
        reserve_end = end_reserve1 if end_reserve1 is not None else reserve_start

        reserve_drop_frac: Optional[Decimal] = None
        if reserve_start is not None and reserve_end is not None:
            if reserve_start > 0:
                reserve_drop_frac = (reserve_start - reserve_end) / reserve_start
            else:
                reserve_drop_frac = Decimal("0")

        price_start = ratio(start_reserve0, start_reserve1) if start_reserve0 is not None else ratio(
            end_reserve0, end_reserve1
        )
        price_end = ratio(end_reserve0, end_reserve1) if end_reserve0 is not None else price_start
        price_change = None
        if price_start is not None and price_end is not None:
            price_change = price_end - price_start

        lp_cum_drawdown: Optional[Decimal] = None
        if lp_peak > 0 and end_lp is not None:
            lp_cum_drawdown = (lp_peak - end_lp) / lp_peak

        if lp_drop_frac is not None and lp_drop_frac >= make_slow_feat.DROP_THRESHOLD:
            if make_slow_feat.MAX_SLOW_DROP_FRAC is None or lp_drop_frac <= make_slow_feat.MAX_SLOW_DROP_FRAC:
                consecutive_drop += 1
            else:
                consecutive_drop = 0
        else:
            consecutive_drop = 0

        burn_to_mint_ratio: Optional[Decimal] = None
        if burn_amount > 0 and mint_amount > 0:
            burn_to_mint_ratio = burn_amount / mint_amount

        # Time since last mint (if any)
        mint_time_since: Optional[float] = None
        if mint_timestamps:
            idx = bisect_right(mint_timestamps, current_ts) - 1
            if idx >= 0:
                delta_sec = (current_ts - mint_timestamps[idx]).total_seconds()
                if delta_sec >= 0:
                    mint_time_since = delta_sec

        lp_tx_ratio: Optional[Decimal] = None
        swap_activity_ratio: Optional[Decimal] = None
        if event_count > 0:
            lp_tx_ratio = Decimal(burn_events + mint_events) / Decimal(event_count)
            swap_activity_ratio = Decimal(swap_events) / Decimal(event_count)

        result[current_ts] = {
            "event_count": Decimal(event_count),
            "burn_events": Decimal(burn_events),
            "mint_events": Decimal(mint_events),
            "swap_events": Decimal(swap_events),
            "lp_start": start_lp,
            "lp_end": end_lp,
            "lp_drop_frac": lp_drop_frac,
            "lp_cum_drawdown": lp_cum_drawdown,
            "burn_to_mint_ratio": burn_to_mint_ratio,
            "lp_burn_amount": burn_amount if burn_amount > 0 else None,
            "lp_mint_amount": mint_amount if mint_amount > 0 else None,
            "time_since_last_mint_sec": mint_time_since,
            "consecutive_drop_windows": Decimal(consecutive_drop),
            "reserve_token_start": reserve_start,
            "reserve_token_end": reserve_end,
            "reserve_token_drop_frac": reserve_drop_frac,
            "price_ratio_start": price_start,
            "price_ratio_end": price_end,
            "price_ratio_change": price_change,
            "lp_tx_ratio": lp_tx_ratio,
            "swap_activity_ratio": swap_activity_ratio,
        }

    return result


# --------------------------------------------------------------------------- #
# Merge pipeline
# --------------------------------------------------------------------------- #


BASE_FIELDNAMES = [
    "win_id",
    "token_addr_idx",
    "win_start_ts",
    "win_start_block",
    "win_tx_count",
    "win_blocks",
    "lp_start_5s",
    "lp_end_5s",
    "lp_drop_amount_5s",
    "burn_frac_5s",
    "reserve_token_start_5s",
    "reserve_token_end_5s",
    "reserve_token_drop_frac_5s",
    "lp_mint_amount_5s",
    "lp_burn_amount_5s",
    "mint_events_5s",
    "burn_events_5s",
    "swap_events_5s",
    "burn_to_mint_ratio_5s",
    "time_since_last_mint_sec_5s",
    "lp_peak_drop_frac_5s",
    "lp_start_peak_frac_5s",
    "swap_base_sell_volume_5s",
    "swap_base_buy_volume_5s",
    "cum_base_minted_5s",
    "cum_base_burned_5s",
    "cum_quote_minted_5s",
    "cum_quote_burned_5s",
]

MINT_FIELDNAMES = [
    "holder_top1_supply_pct",
    "holder_pair_supply_pct",
    "holder_top20_supply_pct",
    "mint_sell_swap_count_60s",
    "mint_sell_base_volume_60s",
    "mint_sell_to_reserve_max_ratio_60s",
    "mint_sell_to_reserve_avg_ratio_60s",
    "mint_sell_base_abs_max_60s",
    "mint_sell_quote_volume_60s",
]

SLOW_FIELDNAMES = [
    "event_count_600s",
    "burn_events_600s",
    "mint_events_600s",
    "swap_events_600s",
    "lp_start_600s",
    "lp_drop_frac_600s",
    "lp_cum_drawdown_600s",
    "lp_burn_amount_600s",
    "lp_mint_amount_600s",
    "time_since_last_mint_sec_600s",
    "consecutive_drop_windows_600s",
    "reserve_token_start_600s",
    "reserve_token_drop_frac_600s",
    "price_ratio_start_600s",
    "price_ratio_end_600s",
    "price_ratio_change_600s",
    "lp_tx_ratio_600s",
    "swap_activity_ratio_600s",
]


def build_output_row(
    base_row: make_hard_feat.FeatureRow,
    mint_metrics: Mapping[str, Optional[Decimal]],
    slow_metrics: Mapping[str, Optional[Decimal]],
    holder_stats: Optional[make_mintndump_feat.HolderStats],
) -> Dict[str, str]:
    """Combine base, mint, and slow features into a single CSV row."""

    output: Dict[str, str] = {}

    # Base features (mirroring make_hard_feat.write_features_csv behaviour)
    output["win_id"] = str(base_row.win_id)
    output["token_addr_idx"] = base_row.token_addr_idx
    output["win_start_ts"] = base_row.win_start_ts
    output["win_start_block"] = "" if base_row.win_start_block is None else str(base_row.win_start_block)
    output["win_tx_count"] = str(base_row.win_tx_count)
    output["win_blocks"] = "" if base_row.win_blocks is None else str(base_row.win_blocks)
    output["lp_start_5s"] = base_row.lp_start or ""
    output["lp_end_5s"] = base_row.lp_end or ""
    output["lp_drop_amount_5s"] = base_row.lp_drop_amount or ""
    output["burn_frac_5s"] = base_row.burn_frac or ""
    output["reserve_token_start_5s"] = base_row.reserve_token_start or ""
    output["reserve_token_end_5s"] = base_row.reserve_token_end or ""
    output["reserve_token_drop_frac_5s"] = base_row.reserve_token_drop_frac or ""
    output["lp_mint_amount_5s"] = base_row.lp_mint_amount or ""
    output["lp_burn_amount_5s"] = base_row.lp_burn_amount or ""
    output["mint_events_5s"] = str(base_row.mint_events)
    output["burn_events_5s"] = str(base_row.burn_events)
    output["swap_events_5s"] = str(base_row.swap_events)
    output["burn_to_mint_ratio_5s"] = base_row.burn_to_mint_ratio or ""
    output["time_since_last_mint_sec_5s"] = base_row.time_since_last_mint_sec or ""
    output["lp_peak_drop_frac_5s"] = base_row.lp_peak_drop_frac or ""
    output["lp_start_peak_frac_5s"] = base_row.lp_start_peak_frac or ""
    output["swap_base_sell_volume_5s"] = base_row.swap_base_sell_volume or ""
    output["swap_base_buy_volume_5s"] = base_row.swap_base_buy_volume or ""
    output["cum_base_minted_5s"] = base_row.cum_base_minted or ""
    output["cum_base_burned_5s"] = base_row.cum_base_burned or ""
    output["cum_quote_minted_5s"] = base_row.cum_quote_minted or ""
    output["cum_quote_burned_5s"] = base_row.cum_quote_burned or ""

    # Holder stats
    holder_top1 = holder_stats.top1_share if holder_stats else None
    holder_pair = holder_stats.pair_share if holder_stats else None
    holder_top20 = holder_stats.top20_share if holder_stats else None
    output["holder_top1_supply_pct"] = decimal_to_str(holder_top1)
    output["holder_pair_supply_pct"] = decimal_to_str(holder_pair)
    output["holder_top20_supply_pct"] = decimal_to_str(holder_top20)

    # Mint rolling metrics
    output["mint_sell_swap_count_60s"] = decimal_to_str(mint_metrics.get("sell_swap_count"))
    output["mint_sell_base_volume_60s"] = decimal_to_str(mint_metrics.get("sell_base_volume"))
    output["mint_sell_to_reserve_max_ratio_60s"] = decimal_to_str(mint_metrics.get("sell_to_reserve_max_ratio"))
    output["mint_sell_to_reserve_avg_ratio_60s"] = decimal_to_str(mint_metrics.get("sell_to_reserve_avg_ratio"))
    output["mint_sell_base_abs_max_60s"] = decimal_to_str(mint_metrics.get("sell_base_abs_max"))
    output["mint_sell_quote_volume_60s"] = decimal_to_str(mint_metrics.get("sell_quote_volume"))

    # Slow rolling metrics
    output["event_count_600s"] = decimal_to_str(slow_metrics.get("event_count"))
    output["burn_events_600s"] = decimal_to_str(slow_metrics.get("burn_events"))
    output["mint_events_600s"] = decimal_to_str(slow_metrics.get("mint_events"))
    output["swap_events_600s"] = decimal_to_str(slow_metrics.get("swap_events"))
    output["lp_start_600s"] = decimal_to_str(slow_metrics.get("lp_start"))
    output["lp_drop_frac_600s"] = decimal_to_str(slow_metrics.get("lp_drop_frac"))
    output["lp_cum_drawdown_600s"] = decimal_to_str(slow_metrics.get("lp_cum_drawdown"))
    output["lp_burn_amount_600s"] = decimal_to_str(slow_metrics.get("lp_burn_amount"))
    output["lp_mint_amount_600s"] = decimal_to_str(slow_metrics.get("lp_mint_amount"))
    output["time_since_last_mint_sec_600s"] = float_to_str(slow_metrics.get("time_since_last_mint_sec"))
    output["consecutive_drop_windows_600s"] = decimal_to_str(slow_metrics.get("consecutive_drop_windows"))
    output["reserve_token_start_600s"] = decimal_to_str(slow_metrics.get("reserve_token_start"))
    output["reserve_token_drop_frac_600s"] = decimal_to_str(slow_metrics.get("reserve_token_drop_frac"))
    output["price_ratio_start_600s"] = decimal_to_str(slow_metrics.get("price_ratio_start"))
    output["price_ratio_end_600s"] = decimal_to_str(slow_metrics.get("price_ratio_end"))
    output["price_ratio_change_600s"] = decimal_to_str(slow_metrics.get("price_ratio_change"))
    output["lp_tx_ratio_600s"] = decimal_to_str(slow_metrics.get("lp_tx_ratio"))
    output["swap_activity_ratio_600s"] = decimal_to_str(slow_metrics.get("swap_activity_ratio"))

    return output


def write_output_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    fieldnames = BASE_FIELDNAMES + MINT_FIELDNAMES + SLOW_FIELDNAMES
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Merge feature sets into 5-second rolling windows")
    parser.add_argument("--token-info", required=True, help="Path to token_information.csv")
    parser.add_argument("--token-events", required=True, help="Path to token_evt.csv")
    parser.add_argument("--pair-events", required=True, help="Path to pair_evt.csv")
    parser.add_argument("--holders", required=True, help="Path to holders.csv")
    parser.add_argument("--output", required=True, help="Output CSV path")

    args = parser.parse_args(argv)

    token_info_path = Path(args.token_info)
    token_events_path = Path(args.token_events)
    pair_events_path = Path(args.pair_events)
    holders_path = Path(args.holders)
    output_path = Path(args.output)

    # ------------------------------------------------------------------ #
    # Base 5-second features (reuse existing implementation)
    # ------------------------------------------------------------------ #
    hard_token_info_map, _, _ = make_hard_feat.load_token_information(token_info_path)
    token_events_map, _, _ = make_hard_feat.load_token_events(token_events_path, hard_token_info_map)
    base_features = make_hard_feat.generate_features(token_events_map, window_seconds=make_hard_feat.WINDOW_SECONDS_DEFAULT)

    rows_by_token: Dict[str, List[Tuple[datetime, make_hard_feat.FeatureRow]]] = defaultdict(list)
    for row in base_features:
        ts = parse_iso_timestamp(row.win_start_ts)
        rows_by_token[row.token_addr_idx].append((ts, row))

    for token_rows in rows_by_token.values():
        token_rows.sort(key=lambda item: item[0])

    tokens = set(rows_by_token.keys())

    # ------------------------------------------------------------------ #
    # Mint-and-dump rolling metrics
    # ------------------------------------------------------------------ #
    mint_token_info_map = make_mintndump_feat.load_token_information(token_info_path)
    holder_entries = make_mintndump_feat.load_holders(holders_path)
    pair_events_map = make_mintndump_feat.load_pair_events(pair_events_path, mint_token_info_map)

    holder_stats_by_token: Dict[str, make_mintndump_feat.HolderStats] = {}
    for token_idx, info in mint_token_info_map.items():
        holders = holder_entries.get(info.token_addr or "", [])
        holder_stats_by_token[token_idx] = make_mintndump_feat.compute_holder_stats(holders, info.pair_addr)

    mint_metrics_by_token: Dict[str, Dict[datetime, Dict[str, Optional[Decimal]]]] = {}
    for token_idx in tokens:
        timepoints = [ts for ts, _ in rows_by_token[token_idx]]
        events = pair_events_map.get(token_idx, [])
        mint_metrics_by_token[token_idx] = compute_mint_metrics_for_token(timepoints, events)

    # ------------------------------------------------------------------ #
    # Slow-drain rolling metrics
    # ------------------------------------------------------------------ #
    slow_events_map = make_slow_feat.read_events(pair_events_path, allowed_tokens=tokens)
    slow_metrics_by_token: Dict[str, Dict[datetime, Dict[str, Optional[Decimal]]]] = {}
    for token_idx in tokens:
        timepoints = [ts for ts, _ in rows_by_token[token_idx]]
        events = slow_events_map.get(token_idx, [])
        slow_metrics_by_token[token_idx] = compute_slow_metrics_for_token(timepoints, events)

    # ------------------------------------------------------------------ #
    # Merge rows
    # ------------------------------------------------------------------ #
    merged_rows: List[Dict[str, str]] = []
    for token_idx in sorted(tokens):
        holder_stats = holder_stats_by_token.get(token_idx)
        timepoint_metrics_mint = mint_metrics_by_token.get(token_idx, {})
        timepoint_metrics_slow = slow_metrics_by_token.get(token_idx, {})

        for ts, base_row in rows_by_token[token_idx]:
            mint_metrics = timepoint_metrics_mint.get(ts, {})
            slow_metrics = timepoint_metrics_slow.get(ts, {})
            merged_rows.append(
                build_output_row(
                    base_row=base_row,
                    mint_metrics=mint_metrics,
                    slow_metrics=slow_metrics,
                    holder_stats=holder_stats,
                )
            )

    write_output_csv(output_path, merged_rows)
    print(f"Merged {len(merged_rows)} rows across {len(tokens)} tokens to {output_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
