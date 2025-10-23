"""Shared data models for rule evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional


@dataclass
class FeatureWindow:
    win_id: int
    token_addr_idx: str
    win_start_ts: datetime
    win_start_block: Optional[int]
    win_tx_count: int
    win_blocks: Optional[int]
    lp_start: Optional[Decimal]
    lp_end: Optional[Decimal]
    lp_drop_amount: Optional[Decimal]
    burn_frac: Optional[Decimal]
    reserve_token_start: Optional[Decimal]
    reserve_token_end: Optional[Decimal]
    reserve_token_drop_frac: Optional[Decimal]
    lp_mint_amount: Optional[Decimal] = None
    lp_burn_amount: Optional[Decimal] = None
    mint_events: Optional[int] = None
    burn_events: Optional[int] = None
    swap_events: Optional[int] = None
    burn_to_mint_ratio: Optional[Decimal] = None
    time_since_last_mint_sec: Optional[Decimal] = None
    lp_peak_drop_frac: Optional[Decimal] = None
    lp_start_peak_frac: Optional[Decimal] = None
    swap_base_sell_volume: Optional[Decimal] = None
    swap_base_buy_volume: Optional[Decimal] = None
    cum_base_minted: Optional[Decimal] = None
    cum_base_burned: Optional[Decimal] = None
    cum_quote_minted: Optional[Decimal] = None
    cum_quote_burned: Optional[Decimal] = None


@dataclass
class EventRecord:
    timestamp: datetime
    tx_hash: Optional[str]


@dataclass
class SlowFeatureWindow:
    window_id: int
    token_addr_idx: str
    window_start_ts: datetime
    event_count: int
    burn_events: int
    mint_events: int
    swap_events: int
    lp_start: Optional[Decimal]
    lp_end: Optional[Decimal]
    lp_drop_frac: Optional[Decimal]
    lp_cum_drawdown: Optional[Decimal]
    burn_to_mint_ratio: Optional[Decimal]
    lp_burn_amount: Optional[Decimal]
    lp_mint_amount: Optional[Decimal]
    time_since_last_mint_sec: Optional[Decimal]
    consecutive_drop_windows: int
    reserve_token_start: Optional[Decimal]
    reserve_token_end: Optional[Decimal]
    reserve_token_drop_frac: Optional[Decimal]
    price_ratio_start: Optional[Decimal]
    price_ratio_end: Optional[Decimal]
    price_ratio_change: Optional[Decimal]
    lp_tx_ratio: Optional[Decimal]
    swap_activity_ratio: Optional[Decimal]


@dataclass
class MintDumpFeatureWindow:
    win_id: int
    token_addr_idx: str
    win_start_ts: datetime
    holder_top1_supply_pct: Optional[Decimal]
    holder_pair_supply_pct: Optional[Decimal]
    holder_top20_supply_pct: Optional[Decimal]
    sell_swap_count: int
    sell_base_volume: Optional[Decimal]
    sell_to_reserve_max_ratio: Optional[Decimal]
    sell_to_reserve_avg_ratio: Optional[Decimal]
    sell_base_abs_max: Optional[Decimal]
    sell_quote_volume: Optional[Decimal]
    sell_first_ts: Optional[datetime]
    sell_last_ts: Optional[datetime]
    sell_window_span_seconds: Optional[float]
