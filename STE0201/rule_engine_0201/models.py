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
    lp_dec: Optional[Decimal]
    burn_frac: Optional[Decimal]
    reserve_start: Optional[Decimal]
    reserve_end: Optional[Decimal]
    reserve_drop_frac: Optional[Decimal]
    lp_increase: Optional[Decimal] = None
    lp_burn_amount: Optional[Decimal] = None
    mint_events: Optional[int] = None
    burn_events: Optional[int] = None
    swap_events: Optional[int] = None
    burn_to_mint_ratio: Optional[Decimal] = None
    time_since_last_mint_sec: Optional[Decimal] = None
    lp_peak_drop_frac: Optional[Decimal] = None


@dataclass
class EventRecord:
    timestamp: datetime
    tx_hash: Optional[str]


@dataclass
class SlowFeatureWindow:
    window_id: int
    token_addr_idx: str
    window_start_ts: datetime
    window_duration_seconds: int
    event_count: int
    burn_events: int
    mint_events: int
    swap_events: int
    lp_start: Optional[Decimal]
    lp_end: Optional[Decimal]
    lp_drop_frac: Optional[Decimal]
    lp_cum_drawdown: Optional[Decimal]
    burn_to_mint_ratio: Optional[Decimal]
    time_since_last_mint_sec: Optional[Decimal]
    consecutive_drop_windows: int
    reserve_token_start: Optional[Decimal]
    reserve_token_end: Optional[Decimal]
    reserve_token_drop_frac: Optional[Decimal]
    price_ratio_start: Optional[Decimal]
    price_ratio_end: Optional[Decimal]
    price_ratio_change: Optional[Decimal]
