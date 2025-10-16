"""Shared helpers for rule evaluation."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

from .models import EventRecord


WINDOW_SECONDS_DEFAULT = 5


def decimal_to_float(value: Optional[Decimal]) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def window_events(
    events: Dict[str, List[EventRecord]],
    token_idx: str,
    start: datetime,
    window_seconds: int,
) -> List[str]:
    bucket = events.get(token_idx, [])
    if not bucket:
        return []
    end = start + timedelta(seconds=window_seconds)
    seen: Dict[str, None] = {}
    for record in bucket:
        if record.timestamp < start:
            continue
        if record.timestamp >= end:
            break
        if record.tx_hash:
            seen.setdefault(record.tx_hash, None)
    return list(seen.keys())
