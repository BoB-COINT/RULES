"""Utilities for loading features, events, and rules."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .models import EventRecord, FeatureWindow, SlowFeatureWindow
from .utils import WINDOW_SECONDS_DEFAULT

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


def parse_iso8601(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def parse_optional_decimal(value: str) -> Optional[Decimal]:
    value = value.strip()
    if not value:
        return None
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid decimal value '{value}'") from exc


def parse_optional_int(value: str) -> Optional[int]:
    value = value.strip()
    if not value:
        return None
    return int(value)


def sanitize_fieldnames(reader: csv.DictReader) -> None:
    if reader.fieldnames is None:
        return
    reader.fieldnames = [
        (name or "").strip().lstrip("\ufeff") for name in reader.fieldnames
    ]


def load_features(path: Path) -> Dict[str, List[FeatureWindow]]:
    features: Dict[str, List[FeatureWindow]] = {}
    with path.open(encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        sanitize_fieldnames(reader)
        for row in reader:
            token_idx = (row.get("token_addr_idx") or "").strip()
            if not token_idx:
                continue
            win_id = int(row["win_id"])
            win_start_ts = parse_iso8601(row["win_start_ts"])
            window = FeatureWindow(
                win_id=win_id,
                token_addr_idx=token_idx,
                win_start_ts=win_start_ts,
                win_start_block=parse_optional_int(row.get("win_start_block", "")),
                win_tx_count=int(row.get("win_tx_count", "0") or 0),
                win_blocks=parse_optional_int(row.get("win_blocks", "")),
                lp_start=parse_optional_decimal(row.get("lp_start", "")),
                lp_end=parse_optional_decimal(row.get("lp_end", "")),
                lp_dec=parse_optional_decimal(row.get("lp_dec", "")),
                burn_frac=parse_optional_decimal(row.get("burn_frac", "")),
                reserve_start=parse_optional_decimal(row.get("reserve_start", "")),
                reserve_end=parse_optional_decimal(row.get("reserve_end", "")),
                reserve_drop_frac=parse_optional_decimal(row.get("reserve_drop_frac", "")),
                lp_increase=parse_optional_decimal(row.get("lp_increase", "")),
                lp_burn_amount=parse_optional_decimal(row.get("lp_burn_amount", "")),
                mint_events=parse_optional_int(row.get("mint_events", "")),
                burn_events=parse_optional_int(row.get("burn_events", "")),
                swap_events=parse_optional_int(row.get("swap_events", "")),
                burn_to_mint_ratio=parse_optional_decimal(row.get("burn_to_mint_ratio", "")),
                time_since_last_mint_sec=parse_optional_decimal(row.get("time_since_last_mint_sec", "")),
                lp_peak_drop_frac=parse_optional_decimal(row.get("lp_peak_drop_frac", "")),
            )
            features.setdefault(token_idx, []).append(window)
    for bucket in features.values():
        bucket.sort(key=lambda w: w.win_id)
    return features


def load_slow_features(path: Path) -> Dict[str, List[SlowFeatureWindow]]:
    features: Dict[str, List[SlowFeatureWindow]] = {}
    with path.open(encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        sanitize_fieldnames(reader)
        for row in reader:
            token_idx = (row.get("token_addr_idx") or "").strip()
            if not token_idx:
                continue

            id_key = "win_id" if "win_id" in row else "window_id"
            try:
                window_id = int(row[id_key])
            except (KeyError, ValueError) as exc:
                raise ValueError("Missing or invalid window_id in slow feature row") from exc

            timestamp_key = "win_start_ts"
            if timestamp_key not in row or not row[timestamp_key].strip():
                timestamp_key = "window_start_ts"
            win_start_ts = parse_iso8601(row[timestamp_key])
            window = SlowFeatureWindow(
                window_id=window_id,
                token_addr_idx=token_idx,
                window_start_ts=win_start_ts,
                window_duration_seconds=WINDOW_SECONDS_DEFAULT,
                event_count=int(row.get("event_count", "0") or 0),
                burn_events=int(row.get("burn_events", "0") or 0),
                mint_events=int(row.get("mint_events", "0") or 0),
                swap_events=int(row.get("swap_events", "0") or 0),
                lp_start=parse_optional_decimal(row.get("lp_start", "")),
                lp_end=parse_optional_decimal(row.get("lp_end", "")),
                lp_drop_frac=parse_optional_decimal(row.get("lp_drop_frac", "")),
                lp_cum_drawdown=parse_optional_decimal(row.get("lp_cum_drawdown", "")),
                burn_to_mint_ratio=parse_optional_decimal(row.get("burn_to_mint_ratio", "")),
                time_since_last_mint_sec=parse_optional_decimal(row.get("time_since_last_mint_sec", "")),
                consecutive_drop_windows=int(row.get("consecutive_drop_windows", "0") or 0),
                reserve_token_start=parse_optional_decimal(row.get("reserve_token_start", "")),
                reserve_token_end=parse_optional_decimal(row.get("reserve_token_end", "")),
                reserve_token_drop_frac=parse_optional_decimal(row.get("reserve_token_drop_frac", "")),
                price_ratio_start=parse_optional_decimal(row.get("price_ratio_start", "")),
                price_ratio_end=parse_optional_decimal(row.get("price_ratio_end", "")),
                price_ratio_change=parse_optional_decimal(row.get("price_ratio_change", "")),
            )
            features.setdefault(token_idx, []).append(window)

    for bucket in features.values():
        bucket.sort(key=lambda w: w.window_id)
    return features


def load_events(path: Optional[Path]) -> Dict[str, List[EventRecord]]:
    if path is None or not path.exists():
        return {}
    events: Dict[str, List[EventRecord]] = {}
    with path.open(encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        sanitize_fieldnames(reader)
        for row in reader:
            token_idx = (row.get("token_addr_idx") or "").strip()
            if not token_idx:
                continue
            ts_raw = row.get("timestamp")
            if not ts_raw:
                continue
            timestamp = parse_iso8601(ts_raw)
            tx_hash = (row.get("tx_hash") or "").strip() or None
            events.setdefault(token_idx, []).append(
                EventRecord(timestamp=timestamp, tx_hash=tx_hash)
            )
    for bucket in events.values():
        bucket.sort(key=lambda e: e.timestamp)
    return events


def parse_scalar(token: str):
    lowered = token.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if token.startswith("\"") and token.endswith("\""):
        return token[1:-1]
    if token.startswith("'") and token.endswith("'"):
        return token[1:-1]
    try:
        if "." in token:
            return float(token)
        return int(token)
    except ValueError:
        return token


def simple_yaml_load(text: str):
    root: Dict[str, object] = {}
    stack: List[Tuple[int, Dict[str, object]]] = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            raise ValueError(f"Unsupported YAML line: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if not value:
            new_map: Dict[str, object] = {}
            current[key] = new_map
            stack.append((indent, new_map))
        else:
            current[key] = parse_scalar(value)
    return root


def load_rule(path: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)  # type: ignore[return-value]
    return simple_yaml_load(text)
