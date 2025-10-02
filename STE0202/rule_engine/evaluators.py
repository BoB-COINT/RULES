"""Detection evaluators for STE0201 rule variants."""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .models import EventRecord, FeatureWindow, SlowFeatureWindow
from .utils import WINDOW_SECONDS_DEFAULT, decimal_to_float, window_events


def evaluate_total_lp_pull(
    token_idx: str,
    windows: Sequence[FeatureWindow],
    params: Dict[str, object],
    scoring: Dict[str, object],
    evidence_cfg: Dict[str, object],
    events: Dict[str, List[EventRecord]],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Detect major liquidity removal via single or rapid multi-step burns."""

    window_seconds = int(params.get("window_seconds", WINDOW_SECONDS_DEFAULT))
    min_single_burn = Decimal(str(params.get("min_burn_frac", 0)))
    min_step_burn = Decimal(
        str(params.get("min_step_burn_frac", params.get("min_burn_frac", 0)))
    )
    min_total_drop = Decimal(str(params.get("min_total_drop_frac", 0)))
    min_sequence_windows = max(2, int(params.get("min_sequence_windows", 2)))
    max_sequence_windows = max(
        min_sequence_windows, int(params.get("max_sequence_windows", min_sequence_windows))
    )
    max_sequence_span_seconds = int(
        params.get(
            "max_sequence_span_seconds",
            window_seconds * max_sequence_windows,
        )
    )
    min_context_burn_events = max(0, int(params.get("min_context_burn_events", 0)))
    min_burn_to_mint_ratio = float(params.get("min_burn_to_mint_ratio", 0))
    min_time_since_mint = float(params.get("min_time_since_last_mint_sec", 0))
    min_peak_drop_frac = float(params.get("min_peak_drop_frac", 0))

    base_score = float(scoring.get("base_score", 0))
    bonus_weight = float(scoring.get("bonus_weight", 0))
    decay_per_window = float(scoring.get("decay_per_window", 0))
    detection_threshold = float(scoring.get("detection_threshold", 0))

    lookback = int(evidence_cfg.get("lookback_windows", 10))
    qualitative_reason = str(evidence_cfg.get("qualitative_reason", ""))

    score = 0.0
    last_win_id = -1
    window_entries: List[Dict[str, object]] = []
    step_candidates: List[bool] = []
    processed_windows: List[FeatureWindow] = []
    max_drop_observed = Decimal(0)

    for window in windows:
        if window.win_id <= last_win_id:
            continue

        single_ok = (
            min_single_burn > 0
            and window.burn_frac is not None
            and window.burn_frac >= min_single_burn
        )
        step_ok = (
            min_step_burn > 0
            and window.burn_frac is not None
            and window.burn_frac >= min_step_burn
        )

        reason_parts: List[str] = []
        if single_ok:
            reason_parts.append(
                "burn_frac {:.3f} >= {:.3f} (single-step)".format(
                    float(window.burn_frac), float(min_single_burn)
                )
            )
        elif step_ok:
            reason_parts.append(
                "burn_frac {:.3f} >= {:.3f} (multi-step candidate)".format(
                    float(window.burn_frac), float(min_step_burn)
                )
            )
        else:
            reason_parts.append("burn_frac below thresholds")

        burn_events = int(window.burn_events or 0)
        mint_events = int(window.mint_events or 0)
        swap_events = int(window.swap_events or 0)
        burn_amount = window.lp_burn_amount or window.lp_dec
        mint_amount = window.lp_increase
        ratio_val: Optional[float] = None
        ratio_label = None
        if burn_amount is not None and burn_amount > 0:
            if mint_amount is not None and mint_amount > 0:
                ratio_val = float(burn_amount / mint_amount)
                ratio_label = f"burn_to_mint_ratio {ratio_val:.3f}"
            else:
                ratio_val = float("inf")
                ratio_label = "burn_to_mint_ratio inf (mint_volume=0)"
        elif burn_amount is None:
            ratio_label = "burn volume unavailable"
        else:
            ratio_label = "burn volume 0"

        context_ok = True

        if min_context_burn_events > 0:
            if burn_events >= min_context_burn_events:
                reason_parts.append(
                    f"burn_events {burn_events} >= {min_context_burn_events}"
                )
            else:
                reason_parts.append(
                    f"burn_events {burn_events} < {min_context_burn_events}"
                )
                context_ok = False

        if min_burn_to_mint_ratio > 0:
            if ratio_val is None:
                reason_parts.append("burn-to-mint ratio unavailable")
                context_ok = False
            elif ratio_val < min_burn_to_mint_ratio:
                reason_parts.append(
                    f"{ratio_label} < {min_burn_to_mint_ratio:.3f}"
                )
                context_ok = False
            else:
                reason_parts.append(
                    f"{ratio_label} >= {min_burn_to_mint_ratio:.3f}"
                )
        elif ratio_label:
            reason_parts.append(ratio_label)

        mint_gap_seconds: Optional[float] = None
        if mint_events > 0:
            mint_gap_seconds = 0.0
        elif window.time_since_last_mint_sec is not None:
            mint_gap_seconds = float(window.time_since_last_mint_sec)

        if min_time_since_mint > 0:
            if mint_gap_seconds is None:
                reason_parts.append("time_since_last_mint unknown")
            elif mint_gap_seconds < min_time_since_mint:
                reason_parts.append(
                    "time_since_last_mint {:.0f}s < {:.0f}s".format(
                        mint_gap_seconds,
                        min_time_since_mint,
                    )
                )
                context_ok = False
            else:
                reason_parts.append(
                    "time_since_last_mint {:.0f}s >= {:.0f}s".format(
                        mint_gap_seconds,
                        min_time_since_mint,
                    )
                )
        elif mint_gap_seconds is not None:
            reason_parts.append(
                "time_since_last_mint {:.0f}s".format(mint_gap_seconds)
            )

        if swap_events:
            reason_parts.append(f"swap_events {swap_events}")

        peak_drop_value: Optional[float] = None
        if window.lp_peak_drop_frac is not None:
            peak_drop_value = float(window.lp_peak_drop_frac)

        if min_peak_drop_frac > 0:
            if peak_drop_value is None:
                reason_parts.append("lp_peak_drop_frac unavailable")
                context_ok = False
            elif peak_drop_value < min_peak_drop_frac:
                reason_parts.append(
                    "lp_peak_drop_frac {:.3f} < {:.3f}".format(
                        peak_drop_value,
                        min_peak_drop_frac,
                    )
                )
                context_ok = False
            else:
                reason_parts.append(
                    "lp_peak_drop_frac {:.3f} >= {:.3f}".format(
                        peak_drop_value,
                        min_peak_drop_frac,
                    )
                )
        elif peak_drop_value is not None:
            reason_parts.append(
                "lp_peak_drop_frac {:.3f}".format(peak_drop_value)
            )

        if context_ok and window.burn_frac is not None:
            max_drop_observed = max(
                max_drop_observed, Decimal(str(window.burn_frac))
            )

        single_ok = single_ok and context_ok
        step_ok = step_ok and context_ok

        tx_hashes = window_events(events, token_idx, window.win_start_ts, window_seconds)

        entry = {
            "win_id": window.win_id,
            "win_start_ts": window.win_start_ts.isoformat().replace("+00:00", "Z"),
            "win_tx_count": window.win_tx_count,
            "win_blocks": window.win_blocks,
            "lp_start": decimal_to_float(window.lp_start),
            "lp_end": decimal_to_float(window.lp_end),
            "lp_dec": decimal_to_float(window.lp_dec),
            "burn_frac": decimal_to_float(window.burn_frac),
            "reserve_start": decimal_to_float(window.reserve_start),
            "reserve_end": decimal_to_float(window.reserve_end),
            "reserve_drop_frac": decimal_to_float(window.reserve_drop_frac),
            "lp_increase": decimal_to_float(window.lp_increase),
            "lp_burn_amount": decimal_to_float(window.lp_burn_amount),
            "mint_events": window.mint_events,
            "burn_events": window.burn_events,
            "swap_events": window.swap_events,
            "burn_to_mint_ratio": decimal_to_float(window.burn_to_mint_ratio),
            "time_since_last_mint_sec": decimal_to_float(
                window.time_since_last_mint_sec
            ),
            "lp_peak_drop_frac": decimal_to_float(window.lp_peak_drop_frac),
            "triggered": single_ok,
            "tx_hashes": tx_hashes,
            "_reason_parts": reason_parts,
        }
        window_entries.append(entry)
        step_candidates.append(step_ok)
        processed_windows.append(window)
        last_win_id = window.win_id

    trigger_indices: Set[int] = set()
    sequences: List[Tuple[int, int, Decimal]] = []

    for start_idx, start_window in enumerate(processed_windows):
        if not step_candidates[start_idx]:
            continue
        if start_idx in trigger_indices:
            continue

        start_lp = start_window.lp_start
        if start_lp is None or start_lp <= 0:
            continue

        for offset in range(
            start_idx, min(start_idx + max_sequence_windows, len(processed_windows))
        ):
            if not step_candidates[offset]:
                break

            span_seconds = (
                processed_windows[offset].win_start_ts - start_window.win_start_ts
            ).total_seconds()
            if max_sequence_span_seconds > 0 and span_seconds > max_sequence_span_seconds:
                break

            end_lp = processed_windows[offset].lp_end
            if end_lp is None or end_lp < 0:
                continue
            if start_lp <= end_lp:
                continue

            total_drop = (start_lp - end_lp) / start_lp
            if offset - start_idx + 1 >= min_sequence_windows and total_drop >= min_total_drop:
                sequences.append((start_idx, offset, total_drop))
                for idx in range(start_idx, offset + 1):
                    trigger_indices.add(idx)
                break

    if sequences:
        for start_idx, end_idx, total_drop in sequences:
            seq_reason = (
                "cumulative burn {:.3f} >= {:.3f} across {} windows".format(
                    float(total_drop), float(min_total_drop), end_idx - start_idx + 1
                )
            )
            max_drop_observed = max(max_drop_observed, total_drop)
            for idx in range(start_idx, end_idx + 1):
                if idx < len(window_entries):
                    window_entries[idx]["triggered"] = True
                    window_entries[idx]["_reason_parts"].append(seq_reason)

    final_windows: List[Dict[str, object]] = []
    for entry in window_entries:
        entry_copy = dict(entry)
        reason_parts = entry_copy.pop("_reason_parts", [])
        entry_copy["reason"] = ", ".join(reason_parts) if reason_parts else qualitative_reason
        final_windows.append(entry_copy)

    recent_windows = (
        final_windows[-lookback:] if lookback and len(final_windows) > lookback else final_windows
    )

    detections: List[Dict[str, object]] = []
    triggered_windows = [w for w in final_windows if w.get("triggered")]
    severity = float(max_drop_observed) if max_drop_observed else 0.0
    if triggered_windows:
        score = base_score + severity * bonus_weight

    token_state = {
        "score": score,
        "last_win_id": last_win_id,
        "recent_windows": recent_windows,
        "severity": severity,
    }

    if triggered_windows and score >= detection_threshold:
        detections.append({
            "token_addr_idx": token_idx,
            "score": score,
            "threshold": detection_threshold,
            "status": "DETECTED",
            "qualitative_reason": qualitative_reason,
            "severity": severity,
            "windows": triggered_windows,
        })

    return token_state, detections


def evaluate_slow_drain(
    token_idx: str,
    windows: Sequence[SlowFeatureWindow],
    params: Dict[str, object],
    scoring: Dict[str, object],
    evidence_cfg: Dict[str, object],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    min_cum_drawdown = Decimal(str(params.get("min_cum_drawdown", 0.6)))
    min_window_drop = Decimal(str(params.get("min_window_drop_frac", 0.02)))
    min_consecutive = max(1, int(params.get("min_consecutive_windows", 3)))
    min_burn_events = max(0, int(params.get("min_burn_events", 1)))
    min_burn_to_mint_ratio = float(params.get("min_burn_to_mint_ratio", 1.5))
    min_time_since_mint = float(params.get("min_time_since_last_mint_sec", 1800))
    max_price_ratio_change = float(params.get("max_price_ratio_change", 0.0))

    base_score = float(scoring.get("base_score", 0))
    bonus_weight = float(scoring.get("bonus_weight", 0))
    detection_threshold = float(scoring.get("detection_threshold", 0))

    lookback = int(evidence_cfg.get("lookback_windows", 10))
    qualitative_reason = str(evidence_cfg.get("qualitative_reason", ""))

    window_records: List[Dict[str, object]] = []
    triggered_windows: List[Dict[str, object]] = []
    severity = 0.0

    for window in windows:
        reasons: List[str] = []

        lp_drop = window.lp_drop_frac
        cum_drawdown = window.lp_cum_drawdown
        burn_ratio = window.burn_to_mint_ratio
        time_since_mint = window.time_since_last_mint_sec
        price_change = window.price_ratio_change

        drop_ok = lp_drop is not None and lp_drop >= min_window_drop
        if drop_ok and lp_drop is not None:
            reasons.append(
                "lp_drop_frac {:.3f} >= {:.3f}".format(float(lp_drop), float(min_window_drop))
            )
        elif lp_drop is not None:
            reasons.append(
                "lp_drop_frac {:.3f} < {:.3f}".format(float(lp_drop), float(min_window_drop))
            )
        else:
            reasons.append("lp_drop_frac unavailable")

        cum_ok = cum_drawdown is not None and cum_drawdown >= min_cum_drawdown
        if cum_ok and cum_drawdown is not None:
            reasons.append(
                "lp_cum_drawdown {:.3f} >= {:.3f}".format(
                    float(cum_drawdown), float(min_cum_drawdown)
                )
            )
        elif cum_drawdown is not None:
            reasons.append(
                "lp_cum_drawdown {:.3f} < {:.3f}".format(
                    float(cum_drawdown), float(min_cum_drawdown)
                )
            )
        else:
            reasons.append("lp_cum_drawdown unavailable")

        consecutive_ok = window.consecutive_drop_windows >= min_consecutive
        reasons.append(
            "consecutive_drop_windows {}{} {}".format(
                window.consecutive_drop_windows,
                ">=" if consecutive_ok else "<",
                min_consecutive,
            )
        )

        burn_ok = window.burn_events >= min_burn_events
        reasons.append(
            "burn_events {}{} {}".format(
                window.burn_events,
                ">=" if burn_ok else "<",
                min_burn_events,
            )
        )

        ratio_ok = True
        if window.burn_events >= min_burn_events:
            if burn_ratio is None:
                ratio_ok = True
            else:
                ratio_ok = float(burn_ratio) >= min_burn_to_mint_ratio
            reasons.append(
                "burn_to_mint_ratio {} {}".format(
                    "N/A" if burn_ratio is None else f"{float(burn_ratio):.3f}",
                    ">=" + f" {min_burn_to_mint_ratio}" if ratio_ok else f"< {min_burn_to_mint_ratio}",
                )
            )
        else:
            reasons.append("burn_to_mint_ratio skipped (insufficient burn events)")

        mint_gap_ok = True
        if min_time_since_mint > 0:
            if time_since_mint is None:
                mint_gap_ok = True
                reasons.append("time_since_last_mint unknown")
            else:
                gap_value = float(time_since_mint)
                mint_gap_ok = gap_value >= min_time_since_mint
                reasons.append(
                    "time_since_last_mint {:.0f}s {} {:.0f}s".format(
                        gap_value,
                        ">=" if mint_gap_ok else "<",
                        min_time_since_mint,
                    )
                )

        price_ok = True
        if max_price_ratio_change > 0 and price_change is not None:
            price_ok = abs(float(price_change)) <= max_price_ratio_change
            reasons.append(
                "price_ratio_change {:.6f} {} {:.6f}".format(
                    float(price_change),
                    "within" if price_ok else "exceeds",
                    max_price_ratio_change,
                )
            )
        elif max_price_ratio_change > 0:
            reasons.append("price_ratio_change unavailable")

        triggered = all(
            [
                drop_ok,
                cum_ok,
                consecutive_ok,
                burn_ok,
                ratio_ok,
                mint_gap_ok,
                price_ok,
            ]
        )

        window_info = {
            "win_id": window.window_id,
            "win_start_ts": window.window_start_ts.isoformat().replace("+00:00", "Z"),
            "win_duration_seconds": window.window_duration_seconds,
            "event_count": window.event_count,
            "burn_events": window.burn_events,
            "mint_events": window.mint_events,
            "swap_events": window.swap_events,
            "lp_start": decimal_to_float(window.lp_start),
            "lp_end": decimal_to_float(window.lp_end),
            "lp_drop_frac": decimal_to_float(lp_drop),
            "lp_cum_drawdown": decimal_to_float(cum_drawdown),
            "burn_to_mint_ratio": decimal_to_float(burn_ratio),
            "time_since_last_mint_sec": decimal_to_float(time_since_mint),
            "consecutive_drop_windows": window.consecutive_drop_windows,
            "reserve_token_start": decimal_to_float(window.reserve_token_start),
            "reserve_token_end": decimal_to_float(window.reserve_token_end),
            "reserve_token_drop_frac": decimal_to_float(window.reserve_token_drop_frac),
            "price_ratio_start": decimal_to_float(window.price_ratio_start),
            "price_ratio_end": decimal_to_float(window.price_ratio_end),
            "price_ratio_change": decimal_to_float(price_change),
            "triggered": triggered,
            "reason": "; ".join(reasons) if reasons else qualitative_reason,
        }

        window_records.append(window_info)
        if triggered:
            triggered_windows.append(window_info)
            if cum_drawdown is not None:
                severity = max(severity, float(min(cum_drawdown, Decimal("1.0"))))

    score = 0.0
    if triggered_windows:
        score = base_score + severity * bonus_weight

    recent_windows = (
        window_records[-lookback:] if lookback and len(window_records) > lookback else window_records
    )

    token_state = {
        "score": score,
        "last_window_id": windows[-1].window_id if windows else -1,
        "recent_windows": recent_windows,
        "severity": severity,
    }

    detections: List[Dict[str, object]] = []
    if triggered_windows and score >= detection_threshold:
        detections.append(
            {
                "token_addr_idx": token_idx,
                "score": score,
                "threshold": detection_threshold,
                "status": "DETECTED",
                "qualitative_reason": qualitative_reason,
                "severity": severity,
                "windows": triggered_windows,
            }
        )

    return token_state, detections


def evaluate_single_drop(
    token_idx: str,
    windows: Sequence[FeatureWindow],
    params: Dict[str, object],
    scoring: Dict[str, object],
    evidence_cfg: Dict[str, object],
    events: Dict[str, List[EventRecord]],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    window_seconds = int(params.get("window_seconds", WINDOW_SECONDS_DEFAULT))
    min_burn = Decimal(str(params.get("min_burn_frac", 0)))

    score_per_hit = float(scoring.get("score_per_hit", 0))
    decay_per_window = float(scoring.get("decay_per_window", 0))
    detection_threshold = float(scoring.get("detection_threshold", 0))

    lookback = int(evidence_cfg.get("lookback_windows", 10))
    qualitative_reason = str(evidence_cfg.get("qualitative_reason", ""))

    score = 0.0
    last_win_id = -1
    window_records: List[Dict[str, object]] = []
    recent_windows: List[Dict[str, object]] = []

    detections: List[Dict[str, object]] = []

    for window in windows:
        if window.win_id <= last_win_id:
            continue

        if decay_per_window:
            score = max(0.0, score - decay_per_window)

        triggered = False
        reason_parts: List[str] = []

        burn_ok = window.burn_frac is not None and window.burn_frac >= min_burn
        if burn_ok:
            triggered = True
            score += score_per_hit
            reason_parts.append(
                f"burn_frac {float(window.burn_frac)} >= {float(min_burn)}"
            )
        else:
            reason_parts.append("burn_frac below threshold")

        tx_hashes = window_events(events, token_idx, window.win_start_ts, window_seconds)

        window_info = {
            "win_id": window.win_id,
            "win_start_ts": window.win_start_ts.isoformat().replace("+00:00", "Z"),
            "win_tx_count": window.win_tx_count,
            "win_blocks": window.win_blocks,
            "lp_start": decimal_to_float(window.lp_start),
            "lp_end": decimal_to_float(window.lp_end),
            "lp_dec": decimal_to_float(window.lp_dec),
            "burn_frac": decimal_to_float(window.burn_frac),
            "reserve_start": decimal_to_float(window.reserve_start),
            "reserve_end": decimal_to_float(window.reserve_end),
            "reserve_drop_frac": decimal_to_float(window.reserve_drop_frac),
            "triggered": triggered,
            "reason": ", ".join(reason_parts) if reason_parts else qualitative_reason,
            "tx_hashes": tx_hashes,
        }
        window_records.append(window_info)
        if lookback and len(window_records) > lookback:
            recent_windows = window_records[-lookback:]
        else:
            recent_windows = list(window_records)

        last_win_id = window.win_id

    token_state = {
        "score": score,
        "last_win_id": last_win_id,
        "recent_windows": recent_windows,
    }

    if score >= detection_threshold:
        triggered_windows = [w for w in window_records if w.get("triggered")]
        if triggered_windows:
            detections.append({
                "token_addr_idx": token_idx,
                "score": score,
                "threshold": detection_threshold,
                "status": "DETECTED",
                "qualitative_reason": qualitative_reason,
                "windows": triggered_windows,
            })

    return token_state, detections


def evaluate_multi_drop(
    token_idx: str,
    windows: Sequence[FeatureWindow],
    params: Dict[str, object],
    scoring: Dict[str, object],
    evidence_cfg: Dict[str, object],
    events: Dict[str, List[EventRecord]],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    window_seconds = int(params.get("window_seconds", WINDOW_SECONDS_DEFAULT))
    min_step_burn = Decimal(str(params.get("min_step_burn_frac", 0)))
    min_total_drop = Decimal(str(params.get("min_total_drop_frac", 0)))
    min_sequence_windows = max(2, int(params.get("min_sequence_windows", 2)))
    max_sequence_windows = max(
        min_sequence_windows, int(params.get("max_sequence_windows", min_sequence_windows))
    )
    max_sequence_span_seconds = int(
        params.get(
            "max_sequence_span_seconds",
            window_seconds * max_sequence_windows,
        )
    )

    score_per_hit = float(scoring.get("score_per_hit", 0))
    decay_per_window = float(scoring.get("decay_per_window", 0))
    detection_threshold = float(scoring.get("detection_threshold", 0))

    lookback = int(evidence_cfg.get("lookback_windows", 10))
    qualitative_reason = str(evidence_cfg.get("qualitative_reason", ""))

    score = 0.0
    last_win_id = -1
    window_entries: List[Dict[str, object]] = []
    step_candidates: List[bool] = []
    processed_windows: List[FeatureWindow] = []

    for window in windows:
        if window.win_id <= last_win_id:
            continue

        if decay_per_window:
            score = max(0.0, score - decay_per_window)

        reason_parts: List[str] = []
        burn_ok = window.burn_frac is not None and window.burn_frac >= min_step_burn
        if burn_ok:
            reason_parts.append(
                f"burn_frac {float(window.burn_frac)} >= {float(min_step_burn)}"
            )
        else:
            reason_parts.append("burn_frac below step threshold")

        tx_hashes = window_events(events, token_idx, window.win_start_ts, window_seconds)

        entry = {
            "win_id": window.win_id,
            "win_start_ts": window.win_start_ts.isoformat().replace("+00:00", "Z"),
            "win_tx_count": window.win_tx_count,
            "win_blocks": window.win_blocks,
            "lp_start": decimal_to_float(window.lp_start),
            "lp_end": decimal_to_float(window.lp_end),
            "lp_dec": decimal_to_float(window.lp_dec),
            "burn_frac": decimal_to_float(window.burn_frac),
            "reserve_start": decimal_to_float(window.reserve_start),
            "reserve_end": decimal_to_float(window.reserve_end),
            "reserve_drop_frac": decimal_to_float(window.reserve_drop_frac),
            "triggered": False,
            "tx_hashes": tx_hashes,
            "_reason_parts": reason_parts,
        }
        window_entries.append(entry)
        step_candidates.append(burn_ok)
        processed_windows.append(window)
        last_win_id = window.win_id

    trigger_indices: Set[int] = set()
    sequences: List[Tuple[int, int, Decimal]] = []

    for start_idx, start_window in enumerate(processed_windows):
        if not step_candidates[start_idx]:
            continue
        if start_idx in trigger_indices:
            continue

        start_lp = start_window.lp_start
        if start_lp is None or start_lp <= 0:
            continue

        for offset in range(
            start_idx, min(start_idx + max_sequence_windows, len(processed_windows))
        ):
            if not step_candidates[offset]:
                break

            span_seconds = (
                processed_windows[offset].win_start_ts - start_window.win_start_ts
            ).total_seconds()
            if max_sequence_span_seconds > 0 and span_seconds > max_sequence_span_seconds:
                break

            end_lp = processed_windows[offset].lp_end
            if end_lp is None or end_lp < 0:
                continue
            if start_lp <= end_lp:
                continue

            total_drop = (start_lp - end_lp) / start_lp
            if offset - start_idx + 1 >= min_sequence_windows and total_drop >= min_total_drop:
                sequences.append((start_idx, offset, total_drop))
                for idx in range(start_idx, offset + 1):
                    trigger_indices.add(idx)
                break

    if sequences:
        for start_idx, end_idx, total_drop in sequences:
            score += score_per_hit
            seq_reason = (
                f"cumulative burn {float(total_drop)} >= {float(min_total_drop)}"
                f" across {end_idx - start_idx + 1} windows"
            )
            for idx in range(start_idx, end_idx + 1):
                if idx < len(window_entries):
                    window_entries[idx]["triggered"] = True
                    window_entries[idx]["_reason_parts"].append(seq_reason)

    final_windows: List[Dict[str, object]] = []
    for entry in window_entries:
        entry_copy = dict(entry)
        reason_parts = entry_copy.pop("_reason_parts", [])
        entry_copy["reason"] = ", ".join(reason_parts) if reason_parts else qualitative_reason
        final_windows.append(entry_copy)

    recent_windows = (
        final_windows[-lookback:] if lookback and len(final_windows) > lookback else final_windows
    )

    token_state = {
        "score": score,
        "last_win_id": last_win_id,
        "recent_windows": recent_windows,
    }

    detections: List[Dict[str, object]] = []
    triggered_windows = [w for w in final_windows if w.get("triggered")]
    if score >= detection_threshold and triggered_windows:
        detections.append({
            "token_addr_idx": token_idx,
            "score": score,
            "threshold": detection_threshold,
            "status": "DETECTED",
            "qualitative_reason": qualitative_reason,
            "windows": triggered_windows,
        })

    return token_state, detections
