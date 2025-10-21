"""Detection evaluators for STE0201 rule variants."""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .models import EventRecord, FeatureWindow, MintDumpFeatureWindow, SlowFeatureWindow
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
    min_start_peak_ratio = float(params.get("min_start_peak_ratio", 0))
    start_peak_override_peak_drop = float(
        params.get("start_peak_ratio_override_peak_drop", 0)
    )
    pre_rug_sell_lookback = max(0, int(params.get("pre_rug_sell_lookback_windows", 12)))
    max_pre_rug_sell_ratio = float(
        params.get("max_pre_rug_sell_base_volume_ratio_for_detection", 0)
    )
    min_base_withdraw_ratio = float(
        params.get("min_net_base_withdraw_ratio_for_detection", 0)
    )
    min_base_withdraw_abs = float(
        params.get("min_net_base_withdraw_abs_for_detection", 0)
    )
    min_quote_profit_ratio = float(
        params.get("min_net_quote_profit_ratio_for_detection", 0)
    )
    min_quote_profit_abs = float(
        params.get("min_net_quote_profit_abs_for_detection", 0)
    )

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

    def entry_value(item: Dict[str, object], key: str) -> float:
        value = item.get(key)
        if value is None:
            return 0.0
        return float(value)

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
        burn_amount = window.lp_burn_amount or window.lp_drop_amount
        mint_amount = window.lp_mint_amount
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

        start_peak_ratio_value: Optional[float] = None
        if window.lp_start_peak_frac is not None:
            start_peak_ratio_value = float(window.lp_start_peak_frac)

        if min_start_peak_ratio > 0:
            if start_peak_ratio_value is None:
                reason_parts.append("lp_start_peak_frac unavailable")
                context_ok = False
            elif start_peak_ratio_value < min_start_peak_ratio:
                if (
                    start_peak_override_peak_drop > 0
                    and peak_drop_value is not None
                    and peak_drop_value >= start_peak_override_peak_drop
                ):
                    reason_parts.append(
                        "lp_start_peak_frac {:.3f} < {:.3f} but lp_peak_drop_frac {:.3f} >= {:.3f} (override)".format(
                            start_peak_ratio_value,
                            min_start_peak_ratio,
                            peak_drop_value,
                            start_peak_override_peak_drop,
                        )
                    )
                else:
                    reason_parts.append(
                        "lp_start_peak_frac {:.3f} < {:.3f}".format(
                            start_peak_ratio_value,
                            min_start_peak_ratio,
                        )
                    )
                    context_ok = False
            else:
                reason_parts.append(
                    "lp_start_peak_frac {:.3f} >= {:.3f}".format(
                        start_peak_ratio_value,
                        min_start_peak_ratio,
                    )
                )
        elif start_peak_ratio_value is not None:
            reason_parts.append("lp_start_peak_frac {:.3f}".format(start_peak_ratio_value))

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
            "lp_drop_amount": decimal_to_float(window.lp_drop_amount),
            "burn_frac": decimal_to_float(window.burn_frac),
            "reserve_token_start": decimal_to_float(window.reserve_token_start),
            "reserve_token_end": decimal_to_float(window.reserve_token_end),
            "reserve_token_drop_frac": decimal_to_float(window.reserve_token_drop_frac),
            "lp_mint_amount": decimal_to_float(window.lp_mint_amount),
            "lp_burn_amount": decimal_to_float(window.lp_burn_amount),
            "mint_events": window.mint_events,
            "burn_events": window.burn_events,
            "swap_events": window.swap_events,
            "burn_to_mint_ratio": decimal_to_float(window.burn_to_mint_ratio),
            "time_since_last_mint_sec": decimal_to_float(
                window.time_since_last_mint_sec
            ),
            "lp_peak_drop_frac": decimal_to_float(window.lp_peak_drop_frac),
            "lp_start_peak_frac": decimal_to_float(window.lp_start_peak_frac),
            "swap_base_sell_volume": decimal_to_float(window.swap_base_sell_volume),
            "swap_base_buy_volume": decimal_to_float(window.swap_base_buy_volume),
            "cum_base_minted": decimal_to_float(window.cum_base_minted),
            "cum_base_burned": decimal_to_float(window.cum_base_burned),
            "cum_quote_minted": decimal_to_float(window.cum_quote_minted),
            "cum_quote_burned": decimal_to_float(window.cum_quote_burned),
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

    for idx, entry in enumerate(window_entries):
        if not entry.get("triggered"):
            continue
        reasons = entry.get("_reason_parts", [])

        minted_total = entry_value(entry, "cum_base_minted")
        burn_total = entry_value(entry, "cum_base_burned")
        quote_minted_total = entry_value(entry, "cum_quote_minted")
        quote_burn_total = entry_value(entry, "cum_quote_burned")
        sell_volume_current = entry_value(entry, "swap_base_sell_volume")

        prev_entry = window_entries[idx - 1] if idx > 0 else None
        prev_minted_total = entry_value(prev_entry, "cum_base_minted") if prev_entry else 0.0
        prev_burn_total = entry_value(prev_entry, "cum_base_burned") if prev_entry else 0.0
        prev_quote_minted_total = entry_value(prev_entry, "cum_quote_minted") if prev_entry else 0.0
        prev_quote_burn_total = entry_value(prev_entry, "cum_quote_burned") if prev_entry else 0.0

        window_base_minted = max(0.0, minted_total - prev_minted_total)
        window_base_burned = max(0.0, burn_total - prev_burn_total)
        window_quote_minted = max(0.0, quote_minted_total - prev_quote_minted_total)
        window_quote_burned = max(0.0, quote_burn_total - prev_quote_burn_total)

        # 1. Pre-rug holder sell activity
        if pre_rug_sell_lookback > 0:
            sell_window_start = max(0, idx - pre_rug_sell_lookback + 1)
            sell_volume = 0.0
            for j in range(sell_window_start, idx + 1):
                sell_volume += entry_value(window_entries[j], "swap_base_sell_volume")
            sell_ratio = None
            if sell_volume > 0 and max_pre_rug_sell_ratio > 0:
                minted_start_value = (
                    entry_value(window_entries[sell_window_start - 1], "cum_base_minted")
                    if sell_window_start > 0
                    else 0.0
                )
                minted_in_lookback = minted_total - minted_start_value
                if minted_in_lookback > 0:
                    sell_ratio = sell_volume / minted_in_lookback
            if (
                max_pre_rug_sell_ratio > 0
                and sell_ratio is not None
                and sell_ratio >= max_pre_rug_sell_ratio
            ):
                reasons.append(
                    "swap_base_sell_volume_ratio {:.3f} >= {:.3f} (treated as normal exit)".format(
                        sell_ratio,
                        max_pre_rug_sell_ratio,
                    )
                )
                entry["triggered"] = False
                continue

        # 2. Net base token reclaimed vs supplied
        net_base_withdraw = max(0.0, window_base_burned - window_base_minted)
        base_ratio = None
        base_threshold_met = True
        if min_base_withdraw_ratio > 0:
            if window_base_minted > 0 and net_base_withdraw > 0:
                base_ratio = net_base_withdraw / window_base_minted
                if base_ratio < min_base_withdraw_ratio:
                    base_threshold_met = False
            elif window_base_minted > 0:
                base_ratio = 0.0
                base_threshold_met = False
            else:
                if net_base_withdraw <= 0:
                    base_threshold_met = False
        if base_threshold_met and min_base_withdraw_abs > 0:
            if net_base_withdraw < min_base_withdraw_abs:
                base_threshold_met = False
        if not base_threshold_met:
            reasons.append(
                "net base withdraw {:.3f} (ratio {}) below thresholds".format(
                    net_base_withdraw,
                    "{:.3f}".format(base_ratio) if base_ratio is not None else "N/A",
                )
            )
            entry["triggered"] = False
            continue
        else:
            reasons.append(
                "net base withdraw {:.3f} ratio {}".format(
                    net_base_withdraw,
                    "{:.3f}".format(base_ratio) if base_ratio is not None else "N/A",
                )
            )

        # 3. Net quote token profit
        quote_profit = max(0.0, window_quote_burned - window_quote_minted)
        quote_profit_ratio = None
        quote_threshold_met = True
        if min_quote_profit_ratio > 0:
            if window_quote_minted > 0 and quote_profit > 0:
                quote_profit_ratio = quote_profit / window_quote_minted
                if quote_profit_ratio < min_quote_profit_ratio:
                    quote_threshold_met = False
            elif window_quote_minted > 0:
                quote_profit_ratio = 0.0
                quote_threshold_met = False
            else:
                if quote_profit <= 0:
                    quote_threshold_met = False
        if quote_threshold_met and min_quote_profit_abs > 0:
            if quote_profit < min_quote_profit_abs:
                quote_threshold_met = False
        if not quote_threshold_met:
            reasons.append(
                "net quote profit {:.6f} (ratio {}) below thresholds".format(
                    quote_profit,
                    "{:.3f}".format(quote_profit_ratio)
                    if quote_profit_ratio is not None
                    else "N/A",
                )
            )
            entry["triggered"] = False
            continue
        else:
            reasons.append(
                "net quote profit {:.6f} ratio {}".format(
                    quote_profit,
                    "{:.3f}".format(quote_profit_ratio)
                    if quote_profit_ratio is not None
                    else "N/A",
                )
            )

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
    max_window_drop_raw = params.get("max_window_drop_frac")
    max_window_drop = (
        Decimal(str(max_window_drop_raw))
        if max_window_drop_raw is not None
        else Decimal("1")
    )
    min_consecutive = max(1, int(params.get("min_consecutive_windows", 3)))
    min_burn_events = max(0, int(params.get("min_burn_events", 1)))
    min_burn_to_mint_ratio = float(
        params.get(
            "min_burn_to_mint_ratio",
            params.get("burn_to_mint_ratio", 1.5),
        )
    )
    min_time_since_mint = float(
        params.get(
            "min_time_since_last_mint_sec",
            params.get("min_time_since_last_mint", 1800),
        )
    )
    max_price_ratio_change = float(params.get("max_price_ratio_change", 0.0))
    min_swap_activity = float(params.get("min_swap_activity", 0.0))
    min_lp_tx_ratio = float(params.get("min_lp_tx_ratio", 0.0))
    max_peak_drop_frac = float(params.get("max_peak_drop_frac", 0.0))

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
        burn_amount = window.lp_burn_amount
        mint_amount = window.lp_mint_amount
        time_since_mint = window.time_since_last_mint_sec
        price_change = window.price_ratio_change
        swap_activity_value = window.swap_activity_ratio
        lp_tx_ratio_value = window.lp_tx_ratio

        drop_ok = False
        if lp_drop is None:
            reasons.append("lp_drop_frac unavailable")
        else:
            if lp_drop < min_window_drop:
                reasons.append(
                    "lp_drop_frac {:.3f} < {:.3f}".format(
                        float(lp_drop), float(min_window_drop)
                    )
                )
            elif max_window_drop > 0 and lp_drop > max_window_drop:
                reasons.append(
                    "lp_drop_frac {:.3f} > {:.3f} (treated as rapid event)".format(
                        float(lp_drop), float(max_window_drop)
                    )
                )
            else:
                drop_ok = True
                reasons.append(
                    "lp_drop_frac {:.3f} within [{:.3f}, {:.3f}]".format(
                        float(lp_drop),
                        float(min_window_drop),
                        float(max_window_drop) if max_window_drop > 0 else float(lp_drop),
                    )
                )
            if drop_ok and max_peak_drop_frac > 0 and float(lp_drop) >= max_peak_drop_frac:
                drop_ok = False
                reasons.append(
                    "lp_drop_frac {:.3f} >= {:.3f} (exceeds soft-rug cap)".format(
                        float(lp_drop),
                        max_peak_drop_frac,
                    )
                )

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
        ratio_display = "N/A"
        ratio_value: Optional[float] = None
        if window.burn_events >= min_burn_events:
            if burn_ratio is not None:
                ratio_value = float(burn_ratio)
            elif burn_amount is not None:
                if mint_amount is None or mint_amount == 0:
                    ratio_value = float("inf")
                elif mint_amount > 0:
                    ratio_value = float(burn_amount / mint_amount)
            if ratio_value is not None:
                ratio_display = f"{ratio_value:.3f}" if ratio_value != float("inf") else "inf"
                ratio_ok = ratio_value >= min_burn_to_mint_ratio
            else:
                ratio_ok = True
            if ratio_value is None:
                reasons.append(
                    "burn_to_mint_ratio {}".format(
                        "N/A (no mint volume)" if burn_amount else "N/A"
                    )
                )
            else:
                reasons.append(
                    "burn_to_mint_ratio {} {} {:.3f}".format(
                        ratio_display,
                        ">=" if ratio_ok else "<",
                        min_burn_to_mint_ratio,
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

        swap_ok = True
        if min_swap_activity > 0:
            if window.swap_events <= 0:
                swap_ok = False
                reasons.append("swap_events 0 < required active pool")
            elif swap_activity_value is None:
                swap_ok = False
                reasons.append("swap_activity_ratio unavailable")
            else:
                swap_ratio_float = float(swap_activity_value)
                if swap_ratio_float < min_swap_activity:
                    swap_ok = False
                    reasons.append(
                        "swap_activity_ratio {:.3f} < {:.3f}".format(
                            swap_ratio_float,
                            min_swap_activity,
                        )
                    )
                else:
                    reasons.append(
                        "swap_activity_ratio {:.3f} >= {:.3f}".format(
                            swap_ratio_float,
                            min_swap_activity,
                        )
                    )
        elif swap_activity_value is not None:
            reasons.append(
                "swap_activity_ratio {:.3f}".format(float(swap_activity_value))
            )

        lp_tx_ok = True
        if min_lp_tx_ratio > 0:
            if lp_tx_ratio_value is None:
                lp_tx_ok = False
                reasons.append("lp_tx_ratio unavailable")
            else:
                lp_tx_float = float(lp_tx_ratio_value)
                if lp_tx_float < min_lp_tx_ratio:
                    lp_tx_ok = False
                    reasons.append(
                        "lp_tx_ratio {:.3f} < {:.3f}".format(
                            lp_tx_float,
                            min_lp_tx_ratio,
                        )
                    )
                else:
                    reasons.append(
                        "lp_tx_ratio {:.3f} >= {:.3f}".format(
                            lp_tx_float,
                            min_lp_tx_ratio,
                        )
                    )
        elif lp_tx_ratio_value is not None:
            reasons.append(
                "lp_tx_ratio {:.3f}".format(float(lp_tx_ratio_value))
            )

        triggered = all(
            [
                drop_ok,
                cum_ok,
                consecutive_ok,
                burn_ok,
                ratio_ok,
                mint_gap_ok,
                price_ok,
                swap_ok,
                lp_tx_ok,
            ]
        )

        burn_ratio_output = decimal_to_float(burn_ratio)
        if burn_ratio_output is None and ratio_value is not None and ratio_value != float("inf"):
            burn_ratio_output = ratio_value

        window_info = {
            "win_id": window.window_id,
            "win_start_ts": window.window_start_ts.isoformat().replace("+00:00", "Z"),
            "event_count": window.event_count,
            "burn_events": window.burn_events,
            "mint_events": window.mint_events,
            "swap_events": window.swap_events,
            "lp_start": decimal_to_float(window.lp_start),
            "lp_end": decimal_to_float(window.lp_end),
            "lp_drop_frac": decimal_to_float(lp_drop),
            "lp_cum_drawdown": decimal_to_float(cum_drawdown),
            "burn_to_mint_ratio": burn_ratio_output,
            "time_since_last_mint_sec": decimal_to_float(time_since_mint),
            "consecutive_drop_windows": window.consecutive_drop_windows,
            "reserve_token_start": decimal_to_float(window.reserve_token_start),
            "reserve_token_end": decimal_to_float(window.reserve_token_end),
            "reserve_token_drop_frac": decimal_to_float(window.reserve_token_drop_frac),
            "price_ratio_start": decimal_to_float(window.price_ratio_start),
            "price_ratio_end": decimal_to_float(window.price_ratio_end),
            "price_ratio_change": decimal_to_float(price_change),
            "lp_burn_amount": decimal_to_float(burn_amount),
            "lp_mint_amount": decimal_to_float(mint_amount),
            "lp_tx_ratio": decimal_to_float(lp_tx_ratio_value),
            "swap_activity_ratio": decimal_to_float(swap_activity_value),
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
            "lp_drop_amount": decimal_to_float(window.lp_drop_amount),
            "burn_frac": decimal_to_float(window.burn_frac),
            "reserve_token_start": decimal_to_float(window.reserve_token_start),
            "reserve_token_end": decimal_to_float(window.reserve_token_end),
            "reserve_token_drop_frac": decimal_to_float(window.reserve_token_drop_frac),
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
            "lp_drop_amount": decimal_to_float(window.lp_drop_amount),
            "burn_frac": decimal_to_float(window.burn_frac),
            "reserve_token_start": decimal_to_float(window.reserve_token_start),
            "reserve_token_end": decimal_to_float(window.reserve_token_end),
            "reserve_token_drop_frac": decimal_to_float(window.reserve_token_drop_frac),
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


def evaluate_mint_and_dump(
    token_idx: str,
    windows: Sequence[MintDumpFeatureWindow],
    params: Dict[str, object],
    scoring: Dict[str, object],
    evidence_cfg: Dict[str, object],
    events: Dict[str, List[EventRecord]],
):
    min_top_holder = Decimal(str(params.get("min_top_holder_supply_pct", 0)))
    min_pair_holder = Decimal(str(params.get("min_pair_contract_supply_pct", 0)))
    min_single_holder = Decimal(str(params.get("min_single_holder_supply_pct", 0)))
    min_top20_holder = Decimal(str(params.get("min_top20_cum_supply_pct", 0)))
    holder_snapshot_limit = float(params.get("holder_snapshot_lag_sec", 0) or 0)

    min_recent_mint_ratio = Decimal(str(params.get("min_recent_mint_supply_ratio", 0)))
    max_mint_to_dump_delay = float(params.get("max_mint_to_dump_delay_sec", 0) or 0)

    min_dump_swaps = int(params.get("min_dump_swap_count", 0) or 0)
    max_dump_span = float(params.get("dump_window_span_seconds", 0) or 0)
    min_dump_supply_ratio = Decimal(str(params.get("min_dump_sell_tot_supply_ratio", 0)))
    min_dump_reserve_ratio = Decimal(str(params.get("min_dump_sell_to_reserve_ratio", 0)))
    min_dump_abs = Decimal(str(params.get("min_dump_sell_abs", 0)))
    min_dump_to_mint_ratio = Decimal(str(params.get("min_dump_to_mint_ratio", 0)))
    min_quote_value = Decimal(str(params.get("min_quote_value_extracted", 0)))
    min_severity_threshold = Decimal(str(params.get("min_severity_threshold", min_dump_reserve_ratio)))

    base_score = float(scoring.get("base_score", 0))
    bonus_weight = float(scoring.get("bonus_weight", 0))
    detection_threshold = float(scoring.get("detection_threshold", 0))

    lookback = int(evidence_cfg.get("lookback_windows", 10) or 0)
    qualitative_reason = str(evidence_cfg.get("qualitative_reason", ""))

    _ = events  # events currently unused for mint-and-dump evaluation

    if not windows:
        token_state = {
            "score": 0.0,
            "last_win_id": -1,
            "severity": 0.0,
            "recent_windows": [],
        }
        return token_state, []

    window_entries: List[Dict[str, object]] = []
    severity_value = 0.0

    for window in windows:
        entry: Dict[str, object] = {
            "win_id": window.win_id,
            "win_start_ts": window.win_start_ts.isoformat().replace("+00:00", "Z"),
            "holder_top1_supply_pct": decimal_to_float(window.holder_top1_supply_pct),
            "holder_pair_supply_pct": decimal_to_float(window.holder_pair_supply_pct),
            "holder_top20_supply_pct": decimal_to_float(window.holder_top20_supply_pct),
            "holder_single_max_pct": decimal_to_float(window.holder_single_max_pct),
            "holder_entropy": decimal_to_float(window.holder_entropy),
            "holder_snapshot_lag_sec": window.holder_snapshot_lag_sec,
            "mint_event_count": window.mint_event_count,
            "mint_to_pair_supply_ratio": decimal_to_float(window.mint_to_pair_supply_ratio),
            "mint_to_pair_amount": decimal_to_float(window.mint_to_pair_amount),
            "dump_swap_count": window.dump_swap_count,
            "dump_sell_tot_supply_ratio": decimal_to_float(window.dump_sell_tot_supply_ratio),
            "dump_sell_to_reserve_max_ratio": decimal_to_float(window.dump_sell_to_reserve_max_ratio),
            "dump_sell_abs_max": decimal_to_float(window.dump_sell_abs_max),
            "dump_quote_out_volume": decimal_to_float(window.dump_quote_out_volume),
            "dump_window_span_seconds": window.dump_window_span_seconds,
            "mint_to_dump_latency_sec": window.mint_to_dump_latency_sec,
            "dump_to_mint_ratio": decimal_to_float(window.dump_to_mint_ratio),
            "triggered": False,
        }

        reason_parts: List[str] = []

        holder_ok = True

        top1_share = window.holder_top1_supply_pct
        if min_top_holder > 0:
            if top1_share is None:
                reason_parts.append("holder_top1_supply_pct unavailable")
                holder_ok = False
            elif top1_share < min_top_holder:
                reason_parts.append(
                    f"holder_top1_supply_pct {float(top1_share):.3f} < {float(min_top_holder):.3f}"
                )
                holder_ok = False
            else:
                reason_parts.append(
                    f"holder_top1_supply_pct {float(top1_share):.3f} >= {float(min_top_holder):.3f}"
                )

        pair_share = window.holder_pair_supply_pct
        if min_pair_holder > 0:
            if pair_share is None:
                reason_parts.append("holder_pair_supply_pct unavailable")
                holder_ok = False
            elif pair_share < min_pair_holder:
                reason_parts.append(
                    f"holder_pair_supply_pct {float(pair_share):.3f} < {float(min_pair_holder):.3f}"
                )
                holder_ok = False
            else:
                reason_parts.append(
                    f"holder_pair_supply_pct {float(pair_share):.3f} >= {float(min_pair_holder):.3f}"
                )

        single_max = window.holder_single_max_pct
        if min_single_holder > 0:
            if single_max is None:
                reason_parts.append("holder_single_max_pct unavailable")
                holder_ok = False
            elif single_max < min_single_holder:
                reason_parts.append(
                    f"holder_single_max_pct {float(single_max):.3f} < {float(min_single_holder):.3f}"
                )
                holder_ok = False
            else:
                reason_parts.append(
                    f"holder_single_max_pct {float(single_max):.3f} >= {float(min_single_holder):.3f}"
                )

        top20_sum = window.holder_top20_supply_pct
        if min_top20_holder > 0:
            if top20_sum is None:
                reason_parts.append("holder_top20_supply_pct unavailable")
                holder_ok = False
            elif top20_sum < min_top20_holder:
                reason_parts.append(
                    f"holder_top20_supply_pct {float(top20_sum):.3f} < {float(min_top20_holder):.3f}"
                )
                holder_ok = False
            else:
                reason_parts.append(
                    f"holder_top20_supply_pct {float(top20_sum):.3f} >= {float(min_top20_holder):.3f}"
                )

        if holder_snapshot_limit > 0:
            lag_value = window.holder_snapshot_lag_sec
            if lag_value is None:
                reason_parts.append("holder_snapshot_lag_sec unavailable")
            elif lag_value > holder_snapshot_limit:
                reason_parts.append(
                    f"holder_snapshot_lag_sec {lag_value:.0f}s > {holder_snapshot_limit:.0f}s"
                )
                holder_ok = False
            else:
                reason_parts.append(
                    f"holder_snapshot_lag_sec {lag_value:.0f}s <= {holder_snapshot_limit:.0f}s"
                )

        holder_violation = 0.0
        for holder_value in (
            window.holder_top1_supply_pct,
            window.holder_pair_supply_pct,
            window.holder_single_max_pct,
        ):
            if holder_value is None:
                continue
            if holder_value > 1:
                holder_violation = max(holder_violation, min(float(holder_value - 1), 1.0))

        mint_ok = True
        mint_ratio = window.mint_to_pair_supply_ratio
        if min_recent_mint_ratio > 0:
            if mint_ratio is None:
                reason_parts.append("mint_to_pair_supply_ratio unavailable")
                mint_ok = False
            elif mint_ratio < min_recent_mint_ratio:
                reason_parts.append(
                    f"mint_to_pair_supply_ratio {float(mint_ratio):.3f} < {float(min_recent_mint_ratio):.3f}"
                )
                mint_ok = False
            else:
                reason_parts.append(
                    f"mint_to_pair_supply_ratio {float(mint_ratio):.3f} >= {float(min_recent_mint_ratio):.3f}"
                )

        dump_ok = True

        if min_dump_swaps > 0:
            if window.dump_swap_count < min_dump_swaps:
                reason_parts.append(
                    f"dump_swap_count {window.dump_swap_count} < {min_dump_swaps}"
                )
                dump_ok = False
            else:
                reason_parts.append(
                    f"dump_swap_count {window.dump_swap_count} >= {min_dump_swaps}"
                )

        dump_span = window.dump_window_span_seconds
        if max_dump_span > 0:
            if dump_span is None:
                reason_parts.append("dump_window_span_seconds unavailable")
                dump_ok = False
            elif dump_span > max_dump_span:
                reason_parts.append(
                    f"dump_window_span_seconds {dump_span:.0f}s > {max_dump_span:.0f}s"
                )
                dump_ok = False
            else:
                reason_parts.append(
                    f"dump_window_span_seconds {dump_span:.0f}s <= {max_dump_span:.0f}s"
                )

        dump_supply_ratio = window.dump_sell_tot_supply_ratio
        if min_dump_supply_ratio > 0:
            if dump_supply_ratio is None:
                reason_parts.append("dump_sell_tot_supply_ratio unavailable")
                dump_ok = False
            elif dump_supply_ratio < min_dump_supply_ratio:
                reason_parts.append(
                    f"dump_sell_tot_supply_ratio {float(dump_supply_ratio):.3f} < {float(min_dump_supply_ratio):.3f}"
                )
                dump_ok = False
            else:
                reason_parts.append(
                    f"dump_sell_tot_supply_ratio {float(dump_supply_ratio):.3f} >= {float(min_dump_supply_ratio):.3f}"
                )

        dump_reserve_ratio = window.dump_sell_to_reserve_max_ratio
        if min_dump_reserve_ratio > 0:
            if dump_reserve_ratio is None:
                reason_parts.append("dump_sell_to_reserve_max_ratio unavailable")
                dump_ok = False
            elif dump_reserve_ratio < min_dump_reserve_ratio:
                reason_parts.append(
                    f"dump_sell_to_reserve_max_ratio {float(dump_reserve_ratio):.3f} < {float(min_dump_reserve_ratio):.3f}"
                )
                dump_ok = False
            else:
                reason_parts.append(
                    f"dump_sell_to_reserve_max_ratio {float(dump_reserve_ratio):.3f} >= {float(min_dump_reserve_ratio):.3f}"
                )

        dump_abs = window.dump_sell_abs_max
        if min_dump_abs > 0:
            if dump_abs is None:
                reason_parts.append("dump_sell_abs_max unavailable")
                dump_ok = False
            elif dump_abs < min_dump_abs:
                reason_parts.append(
                    f"dump_sell_abs_max {float(dump_abs):.3f} < {float(min_dump_abs):.3f}"
                )
                dump_ok = False
            else:
                reason_parts.append(
                    f"dump_sell_abs_max {float(dump_abs):.3f} >= {float(min_dump_abs):.3f}"
                )

        quote_value = window.dump_quote_out_volume
        if min_quote_value > 0:
            if quote_value is None:
                reason_parts.append("dump_quote_out_volume unavailable")
                dump_ok = False
            elif quote_value < min_quote_value:
                reason_parts.append(
                    f"dump_quote_out_volume {float(quote_value):.3f} < {float(min_quote_value):.3f}"
                )
                dump_ok = False
            else:
                reason_parts.append(
                    f"dump_quote_out_volume {float(quote_value):.3f} >= {float(min_quote_value):.3f}"
                )

        mint_latency = window.mint_to_dump_latency_sec
        if max_mint_to_dump_delay > 0:
            if mint_latency is None:
                reason_parts.append("mint_to_dump_latency_sec unavailable")
                dump_ok = False
            elif mint_latency > max_mint_to_dump_delay:
                reason_parts.append(
                    f"mint_to_dump_latency_sec {mint_latency:.0f}s > {max_mint_to_dump_delay:.0f}s"
                )
                dump_ok = False
            else:
                reason_parts.append(
                    f"mint_to_dump_latency_sec {mint_latency:.0f}s <= {max_mint_to_dump_delay:.0f}s"
                )

        dump_to_mint_ratio_value = window.dump_to_mint_ratio
        if min_dump_to_mint_ratio > 0:
            if dump_to_mint_ratio_value is None:
                reason_parts.append("dump_to_mint_ratio unavailable")
                dump_ok = False
            elif dump_to_mint_ratio_value < min_dump_to_mint_ratio:
                reason_parts.append(
                    f"dump_to_mint_ratio {float(dump_to_mint_ratio_value):.3f} < {float(min_dump_to_mint_ratio):.3f}"
                )
                dump_ok = False
            else:
                reason_parts.append(
                    f"dump_to_mint_ratio {float(dump_to_mint_ratio_value):.3f} >= {float(min_dump_to_mint_ratio):.3f}"
                )

        all_conditions_met = holder_ok and mint_ok and dump_ok
        entry["_reason_parts"] = reason_parts
        entry["triggered"] = all_conditions_met

        if all_conditions_met:
            reserve_ratio_val = decimal_to_float(dump_reserve_ratio) or 0.0
            supply_ratio_val = decimal_to_float(dump_supply_ratio) or 0.0
            severity_value = max(severity_value, reserve_ratio_val, supply_ratio_val)
            if holder_violation > 0:
                severity_value = max(severity_value, holder_violation)

        window_entries.append(entry)

    final_windows: List[Dict[str, object]] = []
    for entry in window_entries:
        reason_parts = entry.pop("_reason_parts", [])
        entry["reason"] = ", ".join(reason_parts) if reason_parts else qualitative_reason
        final_windows.append(entry)

    lookback_windows = (
        final_windows[-lookback:] if lookback and len(final_windows) > lookback else final_windows
    )

    triggered_windows = [entry for entry in final_windows if entry.get("triggered")]

    severity = min(1.0, severity_value) if triggered_windows else 0.0
    score = 0.0
    if triggered_windows:
        severity_floor = min(float(min_severity_threshold), 1.0)
        effective_severity = max(severity, severity_floor)
        score = base_score + effective_severity * bonus_weight
        score = min(score, 100.0)

    token_state = {
        "score": score,
        "last_win_id": windows[-1].win_id if windows else -1,
        "severity": severity,
        "recent_windows": lookback_windows,
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
