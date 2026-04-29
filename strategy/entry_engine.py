"""
entry_engine.py
---------------
Leading-entry engine for early altcoin breakouts.

This module is intentionally independent from signal_engine.py. It does not use
lagging confirmation indicators such as RSI/ADX/MACD for entry approval. The
goal is to catch the first volume expansion after a quiet accumulation range.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger("entry_engine")


MIN_BARS = 30
SILENCE_LOOKBACK = 20
MIN_SILENT_BARS = 15
SPIKE_MULTIPLIER = 3.0
HAMMER_VOLUME_MULTIPLIER = 2.0
HAMMER_WICK_BODY_RATIO = 2.0
MAX_PRICE_RISE_PCT = 15.0
MAX_ABSORPTION_ATR_MULTIPLIER = 1.8
BTC_DROP_LOOKBACK = 3
BTC_DROP_BLOCK_PCT = -2.5
ALT_ROTATION_MIN_BREADTH = 5
ALT_ROTATION_MIN_VOLUME_SPIKE = 2.0
ALT_ROTATION_MAX_RISE_PCT = 35.0
ALT_ROTATION_MIN_CANDLE_RETURN_PCT = 1.0


@dataclass
class EntryDecision:
    """Result returned by the leading entry engine."""

    signal: str = "HOLD"
    should_enter: bool = False
    direction: str = "LONG"
    reject_reason: str = ""
    entry_price: float = 0.0
    score: int = 0
    metrics: dict = field(default_factory=dict)


def check(
    symbol: str,
    df_5m: pd.DataFrame,
    df_15m: Optional[pd.DataFrame] = None,
    btc_df: Optional[pd.DataFrame] = None,
    stage1_meta: Optional[dict] = None,
) -> EntryDecision:
    """
    Evaluate a symbol for immediate early LONG entry.

    Entry triggers:
    - priority 1: 15m hammer absorption candle + volume >= 2x quiet average
    - priority 2: 5m quiet range + first volume spike + bullish candle
    - common block: price has not already extended more than 15% and BTC is
      not in a short-term sharp drop

    Parameters
    ----------
    symbol:
        CCXT symbol, for example "TURTLE/USDT:USDT".
    df_5m:
        5m OHLCV dataframe. Must contain open/high/low/close/volume.
    df_15m:
        Optional 15m OHLCV dataframe for hammer/Wyckoff absorption detection.
    btc_df:
        Optional BTC dataframe on the same or similar timeframe.
    stage1_meta:
        Optional metadata from the scanner's first-stage volume filter.
    """
    decision = EntryDecision()

    valid, reason = _validate_df(df_5m, "5m")
    if not valid:
        decision.reject_reason = reason
        return decision

    df_5m_clean = _clean_ohlcv(df_5m)
    if len(df_5m_clean) < MIN_BARS:
        decision.reject_reason = f"not enough 5m candles: {len(df_5m_clean)} < {MIN_BARS}"
        return decision

    squeeze_ok, squeeze_reason, squeeze_metrics = _check_5m_first_spike(df_5m_clean)
    hammer_ok, hammer_reason, hammer_metrics = _check_15m_hammer(df_15m)
    rotation_ok, rotation_reason, rotation_metrics = _check_alt_rotation_entry(
        df_5m_clean,
        stage1_meta,
    )

    metrics = {
        "squeeze_5m": squeeze_metrics,
        "hammer_15m": hammer_metrics,
        "alt_rotation": rotation_metrics,
    }
    if stage1_meta:
        metrics["stage1"] = stage1_meta

    btc_block_reason = _btc_drop_reason(btc_df)
    if btc_block_reason:
        decision.reject_reason = btc_block_reason
        decision.metrics = metrics
        logger.debug(f"[{symbol}] ENTRY HOLD - {btc_block_reason}")
        return decision

    current_close = float(df_5m_clean["close"].iloc[-1])
    decision.entry_price = current_close
    decision.metrics = metrics

    if not hammer_ok and not squeeze_ok and not rotation_ok:
        decision.reject_reason = (
            f"hammer_15m: {hammer_reason}; "
            f"squeeze_5m: {squeeze_reason}; "
            f"alt_rotation: {rotation_reason}"
        )
        logger.debug(f"[{symbol}] ENTRY HOLD - {decision.reject_reason}")
        return decision

    if hammer_ok:
        trigger = "HAMMER_15M"
    elif squeeze_ok:
        trigger = "FIRST_SPIKE_5M"
    else:
        trigger = "ALT_ROTATION"
    decision.signal = "LONG"
    decision.should_enter = True
    decision.score = _score(trigger, metrics)
    decision.metrics["trigger"] = trigger
    logger.info(
        f"[{symbol}] ENTRY LONG - {trigger} "
        f"score={decision.score} price={current_close:.8f}"
    )
    return decision


def _validate_df(df: Optional[pd.DataFrame], timeframe: str) -> tuple[bool, str]:
    if df is None or df.empty:
        return False, f"{timeframe} dataframe is empty"
    missing = {"open", "high", "low", "close", "volume"} - set(df.columns)
    if missing:
        return False, f"{timeframe} dataframe missing columns: {sorted(missing)}"
    return True, ""


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy().dropna(subset=["open", "high", "low", "close", "volume"])


def _check_5m_first_spike(df: pd.DataFrame) -> tuple[bool, str, dict]:
    current = df.iloc[-1]
    quiet = df.iloc[-(SILENCE_LOOKBACK + 1):-1]

    quiet_avg = float(quiet["volume"].mean())
    if quiet_avg <= 0:
        return False, "quiet average volume is zero", {}

    current_volume = float(current["volume"])
    spike_ratio = current_volume / quiet_avg
    silent_bars = int((quiet["volume"] <= quiet_avg).sum())
    prior_spike_count = int((quiet["volume"] >= quiet_avg * SPIKE_MULTIPLIER).sum())

    current_open = float(current["open"])
    current_high = float(current["high"])
    current_low = float(current["low"])
    current_close = float(current["close"])
    quiet_low = float(quiet["low"].min())
    price_rise_pct = _price_rise_pct(current_close, quiet_low)
    candle_return_pct = ((current_close - current_open) / current_open * 100.0) if current_open > 0 else 0.0
    atr = _atr(df)
    range_atr_ratio = ((current_high - current_low) / atr) if atr > 0 else 999.0

    metrics = {
        "quiet_avg_volume": round(quiet_avg, 4),
        "current_volume": round(current_volume, 4),
        "spike_ratio": round(spike_ratio, 2),
        "silent_bars": silent_bars,
        "prior_spike_count": prior_spike_count,
        "quiet_low": round(quiet_low, 8),
        "price_rise_pct": round(price_rise_pct, 2),
        "candle_return_pct": round(candle_return_pct, 2),
        "range_atr_ratio": round(range_atr_ratio, 2),
    }

    reject_reasons = []
    if silent_bars < MIN_SILENT_BARS:
        reject_reasons.append(
            f"quiet bars {silent_bars}/{SILENCE_LOOKBACK} < {MIN_SILENT_BARS}"
        )
    if spike_ratio < SPIKE_MULTIPLIER:
        reject_reasons.append(
            f"volume spike {spike_ratio:.1f}x < {SPIKE_MULTIPLIER:.1f}x"
        )
    if prior_spike_count > 0:
        reject_reasons.append(f"not first spike: prior spikes={prior_spike_count}")
    if price_rise_pct >= MAX_PRICE_RISE_PCT:
        reject_reasons.append(
            f"already extended +{price_rise_pct:.1f}% >= {MAX_PRICE_RISE_PCT:.1f}%"
        )
    if current_close <= current_open:
        reject_reasons.append("current candle is not bullish")
    if range_atr_ratio > MAX_ABSORPTION_ATR_MULTIPLIER:
        reject_reasons.append(
            f"candle range {range_atr_ratio:.1f} ATR > {MAX_ABSORPTION_ATR_MULTIPLIER:.1f} ATR"
        )

    return not reject_reasons, "; ".join(reject_reasons), metrics


def _check_15m_hammer(df_15m: Optional[pd.DataFrame]) -> tuple[bool, str, dict]:
    valid, reason = _validate_df(df_15m, "15m")
    if not valid:
        return False, reason, {}

    df = _clean_ohlcv(df_15m)
    if len(df) < MIN_BARS:
        return False, f"not enough 15m candles: {len(df)} < {MIN_BARS}", {}

    current = df.iloc[-1]
    quiet = df.iloc[-(SILENCE_LOOKBACK + 1):-1]
    quiet_avg = float(quiet["volume"].mean())
    if quiet_avg <= 0:
        return False, "15m quiet average volume is zero", {}

    current_open = float(current["open"])
    current_high = float(current["high"])
    current_low = float(current["low"])
    current_close = float(current["close"])
    current_volume = float(current["volume"])

    body = abs(current_close - current_open)
    body_floor = max(current_close * 0.0001, 1e-12)
    body_for_ratio = max(body, body_floor)
    lower_wick = max(0.0, min(current_open, current_close) - current_low)
    upper_wick = max(0.0, current_high - max(current_open, current_close))
    wick_body_ratio = lower_wick / body_for_ratio
    volume_ratio = current_volume / quiet_avg
    quiet_low = float(quiet["low"].min())
    price_rise_pct = _price_rise_pct(current_close, quiet_low)

    metrics = {
        "quiet_avg_volume": round(quiet_avg, 4),
        "current_volume": round(current_volume, 4),
        "volume_ratio": round(volume_ratio, 2),
        "body": round(body, 8),
        "lower_wick": round(lower_wick, 8),
        "upper_wick": round(upper_wick, 8),
        "wick_body_ratio": round(wick_body_ratio, 2),
        "price_rise_pct": round(price_rise_pct, 2),
    }

    reject_reasons = []
    if wick_body_ratio < HAMMER_WICK_BODY_RATIO:
        reject_reasons.append(
            f"lower wick/body {wick_body_ratio:.1f} < {HAMMER_WICK_BODY_RATIO:.1f}"
        )
    if volume_ratio < HAMMER_VOLUME_MULTIPLIER:
        reject_reasons.append(
            f"15m volume {volume_ratio:.1f}x < {HAMMER_VOLUME_MULTIPLIER:.1f}x"
        )
    if price_rise_pct >= MAX_PRICE_RISE_PCT:
        reject_reasons.append(
            f"already extended +{price_rise_pct:.1f}% >= {MAX_PRICE_RISE_PCT:.1f}%"
        )
    if current_close <= current_open:
        reject_reasons.append("15m hammer candle is not bullish")

    return not reject_reasons, "; ".join(reject_reasons), metrics


def _check_alt_rotation_entry(
    df: pd.DataFrame,
    stage1_meta: Optional[dict],
) -> tuple[bool, str, dict]:
    rotation = (stage1_meta or {}).get("alt_rotation", {})
    active = bool(rotation.get("active"))
    breadth = int(rotation.get("breadth", 0) or 0)

    current = df.iloc[-1]
    quiet = df.iloc[-(SILENCE_LOOKBACK + 1):-1]
    quiet_avg = float(quiet["volume"].mean())
    if quiet_avg <= 0:
        return False, "quiet average volume is zero", {"active": active, "breadth": breadth}

    current_open = float(current["open"])
    current_close = float(current["close"])
    current_volume = float(current["volume"])
    quiet_low = float(quiet["low"].min())
    spike_ratio = current_volume / quiet_avg
    price_rise_pct = _price_rise_pct(current_close, quiet_low)
    candle_return_pct = (
        (current_close - current_open) / current_open * 100.0
        if current_open > 0 else 0.0
    )

    metrics = {
        "active": active,
        "breadth": breadth,
        "spike_ratio": round(spike_ratio, 2),
        "price_rise_pct": round(price_rise_pct, 2),
        "candle_return_pct": round(candle_return_pct, 2),
        "market_spike_count": int(rotation.get("spike_count", 0) or 0),
        "market_gainer_count": int(rotation.get("gainer_count", 0) or 0),
    }

    reject_reasons = []
    if not active:
        reject_reasons.append("alt rotation mode inactive")
    if breadth < ALT_ROTATION_MIN_BREADTH:
        reject_reasons.append(
            f"breadth {breadth} < {ALT_ROTATION_MIN_BREADTH}"
        )
    if spike_ratio < ALT_ROTATION_MIN_VOLUME_SPIKE:
        reject_reasons.append(
            f"volume spike {spike_ratio:.1f}x < {ALT_ROTATION_MIN_VOLUME_SPIKE:.1f}x"
        )
    if price_rise_pct >= ALT_ROTATION_MAX_RISE_PCT:
        reject_reasons.append(
            f"already extended +{price_rise_pct:.1f}% >= {ALT_ROTATION_MAX_RISE_PCT:.1f}%"
        )
    if candle_return_pct < ALT_ROTATION_MIN_CANDLE_RETURN_PCT:
        reject_reasons.append(
            f"candle return {candle_return_pct:.1f}% < {ALT_ROTATION_MIN_CANDLE_RETURN_PCT:.1f}%"
        )
    if current_close <= current_open:
        reject_reasons.append("current candle is not bullish")

    return not reject_reasons, "; ".join(reject_reasons), metrics


def _price_rise_pct(current_close: float, base_low: float) -> float:
    return ((current_close - base_low) / base_low * 100.0) if base_low > 0 else 999.0


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1:
        return 0.0

    recent = df.tail(period + 1).copy()
    prev_close = recent["close"].shift(1)
    true_range = pd.concat(
        [
            recent["high"] - recent["low"],
            (recent["high"] - prev_close).abs(),
            (recent["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return float(true_range.iloc[1:].mean())


def _btc_drop_reason(btc_df: Optional[pd.DataFrame]) -> str:
    if btc_df is None or btc_df.empty or "close" not in btc_df.columns:
        return ""
    df = btc_df.dropna(subset=["close"])
    if len(df) <= BTC_DROP_LOOKBACK:
        return ""

    recent_close = float(df["close"].iloc[-1])
    past_close = float(df["close"].iloc[-(BTC_DROP_LOOKBACK + 1)])
    if past_close <= 0:
        return ""

    change_pct = (recent_close - past_close) / past_close * 100.0
    if change_pct <= BTC_DROP_BLOCK_PCT:
        return f"BTC short-term drop {change_pct:.2f}% <= {BTC_DROP_BLOCK_PCT:.2f}%"
    return ""


def _score(trigger: str, metrics: dict) -> int:
    score = 75 if trigger == "HAMMER_15M" else 70

    if trigger == "HAMMER_15M":
        hammer = metrics.get("hammer_15m", {})
        score += min(10, max(0, int((hammer.get("volume_ratio", 0) - HAMMER_VOLUME_MULTIPLIER) * 5)))
        score += min(10, max(0, int((hammer.get("wick_body_ratio", 0) - HAMMER_WICK_BODY_RATIO) * 3)))
        if hammer.get("price_rise_pct", 999) < 8:
            score += 5
        return min(score, 100)

    if trigger == "ALT_ROTATION":
        rotation = metrics.get("alt_rotation", {})
        score = 68
        score += min(12, max(0, int((rotation.get("spike_ratio", 0) - ALT_ROTATION_MIN_VOLUME_SPIKE) * 4)))
        score += min(10, max(0, rotation.get("breadth", 0) - ALT_ROTATION_MIN_BREADTH))
        score += min(5, max(0, int(rotation.get("candle_return_pct", 0))))
        if rotation.get("price_rise_pct", 999) < 20:
            score += 5
        return min(score, 100)

    squeeze = metrics.get("squeeze_5m", {})
    score += min(15, max(0, int((squeeze.get("spike_ratio", 0) - SPIKE_MULTIPLIER) * 5)))
    score += min(10, max(0, squeeze.get("silent_bars", 0) - MIN_SILENT_BARS))
    if squeeze.get("price_rise_pct", 999) < 8:
        score += 5
    return min(score, 100)


def is_entry_signal(decision: EntryDecision) -> bool:
    """Small helper for main.py readability."""
    return decision.should_enter and decision.signal == "LONG"
