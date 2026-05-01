"""
long_test.py
------------
Experimental Bollinger-band trend strategies for CryptoSniper.

Two modes are available through LONG_TEST_MODE:
  - long_only: bullish/neutral regimes only, long pullback entries
  - long_short: long_only plus tightly filtered shorts in bearish regimes

The module intentionally returns an EntryDecision-shaped object so main.py can
send it through the existing risk, leverage, orderbook, and execution pipeline.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger("long_test")


@dataclass
class LongTestConfig:
    enabled: bool = True
    mode: str = "long_only"
    bb_period: int = 20
    bb_std: float = 2.0
    trend_fast: int = 20
    trend_mid: int = 60
    trend_long: int = 200
    volume_period: int = 20
    min_volume_ratio: float = 0.8
    strong_volume_ratio: float = 1.2
    middle_band_tolerance: float = 0.006
    upper_band_buffer: float = 0.001
    lower_band_buffer: float = 0.001
    max_distance_from_middle: float = 0.035
    btc_drop_block_pct: float = -2.5
    allow_shorts: bool = False
    short_score_cap: int = 72


@dataclass
class LongTestDecision:
    signal: str = "HOLD"
    should_enter: bool = False
    direction: str = "LONG"
    reject_reason: str = ""
    entry_price: float = 0.0
    score: int = 0
    metrics: dict = field(default_factory=dict)


def config_from_env() -> LongTestConfig:
    mode = os.getenv("LONG_TEST_MODE", "long_only").strip().lower()
    if mode not in {"long_only", "long_short"}:
        mode = "long_only"

    cfg = LongTestConfig(
        enabled=_bool_env("LONG_TEST_ENABLED", True),
        mode=mode,
        bb_period=_int_env("LONG_TEST_BB_PERIOD", 20),
        bb_std=_float_env("LONG_TEST_BB_STD", 2.0),
        trend_fast=_int_env("LONG_TEST_FAST_MA", 20),
        trend_mid=_int_env("LONG_TEST_MID_MA", 60),
        trend_long=_int_env("LONG_TEST_LONG_MA", 200),
        volume_period=_int_env("LONG_TEST_VOLUME_PERIOD", 20),
        min_volume_ratio=_float_env("LONG_TEST_MIN_VOLUME_RATIO", 0.8),
        strong_volume_ratio=_float_env("LONG_TEST_STRONG_VOLUME_RATIO", 1.2),
        middle_band_tolerance=_float_env("LONG_TEST_MIDDLE_TOLERANCE", 0.006),
        upper_band_buffer=_float_env("LONG_TEST_UPPER_BUFFER", 0.001),
        lower_band_buffer=_float_env("LONG_TEST_LOWER_BUFFER", 0.001),
        max_distance_from_middle=_float_env("LONG_TEST_MAX_MIDDLE_DISTANCE", 0.035),
        btc_drop_block_pct=_float_env("LONG_TEST_BTC_DROP_BLOCK_PCT", -2.5),
        short_score_cap=_int_env("LONG_TEST_SHORT_SCORE_CAP", 72),
    )
    cfg.allow_shorts = mode == "long_short" and _bool_env("LONG_TEST_ALLOW_SHORTS", True)
    return cfg


def check(
    symbol: str,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    btc_df_1h: Optional[pd.DataFrame] = None,
    cfg: Optional[LongTestConfig] = None,
) -> LongTestDecision:
    cfg = cfg or config_from_env()
    decision = LongTestDecision()

    if not cfg.enabled:
        decision.reject_reason = "long_test disabled"
        return decision

    valid, reason = _validate(df_1h, "1h")
    if not valid:
        decision.reject_reason = reason
        return decision
    valid, reason = _validate(df_15m, "15m")
    if not valid:
        decision.reject_reason = reason
        return decision

    trend_df = _with_trend(df_1h, cfg)
    entry_df = _with_bollinger(df_15m, cfg)
    if len(trend_df) < max(cfg.trend_mid, cfg.bb_period) or len(entry_df) < cfg.bb_period + 2:
        decision.reject_reason = "not enough candles for long_test"
        return decision

    regime = _market_regime(trend_df, cfg)
    btc_regime = _market_regime(_with_trend(btc_df_1h, cfg), cfg) if btc_df_1h is not None else "unknown"
    btc_drop = _recent_change_pct(btc_df_1h, bars=3) if btc_df_1h is not None else 0.0

    latest = entry_df.iloc[-1]
    previous = entry_df.iloc[-2]
    entry_price = float(latest["close"])
    decision.entry_price = entry_price

    volume_ratio = _volume_ratio(entry_df, cfg.volume_period)
    dist_mid = _distance(entry_price, float(latest["bb_mid"]))
    band_width = _distance(float(latest["bb_upper"]), float(latest["bb_lower"]))

    metrics = {
        "strategy": cfg.mode,
        "trigger": "",
        "regime": regime,
        "btc_regime": btc_regime,
        "btc_3bar_change_pct": round(btc_drop, 2),
        "entry_price": round(entry_price, 8),
        "bb_mid": round(float(latest["bb_mid"]), 8),
        "bb_upper": round(float(latest["bb_upper"]), 8),
        "bb_lower": round(float(latest["bb_lower"]), 8),
        "distance_mid_pct": round(dist_mid * 100, 3),
        "band_width_pct": round(band_width * 100, 3),
        "volume_ratio": round(volume_ratio, 2),
    }
    decision.metrics = metrics

    long_ok, long_reason = _long_setup(
        regime, btc_regime, btc_drop, latest, previous, volume_ratio, dist_mid, cfg
    )
    if long_ok:
        decision.signal = "LONG"
        decision.direction = "LONG"
        decision.should_enter = True
        decision.score = _score_long(regime, btc_regime, volume_ratio, dist_mid, band_width)
        decision.metrics["trigger"] = "BB_MID_PULLBACK_LONG"
        logger.info(f"[{symbol}] LONG_TEST LONG score={decision.score} metrics={metrics}")
        return decision

    if cfg.allow_shorts:
        short_ok, short_reason = _short_setup(
            regime, btc_regime, latest, previous, volume_ratio, dist_mid, cfg
        )
        if short_ok:
            decision.signal = "SHORT"
            decision.direction = "SHORT"
            decision.should_enter = True
            decision.score = min(
                cfg.short_score_cap,
                _score_short(regime, btc_regime, volume_ratio, dist_mid, band_width),
            )
            decision.metrics["trigger"] = "BB_MID_REJECTION_SHORT"
            logger.info(f"[{symbol}] LONG_TEST SHORT score={decision.score} metrics={metrics}")
            return decision
    else:
        short_reason = "shorts disabled"

    decision.reject_reason = f"long: {long_reason}; short: {short_reason}"
    return decision


def _validate(df: Optional[pd.DataFrame], label: str) -> tuple[bool, str]:
    if df is None or df.empty:
        return False, f"{label} dataframe is empty"
    missing = {"open", "high", "low", "close", "volume"} - set(df.columns)
    if missing:
        return False, f"{label} dataframe missing columns: {sorted(missing)}"
    return True, ""


def _with_trend(df: Optional[pd.DataFrame], cfg: LongTestConfig) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy().dropna(subset=["close"])
    out["ma_fast"] = out["close"].rolling(cfg.trend_fast).mean()
    out["ma_mid"] = out["close"].rolling(cfg.trend_mid).mean()
    out["ma_long"] = out["close"].rolling(cfg.trend_long).mean()
    out["atr"] = _atr(out)
    return out


def _with_bollinger(df: pd.DataFrame, cfg: LongTestConfig) -> pd.DataFrame:
    out = df.copy().dropna(subset=["open", "high", "low", "close", "volume"])
    mid = out["close"].rolling(cfg.bb_period).mean()
    std = out["close"].rolling(cfg.bb_period).std(ddof=0)
    out["bb_mid"] = mid
    out["bb_upper"] = mid + std * cfg.bb_std
    out["bb_lower"] = mid - std * cfg.bb_std
    out["volume_ma"] = out["volume"].rolling(cfg.volume_period).mean()
    return out.dropna(subset=["bb_mid", "bb_upper", "bb_lower"])


def _market_regime(df: pd.DataFrame, cfg: LongTestConfig) -> str:
    if df is None or df.empty or len(df) < cfg.trend_mid:
        return "unknown"
    latest = df.iloc[-1]
    close = float(latest["close"])
    ma_fast = float(latest["ma_fast"]) if pd.notna(latest["ma_fast"]) else 0.0
    ma_mid = float(latest["ma_mid"]) if pd.notna(latest["ma_mid"]) else 0.0
    ma_long = float(latest["ma_long"]) if "ma_long" in df.columns and pd.notna(latest["ma_long"]) else 0.0
    mid_slope = _slope(df["ma_mid"], 5)

    if close > ma_fast > ma_mid and mid_slope > 0:
        if ma_long <= 0 or ma_mid > ma_long:
            return "bull"
        return "weak_bull"
    if close < ma_fast < ma_mid and mid_slope < 0:
        if ma_long <= 0 or ma_mid < ma_long:
            return "bear"
        return "weak_bear"
    return "range"


def _long_setup(
    regime: str,
    btc_regime: str,
    btc_drop: float,
    latest: pd.Series,
    previous: pd.Series,
    volume_ratio: float,
    dist_mid: float,
    cfg: LongTestConfig,
) -> tuple[bool, str]:
    if regime not in {"bull", "weak_bull", "range"}:
        return False, f"symbol regime is {regime}"
    if btc_regime in {"bear", "weak_bear"}:
        return False, f"BTC regime is {btc_regime}"
    if btc_drop <= cfg.btc_drop_block_pct:
        return False, f"BTC short-term drop {btc_drop:.2f}%"
    if volume_ratio < cfg.min_volume_ratio:
        return False, f"volume ratio {volume_ratio:.2f} < {cfg.min_volume_ratio:.2f}"

    close = float(latest["close"])
    open_ = float(latest["open"])
    mid = float(latest["bb_mid"])
    upper = float(latest["bb_upper"])
    prev_close = float(previous["close"])

    touched_mid = abs(close - mid) / mid <= cfg.middle_band_tolerance or float(latest["low"]) <= mid
    reclaimed_mid = prev_close < float(previous["bb_mid"]) and close >= mid
    not_chasing = close < upper * (1.0 - cfg.upper_band_buffer)
    not_too_far = dist_mid <= cfg.max_distance_from_middle
    bullish_close = close >= open_ or reclaimed_mid

    if not (touched_mid or reclaimed_mid):
        return False, "no middle-band pullback/reclaim"
    if not not_chasing:
        return False, "price too close to upper band"
    if not not_too_far:
        return False, f"too far from middle band {dist_mid*100:.2f}%"
    if not bullish_close:
        return False, "no bullish confirmation candle"
    return True, ""


def _short_setup(
    regime: str,
    btc_regime: str,
    latest: pd.Series,
    previous: pd.Series,
    volume_ratio: float,
    dist_mid: float,
    cfg: LongTestConfig,
) -> tuple[bool, str]:
    if regime not in {"bear", "weak_bear"}:
        return False, f"symbol regime is {regime}"
    if btc_regime not in {"bear", "weak_bear", "unknown"}:
        return False, f"BTC regime is {btc_regime}"
    if volume_ratio < cfg.strong_volume_ratio:
        return False, f"short volume ratio {volume_ratio:.2f} < {cfg.strong_volume_ratio:.2f}"

    close = float(latest["close"])
    open_ = float(latest["open"])
    mid = float(latest["bb_mid"])
    lower = float(latest["bb_lower"])
    prev_close = float(previous["close"])

    rejected_mid = float(latest["high"]) >= mid and close < mid
    failed_reclaim = prev_close > float(previous["bb_mid"]) and close < mid
    not_late = close > lower * (1.0 + cfg.lower_band_buffer)
    not_too_far = dist_mid <= cfg.max_distance_from_middle
    bearish_close = close <= open_ or failed_reclaim

    if not (rejected_mid or failed_reclaim):
        return False, "no middle-band rejection"
    if not not_late:
        return False, "price too close to lower band"
    if not not_too_far:
        return False, f"too far from middle band {dist_mid*100:.2f}%"
    if not bearish_close:
        return False, "no bearish confirmation candle"
    return True, ""


def _score_long(
    regime: str,
    btc_regime: str,
    volume_ratio: float,
    dist_mid: float,
    band_width: float,
) -> int:
    score = 58
    if regime == "bull":
        score += 12
    elif regime == "weak_bull":
        score += 8
    if btc_regime == "bull":
        score += 8
    elif btc_regime == "weak_bull":
        score += 4
    score += min(10, int(max(0.0, volume_ratio - 1.0) * 8))
    score += max(0, 8 - int(dist_mid * 300))
    if band_width >= 0.025:
        score += 4
    return max(0, min(100, score))


def _score_short(
    regime: str,
    btc_regime: str,
    volume_ratio: float,
    dist_mid: float,
    band_width: float,
) -> int:
    score = 54
    if regime == "bear":
        score += 10
    if btc_regime == "bear":
        score += 8
    score += min(8, int(max(0.0, volume_ratio - 1.0) * 6))
    score += max(0, 6 - int(dist_mid * 250))
    if band_width >= 0.03:
        score += 3
    return max(0, min(100, score))


def _volume_ratio(df: pd.DataFrame, period: int) -> float:
    if "volume_ma" in df.columns:
        avg = float(df["volume_ma"].iloc[-1])
    else:
        avg = float(df["volume"].rolling(period).mean().iloc[-1])
    current = float(df["volume"].iloc[-1])
    return current / avg if avg > 0 else 0.0


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _slope(series: pd.Series, bars: int) -> float:
    clean = series.dropna()
    if len(clean) <= bars:
        return 0.0
    start = float(clean.iloc[-(bars + 1)])
    end = float(clean.iloc[-1])
    return (end - start) / start if start > 0 else 0.0


def _recent_change_pct(df: Optional[pd.DataFrame], bars: int) -> float:
    if df is None or df.empty or len(df) <= bars or "close" not in df.columns:
        return 0.0
    close = df["close"].dropna()
    if len(close) <= bars:
        return 0.0
    now = float(close.iloc[-1])
    past = float(close.iloc[-(bars + 1)])
    return (now - past) / past * 100.0 if past > 0 else 0.0


def _distance(a: float, b: float) -> float:
    return abs(a - b) / b if b > 0 else 999.0


def _bool_env(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _float_env(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default
