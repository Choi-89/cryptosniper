"""
trend_indicators.py
-------------------
추세 판단에 사용하는 지표를 계산하는 모듈.

계산 지표:
  - EMA  20 / 50 / 200   : 추세 방향 및 골든/데드 크로스
  - ADX  14              : 추세 강도 (25 이상일 때만 신뢰)
  - 볼린저밴드 (20, 2.0) : 변동성 및 밴드 내 위치
  - ATR  14              : 절대 변동성 (포지션 크기 계산에도 활용)

반환 구조:
  TrendResult 데이터클래스 — signal_engine.py 에서 직접 참조

의존 라이브러리:
  pip install pandas pandas-ta numpy
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

# ── 로거 ───────────────────────────────────────────────────────────────────────
logger = logging.getLogger("trend_indicators")


# ── 파라미터 상수 ──────────────────────────────────────────────────────────────
EMA_SHORT    = 20
EMA_MID      = 50
EMA_LONG     = 200
ADX_PERIOD   = 14
BB_PERIOD    = 20
BB_STD       = 2.0
ATR_PERIOD   = 14

# ADX 강도 구분 임계값
ADX_WEAK     = 25   # 미만: 추세 없음 (횡보)
ADX_STRONG   = 40   # 이상: 강한 추세


# ── 결과 데이터클래스 ──────────────────────────────────────────────────────────

@dataclass
class TrendResult:
    """
    추세 지표 계산 결과.

    Attributes
    ----------
    ema_short       : EMA 20 값
    ema_mid         : EMA 50 값
    ema_long        : EMA 200 값
    adx             : ADX 값
    adx_plus_di     : +DI 값 (매수 방향성)
    adx_minus_di    : -DI 값 (매도 방향성)
    bb_upper        : 볼린저밴드 상단
    bb_mid          : 볼린저밴드 중심선 (EMA 20과 동일)
    bb_lower        : 볼린저밴드 하단
    bb_width        : 밴드 폭 (upper - lower)
    bb_pct          : %B — 현재가의 밴드 내 위치 (0.0 ~ 1.0, 범위 밖도 가능)
    atr             : ATR 값
    atr_ratio       : ATR / 현재가 비율
    close           : 현재 종가 (최신 캔들)

    --- 파생 판단값 ---
    trend_direction : "UP" / "DOWN" / "SIDEWAYS"
    ema_aligned_up  : EMA20 > EMA50 > EMA200 (완전 정배열)
    ema_aligned_dn  : EMA20 < EMA50 < EMA200 (완전 역배열)
    above_bb_mid    : 현재가 > 볼린저밴드 중심선
    near_bb_upper   : 현재가 > 볼린저밴드 상단의 95% 위치 (과매수 근접)
    near_bb_lower   : 현재가 < 볼린저밴드 하단의 105% 위치 (과매도 근접)
    trend_strength  : "STRONG" / "MODERATE" / "WEAK"
    golden_cross    : 이번 캔들에서 EMA20 이 EMA50 을 상향 돌파
    dead_cross      : 이번 캔들에서 EMA20 이 EMA50 을 하향 돌파
    """
    # 원시 지표값
    ema_short:    float = 0.0
    ema_mid:      float = 0.0
    ema_long:     float = 0.0
    adx:          float = 0.0
    adx_plus_di:  float = 0.0
    adx_minus_di: float = 0.0
    bb_upper:     float = 0.0
    bb_mid:       float = 0.0
    bb_lower:     float = 0.0
    bb_width:     float = 0.0
    bb_pct:       float = 0.0
    atr:          float = 0.0
    atr_ratio:    float = 0.0
    close:        float = 0.0

    # 파생 판단값
    trend_direction: str  = "SIDEWAYS"   # "UP" / "DOWN" / "SIDEWAYS"
    ema_aligned_up:  bool = False
    ema_aligned_dn:  bool = False
    above_bb_mid:    bool = False
    near_bb_upper:   bool = False
    near_bb_lower:   bool = False
    trend_strength:  str  = "WEAK"       # "STRONG" / "MODERATE" / "WEAK"
    golden_cross:    bool = False
    dead_cross:      bool = False


# ── 메인 함수 ──────────────────────────────────────────────────────────────────

def calculate(df: pd.DataFrame) -> Optional[TrendResult]:
    """
    OHLCV DataFrame으로 추세 지표 전체를 계산하여 TrendResult 반환.

    Parameters
    ----------
    df : 컬럼 [open, high, low, close, volume] 을 가진 DataFrame.
         인덱스는 datetime. 최소 EMA_LONG + 10 개 (210개 이상) 권장.

    Returns
    -------
    TrendResult or None (데이터 부족 / 계산 실패 시)

    사용 예시:
        from indicators.trend_indicators import calculate, TrendResult
        result: TrendResult = calculate(df)
        if result and result.trend_direction == "UP":
            ...
    """
    if df is None or len(df) < EMA_LONG + 5:
        logger.warning(
            f"데이터 부족: {len(df) if df is not None else 0}개 "
            f"(최소 {EMA_LONG + 5}개 필요)"
        )
        return None

    # DataFrame 복사본 사용 (원본 훼손 방지)
    df = df.copy()

    try:
        result = TrendResult()
        result.close = float(df["close"].iloc[-1])

        # ── 1. EMA 계산 ──────────────────────────────────────────────────────
        _calc_ema(df, result)

        # ── 2. ADX 계산 ──────────────────────────────────────────────────────
        _calc_adx(df, result)

        # ── 3. 볼린저밴드 계산 ───────────────────────────────────────────────
        _calc_bb(df, result)

        # ── 4. ATR 계산 ──────────────────────────────────────────────────────
        _calc_atr(df, result)

        # ── 5. 파생 판단값 계산 ──────────────────────────────────────────────
        _derive_signals(df, result)

        return result

    except Exception as e:
        logger.error(f"추세 지표 계산 오류: {e}", exc_info=True)
        return None


# ── 개별 지표 계산 함수 ────────────────────────────────────────────────────────

def _calc_ema(df: pd.DataFrame, result: TrendResult) -> None:
    """EMA 20 / 50 / 200 계산."""
    for period, attr in [
        (EMA_SHORT, "ema_short"),
        (EMA_MID,   "ema_mid"),
        (EMA_LONG,  "ema_long"),
    ]:
        ema_series = df["close"].ewm(span=period, adjust=False).mean()
        val = ema_series.iloc[-1]
        if pd.isna(val):
            raise ValueError(f"EMA{period} 계산 결과 NaN")
        setattr(result, attr, round(float(val), 8))

    logger.debug(
        f"EMA 계산 완료: "
        f"EMA{EMA_SHORT}={result.ema_short:.4f}  "
        f"EMA{EMA_MID}={result.ema_mid:.4f}  "
        f"EMA{EMA_LONG}={result.ema_long:.4f}"
    )


def _calc_adx(df: pd.DataFrame, result: TrendResult) -> None:
    """ADX / +DI / -DI 계산."""
    adx_df = df.ta.adx(length=ADX_PERIOD)

    if adx_df is None or adx_df.empty:
        raise ValueError("ADX 계산 실패")

    # pandas_ta 컬럼명: ADX_14, DMP_14, DMN_14
    adx_col  = _find_col(adx_df, f"ADX_{ADX_PERIOD}")
    dmp_col  = _find_col(adx_df, "DMP_")   # +DI
    dmn_col  = _find_col(adx_df, "DMN_")   # -DI

    if not all([adx_col, dmp_col, dmn_col]):
        raise ValueError(f"ADX 컬럼 탐색 실패. 컬럼: {list(adx_df.columns)}")

    adx_val = adx_df[adx_col].iloc[-1]
    dmp_val = adx_df[dmp_col].iloc[-1]
    dmn_val = adx_df[dmn_col].iloc[-1]

    if any(pd.isna(v) for v in [adx_val, dmp_val, dmn_val]):
        raise ValueError("ADX 계산 결과 NaN")

    result.adx          = round(float(adx_val), 2)
    result.adx_plus_di  = round(float(dmp_val), 2)
    result.adx_minus_di = round(float(dmn_val), 2)

    logger.debug(
        f"ADX 계산 완료: ADX={result.adx}  "
        f"+DI={result.adx_plus_di}  -DI={result.adx_minus_di}"
    )


def _calc_bb(df: pd.DataFrame, result: TrendResult) -> None:
    """볼린저밴드 (20, 2.0) 계산."""
    bb_df = df.ta.bbands(length=BB_PERIOD, std=BB_STD)

    if bb_df is None or bb_df.empty:
        raise ValueError("볼린저밴드 계산 실패")

    # pandas_ta 컬럼명: BBU_20_2.0 / BBM_20_2.0 / BBL_20_2.0 / BBB_20_2.0 / BBP_20_2.0
    upper_col = _find_col(bb_df, "BBU_")
    mid_col   = _find_col(bb_df, "BBM_")
    lower_col = _find_col(bb_df, "BBL_")
    pct_col   = _find_col(bb_df, "BBP_")   # %B

    if not all([upper_col, mid_col, lower_col]):
        raise ValueError(f"볼린저밴드 컬럼 탐색 실패. 컬럼: {list(bb_df.columns)}")

    upper = bb_df[upper_col].iloc[-1]
    mid   = bb_df[mid_col].iloc[-1]
    lower = bb_df[lower_col].iloc[-1]

    if any(pd.isna(v) for v in [upper, mid, lower]):
        raise ValueError("볼린저밴드 계산 결과 NaN")

    result.bb_upper = round(float(upper), 8)
    result.bb_mid   = round(float(mid),   8)
    result.bb_lower = round(float(lower), 8)
    result.bb_width = round(float(upper - lower), 8)

    # %B: (현재가 - 하단) / (상단 - 하단)
    if pct_col and not pd.isna(bb_df[pct_col].iloc[-1]):
        result.bb_pct = round(float(bb_df[pct_col].iloc[-1]), 4)
    else:
        band_range = upper - lower
        result.bb_pct = round(
            float((result.close - lower) / band_range) if band_range > 0 else 0.5,
            4,
        )

    logger.debug(
        f"BB 계산 완료: Upper={result.bb_upper:.4f}  "
        f"Mid={result.bb_mid:.4f}  Lower={result.bb_lower:.4f}  "
        f"%B={result.bb_pct:.3f}"
    )


def _calc_atr(df: pd.DataFrame, result: TrendResult) -> None:
    """ATR 14 계산."""
    atr_series = df.ta.atr(length=ATR_PERIOD)

    if atr_series is None or atr_series.empty:
        raise ValueError("ATR 계산 실패")

    atr_val = atr_series.iloc[-1]
    if pd.isna(atr_val):
        raise ValueError("ATR 계산 결과 NaN")

    result.atr       = round(float(atr_val), 8)
    result.atr_ratio = round(
        float(atr_val / result.close) if result.close > 0 else 0.0,
        6,
    )

    logger.debug(
        f"ATR 계산 완료: ATR={result.atr:.4f}  ratio={result.atr_ratio*100:.3f}%"
    )


def _derive_signals(df: pd.DataFrame, result: TrendResult) -> None:
    """
    원시 지표값으로 파생 판단값 계산.

    - trend_direction : EMA 배열 + ADX + +DI/-DI 조합
    - trend_strength  : ADX 수치 기준
    - EMA 정배열/역배열
    - 볼린저밴드 위치
    - 골든/데드 크로스 (직전 캔들과 비교)
    """
    c = result.close

    # ── EMA 정배열 / 역배열 ────────────────────────────────────────────────────
    result.ema_aligned_up = (
        result.ema_short > result.ema_mid > result.ema_long
    )
    result.ema_aligned_dn = (
        result.ema_short < result.ema_mid < result.ema_long
    )

    # ── 추세 방향 ─────────────────────────────────────────────────────────────
    # 상승: EMA20 > EMA50, +DI > -DI, ADX 유효
    # 하락: EMA20 < EMA50, -DI > +DI, ADX 유효
    # 횡보: ADX 약하거나 EMA 방향 불일치
    adx_valid    = result.adx >= ADX_WEAK
    ema_bull     = result.ema_short > result.ema_mid
    ema_bear     = result.ema_short < result.ema_mid
    di_bull      = result.adx_plus_di > result.adx_minus_di
    di_bear      = result.adx_plus_di < result.adx_minus_di

    if adx_valid and ema_bull and di_bull:
        result.trend_direction = "UP"
    elif adx_valid and ema_bear and di_bear:
        result.trend_direction = "DOWN"
    else:
        result.trend_direction = "SIDEWAYS"

    # ── 추세 강도 ─────────────────────────────────────────────────────────────
    if result.adx >= ADX_STRONG:
        result.trend_strength = "STRONG"
    elif result.adx >= ADX_WEAK:
        result.trend_strength = "MODERATE"
    else:
        result.trend_strength = "WEAK"

    # ── 볼린저밴드 위치 ───────────────────────────────────────────────────────
    result.above_bb_mid  = c > result.bb_mid
    result.near_bb_upper = c >= result.bb_upper * 0.995   # 상단 0.5% 이내
    result.near_bb_lower = c <= result.bb_lower * 1.005   # 하단 0.5% 이내

    # ── 골든 / 데드 크로스 ────────────────────────────────────────────────────
    # 직전 2개 캔들의 EMA 값 비교
    if len(df) >= 2:
        ema_s_series = df["close"].ewm(span=EMA_SHORT, adjust=False).mean()
        ema_m_series = df["close"].ewm(span=EMA_MID,   adjust=False).mean()

        prev_s = float(ema_s_series.iloc[-2])
        prev_m = float(ema_m_series.iloc[-2])
        curr_s = float(ema_s_series.iloc[-1])
        curr_m = float(ema_m_series.iloc[-1])

        # 골든크로스: 직전엔 EMA20 < EMA50 이었다가 지금 EMA20 > EMA50
        result.golden_cross = (prev_s <= prev_m) and (curr_s > curr_m)
        # 데드크로스: 직전엔 EMA20 > EMA50 이었다가 지금 EMA20 < EMA50
        result.dead_cross   = (prev_s >= prev_m) and (curr_s < curr_m)

    logger.debug(
        f"파생값: direction={result.trend_direction}  "
        f"strength={result.trend_strength}  "
        f"aligned_up={result.ema_aligned_up}  "
        f"golden={result.golden_cross}  dead={result.dead_cross}"
    )


# ── 유틸 ───────────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, keyword: str) -> Optional[str]:
    """keyword 를 포함한 첫 번째 컬럼명 반환 (대소문자 무시)."""
    matches = [c for c in df.columns if keyword.upper() in c.upper()]
    return matches[0] if matches else None


def summary(result: TrendResult) -> str:
    """TrendResult 를 한 줄 요약 문자열로 반환. 로깅/디버깅용."""
    return (
        f"[추세] {result.trend_direction:<9} | "
        f"강도={result.trend_strength:<8} | "
        f"ADX={result.adx:>5.1f} | "
        f"+DI={result.adx_plus_di:>5.1f}  -DI={result.adx_minus_di:>5.1f} | "
        f"EMA20={result.ema_short:.4f}  EMA50={result.ema_mid:.4f}  EMA200={result.ema_long:.4f} | "
        f"BB%={result.bb_pct:.3f} | "
        f"ATR%={result.atr_ratio*100:.3f}% | "
        f"GC={result.golden_cross}  DC={result.dead_cross}"
    )


# ── 단독 실행 테스트 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import time
    import ccxt
    from dotenv import load_dotenv

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    load_dotenv()

    API_KEY    = os.getenv("BINANCE_API_KEY", "")
    API_SECRET = os.getenv("BINANCE_API_SECRET", "")

    exchange = ccxt.binanceusdm({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
    })

    test_cases = [
        ("BTC/USDT:USDT", "1h"),
        ("ETH/USDT:USDT", "15m"),
    ]

    for symbol, tf in test_cases:
        print(f"\n{'='*70}")
        print(f"  {symbol}  [{tf}]")
        print('='*70)

        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=250)
        raw_df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], unit="ms")
        raw_df.set_index("timestamp", inplace=True)
        raw_df = raw_df.astype(float)

        result = calculate(raw_df)

        if result:
            print(summary(result))
            print()
            print(f"  현재가           : {result.close:.4f}")
            print(f"  EMA 정배열 (UP)  : {result.ema_aligned_up}")
            print(f"  EMA 역배열 (DN)  : {result.ema_aligned_dn}")
            print(f"  볼린저밴드 상단   : {result.bb_upper:.4f}")
            print(f"  볼린저밴드 하단   : {result.bb_lower:.4f}")
            print(f"  상단 근접 여부    : {result.near_bb_upper}")
            print(f"  하단 근접 여부    : {result.near_bb_lower}")
            print(f"  ATR              : {result.atr:.4f}")
        else:
            print("  계산 실패")

        time.sleep(0.3)