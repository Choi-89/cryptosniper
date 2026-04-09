"""
momentum_indicators.py
----------------------
모멘텀 지표를 계산하는 모듈.

계산 지표:
  - MACD (12, 26, 9)       : 모멘텀 전환 포착 — 히스토그램 방향이 핵심
  - RSI  14                : 과매수(70↑) / 과매도(30↓) 판단
  - StochRSI (14, 14, 3, 3): RSI 위의 스토캐스틱 — 정밀 진입 타이밍

반환 구조:
  MomentumResult 데이터클래스 — signal_engine.py 에서 직접 참조

의존 라이브러리:
  pip install pandas pandas-ta numpy
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pandas_ta as ta

# ── 로거 ───────────────────────────────────────────────────────────────────────
logger = logging.getLogger("momentum_indicators")


# ── 파라미터 상수 ──────────────────────────────────────────────────────────────
# MACD
MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIGNAL  = 9

# RSI
RSI_PERIOD   = 14
RSI_OB       = 70     # 과매수 (overbought)
RSI_OS       = 30     # 과매도 (oversold)
RSI_BULL_MIN = 40     # 롱 진입 허용 RSI 하한 (과매도 회복 구간)
RSI_BULL_MAX = 65     # 롱 진입 허용 RSI 상한 (과매수 진입 방지)
RSI_BEAR_MIN = 35     # 숏 진입 허용 RSI 하한
RSI_BEAR_MAX = 60     # 숏 진입 허용 RSI 상한

# StochRSI
STOCH_RSI_PERIOD = 14
STOCH_K_PERIOD   = 3
STOCH_D_PERIOD   = 3
STOCH_OB         = 80   # 과매수
STOCH_OS         = 20   # 과매도

# 최소 데이터 요구량
MIN_BARS = MACD_SLOW + MACD_SIGNAL + 10   # 약 45개


# ── 결과 데이터클래스 ──────────────────────────────────────────────────────────

@dataclass
class MomentumResult:
    """
    모멘텀 지표 계산 결과.

    Attributes
    ----------
    [MACD]
    macd            : MACD 라인 (fast EMA - slow EMA)
    macd_signal     : 시그널 라인 (MACD의 EMA)
    macd_hist       : 히스토그램 (macd - signal)
    macd_hist_prev  : 직전 캔들 히스토그램 (전환 감지용)

    [RSI]
    rsi             : RSI 값 (0 ~ 100)
    rsi_prev        : 직전 RSI 값

    [StochRSI]
    stoch_k         : StochRSI %K 선
    stoch_d         : StochRSI %D 선 (K의 이동평균)
    stoch_k_prev    : 직전 %K
    stoch_d_prev    : 직전 %D

    --- 파생 판단값 ---
    macd_bull_cross     : MACD 히스토그램 음→양 전환 (이번 캔들)
    macd_bear_cross     : MACD 히스토그램 양→음 전환 (이번 캔들)
    macd_above_signal   : MACD 라인 > 시그널 라인
    macd_hist_growing   : 히스토그램 절대값 증가 중 (모멘텀 강화)
    rsi_in_bull_zone    : RSI 가 롱 진입 허용 구간 (40~65)
    rsi_in_bear_zone    : RSI 가 숏 진입 허용 구간 (35~60)
    rsi_overbought      : RSI ≥ 70
    rsi_oversold        : RSI ≤ 30
    rsi_rising          : RSI 상승 중 (직전 대비)
    stoch_bull_cross    : %K 가 %D 를 상향 교차 (이번 캔들)
    stoch_bear_cross    : %K 가 %D 를 하향 교차 (이번 캔들)
    stoch_overbought    : %K ≥ 80
    stoch_oversold      : %K ≤ 20
    stoch_k_above_d     : %K > %D
    """
    # ── MACD ──
    macd:           float = 0.0
    macd_signal:    float = 0.0
    macd_hist:      float = 0.0
    macd_hist_prev: float = 0.0

    # ── RSI ──
    rsi:            float = 50.0
    rsi_prev:       float = 50.0

    # ── StochRSI ──
    stoch_k:        float = 50.0
    stoch_d:        float = 50.0
    stoch_k_prev:   float = 50.0
    stoch_d_prev:   float = 50.0

    # ── 파생 판단값: MACD ──
    macd_bull_cross:   bool = False
    macd_bear_cross:   bool = False
    macd_above_signal: bool = False
    macd_hist_growing: bool = False

    # ── 파생 판단값: RSI ──
    rsi_in_bull_zone: bool = False
    rsi_in_bear_zone: bool = False
    rsi_overbought:   bool = False
    rsi_oversold:     bool = False
    rsi_rising:       bool = False

    # ── 파생 판단값: StochRSI ──
    stoch_bull_cross: bool = False
    stoch_bear_cross: bool = False
    stoch_overbought: bool = False
    stoch_oversold:   bool = False
    stoch_k_above_d:  bool = False


# ── 메인 함수 ──────────────────────────────────────────────────────────────────

def calculate(df: pd.DataFrame) -> Optional[MomentumResult]:
    """
    OHLCV DataFrame 으로 모멘텀 지표 전체를 계산하여 MomentumResult 반환.

    Parameters
    ----------
    df : 컬럼 [open, high, low, close, volume] 을 가진 DataFrame.
         인덱스는 datetime. 최소 MIN_BARS (약 45개) 이상 권장.

    Returns
    -------
    MomentumResult or None (데이터 부족 / 계산 실패 시)

    사용 예시:
        from indicators.momentum_indicators import calculate, MomentumResult
        mom: MomentumResult = calculate(df_15m)
        if mom.macd_bull_cross and mom.rsi_in_bull_zone:
            ...
    """
    if df is None or len(df) < MIN_BARS:
        logger.warning(
            f"데이터 부족: {len(df) if df is not None else 0}개 "
            f"(최소 {MIN_BARS}개 필요)"
        )
        return None

    df = df.copy()

    try:
        result = MomentumResult()

        _calc_macd(df, result)
        _calc_rsi(df, result)
        _calc_stoch_rsi(df, result)
        _derive_signals(result)

        return result

    except Exception as e:
        logger.error(f"모멘텀 지표 계산 오류: {e}", exc_info=True)
        return None


# ── 개별 지표 계산 함수 ────────────────────────────────────────────────────────

def _calc_macd(df: pd.DataFrame, result: MomentumResult) -> None:
    """MACD (12, 26, 9) 계산."""
    macd_df = df.ta.macd(
        fast=MACD_FAST,
        slow=MACD_SLOW,
        signal=MACD_SIGNAL,
    )

    if macd_df is None or macd_df.empty:
        raise ValueError("MACD 계산 실패")

    # pandas_ta 컬럼명: MACD_12_26_9 / MACDs_12_26_9 / MACDh_12_26_9
    macd_col  = _find_col(macd_df, f"MACD_{MACD_FAST}_{MACD_SLOW}")
    sig_col   = _find_col(macd_df, "MACDs_")
    hist_col  = _find_col(macd_df, "MACDh_")

    if not all([macd_col, sig_col, hist_col]):
        raise ValueError(f"MACD 컬럼 탐색 실패. 컬럼: {list(macd_df.columns)}")

    macd_val      = macd_df[macd_col].iloc[-1]
    sig_val       = macd_df[sig_col].iloc[-1]
    hist_val      = macd_df[hist_col].iloc[-1]
    hist_prev_val = macd_df[hist_col].iloc[-2]

    if any(pd.isna(v) for v in [macd_val, sig_val, hist_val, hist_prev_val]):
        raise ValueError("MACD 계산 결과 NaN")

    result.macd           = round(float(macd_val),      8)
    result.macd_signal    = round(float(sig_val),       8)
    result.macd_hist      = round(float(hist_val),      8)
    result.macd_hist_prev = round(float(hist_prev_val), 8)

    logger.debug(
        f"MACD: macd={result.macd:.6f}  "
        f"signal={result.macd_signal:.6f}  "
        f"hist={result.macd_hist:.6f}  "
        f"hist_prev={result.macd_hist_prev:.6f}"
    )


def _calc_rsi(df: pd.DataFrame, result: MomentumResult) -> None:
    """RSI 14 계산."""
    rsi_series = df.ta.rsi(length=RSI_PERIOD)

    if rsi_series is None or rsi_series.empty:
        raise ValueError("RSI 계산 실패")

    rsi_val      = rsi_series.iloc[-1]
    rsi_prev_val = rsi_series.iloc[-2]

    if any(pd.isna(v) for v in [rsi_val, rsi_prev_val]):
        raise ValueError("RSI 계산 결과 NaN")

    result.rsi      = round(float(rsi_val),      2)
    result.rsi_prev = round(float(rsi_prev_val), 2)

    logger.debug(f"RSI: {result.rsi:.2f}  (prev={result.rsi_prev:.2f})")


def _calc_stoch_rsi(df: pd.DataFrame, result: MomentumResult) -> None:
    """
    StochRSI 계산.

    pandas_ta 파라미터:
      length = RSI 기간 (14)
      rsi_length = RSI 계산 기간 (14)
      k = %K 스무딩 (3)
      d = %D 스무딩 (3)
    """
    stoch_df = df.ta.stochrsi(
        length     = STOCH_RSI_PERIOD,
        rsi_length = RSI_PERIOD,
        k          = STOCH_K_PERIOD,
        d          = STOCH_D_PERIOD,
    )

    if stoch_df is None or stoch_df.empty:
        raise ValueError("StochRSI 계산 실패")

    # pandas_ta 컬럼명: STOCHRSIk_14_14_3_3 / STOCHRSId_14_14_3_3
    k_col = _find_col(stoch_df, "STOCHRSIk_")
    d_col = _find_col(stoch_df, "STOCHRSId_")

    if not all([k_col, d_col]):
        raise ValueError(f"StochRSI 컬럼 탐색 실패. 컬럼: {list(stoch_df.columns)}")

    k_val      = stoch_df[k_col].iloc[-1]
    d_val      = stoch_df[d_col].iloc[-1]
    k_prev_val = stoch_df[k_col].iloc[-2]
    d_prev_val = stoch_df[d_col].iloc[-2]

    if any(pd.isna(v) for v in [k_val, d_val, k_prev_val, d_prev_val]):
        raise ValueError("StochRSI 계산 결과 NaN")

    result.stoch_k      = round(float(k_val),      2)
    result.stoch_d      = round(float(d_val),      2)
    result.stoch_k_prev = round(float(k_prev_val), 2)
    result.stoch_d_prev = round(float(d_prev_val), 2)

    logger.debug(
        f"StochRSI: K={result.stoch_k:.2f}  D={result.stoch_d:.2f}  "
        f"(K_prev={result.stoch_k_prev:.2f}  D_prev={result.stoch_d_prev:.2f})"
    )


# ── 파생 판단값 계산 ───────────────────────────────────────────────────────────

def _derive_signals(result: MomentumResult) -> None:
    """
    원시 지표값으로 진입에 직접 쓰이는 파생 판단값 계산.

    MACD:
      bull_cross  → 히스토그램이 음수→양수로 전환된 바로 그 캔들
      bear_cross  → 히스토그램이 양수→음수로 전환된 바로 그 캔들
      hist_growing→ 히스토그램 절대값 증가 (모멘텀 강화 중)

    RSI:
      bull_zone   → 40~65: 과매도 아니고 과매수도 아닌 롱 진입 적합 구간
      bear_zone   → 35~60: 숏 진입 적합 구간

    StochRSI:
      bull_cross  → %K 가 %D 를 아래→위로 교차 (직전 K<D, 현재 K>D)
      bear_cross  → %K 가 %D 를 위→아래로 교차 (직전 K>D, 현재 K<D)
    """

    # ── MACD 파생 ─────────────────────────────────────────────────────────────
    # 히스토그램 부호 전환 감지
    hist_was_neg = result.macd_hist_prev < 0
    hist_was_pos = result.macd_hist_prev > 0
    hist_now_pos = result.macd_hist > 0
    hist_now_neg = result.macd_hist < 0

    result.macd_bull_cross   = hist_was_neg and hist_now_pos   # 음→양 전환
    result.macd_bear_cross   = hist_was_pos and hist_now_neg   # 양→음 전환
    result.macd_above_signal = result.macd > result.macd_signal
    result.macd_hist_growing = (
        abs(result.macd_hist) > abs(result.macd_hist_prev)
    )

    # ── RSI 파생 ──────────────────────────────────────────────────────────────
    result.rsi_in_bull_zone = RSI_BULL_MIN <= result.rsi <= RSI_BULL_MAX
    result.rsi_in_bear_zone = RSI_BEAR_MIN <= result.rsi <= RSI_BEAR_MAX
    result.rsi_overbought   = result.rsi >= RSI_OB
    result.rsi_oversold     = result.rsi <= RSI_OS
    result.rsi_rising       = result.rsi > result.rsi_prev

    # ── StochRSI 파생 ─────────────────────────────────────────────────────────
    # 교차 감지: 직전에는 K < D, 지금은 K > D  →  상향 교차
    result.stoch_bull_cross = (
        result.stoch_k_prev <= result.stoch_d_prev
        and result.stoch_k > result.stoch_d
    )
    result.stoch_bear_cross = (
        result.stoch_k_prev >= result.stoch_d_prev
        and result.stoch_k < result.stoch_d
    )
    result.stoch_overbought = result.stoch_k >= STOCH_OB
    result.stoch_oversold   = result.stoch_k <= STOCH_OS
    result.stoch_k_above_d  = result.stoch_k > result.stoch_d

    logger.debug(
        f"파생값: "
        f"macd_bull={result.macd_bull_cross}  "
        f"macd_bear={result.macd_bear_cross}  "
        f"rsi_bull_zone={result.rsi_in_bull_zone}  "
        f"stoch_bull={result.stoch_bull_cross}  "
        f"stoch_bear={result.stoch_bear_cross}"
    )


# ── 유틸 ───────────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, keyword: str) -> Optional[str]:
    """keyword 를 포함한 첫 번째 컬럼명 반환 (대소문자 무시)."""
    matches = [c for c in df.columns if keyword.upper() in c.upper()]
    return matches[0] if matches else None


def summary(result: MomentumResult) -> str:
    """MomentumResult 를 한 줄 요약 문자열로 반환. 로깅/디버깅용."""
    return (
        f"[모멘텀] "
        f"MACD hist={result.macd_hist:+.6f} (prev={result.macd_hist_prev:+.6f}) | "
        f"불크로스={result.macd_bull_cross}  베어크로스={result.macd_bear_cross} | "
        f"RSI={result.rsi:.1f} (bull_zone={result.rsi_in_bull_zone}) | "
        f"StochK={result.stoch_k:.1f}  StochD={result.stoch_d:.1f} | "
        f"stoch_bull={result.stoch_bull_cross}  stoch_bear={result.stoch_bear_cross}"
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
        ("BTC/USDT:USDT", "15m"),
        ("ETH/USDT:USDT", "1h"),
    ]

    for symbol, tf in test_cases:
        print(f"\n{'='*72}")
        print(f"  {symbol}  [{tf}]")
        print('='*72)

        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=150)
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
            print(f"  MACD 라인        : {result.macd:+.6f}")
            print(f"  MACD 시그널      : {result.macd_signal:+.6f}")
            print(f"  MACD 히스토그램  : {result.macd_hist:+.6f}  "
                  f"(직전={result.macd_hist_prev:+.6f})")
            print(f"  MACD 불 크로스   : {result.macd_bull_cross}")
            print(f"  MACD 베어 크로스 : {result.macd_bear_cross}")
            print(f"  히스토그램 강화  : {result.macd_hist_growing}")
            print()
            print(f"  RSI              : {result.rsi:.2f}")
            print(f"  RSI 롱 구간      : {result.rsi_in_bull_zone}  ({RSI_BULL_MIN}~{RSI_BULL_MAX})")
            print(f"  RSI 숏 구간      : {result.rsi_in_bear_zone}  ({RSI_BEAR_MIN}~{RSI_BEAR_MAX})")
            print(f"  RSI 과매수       : {result.rsi_overbought}")
            print(f"  RSI 과매도       : {result.rsi_oversold}")
            print(f"  RSI 상승 중      : {result.rsi_rising}")
            print()
            print(f"  StochRSI %K      : {result.stoch_k:.2f}")
            print(f"  StochRSI %D      : {result.stoch_d:.2f}")
            print(f"  Stoch 불 크로스  : {result.stoch_bull_cross}")
            print(f"  Stoch 베어 크로스: {result.stoch_bear_cross}")
            print(f"  Stoch 과매수     : {result.stoch_overbought}")
            print(f"  Stoch 과매도     : {result.stoch_oversold}")
        else:
            print("  계산 실패")

        time.sleep(0.3)