"""
volume_indicators.py
--------------------
거래량 기반 지표를 계산하는 모듈.

계산 지표:
  - Volume MA (20)  : 평균 대비 거래량 배수 — 급등 감지
  - OBV             : On-Balance Volume — 누적 매수/매도 방향성
  - CVD             : Cumulative Volume Delta — 실제 매수/매도 압력
                      (Binance는 체결 틱 데이터를 제공하지 않으므로
                       캔들 기반 근사치로 계산: 양봉 거래량 - 음봉 거래량)

반환 구조:
  VolumeResult 데이터클래스 — signal_engine.py 에서 직접 참조

의존 라이브러리:
  pip install pandas pandas-ta numpy
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

# ── 로거 ───────────────────────────────────────────────────────────────────────
logger = logging.getLogger("volume_indicators")


# ── 파라미터 상수 ──────────────────────────────────────────────────────────────
VOL_MA_PERIOD      = 20     # 거래량 이동평균 기간
VOL_SURGE_RATIO    = 2.0    # 급등 감지 배수 (평균 대비)
VOL_STRONG_RATIO   = 3.0    # 강한 급등 배수

OBV_MA_PERIOD      = 20     # OBV 추세 확인용 이동평균 기간

CVD_MA_PERIOD      = 14     # CVD 이동평균 기간 (방향성 확인용)

MIN_BARS           = VOL_MA_PERIOD + 5   # 최소 데이터 요구량


# ── 결과 데이터클래스 ──────────────────────────────────────────────────────────

@dataclass
class VolumeResult:
    """
    거래량 지표 계산 결과.

    Attributes
    ----------
    [Volume MA]
    current_volume  : 현재(최신) 캔들 거래량
    volume_ma       : 거래량 이동평균 (20기간)
    volume_ratio    : current_volume / volume_ma — 배수
    prev_volume     : 직전 캔들 거래량 (연속 급등 확인용)
    prev_ratio      : prev_volume / volume_ma

    [OBV]
    obv             : 현재 OBV 값
    obv_ma          : OBV 이동평균 (추세 확인용)
    obv_prev        : 직전 OBV 값

    [CVD — 캔들 기반 근사치]
    cvd             : 누적 볼륨 델타 (최근 CVD_MA_PERIOD 캔들 합산)
    cvd_ma          : CVD 이동평균
    cvd_delta       : 현재 캔들 단일 델타 (양봉: +volume, 음봉: -volume)

    --- 파생 판단값 ---
    volume_surge        : 거래량 급등 여부 (ratio ≥ VOL_SURGE_RATIO)
    volume_strong_surge : 강한 급등 여부 (ratio ≥ VOL_STRONG_RATIO)
    volume_increasing   : 직전 대비 거래량 증가 여부
    obv_rising          : OBV > OBV MA (누적 매수 우세)
    obv_falling         : OBV < OBV MA (누적 매도 우세)
    obv_new_high        : OBV 가 최근 20캔들 신고점 갱신 (강한 매수세)
    obv_new_low         : OBV 가 최근 20캔들 신저점 갱신 (강한 매도세)
    cvd_positive        : CVD > 0 (구간 내 매수 우세)
    cvd_negative        : CVD < 0 (구간 내 매도 우세)
    cvd_rising          : CVD > CVD MA (매수 압력 증가)
    bull_volume         : 현재 캔들이 양봉 + 거래량 급등 (가장 강한 매수 신호)
    bear_volume         : 현재 캔들이 음봉 + 거래량 급등 (가장 강한 매도 신호)
    divergence_bull     : 가격 하락인데 OBV 상승 (강세 다이버전스 — 반등 신호)
    divergence_bear     : 가격 상승인데 OBV 하락 (약세 다이버전스 — 하락 신호)
    """
    # ── Volume MA ──
    current_volume: float = 0.0
    volume_ma:      float = 0.0
    volume_ratio:   float = 0.0
    prev_volume:    float = 0.0
    prev_ratio:     float = 0.0

    # ── OBV ──
    obv:            float = 0.0
    obv_ma:         float = 0.0
    obv_prev:       float = 0.0

    # ── CVD ──
    cvd:            float = 0.0
    cvd_ma:         float = 0.0
    cvd_delta:      float = 0.0

    # ── 파생 판단값 ──
    volume_surge:        bool = False
    volume_strong_surge: bool = False
    volume_increasing:   bool = False
    obv_rising:          bool = False
    obv_falling:         bool = False
    obv_new_high:        bool = False
    obv_new_low:         bool = False
    cvd_positive:        bool = False
    cvd_negative:        bool = False
    cvd_rising:          bool = False
    bull_volume:         bool = False
    bear_volume:         bool = False
    divergence_bull:     bool = False
    divergence_bear:     bool = False


# ── 메인 함수 ──────────────────────────────────────────────────────────────────

def calculate(df: pd.DataFrame) -> Optional[VolumeResult]:
    """
    OHLCV DataFrame 으로 거래량 지표 전체를 계산하여 VolumeResult 반환.

    Parameters
    ----------
    df : 컬럼 [open, high, low, close, volume] 을 가진 DataFrame.
         인덱스는 datetime. 최소 MIN_BARS (25개 이상) 권장.

    Returns
    -------
    VolumeResult or None (데이터 부족 / 계산 실패 시)

    사용 예시:
        from indicators.volume_indicators import calculate, VolumeResult
        vol: VolumeResult = calculate(df_15m)
        if vol.bull_volume and vol.obv_rising:
            score += 2
    """
    if df is None or len(df) < MIN_BARS:
        logger.warning(
            f"데이터 부족: {len(df) if df is not None else 0}개 "
            f"(최소 {MIN_BARS}개 필요)"
        )
        return None

    df = df.copy()
    if df.index.duplicated().any():
        logger.warning(f"중복 타임스탬프 감지 — 마지막 값으로 정리 ({df.index.duplicated().sum()}개)")
        df = df[~df.index.duplicated(keep="last")]

    try:
        result = VolumeResult()

        _calc_volume_ma(df, result)
        _calc_obv(df, result)
        _calc_cvd(df, result)
        _derive_signals(df, result)

        return result

    except Exception as e:
        logger.error(f"거래량 지표 계산 오류: {e}", exc_info=True)
        return None


# ── 개별 지표 계산 함수 ────────────────────────────────────────────────────────

def _calc_volume_ma(df: pd.DataFrame, result: VolumeResult) -> None:
    """거래량 이동평균 및 배수 계산."""
    vol_ma = df["volume"].rolling(VOL_MA_PERIOD).mean()

    ma_val      = vol_ma.iloc[-1]
    curr_vol    = df["volume"].iloc[-1]
    prev_vol    = df["volume"].iloc[-2]

    if pd.isna(ma_val) or ma_val == 0:
        raise ValueError("Volume MA 계산 결과 NaN 또는 0")

    result.current_volume = round(float(curr_vol), 4)
    result.volume_ma      = round(float(ma_val),   4)
    result.volume_ratio   = round(float(curr_vol / ma_val), 4)
    result.prev_volume    = round(float(prev_vol), 4)
    result.prev_ratio     = round(float(prev_vol / ma_val), 4)

    logger.debug(
        f"Volume MA: current={result.current_volume:.2f}  "
        f"ma={result.volume_ma:.2f}  ratio={result.volume_ratio:.2f}x"
    )


def _calc_obv(df: pd.DataFrame, result: VolumeResult) -> None:
    """
    OBV (On-Balance Volume) 계산.

    OBV 규칙:
      현재 종가 > 직전 종가 → OBV += 현재 거래량
      현재 종가 < 직전 종가 → OBV -= 현재 거래량
      현재 종가 == 직전 종가 → OBV 변동 없음
    """
    obv_series = df.ta.obv()

    if obv_series is None or obv_series.empty:
        raise ValueError("OBV 계산 실패")

    obv_val      = obv_series.iloc[-1]
    obv_prev_val = obv_series.iloc[-2]

    if any(pd.isna(v) for v in [obv_val, obv_prev_val]):
        raise ValueError("OBV 계산 결과 NaN")

    # OBV 이동평균 (추세 방향 확인용)
    obv_ma_val = obv_series.rolling(OBV_MA_PERIOD).mean().iloc[-1]

    result.obv      = round(float(obv_val),      2)
    result.obv_prev = round(float(obv_prev_val), 2)
    result.obv_ma   = round(float(obv_ma_val),   2) if not pd.isna(obv_ma_val) else result.obv

    logger.debug(
        f"OBV: {result.obv:.0f}  "
        f"MA={result.obv_ma:.0f}  "
        f"prev={result.obv_prev:.0f}"
    )


def _calc_cvd(df: pd.DataFrame, result: VolumeResult) -> None:
    """
    CVD (Cumulative Volume Delta) 캔들 기반 근사치 계산.

    실제 CVD 는 체결 틱(매수 체결 - 매도 체결)이 필요하지만,
    Binance REST/WebSocket 캔들 스트림에는 틱 데이터가 없음.

    근사 방식 (캔들 기반):
      양봉(close > open) → delta = +volume   (매수 우세 추정)
      음봉(close < open) → delta = -volume   (매도 우세 추정)
      도지(close == open)→ delta = 0

    이 근사치는 실제 CVD 대비 오차가 있지만,
    단기 방향성 편향(bias)을 감지하는 데 실용적으로 활용 가능.
    """
    # 각 캔들의 델타 계산
    close  = df["close"]
    open_  = df["open"]
    volume = df["volume"]

    delta = np.where(close > open_, volume,
            np.where(close < open_, -volume, 0.0))
    df["_delta"] = delta

    # 현재 캔들 단일 델타
    result.cvd_delta = round(float(df["_delta"].iloc[-1]), 4)

    # 누적 델타 (최근 CVD_MA_PERIOD 구간 합산)
    cvd_series = df["_delta"].rolling(CVD_MA_PERIOD).sum()
    cvd_val    = cvd_series.iloc[-1]

    if pd.isna(cvd_val):
        raise ValueError("CVD 계산 결과 NaN")

    # CVD 의 이동평균 (방향성 추세 확인용)
    cvd_ma_val = cvd_series.rolling(CVD_MA_PERIOD).mean().iloc[-1]

    result.cvd    = round(float(cvd_val), 4)
    result.cvd_ma = round(float(cvd_ma_val), 4) if not pd.isna(cvd_ma_val) else 0.0

    logger.debug(
        f"CVD: {result.cvd:.2f}  "
        f"MA={result.cvd_ma:.2f}  "
        f"delta={result.cvd_delta:.2f}"
    )


# ── 파생 판단값 계산 ───────────────────────────────────────────────────────────

def _derive_signals(df: pd.DataFrame, result: VolumeResult) -> None:
    """
    원시 지표값으로 진입 판단에 직접 쓰이는 파생값 계산.

    핵심 신호:
      bull_volume       : 양봉 + 거래량 급등  → 강한 매수 확인
      bear_volume       : 음봉 + 거래량 급등  → 강한 매도 확인
      divergence_bull   : 가격 하락 + OBV 상승 → 매집 중, 반등 임박
      divergence_bear   : 가격 상승 + OBV 하락 → 분산 중, 하락 임박
      obv_new_high/low  : OBV 극단값 — 추세 지속/전환 신호
    """
    # ── 거래량 급등 ───────────────────────────────────────────────────────────
    result.volume_surge        = result.volume_ratio >= VOL_SURGE_RATIO
    result.volume_strong_surge = result.volume_ratio >= VOL_STRONG_RATIO
    result.volume_increasing   = result.current_volume > result.prev_volume

    # ── OBV 방향성 ───────────────────────────────────────────────────────────
    result.obv_rising  = result.obv > result.obv_ma
    result.obv_falling = result.obv < result.obv_ma

    # OBV 신고점 / 신저점 (최근 OBV_MA_PERIOD 캔들 기준)
    obv_series = df.ta.obv()
    if obv_series is not None and len(obv_series) >= OBV_MA_PERIOD:
        recent_obv = obv_series.iloc[-OBV_MA_PERIOD:]
        result.obv_new_high = float(obv_series.iloc[-1]) >= float(recent_obv.max())
        result.obv_new_low  = float(obv_series.iloc[-1]) <= float(recent_obv.min())

    # ── CVD 방향성 ───────────────────────────────────────────────────────────
    result.cvd_positive = result.cvd > 0
    result.cvd_negative = result.cvd < 0
    result.cvd_rising   = result.cvd > result.cvd_ma

    # ── 양봉/음봉 + 거래량 급등 ──────────────────────────────────────────────
    is_bull_candle = df["close"].iloc[-1] > df["open"].iloc[-1]
    is_bear_candle = df["close"].iloc[-1] < df["open"].iloc[-1]

    result.bull_volume = is_bull_candle and result.volume_surge
    result.bear_volume = is_bear_candle and result.volume_surge

    # ── 다이버전스 감지 (최근 5캔들 기준) ─────────────────────────────────────
    #
    # 강세 다이버전스: 가격은 신저점인데 OBV 는 저점을 높이고 있음
    #   → 세력이 조용히 매집 중 → 반등 가능성 높음
    #
    # 약세 다이버전스: 가격은 신고점인데 OBV 는 고점을 낮추고 있음
    #   → 세력이 물량 분산 중 → 하락 가능성 높음
    #
    DIVERGE_WINDOW = 5
    if len(df) >= DIVERGE_WINDOW:
        price_window = df["close"].iloc[-DIVERGE_WINDOW:]
        obv_series   = df.ta.obv()
        obv_window   = obv_series.iloc[-DIVERGE_WINDOW:]

        price_made_low  = float(price_window.iloc[-1]) <= float(price_window.min())
        price_made_high = float(price_window.iloc[-1]) >= float(price_window.max())
        obv_made_high   = float(obv_window.iloc[-1])   >= float(obv_window.max())
        obv_made_low    = float(obv_window.iloc[-1])   <= float(obv_window.min())

        # 강세 다이버전스: 가격 신저점 BUT OBV 신고점
        result.divergence_bull = price_made_low and obv_made_high
        # 약세 다이버전스: 가격 신고점 BUT OBV 신저점
        result.divergence_bear = price_made_high and obv_made_low

    logger.debug(
        f"파생값: "
        f"surge={result.volume_surge}({result.volume_ratio:.1f}x)  "
        f"bull_vol={result.bull_volume}  bear_vol={result.bear_volume}  "
        f"obv_rising={result.obv_rising}  "
        f"cvd_pos={result.cvd_positive}  "
        f"div_bull={result.divergence_bull}  div_bear={result.divergence_bear}"
    )


# ── 유틸 ───────────────────────────────────────────────────────────────────────

def summary(result: VolumeResult) -> str:
    """VolumeResult 를 한 줄 요약 문자열로 반환. 로깅/디버깅용."""
    return (
        f"[거래량] "
        f"ratio={result.volume_ratio:.2f}x  "
        f"surge={result.volume_surge}  "
        f"strong={result.volume_strong_surge}  | "
        f"OBV={result.obv:.0f}  rising={result.obv_rising}  "
        f"new_high={result.obv_new_high}  new_low={result.obv_new_low}  | "
        f"CVD={result.cvd:.1f}  pos={result.cvd_positive}  "
        f"rising={result.cvd_rising}  | "
        f"bull_vol={result.bull_volume}  bear_vol={result.bear_volume}  | "
        f"div_bull={result.divergence_bull}  div_bear={result.divergence_bear}"
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

        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=100)
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
            print(f"  현재 거래량        : {result.current_volume:,.2f}")
            print(f"  20기간 평균 거래량 : {result.volume_ma:,.2f}")
            print(f"  거래량 배수        : {result.volume_ratio:.2f}x")
            print(f"  거래량 급등        : {result.volume_surge}")
            print(f"  강한 급등 (3x↑)   : {result.volume_strong_surge}")
            print(f"  거래량 증가 중     : {result.volume_increasing}")
            print()
            print(f"  OBV               : {result.obv:,.0f}")
            print(f"  OBV MA            : {result.obv_ma:,.0f}")
            print(f"  OBV 상승 추세     : {result.obv_rising}")
            print(f"  OBV 신고점 갱신   : {result.obv_new_high}")
            print(f"  OBV 신저점 갱신   : {result.obv_new_low}")
            print()
            print(f"  CVD (구간 합산)   : {result.cvd:,.2f}")
            print(f"  CVD MA            : {result.cvd_ma:,.2f}")
            print(f"  CVD 양수 (매수↑)  : {result.cvd_positive}")
            print(f"  CVD 상승 추세     : {result.cvd_rising}")
            print(f"  현재 캔들 델타    : {result.cvd_delta:,.2f}")
            print()
            print(f"  양봉+급등 신호    : {result.bull_volume}")
            print(f"  음봉+급등 신호    : {result.bear_volume}")
            print(f"  강세 다이버전스   : {result.divergence_bull}")
            print(f"  약세 다이버전스   : {result.divergence_bear}")
        else:
            print("  계산 실패")

        time.sleep(0.3)