"""
signal_engine.py
----------------
trend / momentum / volume 세 지표 레이어를 조합해
최종 진입 신호(LONG / SHORT / HOLD)와 점수를 반환하는 모듈.

판단 구조:
  1. 필수 조건 (AND) — 하나라도 실패하면 즉시 HOLD 반환
  2. 보조 조건 (OR)  — 충족 개수만큼 점수 가산
  3. 최종 점수 ≥ MIN_SCORE_TO_ENTER 일 때만 실제 진입 허가

타임프레임 역할:
  - df_1h  (1시간봉) → 추세 방향 판단 레이어 (TrendResult)
  - df_15m (15분봉)  → 모멘텀 + 거래량 진입 타이밍 (MomentumResult, VolumeResult)

의존 모듈:
  indicators/trend_indicators.py
  indicators/momentum_indicators.py
  indicators/volume_indicators.py
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from indicators import trend_indicators   as ti
from indicators import momentum_indicators as mi
from indicators import volume_indicators   as vi
from indicators.trend_indicators    import TrendResult
from indicators.momentum_indicators import MomentumResult
from indicators.volume_indicators   import VolumeResult

# ── 로거 ───────────────────────────────────────────────────────────────────────
logger = logging.getLogger("signal_engine")


# ── 파라미터 상수 ──────────────────────────────────────────────────────────────
MIN_SCORE_TO_ENTER = 5      # 진입 허용 최소 점수
MAX_SCORE          = 22     # 실제 보조 조건 합산 최대 (추세7 + 모멘텀7 + 거래량8)


# ── 결과 데이터클래스 ──────────────────────────────────────────────────────────

@dataclass
class SignalResult:
    """
    signal_engine.check() 의 최종 반환값.

    Attributes
    ----------
    signal          : "LONG" / "SHORT" / "HOLD"
    score           : 보조 조건 합산 점수 (0 ~ MAX_SCORE)
    reject_reason   : HOLD 사유 (필수 조건 실패 시 기록)
    passed_required : 필수 조건 통과 여부
    bonus_details   : 보조 조건별 점수 상세 {조건명: 점수}
    trend           : TrendResult 원본 (외부에서 ATR 등 재활용)
    momentum        : MomentumResult 원본
    volume          : VolumeResult 원본
    entry_price     : 현재 종가 (진입 참고가)
    """
    signal:          str   = "HOLD"
    score:           int   = 0
    reject_reason:   str   = ""
    passed_required: bool  = False
    bonus_details:   dict  = field(default_factory=dict)
    trend:           Optional[TrendResult]    = None
    momentum:        Optional[MomentumResult] = None
    volume:          Optional[VolumeResult]   = None
    entry_price:     float = 0.0


# ── 메인 함수 ──────────────────────────────────────────────────────────────────

def check(
    symbol:    str,
    df_1h:     pd.DataFrame,
    df_15m:    pd.DataFrame,
    btc_df_1h: Optional[pd.DataFrame] = None,
    min_score_to_enter: int = MIN_SCORE_TO_ENTER,
) -> SignalResult:
    """
    1H + 15M 데이터로 진입 신호를 판단하여 SignalResult 반환.

    Parameters
    ----------
    symbol    : 예) "BTC/USDT:USDT"  (로깅용)
    df_1h     : 1시간봉 OHLCV DataFrame  (추세 판단)
    df_15m    : 15분봉  OHLCV DataFrame  (진입 타이밍)
    btc_df_1h : BTC 1시간봉 DataFrame (시장 필터용, None이면 BTC 필터 생략)

    Returns
    -------
    SignalResult
      .signal  == "LONG"  → 롱 진입 권고
      .signal  == "SHORT" → 숏 진입 권고
      .signal  == "HOLD"  → 진입 보류
    """
    result = SignalResult()

    # ── Step 0. BTC 시장 방향 필터 ───────────────────────────────────────────
    # BTC가 아닌 알트코인에 대해서만 적용 (BTC 자신은 스킵)
    if btc_df_1h is not None and "BTC" not in symbol:
        btc_reject = _check_btc_market(btc_df_1h)
        if btc_reject:
            result.reject_reason = f"BTC 시장 필터: {btc_reject}"
            logger.info(f"[{symbol}] HOLD — {result.reject_reason}")
            return result

    # ── Step 1. 지표 계산 ─────────────────────────────────────────────────────
    trend_1h  = ti.calculate(df_1h)
    mom_15m   = mi.calculate(df_15m)
    vol_15m   = vi.calculate(df_15m)

    # 지표 계산 실패 시 즉시 HOLD
    if trend_1h is None:
        result.reject_reason = "1H 추세 지표 계산 실패 (데이터 부족)"
        logger.warning(f"[{symbol}] {result.reject_reason}")
        return result
    if mom_15m is None:
        result.reject_reason = "15M 모멘텀 지표 계산 실패 (데이터 부족)"
        logger.warning(f"[{symbol}] {result.reject_reason}")
        return result
    if vol_15m is None:
        result.reject_reason = "15M 거래량 지표 계산 실패 (데이터 부족)"
        logger.warning(f"[{symbol}] {result.reject_reason}")
        return result

    # 계산 결과 저장
    result.trend      = trend_1h
    result.momentum   = mom_15m
    result.volume     = vol_15m
    result.entry_price = trend_1h.close

    # ── Step 2. 방향 판단 ─────────────────────────────────────────────────────
    long_signal, long_reject  = _check_long(symbol, trend_1h, mom_15m, vol_15m)
    short_signal, short_reject = _check_short(symbol, trend_1h, mom_15m, vol_15m)

    if not long_signal and not short_signal:
        # 롱/숏 둘 다 필수 조건 실패
        result.reject_reason = f"롱: [{long_reject}]  숏: [{short_reject}]"
        logger.info(f"[{symbol}] HOLD — {result.reject_reason}")
        return result

    result.passed_required = True

    # ── Step 3. 보조 조건 점수 계산 ──────────────────────────────────────────
    if long_signal:
        score, details = _score_long(trend_1h, mom_15m, vol_15m)
        direction = "LONG"
    else:
        score, details = _score_short(trend_1h, mom_15m, vol_15m)
        direction = "SHORT"

    result.score        = score
    result.bonus_details = details

    # ── Step 4. 최소 점수 필터 ───────────────────────────────────────────────
    if score < min_score_to_enter:
        result.reject_reason = (
            f"점수 부족: {score}/{min_score_to_enter} "
            f"(방향={direction})"
        )
        logger.info(
            f"[{symbol}] HOLD — {result.reject_reason}  "
            f"세부: {details}"
        )
        return result

    # ── Step 5. 최종 신호 확정 ───────────────────────────────────────────────
    result.signal = direction
    logger.info(
        f"[{symbol}] ★ {direction} 신호 확정  "
        f"score={score}/{MAX_SCORE}  "
        f"price={result.entry_price}  "
        f"세부: {details}"
    )
    return result


# ── 롱 필수 조건 체크 ──────────────────────────────────────────────────────────

def _check_long(
    symbol:   str,
    trend:    TrendResult,
    momentum: MomentumResult,
    volume:   VolumeResult,
) -> tuple[bool, str]:
    """
    롱 진입 필수 조건 체크 (AND 로직 — 하나라도 실패 시 False).

    조건:
      [추세]
        T1. 1H EMA20 > EMA50 (단기 상승 추세)
        T2. ADX ≥ 25        (추세 존재)
        T3. +DI > -DI       (방향성: 매수 우세)
      [모멘텀]
        M1. 15M MACD 히스토그램 양수 (매수 모멘텀 존재)
        M2. RSI 40~65 구간  (과매수 아닌 상태)
      [거래량]
        V1. 거래량 급등 (≥ 평균 × 2.0)

    Returns
    -------
    (passed: bool, reject_reason: str)
    """
    # T1. EMA 단기 배열
    if trend.ema_short <= trend.ema_mid:
        return False, f"EMA20({trend.ema_short:.2f}) ≤ EMA50({trend.ema_mid:.2f})"

    # T2. ADX 추세 강도
    # ADX 추세 강도
    if trend.adx < 25:
        return False, f"ADX={trend.adx:.1f} < 25 (추세 없음)"

    # T3. +DI > -DI
    if trend.adx_plus_di <= trend.adx_minus_di:
        return False, f"+DI({trend.adx_plus_di:.1f}) ≤ -DI({trend.adx_minus_di:.1f})"

    # M1. MACD 히스토그램 양수
    if momentum.macd_hist <= 0:
        return False, f"MACD hist={momentum.macd_hist:.6f} ≤ 0"

    # M2. RSI 범위
    # [테스트 완화] 30~75 (원래: 40~65)
    if not (30 <= momentum.rsi <= 75):
        return False, f"RSI={momentum.rsi:.1f} 롱 진입 구간(40~65) 벗어남"

    # V1. 거래량 급등 — 필수 조건 제거, signal_scorer 감점으로만 처리
    # 원래: 2.0x 필수 → 테스트 완화 1.0x → 현재: 조건 제거
    # 이유: 특정 15분 캔들 거래량이 낮다고 진입 차단하면 좋은 종목을 놓침

    return True, ""


# ── 숏 필수 조건 체크 ──────────────────────────────────────────────────────────

def _check_short(
    symbol:   str,
    trend:    TrendResult,
    momentum: MomentumResult,
    volume:   VolumeResult,
) -> tuple[bool, str]:
    """
    숏 진입 필수 조건 체크 (AND 로직).

    조건:
      [추세]
        T1. 1H EMA20 < EMA50 (단기 하락 추세)
        T2. ADX ≥ 25
        T3. -DI > +DI       (방향성: 매도 우세)
      [모멘텀]
        M1. 15M MACD 히스토그램 음수
        M2. RSI 35~60 구간
      [거래량]
        V1. 거래량 급등
    """
    # T1. EMA 단기 역배열
    if trend.ema_short >= trend.ema_mid:
        return False, f"EMA20({trend.ema_short:.2f}) ≥ EMA50({trend.ema_mid:.2f})"

    # T2. ADX 추세 강도
    # ADX 추세 강도
    if trend.adx < 25:
        return False, f"ADX={trend.adx:.1f} < 25"

    # T3. -DI > +DI
    if trend.adx_minus_di <= trend.adx_plus_di:
        return False, f"-DI({trend.adx_minus_di:.1f}) ≤ +DI({trend.adx_plus_di:.1f})"

    # M1. MACD 히스토그램 음수
    if momentum.macd_hist >= 0:
        return False, f"MACD hist={momentum.macd_hist:.6f} ≥ 0"

    # M2. RSI 범위
    # [테스트 완화] 25~70 (원래: 35~60)
    if not (25 <= momentum.rsi <= 70):
        return False, f"RSI={momentum.rsi:.1f} 숏 진입 구간(35~60) 벗어남"

    # V1. 거래량 급등 — 필수 조건 제거, signal_scorer 감점으로만 처리
    # 원래: 2.0x 필수 → 테스트 완화 1.0x → 현재: 조건 제거

    return True, ""


# ── 롱 보조 점수 ───────────────────────────────────────────────────────────────

def _score_long(
    trend:    TrendResult,
    momentum: MomentumResult,
    volume:   VolumeResult,
) -> tuple[int, dict]:
    """
    롱 보조 조건 점수 합산 (OR 로직).
    각 조건 충족 시 점수 가산, 미충족 시 0점.

    총점 기준:
      만점  14점
      진입  5점 이상
    """
    details = {}
    score   = 0

    def add(name: str, pts: int, cond: bool) -> None:
        if cond:
            details[name] = pts
            nonlocal score
            score += pts

    # ── 추세 보조 (최대 5점) ──────────────────────────────────────────────────
    add("EMA 완전 정배열",    2, trend.ema_aligned_up)          # EMA20>50>200
    add("골든크로스",         2, trend.golden_cross)             # 이번 캔들 교차
    add("BB 중심선 위",       1, trend.above_bb_mid)             # 현재가 > BB 중심
    add("BB 하단 근접 반등",  1, trend.near_bb_lower)            # 하단에서 출발
    add("강한 추세(ADX≥40)",  1, trend.trend_strength == "STRONG")

    # ── 모멘텀 보조 (최대 5점) ───────────────────────────────────────────────
    add("MACD 불크로스",      2, momentum.macd_bull_cross)       # 히스토 음→양 전환
    add("StochRSI 불크로스",  2, momentum.stoch_bull_cross)      # K가 D 상향 교차
    add("MACD 모멘텀 강화",   1, momentum.macd_hist_growing)     # 히스토 절대값 증가
    add("RSI 상승 중",        1, momentum.rsi_rising)
    add("StochRSI 과매도 탈출", 1,
        momentum.stoch_oversold is False and momentum.stoch_k_above_d)

    # ── 거래량 보조 (최대 4점) ───────────────────────────────────────────────
    add("양봉+거래량급등",    2, volume.bull_volume)             # 가장 강한 매수 확인
    add("강한 급등(3x↑)",    1, volume.volume_strong_surge)
    add("OBV 상승 추세",      1, volume.obv_rising)
    add("CVD 매수 우세",      1, volume.cvd_positive)
    add("강세 다이버전스",    2, volume.divergence_bull)         # 반등 신호
    add("OBV 신고점",         1, volume.obv_new_high)

    return score, details


# ── 숏 보조 점수 ───────────────────────────────────────────────────────────────

def _score_short(
    trend:    TrendResult,
    momentum: MomentumResult,
    volume:   VolumeResult,
) -> tuple[int, dict]:
    """
    숏 보조 조건 점수 합산 (롱과 대칭 구조).
    """
    details = {}
    score   = 0

    def add(name: str, pts: int, cond: bool) -> None:
        if cond:
            details[name] = pts
            nonlocal score
            score += pts

    # ── 추세 보조 ──────────────────────────────────────────────────────────────
    add("EMA 완전 역배열",    2, trend.ema_aligned_dn)
    add("데드크로스",         2, trend.dead_cross)
    add("BB 중심선 아래",     1, not trend.above_bb_mid)
    add("BB 상단 근접 반락",  1, trend.near_bb_upper)
    add("강한 추세(ADX≥40)",  1, trend.trend_strength == "STRONG")

    # ── 모멘텀 보조 ────────────────────────────────────────────────────────────
    add("MACD 베어크로스",    2, momentum.macd_bear_cross)
    add("StochRSI 베어크로스",2, momentum.stoch_bear_cross)
    add("MACD 모멘텀 강화",   1, momentum.macd_hist_growing)
    add("RSI 하락 중",        1, not momentum.rsi_rising)
    add("StochRSI 과매수 탈출", 1,
        momentum.stoch_overbought is False and not momentum.stoch_k_above_d)

    # ── 거래량 보조 ────────────────────────────────────────────────────────────
    add("음봉+거래량급등",    2, volume.bear_volume)
    add("강한 급등(3x↑)",    1, volume.volume_strong_surge)
    add("OBV 하락 추세",      1, volume.obv_falling)
    add("CVD 매도 우세",      1, volume.cvd_negative)
    add("약세 다이버전스",    2, volume.divergence_bear)
    add("OBV 신저점",         1, volume.obv_new_low)

    return score, details



# ── BTC 시장 방향 필터 ────────────────────────────────────────────────────────

def _check_btc_market(btc_df: pd.DataFrame) -> str:
    """
    BTC 1H 차트로 시장 환경을 판단. 위험한 환경이면 거절 사유 문자열 반환.
    정상이면 빈 문자열 반환.

    판단 기준:
      - BTC가 EMA20 아래 + ADX 상승 중 = 하락 추세 강화 → 롱 위험
      - BTC 1H 캔들 ATR 비율 > 3% = 급변동 구간 → 신호 신뢰도 하락
      - BTC EMA20 < EMA50 = 단기 하락 구조

    Returns
    -------
    str : 거절 사유. 빈 문자열이면 시장 환경 정상.
    """
    btc_trend = ti.calculate(btc_df)
    if btc_trend is None:
        return ""   # BTC 데이터 계산 실패 시 필터 생략 (안전 방향)

    # BTC 급변동 구간 (ATR > 3%) — 알트 신호 신뢰도 하락
    if btc_trend.atr_ratio > 0.03:
        return (
            f"BTC 급변동 구간 (ATR={btc_trend.atr_ratio*100:.2f}% > 3%)"
        )

    # BTC 단기 하락 추세 필터 — 테스트 중 비활성화
    # 알트 개별 급등 케이스(RAVE, BLESS 등)를 막을 수 있어서 주석 처리
    # 실거래 전환 시 재활성화 검토
    # btc_bear = (
    #     btc_trend.ema_short < btc_trend.ema_mid
    #     and btc_trend.adx_minus_di > btc_trend.adx_plus_di
    #     and btc_trend.adx >= 30
    # )
    # if btc_bear:
    #     return (
    #         f"BTC 하락 추세 (EMA20<EMA50, -DI>{btc_trend.adx_plus_di:.1f}, "
    #         f"ADX={btc_trend.adx:.1f})"
    #     )

    return ""

# ── 유틸 ───────────────────────────────────────────────────────────────────────

def summary(result: SignalResult) -> str:
    """SignalResult 한 줄 요약 — 로깅/디버깅용."""
    if result.signal == "HOLD":
        return (
            f"[신호] HOLD  "
            f"사유: {result.reject_reason}"
        )
    return (
        f"[신호] ★ {result.signal}  "
        f"score={result.score}/{MAX_SCORE}  "
        f"price={result.entry_price}  "
        f"bonus={result.bonus_details}"
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

    TEST_SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]

    def fetch_df(symbol: str, tf: str, limit: int) -> pd.DataFrame:
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df.astype(float)

    for sym in TEST_SYMBOLS:
        print(f"\n{'='*72}")
        print(f"  {sym}")
        print('='*72)

        df_1h  = fetch_df(sym, "1h",  250)
        df_15m = fetch_df(sym, "15m", 150)

        result = check(sym, df_1h, df_15m)

        print(summary(result))
        print()

        if result.signal != "HOLD":
            print(f"  진입가    : {result.entry_price}")
            print(f"  점수      : {result.score}/{MAX_SCORE}")
            print(f"  보조 조건 :")
            for k, v in result.bonus_details.items():
                print(f"    +{v}점  {k}")
        else:
            print(f"  거절 사유 : {result.reject_reason}")

        if result.trend:
            print(f"\n  [추세]")
            print(f"    방향={result.trend.trend_direction}  "
                  f"강도={result.trend.trend_strength}  "
                  f"ADX={result.trend.adx:.1f}")
        if result.momentum:
            print(f"  [모멘텀]")
            print(f"    MACD hist={result.momentum.macd_hist:+.6f}  "
                  f"RSI={result.momentum.rsi:.1f}  "
                  f"StochK={result.momentum.stoch_k:.1f}")
        if result.volume:
            print(f"  [거래량]")
            print(f"    ratio={result.volume.volume_ratio:.2f}x  "
                  f"OBV rising={result.volume.obv_rising}  "
                  f"CVD pos={result.volume.cvd_positive}")

        time.sleep(0.5)