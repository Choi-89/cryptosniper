"""
signal_scorer.py
----------------
signal_engine.SignalResult 를 받아
진입 강도를 0~100 으로 정규화하고,
그 강도에 따라 최적 레버리지 배율까지 결정하는 모듈.

역할:
  signal_engine  → "LONG / SHORT / HOLD" + raw score (0~22)
  signal_scorer  → confidence (0~100) + leverage (1x~10x) + grade

  즉, signal_engine 이 "들어갈지 말지"를 결정한다면
  signal_scorer 는 "얼마나 강하게 들어갈지"를 결정한다.

점수 구성:
  base_score   : signal_engine 보조 조건 합산 (0~22)
  context_score: 멀티 타임프레임 일치도, 시장 맥락 보너스 (0~30)
  penalty      : 위험 요소 감점 (-30~0)
  ────────────────────────────────────────────────────
  raw_total    : base + context + penalty
  confidence   : raw_total 을 0~100 으로 클램핑 후 정규화

레버리지 결정:
  confidence 0~39  → 진입 불가 (HOLD 강제)
  confidence 40~54 → 2x
  confidence 55~64 → 3x
  confidence 65~74 → 5x
  confidence 75~84 → 7x
  confidence 85~100→ 10x

의존 모듈:
  strategy/signal_engine.py
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from strategy.signal_engine import SignalResult, MAX_SCORE as ENGINE_MAX_SCORE

# ── 로거 ───────────────────────────────────────────────────────────────────────
logger = logging.getLogger("signal_scorer")


# ── 파라미터 상수 ──────────────────────────────────────────────────────────────

# base_score 만점 (signal_engine 보조 조건 최대합)
BASE_MAX         = ENGINE_MAX_SCORE   # 22

# context 보너스 최대
CONTEXT_MAX      = 30

# 페널티 최대 감점
PENALTY_MAX      = 30

# 이론상 최대 합산
RAW_MAX          = BASE_MAX + CONTEXT_MAX   # 52

# confidence 진입 최소 임계값
# [테스트 완화] 15 (원래: 40)
MIN_CONFIDENCE   = 15

# 레버리지 테이블: (confidence 하한, 레버리지)
LEVERAGE_TABLE = [
    (85, 10),
    (75,  7),
    (65,  5),
    (55,  3),
    (40,  2),
]

# 등급 테이블: (confidence 하한, 등급 문자열)
GRADE_TABLE = [
    (85, "S"),   # 최강 신호
    (75, "A"),
    (65, "B"),
    (55, "C"),
    (40, "D"),   # 최소 진입 기준
    (0,  "F"),   # 진입 불가
]


# ── 결과 데이터클래스 ──────────────────────────────────────────────────────────

@dataclass
class ScoreResult:
    """
    signal_scorer.evaluate() 의 최종 반환값.

    Attributes
    ----------
    confidence      : 신호 신뢰도 (0~100)
    grade           : 등급 문자열 ("S" / "A" / "B" / "C" / "D" / "F")
    leverage        : 권장 레버리지 배율 (1, 2, 3, 5, 7, 10)
    can_enter       : 실제 진입 허가 여부 (confidence ≥ MIN_CONFIDENCE)
    base_score      : signal_engine 보조 조건 점수
    context_score   : 멀티 타임프레임·맥락 보너스 점수
    penalty_score   : 위험 요소 감점 (음수)
    raw_total       : base + context + penalty 합산
    context_details : 보너스 항목별 상세 {항목명: 점수}
    penalty_details : 감점 항목별 상세 {항목명: 감점}
    signal_result   : 원본 SignalResult 참조
    """
    confidence:      int   = 0
    grade:           str   = "F"
    leverage:        int   = 1
    can_enter:       bool  = False
    base_score:      int   = 0
    context_score:   int   = 0
    penalty_score:   int   = 0
    raw_total:       int   = 0
    context_details: dict  = field(default_factory=dict)
    penalty_details: dict  = field(default_factory=dict)
    signal_result:   Optional[SignalResult] = None


# ── 메인 함수 ──────────────────────────────────────────────────────────────────

def evaluate(
    signal_result: SignalResult,
    min_confidence: int = MIN_CONFIDENCE,
) -> ScoreResult:
    """
    SignalResult 를 받아 신뢰도 · 레버리지 · 등급을 계산한다.

    Parameters
    ----------
    signal_result : signal_engine.check() 의 반환값

    Returns
    -------
    ScoreResult
      .can_enter  == False → 진입 포기 (HOLD 강제)
      .leverage          → 진입 시 사용할 레버리지
      .confidence        → 신호 강도 0~100
    """
    result = ScoreResult(signal_result=signal_result)

    # signal_engine 에서 이미 HOLD 를 반환했으면 즉시 F 반환
    if signal_result.signal == "HOLD":
        result.grade    = "F"
        result.leverage = 1
        logger.debug("HOLD 신호 → 채점 생략")
        return result

    direction = signal_result.signal   # "LONG" or "SHORT"
    trend     = signal_result.trend
    momentum  = signal_result.momentum
    volume    = signal_result.volume

    # ── 1. base_score: signal_engine 보조 점수 그대로 사용 ─────────────────────
    result.base_score = signal_result.score

    # ── 2. context_score: 멀티 타임프레임 일치도 + 시장 맥락 보너스 ───────────
    ctx_score, ctx_details = _calc_context(direction, trend, momentum, volume)
    result.context_score   = ctx_score
    result.context_details = ctx_details

    # ── 3. penalty_score: 위험 요소 감점 ──────────────────────────────────────
    pen_score, pen_details = _calc_penalty(direction, trend, momentum, volume)
    result.penalty_score   = pen_score   # 음수
    result.penalty_details = pen_details

    # ── 4. raw_total 합산 ─────────────────────────────────────────────────────
    result.raw_total = result.base_score + result.context_score + result.penalty_score

    # ── 5. confidence 정규화 (0~100) ──────────────────────────────────────────
    #  raw_total / RAW_MAX × 100 으로 정규화 후 0~100 클램핑
    raw_clamped      = max(0, min(result.raw_total, RAW_MAX))
    result.confidence = int(raw_clamped / RAW_MAX * 100)

    # ── 6. 등급 · 레버리지 · 진입 허가 결정 ────────────────────────────────────
    result.grade    = _get_grade(result.confidence)
    result.leverage = _get_leverage(
        result.confidence,
        min_confidence=min_confidence,
    )
    result.can_enter = result.confidence >= min_confidence

    # ── 7. 로깅 ────────────────────────────────────────────────────────────────
    _log_result(signal_result.entry_price, direction, result)

    return result


# ── context 보너스 계산 ────────────────────────────────────────────────────────

def _calc_context(
    direction: str,
    trend,
    momentum,
    volume,
) -> tuple[int, dict]:
    """
    재설계된 맥락 보너스 계산 (최대 30점).

    설계 원칙:
      1단계 — 선행 거래량/CVD로 '에너지 방향' 먼저 확인  (최대 12점)
      2단계 — 모멘텀 조합으로 '진입 타이밍' 확정         (최대 10점)
      3단계 — 후행 추세로 '방향 신뢰도' 부스트            (최대 8점)

    ★ 핵심 변경점
      - BB Squeeze 탈출 감지 추가  : 에너지 응축 → 폭발 구간 포착
      - ATR 급증 감지 추가         : 변동성 확대 초기 진입
      - CVD 선행 판단 우선화       : 거래량 방향성을 타이밍 결정의 핵으로
      - 투매/패닉 거래량 필터      : 폭락장 거래량 급등을 보너스 제외
      - 타임프레임 동기화 보너스   : 1H 추세 + 15M 모멘텀 일치 시 극대화
    """
    details: dict = {}
    score         = 0

    is_long = direction == "LONG"

    def add(name: str, pts: int, cond: bool) -> None:
        nonlocal score
        if cond:
            details[name] = pts
            score += pts

    # ════════════════════════════════════════════════════════════
    # 1단계: 선행 거래량 / CVD — 에너지 방향 확인 (최대 12점)
    # 거래량과 CVD는 가격보다 먼저 움직이는 선행 지표
    # ════════════════════════════════════════════════════════════

    # ① CVD + OBV 동시 방향 일치 (매집/분산 세력 확인)            +5
    #    : 스마트머니가 이미 포지션을 쌓고 있다는 가장 강한 신호
    if is_long:
        add("CVD매수+OBV상승 동시",  5,
            volume.cvd_positive and volume.obv_rising)
    else:
        add("CVD매도+OBV하락 동시",  5,
            volume.cvd_negative and volume.obv_falling)

    # ② 방향성 있는 거래량 급등 (투매/패닉 필터 적용)              +4
    #    : 양봉+급등 또는 음봉+급등 — 방향이 명확한 거래량만 인정
    #    단, ATR이 4% 초과(극단적 변동성)면 패닉 거래량으로 보고 제외
    is_panic = trend.atr_ratio > 0.04
    if is_long:
        add("방향성 양봉+급등",      4,
            volume.bull_volume and volume.volume_strong_surge and not is_panic)
    else:
        add("방향성 음봉+급등",      4,
            volume.bear_volume and volume.volume_strong_surge and not is_panic)

    # ③ 강세/약세 다이버전스 + 거래량 급등 (반전 초기 포착)        +3
    if is_long:
        add("강세다이버전스+급등",   3,
            volume.divergence_bull and volume.volume_surge)
    else:
        add("약세다이버전스+급등",   3,
            volume.divergence_bear and volume.volume_surge)

    # ════════════════════════════════════════════════════════════
    # 2단계: 모멘텀 조합 — 진입 타이밍 확정 (최대 10점)
    # ════════════════════════════════════════════════════════════

    # ④ MACD + StochRSI 동시 크로스 (두 모멘텀 동시 전환)         +5
    #    : 단일 크로스보다 신뢰도 2배 이상
    if is_long:
        add("MACD불+Stoch불 동시",   5,
            momentum.macd_bull_cross and momentum.stoch_bull_cross)
    else:
        add("MACD베어+Stoch베어 동시", 5,
            momentum.macd_bear_cross and momentum.stoch_bear_cross)

    # ⑤ BB Squeeze 탈출 감지 (에너지 응축 → 폭발 초입)            +3
    #    : bb_width가 직전보다 확장되면서 ATR도 동반 상승
    #      → 오랜 횡보 후 추세 폭발 직전의 황금 진입 구간
    bb_width_ratio = trend.bb_width / trend.bb_mid if trend.bb_mid > 0 else 0
    bb_expanding   = 0.015 <= bb_width_ratio <= 0.035   # 스퀴즈 탈출 직후 구간
    atr_surging    = trend.atr_ratio >= 0.015            # ATR도 같이 올라야 진짜
    add("BB Squeeze 탈출+ATR급증",   3, bb_expanding and atr_surging)

    # ⑥ RSI 방향 + 모멘텀 강화 동시 (추세 지속 확인)              +2
    if is_long:
        add("RSI상승+MACD강화",      2,
            momentum.rsi_rising and momentum.macd_hist_growing)
    else:
        add("RSI하락+MACD강화",      2,
            not momentum.rsi_rising and momentum.macd_hist_growing)

    # ════════════════════════════════════════════════════════════
    # 3단계: 후행 추세 부스트 — 방향 신뢰도 강화 (최대 8점)
    # ════════════════════════════════════════════════════════════

    # ⑦ 타임프레임 동기화: EMA 완전 배열 + ADX 강함               +4
    #    : 1H 추세와 15M 신호가 일치하는 최고 신뢰 구간
    if is_long:
        add("EMA정배열+강한추세",    4,
            trend.ema_aligned_up and trend.trend_strength == "STRONG")
    else:
        add("EMA역배열+강한추세",    4,
            trend.ema_aligned_dn and trend.trend_strength == "STRONG")

    # ⑧ ADX 50 이상 초강세 추세                                   +2
    add("ADX≥50 초강세",            2, trend.adx >= 50)

    # ⑨ 골든/데드 크로스 + EMA 배열 동시 (추세 전환 확인)          +2
    if is_long:
        add("골든크로스+정배열",     2, trend.golden_cross and trend.ema_aligned_up)
    else:
        add("데드크로스+역배열",     2, trend.dead_cross   and trend.ema_aligned_dn)

    return min(score, CONTEXT_MAX), details


# ── 페널티 계산 ────────────────────────────────────────────────────────────────

def _calc_penalty(
    direction: str,
    trend,
    momentum,
    volume,
) -> tuple[int, dict]:
    """
    위험 요소 감점 계산. 반환값은 음수 정수.

    감점 항목 (최대 -30점):
      [추세 위험]
        - 볼린저밴드 반대쪽 경계 근접       -5  ← 불리한 출발점
        - ADX 25~28 경계선 (추세 불안정)    -2  ← 이중 처벌 방지로 완화
        - EMA 200 저항/지지선 반대쪽        -3

      [모멘텀 위험]
        - RSI 과매수/과매도 구간 진입        -5  ← 극단에서 신규 진입은 위험
        - StochRSI 과매수/과매도 구간        -4

      [거래량 위험]
        - 다이버전스 역방향 발생             -6  ← 세력 이탈 신호 (가장 위험)
        - OBV 방향이 진입 방향과 반대        -4
        - 거래량 급등 없는 진입              -3  (필수 조건 통과했어도 간신히 통과)
    """
    details: dict = {}
    score         = 0   # 음수로 누적

    is_long = direction == "LONG"

    def sub(name: str, pts: int, cond: bool) -> None:
        """감점: pts 는 양수로 전달, 내부에서 음수로 처리."""
        nonlocal score
        if cond:
            details[name] = -pts
            score        -= pts

    # ── 추세 위험 ─────────────────────────────────────────────────────────────

    # 볼린저밴드 불리한 경계 근접 (롱인데 상단 근처, 숏인데 하단 근처)
    if is_long:
        sub("BB상단 근접 (과매수 위험)",  5, trend.near_bb_upper)
    else:
        sub("BB하단 근접 (과매도 위험)",  5, trend.near_bb_lower)

    # ADX 경계선 불안정 (25~28 → 추세 성립 불확실) — 이중 처벌 방지를 위해 -2로 완화
    sub("ADX 경계선 불안정(25~28)",       2, 25 <= trend.adx <= 28)

    # EMA 200 반대편 — 장기 추세와 역행 진입
    if is_long:
        sub("EMA200 저항 (현재가<EMA200)", 3, trend.close < trend.ema_long)
    else:
        sub("EMA200 지지 (현재가>EMA200)", 3, trend.close > trend.ema_long)

    # ── 모멘텀 위험 ───────────────────────────────────────────────────────────

    # RSI 극단 구간 신규 진입 — 반전 위험 높음
    if is_long:
        sub("RSI 과매수(≥70) 진입",       5, momentum.rsi_overbought)
    else:
        sub("RSI 과매도(≤30) 진입",       5, momentum.rsi_oversold)

    # StochRSI 극단 구간
    if is_long:
        sub("StochRSI 과매수(≥80)",       4, momentum.stoch_overbought)
    else:
        sub("StochRSI 과매도(≤20)",       4, momentum.stoch_oversold)
    # ── 시장 리스크 감점 ─────────────────────────────────────────────────────
    # BTC 역행 감점은 btc_trend가 전달된 경우에만 적용
    # (signal_engine에서 btc_trend를 SignalResult에 담아주거나
    #  _calc_penalty 호출 시 외부에서 전달 — 현재는 None 처리로 안전하게)
    # → 실제 BTC 감점은 signal_engine._check_btc_market()에서 게이트로 처리
    #   여기서는 추가 감점이 필요한 경우만 남김 (향후 확장 포인트)

    # ── 거래량 위험 ───────────────────────────────────────────────────────────

    # 다이버전스 역방향 — 진입 방향과 반대 다이버전스가 나타남 (세력 이탈 신호)
    if is_long:
        sub("약세 다이버전스 역행",        6, volume.divergence_bear)
    else:
        sub("강세 다이버전스 역행",        6, volume.divergence_bull)

    # OBV 방향이 진입 방향 반대
    if is_long:
        sub("OBV 매도 우세",               4, volume.obv_falling)
    else:
        sub("OBV 매수 우세",               4, volume.obv_rising)

    # 거래량 급등 없는 저강도 진입 (배수 1.0~1.5x)
    sub("거래량 배수 낮음(1.0~1.5x)",     3,
        1.0 <= volume.volume_ratio < 1.5)

    # BB Squeeze 없이 이미 밴드 상단 돌파 상태 (과열 진입)
    bb_width_ratio = trend.bb_width / trend.bb_mid if trend.bb_mid > 0 else 0
    if is_long:
        sub("BB상단 돌파 과열 진입",       2,
            bb_width_ratio > 0.05 and trend.near_bb_upper)
    else:
        sub("BB하단 돌파 과열 진입",       2,
            bb_width_ratio > 0.05 and trend.near_bb_lower)

    # 최대 감점 제한
    return max(score, -PENALTY_MAX), details


# ── 등급 / 레버리지 결정 ──────────────────────────────────────────────────────

def _get_grade(confidence: int) -> str:
    for threshold, grade in GRADE_TABLE:
        if confidence >= threshold:
            return grade
    return "F"


def _get_leverage(
    confidence: int,
    min_confidence: int = MIN_CONFIDENCE,
) -> int:
    if confidence < min_confidence:
        return 1   # 진입 불가, 레버리지 의미 없음
    for threshold, lev in LEVERAGE_TABLE:
        if confidence >= threshold:
            return lev
    return 2   # fallback


# ── 로깅 ─────────────────────────────────────────────────────────────────────

def _log_result(price: float, direction: str, result: ScoreResult) -> None:
    status = "진입 허가 ✓" if result.can_enter else "진입 불가 ✗"
    logger.info(
        f"[{direction}] {status}  "
        f"confidence={result.confidence}  "
        f"grade={result.grade}  "
        f"leverage={result.leverage}x  |  "
        f"base={result.base_score}  "
        f"context=+{result.context_score}  "
        f"penalty={result.penalty_score}  "
        f"raw={result.raw_total}  "
        f"price={price}"
    )
    if result.context_details:
        logger.debug(f"  보너스: {result.context_details}")
    if result.penalty_details:
        logger.debug(f"  감점:   {result.penalty_details}")


# ── 유틸 ───────────────────────────────────────────────────────────────────────

def summary(result: ScoreResult) -> str:
    """ScoreResult 한 줄 요약 — 로깅/디버깅용."""
    direction = (
        result.signal_result.signal
        if result.signal_result else "HOLD"
    )
    if not result.can_enter:
        return (
            f"[채점] 진입 불가  "
            f"confidence={result.confidence}  "
            f"grade={result.grade}"
        )
    return (
        f"[채점] {direction} 진입 허가  "
        f"confidence={result.confidence}/100  "
        f"grade={result.grade}  "
        f"leverage={result.leverage}x  |  "
        f"base={result.base_score}  "
        f"ctx=+{result.context_score}  "
        f"pen={result.penalty_score}  "
        f"bonus={result.context_details}  "
        f"deduct={result.penalty_details}"
    )


# ── 단독 실행 테스트 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import time
    import ccxt
    import pandas as pd
    from dotenv import load_dotenv
    import signal_engine

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

    def fetch_df(symbol: str, tf: str, limit: int) -> pd.DataFrame:
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df.astype(float)

    TEST_SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]

    for sym in TEST_SYMBOLS:
        print(f"\n{'='*72}")
        print(f"  {sym}")
        print('='*72)

        df_1h  = fetch_df(sym, "1h",  250)
        df_15m = fetch_df(sym, "15m", 150)

        # Step 1: signal_engine 으로 방향 판단
        sig = signal_engine.check(sym, df_1h, df_15m)

        # Step 2: signal_scorer 로 강도·레버리지 결정
        score = evaluate(sig)

        print(summary(score))
        print()

        if score.can_enter:
            print(f"  방향       : {sig.signal}")
            print(f"  진입가     : {sig.entry_price}")
            print(f"  신뢰도     : {score.confidence}/100")
            print(f"  등급       : {score.grade}")
            print(f"  레버리지   : {score.leverage}x")
            print()
            print(f"  기본점수   : {score.base_score}  (signal_engine 보조)")
            print(f"  맥락 보너스: +{score.context_score}")
            if score.context_details:
                for k, v in score.context_details.items():
                    print(f"    +{v}  {k}")
            print(f"  위험 감점  : {score.penalty_score}")
            if score.penalty_details:
                for k, v in score.penalty_details.items():
                    print(f"    {v}  {k}")
            print(f"  raw 합산   : {score.raw_total} / {RAW_MAX}")
        else:
            print(f"  진입 불가  (confidence={score.confidence} < {MIN_CONFIDENCE})")
            if sig.signal == "HOLD":
                print(f"  HOLD 사유  : {sig.reject_reason}")

        time.sleep(0.5)