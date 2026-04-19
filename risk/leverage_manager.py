"""
leverage_manager.py
-------------------
signal_scorer 의 기본 레버리지에 시장 상황·리스크 상태를 반영해
최종 레버리지를 결정하고 거래소에 실제로 설정하는 모듈.

역할:
  1. signal_scorer 의 confidence 기반 레버리지(기준값) 수신
  2. ADX / ATR / 일일 손실률 / 연속 손절 수에 따라 승수 조정
  3. 심볼별 허용 최대 레버리지 상한 적용 (변동성 큰 알트는 제한)
  4. ccxt 로 거래소에 레버리지 실제 설정
  5. 설정 이력 캐시 (불필요한 API 호출 방지)

최종 레버리지 결정 순서:
  base_leverage    ← signal_scorer.ScoreResult.leverage
  × market_mult    ← ADX / ATR 기반 시장 환경 보정 (0.5 ~ 1.2)
  × risk_mult      ← 일일 손실률 / 연속 손절 보정 (0.5 ~ 1.0)
  → adjusted       = base × market_mult × risk_mult (소수점 버림)
  → final          = clamp(adjusted, MIN_LEV, symbol_max)

의존 모듈:
  strategy/signal_scorer.py  (ScoreResult)
  risk/circuit_breaker.py    (CircuitStatus)
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import ccxt

DEMO_TRADING_URLS = {
    "api": {
        "fapiPublic":    "https://testnet.binancefuture.com/fapi/v1",
        "fapiPrivate":   "https://testnet.binancefuture.com/fapi/v1",
        "fapiPublicV2":  "https://testnet.binancefuture.com/fapi/v2",
        "fapiPrivateV2": "https://testnet.binancefuture.com/fapi/v2",
        "fapiPublicV3":  "https://testnet.binancefuture.com/fapi/v3",
        "fapiPrivateV3": "https://testnet.binancefuture.com/fapi/v3",
        "public":  "https://testnet.binancefuture.com/fapi/v1",
        "private": "https://testnet.binancefuture.com/fapi/v1",
    }
}


from strategy.signal_scorer   import ScoreResult
from circuit_breaker import CircuitStatus, HaltLevel

# ── 로거 ───────────────────────────────────────────────────────────────────────
logger = logging.getLogger("leverage_manager")


# ── 파라미터 상수 ──────────────────────────────────────────────────────────────

# 전역 레버리지 한도
MIN_LEVERAGE      = 1
MAX_LEVERAGE      = 10    # 전체 상한 (어떤 상황에서도 초과 불가)

# ADX 기반 시장 환경 승수
ADX_BOOST_THRESH  = 40    # ADX ≥ 40 → 강한 추세 → 승수 1.2
ADX_REDUCE_THRESH = 28    # ADX < 28 → 약한 추세 → 승수 0.8
ADX_SIDEWAYS      = 25    # ADX < 25 → 횡보 → 승수 0.5 (진입 자체를 막는 건 signal_engine 역할)

# ATR 비율 기반 변동성 승수
ATR_HIGH_RATIO    = 0.04  # ATR/price > 4% → 고변동성 → 승수 0.7
ATR_LOW_RATIO     = 0.015 # ATR/price < 1.5% → 저변동성 → 승수 1.1

# 일일 손실률 기반 리스크 승수
DAILY_LOSS_REDUCE_1 = 0.015  # 1.5% 손실 → 승수 0.8
DAILY_LOSS_REDUCE_2 = 0.025  # 2.5% 손실 → 승수 0.6

# 연속 손절 기반 리스크 승수
CONSEC_REDUCE_1   = 1     # 1회 손절 → 승수 0.9
CONSEC_REDUCE_2   = 2     # 2회 손절 → 승수 0.7

# 심볼별 최대 레버리지 상한
# 변동성이 큰 알트코인은 상한을 낮춰 강제 제한
SYMBOL_MAX_LEVERAGE: dict[str, int] = {
    "BTC/USDT:USDT":  10,
    "ETH/USDT:USDT":  10,
    "BNB/USDT:USDT":   7,
    "SOL/USDT:USDT":   7,
    "XRP/USDT:USDT":   5,
    "DOGE/USDT:USDT":  5,
    "ADA/USDT:USDT":   5,
    # 나머지 알트코인 기본 상한
    "_DEFAULT":        5,
}

# 거래소 레버리지 캐시 유효 시간 (초) — 같은 값이면 API 재호출 불필요
CACHE_TTL = 300


# ── 결과 데이터클래스 ──────────────────────────────────────────────────────────

@dataclass
class LeverageDecision:
    """
    decide() 의 최종 반환값.

    Attributes
    ----------
    final_leverage   : 최종 결정 레버리지 (거래소에 실제 설정될 값)
    base_leverage    : signal_scorer 기준값
    market_mult      : 시장 환경 승수 (ADX + ATR 복합)
    risk_mult        : 리스크 상태 승수 (손실률 + 연속 손절)
    adjusted         : base × market_mult × risk_mult (내림 전)
    symbol_max       : 해당 심볼 상한
    clamp_applied    : 상한/하한 클램핑이 적용됐는지 여부
    reasons          : 각 보정 항목 설명 리스트
    set_on_exchange  : 거래소 설정 성공 여부
    """
    final_leverage:  int   = 1
    base_leverage:   int   = 1
    market_mult:     float = 1.0
    risk_mult:       float = 1.0
    adjusted:        float = 1.0
    symbol_max:      int   = 10
    clamp_applied:   bool  = False
    reasons:         list  = None
    set_on_exchange: bool  = False

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


# ── 메인 클래스 ────────────────────────────────────────────────────────────────

class LeverageManager:
    """
    시장 상황과 리스크 상태를 반영해 최종 레버리지를 결정하고
    거래소에 설정하는 클래스.

    사용 예시:
        lm = LeverageManager(api_key=..., api_secret=...)

        decision = lm.decide(
            symbol  = "BTC/USDT:USDT",
            score   = signal_scorer_result,
            cb_status = circuit_breaker.check(),
        )
        print(f"최종 레버리지: {decision.final_leverage}x")
        # 거래소 설정은 decide() 내부에서 자동 처리
    """

    def __init__(
        self,
        api_key:    str = "",
        api_secret: str = "",
        testnet:    bool = False,
        demo:       bool = False,
    ):
        exchange_cfg = {
            "apiKey":  api_key,
            "secret":  api_secret,
            "options": {
                "defaultType":     "future",
                "fetchCurrencies": False,     # Spot SAPI 호출 차단
                "adjustForTimeDifference": True,
            },
            "enableRateLimit": True,
        }
        if demo or testnet:
            exchange_cfg["urls"] = DEMO_TRADING_URLS
        self.exchange = ccxt.binanceusdm(exchange_cfg)

        # {symbol: (leverage, set_at)} — 캐시
        self._cache: dict[str, tuple[int, float]] = {}

    # ── 퍼블릭 메서드 ──────────────────────────────────────────────────────────

    def decide(
        self,
        symbol:    str,
        score:     ScoreResult,
        cb_status: CircuitStatus,
    ) -> LeverageDecision:
        """
        최종 레버리지 결정 → 거래소 설정 → LeverageDecision 반환.

        Parameters
        ----------
        symbol    : 예) "BTC/USDT:USDT"
        score     : signal_scorer.evaluate() 반환값
        cb_status : circuit_breaker.check() 반환값

        Returns
        -------
        LeverageDecision
        """
        reasons: list[str] = []

        # ── Step 1. 기준 레버리지 ─────────────────────────────────────────────
        base = score.leverage
        reasons.append(f"기준={base}x (confidence={score.confidence})")

        # ── Step 2. 시장 환경 승수 (ADX + ATR) ───────────────────────────────
        market_mult = _calc_market_mult(score, reasons)

        # ── Step 3. 리스크 상태 승수 (손실률 + 연속 손절) ─────────────────────
        risk_mult = _calc_risk_mult(cb_status, reasons)

        # ── Step 4. 승수 적용 후 반올림 ─────────────────────────────────────
        adjusted = base * market_mult * risk_mult
        floored  = max(MIN_LEVERAGE, round(adjusted))   # 반올림 (1.8x → 2x)

        # ── Step 5. 심볼 상한 클램핑 ─────────────────────────────────────────
        sym_max      = _symbol_max(symbol)
        final        = min(floored, sym_max)
        clamp_applied = final != floored

        if clamp_applied:
            reasons.append(f"심볼 상한 클램핑 {floored}x → {final}x (max={sym_max}x)")

        decision = LeverageDecision(
            final_leverage = final,
            base_leverage  = base,
            market_mult    = round(market_mult, 3),
            risk_mult      = round(risk_mult,   3),
            adjusted       = round(adjusted,    3),
            symbol_max     = sym_max,
            clamp_applied  = clamp_applied,
            reasons        = reasons,
        )

        # ── Step 6. 거래소에 실제 설정 ────────────────────────────────────────
        decision.set_on_exchange = self._set_exchange_leverage(symbol, final)

        logger.info(
            f"[{symbol}] 레버리지 결정  "
            f"base={base}x  "
            f"×market={market_mult:.2f}  "
            f"×risk={risk_mult:.2f}  "
            f"→ adjusted={adjusted:.2f}  "
            f"→ final={final}x  "
            f"set={decision.set_on_exchange}"
        )
        return decision

    def force_set(self, symbol: str, leverage: int) -> bool:
        """
        레버리지 강제 설정 (수동 오버라이드 / 테스트용).

        Returns
        -------
        bool : 설정 성공 여부
        """
        leverage = max(MIN_LEVERAGE, min(leverage, MAX_LEVERAGE))
        return self._set_exchange_leverage(symbol, leverage, force=True)

    def get_cached(self, symbol: str) -> Optional[int]:
        """캐시에 저장된 현재 레버리지 반환. 없으면 None."""
        entry = self._cache.get(symbol)
        if entry is None:
            return None
        lev, set_at = entry
        if time.time() - set_at > CACHE_TTL:
            return None
        return lev

    # ── 내부: 거래소 레버리지 설정 ────────────────────────────────────────────

    def _set_exchange_leverage(
        self,
        symbol:   str,
        leverage: int,
        force:    bool = False,
    ) -> bool:
        """
        캐시 확인 후 변경이 필요한 경우에만 거래소 API 호출.

        Parameters
        ----------
        force : True 이면 캐시 무시하고 무조건 설정

        Returns
        -------
        bool : 설정 성공 여부
        """
        # 캐시 히트 — 같은 값이면 API 호출 생략
        if not force:
            cached = self.get_cached(symbol)
            if cached == leverage:
                logger.debug(f"[{symbol}] 레버리지 캐시 히트 ({leverage}x) — API 생략")
                return True

        try:
            # ccxt set_leverage() 사용 (fapiPrivate_post_leverage는 구버전)
            self.exchange.set_leverage(leverage, symbol)
            self._cache[symbol] = (leverage, time.time())
            logger.info(f"[{symbol}] 거래소 레버리지 설정: {leverage}x")
            return True

        except ccxt.ExchangeError as e:
            logger.error(f"[{symbol}] 레버리지 설정 실패 (거래소 오류): {e}")
            return False
        except Exception as e:
            logger.error(f"[{symbol}] 레버리지 설정 실패 (예외): {e}")
            return False

    @staticmethod
    def _to_market_id(symbol: str) -> str:
        """
        ccxt 심볼 → Binance Futures 마켓 ID 변환.
        예) "BTC/USDT:USDT" → "BTCUSDT"
        """
        return symbol.split("/")[0] + "USDT"


# ── 순수 함수: 승수 계산 ───────────────────────────────────────────────────────

def _calc_market_mult(score: ScoreResult, reasons: list) -> float:
    """
    ADX + ATR 기반 시장 환경 승수 계산 (0.5 ~ 1.2).

    ADX:
      ≥ 40 (강한 추세)          → +0.2 보정
      25~28 (추세 불안정 경계)  → -0.2 보정
      < 25 (횡보)               → -0.5 보정

    ATR 비율:
      > 4.0% (고변동성)         → -0.3 보정
      < 1.5% (저변동성)         → +0.1 보정

    최종: 1.0 + adx_delta + atr_delta, clamp(0.5, 1.2)
    """
    if score.signal_result is None or score.signal_result.trend is None:
        reasons.append("시장 승수: 1.0 (추세 데이터 없음)")
        return 1.0

    trend     = score.signal_result.trend
    adx       = trend.adx
    atr_ratio = trend.atr_ratio

    adx_delta = 0.0
    atr_delta = 0.0

    # ADX 보정
    if adx >= ADX_BOOST_THRESH:
        adx_delta = +0.2
        reasons.append(f"ADX={adx:.1f}≥{ADX_BOOST_THRESH} (강한추세) → ×+0.2")
    elif adx < ADX_SIDEWAYS:
        adx_delta = -0.5
        reasons.append(f"ADX={adx:.1f}<{ADX_SIDEWAYS} (횡보) → ×-0.5")
    elif adx < ADX_REDUCE_THRESH:
        adx_delta = -0.2
        reasons.append(f"ADX={adx:.1f}<{ADX_REDUCE_THRESH} (추세 불안정) → ×-0.2")
    else:
        reasons.append(f"ADX={adx:.1f} (보통) → 보정 없음")

    # ATR 보정
    if atr_ratio > ATR_HIGH_RATIO:
        atr_delta = -0.3
        reasons.append(
            f"ATR%={atr_ratio*100:.2f}%>{ATR_HIGH_RATIO*100:.1f}% (고변동성) → ×-0.3"
        )
    elif atr_ratio < ATR_LOW_RATIO:
        atr_delta = +0.1
        reasons.append(
            f"ATR%={atr_ratio*100:.2f}%<{ATR_LOW_RATIO*100:.1f}% (저변동성) → ×+0.1"
        )
    else:
        reasons.append(f"ATR%={atr_ratio*100:.2f}% (보통) → 보정 없음")

    mult = max(0.5, min(1.2, 1.0 + adx_delta + atr_delta))
    return mult


def _calc_risk_mult(cb_status: CircuitStatus, reasons: list) -> float:
    """
    일일 손실률 + 연속 손절 기반 리스크 승수 계산 (0.5 ~ 1.0).

    일일 손실률:
      ≥ 2.5% → 0.6  (한도 임박, 매우 보수적)
      ≥ 1.5% → 0.8  (손실 누적, 보수적)

    연속 손절:
      2회   → ×0.7
      1회   → ×0.9

    최종: loss_mult × consec_mult, clamp(0.5, 1.0)
    """
    daily_loss = cb_status.daily_loss_pct
    consec     = cb_status.consec_losses

    # 일일 손실 승수
    if daily_loss >= DAILY_LOSS_REDUCE_2:
        loss_mult = 0.6
        reasons.append(
            f"일손실={daily_loss*100:.2f}%≥{DAILY_LOSS_REDUCE_2*100:.1f}% → ×0.6"
        )
    elif daily_loss >= DAILY_LOSS_REDUCE_1:
        loss_mult = 0.8
        reasons.append(
            f"일손실={daily_loss*100:.2f}%≥{DAILY_LOSS_REDUCE_1*100:.1f}% → ×0.8"
        )
    else:
        loss_mult = 1.0
        reasons.append(f"일손실={daily_loss*100:.2f}% (정상) → 보정 없음")

    # 연속 손절 승수
    if consec >= CONSEC_REDUCE_2:
        consec_mult = 0.7
        reasons.append(f"연속손절={consec}회≥{CONSEC_REDUCE_2}회 → ×0.7")
    elif consec >= CONSEC_REDUCE_1:
        consec_mult = 0.9
        reasons.append(f"연속손절={consec}회≥{CONSEC_REDUCE_1}회 → ×0.9")
    else:
        consec_mult = 1.0
        reasons.append(f"연속손절={consec}회 (정상) → 보정 없음")

    mult = max(0.5, min(1.0, loss_mult * consec_mult))
    return mult


def _symbol_max(symbol: str) -> int:
    """심볼별 최대 레버리지 반환. 미등록 심볼은 DEFAULT 적용."""
    return SYMBOL_MAX_LEVERAGE.get(symbol, SYMBOL_MAX_LEVERAGE["_DEFAULT"])


# ── 유틸 ───────────────────────────────────────────────────────────────────────

def decision_summary(d: LeverageDecision) -> str:
    """LeverageDecision 한 줄 요약 — 로깅/디버깅용."""
    return (
        f"[레버리지] "
        f"base={d.base_leverage}x  "
        f"×market={d.market_mult:.2f}  "
        f"×risk={d.risk_mult:.2f}  "
        f"→ {d.adjusted:.1f}x  "
        f"→ final={d.final_leverage}x"
        f"{'(클램핑)' if d.clamp_applied else ''}  "
        f"거래소설정={d.set_on_exchange}"
    )


# ── 단독 실행 테스트 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import pandas as pd
    from dotenv import load_dotenv
    import signal_engine
    import signal_scorer
    from circuit_breaker import CircuitBreaker, CircuitStatus

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    load_dotenv()

    API_KEY    = os.getenv("BINANCE_API_KEY", "")
    API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    CAPITAL    = float(os.getenv("TOTAL_CAPITAL", "1000"))

    import ccxt as _ccxt
    exchange = _ccxt.binanceusdm({
        "apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True,
    })

    def fetch_df(symbol, tf, limit):
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(
            ohlcv, columns=["timestamp","open","high","low","close","volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df.astype(float)

    lm = LeverageManager(api_key=API_KEY, api_secret=API_SECRET)
    cb = CircuitBreaker(total_capital=CAPITAL)

    # ── 시나리오별 레버리지 결정 테스트 ────────────────────────────────────────
    scenarios = [
        {"label": "정상 상태",             "daily_loss": 0.0,  "consec": 0},
        {"label": "손실 1.5% 누적",        "daily_loss": 15.0, "consec": 0},
        {"label": "연속 손절 2회",          "daily_loss": 0.0,  "consec": 2},
        {"label": "손실 2.5%+연속손절 1회","daily_loss": 25.0, "consec": 1},
    ]

    sym    = "BTC/USDT:USDT"
    df_1h  = fetch_df(sym, "1h",  250)
    df_15m = fetch_df(sym, "15m", 150)
    sig    = signal_engine.check(sym, df_1h, df_15m)
    score  = signal_scorer.evaluate(sig)

    print(f"\n심볼: {sym}")
    print(f"signal_scorer 기준 레버리지: {score.leverage}x  "
          f"(confidence={score.confidence})\n")

    for sc in scenarios:
        # CircuitStatus 수동 조작 (테스트용)
        cb_status = CircuitStatus(
            blocked         = False,
            daily_loss_pct  = sc["daily_loss"] / CAPITAL,
            daily_loss_usdt = sc["daily_loss"],
            consec_losses   = sc["consec"],
        )

        decision = lm.decide(sym, score, cb_status)

        print(f"[{sc['label']}]")
        print(f"  {decision_summary(decision)}")
        print(f"  보정 내역:")
        for r in decision.reasons:
            print(f"    • {r}")
        print()

    # 심볼 상한 테스트
    print("=" * 60)
    print("[ 심볼별 최대 레버리지 ]")
    for sym_test, max_lev in SYMBOL_MAX_LEVERAGE.items():
        if sym_test == "_DEFAULT":
            print(f"  기타 알트코인 : 최대 {max_lev}x")
        else:
            print(f"  {sym_test:<22} : 최대 {max_lev}x")

    cb.stop()