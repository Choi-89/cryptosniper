"""
risk_manager.py
---------------
포지션 크기 계산, TP/SL 설정, 트레일링 스탑 관리를
책임지는 리스크 관리 모듈.

역할:
  1. 1회 진입 시 최대 손실을 총 자본의 1% 로 제한
  2. ATR 기반 손절가(SL) / 1차 익절(TP1) / 2차 익절(TP2) 계산
  3. TP1 도달 후 손절을 진입가로 이동 (본전 사수)
  4. TP2 이후 트레일링 스탑 활성화
  5. 최대 동시 포지션 수 제한 (3개)
  6. 동일 심볼 중복 진입 차단

포지션 크기 공식:
  risk_amount   = total_capital × RISK_PER_TRADE_PCT
  sl_distance   = ATR × ATR_SL_MULTIPLIER   (가격 단위)
  position_size = risk_amount / (sl_distance / leverage)

의존 모듈:
  strategy/signal_engine.py  (SignalResult)
  strategy/signal_scorer.py  (ScoreResult)
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from strategy.signal_engine import SignalResult
from strategy.signal_scorer import ScoreResult

# ── 로거 ───────────────────────────────────────────────────────────────────────
logger = logging.getLogger("risk_manager")


# ── 파라미터 상수 ──────────────────────────────────────────────────────────────
RISK_PER_TRADE_PCT   = 0.01    # 1회 트레이드 최대 손실 비율 (총 자본 대비)
ATR_SL_MULTIPLIER    = 1.5     # SL 거리 = ATR × 1.5
TP1_RATIO            = 2.0     # TP1 = SL 거리 × 2.0  (Risk:Reward = 1:2)
TP2_RATIO            = 4.0     # TP2 = SL 거리 × 4.0  (Risk:Reward = 1:4)
TP1_CLOSE_PCT        = 0.5     # TP1 도달 시 포지션 50% 청산
TRAILING_TRIGGER_PCT = 0.01    # TP2 이후 고점 대비 -1% 이탈 시 청산
MAX_POSITIONS        = 3       # 최대 동시 포지션 수
MIN_POSITION_USDT    = 10.0    # 최소 포지션 크기 (USDT)


# ── 포지션 상태 Enum ───────────────────────────────────────────────────────────

class PositionState(str, Enum):
    OPEN       = "OPEN"        # 진입 완료, TP1 미도달
    TP1_HIT    = "TP1_HIT"     # TP1 도달, 50% 청산, SL → 진입가
    TP2_HIT    = "TP2_HIT"     # TP2 도달, 트레일링 스탑 활성
    CLOSED     = "CLOSED"      # 청산 완료


# ── 포지션 데이터클래스 ────────────────────────────────────────────────────────

@dataclass
class Position:
    """
    단일 포지션 정보를 담는 데이터클래스.

    Attributes
    ----------
    symbol          : 심볼  예) "BTC/USDT:USDT"
    direction       : "LONG" or "SHORT"
    entry_price     : 진입가
    position_size   : 포지션 크기 (코인 단위, 레버리지 적용 전)
    leverage        : 적용 레버리지
    sl_price        : 현재 손절가 (TP1 도달 시 진입가로 이동)
    tp1_price       : 1차 익절가
    tp2_price       : 2차 익절가
    state           : PositionState
    trailing_high   : 트레일링 스탑 기준 고점 (LONG) 또는 저점 (SHORT)
    remaining_ratio : 잔여 포지션 비율 (1.0 → 0.5 → 0.0)
    opened_at       : 진입 시각 (Unix 초)
    confidence      : 진입 당시 신뢰도 점수
    atr_at_entry    : 진입 당시 ATR 값 (재계산 없이 재활용)
    risk_amount     : 이 포지션의 허용 손실 금액 (USDT)
    """
    symbol:         str
    direction:      str
    entry_price:    float
    position_size:  float
    leverage:       int
    sl_price:       float
    tp1_price:      float
    tp2_price:      float
    state:          PositionState  = PositionState.OPEN
    trailing_high:  float          = 0.0
    remaining_ratio: float         = 1.0
    opened_at:      float          = field(default_factory=time.time)
    confidence:     int            = 0
    atr_at_entry:   float          = 0.0
    risk_amount:    float          = 0.0


@dataclass
class PositionPlan:
    """
    calculate_plan() 의 반환값. 실제 주문 전 계획 정보.
    order_executor.py 가 이 구조를 받아 실제 주문을 넣는다.
    """
    symbol:        str
    direction:     str
    entry_price:   float
    position_size: float      # 코인 단위
    notional:      float      # 명목 포지션 크기 (USDT) = size × entry_price
    leverage:      int
    sl_price:      float
    tp1_price:     float
    tp2_price:     float
    risk_amount:   float      # 허용 손실 USDT
    atr:           float
    confidence:    int
    can_open:      bool       # False 면 진입 불가 (사유: reject_reason)
    reject_reason: str        = ""


# ── 메인 클래스 ────────────────────────────────────────────────────────────────

class RiskManager:
    """
    포지션 크기 계산, 진입 가능 여부 판단, 진행 중 포지션 관리를 담당.

    사용 예시:
        rm = RiskManager(total_capital=1000.0)

        plan = rm.calculate_plan(sig_result, score_result)
        if plan.can_open:
            pos = rm.open_position(plan)
            # order_executor.execute(plan) 으로 실제 주문

        # 매 캔들마다 포지션 업데이트
        actions = rm.update_positions(current_prices)
        for action in actions:
            order_executor.handle(action)
    """

    def __init__(
        self,
        total_capital: float,
        risk_per_trade_pct: float = 0.01,
        max_positions: int = 3,
        min_position_usdt: float = 10.0,
        atr_sl_multiplier: float = 1.5,
        tp1_ratio: float = 2.0,
        tp2_ratio: float = 4.0,
        tp1_close_pct: float = 0.5,
        trailing_trigger_pct: float = 0.01,
    ):
        """
        Parameters
        ----------
        total_capital : 운용 총 자본 (USDT). 실시간 잔고로 주기적 갱신 필요.
        """
        self.total_capital = total_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_positions = max_positions
        self.min_position_usdt = min_position_usdt
        self.atr_sl_multiplier = atr_sl_multiplier
        self.tp1_ratio = tp1_ratio
        self.tp2_ratio = tp2_ratio
        self.tp1_close_pct = tp1_close_pct
        self.trailing_trigger_pct = trailing_trigger_pct

        # 활성 포지션 저장소 {symbol: Position}
        self._positions: dict[str, Position] = {}
        self._lock = threading.Lock()

    # ── 퍼블릭 메서드 ──────────────────────────────────────────────────────────

    def update_capital(self, new_capital: float) -> None:
        """총 자본 갱신 (잔고 조회 후 주기적으로 호출)."""
        self.total_capital = new_capital
        logger.debug(f"총 자본 갱신: {new_capital:.2f} USDT")

    def calculate_plan(
        self,
        sig:   SignalResult,
        score: ScoreResult,
    ) -> PositionPlan:
        """
        SignalResult + ScoreResult 를 받아 PositionPlan 계산.

        진입 가능 여부 판단 → 포지션 크기 → SL/TP 계산.
        실제 주문은 하지 않는다 (order_executor 의 역할).

        Parameters
        ----------
        sig   : signal_engine.check() 반환값
        score : signal_scorer.evaluate() 반환값

        Returns
        -------
        PositionPlan (.can_open 확인 필수)
        """
        symbol    = sig.signal   # 임시, 아래서 sig 구조로부터 추출
        # sig.signal 은 방향이므로 symbol 은 호출자가 별도 전달 필요
        # 여기서는 score.signal_result 를 통해 접근
        direction   = sig.signal
        entry_price = sig.entry_price
        atr         = sig.trend.atr if sig.trend else 0.0
        leverage    = score.leverage
        confidence  = score.confidence

        # ── 진입 가능 여부 체크 ────────────────────────────────────────────────
        # symbol 은 signal_result 에 직접 없으므로 score 로부터 추출
        # (실제 연동 시 check() 에 symbol 파라미터를 추가하거나
        #  PositionPlan 생성 시 symbol 을 별도로 전달한다.)
        # 여기서는 caller 가 symbol 을 전달하는 방식으로 설계
        # → calculate_plan_for(symbol, sig, score) 오버로드 메서드 제공

        reject = self._check_entry_conditions(
            symbol="(unknown)",
            direction=direction,
            score=score,
            atr=atr,
            entry_price=entry_price,
        )
        if reject:
            return PositionPlan(
                symbol="(unknown)", direction=direction,
                entry_price=entry_price, position_size=0, notional=0,
                leverage=leverage, sl_price=0, tp1_price=0, tp2_price=0,
                risk_amount=0, atr=atr, confidence=confidence,
                can_open=False, reject_reason=reject,
            )

        return self._build_plan(
            symbol="(unknown)",
            direction=direction,
            entry_price=entry_price,
            atr=atr,
            leverage=leverage,
            confidence=confidence,
        )

    def calculate_plan_for(
        self,
        symbol:     str,
        sig:        SignalResult,
        score:      ScoreResult,
    ) -> PositionPlan:
        """
        symbol 을 명시적으로 전달하는 calculate_plan 오버로드.
        main.py / order_executor 에서 이 메서드를 사용한다.

        Parameters
        ----------
        symbol : 예) "BTC/USDT:USDT"
        sig    : signal_engine.check() 반환값
        score  : signal_scorer.evaluate() 반환값
        """
        direction   = sig.signal
        entry_price = sig.entry_price
        atr         = sig.trend.atr if sig.trend else 0.0
        leverage    = score.leverage
        confidence  = score.confidence

        reject = self._check_entry_conditions(
            symbol=symbol,
            direction=direction,
            score=score,
            atr=atr,
            entry_price=entry_price,
        )
        if reject:
            return PositionPlan(
                symbol=symbol, direction=direction,
                entry_price=entry_price, position_size=0, notional=0,
                leverage=leverage, sl_price=0, tp1_price=0, tp2_price=0,
                risk_amount=0, atr=atr, confidence=confidence,
                can_open=False, reject_reason=reject,
            )

        return self._build_plan(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            atr=atr,
            leverage=leverage,
            confidence=confidence,
        )

    def open_position(self, plan: PositionPlan) -> Optional[Position]:
        """
        PositionPlan 을 Position 으로 등록 (메모리 저장).
        실제 거래소 주문은 order_executor 가 담당.

        Returns
        -------
        Position or None (등록 실패 시)
        """
        if not plan.can_open:
            logger.warning(f"[{plan.symbol}] open_position 실패: {plan.reject_reason}")
            return None

        pos = Position(
            symbol        = plan.symbol,
            direction     = plan.direction,
            entry_price   = plan.entry_price,
            position_size = plan.position_size,
            leverage      = plan.leverage,
            sl_price      = plan.sl_price,
            tp1_price     = plan.tp1_price,
            tp2_price     = plan.tp2_price,
            trailing_high = plan.entry_price,
            confidence    = plan.confidence,
            atr_at_entry  = plan.atr,
            risk_amount   = plan.risk_amount,
        )

        with self._lock:
            self._positions[plan.symbol] = pos

        logger.info(
            f"[{plan.symbol}] 포지션 등록  "
            f"dir={plan.direction}  "
            f"entry={plan.entry_price:.4f}  "
            f"size={plan.position_size:.6f}  "
            f"lev={plan.leverage}x  "
            f"SL={plan.sl_price:.4f}  "
            f"TP1={plan.tp1_price:.4f}  "
            f"TP2={plan.tp2_price:.4f}"
        )
        return pos

    def update_positions(
        self,
        current_prices: dict[str, float],
    ) -> list[dict]:
        """
        모든 활성 포지션에 대해 현재가 기준으로 상태를 업데이트하고
        필요한 액션(청산 / SL 이동 / 트레일링 업데이트)을 반환.

        매 캔들 닫힘(또는 주기적으로) 호출.

        Parameters
        ----------
        current_prices : {symbol: 현재가(float)}

        Returns
        -------
        list[dict] : 실행해야 할 액션 목록
            {
                "action"  : "CLOSE_PARTIAL" / "CLOSE_FULL" / "MOVE_SL",
                "symbol"  : str,
                "reason"  : str,
                "price"   : float,
                "ratio"   : float,   # 청산 비율 (0.5 = 50%)
                "new_sl"  : float,   # MOVE_SL 시 새 SL 가격
            }
        """
        actions = []

        with self._lock:
            for symbol, pos in list(self._positions.items()):
                price = current_prices.get(symbol)
                if price is None:
                    continue

                pos_actions = self._evaluate_position(pos, price)
                actions.extend(pos_actions)

                # 전량 청산된 포지션 제거
                if pos.state == PositionState.CLOSED:
                    del self._positions[symbol]
                    logger.info(f"[{symbol}] 포지션 제거 완료")

        return actions

    def close_position(self, symbol: str, reason: str = "수동 청산") -> bool:
        """특정 포지션 강제 청산 (Circuit Breaker 등에서 호출)."""
        with self._lock:
            pos = self._positions.pop(symbol, None)
        if pos:
            logger.info(f"[{symbol}] 강제 청산: {reason}")
            return True
        return False

    def get_position(self, symbol: str) -> Optional[Position]:
        """특정 심볼 포지션 조회."""
        with self._lock:
            return self._positions.get(symbol)

    def get_all_positions(self) -> list[Position]:
        """전체 활성 포지션 목록 반환."""
        with self._lock:
            return list(self._positions.values())

    def position_count(self) -> int:
        """현재 활성 포지션 수."""
        with self._lock:
            return len(self._positions)

    # ── 내부: 진입 조건 체크 ──────────────────────────────────────────────────

    def _check_entry_conditions(
        self,
        symbol:      str,
        direction:   str,
        score:       ScoreResult,
        atr:         float,
        entry_price: float,
    ) -> str:
        """
        진입 전 최종 게이트 체크.

        Returns
        -------
        str : 거절 사유. 빈 문자열이면 진입 허가.
        """
        # scorer 가 이미 거절한 경우
        if not score.can_enter:
            return f"신뢰도 부족: confidence={score.confidence}"

        # 최대 포지션 수 초과
        with self._lock:
            if len(self._positions) >= self.max_positions:
                return (
                    f"최대 포지션 수 초과: "
                    f"{len(self._positions)}/{self.max_positions}"
                )

            # 동일 심볼 중복 진입 차단
            if symbol in self._positions:
                return f"이미 열린 포지션 존재: {symbol}"

        # ATR 유효성 체크
        if atr <= 0:
            return "ATR 값 이상 (지표 계산 오류)"

        # 최소 포지션 크기 체크
        risk_amount  = self.total_capital * self.risk_per_trade_pct
        sl_distance  = atr * self.atr_sl_multiplier
        size         = _calc_position_size(
            risk_amount, sl_distance, score.leverage, entry_price
        )
        notional     = size * entry_price

        if notional < self.min_position_usdt:
            return (
                f"포지션 크기 너무 작음: {notional:.2f} USDT "
                f"(최소 {self.min_position_usdt} USDT)"
            )

        return ""  # 모든 조건 통과

    # ── 내부: PositionPlan 생성 ───────────────────────────────────────────────

    def _build_plan(
        self,
        symbol:      str,
        direction:   str,
        entry_price: float,
        atr:         float,
        leverage:    int,
        confidence:  int,
    ) -> PositionPlan:
        """SL / TP / 포지션 크기 계산 후 PositionPlan 반환."""

        risk_amount = self.total_capital * self.risk_per_trade_pct
        sl_distance = atr * self.atr_sl_multiplier    # 가격 단위 SL 거리
        tp1_distance = sl_distance * self.tp1_ratio   # TP1 거리
        tp2_distance = sl_distance * self.tp2_ratio   # TP2 거리

        # 방향에 따라 SL/TP 방향 결정
        if direction == "LONG":
            sl_price  = entry_price - sl_distance
            tp1_price = entry_price + tp1_distance
            tp2_price = entry_price + tp2_distance
        else:  # SHORT
            sl_price  = entry_price + sl_distance
            tp1_price = entry_price - tp1_distance
            tp2_price = entry_price - tp2_distance

        # 포지션 크기 계산
        size     = _calc_position_size(risk_amount, sl_distance, leverage, entry_price)
        notional = size * entry_price

        logger.debug(
            f"[{symbol}] 플랜 계산  "
            f"risk={risk_amount:.2f}U  "
            f"ATR={atr:.4f}  "
            f"SL거리={sl_distance:.4f}  "
            f"size={size:.6f}  "
            f"notional={notional:.2f}U"
        )

        return PositionPlan(
            symbol        = symbol,
            direction     = direction,
            entry_price   = entry_price,
            position_size = round(size, 6),
            notional      = round(notional, 2),
            leverage      = leverage,
            sl_price      = round(sl_price,  4),
            tp1_price     = round(tp1_price, 4),
            tp2_price     = round(tp2_price, 4),
            risk_amount   = round(risk_amount, 2),
            atr           = atr,
            confidence    = confidence,
            can_open      = True,
        )

    # ── 내부: 포지션 상태 업데이트 ────────────────────────────────────────────

    def _evaluate_position(
        self,
        pos:   Position,
        price: float,
    ) -> list[dict]:
        """
        단일 포지션에 대해 현재가로 상태를 평가하고 액션 목록 반환.

        상태 전이:
          OPEN
            → SL 도달      : 전량 손절 (CLOSE_FULL)
            → TP1 도달     : 50% 청산 + SL → 진입가 이동 (CLOSE_PARTIAL + MOVE_SL)
            → TP2 도달     : 트레일링 스탑 활성 (state → TP2_HIT)
          TP1_HIT
            → SL(=진입가) 도달: 나머지 50% 청산 (CLOSE_FULL)
            → TP2 도달       : 트레일링 스탑 활성
          TP2_HIT
            → 트레일링 발동  : 전량 청산 (CLOSE_FULL)
            → 고점 갱신      : trailing_high 업데이트
        """
        actions = []
        is_long = pos.direction == "LONG"

        # ── OPEN 상태 ─────────────────────────────────────────────────────────
        if pos.state == PositionState.OPEN:

            # SL 도달
            if (is_long  and price <= pos.sl_price) or \
               (not is_long and price >= pos.sl_price):
                pos.state = PositionState.CLOSED
                logger.info(
                    f"[{pos.symbol}] 손절  "
                    f"SL={pos.sl_price:.4f}  price={price:.4f}"
                )
                return actions

            # TP2 먼저 체크 (TP1/TP2 동시 돌파 방지)
            if (is_long  and price >= pos.tp2_price) or \
               (not is_long and price <= pos.tp2_price):
                # TP1 + TP2 동시 도달 → 50% 청산 + 트레일링 활성
                pos.remaining_ratio = 1.0 - self.tp1_close_pct
                pos.sl_price        = pos.entry_price
                pos.trailing_high   = price
                pos.state           = PositionState.TP2_HIT
                logger.info(
                    f"[{pos.symbol}] TP2 직행  "
                    f"거래소 브래킷 처리 가정, 트레일링 활성"
                )
                return actions

            # TP1 도달
            if (is_long  and price >= pos.tp1_price) or \
               (not is_long and price <= pos.tp1_price):
                pos.remaining_ratio = 1.0 - self.tp1_close_pct
                pos.sl_price        = pos.entry_price
                pos.state           = PositionState.TP1_HIT
                logger.info(
                    f"[{pos.symbol}] TP1 도달  "
                    f"거래소 브래킷 처리 가정, 상태만 동기화"
                )

        # ── TP1_HIT 상태 ──────────────────────────────────────────────────────
        elif pos.state == PositionState.TP1_HIT:

            # SL (= 진입가) 도달 → 나머지 50% 본전 청산
            if (is_long  and price <= pos.sl_price) or \
               (not is_long and price >= pos.sl_price):
                pos.state = PositionState.CLOSED
                logger.info(f"[{pos.symbol}] 본전 청산  price={price:.4f}")
                return actions

            # TP2 도달 → 트레일링 활성
            if (is_long  and price >= pos.tp2_price) or \
               (not is_long and price <= pos.tp2_price):
                pos.trailing_high = price
                pos.state         = PositionState.TP2_HIT
                logger.info(
                    f"[{pos.symbol}] TP2 도달  트레일링 스탑 활성  "
                    f"trailing_high={price:.4f}"
                )

        # ── TP2_HIT 상태 — 트레일링 스탑 ─────────────────────────────────────
        elif pos.state == PositionState.TP2_HIT:

            if is_long:
                # 고점 갱신
                if price > pos.trailing_high:
                    pos.trailing_high = price
                # 고점 대비 -TRAILING_TRIGGER_PCT 이탈
                trail_stop = pos.trailing_high * (1 - self.trailing_trigger_pct)
                if price <= trail_stop:
                    actions.append(_make_action(
                        "CLOSE_FULL", pos.symbol, price,
                        pos.remaining_ratio,
                        f"트레일링 발동 (고점={pos.trailing_high:.4f})"
                    ))
                    pos.state = PositionState.CLOSED
                    logger.info(
                        f"[{pos.symbol}] 트레일링 청산  "
                        f"high={pos.trailing_high:.4f}  price={price:.4f}"
                    )
            else:
                # 저점 갱신
                if price < pos.trailing_high:
                    pos.trailing_high = price
                # 저점 대비 +TRAILING_TRIGGER_PCT 상승
                trail_stop = pos.trailing_high * (1 + self.trailing_trigger_pct)
                if price >= trail_stop:
                    actions.append(_make_action(
                        "CLOSE_FULL", pos.symbol, price,
                        pos.remaining_ratio,
                        f"트레일링 발동 (저점={pos.trailing_high:.4f})"
                    ))
                    pos.state = PositionState.CLOSED
                    logger.info(
                        f"[{pos.symbol}] 트레일링 청산  "
                        f"low={pos.trailing_high:.4f}  price={price:.4f}"
                    )

        return actions


# ── 순수 함수: 포지션 크기 계산 ────────────────────────────────────────────────

def _calc_position_size(
    risk_amount:  float,
    sl_distance:  float,
    leverage:     int,
    entry_price:  float,
) -> float:
    """
    허용 손실금액으로 포지션 크기(코인 단위) 계산.

    공식:
      sl_pct        = sl_distance / entry_price       (가격 비율)
      actual_sl_pct = sl_pct / leverage                (레버리지 적용)
      size          = risk_amount / (entry_price × actual_sl_pct)
                    = risk_amount × leverage / (entry_price × sl_pct)

    예시:
      자본=1000, risk=1%, sl_distance=ATR×1.5=100,
      entry=50000, leverage=5x
      → sl_pct = 100/50000 = 0.002
      → size   = 10 × 5 / (50000 × 0.002) = 50/100 = 0.5 BTC
    """
    if entry_price <= 0 or sl_distance <= 0:
        return 0.0

    sl_pct = sl_distance / entry_price
    if sl_pct <= 0:
        return 0.0

    size = (risk_amount * leverage) / (entry_price * sl_pct)
    return max(0.0, size)


def _make_action(
    action: str,
    symbol: str,
    price:  float,
    ratio:  float,
    reason: str,
    new_sl: float = 0.0,
) -> dict:
    """액션 딕셔너리 생성 헬퍼."""
    return {
        "action": action,
        "symbol": symbol,
        "price":  price,
        "ratio":  ratio,
        "reason": reason,
        "new_sl": new_sl,
    }


# ── 유틸 ───────────────────────────────────────────────────────────────────────

def plan_summary(plan: PositionPlan) -> str:
    """PositionPlan 한 줄 요약."""
    if not plan.can_open:
        return f"[플랜] 진입 불가 — {plan.reject_reason}"
    rr1 = (plan.tp1_price - plan.entry_price) / (plan.entry_price - plan.sl_price) \
          if plan.direction == "LONG" else \
          (plan.entry_price - plan.tp1_price) / (plan.sl_price - plan.entry_price)
    return (
        f"[플랜] {plan.direction}  "
        f"entry={plan.entry_price:.4f}  "
        f"SL={plan.sl_price:.4f}  "
        f"TP1={plan.tp1_price:.4f}  "
        f"TP2={plan.tp2_price:.4f}  "
        f"size={plan.position_size:.6f}  "
        f"notional={plan.notional:.2f}U  "
        f"lev={plan.leverage}x  "
        f"RR1=1:{rr1:.1f}  "
        f"risk={plan.risk_amount:.2f}U"
    )


# ── 단독 실행 테스트 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import ccxt
    import pandas as pd
    from dotenv import load_dotenv
    import signal_engine
    import signal_scorer

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    load_dotenv()

    API_KEY    = os.getenv("BINANCE_API_KEY", "")
    API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    CAPITAL    = float(os.getenv("TOTAL_CAPITAL", "1000"))

    exchange = ccxt.binanceusdm({
        "apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True,
    })

    def fetch_df(symbol, tf, limit):
        ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df.astype(float)

    rm = RiskManager(total_capital=CAPITAL)

    TEST_SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT"]

    for sym in TEST_SYMBOLS:
        print(f"\n{'='*72}")
        print(f"  {sym}  (자본={CAPITAL:.0f} USDT)")
        print('='*72)

        df_1h  = fetch_df(sym, "1h",  250)
        df_15m = fetch_df(sym, "15m", 150)

        sig   = signal_engine.check(sym, df_1h, df_15m)
        score = signal_scorer.evaluate(sig)
        plan  = rm.calculate_plan_for(sym, sig, score)

        print(plan_summary(plan))

        if plan.can_open:
            print()
            print(f"  진입가         : {plan.entry_price:.4f}")
            print(f"  손절가 (SL)    : {plan.sl_price:.4f}  "
                  f"(ATR×{rm.atr_sl_multiplier} = {plan.atr:.4f}×{rm.atr_sl_multiplier})")
            print(f"  1차 익절 (TP1) : {plan.tp1_price:.4f}  "
                  f"(50% 청산, SL→진입가)")
            print(f"  2차 익절 (TP2) : {plan.tp2_price:.4f}  "
                  f"(트레일링 스탑 활성)")
            print(f"  포지션 크기    : {plan.position_size:.6f} 코인")
            print(f"  명목 크기      : {plan.notional:.2f} USDT")
            print(f"  레버리지       : {plan.leverage}x")
            print(f"  허용 손실      : {plan.risk_amount:.2f} USDT "
                  f"(자본의 {rm.risk_per_trade_pct*100:.0f}%)")
        else:
            print(f"  거절 사유: {plan.reject_reason}")

        time.sleep(0.3)

    # ── 포지션 관리 시뮬레이션 ────────────────────────────────────────────────
    print("\n\n[ 포지션 관리 시뮬레이션 ]")
    dummy_plan = PositionPlan(
        symbol="BTC/USDT:USDT", direction="LONG",
        entry_price=50000.0, position_size=0.01, notional=500.0,
        leverage=5, sl_price=48500.0, tp1_price=53000.0, tp2_price=56000.0,
        risk_amount=10.0, atr=1000.0, confidence=75, can_open=True,
    )
    rm.open_position(dummy_plan)

    price_sequence = [50500, 51000, 53500, 55000, 57000, 56500]
    labels = ["초기", "상승", "TP1 돌파", "TP2 돌파", "고점 갱신", "트레일링 발동"]

    for price, label in zip(price_sequence, labels):
        actions = rm.update_positions({"BTC/USDT:USDT": float(price)})
        pos = rm.get_position("BTC/USDT:USDT")
        state = pos.state if pos else "CLOSED"
        print(
            f"  price={price:>6}  [{label:12s}]  "
            f"state={state:<12}  actions={[a['action']+':'+a['reason'] for a in actions]}"
        )