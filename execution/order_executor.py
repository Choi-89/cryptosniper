"""
order_executor.py
-----------------
PositionPlan 을 받아 Binance Futures 에 실제 주문을 실행하고,
risk_manager 의 액션(CLOSE_PARTIAL / CLOSE_FULL / MOVE_SL)을 처리하는 모듈.

역할:
  1. 진입 주문  : 시장가 진입 + SL/TP 동시 등록
  2. 청산 주문  : 부분 청산(CLOSE_PARTIAL) / 전량 청산(CLOSE_FULL)
  3. SL 이동    : MOVE_SL 액션 수신 시 기존 SL 주문 취소 + 신규 등록
  4. 주문 실패 재시도 : 최대 3회, 지수 백오프
  5. 실제 체결가 / 수수료 반영 후 db_logger 에 기록 전달
  6. 체결 완료 시 circuit_breaker.record_trade() 호출

주문 방식:
  진입   : MARKET  (시장가 — 빠른 체결 우선)
  SL     : STOP_MARKET
  TP1/TP2: TAKE_PROFIT_MARKET

의존 모듈:
  risk/risk_manager.py    (PositionPlan, _make_action dict)
  risk/circuit_breaker.py (CircuitBreaker)
  utils/db_logger.py      (DbLogger)
  utils/telegram_notifier.py (TelegramNotifier)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

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


from risk.risk_manager import PositionPlan, RiskManager
from risk.circuit_breaker import CircuitBreaker

# ── 로거 ───────────────────────────────────────────────────────────────────────
logger = logging.getLogger("order_executor")


# ── 파라미터 상수 ──────────────────────────────────────────────────────────────
MAX_RETRY        = 3      # 주문 실패 시 최대 재시도 횟수
RETRY_BASE_DELAY = 0.5    # 재시도 초기 대기 시간 (초)
TAKER_FEE        = 0.0004 # Binance Futures 테이커 수수료 (0.04%)
MAKER_FEE        = 0.0002 # 메이커 수수료 (0.02%)


# ── 주문 결과 데이터클래스 ─────────────────────────────────────────────────────

@dataclass
class OrderResult:
    """
    단일 주문 실행 결과.

    Attributes
    ----------
    success         : 주문 성공 여부
    order_id        : 거래소 주문 ID
    symbol          : 심볼
    order_type      : 주문 유형 문자열 ("ENTRY" / "SL" / "TP1" / "TP2" / "CLOSE")
    side            : "buy" / "sell"
    amount          : 주문 수량 (코인 단위)
    avg_price       : 평균 체결가 (미체결 시 0.0)
    fee_usdt        : 수수료 (USDT)
    pnl_usdt        : 실현 손익 (청산 주문만 유효, 진입은 0.0)
    raw             : 거래소 원본 응답 dict
    error           : 실패 시 오류 메시지
    attempts        : 실제 시도 횟수
    """
    success:    bool   = False
    order_id:   str    = ""
    symbol:     str    = ""
    order_type: str    = ""
    side:       str    = ""
    amount:     float  = 0.0
    avg_price:  float  = 0.0
    fee_usdt:   float  = 0.0
    pnl_usdt:   float  = 0.0
    raw:        dict   = field(default_factory=dict)
    error:      str    = ""
    attempts:   int    = 0
    close_reason: str  = ""
    position_context: dict = field(default_factory=dict)


@dataclass
class ExecutionReport:
    """
    execute_entry() 의 반환값 — 진입 + SL + TP 주문 묶음 결과.

    Attributes
    ----------
    entry   : 진입 시장가 주문 결과
    sl      : SL STOP_MARKET 주문 결과
    tp1     : TP1 TAKE_PROFIT_MARKET 주문 결과
    tp2     : TP2 TAKE_PROFIT_MARKET 주문 결과
    success : entry + sl 모두 성공했을 때 True
    """
    entry:   Optional[OrderResult] = None
    sl:      Optional[OrderResult] = None
    tp1:     Optional[OrderResult] = None
    tp2:     Optional[OrderResult] = None
    success: bool                  = False


# ── 메인 클래스 ────────────────────────────────────────────────────────────────

class OrderExecutor:
    """
    실제 거래소 주문을 실행하는 클래스.

    사용 예시:
        executor = OrderExecutor(
            api_key=..., api_secret=...,
            risk_manager=rm,
            circuit_breaker=cb,
            on_trade_closed=db_logger.save,
        )

        # 진입
        report = executor.execute_entry(plan)
        if report.success:
            rm.open_position(plan)

        # risk_manager 액션 처리
        actions = rm.update_positions(prices)
        for action in actions:
            executor.handle_action(action, entry_price=plan.entry_price)
    """

    def __init__(
        self,
        api_key:         str,
        api_secret:      str,
        risk_manager:    RiskManager,
        circuit_breaker: CircuitBreaker,
        on_trade_closed: Optional[Callable[[OrderResult], None]] = None,
        on_order_filled: Optional[Callable[[OrderResult], None]] = None,
        testnet:         bool = False,
        demo:            bool = False,
    ):
        """
        Parameters
        ----------
        api_key          : Binance API 키
        api_secret       : Binance API 시크릿
        risk_manager     : RiskManager 인스턴스 (자본 갱신용)
        circuit_breaker  : CircuitBreaker 인스턴스 (거래 결과 기록용)
        on_trade_closed  : 청산 완료 콜백  fn(OrderResult)  → db_logger / telegram
        on_order_filled  : 모든 주문 체결 콜백  fn(OrderResult)
        testnet          : True 이면 테스트넷 사용
        """
        self.exchange = ccxt.binanceusdm({
            "apiKey":  api_key,
            "secret":  api_secret,
            "options": {
                "defaultType":     "future",
                "fetchCurrencies": False,     # Spot SAPI 호출 차단
                "adjustForTimeDifference": True,
            },
            "enableRateLimit": True,
            "urls": DEMO_TRADING_URLS,        # ← 생성 시 바로 주입
        })
        if demo:
            self.exchange.urls.update(DEMO_TRADING_URLS)


        self._rm   = risk_manager
        self._cb   = circuit_breaker
        self._on_trade_closed = on_trade_closed
        self._on_order_filled = on_order_filled

        # 심볼별 활성 주문 ID 캐시 {symbol: {"sl": id, "tp1": id, "tp2": id}}
        self._active_orders: dict[str, dict[str, str]] = {}

    # ── 퍼블릭: 진입 ──────────────────────────────────────────────────────────

    def execute_entry(self, plan: PositionPlan) -> ExecutionReport:
        """
        PositionPlan 기반으로 진입 + SL + TP1 + TP2 주문을 순서대로 실행.

        실행 순서:
          1. 진입 시장가 주문
          2. SL STOP_MARKET 주문
          3. TP1 TAKE_PROFIT_MARKET 주문 (50% 수량)
          4. TP2 TAKE_PROFIT_MARKET 주문 (50% 수량)

        SL 주문 실패 시 즉시 진입 포지션 역방향 청산 후 HOLD 처리.

        Parameters
        ----------
        plan : risk_manager.calculate_plan_for() 반환값

        Returns
        -------
        ExecutionReport
        """
        report = ExecutionReport()

        if not plan.can_open:
            logger.warning(f"[{plan.symbol}] execute_entry 거절: {plan.reject_reason}")
            return report

        is_long   = plan.direction == "LONG"
        entry_side = "buy" if is_long else "sell"
        close_side = "sell" if is_long else "buy"

        # ── 1. 진입 시장가 주문 ────────────────────────────────────────────────
        logger.info(
            f"[{plan.symbol}] 진입 주문  "
            f"{plan.direction}  "
            f"size={plan.position_size:.6f}  "
            f"lev={plan.leverage}x"
        )
        entry_result = self._place_order(
            symbol     = plan.symbol,
            order_type = "ENTRY",
            side       = entry_side,
            amount     = plan.position_size,
            params     = {"reduceOnly": False},
        )
        report.entry = entry_result

        if not entry_result.success:
            logger.error(f"[{plan.symbol}] 진입 주문 실패: {entry_result.error}")
            return report

        # 실제 체결가로 SL/TP 가격 재계산
        actual_entry = entry_result.avg_price or plan.entry_price
        sl, tp1, tp2 = _recalc_levels(plan, actual_entry)

        # ── 2. SL STOP_MARKET 주문 ────────────────────────────────────────────
        sl_result = self._place_order(
            symbol     = plan.symbol,
            order_type = "SL",
            side       = close_side,
            amount     = plan.position_size,
            price      = sl,
            params     = {
                "stopPrice":  sl,
                "reduceOnly": True,
                "type":       "STOP_MARKET",
                "workingType": "MARK_PRICE",
            },
        )
        report.sl = sl_result

        if not sl_result.success:
            # SL 실패 → 즉시 역방향 시장가로 진입 포지션 청산
            logger.error(f"[{plan.symbol}] SL 등록 실패 — 긴급 청산 실행")
            self._emergency_close(plan.symbol, plan.position_size, close_side)
            return report

        # ── 3. TP1 TAKE_PROFIT_MARKET (50%) ──────────────────────────────────
        half_size = round(plan.position_size * 0.5, 6)
        tp1_result = self._place_order(
            symbol     = plan.symbol,
            order_type = "TP1",
            side       = close_side,
            amount     = half_size,
            price      = tp1,
            params     = {
                "stopPrice":  tp1,
                "reduceOnly": True,
                "type":       "TAKE_PROFIT_MARKET",
                "workingType": "MARK_PRICE",
            },
        )
        report.tp1 = tp1_result
        if not tp1_result.success:
            logger.warning(f"[{plan.symbol}] TP1 등록 실패 (진입은 유지): {tp1_result.error}")

        # ── 4. TP2 TAKE_PROFIT_MARKET (나머지 50%) ────────────────────────────
        tp2_result = self._place_order(
            symbol     = plan.symbol,
            order_type = "TP2",
            side       = close_side,
            amount     = half_size,
            price      = tp2,
            params     = {
                "stopPrice":  tp2,
                "reduceOnly": True,
                "type":       "TAKE_PROFIT_MARKET",
                "workingType": "MARK_PRICE",
            },
        )
        report.tp2 = tp2_result
        if not tp2_result.success:
            logger.warning(f"[{plan.symbol}] TP2 등록 실패: {tp2_result.error}")

        # 활성 주문 ID 캐시 저장
        self._active_orders[plan.symbol] = {
            "sl":  sl_result.order_id,
            "tp1": tp1_result.order_id  if tp1_result.success else "",
            "tp2": tp2_result.order_id  if tp2_result.success else "",
        }

        report.success = True
        logger.info(
            f"[{plan.symbol}] 진입 완료  "
            f"체결가={actual_entry:.4f}  "
            f"SL={sl:.4f}  TP1={tp1:.4f}  TP2={tp2:.4f}"
        )
        return report

    # ── 퍼블릭: risk_manager 액션 처리 ────────────────────────────────────────

    def handle_action(
        self,
        action:      dict,
        entry_price: float = 0.0,
        position_context: Optional[dict] = None,
    ) -> Optional[OrderResult]:
        """
        risk_manager.update_positions() 가 반환한 액션 딕셔너리를 처리.

        액션 종류:
          CLOSE_PARTIAL  → 부분 청산 (ratio 만큼)
          CLOSE_FULL     → 전량 청산
          MOVE_SL        → SL 주문 취소 + 신규 SL 등록

        Parameters
        ----------
        action      : risk_manager._make_action() 반환 dict
        entry_price : 진입가 (PnL 계산용)

        Returns
        -------
        OrderResult or None
        """
        act    = action.get("action", "")
        symbol = action.get("symbol", "")
        price  = action.get("price",  0.0)
        ratio  = action.get("ratio",  1.0)
        reason = action.get("reason", "")
        new_sl = action.get("new_sl", 0.0)

        logger.info(f"[{symbol}] 액션 처리: {act}  사유={reason}  price={price:.4f}")

        if act == "CLOSE_FULL":
            return self._close_position(symbol, ratio=1.0, reason=reason,
                                        price=price, entry_price=entry_price,
                                        position_context=position_context)

        elif act == "CLOSE_PARTIAL":
            return self._close_position(symbol, ratio=ratio, reason=reason,
                                        price=price, entry_price=entry_price,
                                        position_context=position_context)

        elif act == "MOVE_SL":
            return self._move_sl(symbol, new_sl=new_sl)

        else:
            logger.warning(f"[{symbol}] 알 수 없는 액션: {act}")
            return None

    # ── 퍼블릭: 전체 포지션 긴급 청산 ────────────────────────────────────────

    def close_all_positions(self, reason: str = "긴급 전량 청산") -> list[OrderResult]:
        """
        모든 활성 포지션을 시장가로 즉시 청산.
        Circuit Breaker 발동 / 수동 긴급 정지 시 호출.
        """
        logger.warning(f"전량 긴급 청산 시작: {reason}")
        results = []
        try:
            positions = self.exchange.fetch_positions()
            for pos in positions:
                size = float(pos.get("contracts", 0) or 0)
                if size <= 0:
                    continue
                symbol    = pos["symbol"]
                side      = pos.get("side", "")
                close_side = "sell" if side == "long" else "buy"
                result = self._place_order(
                    symbol=symbol, order_type="CLOSE",
                    side=close_side, amount=size,
                    params={"reduceOnly": True},
                )
                results.append(result)
                if result.success:
                    self._active_orders.pop(symbol, None)
                    logger.info(f"[{symbol}] 긴급 청산 완료")
        except Exception as e:
            logger.error(f"긴급 청산 중 오류: {e}", exc_info=True)
        return results

    # ── 퍼블릭: 잔고 조회 ─────────────────────────────────────────────────────

    def fetch_usdt_balance(self) -> float:
        """
        Binance Futures USDT 가용 잔고 조회.
        risk_manager.update_capital() 에 전달.
        """
        try:
            balance = self.exchange.fetch_balance()
            usdt = balance.get("USDT", {})
            free = float(usdt.get("free", 0) or 0)
            logger.debug(f"USDT 잔고: {free:.2f}")
            return free
        except Exception as e:
            logger.error(f"잔고 조회 실패: {e}")
            return 0.0

    def fetch_position_size(self, symbol: str) -> float:
        """특정 심볼의 현재 포지션 수량(contracts) 조회."""
        try:
            positions = self.exchange.fetch_positions([symbol])
            pos_info = next((p for p in positions if p["symbol"] == symbol), None)
            if not pos_info:
                return 0.0
            return float(pos_info.get("contracts", 0) or 0)
        except Exception as e:
            logger.error(f"[{symbol}] 포지션 조회 실패: {e}")
            return 0.0

    # ── 내부: 청산 주문 ────────────────────────────────────────────────────────

    def _close_position(
        self,
        symbol:      str,
        ratio:       float,
        reason:      str,
        price:       float,
        entry_price: float,
        position_context: Optional[dict] = None,
    ) -> OrderResult:
        """
        포지션 일부 또는 전체를 시장가로 청산.
        체결 후 PnL 계산 → circuit_breaker 기록 → 콜백 호출.
        """
        try:
            # 현재 포지션 수량 조회
            positions = self.exchange.fetch_positions([symbol])
            pos_info  = next(
                (p for p in positions if p["symbol"] == symbol), None
            )
            if not pos_info:
                logger.warning(f"[{symbol}] 청산할 포지션 없음")
                return OrderResult(symbol=symbol, error="포지션 없음")

            total_size = float(pos_info.get("contracts", 0) or 0)
            side       = pos_info.get("side", "long")
            close_side = "sell" if side == "long" else "buy"
            close_size = round(total_size * ratio, 6)

            if close_size <= 0:
                return OrderResult(symbol=symbol, error="청산 수량 0")

            # 기존 SL/TP 주문 취소 (전량 청산 시에만)
            if ratio >= 1.0:
                self._cancel_active_orders(symbol)

            result = self._place_order(
                symbol=symbol, order_type="CLOSE",
                side=close_side, amount=close_size,
                params={"reduceOnly": True},
            )
            result.close_reason = reason
            if position_context:
                result.position_context = dict(position_context)

            if result.success:
                # PnL 계산
                result.pnl_usdt = _calc_pnl(
                    side       = side,
                    entry_price= entry_price,
                    close_price= result.avg_price or price,
                    size       = close_size,
                    fee_rate   = TAKER_FEE,
                )
                result.fee_usdt = close_size * (result.avg_price or price) * TAKER_FEE

                logger.info(
                    f"[{symbol}] 청산 완료  "
                    f"reason={reason}  "
                    f"size={close_size:.6f}  "
                    f"price={result.avg_price:.4f}  "
                    f"PnL={result.pnl_usdt:+.2f}U"
                )

                # circuit_breaker 에 거래 결과 기록
                self._cb.record_trade(result.pnl_usdt)

                # 자본 갱신
                new_bal = self.fetch_usdt_balance()
                if new_bal > 0:
                    self._rm.update_capital(new_bal)
                    self._cb.update_capital(new_bal)

                # 전량 청산 시 활성 주문 캐시 정리
                if ratio >= 1.0:
                    self._active_orders.pop(symbol, None)

                # 청산 콜백 (db_logger / telegram)
                if self._on_trade_closed:
                    try:
                        self._on_trade_closed(result)
                    except Exception as e:
                        logger.error(f"on_trade_closed 콜백 오류: {e}")

        except Exception as e:
            logger.error(f"[{symbol}] 청산 처리 오류: {e}", exc_info=True)
            result = OrderResult(symbol=symbol, error=str(e))

        return result

    # ── 내부: SL 이동 ─────────────────────────────────────────────────────────

    def _move_sl(self, symbol: str, new_sl: float) -> Optional[OrderResult]:
        """
        기존 SL 주문 취소 → 새 SL 가격으로 재등록.
        TP1 도달 후 SL → 진입가 이동 시 호출.
        """
        orders = self._active_orders.get(symbol, {})
        old_sl_id = orders.get("sl", "")

        # 기존 SL 취소
        if old_sl_id:
            try:
                self.exchange.cancel_order(old_sl_id, symbol)
                logger.info(f"[{symbol}] 기존 SL 취소: {old_sl_id}")
            except Exception as e:
                logger.warning(f"[{symbol}] SL 취소 실패 (이미 체결됐을 수 있음): {e}")

        # 현재 포지션 방향 조회
        try:
            positions  = self.exchange.fetch_positions([symbol])
            pos_info   = next((p for p in positions if p["symbol"] == symbol), None)
            if not pos_info:
                return None
            side       = pos_info.get("side", "long")
            size       = float(pos_info.get("contracts", 0) or 0)
            close_side = "sell" if side == "long" else "buy"
        except Exception as e:
            logger.error(f"[{symbol}] SL 이동 중 포지션 조회 실패: {e}")
            return None

        # 새 SL 등록
        result = self._place_order(
            symbol     = symbol,
            order_type = "SL",
            side       = close_side,
            amount     = size,
            price      = new_sl,
            params     = {
                "stopPrice":  new_sl,
                "reduceOnly": True,
                "type":       "STOP_MARKET",
                "workingType": "MARK_PRICE",
            },
        )

        if result.success:
            self._active_orders.setdefault(symbol, {})["sl"] = result.order_id
            logger.info(f"[{symbol}] SL 이동 완료: {new_sl:.4f}")

        return result

    # ── 내부: 핵심 주문 실행 (재시도 포함) ────────────────────────────────────

    def _place_order(
        self,
        symbol:     str,
        order_type: str,
        side:       str,
        amount:     float,
        price:      float = 0.0,
        params:     dict  = None,
    ) -> OrderResult:
        """
        ccxt create_order() 래퍼. 최대 MAX_RETRY 회 재시도.

        Parameters
        ----------
        symbol     : 예) "BTC/USDT:USDT"
        order_type : "ENTRY" / "SL" / "TP1" / "TP2" / "CLOSE"
        side       : "buy" or "sell"
        amount     : 주문 수량 (코인 단위)
        price      : 지정가 (시장가 주문은 0)
        params     : 거래소 특화 파라미터

        Returns
        -------
        OrderResult
        """
        params    = params or {}
        result    = OrderResult(symbol=symbol, order_type=order_type,
                                side=side, amount=amount)
        raw_type  = params.pop("type", "MARKET")   # params 에서 꺼냄

        for attempt in range(1, MAX_RETRY + 1):
            result.attempts = attempt
            try:
                order = self.exchange.create_order(
                    symbol = symbol,
                    type   = raw_type,
                    side   = side,
                    amount = amount,
                    price  = price if price > 0 else None,
                    params = params,
                )

                avg_price = float(order.get("average") or order.get("price") or price)

                result.success   = True
                result.order_id  = str(order.get("id", ""))
                result.avg_price = avg_price
                result.fee_usdt  = amount * avg_price * TAKER_FEE
                result.raw       = order

                logger.debug(
                    f"[{symbol}] {order_type} 주문 성공  "
                    f"id={result.order_id}  "
                    f"avg={avg_price:.4f}  "
                    f"attempt={attempt}"
                )

                # 체결 콜백
                if self._on_order_filled:
                    try:
                        self._on_order_filled(result)
                    except Exception as e:
                        logger.error(f"on_order_filled 콜백 오류: {e}")

                return result

            except ccxt.InsufficientFunds as e:
                result.error = f"잔고 부족: {e}"
                logger.error(f"[{symbol}] {order_type} 잔고 부족 — 재시도 안 함")
                break   # 잔고 부족은 재시도 의미 없음

            except ccxt.InvalidOrder as e:
                result.error = f"잘못된 주문: {e}"
                logger.error(f"[{symbol}] {order_type} 잘못된 주문 파라미터: {e}")
                break   # 파라미터 오류는 재시도 의미 없음

            except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                result.error = str(e)
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    f"[{symbol}] {order_type} 네트워크 오류 "
                    f"(attempt={attempt}/{MAX_RETRY}) — {delay:.1f}초 후 재시도: {e}"
                )
                if attempt < MAX_RETRY:
                    time.sleep(delay)

            except ccxt.ExchangeError as e:
                result.error = str(e)
                logger.error(f"[{symbol}] {order_type} 거래소 오류: {e}")
                break

            except Exception as e:
                result.error = str(e)
                logger.error(f"[{symbol}] {order_type} 예상치 못한 오류: {e}",
                             exc_info=True)
                break

        if not result.success:
            logger.error(
                f"[{symbol}] {order_type} 최종 실패 "
                f"(attempt={result.attempts}): {result.error}"
            )
        return result

    # ── 내부: 긴급 역방향 청산 ────────────────────────────────────────────────

    def _emergency_close(
        self,
        symbol:    str,
        amount:    float,
        close_side: str,
    ) -> None:
        """SL 등록 실패 시 즉시 역방향 시장가로 진입 포지션 청산."""
        logger.error(f"[{symbol}] 긴급 역방향 청산 실행  size={amount}")
        self._place_order(
            symbol=symbol, order_type="CLOSE",
            side=close_side, amount=amount,
            params={"reduceOnly": True},
        )

    # ── 내부: 활성 주문 전체 취소 ─────────────────────────────────────────────

    def _cancel_active_orders(self, symbol: str) -> None:
        """캐시에 저장된 SL/TP 주문 전부 취소."""
        orders = self._active_orders.get(symbol, {})
        for order_type, order_id in orders.items():
            if not order_id:
                continue
            try:
                self.exchange.cancel_order(order_id, symbol)
                logger.info(f"[{symbol}] {order_type} 주문 취소: {order_id}")
            except Exception as e:
                logger.warning(f"[{symbol}] {order_type} 취소 실패: {e}")


# ── 순수 함수 ──────────────────────────────────────────────────────────────────

def _recalc_levels(
    plan:         PositionPlan,
    actual_entry: float,
) -> tuple[float, float, float]:
    """
    실제 체결가 기준으로 SL / TP1 / TP2 재계산.
    시장가 주문은 예상가와 실제 체결가가 다를 수 있어 재계산이 필요.
    """
    sl_dist  = abs(plan.entry_price - plan.sl_price)
    tp1_dist = abs(plan.tp1_price - plan.entry_price)
    tp2_dist = abs(plan.tp2_price - plan.entry_price)

    if plan.direction == "LONG":
        return (
            round(actual_entry - sl_dist,  4),
            round(actual_entry + tp1_dist, 4),
            round(actual_entry + tp2_dist, 4),
        )
    else:
        return (
            round(actual_entry + sl_dist,  4),
            round(actual_entry - tp1_dist, 4),
            round(actual_entry - tp2_dist, 4),
        )


def _calc_pnl(
    side:        str,
    entry_price: float,
    close_price: float,
    size:        float,
    fee_rate:    float,
) -> float:
    """
    실현 PnL 계산 (수수료 차감).

    PnL (LONG) = (close - entry) × size - 수수료
    PnL (SHORT)= (entry - close) × size - 수수료
    """
    if side == "long":
        raw_pnl = (close_price - entry_price) * size
    else:
        raw_pnl = (entry_price - close_price) * size

    total_fee = (entry_price + close_price) * size * fee_rate
    return round(raw_pnl - total_fee, 4)


# ── 유틸 ───────────────────────────────────────────────────────────────────────

def report_summary(report: ExecutionReport) -> str:
    """ExecutionReport 한 줄 요약."""
    if not report.entry:
        return "[실행] 진입 주문 없음"
    e = report.entry
    lines = [
        f"[실행] {'성공' if report.success else '실패'}  "
        f"entry={e.avg_price:.4f}  "
        f"size={e.amount:.6f}  "
        f"fee={e.fee_usdt:.4f}U"
    ]
    for label, r in [("SL", report.sl), ("TP1", report.tp1), ("TP2", report.tp2)]:
        if r:
            st = "OK" if r.success else f"FAIL({r.error[:30]})"
            lines.append(f"  {label}: {st}")
    return "\n".join(lines)


# ── 단독 실행 테스트 (페이퍼 모드) ────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import pandas as pd
    from dotenv import load_dotenv
    import signal_engine, signal_scorer
    from risk_manager    import RiskManager
    from circuit_breaker import CircuitBreaker

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    load_dotenv()

    API_KEY    = os.getenv("BINANCE_API_KEY", "")
    API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    CAPITAL    = float(os.getenv("TOTAL_CAPITAL", "1000"))
    USE_TESTNET = os.getenv("USE_TESTNET", "true").lower() == "true"

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

    rm = RiskManager(total_capital=CAPITAL)
    cb = CircuitBreaker(total_capital=CAPITAL)

    def on_trade_closed(result: OrderResult):
        logger.info(
            f"[거래 완료] {result.symbol}  "
            f"PnL={result.pnl_usdt:+.2f}U  "
            f"fee={result.fee_usdt:.4f}U"
        )

    executor = OrderExecutor(
        api_key         = API_KEY,
        api_secret      = API_SECRET,
        risk_manager    = rm,
        circuit_breaker = cb,
        on_trade_closed = on_trade_closed,
        testnet         = USE_TESTNET,
    )

    sym    = "BTC/USDT:USDT"
    df_1h  = fetch_df(sym, "1h",  250)
    df_15m = fetch_df(sym, "15m", 150)

    sig   = signal_engine.check(sym, df_1h, df_15m)
    score = signal_scorer.evaluate(sig)
    plan  = rm.calculate_plan_for(sym, sig, score)

    print(f"\n심볼: {sym}")
    print(f"신호: {sig.signal}  confidence={score.confidence}  leverage={score.leverage}x")

    if plan.can_open:
        print(f"\n계획:")
        print(f"  진입가  : {plan.entry_price:.4f}")
        print(f"  SL      : {plan.sl_price:.4f}")
        print(f"  TP1     : {plan.tp1_price:.4f}")
        print(f"  TP2     : {plan.tp2_price:.4f}")
        print(f"  크기    : {plan.position_size:.6f} 코인")
        print(f"  명목    : {plan.notional:.2f} USDT")
        print()

        if USE_TESTNET:
            print("[ 테스트넷 실행 ]")
            report = executor.execute_entry(plan)
            print(report_summary(report))
            if report.success:
                rm.open_position(plan)
        else:
            print("[ 실거래 모드 — 실행 건너뜀 (USE_TESTNET=true 로 설정 후 테스트) ]")
    else:
        print(f"\n진입 불가: {plan.reject_reason}")

    cb.stop()