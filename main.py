"""
main.py
-------
CryptoSniper Bot 진입점.

전체 실행 흐름:
  1. 설정 로드 (config.py)
  2. 로깅 초기화
  3. 모듈 인스턴스 생성 및 의존성 주입
  4. CoinScanner 로 초기 후보 코인 선정
  5. DataFetcher WebSocket 구독 시작
  6. 매 캔들 닫힘 → on_candle_close() 콜백
       └─ CircuitBreaker 차단 확인
       └─ signal_engine → signal_scorer → leverage_manager
       └─ risk_manager.calculate_plan_for()
       └─ orderbook 유동성 최종 점검
       └─ order_executor.execute_entry()
  7. 매 5분 주기 → run_scan_cycle()
       └─ CoinScanner 재스캔 → 구독 심볼 업데이트
  8. APScheduler 로 스케줄 관리
  9. 일별 리포트 자정 자동 전송
 10. KeyboardInterrupt / SIGTERM 시 정상 종료
"""

import logging
import logging.handlers
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# ── sys.path 설정 — 반드시 내부 모듈 임포트보다 먼저 실행 ──────────────────────
_ROOT = Path(__file__).resolve().parent
for _sub in ("", "data", "indicators", "strategy", "risk", "execution", "utils"):
    _p = str(_ROOT / _sub) if _sub else str(_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

# ── 내부 모듈 ──────────────────────────────────────────────────────────────────
from config              import get_config
from coin_scanner        import CoinScanner
from data_fetcher        import DataFetcher
from orderbook           import OrderBookManager
import signal_engine
import signal_scorer
from risk_manager        import RiskManager
from leverage_manager    import LeverageManager
from circuit_breaker     import CircuitBreaker
from order_executor      import OrderExecutor, OrderResult
from db_logger           import DbLogger, TradeRecord, SignalRecord
from telegram_notifier   import TelegramNotifier, NotifyLevel


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 로깅 초기화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _setup_logging(log_level: str, log_file: str) -> None:
    """
    콘솔 + 파일 핸들러 설정.
    파일은 10 MB 도달 시 롤링, 최대 5개 백업 유지.
    """
    level   = getattr(logging, log_level.upper(), logging.INFO)
    fmt     = "%(asctime)s [%(levelname)s] %(name)-20s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(level)

    # 콘솔 핸들러
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(sh)

    # 파일 핸들러 (롤링)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5,
            encoding="utf-8",
        )
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        root.addHandler(fh)

    # 외부 라이브러리 로그 수준 억제
    for lib in ("ccxt", "websocket", "urllib3", "apscheduler"):
        logging.getLogger(lib).setLevel(logging.WARNING)


logger = logging.getLogger("main")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 봇 메인 클래스
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CryptoSniperBot:
    """
    전체 봇 라이프사이클을 관리하는 최상위 클래스.

    start() → 봇 시작 (블로킹)
    stop()  → 정상 종료
    """

    def __init__(self):
        self.cfg = get_config()
        self._running = False
        self._stop_event = threading.Event()

        # ── 모듈 인스턴스 ────────────────────────────────────────────────────
        self.db         = DbLogger(self.cfg.system.db_path)
        self.notifier   = TelegramNotifier(
            token    = self.cfg.notification.telegram_token,
            chat_id  = self.cfg.notification.telegram_chat_id,
            bot_name = self.cfg.notification.bot_name,
        )
        self.cb = CircuitBreaker(
            total_capital        = self.cfg.risk.initial_capital,
            daily_loss_limit_pct = self.cfg.risk.daily_loss_limit_pct,
            consec_loss_limit    = self.cfg.risk.consec_loss_limit,
            cooldown_seconds     = self.cfg.risk.cooldown_seconds,
            on_halt              = self.notifier.send_halt    if self.cfg.notification.notify_halt else None,
            on_release           = self.notifier.send_release if self.cfg.notification.notify_halt else None,
        )
        self.rm = RiskManager(
            total_capital        = self.cfg.risk.initial_capital,
            risk_per_trade_pct   = self.cfg.risk.risk_per_trade_pct,
            max_positions        = self.cfg.risk.max_positions,
            min_position_usdt    = self.cfg.risk.min_position_usdt,
            atr_sl_multiplier    = self.cfg.risk.atr_sl_multiplier,
            tp1_ratio            = self.cfg.risk.tp1_ratio,
            tp2_ratio            = self.cfg.risk.tp2_ratio,
            tp1_close_pct        = self.cfg.risk.tp1_close_pct,
            trailing_trigger_pct = self.cfg.risk.trailing_trigger_pct,
            max_notional_pct     = self.cfg.risk.max_notional_pct,
            max_notional_abs     = self.cfg.risk.max_notional_abs,
        )
        self.lm = LeverageManager(
            api_key    = self.cfg.exchange.api_key,
            api_secret = self.cfg.exchange.api_secret,
            testnet    = self.cfg.exchange.testnet,
        )
        self.scanner = CoinScanner(
            api_key    = self.cfg.exchange.api_key,
            api_secret = self.cfg.exchange.api_secret,
            testnet    = self.cfg.exchange.testnet,
        )
        self.ob = OrderBookManager(symbols=[])   # 심볼은 스캔 후 설정

        self.executor = OrderExecutor(
            api_key         = self.cfg.exchange.api_key,
            api_secret      = self.cfg.exchange.api_secret,
            risk_manager    = self.rm,
            circuit_breaker = self.cb,
            on_trade_closed = self._on_trade_closed,
            testnet         = self.cfg.exchange.testnet,
        )

        # DataFetcher 는 스캔 후 심볼 확정 시점에 생성
        self.fetcher: DataFetcher | None = None

        # 스케줄러
        self.scheduler = BackgroundScheduler(timezone="UTC")

        # 현재 구독 심볼 캐시
        self._subscribed_symbols: list[str] = []
        self._symbols_lock = threading.Lock()

        # 진행 중인 진입 처리 중복 방지 {symbol: True}
        self._processing: dict[str, bool] = {}
        self._processing_lock = threading.Lock()

        # 청산 후 쿨다운 {symbol: close_time}
        # 같은 종목 재진입 시 COOLDOWN_MINUTES 경과 여부 체크
        self._close_cooldown: dict[str, float] = {}
        self._cooldown_minutes = 60   # 기본 60분 쿨다운

    # ── 라이프사이클 ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """봇 시작. Ctrl+C 또는 stop() 호출 시까지 블로킹."""
        logger.info("=" * 60)
        logger.info(f"  CryptoSniper Bot 시작")
        logger.info(f"  자본: {self.cfg.risk.initial_capital:,.2f} USDT")
        logger.info(f"  모드: {'테스트넷' if self.cfg.exchange.testnet else '실거래'}"
                    f"{'  [DRY RUN]' if self.cfg.system.dry_run else ''}")
        logger.info(f"  ── 실효 전략 파라미터 ──")
        logger.info(f"  min_score_to_enter : {self.cfg.strategy.min_score_to_enter}")
        logger.info(f"  min_confidence     : {self.cfg.strategy.min_confidence}")
        logger.info(f"  risk_per_trade_pct : {self.cfg.risk.risk_per_trade_pct*100:.1f}%")
        logger.info(f"  max_positions      : {self.cfg.risk.max_positions}")
        logger.info("=" * 60)

        self._running = True

        # 1. 텔레그램 알림 시작
        if self.cfg.notification.telegram_token:
            self.notifier.start()
            self._register_telegram_commands()

        # 2. 관심 심볼 초기 갱신 (REST 대량 호출 — 1회만)
        logger.info("관심 심볼 초기 갱신 중...")
        self._refresh_watchlist()

        # 3. 관심 심볼 기반 초기 스캔 (30개 대상)
        logger.info("초기 코인 스캔 시작...")
        candidates = self._run_scan()
        # 수정안 2: 후보 유무와 관계없이 watchlist 전체를 기본 구독
        # scanner 후보는 우선순위 참고용이지, 감시 범위를 축소하지 않음
        watchlist_symbols = self.scanner.get_watchlist() or ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        if not candidates:
            logger.warning("초기 스캔에서 후보 코인 없음 — 관심 심볼 전체로 시작")
        else:
            logger.info(f"초기 후보 {len(candidates)}개 — watchlist 전체({len(watchlist_symbols)}개) 구독")
        self._start_feeds(watchlist_symbols)

        # 4. 스케줄러 등록
        self._register_schedules()
        self.scheduler.start()
        logger.info("스케줄러 시작 완료")

        # 5. 메인 루프 (stop_event 대기)
        logger.info("봇 정상 가동 중... (Ctrl+C 로 종료)")
        try:
            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=1.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """정상 종료 — 모든 서브시스템 순서대로 종료."""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()

        logger.info("봇 종료 시작...")

        self.scheduler.shutdown(wait=False)

        if self.fetcher:
            self.fetcher.stop()

        self.ob.stop()
        self.cb.stop()

        if self.cfg.notification.telegram_token:
            self.notifier.send_raw("🛑 *CryptoSniper Bot 종료*", NotifyLevel.HIGH)
            time.sleep(2)
            self.notifier.stop()

        logger.info("봇 종료 완료")

    # ── 피드 관리 ─────────────────────────────────────────────────────────────

    def _start_feeds(self, symbols: list[str]) -> None:
        """DataFetcher / OrderBookManager 심볼 설정 후 시작."""
        # 포지션 보유 중인 심볼은 항상 포함 (호가창 유실 방지)
        position_symbols = {pos.symbol for pos in self.rm.get_all_positions()}
        all_symbols = list(dict.fromkeys(list(symbols) + list(position_symbols)))
        if position_symbols - set(symbols):
            logger.debug(
                f"포지션 심볼 피드 유지: {position_symbols - set(symbols)}"
            )

        with self._symbols_lock:
            self._subscribed_symbols = list(all_symbols)
        symbols = all_symbols

        timeframes = [
            self.cfg.strategy.entry_timeframe,    # 15m
            self.cfg.strategy.trend_timeframe,    # 1h
            self.cfg.strategy.confirm_timeframe,  # 5m (눌림목 전략용)
        ]

        # DataFetcher
        if self.fetcher:
            self.fetcher.stop()

        self.fetcher = DataFetcher(
            symbols          = symbols,
            timeframes       = timeframes,
            on_closed_candle = self._on_candle_close,
            api_key          = self.cfg.exchange.api_key,
            api_secret       = self.cfg.exchange.api_secret,
            testnet          = self.cfg.exchange.testnet,
        )
        self.fetcher.start()

        # OrderBookManager
        self.ob.stop()
        self.ob = OrderBookManager(symbols=symbols)
        self.ob.start()

        logger.info(f"피드 시작: {len(symbols)}개 심볼  {timeframes}")

    # ── 캔들 닫힘 콜백 (핵심 진입 로직) ──────────────────────────────────────

    def _on_candle_close(
        self,
        symbol:    str,
        timeframe: str,
        df:        pd.DataFrame,
    ) -> None:
        """
        DataFetcher 가 캔들 닫힘을 감지할 때마다 호출.

        15M 캔들 닫힘 시에만 신호 판단 실행.
        (1H 캔들은 추세 데이터로만 사용, 별도 진입 판단 안 함)
        """
        # 15m 캔들 닫힘 시에만 진입 판단 (5m은 데이터 축적용, 1h는 추세용)
        if timeframe != self.cfg.strategy.entry_timeframe:
            return

        # 동일 심볼 중복 처리 방지
        with self._processing_lock:
            if self._processing.get(symbol):
                return
            self._processing[symbol] = True

        try:
            self._process_signal(symbol)
        except Exception as e:
            logger.error(f"[{symbol}] 신호 처리 중 예외: {e}", exc_info=True)
            self.db.save_error("main._on_candle_close", str(e), symbol)
        finally:
            with self._processing_lock:
                self._processing[symbol] = False

    def _process_signal(self, symbol: str) -> None:
        """
        단일 심볼에 대한 전체 진입 판단 파이프라인.

        Step 1. Circuit Breaker 차단 확인
        Step 2. 이미 포지션 있으면 포지션 업데이트 후 스킵
        Step 3. 1H / 15M DataFrame 가져오기
        Step 4. signal_engine → signal_scorer
        Step 5. 신호 기록 (DB)
        Step 6. HOLD 이면 종료
        Step 7. leverage_manager 최종 레버리지 결정
        Step 8. risk_manager 플랜 계산
        Step 9. orderbook 유동성 최종 점검
        Step 10. order_executor 실행
        """
        cfg = self.cfg

        # ── Step 1. Circuit Breaker ────────────────────────────────────────
        cb_status = self.cb.check()
        if cb_status.blocked:
            logger.info(f"[{symbol}] CB 차단 중 — 진입 스킵: {cb_status.reason}")
            return

        # 텔레그램 /pause 명령으로 일시 중단 중이면 신규 진입 스킵
        if getattr(self, "_paused", False):
            logger.debug(f"[{symbol}] 일시 중단 중 — 진입 스킵")
            return

        # 쿨다운 체크 — 최근 청산 후 일정 시간 재진입 차단
        import time as _time
        cooldown_end = self._close_cooldown.get(symbol, 0)
        if _time.time() < cooldown_end:
            remaining = int((cooldown_end - _time.time()) / 60)
            logger.debug(f"[{symbol}] 쿨다운 중 — {remaining}분 후 재진입 가능")
            return

        # ── Step 2. 포지션 업데이트 ───────────────────────────────────────
        pos = self.rm.get_position(symbol)
        if pos:
            exchange_size = self.executor.fetch_position_size(symbol)
            if exchange_size is None:
                logger.warning(f"[{symbol}] 포지션 조회 API 오류 — 이번 사이클 스킵")
                return
            if exchange_size <= 0:
                self._close_position_with_notify(symbol, reason="거래소 포지션 종료 감지")
                return

            current_price = self._get_current_price(symbol)
            if current_price > 0:
                actions = self.rm.update_positions({symbol: current_price})
                for action in actions:
                    # SL 손절 / TP1 50% 익절 / TP2 전량 익절 모두 처리
                    act_type = action.get("action", "")
                    if act_type in ("CLOSE_FULL", "CLOSE_PARTIAL", "MOVE_SL"):
                        result = self.executor.handle_action(
                            action,
                            entry_price=pos.entry_price,
                            position_context={
                                "direction":    pos.direction,
                                "entry_price":  pos.entry_price,
                                "leverage":     pos.leverage,
                                "atr_at_entry": pos.atr_at_entry,
                                "confidence":   pos.confidence,
                            },
                        )
                        if result and result.success:
                            logger.info(
                                f"[{symbol}] 청산 완료  "
                                f"사유={action.get('reason')}  "
                                f"price={current_price:.4f}"
                            )
            return

        # ── Step 3. DataFrame 수집 ────────────────────────────────────────
        df_1h  = self.fetcher.get_df(symbol, cfg.strategy.trend_timeframe)
        df_15m = self.fetcher.get_df(symbol, cfg.strategy.entry_timeframe)
        df_5m  = self.fetcher.get_df(symbol, cfg.strategy.confirm_timeframe)

        if df_1h is None or df_15m is None:
            logger.debug(f"[{symbol}] 데이터 미준비 — 스킵")
            return
        if len(df_1h) < 205 or len(df_15m) < 50:
            logger.debug(f"[{symbol}] 캔들 수 부족 — 스킵")
            return

        # BTC 1H 데이터 수집 (알트코인 시장 필터용)
        btc_df_1h = None
        if "BTC" not in symbol:
            btc_df_1h = self.fetcher.get_df("BTC/USDT:USDT", cfg.strategy.trend_timeframe)

        # ── Step 4. 신호 판단 ─────────────────────────────────────────────
        sig = signal_engine.check(
            symbol,
            df_1h,
            df_15m,
            btc_df_1h=btc_df_1h,
            df_5m=df_5m if (df_5m is not None and len(df_5m) >= 30) else None,
            min_score_to_enter=cfg.strategy.min_score_to_enter,
        )
        score = signal_scorer.evaluate(
            sig,
            min_confidence=cfg.strategy.min_confidence,
        )

        # ── Step 5. 신호 DB 기록 ──────────────────────────────────────────
        self._save_signal_record(symbol, sig, score)

        # ── Step 6. HOLD 처리 ─────────────────────────────────────────────
        if sig.signal == "HOLD" or not score.can_enter:
            logger.debug(
                f"[{symbol}] HOLD  "
                f"score={sig.score}  "
                f"confidence={score.confidence}  "
                f"reason={sig.reject_reason or '점수 부족'}"
            )
            return

        # ── Step 7. 레버리지 결정 ─────────────────────────────────────────
        decision = self.lm.decide(symbol, score, cb_status)
        if not decision.set_on_exchange:
            logger.warning(f"[{symbol}] 레버리지 설정 실패 — 진입 스킵")
            return

        # ── Step 8. PositionPlan 계산 ─────────────────────────────────────
        # 최종 레버리지를 score 에 반영한 임시 객체 생성
        final_score = _clone_score_with_leverage(score, decision.final_leverage)
        plan = self.rm.calculate_plan_for(symbol, sig, final_score)

        if not plan.can_open:
            logger.info(f"[{symbol}] 진입 플랜 거절: {plan.reject_reason}")
            return

        # ── Step 9. 호가창 유동성 최종 점검 ──────────────────────────────
        safe, ob_reason = self.ob.is_safe_to_enter(
            symbol,
            direction  = sig.signal,
            target_qty = plan.position_size,
        )
        if not safe:
            logger.info(f"[{symbol}] 유동성 불충분 — 진입 스킵: {ob_reason}")
            return

        # ── Step 10. 주문 실행 ────────────────────────────────────────────
        if cfg.system.dry_run:
            logger.info(
                f"[DRY RUN] {symbol}  {sig.signal}  "
                f"entry={plan.entry_price:.4f}  "
                f"SL={plan.sl_price:.4f}  "
                f"lev={plan.leverage}x  "
                f"size={plan.position_size:.6f}"
            )
            return

        report = self.executor.execute_entry(plan)

        if report.success:
            from dataclasses import replace

            from order_executor import _recalc_levels

            actual_entry = report.entry.avg_price or plan.entry_price
            actual_sl, actual_tp1, actual_tp2 = _recalc_levels(plan, actual_entry)
            actual_plan = replace(
                plan,
                entry_price=actual_entry,
                sl_price=actual_sl,
                tp1_price=actual_tp1,
                tp2_price=actual_tp2,
                notional=round(plan.position_size * actual_entry, 2),
            )
            self.rm.open_position(actual_plan)
            logger.info(
                f"[{symbol}] 진입 성공  "
                f"{sig.signal} × {plan.leverage}x  "
                f"entry={actual_entry:.4f}  "
                f"SL={actual_sl:.4f}  TP1={actual_tp1:.4f}"
            )

            if cfg.notification.notify_entry:
                self.notifier.send_entry(
                    symbol         = symbol,
                    direction      = sig.signal,
                    entry_price    = actual_entry,
                    sl_price       = actual_sl,
                    tp1_price      = actual_tp1,
                    tp2_price      = actual_tp2,
                    position_size  = plan.position_size,
                    leverage       = plan.leverage,
                    confidence     = score.confidence,
                    notional       = actual_plan.notional,
                )
        else:
            logger.error(
                f"[{symbol}] 진입 실패: "
                f"{report.entry.error if report.entry else '알 수 없음'}"
            )

    # ── 스캔 사이클 ───────────────────────────────────────────────────────────

    def _refresh_watchlist_b(self) -> None:
        """B군 watchlist 1시간마다 갱신 — 지금 움직이는 종목 교체."""
        try:
            if not hasattr(self.scanner, "refresh_watchlist_b"):
                return
            watchlist = self.scanner.refresh_watchlist_b()
            logger.info(f"B군 갱신 완료: watchlist={len(watchlist)}개")
            # 피드 재구독 (새 종목 추가됐을 수 있음)
            surge_symbols = set(self.scanner.get_surge_symbols().keys())
            all_symbols   = list(set(watchlist) | surge_symbols)
            self._start_feeds(all_symbols)
        except Exception as e:
            logger.error(f"B군 갱신 오류: {e}", exc_info=True)

    def _refresh_watchlist(self) -> None:
        """
        관심 심볼 30개 갱신 — 자정마다 스케줄러 호출.
        REST API 대량 호출(수백 회)은 오직 이 메서드에서만 발생.
        """
        try:
            watchlist = self.scanner.refresh_watchlist()
            logger.info(
                f"관심 심볼 갱신 완료: {len(watchlist)}개  "
                f"(예: {watchlist[:3]}...)"
            )
            if self.cfg.notification.telegram_token:
                self.notifier.send_raw(
                    f"🔄 관심 심볼 갱신 완료 ({len(watchlist)}개)",
                    NotifyLevel.LOW,
                )
        except Exception as e:
            logger.error(f"관심 심볼 갱신 오류: {e}", exc_info=True)

    def _run_scan(self) -> list[dict]:
        """관심 심볼(30개) 대상으로 스캔 실행 후 후보 목록 반환."""
        try:
            candidates = self.scanner.scan()
            logger.info(f"스캔 완료: {len(candidates)}개 후보")
            return candidates
        except Exception as e:
            logger.error(f"코인 스캔 오류: {e}", exc_info=True)
            return []

    def _scheduled_scan(self) -> None:
        """스케줄러에서 5분마다 호출 — 후보 변경 시 피드 재구독."""
        logger.info("정기 스캔 시작...")
        candidates = self._run_scan()
        if not candidates:
            return

        new_symbols = [c["symbol"] for c in candidates]

        with self._symbols_lock:
            current = set(self._subscribed_symbols)

        new_set = set(new_symbols)
        if new_set == current:
            logger.debug("스캔 결과 동일 — 피드 유지")
            return

        added   = new_set - current
        removed = current - new_set
        logger.info(
            f"심볼 변경 감지  "
            f"추가={list(added)}  제거={list(removed)}"
        )

        # 제거 심볼에 활성 포지션 있으면 제외하고 유지
        safe_to_remove = set()
        for sym in removed:
            if not self.rm.get_position(sym):
                safe_to_remove.add(sym)

        final_symbols = list((current - safe_to_remove) | new_set)
        self._start_feeds(final_symbols)

    # ── 스케줄 등록 ──────────────────────────────────────────────────────────

    def _register_schedules(self) -> None:
        """APScheduler 작업 등록."""

        # ── 자정 00:01 UTC: 관심 심볼 갱신 (REST 대량 호출은 여기서만 발생)
        self.scheduler.add_job(
            self._refresh_watchlist,
            trigger = CronTrigger(hour=0, minute=1, timezone="UTC"),
            id      = "watchlist_refresh",
            name    = "관심 심볼 갱신",
            misfire_grace_time = 60,
        )

        # ── 5분마다: 관심 심볼(30개)만 분석 → 후보 선정
        #    scan_interval 기본값 300초(5분). 30개만 조회하므로 429 없음
        self.scheduler.add_job(
            self._scheduled_scan,
            trigger = IntervalTrigger(seconds=self.cfg.scanner.scan_interval),
            id      = "scan",
            name    = "코인 스캔",
            misfire_grace_time = 30,
        )

        # ── 1분마다: 포지션 SL / 트레일링 체크
        self.scheduler.add_job(
            self._update_all_positions,
            trigger = IntervalTrigger(seconds=60),
            id      = "position_update",
            name    = "포지션 업데이트",
            misfire_grace_time = 10,
        )

        # ── 자정 00:05 UTC: 일별 성과 리포트 전송
        self.scheduler.add_job(
            self._send_daily_report,
            trigger = CronTrigger(
                hour     = self.cfg.notification.daily_report_hour,
                minute   = self.cfg.notification.daily_report_minute,
                timezone = "UTC",
            ),
            id   = "daily_report",
            name = "일별 리포트",
        )

        # ── 1시간마다: B군 watchlist 갱신 (지금 움직이는 종목 교체)
        self.scheduler.add_job(
            self._refresh_watchlist_b,
            trigger = IntervalTrigger(minutes=60),
            id      = "watchlist_b_refresh",
            name    = "B군 watchlist 갱신",
            misfire_grace_time = 30,
        )

        # ── 5분마다: 잔고 갱신
        self.scheduler.add_job(
            self._refresh_balance,
            trigger = IntervalTrigger(seconds=300),
            id      = "balance_refresh",
            name    = "잔고 갱신",
            misfire_grace_time = 30,
        )

        logger.debug("스케줄 등록 완료")

    # ── 주기적 작업 ──────────────────────────────────────────────────────────

    def _update_all_positions(self) -> None:
        """1분마다 모든 포지션에 대해 현재가 기반 업데이트."""
        positions = self.rm.get_all_positions()
        if not positions:
            return

        for pos in positions:
            exchange_size = self.executor.fetch_position_size(pos.symbol)
            if exchange_size is None:
                logger.warning(f"[{pos.symbol}] 포지션 조회 API 오류 — 이번 사이클 스킵")
                continue
            if exchange_size <= 0:
                self._close_position_with_notify(pos.symbol, reason="거래소 포지션 종료 감지")
                continue
            price = self._get_current_price(pos.symbol)
            if price <= 0:
                continue
            actions = self.rm.update_positions({pos.symbol: price})
            for action in actions:
                # ★ SIMPLE MODE: CLOSE_FULL / CLOSE_PARTIAL 모두 즉시 실행
                # ★ TRAILING MODE로 전환 시:
                #   if action.get("reason", "").startswith("트레일링"): 조건 복원
                act = action.get("action", "")
                if act in ("CLOSE_FULL", "CLOSE_PARTIAL"):
                    result = self.executor.handle_action(
                        action,
                        entry_price=pos.entry_price,
                        position_context={
                            "direction":    pos.direction,
                            "entry_price":  pos.entry_price,
                            "leverage":     pos.leverage,
                            "atr_at_entry": pos.atr_at_entry,
                            "confidence":   pos.confidence,
                        },
                    )
                    if result and result.success:
                        if act == "CLOSE_FULL":
                            self._close_position_with_notify(
                                pos.symbol,
                                reason=action.get("reason", "청산")
                            )
                # MOVE_SL은 TRAILING MODE에서만 사용 (현재 비활성)
                # elif act == "MOVE_SL":
                #     self.executor.handle_action(action, entry_price=pos.entry_price)

    def _refresh_balance(self) -> None:
        """5분마다 USDT 잔고를 거래소에서 조회해 자본 갱신."""
        try:
            balance = self.executor.fetch_usdt_balance()
            if balance > 0:
                self.rm.update_capital(balance)
                self.cb.update_capital(balance)
                logger.debug(f"잔고 갱신: {balance:.2f} USDT")
        except Exception as e:
            logger.error(f"잔고 갱신 오류: {e}")

    def _send_daily_report(self) -> None:
        """자정 일별 성과 리포트 전송."""
        try:
            from datetime import date, timedelta
            yesterday = date.today() - timedelta(days=1)
            stats = self.db.get_daily_stats(yesterday)
            balance = self.executor.fetch_usdt_balance()
            stats.capital_end = balance
            self.db.save_error  # dummy — 실제로는 daily_stats 테이블 저장 가능
            if self.cfg.notification.notify_daily:
                self.notifier.send_daily_report(stats, capital=balance)
            logger.info(
                f"일별 리포트 전송 완료  "
                f"{stats.date_str}  "
                f"PnL={stats.total_pnl:+.2f}U  "
                f"승률={stats.win_rate*100:.1f}%"
            )
        except Exception as e:
            logger.error(f"일별 리포트 오류: {e}")

    # ── 텔레그램 명령 핸들러 ──────────────────────────────────────────────────

    def _register_telegram_commands(self) -> None:
        n = self.notifier
        n.register_command("/status",    self._cmd_status)
        n.register_command("/positions", self._cmd_positions)
        n.register_command("/watchlist", self._cmd_watchlist)
        n.register_command("/close",     self._cmd_close)
        n.register_command("/closeall",  self._cmd_closeall)
        n.register_command("/pause",     self._cmd_pause)
        n.register_command("/resume",    self._cmd_resume)
        n.register_command("/uncool",    self._cmd_uncool)
        n.register_command("/help",      self._cmd_help)
        logger.info("텔레그램 명령 핸들러 등록 완료")

    def _cmd_help(self, args: str) -> None:
        text = (
            "📋 *CryptoSniper 명령어*\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "📊 `/status`  봇 상태 요약\n"
            "💰 `/positions`  보유 포지션\n"
            "👁 `/watchlist`  감시 종목\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "⏹ `/close SYMBOL`  종목 청산\n"
            "⏹ `/closeall`  전체 청산\n"
            "⏸ `/pause`  신규 진입 중단\n"
            "▶️ `/resume`  신규 진입 재개\n"
            "🔓 `/uncool SYMBOL`  쿨다운 해제 (재진입 허용)"
        )
        self.notifier.send_raw(text)

    def _cmd_status(self, args: str) -> None:
        positions = self.rm.get_all_positions()
        watchlist = self.scanner.get_watchlist()
        surge     = self.scanner.get_surge_symbols()
        cb_status = self.cb.check()
        paused    = getattr(self, "_paused", False)
        status_icon = "⏸ 일시중단" if paused else "🟢 가동 중"

        pos_lines = []
        for pos in positions:
            price = self._get_current_price(pos.symbol)
            if price > 0 and pos.entry_price > 0:
                pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                if pos.direction == "SHORT":
                    pnl_pct = -pnl_pct
                pnl_pct *= pos.leverage
                sign = "+" if pnl_pct >= 0 else ""
                sym = pos.symbol.split("/")[0]
                pos_lines.append(
                    f"  {sym} {pos.direction}x{pos.leverage} "
                    f"진입={pos.entry_price:.4f} {sign}{pnl_pct:.1f}%"
                )

        text = (
            f"🤖 *CryptoSniper 상태*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"상태: {status_icon}\n"
            f"CB: {'🔴 차단' if cb_status.blocked else '🟢 정상'}\n"
            f"자본: `{self.rm.total_capital:,.2f} USDT`\n"
            f"포지션: {len(positions)}개\n"
        )
        if pos_lines:
            text += "\n".join(pos_lines) + "\n"
        text += f"감시: {len(watchlist)}개"
        if surge:
            text += f" +급증 {len(surge)}개"

        buttons = [
            [
                {"text": "💰 포지션", "callback_data": "cb_positions"},
                {"text": "👁 감시목록", "callback_data": "cb_watchlist"},
            ],
            [{"text": "⏹ 전체청산", "callback_data": "cb_closeall"}],
        ]
        self.notifier.register_callback("cb_positions", lambda: self._cmd_positions(""))
        self.notifier.register_callback("cb_watchlist", lambda: self._cmd_watchlist(""))
        self.notifier.register_callback("cb_closeall",  lambda: self._cmd_closeall(""))
        self.notifier.send_with_keyboard(text, buttons)

    def _cmd_positions(self, args: str) -> None:
        positions = self.rm.get_all_positions()
        if not positions:
            self.notifier.send_raw("📭 현재 보유 포지션 없음")
            return

        lines = ["💰 *보유 포지션*\n━━━━━━━━━━━━━━━━━━━━"]
        for pos in positions:
            price = self._get_current_price(pos.symbol)
            pnl_pct = 0.0
            if price > 0 and pos.entry_price > 0:
                pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                if pos.direction == "SHORT":
                    pnl_pct = -pnl_pct
                pnl_pct *= pos.leverage
            sign  = "+" if pnl_pct >= 0 else ""
            emoji = "🟢" if pnl_pct >= 0 else "🔴"
            sym   = pos.symbol.split("/")[0]
            lines.append(
                f"{emoji} *{sym}* {pos.direction}x{pos.leverage}\n"
                f"  진입: `{pos.entry_price:.4f}`  현재: `{price:.4f}`\n"
                f"  손익: `{sign}{pnl_pct:.2f}%`  SL: `{pos.sl_price:.4f}`"
            )

        buttons = [
            [{"text": f"⏹ {pos.symbol.split('/')[0]} 청산",
              "callback_data": f"close_{pos.symbol}"}]
            for pos in positions
        ]
        for pos in positions:
            sym = pos.symbol
            self.notifier.register_callback(
                f"close_{sym}", lambda s=sym: self._cmd_close(s)
            )
        self.notifier.send_with_keyboard("\n".join(lines), buttons)

    def _cmd_watchlist(self, args: str) -> None:
        watchlist = self.scanner.get_watchlist()
        surge     = self.scanner.get_surge_symbols()
        syms = [s.split("/")[0] for s in watchlist]
        text = f"👁 *감시 중인 종목* ({len(watchlist)}개)\n"
        text += "  " + "  ".join(syms[:15])
        if len(syms) > 15:
            text += "\n  " + "  ".join(syms[15:])
        if surge:
            text += f"\n\n⚡ *급증 감지* ({len(surge)}개)\n"
            for sym, added_at in surge.items():
                text += f"  {sym.split('/')[0]} ({added_at.strftime('%H:%M')} 추가)\n"
        self.notifier.send_raw(text)

    def _cmd_close(self, args: str) -> None:
        symbol = args.strip().upper()
        if not symbol:
            self.notifier.send_raw("❌ 사용법: `/close SYMBOL`\n예) `/close APR`")
            return
        if "/" not in symbol:
            symbol = f"{symbol}/USDT:USDT"
        pos = self.rm.get_position(symbol)
        if not pos:
            self.notifier.send_raw(f"❌ `{symbol}` 포지션 없음")
            return
        try:
            # handle_action으로 전량 청산 실행
            action = {
                "action": "CLOSE_FULL",
                "symbol": symbol,
                "price":  self._get_current_price(symbol),
                "ratio":  1.0,
                "reason": "텔레그램 수동 청산",
            }
            result = self.executor.handle_action(
                action,
                entry_price=pos.entry_price,
            )
            if result and result.success:
                self._close_position_with_notify(symbol, reason="텔레그램 수동 청산")
                self.notifier.send_raw(f"✅ `{symbol}` 청산 완료")
            else:
                self.notifier.send_raw(f"❌ `{symbol}` 청산 실패")
        except Exception as e:
            logger.error(f"텔레그램 청산 오류 [{symbol}]: {e}", exc_info=True)
            self.notifier.send_raw(f"❌ 청산 오류: {e}")

    def _cmd_closeall(self, args: str) -> None:
        positions = self.rm.get_all_positions()
        if not positions:
            self.notifier.send_raw("📭 청산할 포지션 없음")
            return
        self.notifier.send_raw(f"⏹ 전체 {len(positions)}개 청산 시작...")
        for pos in list(positions):
            self._cmd_close(pos.symbol)

    def _cmd_pause(self, args: str) -> None:
        self._paused = True
        self.notifier.send_raw("⏸ 신규 진입 *일시 중단*\n`/resume` 으로 재개")
        logger.info("텔레그램: 신규 진입 일시 중단")

    def _cmd_resume(self, args: str) -> None:
        self._paused = False
        self.notifier.send_raw("▶️ 신규 진입 *재개*")
        logger.info("텔레그램: 신규 진입 재개")

    def _cmd_uncool(self, args: str) -> None:
        """특정 종목 쿨다운 해제 — 조정 후 재진입 허용."""
        symbol = args.strip().upper()
        if not symbol:
            self.notifier.send_raw("❌ 사용법: `/uncool SYMBOL`\n예) `/uncool SPACE`")
            return
        if "/" not in symbol:
            symbol = f"{symbol}/USDT:USDT"
        if symbol in self._close_cooldown:
            del self._close_cooldown[symbol]
            self.notifier.send_raw(f"🔓 `{symbol}` 쿨다운 해제 — 재진입 가능")
            logger.info(f"텔레그램: {symbol} 쿨다운 해제")
        else:
            self.notifier.send_raw(f"ℹ️ `{symbol}` 쿨다운 없음")

    # ── 청산 완료 콜백 ────────────────────────────────────────────────────────

    def _close_position_with_notify(self, symbol: str, reason: str) -> None:
        """
        강제 청산 시 로컬 포지션 제거 + DB 기록 + 텔레그램 알림.
        API 오류로 인한 강제 청산(거래소 포지션 종료 감지 등)에서 호출.
        """
        pos = self.rm.get_position(symbol)
        self.rm.close_position(symbol, reason=reason)
        logger.info(f"[{symbol}] {reason} — 로컬 상태 정리")

        if pos is None:
            return

        # 현재가 조회
        current_price = self._get_current_price(symbol)
        if current_price <= 0:
            current_price = pos.entry_price  # fallback

        # PnL 추정 (실제 체결가 없으므로 추정값)
        direction_mult = 1 if pos.direction == "LONG" else -1
        pnl_usdt = (current_price - pos.entry_price) * pos.position_size * direction_mult * pos.leverage

        from db_logger import TradeRecord
        from datetime import datetime, timezone
        trade = TradeRecord(
            symbol         = symbol,
            direction      = pos.direction,
            entry_price    = pos.entry_price,
            close_price    = current_price,
            position_size  = pos.position_size,
            leverage       = pos.leverage,
            pnl_usdt       = round(pnl_usdt, 4),
            fee_usdt       = 0.0,
            close_reason   = reason,
            close_order_id = "",
            atr_at_entry   = pos.atr_at_entry,
            confidence     = pos.confidence,
            signal_score   = getattr(pos, "signal_score", 0),
            adx_at_entry   = getattr(pos, "adx_at_entry", 0.0),
            rsi_at_entry   = getattr(pos, "rsi_at_entry", 0.0),
            volume_ratio   = getattr(pos, "volume_ratio",  0.0),
            entry_at       = getattr(pos, "entry_at", ""),
            close_at       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        )
        self.db.save_trade(trade)

        if self.cfg.notification.notify_close:
            self.notifier.send_close(trade)

        # 쿨다운 등록
        import time as _time
        self._close_cooldown[symbol] = _time.time() + self._cooldown_minutes * 60
        logger.info(f"[{symbol}] 쿨다운 등록: {self._cooldown_minutes}분")

    def _on_trade_closed(self, result: OrderResult) -> None:
        """
        order_executor 가 청산 완료 시 호출.
        DB 저장 + 텔레그램 알림 처리.
        """
        pos = self.rm.get_position(result.symbol)
        ctx = result.position_context or {}

        from datetime import datetime, timezone
        trade = TradeRecord(
            symbol         = result.symbol,
            direction      = pos.direction if pos else ctx.get("direction", "LONG" if result.side == "sell" else "SHORT"),
            entry_price    = pos.entry_price if pos else ctx.get("entry_price", 0.0),
            close_price    = result.avg_price,
            position_size  = result.amount,
            leverage       = pos.leverage if pos else ctx.get("leverage", 1),
            pnl_usdt       = result.pnl_usdt,
            fee_usdt       = result.fee_usdt,
            close_reason   = result.close_reason or "청산",
            close_order_id = result.order_id,
            atr_at_entry   = pos.atr_at_entry if pos else ctx.get("atr_at_entry", 0.0),
            confidence     = pos.confidence if pos else ctx.get("confidence", 0),
            signal_score   = getattr(pos, "signal_score", 0) if pos else 0,
            adx_at_entry   = getattr(pos, "adx_at_entry", 0.0) if pos else 0.0,
            rsi_at_entry   = getattr(pos, "rsi_at_entry", 0.0) if pos else 0.0,
            volume_ratio   = getattr(pos, "volume_ratio",  0.0) if pos else 0.0,
            entry_at       = getattr(pos, "entry_at", "") if pos else "",
            close_at       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        )
        self.db.save_trade(trade)

        if self.cfg.notification.notify_close:
            self.notifier.send_close(trade)

        # 쿨다운 등록
        import time as _time
        self._close_cooldown[result.symbol] = _time.time() + self._cooldown_minutes * 60
        logger.info(f"[{result.symbol}] 쿨다운 등록: {self._cooldown_minutes}분")

    # ── 유틸 ─────────────────────────────────────────────────────────────────

    def _get_current_price(self, symbol: str) -> float:
        """호가창 mid_price 또는 최신 캔들 종가 반환."""
        snap = self.ob.get_snapshot(symbol)
        if snap and snap.mid_price > 0:
            return snap.mid_price
        df = self.fetcher.get_df(symbol, self.cfg.strategy.entry_timeframe) if self.fetcher else None
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1])
        return 0.0

    def _save_signal_record(self, symbol: str, sig, score) -> None:
        """신호 발생 기록 DB 저장."""
        try:
            trend = sig.trend
            mom   = sig.momentum
            vol   = sig.volume
            rec = SignalRecord(
                symbol          = symbol,
                timeframe       = self.cfg.strategy.entry_timeframe,
                signal          = sig.signal,
                score           = sig.score,
                confidence      = score.confidence,
                reject_reason   = sig.reject_reason,
                trend_direction = trend.trend_direction if trend else "SIDEWAYS",
                adx             = trend.adx             if trend else 0.0,
                rsi             = mom.rsi               if mom   else 0.0,
                macd_hist       = mom.macd_hist         if mom   else 0.0,
                volume_ratio    = vol.volume_ratio       if vol   else 0.0,
            )
            self.db.save_signal(rec)
        except Exception as e:
            logger.debug(f"신호 기록 저장 실패: {e}")


# ── 순수 함수 ──────────────────────────────────────────────────────────────────

def _clone_score_with_leverage(score, leverage: int):
    """
    ScoreResult 의 leverage 필드만 교체한 새 인스턴스 반환.
    dataclasses.replace() 사용.
    """
    from dataclasses import replace
    return replace(score, leverage=leverage)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 진입점
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _verify_api_key(cfg) -> bool:
    """
    바이낸스 API 키 유효성 사전 검증.
    봇 시작 전에 호출하여 키 오류를 조기에 발견.

    체크 항목:
      1. API 키 / Secret 공백 여부
      2. 실거래 vs Demo 모드 일치 여부
      3. 실제 API 호출 (잔고 조회) 성공 여부
      4. Futures 권한 여부
    """
    import ccxt

    key    = cfg.exchange.api_key
    secret = cfg.exchange.api_secret
    import os
    demo    = os.getenv("USE_DEMO", "false").lower() == "true"
    testnet = cfg.exchange.testnet

    print("=" * 50)
    print("  바이낸스 API 키 검증 중...")
    print("=" * 50)

    # ── 1. 키 공백 체크 ───────────────────────────────
    if not key or not secret:
        print("❌ API 키 또는 Secret이 비어있습니다.")
        print("   .env 파일에서 API_KEY, API_SECRET을 확인하세요.")
        return False

    print(f"  모드    : {'Demo Trading' if demo else ('테스트넷' if testnet else '실거래')}")
    print(f"  API Key : {key[:8]}...{key[-4:]}")

    # ── 2. Demo 모드 경고 ─────────────────────────────
    if demo:
        print("  ⚠️  Demo Trading 모드입니다. 실거래 전환 시 USE_DEMO=false로 변경하세요.")

    # ── 3. 실제 API 연결 테스트 ───────────────────────
    try:
        exchange_cfg = {
            "apiKey":         key,
            "secret":         secret,
            "enableRateLimit": True,
            "options": {
                "defaultType":              "future",
                "adjustForTimeDifference":  True,
            },
        }
        if testnet:
            exchange_cfg["urls"] = {
                "api": {
                    "fapiPublic":    "https://testnet.binancefuture.com/fapi/v1",
                    "fapiPrivate":   "https://testnet.binancefuture.com/fapi/v1",
                    "fapiPublicV2":  "https://testnet.binancefuture.com/fapi/v2",
                    "fapiPrivateV2": "https://testnet.binancefuture.com/fapi/v2",
                }
            }
        if demo:
            exchange_cfg["options"]["portfolioMargin"] = False
            exchange_cfg["headers"] = {"X-MBX-APIKEY": key}
            exchange_cfg["urls"] = {
                "api": {
                    "fapiPublic":    "https://testnet.binancefuture.com/fapi/v1",
                    "fapiPrivate":   "https://testnet.binancefuture.com/fapi/v1",
                    "fapiPublicV2":  "https://testnet.binancefuture.com/fapi/v2",
                    "fapiPrivateV2": "https://testnet.binancefuture.com/fapi/v2",
                }
            }

        exchange = ccxt.binanceusdm(exchange_cfg)

        # 잔고 조회로 키 유효성 검증
        balance = exchange.fetch_balance()
        usdt = float(balance.get("USDT", {}).get("free", 0) or 0)
        print(f"  ✅ API 연결 성공")
        print(f"  잔고    : {usdt:.2f} USDT")

    except ccxt.AuthenticationError as e:
        print(f"❌ API 인증 실패: {e}")
        if not demo:
            print("   → 실거래 API 키인지 확인하세요.")
            print("   → 바이낸스 프로필 > API Management에서 발급한 키를 사용해야 합니다.")
            print("   → Demo Trading 키는 실거래에서 사용 불가합니다.")
        else:
            print("   → Demo Trading 키를 확인하세요.")
            print("   → 바이낸스 선물 > Demo Trading > API Management에서 발급한 키를 사용해야 합니다.")
        return False

    except ccxt.PermissionDenied as e:
        print(f"❌ API 권한 부족: {e}")
        print("   → API 키에서 'Enable Futures' 권한을 활성화하세요.")
        return False

    except ccxt.NetworkError as e:
        print(f"⚠️  네트워크 오류 (API 키 검증 생략): {e}")
        print("   → 인터넷 연결을 확인하세요. 봇은 계속 시작합니다.")
        return True  # 네트워크 오류는 봇 시작 막지 않음

    except Exception as e:
        err = str(e)
        if "-2015" in err:
            print(f"❌ API 권한 오류 (-2015): {e}")
            print("   → IP 제한 또는 권한 문제입니다.")
            print("   → 1. API 키 IP 제한 설정 확인 (현재 IP 등록 여부)")
            print("   → 2. Demo 키를 실거래에 사용하고 있지 않은지 확인")
            print("   → 3. Enable Futures 권한 활성화 확인")
            return False
        elif "-1021" in err:
            print(f"⚠️  서버 시간 불일치: {e}")
            print("   → Windows 시간 동기화: 설정 > 시간 및 언어 > '지금 동기화'")
            return False
        else:
            print(f"⚠️  API 검증 중 예외 발생: {e}")
            print("   → 봇은 계속 시작합니다.")
            return True

    # ── 4. Futures 권한 확인 ──────────────────────────
    try:
        exchange.fetch_positions(["BTC/USDT:USDT"])
        print(f"  ✅ Futures 권한 확인")
    except ccxt.PermissionDenied:
        print("❌ Futures 권한 없음")
        print("   → API 키에서 'Enable Futures'를 체크하세요.")
        return False
    except Exception:
        pass  # 포지션 조회 실패는 권한 이외 이유일 수 있음

    print("=" * 50)
    print("  ✅ API 키 검증 완료 — 봇을 시작합니다.")
    print("=" * 50)
    return True


def main() -> None:
    """main() — 설정 로드 후 봇 시작."""

    # 설정 로드 (유효성 검사 포함)
    try:
        cfg = get_config()
    except ValueError as e:
        print(f"[오류] 설정 유효성 검사 실패:\n{e}")
        sys.exit(1)

    # 로깅 초기화
    _setup_logging(cfg.system.log_level, cfg.system.log_file)

    # ── API 키 사전 검증 ──────────────────────────────
    if not _verify_api_key(cfg):
        print("\n봇 시작을 중단합니다. 위 오류를 해결 후 다시 실행하세요.")
        sys.exit(1)

    # SIGTERM 처리 (systemd / Docker 종료 신호)
    bot = CryptoSniperBot()

    def _handle_sigterm(signum, frame):
        logger.info("SIGTERM 수신 — 정상 종료 시작")
        bot.stop()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # 봇 시작 (블로킹)
    bot.start()


if __name__ == "__main__":
    main()