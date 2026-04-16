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
        with self._symbols_lock:
            self._subscribed_symbols = list(symbols)

        timeframes = [
            self.cfg.strategy.entry_timeframe,   # 15m
            self.cfg.strategy.trend_timeframe,    # 1h
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

        # ── Step 2. 포지션 업데이트 ───────────────────────────────────────
        pos = self.rm.get_position(symbol)
        if pos:
            exchange_size = self.executor.fetch_position_size(symbol)
            if exchange_size is None:
                # API 오류 (429 등) — 포지션 없음으로 오판 방지, 이번 사이클 스킵
                logger.warning(f"[{symbol}] 포지션 조회 API 오류 — 이번 사이클 스킵")
                return
            if exchange_size <= 0:
                self.rm.close_position(symbol, reason="거래소 포지션 종료 감지")
                logger.info(f"[{symbol}] 거래소 포지션 종료 감지 — 로컬 상태 정리")
                return

            current_price = self._get_current_price(symbol)
            if current_price > 0:
                actions = self.rm.update_positions({symbol: current_price})
                for action in actions:
                    # 트레일링 스탑(CLOSE_FULL, TP2 이후)만 로컬에서 처리
                    if action.get("reason", "").startswith("트레일링"):
                        self.executor.handle_action(
                            action,
                            entry_price=pos.entry_price,
                            position_context={
                                "direction": pos.direction,
                                "entry_price": pos.entry_price,
                                "leverage": pos.leverage,
                                "atr_at_entry": pos.atr_at_entry,
                                "confidence": pos.confidence,
                            },
                        )
            return

        # ── Step 3. DataFrame 수집 ────────────────────────────────────────
        df_1h  = self.fetcher.get_df(symbol, cfg.strategy.trend_timeframe)
        df_15m = self.fetcher.get_df(symbol, cfg.strategy.entry_timeframe)

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
                # API 오류 — 포지션 없음으로 오판 방지, 이번 사이클 스킵
                logger.warning(f"[{pos.symbol}] 포지션 조회 API 오류 — 이번 사이클 스킵")
                continue
            if exchange_size <= 0:
                self.rm.close_position(pos.symbol, reason="거래소 포지션 종료 감지")
                continue
            price = self._get_current_price(pos.symbol)
            if price <= 0:
                continue
            actions = self.rm.update_positions({pos.symbol: price})
            for action in actions:
                if action.get("reason", "").startswith("트레일링"):
                    self.executor.handle_action(
                        action,
                        entry_price=pos.entry_price,
                        position_context={
                            "direction": pos.direction,
                            "entry_price": pos.entry_price,
                            "leverage": pos.leverage,
                            "atr_at_entry": pos.atr_at_entry,
                            "confidence": pos.confidence,
                        },
                    )

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

    # ── 청산 완료 콜백 ────────────────────────────────────────────────────────

    def _on_trade_closed(self, result: OrderResult) -> None:
        """
        order_executor 가 청산 완료 시 호출.
        DB 저장 + 텔레그램 알림 처리.
        """
        pos = self.rm.get_position(result.symbol)
        ctx = result.position_context or {}

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
        )
        self.db.save_trade(trade)

        if self.cfg.notification.notify_close:
            self.notifier.send_close(trade)

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