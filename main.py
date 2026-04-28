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
try:
    import pandas_ta as pta  # Squeeze/조기청산 RSI·MACD 계산용
except ImportError:
    pta = None
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

    # 기존 핸들러 제거 (재시작 시 중복 핸들러 누적 방지)
    root.handlers.clear()

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

        # Feed restarts near 15m candle boundaries can miss Binance's
        # one-shot kline close event. Defer non-initial restarts around
        # 00/15/30/45 so the existing WebSocket can receive the close.
        self._feed_restart_lock = threading.Lock()
        self._pending_feed_symbols: list[str] | None = None
        self._feed_restart_timer: threading.Timer | None = None
        self._feed_restart_guard_before_sec = 120
        self._feed_restart_guard_after_sec = 75
        self._last_closed_candles: set[tuple[str, str, str]] = set()

        # 청산 후 쿨다운 {symbol: close_time}
        # 같은 종목 재진입 시 COOLDOWN_MINUTES 경과 여부 체크
        self._close_cooldown: dict[str, float] = {}
        self._cooldown_minutes = 60   # 기본 60분 쿨다운
        # 최근 스캔 후보 심볼 집합 — 진입 허가 게이트로만 사용 (피드 재시작 안 함)
        self._scan_candidates: set[str] = set()
        # Squeeze Breakout 레이어 — 종목별 거래량 침묵 상태 추적
        # {symbol: {"silence_avg": float, "silence_bars": int, "detected_at": str}}
        self._squeeze_state: dict[str, dict] = {}

    def _desired_feed_symbols(self, watchlist: list[str] | None = None) -> list[str]:
        """
        현재 유지해야 할 목표 feed 심볼 집합 계산.

        구성:
        - 최신 watchlist
        - 열린 포지션 심볼
        - 최근 스캔 후보 심볼
        """
        base = list(watchlist) if watchlist is not None else self.scanner.get_watchlist()
        position_symbols = {pos.symbol for pos in self.rm.get_all_positions()}
        desired = list(dict.fromkeys(base + list(position_symbols) + list(self._scan_candidates)))
        return desired

    def _feed_restart_delay_seconds(self) -> int:
        """Return delay needed to avoid restarting near 15m candle close."""
        now = time.time()
        tm = time.localtime(now)
        seconds_into_15m = (tm.tm_min % 15) * 60 + tm.tm_sec
        seconds_to_next = 900 - seconds_into_15m
        if seconds_to_next == 900:
            seconds_to_next = 0

        before = self._feed_restart_guard_before_sec
        after = self._feed_restart_guard_after_sec

        if seconds_to_next <= before:
            return int(seconds_to_next + after + 10)
        if seconds_into_15m <= after:
            return int(after - seconds_into_15m + 10)
        return 0

    def _request_feed_restart(self, symbols: list[str], reason: str) -> None:
        """
        Restart feeds unless we are close to a 15m candle boundary.

        The delayed path coalesces multiple requests so scan/B-group churn cannot
        repeatedly disconnect the WebSocket around 00/15/30/45.
        """
        symbols = list(dict.fromkeys(symbols))
        with self._symbols_lock:
            has_active_feed = bool(self._subscribed_symbols)
        if not has_active_feed:
            self._start_feeds(symbols)
            return

        delay = self._feed_restart_delay_seconds()
        if delay <= 0:
            self._start_feeds(symbols)
            return

        with self._feed_restart_lock:
            self._pending_feed_symbols = symbols
            timer_alive = (
                self._feed_restart_timer is not None
                and self._feed_restart_timer.is_alive()
            )
            if timer_alive:
                logger.info(
                    f"feed restart deferred update ({reason}) -> {len(symbols)} symbols"
                )
                return

            logger.info(
                f"feed restart deferred {delay}s near 15m boundary ({reason})"
            )
            self._feed_restart_timer = threading.Timer(
                delay,
                self._run_deferred_feed_restart,
            )
            self._feed_restart_timer.daemon = True
            self._feed_restart_timer.start()

    def _run_deferred_feed_restart(self) -> None:
        """Run the latest deferred feed restart when the candle boundary is safe."""
        with self._feed_restart_lock:
            symbols = self._pending_feed_symbols
            self._pending_feed_symbols = None
            self._feed_restart_timer = None

        if not symbols or not self._running:
            return

        delay = self._feed_restart_delay_seconds()
        if delay > 0:
            self._request_feed_restart(symbols, "deferred_retry")
            return

        logger.info(f"deferred feed restart executing: {len(symbols)} symbols")
        self._start_feeds(symbols)

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

        # 3. 거래소 열린 포지션 복구 (재시작 시 기존 포지션 동기화)
        self._restore_positions()

        # 4. 관심 심볼 기반 초기 스캔 (30개 대상)
        logger.info("초기 코인 스캔 시작...")
        candidates = self._run_scan()
        # 수정안 2: 후보 유무와 관계없이 watchlist 전체를 기본 구독
        # scanner 후보는 우선순위 참고용이지, 감시 범위를 축소하지 않음
        watchlist_symbols = self.scanner.get_watchlist() or ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        # 복구된 포지션 심볼도 피드에 포함
        restored_symbols = {pos.symbol for pos in self.rm.get_all_positions()}
        watchlist_symbols = list(set(watchlist_symbols) | restored_symbols)
        if not candidates:
            logger.warning("초기 스캔에서 후보 코인 없음 — 관심 심볼 전체로 시작")
        else:
            logger.info(f"초기 후보 {len(candidates)}개 — watchlist 전체({len(watchlist_symbols)}개) 구독")
            # 초기 후보를 진입 허가 게이트에 등록
            self._scan_candidates = {c["symbol"] for c in candidates}
        self._start_feeds(watchlist_symbols)

        # 5. 스케줄러 등록
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

        with self._feed_restart_lock:
            if self._feed_restart_timer and self._feed_restart_timer.is_alive():
                self._feed_restart_timer.cancel()
            self._feed_restart_timer = None
            self._pending_feed_symbols = None

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

        candle_ts = str(df.index[-1]) if df is not None and not df.empty else ""
        candle_key = (symbol, timeframe, candle_ts)
        with self._processing_lock:
            if candle_key in self._last_closed_candles:
                logger.debug(f"[{symbol}] duplicate {timeframe} candle ignored: {candle_ts}")
                return
            self._last_closed_candles.add(candle_key)
            if len(self._last_closed_candles) > 5000:
                self._last_closed_candles = set(list(self._last_closed_candles)[-2500:])

        logger.info(f"[{symbol}] 15m 캔들 닫힘 감지 → 신호 판단 시작")

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

        # ── Step 0. 스캔 후보 게이트 ─────────────────────────────────────────
        # 포지션 없는 심볼은 최근 스캔에서 후보로 선정된 경우에만 진입 판단 진행
        # → 구독은 watchlist 전체를 유지하고 진입은 후보만 허용
        # _scan_candidates가 비어있으면(초기 스캔 전) 전종목 허용
        if (
            self._scan_candidates                     # 후보 목록이 확정됐고
            and symbol not in self._scan_candidates   # 이 심볼이 후보 밖이고
            and not self.rm.get_position(symbol)      # 포지션도 없으면
        ):
            logger.info(f"[{symbol}] 스캔 후보 아님 — 진입 스킵")  # debug→info로 변경
            return

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

            from execution.order_executor import _recalc_levels

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

    def _surge_scan_loop(self) -> None:
        """
        1분마다 실행 — 거래량 급증 종목 감지 후 즉시 스캔 트리거.

        기존 정기 스캔(5분)과 별개로 급증 감지 시 즉시 해당 종목의
        신호를 체크하여 급등 초입 포착 타이밍을 앞당김.

        흐름:
          detect_volume_surge() → 새로 감지된 종목 필터링
          → _start_feeds()에 추가 (WebSocket 구독)
          → 각 종목 즉시 _process_signal() 호출
        """
        try:
            new_surges = self.scanner.detect_volume_surge()
            if not new_surges:
                return

            # 이미 watchlist에 있는 종목 제외 (새로 감지된 것만)
            current = set(self.scanner.get_watchlist())
            truly_new = [s for s in new_surges if s not in current]
            if not truly_new:
                return

            logger.info(f"⚡ 급증 감지 → 즉시 스캔: {[s.split('/')[0] for s in truly_new]}")

            # 피드 구독 추가
            all_symbols = list(set(self.scanner.get_watchlist()) |
                               set(new_surges) |
                               {pos.symbol for pos in self.rm.get_all_positions()})
            self._request_feed_restart(all_symbols, "surge_scan")

            # 즉시 신호 체크 (캔들 닫힘 기다리지 않고 현재 데이터로)
            for symbol in truly_new:
                try:
                    self._process_signal(symbol)
                except Exception as e:
                    logger.debug(f"[{symbol}] 급증 즉시 스캔 오류: {e}")

        except Exception as e:
            logger.error(f"급증 스캔 루프 오류: {e}", exc_info=True)

    def _refresh_watchlist_b(self) -> None:
        """B군 watchlist 30분마다 갱신 — 신규 심볼 있을 때만 feed 재시작.

        기존 feed 심볼 vs 새 watchlist를 비교해 신규 심볼이 생긴 경우에만
        _start_feeds 재시작. 없으면 재시작 없이 candidates 갱신만.
        → 5분마다 재시작하던 것과 달리 최대 30분에 1회, 실제 신규 때만 재시작.
        """
        try:
            if not hasattr(self.scanner, "refresh_watchlist_b"):
                return
            new_watchlist = self.scanner.refresh_watchlist_b()
            desired_feed = set(self._desired_feed_symbols(new_watchlist))

            with self._symbols_lock:
                current_feed = set(self._subscribed_symbols)

            added = desired_feed - current_feed
            removed = current_feed - desired_feed
            if added or removed:
                logger.info(
                    f"B군 갱신 → feed 재구독 ({len(desired_feed)}개)  "
                    f"추가={ [s.split('/')[0] for s in sorted(added)] if added else [] }  "
                    f"제거={ [s.split('/')[0] for s in sorted(removed)] if removed else [] }"
                )
                self._request_feed_restart(sorted(desired_feed), "watchlist_b_refresh")
            else:
                logger.info(
                    f"B군 갱신 완료: watchlist={len(new_watchlist)}개 "
                    f"(변경 없음 — 피드 유지)"
                )
        except Exception as e:
            logger.error(f"B군 갱신 오류: {e}", exc_info=True)

    def _restore_positions(self) -> None:
        """
        봇 재시작 시 거래소에서 열린 포지션을 조회하여 RiskManager에 복구.

        왜 필요한가:
          봇 재시작 시 RiskManager 메모리가 초기화됨
          → 기존 포지션을 봇이 모르는 상태
          → TP/SL 체결 시 DB 기록 안 됨, 텔레그램 알림 안 됨
          → 같은 종목 중복 진입 가능

        복구 방식:
          거래소 fetch_positions() → size > 0 포지션 → Position 객체 생성
          단, SL/TP 가격은 거래소 open orders에서 조회
          알 수 없는 값(confidence, atr 등)은 기본값으로 채움
        """
        logger.info("=== 거래소 포지션 복구 시작 ===")
        try:
            positions = self.executor.exchange.fetch_positions()
        except Exception as e:
            logger.warning(f"포지션 복구 실패 (거래소 조회 오류): {e}")
            return

        restored = 0
        for pos_info in positions:
            size = float(pos_info.get("contracts", 0) or 0)
            if size <= 0:
                continue

            symbol     = pos_info.get("symbol", "")
            side       = pos_info.get("side", "long")
            direction  = "LONG" if side == "long" else "SHORT"
            entry_price = float(pos_info.get("entryPrice") or
                                pos_info.get("info", {}).get("entryPrice", 0) or 0)
            # 레버리지: ccxt는 info.leverage에 실제 값이 있음
            _lev_raw = (
                pos_info.get("info", {}).get("leverage") or
                pos_info.get("leverage") or 1
            )
            leverage = max(1, int(float(_lev_raw)))
            notional   = float(pos_info.get("notional") or
                               pos_info.get("info", {}).get("notional", 0) or 0)

            if entry_price <= 0:
                logger.warning(f"[{symbol}] 포지션 복구 스킵: 진입가 없음")
                continue

            # 미체결 주문에서 SL/TP 가격 조회
            # 바이낸스 선물의 Stop/TP 주문은 일반 open_orders가 아닌
            # 조건부 주문(conditional orders)으로 분류됨
            # → params={"type": "future"} 추가 또는 fapiPrivate API 직접 호출
            sl_price = tp1_price = tp2_price = 0.0
            try:
                # 방법 1: 일반 미체결 주문 조회
                open_orders = self.executor.exchange.fetch_open_orders(symbol)
                # 방법 2: 바이낸스 선물 전용 조건부 주문 조회 (Stop/TP)
                # 바이낸스 선물 심볼 변환: "NAORIS/USDT:USDT" → "NAORIUSUSDT"
                # ccxt의 market_id 사용이 가장 정확
                try:
                    market = self.executor.exchange.market(symbol)
                    bn_symbol = market.get("id", symbol.split("/")[0] + "USDT")
                except Exception:
                    bn_symbol = symbol.split("/")[0] + "USDT"

                try:
                    cond_orders = self.executor.exchange.fapiPrivateGetOpenOrders(
                        {"symbol": bn_symbol}
                    )
                    logger.debug(
                        f"[{symbol}] 조건부 주문 조회: {len(cond_orders) if isinstance(cond_orders, list) else type(cond_orders)} "
                        f"(심볼={bn_symbol})"
                    )
                    if isinstance(cond_orders, list):
                        for raw in cond_orders:
                            open_orders.append({
                                "type": raw.get("type", "").lower(),
                                "stopPrice": raw.get("stopPrice", 0),
                                "info": raw,
                            })
                except Exception as ce:
                    logger.debug(f"[{symbol}] 조건부 주문 조회 실패: {ce}")

                tp_prices = []
                logger.debug(f"[{symbol}] 전체 주문 {len(open_orders)}건 파싱 시작")
                for order in open_orders:
                    order_type = str(order.get("type", "")).lower()
                    stop_price = float(
                        order.get("stopPrice") or
                        order.get("info", {}).get("stopPrice") or 0
                    )
                    logger.debug(
                        f"[{symbol}] 주문: type={order_type} stopPrice={stop_price}"
                    )
                    if stop_price <= 0:
                        continue
                    if "stop_market" in order_type or order_type == "stop":
                        sl_price = stop_price
                    elif "take_profit" in order_type:
                        tp_prices.append(stop_price)

                # TP 가격 정렬 (LONG: 낮은게 TP1, 높은게 TP2 / SHORT: 반대)
                tp_prices.sort(reverse=(direction == "SHORT"))
                if len(tp_prices) >= 1:
                    tp1_price = tp_prices[0]
                if len(tp_prices) >= 2:
                    tp2_price = tp_prices[1]

            except Exception as e:
                logger.warning(f"[{symbol}] 미체결 주문 조회 실패: {e}")

            # SL/TP 없으면 ATR 기반 추정 (안전 기본값)
            if sl_price <= 0:
                sl_price = entry_price * (0.97 if direction == "LONG" else 1.03)
                logger.warning(f"[{symbol}] SL 정보 없음 — 기본값 사용: {sl_price:.5f}")
            if tp1_price <= 0:
                tp1_price = entry_price * (1.06 if direction == "LONG" else 0.94)
            if tp2_price <= 0:
                tp2_price = entry_price * (1.12 if direction == "LONG" else 0.88)

            # Position 객체 생성 후 RiskManager에 등록
            from risk.risk_manager import Position, PositionState
            import time as _t
            pos = Position(
                symbol        = symbol,
                direction     = direction,
                entry_price   = entry_price,
                position_size = size,
                leverage      = leverage,
                sl_price      = round(sl_price, 6),
                tp1_price     = round(tp1_price, 6),
                tp2_price     = round(tp2_price, 6),
                trailing_high = entry_price,
                state         = PositionState.OPEN,
                confidence    = 0,    # 알 수 없음
                atr_at_entry  = 0.0,  # 알 수 없음
                entry_at      = "",
            )
            self.rm._positions[symbol] = pos
            restored += 1

            logger.info(
                f"[{symbol}] 포지션 복구 완료  "
                f"{direction} x{leverage}  "
                f"진입가={entry_price:.5f}  size={size:.4f}  "
                f"SL={sl_price:.5f}  TP1={tp1_price:.5f}  TP2={tp2_price:.5f}"
            )
            if self.cfg.notification.telegram_token:
                self.notifier.send_raw(
                    f"♻️ 포지션 복구: *{symbol.split('/')[0]}* {direction} x{leverage}\n"
                    f"  진입가: `{entry_price:.5f}`  SL: `{sl_price:.5f}`"
                )

        if restored == 0:
            logger.info("=== 복구할 포지션 없음 ===")
        else:
            logger.info(f"=== 포지션 복구 완료: {restored}개 ===")

    def _refresh_watchlist(self) -> None:
        """
        관심 심볼 30개 갱신 — 자정마다 스케줄러 호출.
        REST API 대량 호출(수백 회)은 오직 이 메서드에서만 발생.
        신규 심볼이 있을 때만 feed 재시작 (B군 갱신과 동일한 전략).
        """
        try:
            new_watchlist = self.scanner.refresh_watchlist()
            desired_feed = set(self._desired_feed_symbols(new_watchlist))

            with self._symbols_lock:
                current_feed = set(self._subscribed_symbols)

            added = desired_feed - current_feed
            removed = current_feed - desired_feed
            if added or removed:
                logger.info(
                    f"관심 심볼 갱신 → feed 재구독 ({len(desired_feed)}개)  "
                    f"추가={ [s.split('/')[0] for s in sorted(added)] if added else [] }  "
                    f"제거={ [s.split('/')[0] for s in sorted(removed)] if removed else [] }"
                )
                self._request_feed_restart(sorted(desired_feed), "watchlist_refresh")
            else:
                logger.info(
                    f"관심 심볼 갱신 완료: {len(new_watchlist)}개 "
                    f"(변경 없음 — 피드 유지)"
                )

            if self.cfg.notification.telegram_token:
                self.notifier.send_raw(
                    f"🔄 관심 심볼 갱신 완료 ({len(new_watchlist)}개)",
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

    def _squeeze_scan(self) -> None:
        """
        Squeeze Breakout 조기 감지 레이어 — 5분마다 실행 (Shadow Mode).

        현재 방식(15m 캔들 닫힘 후 감지)보다 선행해서 상승 초입을 포착.
        실주문은 없고 DB에만 기록 → 기존 엔진 결과와 비교 분석용.

        필수 필터 (하나라도 실패 시 즉시 제외):
          - 양봉 여부
          - EMA 정배열: MA7 >= MA25
          - MACD hist > 0 (방향성 확인)
          - RSI < 70 (과열 구간 완전 차단)

        점수 계산 (기준선 50점):
          +40  거래량 스파이크 (필수)
          +15  MA 수렴 (ATR×0.8 이내)
          +20  StochRSI 과매도 반등 (K < 20 → 상승)
          +20  박스 상단 돌파
          +15  호가창 ask 감소
          +0~10 침묵 지속 보너스
          +15  황금 패턴 (RSI<63 + StochRSI반등 + MA수렴 동시)
          -10  RSI 60~63
          -40  RSI 63~70

        돌파✗ 종목: 30분간 박스 돌파 모니터링 (pending)
        돌파 확인 시: [SQUEEZE:진입트리거] 로그 + actual_result 업데이트
        """
        if self.fetcher is None:
            return

        watchlist = self.scanner.get_watchlist() if hasattr(self.scanner, "get_watchlist") else []
        if not watchlist:
            return

        tf_5m = self.cfg.strategy.confirm_timeframe  # "5m"
        detected = []
        import time as _time
        now_ts = _time.time()

        # ── Pending 모니터링: 스파이크 감지 후 박스 돌파 대기 종목 체크 ─────
        for p_sym, p_state in list(self._squeeze_state.items()):
            pending = p_state.get("pending")
            if not pending:
                continue
            if now_ts > pending["expires_at"]:
                logger.debug(f"[SQUEEZE:만료] {p_sym.split('/')[0]} — 30분 내 돌파 없음")
                p_state.pop("pending", None)
                continue
            try:
                p_df = self.fetcher.get_df(p_sym, tf_5m)
                if p_df is None or len(p_df) < 5:
                    continue
                p_close = float(p_df["close"].iloc[-1])
                p_box_high = pending["box_high_at_detect"]
                p_elapsed_min = round((now_ts - pending["detect_ts"]) / 60)
                if p_close > p_box_high * 1.002:
                    pct_from_detect = round(
                        (p_close - pending["price_at_detect"])
                        / pending["price_at_detect"] * 100, 2
                    )
                    logger.info(
                        f"[SQUEEZE:진입트리거] {p_sym.split('/')[0]}  "
                        f"경과={p_elapsed_min}분  "
                        f"감지가={pending['price_at_detect']:.6f}  "
                        f"현재가={p_close:.6f}  "
                        f"+{pct_from_detect:.2f}%  "
                        f"score={pending['score_at_detect']}"
                    )
                    self._update_squeeze_trigger(p_sym, pending, p_close, pct_from_detect, p_elapsed_min)
                    p_state.pop("pending", None)
            except Exception as e:
                logger.debug(f"[SQUEEZE] {p_sym} pending 체크 오류: {e}")

        for symbol in watchlist:
            try:
                df = self.fetcher.get_df(symbol, tf_5m)
                if df is None or len(df) < 30:
                    continue

                closes  = df["close"].values
                volumes = df["volume"].values
                highs   = df["high"].values
                lows    = df["low"].values

                # ── 1순위: 거래량 침묵 → 첫 스파이크 ────────────────────────
                silence_vols = volumes[-26:-1]
                current_vol  = volumes[-1]

                nonzero = silence_vols[silence_vols > 0]
                if len(nonzero) < 10:
                    continue
                silence_avg = float(nonzero.mean())
                if silence_avg <= 0:
                    continue

                all_vols_avg = float(volumes[-50:].mean()) if len(volumes) >= 50 else silence_avg
                spike_mult   = 2.0 if silence_avg < all_vols_avg * 0.6 else 2.5
                is_spike     = current_vol >= silence_avg * spike_mult
                silence_bars = int((silence_vols < silence_avg * 0.7).sum())

                if not is_spike or silence_bars < 8:
                    self._squeeze_state[symbol] = {
                        "silence_avg":  silence_avg,
                        "silence_bars": silence_bars,
                        "spike_mult":   spike_mult,
                    }
                    continue

                # ── 필수 필터 1: 양봉 ────────────────────────────────────────
                open_prices = df["open"].values
                if not (closes[-1] > open_prices[-1]):
                    continue

                # ── 필수 필터 2: EMA 정배열 (MA7 >= MA25) ────────────────────
                close_s = pd.Series(closes)
                ma7  = close_s.rolling(7).mean().iloc[-1]
                ma25 = close_s.rolling(25).mean().iloc[-1]
                if ma7 < ma25:
                    logger.debug(
                        f"[SQUEEZE] {symbol.split('/')[0]} MA 역배열 차단 "
                        f"(MA7={ma7:.4f} < MA25={ma25:.4f})"
                    )
                    continue

                # ── 필수 필터 3: MACD hist 증가 추세 ────────────────────────────
                # hist > 0 대신 '현재 hist > 직전 hist' (방향성 기준)
                # → 음수→양수 전환 첫 캔들도 포착 가능 (PRL/DAM 케이스 대응)
                # 단, 강한 음수(-ATR의 1% 이하)이면 차단
                macd_ok = False
                try:
                    if pta:
                        macd_df = pta.macd(close_s, fast=12, slow=26, signal=9)
                        if macd_df is not None and len(macd_df) >= 2:
                            hist_col = [c for c in macd_df.columns if "h" in c.lower()]
                            if hist_col:
                                macd_hist_now  = float(macd_df[hist_col[0]].iloc[-1])
                                macd_hist_prev = float(macd_df[hist_col[0]].iloc[-2])
                                hist_rising    = macd_hist_now > macd_hist_prev
                                # 강한 음수 차단: hist < -(현재가×0.005) 이면 하락 모멘텀 강함
                                price_threshold = float(closes[-1]) * 0.005
                                strong_negative = macd_hist_now < -price_threshold
                                macd_ok = hist_rising and not strong_negative
                except Exception:
                    macd_ok = True  # 계산 실패 시 통과
                if not macd_ok:
                    logger.debug(
                        f"[SQUEEZE] {symbol.split('/')[0]} MACD 방향성 없음 — 차단"
                    )
                    continue

                # ── ATR 계산 ────────────────────────────────────────────────
                tr = [max(highs[i] - lows[i],
                          abs(highs[i] - closes[i-1]),
                          abs(lows[i]  - closes[i-1]))
                      for i in range(-15, -1)]
                atr = sum(tr) / len(tr) if tr else 0.001

                # ── 2순위: MA 수렴도 (ATR×0.8 이내) ─────────────────────────
                ma_gap = abs(ma7 - ma25)
                ma_convergence = ma_gap / atr if atr > 0 else 99
                is_converged   = ma_convergence < 0.8

                # ── 2.5순위: StochRSI 과매도 반등 ────────────────────────────
                stoch_recovering = False
                try:
                    if pta:
                        stoch_df = pta.stochrsi(close_s, length=14, rsi_length=14, k=3, d=3)
                        if stoch_df is not None and not stoch_df.empty:
                            k_col = [c for c in stoch_df.columns if "k" in c.lower()]
                            if k_col:
                                sk_now  = float(stoch_df[k_col[0]].iloc[-1])
                                sk_prev = float(stoch_df[k_col[0]].iloc[-2])
                                stoch_recovering = sk_prev < 20 and sk_now > sk_prev
                except Exception:
                    stoch_recovering = False

                # ── 3순위: 박스 상단 이탈 ────────────────────────────────────
                box_high      = float(max(highs[-21:-1]))
                current_close = float(closes[-1])
                is_breakout   = current_close > box_high * 1.002

                # ── 4순위: 호가창 ask 감소 ───────────────────────────────────
                ask_shrinking = False
                if hasattr(self, "ob") and self.ob is not None:
                    try:
                        snap = self.ob.get_snapshot(symbol)
                        if snap is not None:
                            asks_raw = getattr(snap, "asks", None)
                            if asks_raw is not None and len(asks_raw) > 0:
                                try:
                                    ask_total = sum(
                                        (item[1] if isinstance(item, (list, tuple))
                                         else item.size)
                                        for item in asks_raw[:10]
                                    )
                                except Exception:
                                    ask_total = 0.0
                                if ask_total > 0:
                                    prev_state = self._squeeze_state.get(symbol, {})
                                    prev_ask   = prev_state.get("prev_ask_total", ask_total)
                                    ask_shrinking = ask_total < prev_ask * 0.90
                                    if symbol not in self._squeeze_state:
                                        self._squeeze_state[symbol] = {}
                                    self._squeeze_state[symbol]["prev_ask_total"] = ask_total
                    except Exception as e:
                        logger.debug(f"[SQUEEZE] {symbol} ask 조회 실패: {e}")

                # ── 필수 필터 4: RSI 계산 + RSI≥70 완전 차단 ────────────────
                rsi_s   = pta.rsi(close_s, length=14) if pta else None
                rsi_now = float(rsi_s.iloc[-1]) if rsi_s is not None else 50.0
                if rsi_now >= 70:
                    logger.debug(f"[SQUEEZE] {symbol.split('/')[0]} RSI={rsi_now:.1f}≥70 차단")
                    continue

                # ── 종합 점수 계산 ────────────────────────────────────────────
                score = 0
                score += 40                            # 스파이크 (필수)
                score += 15 if is_converged   else 0   # MA 수렴
                score += 20 if stoch_recovering else 0  # StochRSI 과매도 반등
                score += 20 if is_breakout    else 0   # 박스 돌파
                score += 15 if ask_shrinking  else 0   # ask 감소
                score += min(silence_bars // 4, 10)    # 침묵 보너스

                # RSI 패널티
                if rsi_now >= 63:
                    score -= 40   # RSI 63~70 강화 패널티
                elif rsi_now >= 60:
                    score -= 10   # RSI 60~63 경계 패널티

                # 황금 패턴 보너스: RSI 양호 + StochRSI 반등 + MA 수렴 동시 충족
                if rsi_now < 63 and stoch_recovering and is_converged:
                    score += 15
                    logger.debug(
                        f"[SQUEEZE] {symbol.split('/')[0]} 황금 패턴 +15 "
                        f"(RSI={rsi_now:.1f} + StochRSI 반등 + MA 수렴)"
                    )

                # 돌파 여부 태그
                breakout_tag = "돌파" if is_breakout else "초입"

                MIN_SCORE = 50
                import datetime as _dt
                now_str = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

                result = {
                    "symbol":           symbol,
                    "score":            score,
                    "silence_bars":     silence_bars,
                    "silence_avg":      round(silence_avg, 2),
                    "current_vol":      round(float(current_vol), 2),
                    "spike_ratio":      round(float(current_vol) / silence_avg, 2),
                    "ma_convergence":   round(ma_convergence, 2),
                    "is_converged":     is_converged,
                    "is_breakout":      is_breakout,
                    "ask_shrinking":    ask_shrinking,
                    "rsi":              round(rsi_now, 1),
                    "stoch_recovering": stoch_recovering,
                    "breakout_tag":     breakout_tag,
                    "macd_ok":          macd_ok,
                    "box_high":         round(box_high, 6),
                    "price_at_detect":  round(float(closes[-1]), 6),
                    "detected_at":      now_str,
                    "passed":           score >= MIN_SCORE,
                }
                detected.append(result)

                if score >= MIN_SCORE:
                    # 30분 쿨다운 체크
                    last_rec = self._squeeze_state.get(symbol, {}).get("last_recorded", 0)
                    if _time.time() - last_rec < 1800:
                        logger.debug(f"[SQUEEZE] {symbol.split('/')[0]} 쿨다운 — 스킵")
                    else:
                        logger.info(
                            f"[SQUEEZE:{breakout_tag}] {symbol.split('/')[0]}  "
                            f"score={score}  spike={result['spike_ratio']:.1f}×  "
                            f"silence={silence_bars}봉  RSI={rsi_now:.1f}  "
                            f"stoch_rec={stoch_recovering}  "
                            f"converged={is_converged}  breakout={is_breakout}  "
                            f"ask_shrink={ask_shrinking}"
                        )
                        self._save_squeeze_signal(result)
                        if symbol not in self._squeeze_state:
                            self._squeeze_state[symbol] = {}
                        self._squeeze_state[symbol]["last_recorded"] = _time.time()

                        # pending 등록: 돌파✗ 종목만 30분 모니터링
                        if not is_breakout:
                            self._squeeze_state[symbol]["pending"] = {
                                "price_at_detect":    result["price_at_detect"],
                                "box_high_at_detect": result["box_high"],
                                "detected_at":        now_str,
                                "detect_ts":          _time.time(),
                                "score_at_detect":    score,
                                "expires_at":         _time.time() + 14400,  # 4시간
                            }
                            logger.info(
                                f"[SQUEEZE:대기] {symbol.split('/')[0]} — "
                                f"박스 돌파 모니터링 시작 "
                                f"(기준선={result['box_high']:.6f}  "
                                f"감지가={result['price_at_detect']:.6f}  만료=4시간)"
                            )
                else:
                    logger.debug(f"[SQUEEZE-WEAK] {symbol.split('/')[0]}  score={score}")

                # 상태 갱신
                if symbol not in self._squeeze_state:
                    self._squeeze_state[symbol] = {}
                self._squeeze_state[symbol].update({
                    "silence_avg":  silence_avg,
                    "silence_bars": silence_bars,
                    "spike_mult":   spike_mult,
                })

            except Exception as e:
                logger.debug(f"[SQUEEZE] {symbol} 계산 오류: {e}")

        if detected:
            passed = [r for r in detected if r["passed"]]
            logger.info(
                f"[SQUEEZE] 스캔 완료: {len(detected)}개 스파이크 감지, "
                f"{len(passed)}개 기준 통과"
            )


    def _update_squeeze_trigger(
        self,
        symbol: str,
        pending: dict,
        trigger_price: float,
        pct_from_detect: float,
        elapsed_min: int,
    ) -> None:
        """박스 돌파 확인 시 squeeze_signals.actual_result 업데이트."""
        import json as _json, sqlite3
        actual = {
            "trigger_price":      round(trigger_price, 6),
            "price_at_detect":    pending["price_at_detect"],
            "pct_from_detect":    pct_from_detect,
            "elapsed_min":        elapsed_min,
            "breakout_confirmed": True,
        }
        try:
            conn = sqlite3.connect(self.cfg.system.db_path)
            cur  = conn.cursor()
            # actual_result 컬럼이 없으면 추가
            try:
                cur.execute("ALTER TABLE squeeze_signals ADD COLUMN actual_result TEXT DEFAULT ''")
                conn.commit()
            except Exception:
                pass  # 이미 존재하면 무시
            cur.execute("""
                UPDATE squeeze_signals
                SET    actual_result = ?
                WHERE  symbol      = ?
                  AND  detected_at = ?
                  AND  (actual_result IS NULL OR actual_result = '')
            """, (_json.dumps(actual, ensure_ascii=False), symbol, pending["detected_at"]))
            conn.commit()
            conn.close()
            logger.info(
                f"[SQUEEZE] {symbol.split('/')[0]} actual_result 업데이트 "
                f"(+{pct_from_detect:.2f}%  {elapsed_min}분 후 돌파)"
            )
        except Exception as e:
            logger.warning(f"[SQUEEZE] {symbol} actual_result 업데이트 실패: {e}")

    def _save_squeeze_signal(self, result: dict) -> None:
        """Squeeze Breakout 감지 결과를 DB에 Shadow 기록."""
        try:
            import sqlite3, json
            db_path = self.cfg.system.db_path
            conn = sqlite3.connect(db_path)
            cur  = conn.cursor()
            # 테이블 없으면 생성
            cur.execute("""
                CREATE TABLE IF NOT EXISTS squeeze_signals (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol       TEXT    NOT NULL,
                    score        INTEGER NOT NULL,
                    silence_bars INTEGER NOT NULL,
                    spike_ratio  REAL    NOT NULL,
                    ma_convergence REAL  NOT NULL,
                    is_converged INTEGER NOT NULL,
                    is_breakout  INTEGER NOT NULL,
                    ask_shrinking INTEGER NOT NULL,
                    rsi          REAL    NOT NULL,
                    detected_at  TEXT    NOT NULL,
                    actual_result TEXT   DEFAULT ''
                )
            """)
            cur.execute("""
                INSERT INTO squeeze_signals
                  (symbol, score, silence_bars, spike_ratio,
                   ma_convergence, is_converged, is_breakout,
                   ask_shrinking, rsi, detected_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                result["symbol"],
                result["score"],
                result["silence_bars"],
                result["spike_ratio"],
                result["ma_convergence"],
                int(result["is_converged"]),
                int(result["is_breakout"]),
                int(result["ask_shrinking"]),
                result["rsi"],
                result["detected_at"],
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"[SQUEEZE] DB 기록 실패: {e}")

    def _poll_entry_candle_close(self) -> None:
        """REST fallback for 15m candle close when WebSocket close events are silent."""
        if self.fetcher is None:
            return

        timeframe = self.cfg.strategy.entry_timeframe
        symbols = set(self._scan_candidates)
        symbols.update(pos.symbol for pos in self.rm.get_all_positions())
        if not symbols:
            logger.info("15m candle poll skipped: no scan candidates")
            return

        logger.info(
            f"15m candle poll start: {len(symbols)} symbols "
            f"{[s.split('/')[0] for s in sorted(symbols)[:8]]}"
        )
        for symbol in sorted(symbols):
            try:
                self.fetcher._fetch_and_store(symbol, timeframe)
                df = self.fetcher.get_df(symbol, timeframe)
                if df is not None and not df.empty:
                    self._on_candle_close(symbol, timeframe, df)
            except Exception as e:
                logger.debug(f"[{symbol}] 15m candle poll failed: {e}")

    def _scheduled_scan(self) -> None:
        """스케줄러에서 5분마다 호출 — 후보 목록 갱신, 신규 feed 추가.

        ① _scan_candidates 갱신 (진입 허가 게이트)
        ② 후보 중 feed 밖 신규 심볼이 있으면 _start_feeds 재시작
           → watchlist 갱신(30분)이 아닌 스캔 결과로도 신규 종목 즉시 반영
           → 단, 재시작 조건: '실제 신규 심볼'이 있을 때만 (5분마다 무조건 X)
        """
        logger.info("정기 스캔 시작...")
        candidates = self._run_scan()

        new_symbols = {c["symbol"] for c in candidates} if candidates else set()

        # _scan_candidates 갱신 (진입 허가 게이트용)
        self._scan_candidates = new_symbols

        if not new_symbols:
            logger.debug("진입 후보 없음 — 캔들 닫힘 시 전 종목 HOLD")
            return

        logger.debug(
            f"진입 후보 갱신: {len(new_symbols)}개 "
            f"{[s.split('/')[0] for s in sorted(new_symbols)]}"
        )

        # 후보 중 현재 feed에 없는 신규 심볼 확인
        with self._symbols_lock:
            current_feed = set(self._subscribed_symbols)

        truly_new = new_symbols - current_feed
        if truly_new:
            # 기존 feed 유지 + 신규 심볼 추가
            merged = list(current_feed | new_symbols)
            logger.info(
                f"스캔 후보 중 feed 밖 신규 {len(truly_new)}개 → feed 추가 재구독: "
                f"{[s.split('/')[0] for s in sorted(truly_new)]}"
            )
            self._request_feed_restart(merged, "scheduled_scan_new_symbols")

    # ── 스케줄 등록 ──────────────────────────────────────────────────────────

    def _register_schedules(self) -> None:
        self.scheduler.add_job(
            self._poll_entry_candle_close,
            trigger = CronTrigger(minute="0,15,30,45", second=10, timezone="UTC"),
            id      = "entry_candle_poll",
            name    = "15m candle close REST fallback",
            misfire_grace_time = 20,
        )
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
        # Squeeze Breakout 조기 감지 — 기존 스캔과 동일 주기, Shadow Mode
        self.scheduler.add_job(
            self._squeeze_scan,
            trigger = IntervalTrigger(seconds=self.cfg.scanner.scan_interval),
            id      = "squeeze_scan",
            name    = "Squeeze Breakout 조기 감지 (Shadow)",
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

        # ── 30분마다: B군 watchlist 갱신 (지금 움직이는 종목 교체)
        # 60분 → 30분으로 단축: 급등 초입 종목을 더 빠르게 watchlist에 포함
        self.scheduler.add_job(
            self._refresh_watchlist_b,
            trigger = IntervalTrigger(minutes=30),
            id      = "watchlist_b_refresh",
            name    = "B군 watchlist 갱신",
            misfire_grace_time = 30,
        )

        # ── 1분마다: 거래량 급증 감지 → 즉시 해당 종목 스캔 트리거
        # ★ 비활성화 유지: DataFetcher 연속 재시작 유발 문제로 비활성화
        #   (피드 안정성 > 급증 포착 우선 — 재활성화 시 아래 주석 해제)
        # self.scheduler.add_job(
        #     self._surge_scan_loop,
        #     trigger = IntervalTrigger(seconds=60),
        #     id      = "surge_scan",
        #     name    = "급증 감지 스캔",
        #     misfire_grace_time = 10,
        # )

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

    def _check_early_exit(self, pos, price: float) -> bool:
        """
        조기 청산 트리거 — TP1 미달 하락 패턴 대응.

        두 가지 조건 중 하나라도 충족되면 즉시 시장가 청산.

        트리거 A — ATR 기반 동적 트레일링 스탑
          진입 후 peak_price 대비 ATR×0.5 이상 하락 시 청산.
          단, 수익 구간(현재가 > 진입가)에서만 발동.
          → 손실 구간에서는 원래 SL에 맡김 (이중 청산 방지)

        트리거 B — 5m 모멘텀 이탈
          5m RSI < 45 AND 5m 직전 캔들 종가가 2캔들 전보다 낮음
          AND 현재 수익 > 0 인 상태.
          → 단기 모멘텀이 꺾이면서 수익 구간에서 청산.

        Returns
        -------
        True  : 청산 실행됨 (호출자는 continue로 이번 pos 스킵)
        False : 청산 안 함
        """
        is_long = pos.direction == "LONG"
        in_profit = (price > pos.entry_price) if is_long else (price < pos.entry_price)

        # ── 트리거 A: ATR 기반 동적 트레일링 스탑 ──────────────────────────────
        peak_price = getattr(pos, "peak_price", 0.0)  # 구버전 포지션 호환
        if peak_price == 0.0:
            peak_price = price
            pos.peak_price = price

        if in_profit and pos.atr_at_entry > 0 and peak_price > 0:
            trail_gap = pos.atr_at_entry * 0.5   # ATR의 절반 → 변동성 대응
            if is_long:
                trail_sl = peak_price - trail_gap
                triggered = price <= trail_sl
            else:
                trail_sl = peak_price + trail_gap
                triggered = price >= trail_sl

            if triggered:
                pnl_pct = (
                    (price - pos.entry_price) / pos.entry_price * 100
                    if is_long
                    else (pos.entry_price - price) / pos.entry_price * 100
                )
                reason = (
                    f"트레일링 스탑 — 고점({peak_price:.4f}) 대비 "
                    f"ATR×0.5({trail_gap:.4f}) 이탈 | 수익 {pnl_pct:+.2f}%"
                )
                logger.info(f"[{pos.symbol}] 조기 청산(트레일링): {reason}")
                self._close_position_with_notify(pos.symbol, reason=reason)
                return True

        # ── 트리거 B: 5m 모멘텀 이탈 ───────────────────────────────────────────
        if in_profit and self.fetcher is not None:
            df_5m = self.fetcher.get_df(pos.symbol, self.cfg.strategy.confirm_timeframe)
            if df_5m is not None and len(df_5m) >= 10:
                try:
                    # RSI < 45 확인 (talib 또는 ta 라이브러리 활용)
                    import pandas_ta as pta
                    rsi_5m = pta.rsi(df_5m["close"], length=14)
                    if rsi_5m is not None and len(rsi_5m) >= 2:
                        rsi_now = rsi_5m.iloc[-1]
                        # 5m 종가 하락 추세: 최근 캔들이 2캔들 전보다 낮음
                        close_falling = df_5m["close"].iloc[-1] < df_5m["close"].iloc[-3]
                        if is_long and rsi_now < 45 and close_falling:
                            pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                            reason = (
                                f"5m 모멘텀 이탈 — RSI={rsi_now:.1f}<45 + 종가 하락 "
                                f"| 수익 {pnl_pct:+.2f}%"
                            )
                            logger.info(f"[{pos.symbol}] 조기 청산(모멘텀): {reason}")
                            self._close_position_with_notify(pos.symbol, reason=reason)
                            return True
                except Exception as e:
                    logger.debug(f"[{pos.symbol}] 5m 모멘텀 계산 실패: {e}")

        return False

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

            # ── 조기 청산 트리거 (TP1 미달 하락 패턴 대응) ──────────────────
            if self._check_early_exit(pos, price):
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
                            # handle_action → on_trade_closed 콜백에서 DB 저장됨
                            # 로컬 정리 + 쿨다운만
                            self.rm.close_position(pos.symbol, reason=action.get("reason","청산"))
                            import time as _t
                            self._close_cooldown[pos.symbol] = _t.time() + self._cooldown_minutes * 60
                            logger.info(f"[{pos.symbol}] 쿨다운 등록: {self._cooldown_minutes}분")
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
            sym   = pos.symbol.split("/")[0]

            notional = pos.entry_price * pos.position_size

            pnl_pct  = 0.0
            pnl_usdt = 0.0
            if price > 0 and pos.entry_price > 0:
                price_chg = (price - pos.entry_price) / pos.entry_price
                if pos.direction == "SHORT":
                    price_chg = -price_chg
                pnl_pct  = price_chg * pos.leverage * 100
                pnl_usdt = price_chg * notional * pos.leverage

            sign  = "+" if pnl_pct >= 0 else ""
            emoji = "🟢" if pnl_pct >= 0 else "🔴"

            lines.append(
                f"{emoji} *{sym}* {pos.direction} x{pos.leverage}\n"
                f"  진입가: `{pos.entry_price:.5f}`  현재가: `{price:.5f}`\n"
                f"  투자금: `{notional:.2f} USDT`\n"
                f"  손익률: `{sign}{pnl_pct:.2f}%`  손익금: `{sign}{pnl_usdt:.2f} USDT`\n"
                f"  SL: `{pos.sl_price:.5f}`  TP1: `{pos.tp1_price:.5f}`"
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
                # handle_action → on_trade_closed 콜백에서 이미 DB 저장됨
                # _close_position_with_notify 호출하면 DB 2중 저장 → 로컬 정리만
                self.rm.close_position(symbol, reason="텔레그램 수동 청산")
                self.notifier.send_raw(f"✅ `{symbol}` 청산 완료")
            else:
                # 거래소 주문 실패여도 로컬 포지션은 강제 정리
                self.rm.close_position(symbol, reason="텔레그램 수동 청산 (강제 정리)")
                self.notifier.send_raw(f"⚠️ `{symbol}` 청산 주문 실패 — 로컬 강제 정리")
            # 수동 청산 쿨다운 보장 — 성공/실패 무관하게 항상 등록
            import time as _t
            self._close_cooldown[symbol] = _t.time() + self._cooldown_minutes * 60
            logger.info(f"[{symbol}] 수동 청산 쿨다운 등록: {self._cooldown_minutes}분")
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
        거래소에서 SL/TP 체결로 포지션이 사라졌을 때도 여기서 처리.

        미체결 주문 취소 이유:
          SL이 체결되면 TP1/TP2 주문이 거래소에 남음.
          다음 진입 시 이 잔여 주문이 새 포지션을 의도치 않게 청산할 수 있어
          → 포지션 종료 감지 즉시 잔여 주문 전부 취소
        """
        pos = self.rm.get_position(symbol)

        # 잔여 미체결 주문 취소 (SL 체결 후 남은 TP 등)
        try:
            self.executor._cancel_active_orders(symbol)
            logger.info(f"[{symbol}] 잔여 미체결 주문 취소 완료")
        except Exception as e:
            logger.warning(f"[{symbol}] 잔여 주문 취소 중 오류: {e}")

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
        # position_size는 이미 레버리지가 반영된 계약 수량
        # → leverage를 다시 곱하면 이중 계산됨 (제거)
        pnl_usdt = (current_price - pos.entry_price) * pos.position_size * direction_mult

        from utils.db_logger import TradeRecord
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
