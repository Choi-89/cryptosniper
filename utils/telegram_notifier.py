"""
telegram_notifier.py
--------------------
Telegram Bot API 를 통해 봇 운영 상황을 실시간 알림으로 전송하는 모듈.

알림 종류:
  ENTRY      : 진입 주문 완료
  CLOSE      : 청산 완료 (TP / SL / 트레일링)
  HALT       : Circuit Breaker 차단 발동
  RELEASE    : Circuit Breaker 차단 해제
  DAILY      : 일별 성과 리포트 (자정 자동 전송)
  ERROR      : 주문 오류 / 시스템 오류
  HEARTBEAT  : 봇 생존 신호 (1시간마다)

설계 원칙:
  - 전송 실패 시 최대 3회 재시도 (지수 백오프)
  - 메시지 큐 + 백그라운드 스레드로 비동기 전송 (봇 메인 루프 블로킹 방지)
  - 같은 내용의 알림은 THROTTLE_SEC 이내 중복 전송 차단
  - 긴급 알림(HALT / ERROR)은 큐 우선 순위 높음

의존 라이브러리:
  pip install requests
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import requests

from db_logger       import TradeRecord, DailyStats
from circuit_breaker import HaltEvent

# ── 로거 ───────────────────────────────────────────────────────────────────────
logger = logging.getLogger("telegram_notifier")


# ── 상수 ───────────────────────────────────────────────────────────────────────
TELEGRAM_API        = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_API_BASE   = "https://api.telegram.org/bot{token}"  # polling용
POLLING_TIMEOUT     = 30   # long polling 타임아웃 (초)
POLLING_INTERVAL    = 0.5  # 폴링 루프 sleep (초)
MAX_MSG_LEN    = 4096          # Telegram 메시지 최대 길이
MAX_RETRY      = 3
RETRY_DELAY    = 1.0           # 재시도 초기 대기 (초)
QUEUE_TIMEOUT  = 5             # 큐 대기 타임아웃 (초)
THROTTLE_SEC   = 30            # 동일 종류 알림 중복 차단 시간 (초)
HEARTBEAT_SEC  = 3600          # 생존 신호 간격 (초)


# ── 알림 우선순위 Enum ─────────────────────────────────────────────────────────

class NotifyLevel(int, Enum):
    LOW    = 3   # DAILY / HEARTBEAT
    NORMAL = 2   # ENTRY / CLOSE
    HIGH   = 1   # HALT / RELEASE / ERROR  (숫자 작을수록 우선 처리)


# ── 메시지 데이터클래스 ────────────────────────────────────────────────────────

@dataclass
class TelegramMessage:
    """
    전송 큐에 쌓이는 메시지 단위.

    Attributes
    ----------
    text        : 전송할 메시지 본문 (Markdown)
    level       : 우선순위
    msg_type    : 알림 종류 문자열 (중복 차단 키)
    parse_mode  : "Markdown" or "HTML"
    """
    text:       str
    level:      NotifyLevel = NotifyLevel.NORMAL
    msg_type:   str         = ""
    parse_mode: str         = "Markdown"
    extra_data: dict        = None   # 인라인 키보드 등 추가 파라미터


# ── 메인 클래스 ────────────────────────────────────────────────────────────────

class TelegramNotifier:
    """
    백그라운드 스레드 기반 Telegram 알림 발송기.

    사용 예시:
        notifier = TelegramNotifier(
            token="YOUR_BOT_TOKEN",
            chat_id="YOUR_CHAT_ID",
        )
        notifier.start()

        # 진입 알림
        notifier.send_entry(plan, actual_price=50100.0)

        # 청산 알림
        notifier.send_close(trade_record)

        # Circuit Breaker 알림 (cb 콜백으로 직접 연결)
        cb = CircuitBreaker(on_halt=notifier.send_halt)

        notifier.stop()
    """

    def __init__(
        self,
        token:   str,
        chat_id: str,
        bot_name: str = "CryptoSniper",
    ):
        """
        Parameters
        ----------
        token    : Telegram Bot API 토큰  예) "123456:ABC-DEF..."
        chat_id  : 메시지를 받을 채팅 ID  예) "-1001234567890"
        bot_name : 알림 헤더에 표시될 봇 이름
        """
        self._token    = token
        self._chat_id  = str(chat_id)
        self._bot_name = bot_name
        self._url      = TELEGRAM_API.format(token=token)

        # 우선순위 큐 (level 숫자 작을수록 먼저)
        self._queue: queue.PriorityQueue = queue.PriorityQueue()

        # 중복 차단: {msg_type: last_sent_ts}
        self._throttle: dict[str, float] = {}
        self._throttle_lock = threading.Lock()

        # 백그라운드 전송 스레드
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Heartbeat 스레드
        self._hb_thread: Optional[threading.Thread] = None

        # Polling 스레드 (명령 수신)
        self._poll_thread: Optional[threading.Thread] = None
        self._update_offset: int = 0   # getUpdates offset

        # 명령 핸들러 콜백 {command: callable}
        self._command_handlers: dict = {}
        self._callback_handlers: dict = {}  # 인라인 버튼 콜백 {data: callable}

    # ── 라이프사이클 ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """백그라운드 전송 스레드 + polling 스레드 시작."""
        self._running = True
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="tg-sender"
        )
        self._thread.start()

        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="tg-heartbeat"
        )
        self._hb_thread.start()

        # 명령 polling 스레드
        self._poll_thread = threading.Thread(
            target=self._polling_loop, daemon=True, name="tg-polling"
        )
        self._poll_thread.start()
        logger.info("TelegramNotifier polling 시작")

        logger.info(f"TelegramNotifier 시작 (bot={self._bot_name})")
        self._enqueue(TelegramMessage(
            text     = f"🤖 *{self._bot_name}* 시작\n`{_now()}`",
            level    = NotifyLevel.NORMAL,
            msg_type = "BOT_START",
        ))

    def stop(self) -> None:
        """전송 완료 후 스레드 종료."""
        logger.info("TelegramNotifier 종료 중...")
        self._running = False
        # 종료 신호용 sentinel
        self._queue.put((99, TelegramMessage(text="__STOP__")))
        if self._thread:
            self._thread.join(timeout=10)

    # ── 퍼블릭: 알림 전송 메서드 ─────────────────────────────────────────────

    def send_entry(
        self,
        symbol:       str,
        direction:    str,
        entry_price:  float,
        sl_price:     float,
        tp1_price:    float,
        tp2_price:    float,
        position_size: float,
        leverage:     int,
        confidence:   int,
        notional:     float,
    ) -> None:
        """
        진입 주문 완료 알림.

        Parameters
        ----------
        symbol, direction : 심볼 / 방향
        entry_price       : 실제 체결가
        sl_price          : 손절가
        tp1_price         : 1차 익절가
        tp2_price         : 2차 익절가
        position_size     : 포지션 크기 (코인)
        leverage          : 레버리지
        confidence        : 신뢰도 점수
        notional          : 명목 포지션 크기 (USDT)
        """
        emoji  = "🟢" if direction == "LONG" else "🔴"
        arrow  = "▲" if direction == "LONG" else "▼"
        sym    = _fmt_symbol(symbol)

        # SL/TP 거리(%)
        sl_pct  = abs(entry_price - sl_price)  / entry_price * 100
        tp1_pct = abs(tp1_price   - entry_price) / entry_price * 100
        tp2_pct = abs(tp2_price   - entry_price) / entry_price * 100

        text = (
            f"{emoji} *진입* {arrow} `{sym}`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"방향       `{direction}` × `{leverage}x`\n"
            f"진입가     `{entry_price:,.4f}`\n"
            f"손절 (SL)  `{sl_price:,.4f}`  _{sl_pct:.2f}%_\n"
            f"TP1        `{tp1_price:,.4f}`  _{tp1_pct:.2f}%_\n"
            f"TP2        `{tp2_price:,.4f}`  _{tp2_pct:.2f}%_\n"
            f"수량       `{position_size:.6f}` / `{notional:.2f} USDT`\n"
            f"신뢰도     `{confidence}/100`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"_{_now()}_"
        )
        self._enqueue(TelegramMessage(
            text     = text,
            level    = NotifyLevel.NORMAL,
            msg_type = f"ENTRY_{symbol}",
        ))

    def send_close(self, trade: TradeRecord) -> None:
        """청산 완료 알림."""
        is_profit = trade.pnl_usdt >= 0
        emoji     = "✅" if is_profit else "❌"
        pnl_sign  = "+" if is_profit else ""
        sym       = _fmt_symbol(trade.symbol)

        # 청산 사유별 이모지
        reason_emoji = {
            "TP1 도달":    "🎯",
            "TP2 도달":    "🎯🎯",
            "트레일링 발동": "🏹",
            "SL 도달":     "🛑",
        }.get(trade.close_reason, "📌")

        text = (
            f"{emoji} *청산* {reason_emoji} `{sym}`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"방향       `{trade.direction}` × `{trade.leverage}x`\n"
            f"진입가     `{trade.entry_price:,.4f}`\n"
            f"청산가     `{trade.close_price:,.4f}`\n"
            f"사유       `{trade.close_reason}`\n"
            f"손익       `{pnl_sign}{trade.pnl_usdt:.2f} USDT`"
            f"  _({pnl_sign}{trade.pnl_pct:.3f}%)_\n"
            f"수수료     `-{trade.fee_usdt:.4f} USDT`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"_{_now()}_"
        )
        self._enqueue(TelegramMessage(
            text     = text,
            level    = NotifyLevel.NORMAL,
            msg_type = f"CLOSE_{trade.symbol}",
        ))

    def send_halt(self, event: HaltEvent) -> None:
        """Circuit Breaker 차단 발동 알림 (긴급 — 우선 처리)."""
        level_emoji = {
            "COOLDOWN":    "⏸",
            "DAILY_HALT":  "🚫",
            "MANUAL_HALT": "🔴",
        }.get(event.level.value, "⚠️")

        text = (
            f"🚨 *거래 차단 발동* {level_emoji}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"레벨       `{event.level.value}`\n"
            f"사유       {event.reason}\n"
            f"일일 손실  `{event.daily_loss_pct*100:.2f}%`\n"
            f"연속 손절  `{event.consec_losses}회`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"_{_now()}_"
        )
        self._enqueue(TelegramMessage(
            text     = text,
            level    = NotifyLevel.HIGH,
            msg_type = f"HALT_{event.level.value}",
        ))

    def send_release(self, event: HaltEvent) -> None:
        """Circuit Breaker 차단 해제 알림."""
        text = (
            f"✅ *거래 차단 해제*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"레벨       `{event.level.value}`\n"
            f"_{_now()}_"
        )
        self._enqueue(TelegramMessage(
            text     = text,
            level    = NotifyLevel.HIGH,
            msg_type = "RELEASE",
        ))

    def send_daily_report(self, stats: DailyStats, capital: float = 0.0) -> None:
        """일별 성과 리포트 알림."""
        win_emoji  = "📈" if stats.total_pnl >= 0 else "📉"
        pnl_sign   = "+" if stats.total_pnl >= 0 else ""

        # 승률 바 시각화 (10칸)
        filled = round(stats.win_rate * 10)
        bar    = "█" * filled + "░" * (10 - filled)

        text = (
            f"{win_emoji} *일별 성과 리포트*\n"
            f"📅 `{stats.date_str}`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"총 트레이드  `{stats.total_trades}건`\n"
            f"승 / 패      `{stats.wins}승` / `{stats.losses}패`\n"
            f"승률         `{stats.win_rate*100:.1f}%`  `[{bar}]`\n"
            f"총 손익      `{pnl_sign}{stats.total_pnl:.2f} USDT`\n"
            f"평균 손익    `{pnl_sign}{stats.avg_pnl:.2f} USDT`\n"
            f"최고 트레이드  `+{stats.best_trade:.2f} USDT`\n"
            f"최악 트레이드  `{stats.worst_trade:.2f} USDT`\n"
            f"최대 연속 수익 `{stats.max_consec_wins}회`\n"
            f"최대 연속 손절 `{stats.max_consec_loss}회`\n"
        )
        if capital > 0:
            text += f"잔고          `{capital:.2f} USDT`\n"
        text += (
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"_{_now()}_"
        )
        self._enqueue(TelegramMessage(
            text     = text,
            level    = NotifyLevel.LOW,
            msg_type = f"DAILY_{stats.date_str}",
        ))

    def send_error(self, module: str, message: str, detail: str = "") -> None:
        """시스템 오류 알림 (긴급)."""
        text = (
            f"⚠️ *오류 발생*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"모듈    `{module}`\n"
            f"내용    {message}\n"
        )
        if detail:
            # 너무 길면 앞 200자만 표시
            snippet = detail[:200] + ("..." if len(detail) > 200 else "")
            text += f"상세    `{snippet}`\n"
        text += f"_{_now()}_"

        self._enqueue(TelegramMessage(
            text     = text,
            level    = NotifyLevel.HIGH,
            msg_type = f"ERROR_{module}",
        ))

    def send_raw(self, text: str, level: NotifyLevel = NotifyLevel.NORMAL) -> None:
        """자유 형식 메시지 전송."""
        self._enqueue(TelegramMessage(
            text  = text,
            level = level,
        ))

    # ── 내부: 큐 관리 ─────────────────────────────────────────────────────────

    def _enqueue(self, msg: TelegramMessage) -> None:
        """
        중복 차단 체크 후 큐에 삽입.
        동일 msg_type 은 THROTTLE_SEC 이내 재전송 차단.
        """
        if msg.msg_type:
            with self._throttle_lock:
                last = self._throttle.get(msg.msg_type, 0.0)
                if time.time() - last < THROTTLE_SEC:
                    logger.debug(f"중복 알림 차단: {msg.msg_type}")
                    return
                self._throttle[msg.msg_type] = time.time()

        # PriorityQueue: (우선순위, 삽입순서, 메시지)
        seq = int(time.time() * 1000)
        self._queue.put((msg.level.value, seq, msg))

    # ── 내부: 백그라운드 전송 워커 ────────────────────────────────────────────

    def _worker(self) -> None:
        """큐에서 메시지를 꺼내 순차 전송하는 백그라운드 스레드."""
        logger.debug("전송 워커 시작")
        while self._running:
            try:
                priority, _, msg = self._queue.get(timeout=QUEUE_TIMEOUT)
            except queue.Empty:
                continue

            if msg.text == "__STOP__":
                break

            self._send_with_retry(msg)
            self._queue.task_done()

        logger.debug("전송 워커 종료")

    def _send_with_retry(self, msg: TelegramMessage) -> bool:
        """
        Telegram API 호출 (재시도 포함).

        Returns
        -------
        bool : 최종 성공 여부
        """
        text = msg.text[:MAX_MSG_LEN]  # 길이 제한

        for attempt in range(1, MAX_RETRY + 1):
            try:
                payload = {
                    "chat_id":    self._chat_id,
                    "text":       text,
                    "parse_mode": msg.parse_mode,
                    "disable_web_page_preview": True,
                }
                if msg.extra_data:
                    payload.update(msg.extra_data)
                resp = requests.post(
                    self._url,
                    json=payload,
                    timeout=10,
                )
                data = resp.json()

                if resp.status_code == 200 and data.get("ok"):
                    logger.debug(
                        f"텔레그램 전송 성공 "
                        f"(attempt={attempt}, type={msg.msg_type})"
                    )
                    return True

                # 429 Too Many Requests — rate limit
                if resp.status_code == 429:
                    retry_after = data.get("parameters", {}).get("retry_after", 5)
                    logger.warning(f"텔레그램 Rate Limit — {retry_after}초 대기")
                    time.sleep(retry_after)
                    continue

                logger.error(
                    f"텔레그램 오류 응답: "
                    f"status={resp.status_code}  "
                    f"body={data}"
                )

            except requests.Timeout:
                logger.warning(f"텔레그램 타임아웃 (attempt={attempt})")
            except requests.ConnectionError as e:
                logger.warning(f"텔레그램 연결 오류 (attempt={attempt}): {e}")
            except Exception as e:
                logger.error(f"텔레그램 예상치 못한 오류: {e}", exc_info=True)
                break

            if attempt < MAX_RETRY:
                delay = RETRY_DELAY * (2 ** (attempt - 1))
                time.sleep(delay)

        logger.error(f"텔레그램 최종 전송 실패 (type={msg.msg_type})")
        return False

    # ── 퍼블릭: 명령 핸들러 등록 ─────────────────────────────────────────────

    def register_command(self, command: str, handler) -> None:
        """
        텔레그램 명령어 핸들러 등록.

        Parameters
        ----------
        command : "/status", "/positions" 등 슬래시 포함
        handler : fn(args: str) → None  (args는 명령어 뒤 텍스트)
        """
        self._command_handlers[command.lower()] = handler

    def register_callback(self, data: str, handler) -> None:
        """
        인라인 버튼 콜백 핸들러 등록.

        Parameters
        ----------
        data    : 버튼의 callback_data 값
        handler : fn() → None
        """
        self._callback_handlers[data] = handler

    def send_with_keyboard(
        self,
        text: str,
        buttons: list[list[dict]],
        level: "NotifyLevel" = None,
    ) -> None:
        """
        인라인 키보드 버튼과 함께 메시지 전송.

        Parameters
        ----------
        buttons : [[{"text": "버튼명", "callback_data": "data"}, ...], ...]
        """
        if level is None:
            level = NotifyLevel.NORMAL
        import json
        keyboard = {"inline_keyboard": buttons}
        self._enqueue(TelegramMessage(
            text       = text,
            level      = level,
            msg_type   = "keyboard",
            extra_data = {"reply_markup": json.dumps(keyboard)},
        ))

    # ── 내부: polling 루프 ────────────────────────────────────────────────────

    def _polling_loop(self) -> None:
        """getUpdates long polling — 명령 및 버튼 콜백 수신."""
        base_url = TELEGRAM_API_BASE.format(token=self._token)
        logger.debug("polling 루프 시작")

        while self._running:
            try:
                resp = requests.get(
                    f"{base_url}/getUpdates",
                    params={
                        "offset":  self._update_offset,
                        "timeout": POLLING_TIMEOUT,
                        "allowed_updates": ["message", "callback_query"],
                    },
                    timeout=POLLING_TIMEOUT + 5,
                )
                data = resp.json()

                if not data.get("ok"):
                    time.sleep(POLLING_INTERVAL)
                    continue

                for update in data.get("result", []):
                    self._update_offset = update["update_id"] + 1
                    self._handle_update(update)

            except requests.Timeout:
                pass  # long polling 정상 타임아웃
            except Exception as e:
                logger.warning(f"polling 오류: {e}")
                time.sleep(5)

    def _handle_update(self, update: dict) -> None:
        """수신된 update 처리 — 명령어 또는 버튼 콜백."""
        # ── 텍스트 명령 ───────────────────────────────────────────────────────
        message = update.get("message", {})
        text    = message.get("text", "")
        chat_id = str(message.get("chat", {}).get("id", ""))

        if text and chat_id == self._chat_id:
            parts   = text.strip().split(maxsplit=1)
            command = parts[0].lower()
            args    = parts[1] if len(parts) > 1 else ""

            # @봇이름 접미사 제거
            if "@" in command:
                command = command.split("@")[0]

            handler = self._command_handlers.get(command)
            if handler:
                try:
                    logger.info(f"명령 수신: {command} {args}")
                    handler(args)
                except Exception as e:
                    logger.error(f"명령 처리 오류 [{command}]: {e}", exc_info=True)
                    self.send_raw(f"⚠️ 명령 처리 오류: {e}", NotifyLevel.HIGH)
            elif text.startswith("/"):
                self.send_raw(
                    f"❓ 알 수 없는 명령입니다: `{command}`\n/help 로 명령어 목록을 확인하세요.",
                    NotifyLevel.NORMAL,
                )

        # ── 인라인 버튼 콜백 ──────────────────────────────────────────────────
        callback_query = update.get("callback_query", {})
        if callback_query:
            cb_chat_id = str(callback_query.get("message", {})
                             .get("chat", {}).get("id", ""))
            cb_data    = callback_query.get("data", "")
            cb_id      = callback_query.get("id", "")

            # 버튼 응답 (로딩 표시 제거)
            try:
                base_url = TELEGRAM_API_BASE.format(token=self._token)
                requests.post(
                    f"{base_url}/answerCallbackQuery",
                    json={"callback_query_id": cb_id},
                    timeout=5,
                )
            except Exception:
                pass

            if cb_chat_id == self._chat_id:
                handler = self._callback_handlers.get(cb_data)
                if handler:
                    try:
                        logger.info(f"버튼 콜백 수신: {cb_data}")
                        handler()
                    except Exception as e:
                        logger.error(f"버튼 콜백 오류 [{cb_data}]: {e}")
                        self.send_raw(f"⚠️ 처리 오류: {e}", NotifyLevel.HIGH)

    # ── 내부: 하트비트 루프 ────────────────────────────────────────────────────

    def _heartbeat_loop(self) -> None:
        """1시간마다 봇 생존 신호 전송."""
        time.sleep(HEARTBEAT_SEC)   # 시작 후 1시간 뒤 첫 전송
        while self._running:
            self._enqueue(TelegramMessage(
                text = (
                    f"💓 *{self._bot_name}* 정상 운영 중\n"
                    f"_{_now()}_"
                ),
                level    = NotifyLevel.LOW,
                msg_type = "HEARTBEAT",
            ))
            time.sleep(HEARTBEAT_SEC)


# ── 유틸 ───────────────────────────────────────────────────────────────────────

def _now() -> str:
    """현재 UTC 시각 문자열 반환."""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt_symbol(symbol: str) -> str:
    """'BTC/USDT:USDT' → 'BTC/USDT' 로 간결하게."""
    return symbol.split(":")[0]


# ── 단독 실행 테스트 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from db_logger import TradeRecord, DailyStats
    from circuit_breaker import HaltEvent, HaltLevel

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    load_dotenv()

    TOKEN   = os.getenv("TELEGRAM_TOKEN",  "")
    CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

    if not TOKEN or not CHAT_ID:
        print("❌ .env 에 TELEGRAM_TOKEN / TELEGRAM_CHAT_ID 를 설정하세요.")
        print("   TELEGRAM_TOKEN=123456:ABC-DEF...")
        print("   TELEGRAM_CHAT_ID=-1001234567890")
        exit(1)

    notifier = TelegramNotifier(
        token    = TOKEN,
        chat_id  = CHAT_ID,
        bot_name = "CryptoSniper",
    )
    notifier.start()
    time.sleep(1)

    # ── 1. 진입 알림 테스트 ────────────────────────────────────────────────────
    print("1. 진입 알림 전송...")
    notifier.send_entry(
        symbol        = "BTC/USDT:USDT",
        direction     = "LONG",
        entry_price   = 91600.0,
        sl_price      = 90100.0,
        tp1_price     = 94600.0,
        tp2_price     = 97600.0,
        position_size = 0.01,
        leverage      = 5,
        confidence    = 78,
        notional      = 916.0,
    )
    time.sleep(2)

    # ── 2. 청산 알림 테스트 ────────────────────────────────────────────────────
    print("2. 청산 알림 전송 (TP1)...")
    trade = TradeRecord(
        symbol        = "BTC/USDT:USDT",
        direction     = "LONG",
        entry_price   = 91600.0,
        close_price   = 94600.0,
        position_size = 0.005,
        leverage      = 5,
        pnl_usdt      = 15.0,
        fee_usdt      = 0.37,
        close_reason  = "TP1 도달",
        confidence    = 78,
        pnl_pct       = 3.27,
    )
    notifier.send_close(trade)
    time.sleep(2)

    # ── 3. Circuit Breaker 차단 알림 ─────────────────────────────────────────
    print("3. Circuit Breaker 차단 알림...")
    evt = HaltEvent(
        level          = HaltLevel.COOLDOWN,
        reason         = "연속 손절 3회 — 60분 쿨다운",
        triggered_at   = time.time(),
        daily_loss_pct = 0.021,
        consec_losses  = 3,
    )
    notifier.send_halt(evt)
    time.sleep(2)

    # ── 4. 일별 리포트 알림 ────────────────────────────────────────────────────
    print("4. 일별 리포트 전송...")
    stats = DailyStats(
        date_str        = "2026-04-08",
        total_trades    = 12,
        wins            = 8,
        losses          = 4,
        win_rate        = 0.667,
        total_pnl       = 47.32,
        total_fee       = 3.21,
        avg_pnl         = 3.94,
        best_trade      = 24.10,
        worst_trade     = -11.50,
        max_consec_wins = 4,
        max_consec_loss = 2,
        capital_end     = 1047.32,
    )
    notifier.send_daily_report(stats, capital=1047.32)
    time.sleep(2)

    # ── 5. 오류 알림 ──────────────────────────────────────────────────────────
    print("5. 오류 알림 전송...")
    notifier.send_error(
        module  = "order_executor",
        message = "SL 주문 등록 실패 — 긴급 청산 실행",
        detail  = "ccxt.ExchangeError: Order would immediately trigger",
    )
    time.sleep(3)

    notifier.stop()
    print("\n✅ 모든 테스트 알림 전송 완료")