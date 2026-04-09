"""
circuit_breaker.py
------------------
일일 손실 한도 / 연속 손절 감지 / 쿨다운 타이머를 관리하는 모듈.

역할:
  1. 일일 누적 손실이 -3% 도달 시 당일 거래 전면 차단
  2. 연속 손절 3회 도달 시 1시간 거래 차단 (쿨다운)
  3. 자정 기준 일일 리셋
  4. 긴급 수동 차단 / 해제 지원
  5. 모든 차단 이벤트를 텔레그램 알림용 콜백으로 전달

차단 레벨:
  CLEAR       : 정상, 진입 가능
  COOLDOWN    : 연속 손절 쿨다운 중 (임시 차단, 자동 해제)
  DAILY_HALT  : 일일 손실 한도 초과 (자정까지 차단)
  MANUAL_HALT : 수동 긴급 차단 (명시적 해제 필요)

의존 라이브러리:
  표준 라이브러리만 사용 (외부 패키지 없음)
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Callable, Optional

# ── 로거 ───────────────────────────────────────────────────────────────────────
logger = logging.getLogger("circuit_breaker")


# ── 파라미터 상수 ──────────────────────────────────────────────────────────────
DAILY_LOSS_LIMIT_PCT   = 0.03   # 일일 손실 한도 (총 자본 대비 3%)
CONSEC_LOSS_LIMIT      = 3      # 연속 손절 허용 횟수
COOLDOWN_SECONDS       = 3600   # 쿨다운 시간 (초) — 1시간
RESET_HOUR             = 0      # 일일 리셋 시각 (0 = 자정)
RESET_MINUTE           = 0


# ── 차단 레벨 Enum ─────────────────────────────────────────────────────────────

class HaltLevel(str, Enum):
    CLEAR       = "CLEAR"        # 정상
    COOLDOWN    = "COOLDOWN"     # 연속 손절 쿨다운
    DAILY_HALT  = "DAILY_HALT"   # 일일 손실 한도 초과
    MANUAL_HALT = "MANUAL_HALT"  # 수동 긴급 차단


# ── 이벤트 데이터클래스 ────────────────────────────────────────────────────────

@dataclass
class HaltEvent:
    """
    차단 발동/해제 이벤트 기록.

    Attributes
    ----------
    level           : 차단 레벨
    reason          : 차단 사유 문자열
    triggered_at    : 발동 시각 (Unix 초)
    released_at     : 해제 시각 (None = 아직 차단 중)
    daily_loss_pct  : 발동 당시 일일 손실률
    consec_losses   : 발동 당시 연속 손절 횟수
    """
    level:          HaltLevel
    reason:         str
    triggered_at:   float
    released_at:    Optional[float] = None
    daily_loss_pct: float           = 0.0
    consec_losses:  int             = 0


@dataclass
class CircuitStatus:
    """
    is_blocked() 호출 시 반환하는 현재 상태 스냅샷.

    Attributes
    ----------
    blocked         : 거래 차단 여부
    level           : 현재 차단 레벨
    reason          : 차단 사유 (CLEAR 이면 빈 문자열)
    daily_loss_pct  : 현재 일일 손실률 (0.0 ~ )
    daily_loss_usdt : 현재 일일 손실 금액 (USDT)
    consec_losses   : 현재 연속 손절 횟수
    cooldown_remain : 쿨다운 잔여 시간 (초), 쿨다운 아니면 0
    today_trades    : 오늘 총 트레이드 횟수
    today_wins      : 오늘 수익 트레이드 횟수
    today_losses    : 오늘 손실 트레이드 횟수
    """
    blocked:          bool      = False
    level:            HaltLevel = HaltLevel.CLEAR
    reason:           str       = ""
    daily_loss_pct:   float     = 0.0
    daily_loss_usdt:  float     = 0.0
    consec_losses:    int       = 0
    cooldown_remain:  float     = 0.0
    today_trades:     int       = 0
    today_wins:       int       = 0
    today_losses:     int       = 0


# ── 메인 클래스 ────────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    거래 차단 조건을 감시하고 진입 허가/차단 여부를 결정하는 클래스.

    사용 예시:
        cb = CircuitBreaker(
            total_capital=1000.0,
            on_halt=lambda evt: telegram.send(f"차단: {evt.reason}"),
        )

        # 트레이드 결과 기록
        cb.record_trade(pnl_usdt=-12.5)   # 손실
        cb.record_trade(pnl_usdt= 24.0)   # 수익

        # 진입 전 반드시 체크
        status = cb.check()
        if status.blocked:
            return  # 진입 포기
    """

    def __init__(
        self,
        total_capital:   float,
        daily_loss_limit_pct: float = DAILY_LOSS_LIMIT_PCT,
        consec_loss_limit: int = CONSEC_LOSS_LIMIT,
        cooldown_seconds: int = COOLDOWN_SECONDS,
        on_halt:         Optional[Callable[[HaltEvent], None]] = None,
        on_release:      Optional[Callable[[HaltEvent], None]] = None,
    ):
        """
        Parameters
        ----------
        total_capital : 총 운용 자본 (USDT). 잔고 변경 시 update_capital() 로 갱신.
        on_halt       : 차단 발동 시 호출될 콜백  fn(HaltEvent)
        on_release    : 차단 해제 시 호출될 콜백  fn(HaltEvent)
        """
        self.total_capital = total_capital
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.consec_loss_limit = consec_loss_limit
        self.cooldown_seconds = cooldown_seconds
        self._on_halt      = on_halt
        self._on_release   = on_release

        self._lock = threading.Lock()

        # ── 일일 통계 ────────────────────────────────────────────────────────
        self._daily_loss_usdt: float = 0.0   # 오늘 누적 손실 (USDT, 항상 ≥ 0)
        self._today_trades:    int   = 0
        self._today_wins:      int   = 0
        self._today_losses:    int   = 0
        self._last_reset_date: date  = date.today()

        # ── 연속 손절 ────────────────────────────────────────────────────────
        self._consec_losses: int = 0

        # ── 차단 상태 ────────────────────────────────────────────────────────
        self._halt_level:    HaltLevel        = HaltLevel.CLEAR
        self._halt_reason:   str              = ""
        self._cooldown_until: float           = 0.0   # Unix 초
        self._current_event: Optional[HaltEvent] = None

        # ── 이벤트 히스토리 ──────────────────────────────────────────────────
        self._history: list[HaltEvent] = []

        # ── 자정 리셋 백그라운드 스레드 ─────────────────────────────────────
        self._running = True
        self._reset_thread = threading.Thread(
            target=self._midnight_reset_loop,
            daemon=True,
            name="cb-reset",
        )
        self._reset_thread.start()
        logger.info(
            f"CircuitBreaker 시작  "
            f"자본={total_capital:.2f}U  "
            f"일일손실한도={self.daily_loss_limit_pct*100:.1f}%  "
            f"연속손절한도={self.consec_loss_limit}회"
        )

    # ── 퍼블릭: 진입 전 체크 ──────────────────────────────────────────────────

    def check(self) -> CircuitStatus:
        """
        현재 거래 가능 여부를 판단하고 CircuitStatus 반환.
        진입 직전 반드시 호출해야 한다.

        Returns
        -------
        CircuitStatus
          .blocked == True  → 진입 포기
          .blocked == False → 진입 가능
        """
        with self._lock:
            self._auto_reset_if_needed()
            self._release_cooldown_if_expired()
            return self._build_status()

    def is_blocked(self) -> bool:
        """빠른 차단 여부 확인 (bool 반환)."""
        return self.check().blocked

    # ── 퍼블릭: 트레이드 결과 기록 ───────────────────────────────────────────

    def record_trade(self, pnl_usdt: float) -> CircuitStatus:
        """
        트레이드 결과를 기록하고 차단 조건을 재평가한다.
        order_executor 에서 청산 완료 후 호출.

        Parameters
        ----------
        pnl_usdt : 실현 손익 (USDT). 수익 = 양수, 손실 = 음수.

        Returns
        -------
        CircuitStatus : 기록 후 즉시 평가한 현재 상태
        """
        with self._lock:
            self._auto_reset_if_needed()

            self._today_trades += 1

            if pnl_usdt >= 0:
                # ── 수익 트레이드 ─────────────────────────────────────────
                self._today_wins    += 1
                self._consec_losses  = 0   # 연속 손절 리셋
                logger.info(
                    f"수익 기록: +{pnl_usdt:.2f}U  "
                    f"연속손절={self._consec_losses}  "
                    f"일일손실={self._daily_loss_usdt:.2f}U"
                )
            else:
                # ── 손실 트레이드 ─────────────────────────────────────────
                loss = abs(pnl_usdt)
                self._today_losses    += 1
                self._daily_loss_usdt += loss
                self._consec_losses   += 1
                logger.info(
                    f"손실 기록: -{loss:.2f}U  "
                    f"연속손절={self._consec_losses}  "
                    f"일일손실={self._daily_loss_usdt:.2f}U / "
                    f"한도={self.total_capital * self.daily_loss_limit_pct:.2f}U"
                )
                # 차단 조건 평가
                self._evaluate_halt_conditions()

            return self._build_status()

    # ── 퍼블릭: 수동 제어 ────────────────────────────────────────────────────

    def manual_halt(self, reason: str = "운영자 수동 차단") -> None:
        """긴급 수동 차단 (봇 오작동, 시장 이상 시 호출)."""
        with self._lock:
            self._set_halt(HaltLevel.MANUAL_HALT, reason)
        logger.warning(f"수동 차단 발동: {reason}")

    def manual_release(self) -> None:
        """수동 차단 해제. MANUAL_HALT 에만 적용."""
        with self._lock:
            if self._halt_level == HaltLevel.MANUAL_HALT:
                self._release_halt("수동 해제")
            else:
                logger.warning(
                    f"수동 해제 무시 — 현재 레벨: {self._halt_level.value}"
                )

    def update_capital(self, new_capital: float) -> None:
        """총 자본 갱신 (실시간 잔고 반영)."""
        with self._lock:
            self.total_capital = new_capital
        logger.debug(f"자본 갱신: {new_capital:.2f}U")

    def get_history(self) -> list[HaltEvent]:
        """차단 이벤트 히스토리 반환."""
        with self._lock:
            return list(self._history)

    def reset_daily(self) -> None:
        """일일 통계 수동 리셋 (테스트 / 날짜 경계 수동 처리용)."""
        with self._lock:
            self._do_daily_reset()

    def stop(self) -> None:
        """백그라운드 리셋 스레드 종료."""
        self._running = False

    # ── 내부: 차단 조건 평가 ──────────────────────────────────────────────────

    def _evaluate_halt_conditions(self) -> None:
        """
        손실 기록 후 차단 조건 체크.
        우선순위: DAILY_HALT > COOLDOWN
        이미 MANUAL_HALT 중이면 평가 생략.
        """
        if self._halt_level == HaltLevel.MANUAL_HALT:
            return

        # ── 일일 손실 한도 체크 ───────────────────────────────────────────
        daily_limit = self.total_capital * self.daily_loss_limit_pct
        if self._daily_loss_usdt >= daily_limit:
            reason = (
                f"일일 손실 한도 초과: "
                f"{self._daily_loss_usdt:.2f}U / {daily_limit:.2f}U "
                f"({self._daily_loss_usdt/self.total_capital*100:.2f}%)"
            )
            self._set_halt(HaltLevel.DAILY_HALT, reason)
            return

        # ── 연속 손절 한도 체크 ───────────────────────────────────────────
        if self._consec_losses >= self.consec_loss_limit:
            # 이미 쿨다운 중이면 타이머 연장하지 않음
            if self._halt_level == HaltLevel.COOLDOWN:
                return
            reason = (
                f"연속 손절 {self._consec_losses}회 — "
                f"{self.cooldown_seconds//60}분 쿨다운 시작"
            )
            self._cooldown_until = time.time() + self.cooldown_seconds
            self._set_halt(HaltLevel.COOLDOWN, reason)

    # ── 내부: 상태 전이 ───────────────────────────────────────────────────────

    def _set_halt(self, level: HaltLevel, reason: str) -> None:
        """차단 발동. 현재 레벨보다 심각한 경우만 덮어씀."""
        # 우선순위: MANUAL > DAILY > COOLDOWN > CLEAR
        _priority = {
            HaltLevel.CLEAR:       0,
            HaltLevel.COOLDOWN:    1,
            HaltLevel.DAILY_HALT:  2,
            HaltLevel.MANUAL_HALT: 3,
        }
        if _priority[level] <= _priority[self._halt_level]:
            # 현재보다 약하거나 같은 레벨 → 덮어쓰지 않음
            if level == self._halt_level:
                return
            return

        self._halt_level  = level
        self._halt_reason = reason

        event = HaltEvent(
            level          = level,
            reason         = reason,
            triggered_at   = time.time(),
            daily_loss_pct = self._daily_loss_usdt / self.total_capital,
            consec_losses  = self._consec_losses,
        )
        self._current_event = event
        self._history.append(event)

        logger.warning(f"[차단 발동] level={level.value}  reason={reason}")

        if self._on_halt:
            try:
                self._on_halt(event)
            except Exception as e:
                logger.error(f"on_halt 콜백 오류: {e}")

    def _release_halt(self, reason: str) -> None:
        """차단 해제."""
        if self._halt_level == HaltLevel.CLEAR:
            return

        prev_level        = self._halt_level
        self._halt_level  = HaltLevel.CLEAR
        self._halt_reason = ""

        if self._current_event:
            self._current_event.released_at = time.time()

        logger.info(f"[차단 해제] prev={prev_level.value}  reason={reason}")

        if self._on_release and self._current_event:
            try:
                self._on_release(self._current_event)
            except Exception as e:
                logger.error(f"on_release 콜백 오류: {e}")

        self._current_event = None

    def _release_cooldown_if_expired(self) -> None:
        """쿨다운 타이머가 만료됐으면 자동 해제."""
        if (
            self._halt_level == HaltLevel.COOLDOWN
            and time.time() >= self._cooldown_until
        ):
            self._consec_losses = 0   # 연속 손절 카운터 리셋
            self._release_halt(
                f"쿨다운 {self.cooldown_seconds//60}분 경과 — 자동 해제"
            )

    # ── 내부: 일일 리셋 ───────────────────────────────────────────────────────

    def _auto_reset_if_needed(self) -> None:
        """날짜가 바뀌었으면 자동 리셋."""
        today = date.today()
        if today != self._last_reset_date:
            self._do_daily_reset()

    def _do_daily_reset(self) -> None:
        """일일 통계 초기화 및 DAILY_HALT 해제."""
        prev_date              = self._last_reset_date
        self._last_reset_date  = date.today()
        self._daily_loss_usdt  = 0.0
        self._today_trades     = 0
        self._today_wins       = 0
        self._today_losses     = 0
        self._consec_losses    = 0

        logger.info(
            f"일일 리셋 완료  "
            f"({prev_date} → {self._last_reset_date})"
        )

        # DAILY_HALT 는 자정에 자동 해제
        # MANUAL_HALT 는 명시적 해제 필요 → 유지
        if self._halt_level == HaltLevel.DAILY_HALT:
            self._release_halt("자정 자동 해제 (일일 리셋)")

    def _midnight_reset_loop(self) -> None:
        """자정 리셋 전용 백그라운드 스레드."""
        while self._running:
            now    = datetime.now()
            next_reset = now.replace(
                hour=RESET_HOUR, minute=RESET_MINUTE,
                second=5, microsecond=0,   # 자정 5초 후 (경계 안전 마진)
            )
            if next_reset <= now:
                # 이미 지났으면 내일로
                from datetime import timedelta
                next_reset += timedelta(days=1)

            wait_secs = (next_reset - now).total_seconds()
            logger.debug(f"다음 일일 리셋까지 {wait_secs:.0f}초")
            time.sleep(min(wait_secs, 60))   # 최대 60초마다 깨어나 재확인

            with self._lock:
                self._auto_reset_if_needed()

    # ── 내부: 상태 스냅샷 생성 ────────────────────────────────────────────────

    def _build_status(self) -> CircuitStatus:
        """현재 내부 상태로 CircuitStatus 생성 (lock 보유 상태에서 호출)."""
        blocked  = self._halt_level != HaltLevel.CLEAR
        cooldown = 0.0
        if self._halt_level == HaltLevel.COOLDOWN:
            cooldown = max(0.0, self._cooldown_until - time.time())

        return CircuitStatus(
            blocked          = blocked,
            level            = self._halt_level,
            reason           = self._halt_reason,
            daily_loss_pct   = (
                self._daily_loss_usdt / self.total_capital
                if self.total_capital > 0 else 0.0
            ),
            daily_loss_usdt  = self._daily_loss_usdt,
            consec_losses    = self._consec_losses,
            cooldown_remain  = cooldown,
            today_trades     = self._today_trades,
            today_wins       = self._today_wins,
            today_losses     = self._today_losses,
        )


# ── 유틸 ───────────────────────────────────────────────────────────────────────

def status_summary(s: CircuitStatus) -> str:
    """CircuitStatus 한 줄 요약 — 로깅/디버깅용."""
    if not s.blocked:
        return (
            f"[CB] CLEAR  "
            f"일손실={s.daily_loss_pct*100:.2f}%  "
            f"연속손절={s.consec_losses}회  "
            f"오늘 {s.today_wins}승/{s.today_losses}패"
        )
    extra = ""
    if s.level == HaltLevel.COOLDOWN:
        extra = f"  잔여={s.cooldown_remain/60:.1f}분"
    return (
        f"[CB] ★ {s.level.value}  "
        f"사유={s.reason}{extra}  "
        f"일손실={s.daily_loss_pct*100:.2f}%  "
        f"연속손절={s.consec_losses}회"
    )


# ── 단독 실행 테스트 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    # 알림 콜백 (실제로는 telegram_notifier 호출)
    def on_halt(evt: HaltEvent) -> None:
        print(f"\n  🚨 차단 발동: [{evt.level.value}] {evt.reason}\n")

    def on_release(evt: HaltEvent) -> None:
        print(f"\n  ✅ 차단 해제: [{evt.level.value}]\n")

    CAPITAL = 1000.0
    cb = CircuitBreaker(
        total_capital=CAPITAL,
        on_halt=on_halt,
        on_release=on_release,
    )

    print("=" * 60)
    print("  시나리오 1: 연속 손절 3회 → 쿨다운")
    print("=" * 60)
    trades_1 = [-8.0, -9.0, -11.0]
    for pnl in trades_1:
        s = cb.record_trade(pnl)
        print(f"  PnL={pnl:+.2f}U  →  {status_summary(s)}")

    print(f"\n  → 차단 상태: {cb.check().blocked}  level={cb.check().level.value}")
    print(f"  → 쿨다운 잔여: {cb.check().cooldown_remain:.0f}초")

    print("\n  (연속 손절 카운터 수동 리셋 후 수익 기록)")
    cb._consec_losses = 0
    cb._halt_level    = HaltLevel.CLEAR   # 테스트용 강제 해제
    s = cb.record_trade(+20.0)
    print(f"  PnL=+20.0U  →  {status_summary(s)}")

    print("\n" + "=" * 60)
    print("  시나리오 2: 일일 손실 한도 초과 (자본의 3% = 30U)")
    print("=" * 60)
    cb2 = CircuitBreaker(total_capital=CAPITAL, on_halt=on_halt)
    trades_2 = [-10.0, -10.0, -12.0]   # 합계 32U > 30U
    for pnl in trades_2:
        s = cb2.record_trade(pnl)
        print(f"  PnL={pnl:+.2f}U  →  {status_summary(s)}")

    print(f"\n  → 차단 상태: {cb2.check().blocked}  level={cb2.check().level.value}")
    print(f"  → 일일 손실: {cb2.check().daily_loss_usdt:.2f}U / {CAPITAL*DAILY_LOSS_LIMIT_PCT:.2f}U")

    print("\n" + "=" * 60)
    print("  시나리오 3: 수동 긴급 차단 / 해제")
    print("=" * 60)
    cb3 = CircuitBreaker(total_capital=CAPITAL, on_halt=on_halt, on_release=on_release)
    cb3.manual_halt("테스트 긴급 차단")
    print(f"  → {status_summary(cb3.check())}")
    cb3.manual_release()
    print(f"  → {status_summary(cb3.check())}")

    print("\n" + "=" * 60)
    print("  이벤트 히스토리")
    print("=" * 60)
    for evt in cb2.get_history():
        ts = datetime.fromtimestamp(evt.triggered_at).strftime("%H:%M:%S")
        print(
            f"  [{ts}] {evt.level.value:<14}  "
            f"손실={evt.daily_loss_pct*100:.2f}%  "
            f"연속={evt.consec_losses}회  "
            f"사유={evt.reason}"
        )

    cb.stop(); cb2.stop(); cb3.stop()
