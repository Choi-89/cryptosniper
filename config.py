"""
config.py
---------
CryptoSniper Bot 전체 설정을 한 곳에서 관리하는 중앙 설정 모듈.

구조:
  1. 환경변수 / .env 파일에서 민감 정보 로드 (API 키, 토큰)
  2. 섹션별 설정 dataclass (Exchange / Scanner / Indicator / Strategy /
                             Risk / Execution / Notification / System)
  3. 최상위 Config 객체가 모든 섹션을 포함
  4. get_config() 싱글턴으로 어디서나 동일 인스턴스 접근
  5. validate() 로 필수값 누락 / 범위 오류 조기 감지

사용 예시:
  from config import get_config
  cfg = get_config()

  # 각 모듈에서 직접 참조
  scanner = CoinScanner(
      api_key    = cfg.exchange.api_key,
      api_secret = cfg.exchange.api_secret,
  )
  cb = CircuitBreaker(
      total_capital = cfg.risk.initial_capital,
  )

.env 파일 예시:
  BINANCE_API_KEY=your_key_here
  BINANCE_API_SECRET=your_secret_here
  TELEGRAM_TOKEN=123456:ABC-DEF
  TELEGRAM_CHAT_ID=-1001234567890
  TOTAL_CAPITAL=1000.0
  USE_TESTNET=false
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ── 로거 ───────────────────────────────────────────────────────────────────────
logger = logging.getLogger("config")

# .env 파일 로드 (프로젝트 루트 기준)
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 섹션별 설정 dataclass
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ExchangeConfig:
    """거래소 연결 설정."""

    api_key:    str  = ""
    api_secret: str  = ""
    testnet:    bool = False

    # Binance Futures WebSocket / REST
    ws_base_url:      str = "wss://fstream.binance.com/stream?streams="
    rest_ohlcv_limit: int = 200        # 초기 REST 캔들 로드 수
    request_delay:    float = 0.15     # API 호출 간 딜레이 (초)
    ping_interval:    int   = 20       # WebSocket ping 간격 (초)
    reconnect_delay:  int   = 5        # 재연결 초기 대기 (초)
    reconnect_max:    int   = 60       # 재연결 최대 대기 (초)

    # 수수료
    taker_fee: float = 0.0004          # 0.04%
    maker_fee: float = 0.0002          # 0.02%


@dataclass
class ScannerConfig:
    """코인 스캐너 설정."""

    # 필터 조건
    min_quote_volume:   float = 50_000_000  # 24H 거래대금 최소 (USDT)
    min_atr_ratio:      float = 0.015       # ATR / 가격 최소 비율 (1.5%)
    volume_surge_ratio: float = 2.0         # 거래량 급등 기준 배수
    min_adx:            float = 25.0        # 추세 존재 최소 ADX

    # 스캔 설정
    scan_timeframe:   str = "1h"            # 스캔 기준 타임프레임
    scan_interval:    int = 300             # 스캔 주기 (초) — 5분
    max_candidates:   int = 10             # 최종 후보 최대 수
    ohlcv_limit:      int = 100            # 지표 계산용 캔들 수


@dataclass
class IndicatorConfig:
    """기술적 지표 파라미터."""

    # EMA
    ema_short:   int = 20
    ema_mid:     int = 50
    ema_long:    int = 200

    # ADX
    adx_period:  int   = 14
    adx_weak:    float = 25.0   # 미만: 횡보
    adx_strong:  float = 40.0   # 이상: 강한 추세

    # 볼린저밴드
    bb_period:   int   = 20
    bb_std:      float = 2.0

    # ATR
    atr_period:  int = 14

    # MACD
    macd_fast:   int = 12
    macd_slow:   int = 26
    macd_signal: int = 9

    # RSI
    rsi_period:      int   = 14
    rsi_overbought:  float = 70.0
    rsi_oversold:    float = 30.0
    rsi_bull_min:    float = 40.0   # 롱 진입 허용 RSI 하한
    rsi_bull_max:    float = 65.0   # 롱 진입 허용 RSI 상한
    rsi_bear_min:    float = 35.0   # 숏 진입 허용 RSI 하한
    rsi_bear_max:    float = 60.0   # 숏 진입 허용 RSI 상한

    # StochRSI
    stoch_rsi_period: int   = 14
    stoch_k_period:   int   = 3
    stoch_d_period:   int   = 3
    stoch_overbought: float = 80.0
    stoch_oversold:   float = 20.0

    # 거래량
    vol_ma_period:    int   = 20
    vol_surge_ratio:  float = 2.0
    vol_strong_ratio: float = 3.0
    obv_ma_period:    int   = 20
    cvd_ma_period:    int   = 14

    # 호가창
    depth_level:      int   = 20
    obi_depth:        int   = 10
    max_slippage_pct: float = 0.003    # 슬리피지 허용 상한 (0.3%)


@dataclass
class StrategyConfig:
    """신호 판단 및 점수 전략 설정."""

    # signal_engine
    min_score_to_enter: int = 5        # 진입 허용 최소 점수

    # signal_scorer
    min_confidence:     int = 40       # 진입 허용 최소 신뢰도 (0~100)
    base_score_max:     int = 22       # signal_engine 보조 점수 최대
    context_score_max:  int = 30       # 맥락 보너스 최대
    penalty_score_max:  int = 30       # 위험 감점 최대

    # 레버리지 결정 테이블: [(confidence 하한, 배율), ...]
    leverage_table: list = field(default_factory=lambda: [
        (85, 10),
        (75,  7),
        (65,  5),
        (55,  3),
        (40,  2),
    ])

    # 멀티 타임프레임
    trend_timeframe:  str = "1h"       # 추세 판단
    entry_timeframe:  str = "15m"      # 진입 타이밍
    confirm_timeframe: str = "5m"      # 정밀 확인 (옵션)


@dataclass
class RiskConfig:
    """리스크 관리 설정."""

    # 자본 / 포지션
    initial_capital:     float = 1000.0    # 초기 운용 자본 (USDT)
    risk_per_trade_pct:  float = 0.01      # 1회 최대 손실 비율 (1%)
    max_positions:       int   = 3         # 최대 동시 포지션 수
    min_position_usdt:   float = 5.0      # 최소 포지션 크기 (USDT)

    # SL / TP
    atr_sl_multiplier:    float = 1.5      # SL 거리 = ATR × 1.5
    tp1_ratio:            float = 2.0      # TP1 = SL거리 × 2  (R:R = 1:2)
    tp2_ratio:            float = 4.0      # TP2 = SL거리 × 4  (R:R = 1:4)
    tp1_close_pct:        float = 0.5      # TP1 도달 시 50% 청산
    trailing_trigger_pct: float = 0.01     # 고점 대비 -1% 이탈 시 청산

    # 레버리지 전역 한도
    min_leverage: int = 1
    max_leverage: int = 10

    # 포지션 명목가치 상한 (USDT)
    # 레버리지가 높아도 단일 포지션 명목가치를 이 값으로 제한
    # 예: 자본 1000 USDT × 30% = 300 USDT 명목 상한
    max_notional_pct: float = 0.30   # 자본 대비 단일 포지션 명목가치 상한 (30%)
    max_notional_abs: float = 500.0  # 절대 상한 (USDT), 0이면 비활성

    # 심볼별 최대 레버리지
    symbol_max_leverage: dict = field(default_factory=lambda: {
        "BTC/USDT:USDT":  10,
        "ETH/USDT:USDT":  10,
        "BNB/USDT:USDT":   7,
        "SOL/USDT:USDT":   7,
        "XRP/USDT:USDT":   5,
        "DOGE/USDT:USDT":  5,
        "ADA/USDT:USDT":   5,
        "_DEFAULT":         5,
    })

    # leverage_manager 시장 승수
    adx_boost_thresh:  float = 40.0
    adx_reduce_thresh: float = 28.0
    adx_sideways:      float = 25.0
    atr_high_ratio:    float = 0.04
    atr_low_ratio:     float = 0.015

    # leverage_manager 리스크 승수
    daily_loss_reduce_1: float = 0.015   # 1.5% 손실 → ×0.8
    daily_loss_reduce_2: float = 0.025   # 2.5% 손실 → ×0.6
    consec_reduce_1:     int   = 1       # 연속 1회 → ×0.9
    consec_reduce_2:     int   = 2       # 연속 2회 → ×0.7

    # Circuit Breaker
    daily_loss_limit_pct: float = 0.03   # 일일 손실 한도 (3%)
    consec_loss_limit:    int   = 3      # 연속 손절 한도
    cooldown_seconds:     int   = 3600   # 쿨다운 시간 (1시간)


@dataclass
class ExecutionConfig:
    """주문 실행 설정."""

    max_retry:        int   = 3      # 주문 실패 시 최대 재시도
    retry_base_delay: float = 0.5    # 재시도 초기 대기 (초)
    order_type_entry: str   = "MARKET"
    order_type_sl:    str   = "STOP_MARKET"
    order_type_tp:    str   = "TAKE_PROFIT_MARKET"
    working_type:     str   = "MARK_PRICE"   # SL/TP 트리거 기준


@dataclass
class NotificationConfig:
    """알림 설정."""

    telegram_token:   str  = ""
    telegram_chat_id: str  = ""
    bot_name:         str  = "CryptoSniper"

    # 알림 활성화 플래그
    notify_entry:     bool = True
    notify_close:     bool = True
    notify_halt:      bool = True
    notify_daily:     bool = True
    notify_error:     bool = True
    notify_heartbeat: bool = True

    # 중복 차단 / 하트비트
    throttle_sec:     int  = 30
    heartbeat_sec:    int  = 3600

    # 일별 리포트 전송 시각 (UTC)
    daily_report_hour:   int = 0    # 자정
    daily_report_minute: int = 5    # 00:05 UTC


@dataclass
class SystemConfig:
    """시스템 / 로깅 설정."""

    db_path:        str  = "cryptosniper.db"
    log_level:      str  = "INFO"
    log_file:       str  = "cryptosniper.log"
    log_max_bytes:  int  = 10 * 1024 * 1024   # 10 MB
    log_backup_cnt: int  = 5

    # 타임존 (로그 표시용, 실제 연산은 UTC)
    timezone:       str  = "UTC"

    # 운용 모드
    dry_run:        bool = False   # True 이면 실제 주문 전송 안 함


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 최상위 Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Config:
    """
    전체 설정 컨테이너.
    get_config() 를 통해 싱글턴으로 접근.
    """
    exchange:     ExchangeConfig     = field(default_factory=ExchangeConfig)
    scanner:      ScannerConfig      = field(default_factory=ScannerConfig)
    indicator:    IndicatorConfig    = field(default_factory=IndicatorConfig)
    strategy:     StrategyConfig     = field(default_factory=StrategyConfig)
    risk:         RiskConfig         = field(default_factory=RiskConfig)
    execution:    ExecutionConfig    = field(default_factory=ExecutionConfig)
    notification: NotificationConfig = field(default_factory=NotificationConfig)
    system:       SystemConfig       = field(default_factory=SystemConfig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 빌더 함수 — 환경변수 → Config 조립
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_config() -> Config:
    """환경변수를 읽어 Config 인스턴스를 생성한다."""

    def _bool(key: str, default: bool) -> bool:
        val = os.getenv(key, "").lower()
        return {"true": True, "false": False}.get(val, default)

    def _float(key: str, default: float) -> float:
        try:
            return float(os.getenv(key, default))
        except (ValueError, TypeError):
            return default

    def _int(key: str, default: int) -> int:
        try:
            return int(os.getenv(key, default))
        except (ValueError, TypeError):
            return default

    cfg = Config()

    # ── 거래소 ──────────────────────────────────────────────────────────────
    cfg.exchange.api_key    = os.getenv("BINANCE_API_KEY",    "")
    cfg.exchange.api_secret = os.getenv("BINANCE_API_SECRET", "")
    cfg.exchange.testnet    = _bool("USE_TESTNET", False)

    # ── 자본 ────────────────────────────────────────────────────────────────
    cfg.risk.initial_capital = _float("TOTAL_CAPITAL", 1000.0)

    # ── 알림 ────────────────────────────────────────────────────────────────
    cfg.notification.telegram_token   = os.getenv("TELEGRAM_TOKEN",   "")
    cfg.notification.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    cfg.notification.bot_name         = os.getenv("BOT_NAME", "CryptoSniper")

    # ── 시스템 ──────────────────────────────────────────────────────────────
    cfg.system.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    cfg.system.db_path   = os.getenv("DB_PATH",   "cryptosniper.db")
    cfg.system.dry_run   = _bool("DRY_RUN", False)

    # ── 전략 파라미터 (선택적 오버라이드) ──────────────────────────────────
    if os.getenv("MIN_SCORE_TO_ENTER"):
        cfg.strategy.min_score_to_enter = _int("MIN_SCORE_TO_ENTER", 5)
    if os.getenv("MIN_CONFIDENCE"):
        cfg.strategy.min_confidence     = _int("MIN_CONFIDENCE", 40)
    if os.getenv("RISK_PER_TRADE_PCT"):
        cfg.risk.risk_per_trade_pct     = _float("RISK_PER_TRADE_PCT", 0.01)
    if os.getenv("MAX_POSITIONS"):
        cfg.risk.max_positions          = _int("MAX_POSITIONS", 3)
    if os.getenv("DAILY_LOSS_LIMIT_PCT"):
        cfg.risk.daily_loss_limit_pct   = _float("DAILY_LOSS_LIMIT_PCT", 0.03)

    return cfg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 유효성 검사
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate(cfg: Config) -> list[str]:
    """
    설정값 유효성 검사.

    Returns
    -------
    list[str] : 오류 메시지 목록. 빈 리스트면 통과.
    """
    errors: list[str] = []

    # ── 필수값 ─────────────────────────────────────────────────────────────
    if not cfg.exchange.api_key:
        errors.append("BINANCE_API_KEY 가 설정되지 않았습니다.")
    if not cfg.exchange.api_secret:
        errors.append("BINANCE_API_SECRET 가 설정되지 않았습니다.")

    # 알림 설정 (없으면 경고만 — 알림 없이도 봇 동작 가능)
    if not cfg.notification.telegram_token:
        logger.warning("TELEGRAM_TOKEN 미설정 — 텔레그램 알림 비활성화")
    if not cfg.notification.telegram_chat_id:
        logger.warning("TELEGRAM_CHAT_ID 미설정 — 텔레그램 알림 비활성화")

    # ── 자본 범위 ──────────────────────────────────────────────────────────
    if cfg.risk.initial_capital <= 0:
        errors.append(f"TOTAL_CAPITAL 은 0 보다 커야 합니다. (현재: {cfg.risk.initial_capital})")

    # ── 리스크 범위 ────────────────────────────────────────────────────────
    if not 0 < cfg.risk.risk_per_trade_pct <= 0.05:
        errors.append(
            f"risk_per_trade_pct 는 0~5% 범위여야 합니다. "
            f"(현재: {cfg.risk.risk_per_trade_pct*100:.1f}%)"
        )
    if not 0 < cfg.risk.daily_loss_limit_pct <= 0.10:
        errors.append(
            f"daily_loss_limit_pct 는 0~10% 범위여야 합니다. "
            f"(현재: {cfg.risk.daily_loss_limit_pct*100:.1f}%)"
        )
    if cfg.risk.max_positions < 1 or cfg.risk.max_positions > 10:
        errors.append(
            f"max_positions 는 1~10 이어야 합니다. "
            f"(현재: {cfg.risk.max_positions})"
        )
    if cfg.risk.atr_sl_multiplier < 0.5 or cfg.risk.atr_sl_multiplier > 5.0:
        errors.append(
            f"atr_sl_multiplier 는 0.5~5.0 이어야 합니다. "
            f"(현재: {cfg.risk.atr_sl_multiplier})"
        )
    if cfg.risk.tp1_ratio <= 1.0:
        errors.append(
            f"tp1_ratio 는 1.0 보다 커야 합니다 (R:R ≥ 1:1). "
            f"(현재: {cfg.risk.tp1_ratio})"
        )

    # ── 전략 범위 ──────────────────────────────────────────────────────────
    if cfg.strategy.min_score_to_enter < 1:
        errors.append(
            f"min_score_to_enter 는 1 이상이어야 합니다. "
            f"(현재: {cfg.strategy.min_score_to_enter})"
        )
    if not 0 <= cfg.strategy.min_confidence <= 100:
        errors.append(
            f"min_confidence 는 0~100 이어야 합니다. "
            f"(현재: {cfg.strategy.min_confidence})"
        )

    # ── 지표 파라미터 ──────────────────────────────────────────────────────
    if cfg.indicator.ema_short >= cfg.indicator.ema_mid:
        errors.append(
            f"ema_short({cfg.indicator.ema_short}) < "
            f"ema_mid({cfg.indicator.ema_mid}) 이어야 합니다."
        )
    if cfg.indicator.ema_mid >= cfg.indicator.ema_long:
        errors.append(
            f"ema_mid({cfg.indicator.ema_mid}) < "
            f"ema_long({cfg.indicator.ema_long}) 이어야 합니다."
        )

    # ── testnet 경고 ───────────────────────────────────────────────────────
    if cfg.exchange.testnet:
        logger.warning("⚠️  테스트넷 모드 활성화 — 실제 자금이 사용되지 않습니다.")
    if cfg.system.dry_run:
        logger.warning("⚠️  DRY RUN 모드 — 실제 주문이 전송되지 않습니다.")

    return errors


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 싱글턴 접근
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_config_instance: Optional[Config] = None


def get_config() -> Config:
    """
    Config 싱글턴 반환.
    최초 호출 시 .env 로드 + 빌드 + 유효성 검사 실행.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = _build_config()
        errors = validate(_config_instance)
        if errors:
            for e in errors:
                logger.error(f"설정 오류: {e}")
            raise ValueError(
                f"Config 유효성 검사 실패 ({len(errors)}건):\n"
                + "\n".join(f"  - {e}" for e in errors)
            )
        logger.info("Config 로드 완료")
    return _config_instance


def reload_config() -> Config:
    """설정 강제 재로드 (테스트 / 핫리로드용)."""
    global _config_instance
    _config_instance = None
    return get_config()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 단독 실행: 설정 덤프 및 검증
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import dataclasses
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    print("=" * 60)
    print("  CryptoSniper Bot — 설정 로드 및 검증")
    print("=" * 60)

    try:
        cfg = get_config()
        print("✅ 설정 유효성 검사 통과\n")
    except ValueError as e:
        print(f"❌ {e}")
        exit(1)

    # 민감 정보 마스킹 후 출력
    def _mask(s: str) -> str:
        return s[:4] + "****" + s[-4:] if len(s) > 8 else "****"

    print(f"[ 거래소 ]")
    print(f"  API Key    : {_mask(cfg.exchange.api_key) if cfg.exchange.api_key else '미설정'}")
    print(f"  API Secret : {_mask(cfg.exchange.api_secret) if cfg.exchange.api_secret else '미설정'}")
    print(f"  Testnet    : {cfg.exchange.testnet}")

    print(f"\n[ 자본 / 리스크 ]")
    print(f"  초기 자본       : {cfg.risk.initial_capital:,.2f} USDT")
    print(f"  1회 손실 한도   : {cfg.risk.risk_per_trade_pct*100:.1f}%  "
          f"= {cfg.risk.initial_capital * cfg.risk.risk_per_trade_pct:.2f} USDT")
    print(f"  일일 손실 한도  : {cfg.risk.daily_loss_limit_pct*100:.1f}%  "
          f"= {cfg.risk.initial_capital * cfg.risk.daily_loss_limit_pct:.2f} USDT")
    print(f"  최대 동시 포지션: {cfg.risk.max_positions}개")
    print(f"  ATR SL 배수     : {cfg.risk.atr_sl_multiplier}x")
    print(f"  TP1 / TP2       : SL×{cfg.risk.tp1_ratio} / SL×{cfg.risk.tp2_ratio}")
    print(f"  트레일링 트리거 : -{cfg.risk.trailing_trigger_pct*100:.1f}%")

    print(f"\n[ 전략 ]")
    print(f"  진입 최소 점수  : {cfg.strategy.min_score_to_enter}점")
    print(f"  진입 최소 신뢰도: {cfg.strategy.min_confidence}/100")
    print(f"  추세 타임프레임 : {cfg.strategy.trend_timeframe}")
    print(f"  진입 타임프레임 : {cfg.strategy.entry_timeframe}")

    print(f"\n[ 지표 ]")
    print(f"  EMA             : {cfg.indicator.ema_short} / {cfg.indicator.ema_mid} / {cfg.indicator.ema_long}")
    print(f"  MACD            : {cfg.indicator.macd_fast} / {cfg.indicator.macd_slow} / {cfg.indicator.macd_signal}")
    print(f"  RSI 기간        : {cfg.indicator.rsi_period}  (롱={cfg.indicator.rsi_bull_min}~{cfg.indicator.rsi_bull_max})")
    print(f"  ADX 기간        : {cfg.indicator.adx_period}  (약={cfg.indicator.adx_weak} 강={cfg.indicator.adx_strong})")
    print(f"  볼린저밴드      : {cfg.indicator.bb_period} / {cfg.indicator.bb_std}")

    print(f"\n[ 알림 ]")
    token_status = "설정됨" if cfg.notification.telegram_token else "미설정"
    print(f"  Telegram Token  : {token_status}")
    print(f"  봇 이름         : {cfg.notification.bot_name}")

    print(f"\n[ 시스템 ]")
    print(f"  DB 경로         : {cfg.system.db_path}")
    print(f"  로그 레벨       : {cfg.system.log_level}")
    print(f"  Dry Run 모드    : {cfg.system.dry_run}")