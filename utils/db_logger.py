"""
db_logger.py
------------
모든 거래 기록을 SQLite 에 저장하고
일별/심볼별 성과를 집계하여 조회하는 모듈.

테이블 구조:
  trades      : 개별 트레이드 기록 (진입~청산 1건 = 1행)
  daily_stats : 일별 집계 통계 (매일 자정 갱신)
  signals     : 신호 발생 기록 (진입 미발생 신호 포함, 전략 분석용)
  errors      : 주문 오류 기록

역할:
  - order_executor.on_trade_closed 콜백에서 호출 → trades 저장
  - signal_engine 결과 전체 저장 → signals 저장 (백테스트 분석용)
  - 일별 승률 / MDD / Sharpe 집계
  - 최근 N건 / 특정 심볼 / 날짜 범위 조회

의존 라이브러리:
  표준 라이브러리만 사용 (sqlite3, json, datetime)
"""

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

# ── 로거 ───────────────────────────────────────────────────────────────────────
logger = logging.getLogger("db_logger")


# ── 상수 ───────────────────────────────────────────────────────────────────────
DEFAULT_DB_PATH = "cryptosniper.db"
SCHEMA_VERSION  = 1


# ── 저장용 데이터클래스 ────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """
    단일 트레이드 완전 기록.
    order_executor.on_trade_closed 콜백에서 생성해 save_trade() 에 전달.

    Attributes
    ----------
    symbol          : 심볼  예) "BTC/USDT:USDT"
    direction       : "LONG" / "SHORT"
    entry_price     : 진입가
    close_price     : 청산가
    position_size   : 포지션 크기 (코인 단위)
    leverage        : 레버리지 배율
    pnl_usdt        : 실현 손익 (USDT, 수수료 차감)
    fee_usdt        : 총 수수료 (USDT)
    pnl_pct         : 손익률 (pnl / risk_amount)
    close_reason    : 청산 사유  예) "TP1 도달" / "SL 도달" / "트레일링 발동"
    signal_score    : 진입 당시 signal_engine 점수
    confidence      : 진입 당시 signal_scorer confidence
    entry_order_id  : 진입 주문 ID
    close_order_id  : 청산 주문 ID
    entry_at        : 진입 시각 (ISO 8601)
    close_at        : 청산 시각 (ISO 8601)
    atr_at_entry    : 진입 당시 ATR
    adx_at_entry    : 진입 당시 ADX
    rsi_at_entry    : 진입 당시 RSI
    volume_ratio    : 진입 당시 거래량 배수
    """
    symbol:         str
    direction:      str
    entry_price:    float
    close_price:    float
    position_size:  float
    leverage:       int
    pnl_usdt:       float
    fee_usdt:       float
    close_reason:   str
    signal_score:   int    = 0
    confidence:     int    = 0
    entry_order_id: str    = ""
    close_order_id: str    = ""
    entry_at:       str    = ""
    close_at:       str    = ""
    atr_at_entry:   float  = 0.0
    adx_at_entry:   float  = 0.0
    rsi_at_entry:   float  = 0.0
    volume_ratio:   float  = 0.0
    pnl_pct:        float  = 0.0   # 자동 계산


@dataclass
class SignalRecord:
    """
    신호 발생 기록. 진입 여부와 무관하게 저장 → 전략 분석용.

    Attributes
    ----------
    symbol          : 심볼
    timeframe       : 판단 기준 타임프레임 ("15m" 등)
    signal          : "LONG" / "SHORT" / "HOLD"
    score           : signal_engine 점수
    confidence      : signal_scorer confidence
    reject_reason   : HOLD 사유 (진입 시 빈 문자열)
    trend_direction : "UP" / "DOWN" / "SIDEWAYS"
    adx             : ADX 값
    rsi             : RSI 값
    macd_hist       : MACD 히스토그램
    volume_ratio    : 거래량 배수
    recorded_at     : 기록 시각 (ISO 8601)
    """
    symbol:          str
    timeframe:       str
    signal:          str
    score:           int
    confidence:      int
    reject_reason:   str   = ""
    trend_direction: str   = "SIDEWAYS"
    adx:             float = 0.0
    rsi:             float = 0.0
    macd_hist:       float = 0.0
    volume_ratio:    float = 0.0
    recorded_at:     str   = ""


@dataclass
class DailyStats:
    """
    일별 집계 통계.

    Attributes
    ----------
    date_str        : 날짜 문자열 "YYYY-MM-DD"
    total_trades    : 총 트레이드 수
    wins            : 수익 트레이드 수
    losses          : 손실 트레이드 수
    win_rate        : 승률 (0.0 ~ 1.0)
    total_pnl       : 총 손익 (USDT)
    total_fee       : 총 수수료 (USDT)
    avg_pnl         : 평균 손익 (USDT)
    best_trade      : 최대 수익 트레이드 PnL
    worst_trade     : 최대 손실 트레이드 PnL
    max_consec_wins : 최대 연속 수익
    max_consec_loss : 최대 연속 손절
    capital_end     : 당일 종료 자본 (USDT)
    """
    date_str:         str   = ""
    total_trades:     int   = 0
    wins:             int   = 0
    losses:           int   = 0
    win_rate:         float = 0.0
    total_pnl:        float = 0.0
    total_fee:        float = 0.0
    avg_pnl:          float = 0.0
    best_trade:       float = 0.0
    worst_trade:      float = 0.0
    max_consec_wins:  int   = 0
    max_consec_loss:  int   = 0
    capital_end:      float = 0.0


# ── 메인 클래스 ────────────────────────────────────────────────────────────────

class DbLogger:
    """
    SQLite 기반 거래 기록 저장소.

    사용 예시:
        db = DbLogger("cryptosniper.db")

        # 트레이드 저장
        db.save_trade(trade_record)

        # 신호 저장
        db.save_signal(signal_record)

        # 오늘 성과 집계
        stats = db.get_daily_stats()
        print(f"오늘 승률: {stats.win_rate*100:.1f}%")

        # 최근 20건 조회
        trades = db.get_recent_trades(limit=20)
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """
        Parameters
        ----------
        db_path : SQLite DB 파일 경로 (없으면 자동 생성)
        """
        self._db_path = str(Path(db_path).resolve())
        self._lock    = threading.Lock()
        self._init_db()
        logger.info(f"DbLogger 초기화: {self._db_path}")

    # ── 퍼블릭: 저장 ──────────────────────────────────────────────────────────

    def save_trade(self, trade: TradeRecord) -> int:
        """
        트레이드 기록 저장.

        pnl_pct 는 진입가 × 크기 기준으로 자동 계산.

        Returns
        -------
        int : 삽입된 행의 rowid (-1 이면 실패)
        """
        # pnl_pct 자동 계산
        notional = trade.entry_price * trade.position_size
        if notional > 0:
            trade.pnl_pct = round(trade.pnl_usdt / notional * 100, 4)

        if not trade.close_at:
            trade.close_at = _now_iso()
        if not trade.entry_at:
            trade.entry_at = _now_iso()

        sql = """
            INSERT INTO trades (
                symbol, direction, entry_price, close_price,
                position_size, leverage, pnl_usdt, fee_usdt, pnl_pct,
                close_reason, signal_score, confidence,
                entry_order_id, close_order_id,
                entry_at, close_at,
                atr_at_entry, adx_at_entry, rsi_at_entry, volume_ratio
            ) VALUES (
                :symbol, :direction, :entry_price, :close_price,
                :position_size, :leverage, :pnl_usdt, :fee_usdt, :pnl_pct,
                :close_reason, :signal_score, :confidence,
                :entry_order_id, :close_order_id,
                :entry_at, :close_at,
                :atr_at_entry, :adx_at_entry, :rsi_at_entry, :volume_ratio
            )
        """
        try:
            with self._conn() as conn:
                cur = conn.execute(sql, asdict(trade))
                rowid = cur.lastrowid
                logger.info(
                    f"[DB] 트레이드 저장 id={rowid}  "
                    f"{trade.symbol}  {trade.direction}  "
                    f"PnL={trade.pnl_usdt:+.2f}U"
                )
                return rowid
        except Exception as e:
            logger.error(f"[DB] 트레이드 저장 실패: {e}", exc_info=True)
            return -1

    def save_signal(self, sig: SignalRecord) -> int:
        """신호 기록 저장."""
        if not sig.recorded_at:
            sig.recorded_at = _now_iso()

        sql = """
            INSERT INTO signals (
                symbol, timeframe, signal, score, confidence,
                reject_reason, trend_direction,
                adx, rsi, macd_hist, volume_ratio, recorded_at
            ) VALUES (
                :symbol, :timeframe, :signal, :score, :confidence,
                :reject_reason, :trend_direction,
                :adx, :rsi, :macd_hist, :volume_ratio, :recorded_at
            )
        """
        try:
            with self._conn() as conn:
                cur = conn.execute(sql, asdict(sig))
                return cur.lastrowid
        except Exception as e:
            logger.error(f"[DB] 신호 저장 실패: {e}")
            return -1

    def save_error(
        self,
        module:  str,
        message: str,
        detail:  str = "",
    ) -> None:
        """오류 기록 저장."""
        sql = """
            INSERT INTO errors (module, message, detail, occurred_at)
            VALUES (?, ?, ?, ?)
        """
        try:
            with self._conn() as conn:
                conn.execute(sql, (module, message, detail, _now_iso()))
        except Exception as e:
            logger.error(f"[DB] 오류 저장 실패: {e}")

    # ── 퍼블릭: 조회 ──────────────────────────────────────────────────────────

    def get_recent_trades(
        self,
        limit:  int           = 20,
        symbol: Optional[str] = None,
    ) -> list[dict]:
        """
        최근 트레이드 조회.

        Parameters
        ----------
        limit  : 최대 반환 건수
        symbol : 특정 심볼 필터 (None 이면 전체)

        Returns
        -------
        list[dict] : trades 테이블 행 목록 (최신순)
        """
        if symbol:
            sql    = "SELECT * FROM trades WHERE symbol=? ORDER BY id DESC LIMIT ?"
            params = (symbol, limit)
        else:
            sql    = "SELECT * FROM trades ORDER BY id DESC LIMIT ?"
            params = (limit,)

        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"[DB] 트레이드 조회 실패: {e}")
            return []

    def get_daily_stats(
        self,
        target_date: Optional[date] = None,
    ) -> DailyStats:
        """
        특정 날짜 (기본: 오늘) 트레이드 통계 집계.

        Parameters
        ----------
        target_date : 집계 대상 날짜 (None 이면 오늘)

        Returns
        -------
        DailyStats
        """
        d        = target_date or date.today()
        date_str = d.isoformat()
        next_str = (d + timedelta(days=1)).isoformat()

        sql = """
            SELECT pnl_usdt, fee_usdt
            FROM trades
            WHERE close_at >= ? AND close_at < ?
            ORDER BY close_at ASC
        """
        try:
            with self._conn() as conn:
                rows = conn.execute(sql, (date_str, next_str)).fetchall()
        except Exception as e:
            logger.error(f"[DB] 일별 통계 조회 실패: {e}")
            return DailyStats(date_str=date_str)

        if not rows:
            return DailyStats(date_str=date_str)

        pnls = [r[0] for r in rows]
        fees = [r[1] for r in rows]

        wins   = [p for p in pnls if p >= 0]
        losses = [p for p in pnls if p  < 0]

        # 최대 연속 수익/손절 계산
        max_cw, max_cl, cur_cw, cur_cl = 0, 0, 0, 0
        for p in pnls:
            if p >= 0:
                cur_cw += 1; cur_cl = 0
            else:
                cur_cl += 1; cur_cw = 0
            max_cw = max(max_cw, cur_cw)
            max_cl = max(max_cl, cur_cl)

        total = len(pnls)
        return DailyStats(
            date_str        = date_str,
            total_trades    = total,
            wins            = len(wins),
            losses          = len(losses),
            win_rate        = round(len(wins) / total, 4) if total else 0.0,
            total_pnl       = round(sum(pnls), 4),
            total_fee       = round(sum(fees), 4),
            avg_pnl         = round(sum(pnls) / total, 4) if total else 0.0,
            best_trade      = round(max(pnls), 4),
            worst_trade     = round(min(pnls), 4),
            max_consec_wins = max_cw,
            max_consec_loss = max_cl,
        )

    def get_stats_range(
        self,
        start_date: date,
        end_date:   date,
    ) -> list[DailyStats]:
        """날짜 범위의 일별 통계 목록 반환."""
        results = []
        d = start_date
        while d <= end_date:
            results.append(self.get_daily_stats(d))
            d += timedelta(days=1)
        return results

    def get_total_pnl(self) -> float:
        """전체 누적 손익 (USDT)."""
        try:
            with self._conn() as conn:
                row = conn.execute("SELECT SUM(pnl_usdt) FROM trades").fetchone()
                return round(float(row[0] or 0), 4)
        except Exception as e:
            logger.error(f"[DB] 누적 PnL 조회 실패: {e}")
            return 0.0

    def get_mdd(self) -> float:
        """
        전체 기간 최대 낙폭(MDD) 계산.

        MDD = (최고 누적 PnL - 이후 최저 누적 PnL) / |최고 누적 PnL|

        Returns
        -------
        float : MDD 비율 (0.0 ~ 1.0). 손실 없으면 0.0.
        """
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT pnl_usdt FROM trades ORDER BY id ASC"
                ).fetchall()
        except Exception as e:
            logger.error(f"[DB] MDD 조회 실패: {e}")
            return 0.0

        if not rows:
            return 0.0

        peak       = 0.0
        max_dd     = 0.0
        cumulative = 0.0

        for (pnl,) in rows:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            drawdown = peak - cumulative
            if peak > 0:
                dd_pct = drawdown / peak
                if dd_pct > max_dd:
                    max_dd = dd_pct

        return round(max_dd, 6)

    def get_win_rate(self, days: int = 30) -> float:
        """최근 N일 승률."""
        since = (date.today() - timedelta(days=days)).isoformat()
        try:
            with self._conn() as conn:
                total = conn.execute(
                    "SELECT COUNT(*) FROM trades WHERE close_at >= ?", (since,)
                ).fetchone()[0]
                wins = conn.execute(
                    "SELECT COUNT(*) FROM trades WHERE close_at >= ? AND pnl_usdt >= 0",
                    (since,)
                ).fetchone()[0]
                return round(wins / total, 4) if total else 0.0
        except Exception as e:
            logger.error(f"[DB] 승률 조회 실패: {e}")
            return 0.0

    def get_symbol_stats(self) -> list[dict]:
        """심볼별 성과 집계."""
        sql = """
            SELECT
                symbol,
                COUNT(*)                            AS total,
                SUM(CASE WHEN pnl_usdt>=0 THEN 1 ELSE 0 END) AS wins,
                ROUND(SUM(pnl_usdt), 4)             AS total_pnl,
                ROUND(AVG(pnl_usdt), 4)             AS avg_pnl,
                ROUND(MAX(pnl_usdt), 4)             AS best,
                ROUND(MIN(pnl_usdt), 4)             AS worst
            FROM trades
            GROUP BY symbol
            ORDER BY total_pnl DESC
        """
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(sql).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"[DB] 심볼 통계 조회 실패: {e}")
            return []

    def get_close_reason_stats(self) -> list[dict]:
        """청산 사유별 통계 (TP/SL/트레일링 비율 확인용)."""
        sql = """
            SELECT
                close_reason,
                COUNT(*)                            AS count,
                ROUND(AVG(pnl_usdt), 4)             AS avg_pnl,
                ROUND(SUM(pnl_usdt), 4)             AS total_pnl
            FROM trades
            GROUP BY close_reason
            ORDER BY count DESC
        """
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(sql).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"[DB] 청산 사유 통계 조회 실패: {e}")
            return []

    def get_recent_signals(
        self,
        limit:  int           = 50,
        signal: Optional[str] = None,
    ) -> list[dict]:
        """최근 신호 기록 조회."""
        if signal:
            sql    = "SELECT * FROM signals WHERE signal=? ORDER BY id DESC LIMIT ?"
            params = (signal, limit)
        else:
            sql    = "SELECT * FROM signals ORDER BY id DESC LIMIT ?"
            params = (limit,)
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"[DB] 신호 조회 실패: {e}")
            return []

    # ── 내부: DB 초기화 ───────────────────────────────────────────────────────

    def _init_db(self) -> None:
        """DB 파일 생성 및 테이블 초기화."""
        with self._conn() as conn:
            conn.executescript("""
                PRAGMA journal_mode = WAL;
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS trades (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol          TEXT    NOT NULL,
                    direction       TEXT    NOT NULL,
                    entry_price     REAL    NOT NULL,
                    close_price     REAL    NOT NULL,
                    position_size   REAL    NOT NULL,
                    leverage        INTEGER NOT NULL,
                    pnl_usdt        REAL    NOT NULL,
                    fee_usdt        REAL    NOT NULL DEFAULT 0,
                    pnl_pct         REAL    NOT NULL DEFAULT 0,
                    close_reason    TEXT    NOT NULL DEFAULT '',
                    signal_score    INTEGER NOT NULL DEFAULT 0,
                    confidence      INTEGER NOT NULL DEFAULT 0,
                    entry_order_id  TEXT    NOT NULL DEFAULT '',
                    close_order_id  TEXT    NOT NULL DEFAULT '',
                    entry_at        TEXT    NOT NULL DEFAULT '',
                    close_at        TEXT    NOT NULL DEFAULT '',
                    atr_at_entry    REAL    NOT NULL DEFAULT 0,
                    adx_at_entry    REAL    NOT NULL DEFAULT 0,
                    rsi_at_entry    REAL    NOT NULL DEFAULT 0,
                    volume_ratio    REAL    NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_trades_symbol
                    ON trades(symbol);
                CREATE INDEX IF NOT EXISTS idx_trades_close_at
                    ON trades(close_at);

                CREATE TABLE IF NOT EXISTS signals (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol          TEXT    NOT NULL,
                    timeframe       TEXT    NOT NULL DEFAULT '15m',
                    signal          TEXT    NOT NULL,
                    score           INTEGER NOT NULL DEFAULT 0,
                    confidence      INTEGER NOT NULL DEFAULT 0,
                    reject_reason   TEXT    NOT NULL DEFAULT '',
                    trend_direction TEXT    NOT NULL DEFAULT 'SIDEWAYS',
                    adx             REAL    NOT NULL DEFAULT 0,
                    rsi             REAL    NOT NULL DEFAULT 0,
                    macd_hist       REAL    NOT NULL DEFAULT 0,
                    volume_ratio    REAL    NOT NULL DEFAULT 0,
                    recorded_at     TEXT    NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_signals_symbol
                    ON signals(symbol);
                CREATE INDEX IF NOT EXISTS idx_signals_recorded_at
                    ON signals(recorded_at);

                CREATE TABLE IF NOT EXISTS daily_stats (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    date_str        TEXT    UNIQUE NOT NULL,
                    total_trades    INTEGER NOT NULL DEFAULT 0,
                    wins            INTEGER NOT NULL DEFAULT 0,
                    losses          INTEGER NOT NULL DEFAULT 0,
                    win_rate        REAL    NOT NULL DEFAULT 0,
                    total_pnl       REAL    NOT NULL DEFAULT 0,
                    total_fee       REAL    NOT NULL DEFAULT 0,
                    avg_pnl         REAL    NOT NULL DEFAULT 0,
                    best_trade      REAL    NOT NULL DEFAULT 0,
                    worst_trade     REAL    NOT NULL DEFAULT 0,
                    max_consec_wins INTEGER NOT NULL DEFAULT 0,
                    max_consec_loss INTEGER NOT NULL DEFAULT 0,
                    capital_end     REAL    NOT NULL DEFAULT 0,
                    updated_at      TEXT    NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS errors (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    module      TEXT NOT NULL DEFAULT '',
                    message     TEXT NOT NULL DEFAULT '',
                    detail      TEXT NOT NULL DEFAULT '',
                    occurred_at TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                INSERT OR IGNORE INTO meta VALUES ('schema_version', '1');
            """)
        logger.debug("DB 스키마 초기화 완료")

    # ── 내부: 연결 컨텍스트 매니저 ───────────────────────────────────────────

    @contextmanager
    def _conn(self):
        """
        thread-safe SQLite 연결 컨텍스트 매니저.
        check_same_thread=False + Lock 으로 멀티스레드 안전 보장.
        """
        with self._lock:
            conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
                timeout=10,
            )
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()


# ── 유틸 ───────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    """현재 시각 ISO 8601 문자열 반환."""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def print_daily_report(stats: DailyStats) -> None:
    """DailyStats 를 보기 좋게 출력."""
    print(f"\n{'='*52}")
    print(f"  📊 일별 성과 리포트  [{stats.date_str}]")
    print(f"{'='*52}")
    print(f"  총 트레이드   : {stats.total_trades}건")
    print(f"  승 / 패       : {stats.wins}승 / {stats.losses}패")
    print(f"  승률          : {stats.win_rate*100:.1f}%")
    print(f"  총 손익       : {stats.total_pnl:+.2f} USDT")
    print(f"  총 수수료     : -{stats.total_fee:.2f} USDT")
    print(f"  평균 손익     : {stats.avg_pnl:+.2f} USDT")
    print(f"  최고 트레이드 : {stats.best_trade:+.2f} USDT")
    print(f"  최악 트레이드 : {stats.worst_trade:+.2f} USDT")
    print(f"  최대 연속 수익: {stats.max_consec_wins}회")
    print(f"  최대 연속 손절: {stats.max_consec_loss}회")


# ── 단독 실행 테스트 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    DB_PATH = "test_cryptosniper.db"
    db = DbLogger(DB_PATH)

    # ── 더미 트레이드 삽입 ─────────────────────────────────────────────────────
    print("\n[ 더미 트레이드 30건 삽입 ]")
    symbols    = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
    directions = ["LONG", "SHORT"]
    reasons    = ["TP1 도달", "TP2 도달", "SL 도달", "트레일링 발동"]

    for i in range(30):
        entry = round(random.uniform(40000, 60000), 2)
        pnl   = round(random.uniform(-15, 25), 4)
        tr = TradeRecord(
            symbol         = random.choice(symbols),
            direction      = random.choice(directions),
            entry_price    = entry,
            close_price    = round(entry + pnl * 10, 2),
            position_size  = round(random.uniform(0.001, 0.02), 6),
            leverage       = random.choice([3, 5, 7, 10]),
            pnl_usdt       = pnl,
            fee_usdt       = round(abs(pnl) * 0.04, 4),
            close_reason   = random.choice(reasons),
            signal_score   = random.randint(5, 14),
            confidence     = random.randint(40, 100),
            adx_at_entry   = round(random.uniform(25, 55), 1),
            rsi_at_entry   = round(random.uniform(35, 65), 1),
            volume_ratio   = round(random.uniform(2.0, 5.0), 2),
        )
        db.save_trade(tr)

    # ── 일별 통계 ──────────────────────────────────────────────────────────────
    stats = db.get_daily_stats()
    print_daily_report(stats)

    # ── MDD / 승률 ─────────────────────────────────────────────────────────────
    print(f"\n  전체 누적 PnL : {db.get_total_pnl():+.2f} USDT")
    print(f"  MDD           : {db.get_mdd()*100:.2f}%")
    print(f"  30일 승률     : {db.get_win_rate(30)*100:.1f}%")

    # ── 심볼별 통계 ────────────────────────────────────────────────────────────
    print("\n[ 심볼별 성과 ]")
    for row in db.get_symbol_stats():
        wr = row["wins"] / row["total"] * 100 if row["total"] else 0
        print(
            f"  {row['symbol']:<22}  "
            f"총 {row['total']}건  "
            f"승률 {wr:.0f}%  "
            f"PnL {row['total_pnl']:+.2f}U"
        )

    # ── 청산 사유별 통계 ────────────────────────────────────────────────────────
    print("\n[ 청산 사유별 통계 ]")
    for row in db.get_close_reason_stats():
        print(
            f"  {row['close_reason']:<16}  "
            f"{row['count']}건  "
            f"평균 PnL {row['avg_pnl']:+.2f}U"
        )

    # ── 최근 5건 조회 ──────────────────────────────────────────────────────────
    print("\n[ 최근 5건 트레이드 ]")
    for t in db.get_recent_trades(limit=5):
        print(
            f"  id={t['id']}  {t['symbol']:<22}  "
            f"{t['direction']}  "
            f"PnL={t['pnl_usdt']:+.2f}U  "
            f"사유={t['close_reason']}"
        )

    # 테스트 DB 삭제
    Path(DB_PATH).unlink(missing_ok=True)
    print("\n  (테스트 DB 삭제 완료)")