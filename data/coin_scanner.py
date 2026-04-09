"""
coin_scanner.py
---------------
Binance Futures 전체 종목을 스캔하여 진입 후보 코인 리스트를 반환하는 모듈.

동작 방식:
  1. 자정마다 1회: 거래대금 상위 WATCHLIST_SIZE(30)개 고정 관심 심볼 선정
     → REST API 대량 호출은 이 시점에만 발생
  2. 이후 scan() 호출 시: 고정 심볼 30개만 OHLCV 조회 → 지표 계산 → 후보 반환
     → REST 호출 30회로 고정, 429 오류 대폭 감소

필터 조건:
  1. 24H 거래대금 > MIN_QUOTE_VOLUME (기본 $50M)
  2. ATR(14) / 현재가 > MIN_ATR_RATIO (기본 1.5%)
  3. 현재 거래량 > 20일 평균 거래량 × VOLUME_SURGE_RATIO (기본 1.2)
  4. ADX(14) > MIN_ADX (기본 25) — 추세 존재 확인
"""

import time
import logging
from datetime import date
from typing import Optional

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta

# ── 로거 ───────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("coin_scanner")


# ── 상수 ───────────────────────────────────────────────────────────────────────
MIN_QUOTE_VOLUME   = 100_000_000  # 24H 거래대금 최소값 (USDT) — $1억
MIN_ATR_RATIO      = 0.015        # ATR / 현재가 최소 비율 (1.5%)
VOLUME_SURGE_RATIO = 1.2          # 현재 거래량 / 20일 평균 배수 (완화)
MIN_ADX            = 25           # ADX 최소값
ATR_PERIOD         = 14
ADX_PERIOD         = 14
VOLUME_MA_PERIOD   = 20
OHLCV_LIMIT        = 50           # 지표 계산용 캔들 수
TIMEFRAME          = "1h"
MAX_CANDIDATES     = 10           # scan() 최종 반환 후보 수
REQUEST_DELAY      = 0.5          # API 호출 간 딜레이 (초)

WATCHLIST_SIZE     = 30           # 고정 관심 심볼 수 (자정마다 갱신)


class CoinScanner:
    """
    Binance Futures USDT-M 마켓에서 조건을 충족하는
    후보 코인 목록을 스캔하여 반환한다.

    사용 예시:
        scanner = CoinScanner(api_key="...", api_secret="...")

        # 자정마다 1회 호출 — 관심 심볼 30개 선정 (REST 대량 호출)
        scanner.refresh_watchlist()

        # 5분마다 호출 — 관심 심볼 30개만 분석 (REST 30회)
        candidates = scanner.scan()
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = False,
    ):
        self.exchange = ccxt.binanceusdm(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "options": {"defaultType": "future"},
                "enableRateLimit": True,
            }
        )
        if testnet:
            self.exchange.set_sandbox_mode(True)

        self._markets: dict = {}

        # 고정 관심 심볼 리스트 (자정마다 갱신)
        self._watchlist: list[str] = []
        self._watchlist_date: Optional[date] = None   # 마지막 갱신 날짜

    # ── 퍼블릭 메서드 ──────────────────────────────────────────────────────────

    def refresh_watchlist(self) -> list[str]:
        """
        거래대금 상위 WATCHLIST_SIZE 개 심볼을 선정해 고정 관심 리스트 갱신.

        자정마다 1회 호출 (main.py 스케줄러 등록).
        REST API 대량 호출이 발생하는 유일한 지점.

        Returns
        -------
        list[str] : 갱신된 관심 심볼 목록
        """
        logger.info("=== 관심 심볼 갱신 시작 ===")

        if not self._markets:
            self._load_markets()

        # 티커 1회 조회 → 거래대금 정렬 → 상위 WATCHLIST_SIZE 개 선정
        tickers = self._fetch_tickers()
        volume_ranked = self._rank_by_volume(tickers)
        self._watchlist      = volume_ranked[:WATCHLIST_SIZE]
        self._watchlist_date = date.today()

        logger.info(
            f"=== 관심 심볼 갱신 완료: {len(self._watchlist)}개 ===  "
            f"({self._watchlist_date})"
        )
        for i, sym in enumerate(self._watchlist, 1):
            logger.info(f"  {i:>2}. {sym}")

        return self._watchlist

    def scan(self) -> list[dict]:
        """
        고정 관심 심볼(30개)만 분석하여 진입 후보 반환.

        watchlist 가 비어 있으면 자동으로 refresh_watchlist() 먼저 실행.
        REST API 호출 = 최대 WATCHLIST_SIZE(30)회로 고정.

        Returns
        -------
        list[dict] : score 기준 내림차순 정렬된 후보 목록
        """
        # 관심 심볼이 없으면 먼저 갱신
        if not self._watchlist:
            logger.info("관심 심볼 없음 — refresh_watchlist() 자동 실행")
            self.refresh_watchlist()

        logger.info(
            f"=== 코인 스캔 시작 "
            f"(관심 심볼 {len(self._watchlist)}개 대상) ==="
        )

        candidates    = []
        consecutive_429 = 0

        for symbol in self._watchlist:
            try:
                result = self._analyze_symbol(symbol)
                if result:
                    candidates.append(result)
                consecutive_429 = 0
                time.sleep(REQUEST_DELAY)

            except ccxt.NetworkError as e:
                if "429" in str(e):
                    consecutive_429 += 1
                    wait = min(5 * consecutive_429, 60)
                    logger.warning(
                        f"[{symbol}] 429 감지 — {wait}초 대기 "
                        f"(연속 {consecutive_429}회)"
                    )
                    time.sleep(wait)
                else:
                    logger.warning(f"[{symbol}] 네트워크 오류: {e}")

            except ccxt.ExchangeError as e:
                logger.warning(f"[{symbol}] 거래소 오류: {e}")

            except Exception as e:
                logger.error(f"[{symbol}] 예외 발생: {e}", exc_info=True)

        # score 기준 정렬 → 상위 MAX_CANDIDATES 개 반환
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top = candidates[:MAX_CANDIDATES]

        logger.info(f"=== 스캔 완료: 최종 후보 {len(top)}개 ===")
        for c in top:
            logger.info(
                f"  {c['symbol']:20s} | score={c['score']:.1f} | "
                f"ADX={c['adx']:.1f} | ATR%={c['atr_ratio']*100:.2f}% | "
                f"VolRatio={c['volume_ratio']:.1f}x"
            )
        return top

    def get_watchlist(self) -> list[str]:
        """현재 관심 심볼 목록 반환."""
        return list(self._watchlist)

    def get_watchlist_date(self) -> Optional[date]:
        """마지막 관심 심볼 갱신 날짜 반환."""
        return self._watchlist_date

    # ── 내부 메서드 ────────────────────────────────────────────────────────────

    def _load_markets(self) -> None:
        """거래소 마켓 정보 로드 및 캐시."""
        logger.info("마켓 정보 로드 중...")
        self._markets = self.exchange.load_markets()
        logger.info(f"총 {len(self._markets)}개 마켓 로드 완료")

    def _fetch_tickers(self) -> dict:
        """전체 USDT Perp 티커 조회."""
        logger.info("전체 티커 조회 중...")
        tickers = self.exchange.fetch_tickers()
        return tickers

    def _rank_by_volume(self, tickers: dict) -> list[str]:
        """
        USDT 무기한 선물 심볼을 24H 거래대금 기준 내림차순 정렬 후 반환.
        refresh_watchlist() 에서 상위 WATCHLIST_SIZE 개 선정에 사용.

        Parameters
        ----------
        tickers : _fetch_tickers() 반환값

        Returns
        -------
        list[str] : 거래대금 내림차순 정렬된 심볼 목록
        """
        scored = []
        for symbol, ticker in tickers.items():
            if not symbol.endswith("/USDT:USDT"):
                continue
            base = symbol.split("/")[0]
            if base in ("BUSD", "USDC", "TUSD", "DAI", "USDP"):
                continue
            market = self._markets.get(symbol, {})
            if not market.get("active", True):
                continue
            quote_vol = float(ticker.get("quoteVolume") or 0)
            if quote_vol < MIN_QUOTE_VOLUME:
                continue
            scored.append((symbol, quote_vol))

        scored.sort(key=lambda x: x[1], reverse=True)
        logger.info(
            f"거래대금 필터 통과: {len(scored)}개 "
            f"(상위 {WATCHLIST_SIZE}개 선정)"
        )
        return [s[0] for s in scored]

    def _analyze_symbol(self, symbol: str) -> Optional[dict]:
        """
        단일 종목 OHLCV 데이터로 지표 계산 후 조건 충족 여부 판단.

        Parameters
        ----------
        symbol : 예) 'BTC/USDT:USDT'

        Returns
        -------
        dict or None : 조건 충족 시 분석 결과 딕셔너리, 미충족 시 None
        """
        ohlcv = self.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=OHLCV_LIMIT)
        if len(ohlcv) < ATR_PERIOD + 1:
            return None

        df = self._ohlcv_to_df(ohlcv)

        # ── 지표 계산 ──────────────────────────────────────────────────────────

        # ATR
        df.ta.atr(length=ATR_PERIOD, append=True)
        atr_col = self._find_col(df, "ATR")
        if atr_col is None:
            return None

        # ADX
        df.ta.adx(length=ADX_PERIOD, append=True)
        adx_col = self._find_col(df, f"ADX_{ADX_PERIOD}")
        if adx_col is None:
            return None

        # 거래량 이동평균
        df["volume_ma"] = df["volume"].rolling(VOLUME_MA_PERIOD).mean()

        # ── 최신 값 추출 ────────────────────────────────────────────────────────
        latest        = df.iloc[-1]
        current_price = latest["close"]
        atr_value     = latest[atr_col]
        adx_value     = latest[adx_col]
        current_vol   = latest["volume"]
        avg_vol       = latest["volume_ma"]

        # NaN 체크
        if any(pd.isna(v) for v in [atr_value, adx_value, avg_vol]):
            return None

        atr_ratio    = atr_value / current_price if current_price > 0 else 0
        volume_ratio = current_vol / avg_vol if avg_vol > 0 else 0

        # ── 필터 조건 체크 ──────────────────────────────────────────────────────
        if adx_value < MIN_ADX:
            return None
        if atr_ratio < MIN_ATR_RATIO:
            return None
        if volume_ratio < VOLUME_SURGE_RATIO:
            return None

        # ── 점수 계산 (0~10) ────────────────────────────────────────────────────
        score = self._calculate_score(adx_value, atr_ratio, volume_ratio)

        return {
            "symbol":        symbol,
            "score":         round(score, 2),
            "adx":           round(adx_value, 2),
            "atr_ratio":     round(atr_ratio, 5),
            "volume_ratio":  round(volume_ratio, 2),
            "current_price": round(current_price, 6),
        }

    def _calculate_score(
        self,
        adx: float,
        atr_ratio: float,
        volume_ratio: float,
    ) -> float:
        """
        각 지표값으로 진입 매력도 점수 계산 (0 ~ 10점).

        배점:
          - ADX          : 최대 4점 (25~60 구간 선형)
          - ATR ratio    : 최대 3점 (1.5%~4.0% 구간)
          - volume ratio : 최대 3점 (2.0x~5.0x 구간)
        """
        adx_score = min(4.0, max(0.0, (adx - 25) / (60 - 25) * 4))
        atr_score = min(3.0, max(0.0, (atr_ratio - 0.015) / (0.04 - 0.015) * 3))
        vol_score = min(3.0, max(0.0, (volume_ratio - 2.0) / (5.0 - 2.0) * 3))
        return adx_score + atr_score + vol_score

    @staticmethod
    def _ohlcv_to_df(ohlcv: list) -> pd.DataFrame:
        """ccxt OHLCV 리스트 → pandas DataFrame 변환."""
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)
        return df

    @staticmethod
    def _find_col(df: pd.DataFrame, keyword: str) -> Optional[str]:
        """
        DataFrame에서 keyword를 포함한 첫 번째 컬럼명 반환.
        pandas_ta 버전에 따라 컬럼명 형식이 달라 유연하게 처리.
        """
        matches = [c for c in df.columns if keyword.upper() in c.upper()]
        return matches[0] if matches else None


# ── 단독 실행 테스트 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()   # 프로젝트 루트의 .env 파일에서 키 로드

    API_KEY    = os.getenv("BINANCE_API_KEY", "")
    API_SECRET = os.getenv("BINANCE_API_SECRET", "")

    scanner = CoinScanner(
        api_key=API_KEY,
        api_secret=API_SECRET,
        testnet=False,  # 읽기 전용 스캔, 잔고 조회 없음
    )

    candidates = scanner.scan()

    print("\n[ 최종 후보 코인 ]")
    print(f"{'순위':<4} {'심볼':<22} {'점수':<8} {'ADX':<8} {'ATR%':<8} {'거래량배수'}")
    print("-" * 68)
    for i, c in enumerate(candidates, 1):
        print(
            f"{i:<4} {c['symbol']:<22} {c['score']:<8.1f} "
            f"{c['adx']:<8.1f} {c['atr_ratio']*100:<8.2f} {c['volume_ratio']:.1f}x"
        )