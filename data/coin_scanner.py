"""
coin_scanner.py
---------------
Binance Futures 전체 종목을 스캔하여 진입 후보 코인 리스트를 반환하는 모듈.

동작 방식:
  1. 자정마다 1회: 거래대금 상위 WATCHLIST_SIZE(30)개 고정 관심 심볼 선정
  2. 5분마다: 고정 심볼 30개 OHLCV 조회 → 지표 계산 → 후보 반환
  3. 1분마다: 전체 티커 1회 호출 → 거래량 급증 종목 감지 → watchlist 임시 추가
     (RAVE 같은 수직 급등 종목 조기 포착용)
"""

import time
import logging
from datetime import date, datetime
from typing import Optional

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

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger("coin_scanner")

# ── 기존 상수 ──────────────────────────────────────────────────────────────────
MIN_QUOTE_VOLUME   = 200_000_000
MIN_ATR_RATIO      = 0.015
VOLUME_SURGE_RATIO = 1.2
MIN_ADX            = 25
ATR_PERIOD         = 14
ADX_PERIOD         = 14
VOLUME_MA_PERIOD   = 20
OHLCV_LIMIT        = 50
TIMEFRAME          = "1h"
MAX_CANDIDATES     = 10
REQUEST_DELAY      = 0.5
WATCHLIST_SIZE     = 30

# ── 거래량 급증 감지 상수 ──────────────────────────────────────────────────────
SURGE_VOLUME_MULTIPLIER = 5.0    # 직전 평균 대비 몇 배 이상이면 급증으로 판단
SURGE_MIN_QUOTE_VOLUME  = 500_000  # 급증 감지 최소 거래대금 ($50만, 너무 소형 제외)
SURGE_WATCHLIST_MINUTES = 60     # 급증 감지 종목을 watchlist에 유지할 시간 (분)
SURGE_MAX_SYMBOLS       = 5      # 급증 감지로 추가할 최대 심볼 수


class CoinScanner:

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = False,
        demo: bool = False,
    ):
        self.exchange = ccxt.binanceusdm({
            "apiKey":  api_key,
            "secret":  api_secret,
            "options": {
                "defaultType":     "future",
                "fetchCurrencies": False,
                "adjustForTimeDifference": True,
            },
            "enableRateLimit": True,
            "urls": DEMO_TRADING_URLS,
        })
        if demo:
            self.exchange.urls.update(DEMO_TRADING_URLS)

        self._markets: dict = {}
        self._watchlist: list[str] = []
        self._watchlist_date: Optional[date] = None

        # 거래량 급증 감지: {symbol: 추가된 datetime}
        self._surge_symbols: dict[str, datetime] = {}

        # 티커 히스토리: 급증 판단을 위한 직전 값 저장
        # {symbol: [quoteVolume, quoteVolume, ...]}  최근 N개 유지
        self._ticker_history: dict[str, list[float]] = {}
        self._ticker_history_size = 20   # 20회 평균으로 판단

    # ── 퍼블릭 메서드 ──────────────────────────────────────────────────────────

    def refresh_watchlist(self) -> list[str]:
        logger.info("=== 관심 심볼 갱신 시작 ===")
        if not self._markets:
            self._load_markets()
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
        if not self._watchlist:
            logger.info("관심 심볼 없음 — refresh_watchlist() 자동 실행")
            self.refresh_watchlist()

        # 만료된 급증 심볼 정리
        self._cleanup_surge_symbols()

        # 급증 심볼을 watchlist에 임시 합산
        all_symbols = list(self._watchlist)
        for sym in self._surge_symbols:
            if sym not in all_symbols:
                all_symbols.append(sym)

        logger.info(
            f"=== 코인 스캔 시작 "
            f"(관심 심볼 {len(self._watchlist)}개 "
            f"+ 급증 감지 {len(self._surge_symbols)}개 대상) ==="
        )

        candidates = []
        consecutive_429 = 0

        for symbol in all_symbols:
            try:
                result = self._analyze_symbol(symbol)
                if result:
                    # 급증 감지로 추가된 종목이면 표시
                    if symbol in self._surge_symbols:
                        result["surge_detected"] = True
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

        candidates.sort(key=lambda x: x["score"], reverse=True)
        top = candidates[:MAX_CANDIDATES]

        logger.info(f"=== 스캔 완료: 최종 후보 {len(top)}개 ===")
        for c in top:
            surge_tag = " ★급증감지" if c.get("surge_detected") else ""
            logger.info(
                f"  {c['symbol']:20s} | score={c['score']:.1f} | "
                f"ADX={c['adx']:.1f} | ATR%={c['atr_ratio']*100:.2f}% | "
                f"VolRatio={c['volume_ratio']:.1f}x{surge_tag}"
            )
        return top

    def detect_volume_surge(self) -> list[str]:
        """
        전체 티커를 1회 조회하여 거래량이 급증한 종목을 감지하고
        _surge_symbols에 추가한다.

        main.py 스케줄러에서 1분마다 호출 권장.

        동작 방식:
          - 전체 티커 1회 REST 호출 (개별 호출 아님 → 빠름)
          - 심볼별로 직전 N회 quoteVolume 히스토리와 현재값 비교
          - 현재값이 히스토리 평균의 SURGE_VOLUME_MULTIPLIER배 이상이면 급증으로 판단
          - 최소 MIN_QUOTE_VOLUME 이상인 종목만 대상

        Returns
        -------
        list[str] : 새로 감지된 급증 심볼 목록
        """
        try:
            tickers = self.exchange.fetch_tickers()
        except Exception as e:
            logger.warning(f"거래량 급증 감지 — 티커 조회 실패: {e}")
            return []

        newly_detected = []

        for symbol, ticker in tickers.items():
            if not symbol.endswith("/USDT:USDT"):
                continue
            base = symbol.split("/")[0]
            if base in ("BUSD", "USDC", "TUSD", "DAI", "USDP"):
                continue

            quote_vol = float(ticker.get("quoteVolume") or 0)

            # 최소 거래대금 미달 → 스킵
            if quote_vol < SURGE_MIN_QUOTE_VOLUME:
                continue

            # 히스토리 업데이트
            if symbol not in self._ticker_history:
                self._ticker_history[symbol] = []
            history = self._ticker_history[symbol]
            history.append(quote_vol)

            # 히스토리가 충분히 쌓이기 전에는 판단하지 않음
            if len(history) < 5:
                # 최대 크기 유지
                if len(history) > self._ticker_history_size:
                    self._ticker_history[symbol] = history[-self._ticker_history_size:]
                continue

            # 직전 값들의 평균 (현재값 제외)
            prev_avg = sum(history[:-1]) / len(history[:-1])

            # 최대 크기 유지
            if len(history) > self._ticker_history_size:
                self._ticker_history[symbol] = history[-self._ticker_history_size:]

            if prev_avg <= 0:
                continue

            surge_ratio = quote_vol / prev_avg

            # 급증 감지
            if surge_ratio >= SURGE_VOLUME_MULTIPLIER:
                # 이미 watchlist에 있으면 추가 불필요
                if symbol in self._watchlist:
                    continue
                # 이미 surge_symbols에 있으면 스킵
                if symbol in self._surge_symbols:
                    continue
                # 최대 개수 초과 시 스킵
                if len(self._surge_symbols) >= SURGE_MAX_SYMBOLS:
                    continue

                self._surge_symbols[symbol] = datetime.now()
                newly_detected.append(symbol)
                logger.info(
                    f"[거래량 급증 감지] {symbol}  "
                    f"현재={quote_vol/1e6:.1f}M  "
                    f"평균={prev_avg/1e6:.1f}M  "
                    f"배수={surge_ratio:.1f}x  "
                    f"→ watchlist 임시 추가 ({SURGE_WATCHLIST_MINUTES}분간)"
                )

        if newly_detected:
            logger.info(
                f"거래량 급증 감지 완료: {len(newly_detected)}개 추가 "
                f"({newly_detected})"
            )

        return newly_detected

    def get_watchlist(self) -> list[str]:
        return list(self._watchlist)

    def get_watchlist_date(self) -> Optional[date]:
        return self._watchlist_date

    def get_surge_symbols(self) -> dict:
        """현재 급증 감지로 추가된 심볼과 추가 시각 반환."""
        return dict(self._surge_symbols)

    # ── 내부 메서드 ────────────────────────────────────────────────────────────

    def _cleanup_surge_symbols(self) -> None:
        """SURGE_WATCHLIST_MINUTES 이상 경과한 급증 심볼 제거."""
        now = datetime.now()
        expired = [
            sym for sym, added_at in self._surge_symbols.items()
            if (now - added_at).total_seconds() > SURGE_WATCHLIST_MINUTES * 60
        ]
        for sym in expired:
            del self._surge_symbols[sym]
            logger.info(f"[급증 감지] {sym} watchlist 임시 제거 (유지 시간 만료)")

    def _load_markets(self) -> None:
        logger.info("마켓 정보 로드 중...")
        self._markets = self.exchange.load_markets()
        logger.info(f"총 {len(self._markets)}개 마켓 로드 완료")

    def _fetch_tickers(self) -> dict:
        logger.info("전체 티커 조회 중...")
        return self.exchange.fetch_tickers()

    def _rank_by_volume(self, tickers: dict) -> list[str]:
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
        ohlcv = self.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=OHLCV_LIMIT)
        if len(ohlcv) < ATR_PERIOD + 1:
            return None

        df = self._ohlcv_to_df(ohlcv)

        df.ta.atr(length=ATR_PERIOD, append=True)
        atr_col = self._find_col(df, "ATR")
        if atr_col is None:
            return None

        df.ta.adx(length=ADX_PERIOD, append=True)
        adx_col = self._find_col(df, f"ADX_{ADX_PERIOD}")
        if adx_col is None:
            return None

        df["volume_ma"] = df["volume"].rolling(VOLUME_MA_PERIOD).mean()

        latest        = df.iloc[-1]
        current_price = latest["close"]
        atr_value     = latest[atr_col]
        adx_value     = latest[adx_col]
        current_vol   = latest["volume"]
        avg_vol       = latest["volume_ma"]

        if any(pd.isna(v) for v in [atr_value, adx_value, avg_vol]):
            return None

        atr_ratio    = atr_value / current_price if current_price > 0 else 0
        volume_ratio = current_vol / avg_vol if avg_vol > 0 else 0

        if adx_value < MIN_ADX:
            return None
        if atr_ratio < MIN_ATR_RATIO:
            return None
        if volume_ratio < VOLUME_SURGE_RATIO:
            return None

        score = self._calculate_score(adx_value, atr_ratio, volume_ratio)

        return {
            "symbol":        symbol,
            "score":         round(score, 2),
            "adx":           round(adx_value, 2),
            "atr_ratio":     round(atr_ratio, 5),
            "volume_ratio":  round(volume_ratio, 2),
            "current_price": round(current_price, 6),
            "surge_detected": False,
        }

    def _calculate_score(self, adx, atr_ratio, volume_ratio) -> float:
        adx_score = min(4.0, max(0.0, (adx - 25) / (60 - 25) * 4))
        atr_score = min(3.0, max(0.0, (atr_ratio - 0.015) / (0.04 - 0.015) * 3))
        vol_score = min(3.0, max(0.0, (volume_ratio - 2.0) / (5.0 - 2.0) * 3))
        return adx_score + atr_score + vol_score

    @staticmethod
    def _ohlcv_to_df(ohlcv: list) -> pd.DataFrame:
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df.astype(float)

    @staticmethod
    def _find_col(df: pd.DataFrame, keyword: str) -> Optional[str]:
        matches = [c for c in df.columns if keyword.upper() in c.upper()]
        return matches[0] if matches else None


# ── 단독 실행 테스트 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    API_KEY    = os.getenv("BINANCE_API_KEY", "")
    API_SECRET = os.getenv("BINANCE_API_SECRET", "")

    scanner = CoinScanner(api_key=API_KEY, api_secret=API_SECRET)
    candidates = scanner.scan()

    print("\n[ 최종 후보 코인 ]")
    print(f"{'순위':<4} {'심볼':<22} {'점수':<8} {'ADX':<8} {'ATR%':<8} {'거래량배수'}")
    print("-" * 68)
    for i, c in enumerate(candidates, 1):
        print(
            f"{i:<4} {c['symbol']:<22} {c['score']:<8.1f} "
            f"{c['adx']:<8.1f} {c['atr_ratio']*100:<8.2f} {c['volume_ratio']:.1f}x"
        )