"""
coin_scanner.py
---------------
Binance Futures 전체 종목을 스캔하여 진입 후보 코인 리스트를 반환하는 모듈.

동작 방식:
  1. 자정마다 1회: 거래대금 상위 WATCHLIST_SIZE(30)개 고정 관심 심볼 선정
  2. 5분마다: 고정 심볼 30개 OHLCV 조회 → 지표 계산 → 후보 반환
  3. B군: 전체 알트 5m OHLCV 기준 침묵 후 첫 스파이크 종목 선정
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
MIN_QUOTE_VOLUME   = 200_000_000  # A군 최소 거래대금 ($2억)
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

# ── A군 / B군 분리 watchlist 상수 ──────────────────────────────────────────────
WATCHLIST_A_SIZE        = 10     # A군: 유동성 안정 종목 수
WATCHLIST_B_SIZE        = 20     # B군: 지금 움직이는 종목 수
WATCHLIST_B_MIN_QUOTE   = 500_000     # B군 최소 거래대금 ($50만, 저유동성 알트 포함)
WATCHLIST_B_REFRESH_MIN = 30     # B군 갱신 주기 (분) — 급등 종목 조기 포착
WATCHLIST_B_VOLUME_LOOKBACK = 24 # B군 판단 기준: 최근 N시간 대비 현재 1시간 배수
WATCHLIST_B_5M_TIMEFRAME = "5m"
WATCHLIST_B_5M_LIMIT = 30
WATCHLIST_B_SILENCE_LOOKBACK = 20
WATCHLIST_B_MAX_RISE_PCT = 15.0
WATCHLIST_B_REQUEST_DELAY = 0.03
WATCHLIST_B_RECENT_SPIKE_BARS = 6
WATCHLIST_B_MIN_SILENT_BARS = 13
WATCHLIST_B_SPIKE_RATIO = 3.0

# ── watchlist 혼합 점수 가중치 ────────────────────────────────────────────────
# DB 분석 결과: 24시간 거래대금 단일 기준 → BTC/ETH/XRP 등 메이저 코인 위주
# → 이 종목들은 전략이 원하는 "모멘텀 폭발" 패턴이 약해서 confidence가 매우 낮음
# 개선: 거래대금(유동성) + 거래량 증가율(모멘텀) 혼합 점수로 선정
WATCHLIST_VOLUME_WEIGHT    = 0.5   # 24시간 거래대금 가중치
WATCHLIST_MOMENTUM_WEIGHT  = 0.5   # 거래량 증가율 가중치
WATCHLIST_MIN_MOMENTUM     = 1.0   # 최소 거래량 증가율 (1.0 = 평균 이상)
WATCHLIST_MAX_MOMENTUM     = 10.0  # 이상치 방지용 증가율 상한

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
        })
        if demo:
            self.exchange.urls.update(DEMO_TRADING_URLS)

        self._markets: dict = {}
        self._watchlist: list[str] = []
        self._watchlist_date: Optional[date] = None

        # A군/B군 분리 관리
        self._watchlist_a: list[str] = []   # 유동성 안정 종목 (자정 갱신)
        self._watchlist_b: list[str] = []   # 지금 움직이는 종목 (1시간 갱신)
        self._watchlist_b_updated: Optional[datetime] = None

    # ── 퍼블릭 메서드 ──────────────────────────────────────────────────────────

    def refresh_watchlist(self) -> list[str]:
        """
        A군 + B군 혼합 watchlist 갱신.

        A군 (WATCHLIST_A_SIZE=10개): 24시간 거래대금 상위 — 유동성 안정
        B군 (WATCHLIST_B_SIZE=20개): 5m 침묵 후 첫 거래량 스파이크 — 지금 움직이는 알트

        호출 주기:
          - A군: 자정 1회 (main.py CronTrigger)
          - B군: 1시간마다 (main.py IntervalTrigger)
          - 이 메서드는 두 군 모두 갱신 (초기 시작 시 또는 강제 갱신 시)
        """
        logger.info("=== 관심 심볼 갱신 시작 (A군+B군) ===")
        if not self._markets:
            self._load_markets()
        tickers = self._fetch_tickers()

        # A군: 거래대금 상위 (유동성 안정)
        self._watchlist_a = self._select_group_a(tickers)

        # B군: 지금 움직이는 종목
        self._watchlist_b = self._select_group_b(tickers, exclude=set(self._watchlist_a))
        self._watchlist_b_updated = datetime.now()

        # 합산 (중복 제거)
        combined = list(dict.fromkeys(self._watchlist_a + self._watchlist_b))
        self._watchlist      = combined
        self._watchlist_date = date.today()

        logger.info(
            f"=== 관심 심볼 갱신 완료: {len(self._watchlist)}개 "
            f"(A군={len(self._watchlist_a)} B군={len(self._watchlist_b)}) ==="
        )
        logger.info(f"  A군: {[s.split('/')[0] for s in self._watchlist_a]}")
        logger.info(f"  B군: {[s.split('/')[0] for s in self._watchlist_b]}")
        return self._watchlist

    def refresh_watchlist_b(self) -> list[str]:
        """
        B군만 갱신 — 1시간마다 호출.
        지금 막 움직이기 시작한 종목을 실시간으로 교체.
        """
        if not self._markets:
            self._load_markets()
        tickers = self._fetch_tickers()

        new_b = self._select_group_b(tickers, exclude=set(self._watchlist_a))
        added   = set(new_b) - set(self._watchlist_b)
        removed = set(self._watchlist_b) - set(new_b)

        self._watchlist_b = new_b
        self._watchlist_b_updated = datetime.now()

        # watchlist 재합산
        combined = list(dict.fromkeys(self._watchlist_a + self._watchlist_b))
        self._watchlist = combined

        if added or removed:
            logger.info(
                f"B군 갱신: +{[s.split('/')[0] for s in added]} "
                f"-{[s.split('/')[0] for s in removed]}"
            )
        else:
            logger.debug("B군 갱신: 변경 없음")
        return self._watchlist

    def scan(self) -> list[dict]:
        if not self._watchlist:
            logger.info("관심 심볼 없음 — refresh_watchlist() 자동 실행")
            self.refresh_watchlist()

        all_symbols = list(self._watchlist)

        logger.info(
            f"=== 코인 스캔 시작 "
            f"(관심 심볼 {len(self._watchlist)}개 대상) ==="
        )

        candidates = []
        consecutive_429 = 0

        for symbol in all_symbols:
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
        return list(self._watchlist)

    def get_watchlist_date(self) -> Optional[date]:
        return self._watchlist_date

    def _load_markets(self) -> None:
        logger.info("마켓 정보 로드 중...")
        self._markets = self.exchange.load_markets()
        logger.info(f"총 {len(self._markets)}개 마켓 로드 완료")

    def _fetch_tickers(self) -> dict:
        logger.info("전체 티커 조회 중...")
        return self.exchange.fetch_tickers()

    def _select_group_a(self, tickers: dict) -> list[str]:
        """
        A군 선정 — 24시간 거래대금 상위 WATCHLIST_A_SIZE개.
        유동성 안정 종목. MIN_QUOTE_VOLUME($2억) 이상만.
        """
        scored = []
        for symbol, ticker in tickers.items():
            if not symbol.endswith("/USDT:USDT"):
                continue
            base = symbol.split("/")[0]
            if base in ("BUSD", "USDC", "TUSD", "DAI", "USDP"):
                continue
            if not base.isascii():
                continue
            market = self._markets.get(symbol, {})
            if not market.get("active", True):
                continue
            quote_vol = float(ticker.get("quoteVolume") or 0)
            if quote_vol < MIN_QUOTE_VOLUME:
                continue
            scored.append((symbol, quote_vol))

        scored.sort(key=lambda x: x[1], reverse=True)
        result = [s[0] for s in scored[:WATCHLIST_A_SIZE]]
        logger.debug(f"A군 선정: {len(result)}개 ({[s.split('/')[0] for s in result]})")
        return result

    def _select_group_b(self, tickers: dict, exclude: set = None) -> list[str]:
        """
        B군 선정 — 전체 알트의 5m 거래량 침묵 후 첫 스파이크 종목.

        티커 누적 거래대금의 호출 간 변화량은 시작 직후 히스토리가 없어
        B군이 0개가 되기 쉽다. 그래서 B군은 5m OHLCV를 직접 확인해
        entry_engine이 원하는 "침묵 → 첫 거래량 변화" 후보만 올린다.
        """
        if exclude is None:
            exclude = set()

        candidates = []
        for symbol, ticker in tickers.items():
            if not symbol.endswith("/USDT:USDT"):
                continue
            if symbol in exclude:
                continue
            base = symbol.split("/")[0]
            if base in ("BTC", "ETH", "BUSD", "USDC", "TUSD", "DAI", "USDP"):
                continue
            if not base.isascii():
                continue
            market = self._markets.get(symbol, {})
            if not market.get("active", True):
                continue
            quote_vol = float(ticker.get("quoteVolume") or 0)
            if quote_vol < WATCHLIST_B_MIN_QUOTE:
                continue
            candidates.append((symbol, quote_vol))

        # 저유동성 알트 타겟: 50만~1000만 범위만 2단계 처리
        # 너무 크면 이미 활발한 종목, 너무 작으면 진입/청산 불가
        WATCHLIST_B_MAX_QUOTE = 10_000_000  # 1000만 상한
        candidates = [
            (s, v) for s, v in candidates
            if v <= WATCHLIST_B_MAX_QUOTE
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:150]  # 범위 내 상위 150개

        scored = []
        for symbol, quote_vol in candidates:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    WATCHLIST_B_5M_TIMEFRAME,
                    limit=WATCHLIST_B_5M_LIMIT,
                )
                time.sleep(WATCHLIST_B_REQUEST_DELAY)
            except Exception as e:
                logger.debug(f"[{symbol}] B군 5m OHLCV 조회 실패: {e}")
                continue

            if len(ohlcv) < WATCHLIST_B_SILENCE_LOOKBACK + 1:
                continue

            df = self._ohlcv_to_df(ohlcv)
            if len(df) < WATCHLIST_B_SILENCE_LOOKBACK + 1:
                continue

            best = None
            start_idx = max(
                WATCHLIST_B_SILENCE_LOOKBACK,
                len(df) - WATCHLIST_B_RECENT_SPIKE_BARS,
            )

            for idx in range(start_idx, len(df)):
                candle = df.iloc[idx]
                quiet = df.iloc[idx - WATCHLIST_B_SILENCE_LOOKBACK:idx]

                quiet_avg = float(quiet["volume"].mean())
                if quiet_avg <= 0:
                    continue

                current_vol = float(candle["volume"])
                spike_ratio = current_vol / quiet_avg
                silent_bars = int((quiet["volume"] <= quiet_avg).sum())
                prior_spikes = int((quiet["volume"] >= quiet_avg * WATCHLIST_B_SPIKE_RATIO).sum())
                quiet_low = float(quiet["low"].min())
                close = float(candle["close"])
                open_ = float(candle["open"])
                rise_pct = ((close - quiet_low) / quiet_low * 100.0) if quiet_low > 0 else 999.0

                if silent_bars < WATCHLIST_B_MIN_SILENT_BARS:
                    continue
                if spike_ratio < WATCHLIST_B_SPIKE_RATIO:
                    continue
                if prior_spikes > 0:
                    continue
                if rise_pct >= WATCHLIST_B_MAX_RISE_PCT:
                    continue

                score = (
                    spike_ratio * 10
                    + silent_bars
                    + (5.0 if close > open_ else 0.0)
                    + (idx - start_idx)
                    + min(20.0, quote_vol / 1_000_000)
                    - rise_pct
                )
                candidate = (symbol, score, spike_ratio, silent_bars, rise_pct, quote_vol)
                if best is None or candidate[1] > best[1]:
                    best = candidate

            if best:
                scored.append(best)

        scored.sort(key=lambda x: x[1], reverse=True)

        result = [s[0] for s in scored[:WATCHLIST_B_SIZE]]
        if result:
            top3 = [
                (s[0].split('/')[0], f"spike={s[2]:.1f}x", f"silent={s[3]}/20")
                for s in scored[:3]
            ]
            logger.info(f"B군 선정: {len(result)}개  상위3={top3}")
        else:
            logger.info("B군 선정: 0개 (조건 충족 종목 없음)")
        return result

    def _rank_by_volume(self, tickers: dict) -> list[str]:
        """
        혼합 점수로 watchlist 상위 종목 선정.

        기존: 24시간 거래대금 단일 기준 정렬
              → BTC/ETH/XRP 등 메이저 코인 위주 → confidence 낮음 (DB 분석 결과)

        개선: 거래대금(유동성) × 거래량 증가율(모멘텀) 혼합 점수
              → 유동성은 확보하면서 지금 움직이는 종목을 우선 선정
              → WATCHLIST_VOLUME_WEIGHT + WATCHLIST_MOMENTUM_WEIGHT = 1.0

        혼합 점수 계산:
          1. 거래대금 정규화: quote_vol / 전체 평균 (상대 크기)
          2. 거래량 증가율: baseVolume / 평균 baseVolume 히스토리
             (히스토리 없으면 1.0으로 처리 — 중립)
          3. 혼합 점수 = vol_norm^WEIGHT × momentum^WEIGHT
        """
        candidates = []

        for symbol, ticker in tickers.items():
            if not symbol.endswith("/USDT:USDT"):
                continue
            base = symbol.split("/")[0]
            if base in ("BUSD", "USDC", "TUSD", "DAI", "USDP"):
                continue
            if not base.isascii():
                continue
            market = self._markets.get(symbol, {})
            if not market.get("active", True):
                continue

            quote_vol = float(ticker.get("quoteVolume") or 0)
            if quote_vol < MIN_QUOTE_VOLUME:
                continue

            # 거래량 증가율 계산 (히스토리 있으면 사용, 없으면 중립 1.0)
            momentum = 1.0
            history = self._ticker_history.get(symbol, [])
            if len(history) >= 5:
                base_vol = float(ticker.get("baseVolume") or 0)
                early_avg = sum(history[:max(5, len(history)//2)]) / max(5, len(history)//2)
                if early_avg > 0 and base_vol > 0:
                    raw_momentum = base_vol / early_avg
                    # 이상치 방지: 상한 클램핑
                    momentum = min(raw_momentum, WATCHLIST_MAX_MOMENTUM)
                    momentum = max(momentum, WATCHLIST_MIN_MOMENTUM)

            candidates.append((symbol, quote_vol, momentum))

        if not candidates:
            return []

        # 거래대금 정규화 (전체 평균 대비)
        avg_quote = sum(c[1] for c in candidates) / len(candidates)

        scored = []
        for symbol, quote_vol, momentum in candidates:
            vol_norm = quote_vol / avg_quote if avg_quote > 0 else 1.0

            # 혼합 점수: 거래대금^0.5 × 모멘텀^0.5
            # 둘 다 높을수록 높은 점수, 한쪽이 0이면 0
            mixed = (vol_norm ** WATCHLIST_VOLUME_WEIGHT) * \
                    (momentum ** WATCHLIST_MOMENTUM_WEIGHT)
            scored.append((symbol, mixed, quote_vol, momentum))

        scored.sort(key=lambda x: x[1], reverse=True)

        logger.info(
            f"거래대금 필터 통과: {len(scored)}개 "
            f"(상위 {WATCHLIST_SIZE}개 선정, 혼합점수 기준)"
        )
        # 상위 WATCHLIST_SIZE개 선정 후 로그
        top = scored[:WATCHLIST_SIZE]
        for sym, score, qvol, mom in top[:5]:  # 상위 5개만 로그
            logger.debug(
                f"  watchlist 선정: {sym:22s} "
                f"거래대금={qvol/1e6:.0f}M  모멘텀={mom:.2f}x  혼합={score:.3f}"
            )
        return [s[0] for s in top]

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
