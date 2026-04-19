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
WATCHLIST_A_SIZE        = 15     # A군: 유동성 안정 종목 수
WATCHLIST_B_SIZE        = 15     # B군: 지금 움직이는 종목 수
WATCHLIST_B_MIN_QUOTE   = 5_000_000   # B군 최소 거래대금 ($500만, 저유동성 차단)
WATCHLIST_B_REFRESH_MIN = 60     # B군 갱신 주기 (분)
WATCHLIST_B_VOLUME_LOOKBACK = 24 # B군 판단 기준: 최근 N시간 대비 현재 1시간 배수

# ── watchlist 혼합 점수 가중치 ────────────────────────────────────────────────
# DB 분석 결과: 24시간 거래대금 단일 기준 → BTC/ETH/XRP 등 메이저 코인 위주
# → 이 종목들은 전략이 원하는 "모멘텀 폭발" 패턴이 약해서 confidence가 매우 낮음
# 개선: 거래대금(유동성) + 거래량 증가율(모멘텀) 혼합 점수로 선정
WATCHLIST_VOLUME_WEIGHT    = 0.5   # 24시간 거래대금 가중치
WATCHLIST_MOMENTUM_WEIGHT  = 0.5   # 거래량 증가율 가중치
WATCHLIST_MIN_MOMENTUM     = 1.0   # 최소 거래량 증가율 (1.0 = 평균 이상)
WATCHLIST_MAX_MOMENTUM     = 10.0  # 이상치 방지용 증가율 상한
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
        exchange_cfg = {
            "apiKey":  api_key,
            "secret":  api_secret,
            "options": {
                "defaultType":     "future",
                "fetchCurrencies": False,     # Spot SAPI 호출 차단
                "adjustForTimeDifference": True,
            },
            "enableRateLimit": True,
        }
        if demo or testnet:
            exchange_cfg["urls"] = DEMO_TRADING_URLS
        self.exchange = ccxt.binanceusdm(exchange_cfg)

        self._markets: dict = {}
        self._watchlist: list[str] = []
        self._watchlist_date: Optional[date] = None

        # 거래량 급증 감지: {symbol: 추가된 datetime}
        self._surge_symbols: dict[str, datetime] = {}

        # 티커 히스토리: 급증 판단을 위한 직전 값 저장
        # {symbol: [quoteVolume, quoteVolume, ...]}  최근 N개 유지
        self._ticker_history: dict[str, list[float]] = {}
        self._ticker_history_size = 20   # 20회 평균으로 판단

        # A군/B군 분리 관리
        self._watchlist_a: list[str] = []   # 유동성 안정 종목 (자정 갱신)
        self._watchlist_b: list[str] = []   # 지금 움직이는 종목 (1시간 갱신)
        self._watchlist_b_updated: Optional[datetime] = None

    # ── 퍼블릭 메서드 ──────────────────────────────────────────────────────────

    def refresh_watchlist(self) -> list[str]:
        """
        A군 + B군 혼합 watchlist 갱신.

        A군 (WATCHLIST_A_SIZE=15개): 24시간 거래대금 상위 — 유동성 안정
        B군 (WATCHLIST_B_SIZE=15개): 최근 1시간 거래량 증가율 상위 — 지금 움직이는 종목

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
          - baseVolume(현재 캔들 거래량) 히스토리와 비교
            → quoteVolume(24시간 누적)은 천천히 변해서 급등 감지 불가
          - 현재 baseVolume이 히스토리 평균의 SURGE_VOLUME_MULTIPLIER배 이상이면 급증
          - 최소 quoteVolume 필터는 유지 (극소형 종목 제외용)

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

            # 최소 24시간 거래대금 필터 (극소형 제외)
            quote_vol = float(ticker.get("quoteVolume") or 0)
            if quote_vol < SURGE_MIN_QUOTE_VOLUME:
                continue

            # ★ 핵심 변경: baseVolume(현재 캔들 거래량)으로 급증 판단
            # quoteVolume은 24시간 누적이라 1분 급등을 감지 불가
            base_vol = float(ticker.get("baseVolume") or 0)
            if base_vol <= 0:
                continue

            # 히스토리 업데이트
            if symbol not in self._ticker_history:
                self._ticker_history[symbol] = []
            history = self._ticker_history[symbol]
            history.append(base_vol)

            # 최대 크기 유지
            if len(history) > self._ticker_history_size:
                self._ticker_history[symbol] = history[-self._ticker_history_size:]

            # 히스토리 5회 미만이면 판단 생략
            if len(history) < 5:
                continue

            # 직전 값들의 평균 (현재값 제외)
            prev_values = history[:-1]
            prev_avg = sum(prev_values) / len(prev_values)

            if prev_avg <= 0:
                continue

            surge_ratio = base_vol / prev_avg

            # 급증 감지
            if surge_ratio >= SURGE_VOLUME_MULTIPLIER:
                if symbol in self._watchlist:
                    continue
                if symbol in self._surge_symbols:
                    continue
                if len(self._surge_symbols) >= SURGE_MAX_SYMBOLS:
                    continue

                self._surge_symbols[symbol] = datetime.now()
                newly_detected.append(symbol)
                logger.info(
                    f"[거래량 급증 감지] {symbol}  "
                    f"현재거래량={base_vol:.0f}  "
                    f"평균거래량={prev_avg:.0f}  "
                    f"배수={surge_ratio:.1f}x  "
                    f"24H거래대금={quote_vol/1e6:.1f}M  "
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
        B군 선정 — 지금 이 시간에 유독 많이 거래되는 종목 WATCHLIST_B_SIZE개.

        선정 기준:
          recent_ratio = baseVolume(현재) / (quoteVolume / 24)
          = 현재 1시간 거래량 / 24시간 평균 1시간 거래량
          이 값이 높을수록 "지금 막 터지는 중"인 종목

        MIN_QUOTE_VOLUME보다 낮아도 WATCHLIST_B_MIN_QUOTE 이상이면 포함
        → NEIRO처럼 아직 24시간 거래대금은 낮지만 지금 폭발하는 종목 포착
        """
        if exclude is None:
            exclude = set()

        scored = []
        for symbol, ticker in tickers.items():
            if not symbol.endswith("/USDT:USDT"):
                continue
            if symbol in exclude:
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
            base_vol  = float(ticker.get("baseVolume") or 0)
            last_price = float(ticker.get("last") or ticker.get("close") or 0)

            # B군 최소 거래대금 필터 (A군보다 낮음 — 저유동성만 차단)
            if quote_vol < WATCHLIST_B_MIN_QUOTE:
                continue
            if base_vol <= 0 or last_price <= 0:
                continue

            # 현재 1시간 거래량(USDT) 추정
            # baseVolume × 현재가 ≈ 지금 이 캔들의 거래대금
            current_1h_usdt = base_vol * last_price

            # 24시간 평균 1시간 거래대금
            avg_1h_usdt = quote_vol / WATCHLIST_B_VOLUME_LOOKBACK
            if avg_1h_usdt <= 0:
                continue

            # 현재 1시간이 평균 대비 몇 배인지
            recent_ratio = current_1h_usdt / avg_1h_usdt

            # 최소 1.5배 이상인 종목만 (평균보다 50% 이상 많이 거래 중)
            if recent_ratio < 1.5:
                continue

            scored.append((symbol, recent_ratio, quote_vol))

        # recent_ratio 내림차순 정렬
        scored.sort(key=lambda x: x[1], reverse=True)

        result = [s[0] for s in scored[:WATCHLIST_B_SIZE]]
        if result:
            top3 = [(s[0].split('/')[0], f"{s[1]:.1f}x") for s in scored[:3]]
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