"""
data_fetcher.py
---------------
Binance Futures WebSocket으로 실시간 캔들 데이터를 수신하고,
REST API로 과거 OHLCV를 초기 로드하여 DataFrame을 최신 상태로 유지하는 모듈.

역할:
  - 여러 심볼 × 여러 타임프레임 동시 구독
  - 캔들 닫힘(is_closed=True) 시 signal_engine에 콜백 전달
  - 네트워크 끊김 시 자동 재연결 (지수 백오프)
  - get_df(symbol, timeframe) 으로 언제든 최신 DataFrame 조회 가능

의존 라이브러리:
  pip install ccxt websocket-client pandas pandas-ta python-dotenv
"""

import json
import logging
import threading
import time
from collections import defaultdict
from typing import Callable, Optional

import ccxt
import pandas as pd
import pandas_ta as ta
import websocket  # websocket-client 패키지

# ── 로거 ───────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("data_fetcher")

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

# ── 상수 ───────────────────────────────────────────────────────────────────────
WS_BASE_URL       = "wss://fstream.binance.com/stream?streams="  # Futures 복합 스트림
REST_OHLCV_LIMIT  = 300        # 초기 REST 로드 캔들 수
RECONNECT_DELAY   = 5          # 재연결 초기 대기 시간 (초)
RECONNECT_MAX     = 60         # 재연결 최대 대기 시간 (초)
PING_INTERVAL     = 20         # WebSocket ping 간격 (초)

# 지원 타임프레임 → Binance 스트림명 접미사 매핑
TF_MAP = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1d",
}


class DataFetcher:
    """
    여러 심볼 × 여러 타임프레임의 OHLCV를 실시간으로 유지하는 클래스.

    사용 예시:
        def on_candle_close(symbol, timeframe, df):
            print(f"{symbol} {timeframe} 캔들 닫힘: {df.iloc[-1]}")

        fetcher = DataFetcher(
            symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
            timeframes=["15m", "1h"],
            on_closed_candle=on_candle_close,
        )
        fetcher.start()
        # 백그라운드 스레드에서 실행됨
        # ...
        fetcher.stop()

        # 언제든 최신 DataFrame 조회
        df = fetcher.get_df("BTC/USDT:USDT", "1h")
    """

    def __init__(
        self,
        symbols: list[str],
        timeframes: list[str],
        on_closed_candle: Optional[Callable] = None,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = False,
        demo: bool = False,
    ):
        """
        Parameters
        ----------
        symbols          : 구독할 심볼 목록  예) ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        timeframes       : 구독할 타임프레임 예) ["15m", "1h"]
        on_closed_candle : 캔들 닫힘 시 호출될 콜백
                           signature: fn(symbol: str, timeframe: str, df: DataFrame)
        api_key          : Binance API 키 (공개 스트림은 불필요)
        api_secret       : Binance API 시크릿
        testnet          : True 이면 테스트넷 사용
        """
        self.symbols    = symbols
        self.timeframes = timeframes
        self.on_closed_candle = on_closed_candle

        # REST 클라이언트 (초기 데이터 로드용)
        self.exchange = ccxt.binanceusdm({
            "apiKey":  api_key,
            "secret":  api_secret,
            "options": {
                "defaultType":     "future",
                "fetchCurrencies": False,     # Spot SAPI 호출 차단
                "adjustForTimeDifference": True,
            },
            "enableRateLimit": True,
            "urls": DEMO_TRADING_URLS,        # ← 생성 시 바로 주입
        })
        if demo:
            self.exchange.urls.update(DEMO_TRADING_URLS)

        # DataFrame 저장소: _store[symbol][timeframe] = pd.DataFrame
        self._store: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict)
        self._store_lock = threading.Lock()

        # WebSocket 상태
        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._running = False
        self._reconnect_delay = RECONNECT_DELAY

    # ── 퍼블릭 메서드 ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """REST로 초기 데이터 로드 후 WebSocket 구독 시작 (백그라운드 스레드)."""
        logger.info("DataFetcher 시작")
        self._running = True

        # 1. 초기 OHLCV REST 로드
        self._load_initial_data()

        # 2. WebSocket 백그라운드 스레드 시작
        self._ws_thread = threading.Thread(
            target=self._run_ws_loop, daemon=True, name="ws-loop"
        )
        self._ws_thread.start()
        logger.info("WebSocket 스레드 시작 완료")

    def stop(self) -> None:
        """WebSocket 연결 종료 및 스레드 정리."""
        logger.info("DataFetcher 종료 중...")
        self._running = False
        if self._ws:
            self._ws.close()
        if self._ws_thread:
            self._ws_thread.join(timeout=5)
        logger.info("DataFetcher 종료 완료")

    def get_df(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        저장된 DataFrame의 복사본 반환 (thread-safe).

        Parameters
        ----------
        symbol    : 예) "BTC/USDT:USDT"
        timeframe : 예) "1h"

        Returns
        -------
        pd.DataFrame or None
        """
        with self._store_lock:
            tf_data = self._store.get(symbol, {})
            df = tf_data.get(timeframe)
            return df.copy() if df is not None else None

    def get_latest(self, symbol: str, timeframe: str) -> Optional[pd.Series]:
        """
        가장 최근 캔들 데이터(Series) 반환.

        Returns
        -------
        pd.Series or None
        """
        df = self.get_df(symbol, timeframe)
        return df.iloc[-1] if df is not None and not df.empty else None

    # ── 초기 데이터 로드 (REST) ────────────────────────────────────────────────

    def _load_initial_data(self) -> None:
        """모든 심볼 × 타임프레임 조합의 과거 OHLCV를 REST로 로드."""
        logger.info("초기 OHLCV 데이터 REST 로드 중...")
        for symbol in self.symbols:
            for tf in self.timeframes:
                try:
                    self._fetch_and_store(symbol, tf)
                    time.sleep(0.1)  # 레이트 리밋 방지
                except Exception as e:
                    logger.error(f"초기 로드 실패 [{symbol} {tf}]: {e}")
        logger.info("초기 데이터 로드 완료")

    def _fetch_and_store(self, symbol: str, timeframe: str) -> None:
        """단일 심볼+타임프레임 OHLCV 조회 후 저장소에 저장."""
        ohlcv = self.exchange.fetch_ohlcv(
            symbol, timeframe, limit=REST_OHLCV_LIMIT
        )
        df = self._to_df(ohlcv)
        with self._store_lock:
            self._store[symbol][timeframe] = df
        logger.debug(f"REST 로드: {symbol} {timeframe} → {len(df)}개 캔들")

    # ── WebSocket ──────────────────────────────────────────────────────────────

    def _run_ws_loop(self) -> None:
        """재연결 루프: 연결 끊기면 지수 백오프 후 재시도."""
        while self._running:
            try:
                self._connect_ws()
            except Exception as e:
                logger.error(f"WebSocket 예외: {e}")

            if not self._running:
                break

            logger.warning(f"{self._reconnect_delay}초 후 재연결 시도...")
            time.sleep(self._reconnect_delay)
            # 지수 백오프 (최대 RECONNECT_MAX초)
            self._reconnect_delay = min(
                self._reconnect_delay * 2, RECONNECT_MAX
            )

    def _connect_ws(self) -> None:
        """WebSocket 연결 생성 및 실행."""
        url = self._build_ws_url()
        logger.info(f"WebSocket 연결: {url[:80]}...")

        self._ws = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        # run_forever: ping으로 연결 유지
        self._ws.run_forever(ping_interval=PING_INTERVAL, ping_timeout=10)

    def _build_ws_url(self) -> str:
        """
        구독할 스트림 URL 생성.
        형식: wss://fstream.binance.com/stream?streams=btcusdt@kline_1h/ethusdt@kline_1h/...

        ccxt 심볼("BTC/USDT:USDT") → Binance 스트림명("btcusdt") 변환 포함.
        """
        streams = []
        for symbol in self.symbols:
            # "BTC/USDT:USDT" → "btcusdt"
            raw = symbol.split("/")[0] + "USDT"
            stream_symbol = raw.lower()
            for tf in self.timeframes:
                if tf not in TF_MAP:
                    logger.warning(f"지원하지 않는 타임프레임: {tf}, 건너뜀")
                    continue
                streams.append(f"{stream_symbol}@kline_{TF_MAP[tf]}")

        return WS_BASE_URL + "/".join(streams)

    # ── WebSocket 이벤트 핸들러 ────────────────────────────────────────────────

    def _on_open(self, ws) -> None:
        logger.info("WebSocket 연결 성공")
        self._reconnect_delay = RECONNECT_DELAY  # 성공 시 딜레이 초기화

    def _on_message(self, ws, raw: str) -> None:
        """
        메시지 수신 → 캔들 파싱 → DataFrame 업데이트.

        Binance 복합 스트림 메시지 구조:
        {
            "stream": "btcusdt@kline_1h",
            "data": {
                "e": "kline",
                "k": {
                    "t": 1234567890000,  # 캔들 시작 시각 (ms)
                    "o": "100.0",        # 시가
                    "h": "110.0",        # 고가
                    "l": "95.0",         # 저가
                    "c": "105.0",        # 종가
                    "v": "1000.0",       # 거래량
                    "x": true            # 캔들 닫힘 여부
                }
            }
        }
        """
        try:
            msg  = json.loads(raw)
            data = msg.get("data", {})

            # ← 임시 추가
            logger.debug(f"WS 메시지 수신: stream={msg.get('stream', '')}  e={data.get('e', '')}")

            if data.get("e") != "kline":
                return

            k          = data["k"]
            stream     = msg.get("stream", "")          # "btcusdt@kline_1h"
            symbol_raw = stream.split("@")[0].upper()   # "BTCUSDT"
            tf_raw     = stream.split("_")[-1]          # "1h"

            # Binance 스트림 심볼 → ccxt 심볼 역변환
            symbol = self._stream_to_ccxt_symbol(symbol_raw)

            candle = {
                "timestamp": pd.to_datetime(k["t"], unit="ms"),
                "open":      float(k["o"]),
                "high":      float(k["h"]),
                "low":       float(k["l"]),
                "close":     float(k["c"]),
                "volume":    float(k["v"]),
            }
            is_closed = bool(k["x"])

            # DataFrame 업데이트
            self._update_store(symbol, tf_raw, candle, is_closed)

            # 캔들 닫힘 시 콜백 호출
            if is_closed and self.on_closed_candle:
                df = self.get_df(symbol, tf_raw)
                if df is not None:
                    self.on_closed_candle(symbol, tf_raw, df)

        except Exception as e:
            logger.error(f"메시지 처리 오류: {e}", exc_info=True)

    def _on_error(self, ws, error) -> None:
        logger.error(f"WebSocket 오류: {error}")

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        logger.warning(f"WebSocket 닫힘 (code={close_status_code}, msg={close_msg})")

    # ── DataFrame 업데이트 ─────────────────────────────────────────────────────

    def _update_store(
        self,
        symbol: str,
        timeframe: str,
        candle: dict,
        is_closed: bool,
    ) -> None:
        """
        수신한 캔들로 저장소 DataFrame 업데이트 (thread-safe).

        - 진행 중인 캔들: 마지막 행을 실시간으로 덮어씀
        - 닫힌 캔들    : 새 행 추가 후 오래된 행 제거 (최대 REST_OHLCV_LIMIT 유지)
        """
        with self._store_lock:
            df = self._store.get(symbol, {}).get(timeframe)

            new_row = pd.DataFrame([candle]).set_index("timestamp")

            if df is None or df.empty:
                self._store[symbol][timeframe] = new_row
                return

            if is_closed:
                # 새 캔들 추가
                df = pd.concat([df, new_row])
                df = df[~df.index.duplicated(keep="last")]
                # 최대 행 수 유지 (메모리 관리)
                if len(df) > REST_OHLCV_LIMIT:
                    df = df.iloc[-REST_OHLCV_LIMIT:]
            else:
                # 진행 중 캔들: 마지막 행 갱신
                if df.index[-1] == new_row.index[0]:
                    df.iloc[-1] = new_row.iloc[0]
                else:
                    # 새 타임스탬프면 행 추가
                    df = pd.concat([df, new_row])
                    df = df[~df.index.duplicated(keep="last")]

            self._store[symbol][timeframe] = df

    # ── 유틸 ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_df(ohlcv: list) -> pd.DataFrame:
        """ccxt OHLCV 리스트 → DataFrame 변환."""
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df.astype(float)

    @staticmethod
    def _stream_to_ccxt_symbol(stream_symbol: str) -> str:
        """
        Binance 스트림 심볼 → ccxt 심볼 변환.
        예) "BTCUSDT" → "BTC/USDT:USDT"

        간단한 USDT 무기한 선물 기준. 더 복잡한 쌍은 마켓 테이블 참조 필요.
        """
        if stream_symbol.endswith("USDT"):
            base = stream_symbol[:-4]
            return f"{base}/USDT:USDT"
        return stream_symbol  # 변환 불가 시 원본 반환


# ── 단독 실행 테스트 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    API_KEY    = os.getenv("BINANCE_API_KEY", "")
    API_SECRET = os.getenv("BINANCE_API_SECRET", "")

    # 캔들 닫힘 콜백 — 실제로는 여기서 signal_engine 호출
    def on_closed(symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        latest = df.iloc[-1]
        logger.info(
            f"[캔들 닫힘] {symbol} {timeframe} | "
            f"close={latest['close']:.4f} | vol={latest['volume']:.2f}"
        )

    fetcher = DataFetcher(
        symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
        timeframes=["15m", "1h"],
        on_closed_candle=on_closed,
        api_key=API_KEY,
        api_secret=API_SECRET,
    )

    fetcher.start()

    try:
        print("DataFetcher 실행 중... Ctrl+C 로 종료")
        while True:
            time.sleep(10)
            # 현재 저장된 BTC 1H 마지막 5개 캔들 출력
            df = fetcher.get_df("BTC/USDT:USDT", "1h")
            if df is not None:
                print("\n[ BTC/USDT 1H 최근 5캔들 ]")
                print(df.tail(5).to_string())
    except KeyboardInterrupt:
        fetcher.stop()
        print("종료")