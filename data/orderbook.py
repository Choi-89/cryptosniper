"""
orderbook.py
------------
Binance Futures 호가창(Order Book)을 실시간으로 유지하고
진입 전 유동성 / 슬리피지 / 매수·매도 압력을 분석하는 모듈.

역할:
  - WebSocket depth 스트림으로 호가창 실시간 업데이트 (로컬 미러링)
  - 진입 전 슬리피지 추정 (목표 수량 기준 체결 시뮬레이션)
  - 매수벽 / 매도벽 두께 분석
  - 호가 불균형 지수(OBI) 계산 → 단기 방향성 힌트
  - get_snapshot() 으로 언제든 현재 호가창 조회

의존 라이브러리:
  pip install ccxt websocket-client python-dotenv
"""

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import websocket  # websocket-client

# ── 로거 ───────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("orderbook")


# ── 상수 ───────────────────────────────────────────────────────────────────────
WS_BASE_URL      = "wss://fstream.binance.com/stream?streams="
DEPTH_LEVEL      = 20          # 호가창 구독 depth (5 / 10 / 20)
UPDATE_SPEED_MS  = 500         # 업데이트 속도 ms (100ms / 250ms / 500ms)
RECONNECT_DELAY  = 5
RECONNECT_MAX    = 60
PING_INTERVAL    = 30
PING_TIMEOUT     = 15

# 슬리피지 경고 임계값
MAX_SLIPPAGE_PCT = 0.003       # 0.3% 초과 시 진입 비권고
# 호가 불균형 지수(OBI) 계산에 사용할 상위 depth
OBI_DEPTH        = 10          # 매수/매도 각 상위 N개 레벨


# ── 데이터 구조 ────────────────────────────────────────────────────────────────

@dataclass
class OrderBookSnapshot:
    """
    호가창 분석 결과 스냅샷.

    Attributes
    ----------
    symbol          : 심볼  예) "BTC/USDT:USDT"
    timestamp       : 스냅샷 생성 시각 (Unix ms)
    best_bid        : 최우선 매수 호가
    best_ask        : 최우선 매도 호가
    spread          : 스프레드 (best_ask - best_bid)
    spread_pct      : 스프레드 비율 (spread / mid_price)
    mid_price       : 중간 가격 ((best_bid + best_ask) / 2)
    bid_wall        : 상위 OBI_DEPTH 레벨 매수 잔량 합계
    ask_wall        : 상위 OBI_DEPTH 레벨 매도 잔량 합계
    obi             : 호가 불균형 지수 (-1.0 ~ +1.0)
                      양수 = 매수 우세, 음수 = 매도 우세
    long_slippage   : 목표 수량 롱 진입 시 예상 슬리피지 비율
    short_slippage  : 목표 수량 숏 진입 시 예상 슬리피지 비율
    is_liquid       : 슬리피지 기준 유동성 충분 여부
    bids            : 매수 호가 리스트 [(가격, 수량), ...]
    asks            : 매도 호가 리스트 [(가격, 수량), ...]
    """
    symbol:         str
    timestamp:      int
    best_bid:       float
    best_ask:       float
    spread:         float
    spread_pct:     float
    mid_price:      float
    bid_wall:       float
    ask_wall:       float
    obi:            float
    long_slippage:  float
    short_slippage: float
    is_liquid:      bool
    bids:           list = field(default_factory=list)
    asks:           list = field(default_factory=list)


# ── 내부 호가창 저장 구조 ──────────────────────────────────────────────────────

class _LocalOrderBook:
    """
    단일 심볼의 호가창을 로컬에서 유지하는 내부 클래스.

    Binance depth 스트림은 전체 스냅샷을 일정 주기로 내려주므로
    (partial book depth stream 방식) 매번 전체를 교체한다.
    """

    def __init__(self):
        # {가격(float): 수량(float)}  — 수량 0이면 삭제
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.last_update_id: int = 0
        self.lock = threading.Lock()

    def update(self, bids_raw: list, asks_raw: list, update_id: int) -> None:
        """
        Binance partial depth 스트림은 매번 top-N 전체를 내려줌.
        수량이 0인 레벨은 삭제, 나머지는 덮어쓰기.
        """
        with self.lock:
            # bids 갱신
            self.bids.clear()
            for price_str, qty_str in bids_raw:
                price, qty = float(price_str), float(qty_str)
                if qty > 0:
                    self.bids[price] = qty

            # asks 갱신
            self.asks.clear()
            for price_str, qty_str in asks_raw:
                price, qty = float(price_str), float(qty_str)
                if qty > 0:
                    self.asks[price] = qty

            self.last_update_id = update_id

    def get_sorted(self, depth: int = DEPTH_LEVEL) -> tuple[list, list]:
        """
        정렬된 호가 리스트 반환 (thread-safe).

        Returns
        -------
        bids : [(가격, 수량), ...] 내림차순 (높은 가격 = 최우선 매수)
        asks : [(가격, 수량), ...] 오름차순 (낮은 가격 = 최우선 매도)
        """
        with self.lock:
            bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:depth]
            asks = sorted(self.asks.items(), key=lambda x: x[0])[:depth]
        return bids, asks


# ── 메인 클래스 ────────────────────────────────────────────────────────────────

class OrderBookManager:
    """
    여러 심볼의 호가창을 WebSocket으로 실시간 유지하고 분석 결과를 제공.

    사용 예시:
        ob = OrderBookManager(symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"])
        ob.start()

        # 진입 전 유동성 체크
        snap = ob.get_snapshot("BTC/USDT:USDT", target_qty=0.01)
        if snap.is_liquid:
            print(f"OBI={snap.obi:.2f}, 슬리피지={snap.long_slippage*100:.3f}%")

        ob.stop()
    """

    def __init__(self, symbols: list[str]):
        """
        Parameters
        ----------
        symbols : 구독할 심볼 목록  예) ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        """
        self.symbols = symbols

        # 심볼별 로컬 호가창
        self._books: dict[str, _LocalOrderBook] = {
            s: _LocalOrderBook() for s in symbols
        }

        # stream_name → ccxt symbol 역매핑  예) "btcusdt" → "BTC/USDT:USDT"
        self._stream_to_symbol: dict[str, str] = {}
        for s in symbols:
            base = s.split("/")[0].lower()
            self._stream_to_symbol[f"{base}usdt"] = s

        # WebSocket
        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._running = False
        self._reconnect_delay = RECONNECT_DELAY

    # ── 퍼블릭 메서드 ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """WebSocket 구독 시작 (백그라운드 스레드)."""
        logger.info("OrderBookManager 시작")
        self._running = True
        self._ws_thread = threading.Thread(
            target=self._run_ws_loop, daemon=True, name="ob-ws-loop"
        )
        self._ws_thread.start()

    def stop(self) -> None:
        """WebSocket 종료."""
        logger.info("OrderBookManager 종료")
        self._running = False
        if self._ws:
            self._ws.close()
        if self._ws_thread:
            self._ws_thread.join(timeout=5)

    def get_snapshot(
        self,
        symbol: str,
        target_qty: float = 0.0,
        depth: int = DEPTH_LEVEL,
    ) -> Optional[OrderBookSnapshot]:
        """
        현재 호가창 분석 스냅샷 반환.

        Parameters
        ----------
        symbol     : 예) "BTC/USDT:USDT"
        target_qty : 슬리피지 추정에 사용할 목표 진입 수량 (코인 단위)
                     0이면 슬리피지 계산 생략 (0.0 반환)
        depth      : 반환할 호가 레벨 수

        Returns
        -------
        OrderBookSnapshot or None (데이터 없을 시)
        """
        book = self._books.get(symbol)
        if book is None:
            logger.warning(f"[{symbol}] 호가창 없음 (구독 목록 확인 필요)")
            return None

        bids, asks = book.get_sorted(depth=depth)

        if not bids or not asks:
            logger.debug(f"[{symbol}] 호가창 비어 있음 (아직 수신 대기 중)")
            return None

        best_bid  = bids[0][0]
        best_ask  = asks[0][0]
        mid_price = (best_bid + best_ask) / 2
        spread    = best_ask - best_bid
        spread_pct = spread / mid_price if mid_price > 0 else 0.0

        # 매수벽 / 매도벽 (상위 OBI_DEPTH 레벨 수량 합계)
        bid_wall = sum(qty for _, qty in bids[:OBI_DEPTH])
        ask_wall = sum(qty for _, qty in asks[:OBI_DEPTH])

        # 호가 불균형 지수 (Order Book Imbalance)
        # OBI = (bid_wall - ask_wall) / (bid_wall + ask_wall)
        # +1.0 = 완전 매수 우세, -1.0 = 완전 매도 우세
        total_wall = bid_wall + ask_wall
        obi = (bid_wall - ask_wall) / total_wall if total_wall > 0 else 0.0

        # 슬리피지 추정
        long_slip  = 0.0
        short_slip = 0.0
        if target_qty > 0:
            long_slip  = self._estimate_slippage(asks, target_qty, mid_price)
            short_slip = self._estimate_slippage(bids, target_qty, mid_price)

        is_liquid = max(long_slip, short_slip) <= MAX_SLIPPAGE_PCT

        return OrderBookSnapshot(
            symbol         = symbol,
            timestamp      = int(time.time() * 1000),
            best_bid       = best_bid,
            best_ask       = best_ask,
            spread         = spread,
            spread_pct     = spread_pct,
            mid_price      = mid_price,
            bid_wall       = bid_wall,
            ask_wall       = ask_wall,
            obi            = round(obi, 4),
            long_slippage  = round(long_slip, 6),
            short_slippage = round(short_slip, 6),
            is_liquid      = is_liquid,
            bids           = bids,
            asks           = asks,
        )

    def is_safe_to_enter(
        self,
        symbol: str,
        direction: str,
        target_qty: float,
    ) -> tuple[bool, str]:
        """
        진입 전 최종 유동성 안전 체크.

        Parameters
        ----------
        symbol     : 예) "BTC/USDT:USDT"
        direction  : "LONG" 또는 "SHORT"
        target_qty : 진입 수량 (코인 단위)

        Returns
        -------
        (safe: bool, reason: str)
        """
        snap = self.get_snapshot(symbol, target_qty=target_qty)

        if snap is None:
            return False, "호가창 데이터 없음"

        # 스프레드 체크
        if snap.spread_pct > MAX_SLIPPAGE_PCT:
            return False, f"스프레드 과다: {snap.spread_pct*100:.3f}%"

        # 슬리피지 체크
        slip = snap.long_slippage if direction == "LONG" else snap.short_slippage
        if slip > MAX_SLIPPAGE_PCT:
            return False, f"슬리피지 과다: {slip*100:.3f}%"

        # OBI 방향 역행 체크 (강한 반대 신호 시 경고)
        if direction == "LONG"  and snap.obi < -0.3:
            return False, f"OBI 매도 우세: {snap.obi:.2f}"
        if direction == "SHORT" and snap.obi >  0.3:
            return False, f"OBI 매수 우세: {snap.obi:.2f}"

        return True, "OK"

    # ── WebSocket ──────────────────────────────────────────────────────────────

    def _run_ws_loop(self) -> None:
        """재연결 루프."""
        while self._running:
            try:
                self._connect_ws()
            except Exception as e:
                logger.error(f"WebSocket 예외: {e}")

            if not self._running:
                break

            logger.warning(f"{self._reconnect_delay}초 후 재연결...")
            time.sleep(self._reconnect_delay)
            self._reconnect_delay = min(self._reconnect_delay * 2, RECONNECT_MAX)

    def _connect_ws(self) -> None:
        url = self._build_ws_url()
        logger.info(f"호가창 WebSocket 연결: {url[:80]}...")

        self._ws = websocket.WebSocketApp(
            url,
            on_open    = self._on_open,
            on_message = self._on_message,
            on_error   = self._on_error,
            on_close   = self._on_close,
        )
        self._ws.run_forever(ping_interval=PING_INTERVAL, ping_timeout=PING_TIMEOUT)

    def _build_ws_url(self) -> str:
        """
        Binance partial book depth 스트림 URL 생성.
        형식: <symbol>@depth<N>@<speed>ms
        예)   btcusdt@depth20@100ms
        """
        streams = []
        for symbol in self.symbols:
            base = symbol.split("/")[0].lower()
            streams.append(f"{base}usdt@depth{DEPTH_LEVEL}@{UPDATE_SPEED_MS}ms")
        return WS_BASE_URL + "/".join(streams)

    def _on_open(self, ws) -> None:
        logger.info("호가창 WebSocket 연결 성공")
        self._reconnect_delay = RECONNECT_DELAY

    def _on_message(self, ws, raw: str) -> None:
        """
        Binance partial depth 메시지 파싱 후 로컬 호가창 업데이트.

        메시지 구조:
        {
            "stream": "btcusdt@depth20@100ms",
            "data": {
                "e": "depthUpdate",
                "E": 1234567890000,   # 이벤트 시각
                "T": 1234567890000,   # 트랜잭션 시각
                "s": "BTCUSDT",
                "U": 157,             # first update ID
                "u": 160,             # final update ID
                "pu": 149,            # last update ID (이전 이벤트)
                "b": [["50000.0", "1.5"], ...],  # 매수 호가
                "a": [["50001.0", "0.8"], ...]   # 매도 호가
            }
        }
        """
        try:
            msg  = json.loads(raw)
            data = msg.get("data", {})

            if data.get("e") not in ("depthUpdate", "depth"):
                return

            stream      = msg.get("stream", "")
            stream_name = stream.split("@")[0]          # "btcusdt"
            symbol      = self._stream_to_symbol.get(stream_name)

            if symbol is None:
                return

            book = self._books.get(symbol)
            if book is None:
                return

            bids_raw  = data.get("b", [])
            asks_raw  = data.get("a", [])
            update_id = data.get("u", 0)

            book.update(bids_raw, asks_raw, update_id)

        except Exception as e:
            logger.error(f"호가창 메시지 처리 오류: {e}", exc_info=True)

    def _on_error(self, ws, error) -> None:
        logger.error(f"호가창 WebSocket 오류: {error}")

    def _on_close(self, ws, code, msg) -> None:
        logger.warning(f"호가창 WebSocket 닫힘 (code={code})")

    # ── 슬리피지 추정 ─────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_slippage(
        levels: list[tuple[float, float]],
        target_qty: float,
        mid_price: float,
    ) -> float:
        """
        목표 수량만큼 시장가 체결 시뮬레이션 후 슬리피지 비율 반환.

        매수(롱) → asks 리스트 전달 (낮은 가격부터 체결)
        매도(숏) → bids 리스트 전달 (높은 가격부터 체결)

        Parameters
        ----------
        levels     : 정렬된 호가 리스트 [(가격, 수량), ...]
        target_qty : 목표 체결 수량 (코인 단위)
        mid_price  : 기준 중간 가격

        Returns
        -------
        float : 슬리피지 비율  예) 0.002 = 0.2%
        """
        if not levels or mid_price <= 0:
            return 0.0

        filled_qty   = 0.0
        filled_cost  = 0.0

        for price, qty in levels:
            remaining = target_qty - filled_qty
            if remaining <= 0:
                break
            take    = min(qty, remaining)
            filled_qty  += take
            filled_cost += take * price

        if filled_qty <= 0:
            return 0.0

        avg_price = filled_cost / filled_qty

        # 슬리피지 = |평균체결가 - 기준가| / 기준가
        slippage = abs(avg_price - mid_price) / mid_price
        return slippage


# ── 단독 실행 테스트 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT"]

    ob = OrderBookManager(symbols=SYMBOLS)
    ob.start()

    print("호가창 수신 중... (3초 대기 후 스냅샷 출력)\n")
    time.sleep(3)

    try:
        while True:
            for sym in SYMBOLS:
                # BTC 기준 0.01 BTC 진입 슬리피지 추정
                snap = ob.get_snapshot(sym, target_qty=0.01)
                if snap:
                    print(
                        f"[{sym}]  "
                        f"bid={snap.best_bid:.2f}  "
                        f"ask={snap.best_ask:.2f}  "
                        f"spread={snap.spread_pct*100:.4f}%  "
                        f"OBI={snap.obi:+.3f}  "
                        f"long_slip={snap.long_slippage*100:.4f}%  "
                        f"liquid={snap.is_liquid}"
                    )

                # 진입 안전 체크
                safe, reason = ob.is_safe_to_enter(sym, "LONG", target_qty=0.01)
                print(f"  → 롱 진입 가능: {safe}  ({reason})\n")

            time.sleep(5)

    except KeyboardInterrupt:
        ob.stop()
        print("종료")
