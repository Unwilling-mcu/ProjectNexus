"""
Project Nexus — TouchField Data Bridge
========================================
Receives live player/ball position data from the stadium SAOT WebSocket feed,
normalises the 105m × 68m pitch to the 20×12 haptic grid,
and transmits JSON packets to the Arduino over serial at 30fps.

Run: python touchfield_bridge.py --port /dev/ttyUSB0 --ws ws://stadium-feed:8765

Author  : Sanchayan (Unwilling-mcu)
GitHub  : github.com/Unwilling-mcu/ProjectNexus
"""

import json
import time
import asyncio
import argparse
import threading
import logging
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

# Optional imports — graceful stubs for demo mode
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[Bridge] pyserial not found. Install: pip install pyserial")

try:
    import websockets
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False
    print("[Bridge] websockets not found. Install: pip install websockets")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("TouchField")


# ─────────────────────────────────────────────────────────────────
# 1. CONSTANTS & ENUMS
# ─────────────────────────────────────────────────────────────────

PITCH_WIDTH_M  = 105.0   # FIFA standard length (x-axis)
PITCH_HEIGHT_M =  68.0   # FIFA standard width  (y-axis)

GRID_COLS = 20
GRID_ROWS = 12

# Cell state codes (must match Arduino firmware)
CELL_EMPTY  = 0
CELL_HOME   = 1
CELL_AWAY   = 2
CELL_BALL   = 3
CELL_REF    = 4
CELL_GOAL   = 5

FRAME_INTERVAL_S = 1.0 / 30.0   # 30fps target


# ─────────────────────────────────────────────────────────────────
# 2. DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────

@dataclass
class PlayerPosition:
    player_id: int
    x: float        # metres from left goal line
    y: float        # metres from bottom touchline
    team: str       # "home" | "away" | "referee"

@dataclass
class BallPosition:
    x: float
    y: float
    z: float = 0.0

@dataclass
class MatchState:
    timestamp: float
    players: list
    ball: Optional[BallPosition] = None
    event: str = ""           # "goal", "foul", "offside", etc.
    score: str = "0-0"


# ─────────────────────────────────────────────────────────────────
# 3. COORDINATE NORMALISER
# ─────────────────────────────────────────────────────────────────

class PitchNormaliser:
    """
    Maps real-world pitch coordinates (metres) → grid cell indices.
    Pitch origin is bottom-left corner of the pitch.

    Grid:
      col 0   = left goal line
      col 19  = right goal line
      row 0   = bottom touchline
      row 11  = top touchline
    """

    def __init__(self, cols: int = GRID_COLS, rows: int = GRID_ROWS):
        self.cols = cols
        self.rows = rows

    def to_grid(self, x_m: float, y_m: float) -> tuple[int, int]:
        """Returns (col, row) — both clamped to grid bounds."""
        col = int(np.clip(x_m / PITCH_WIDTH_M  * self.cols, 0, self.cols - 1))
        row = int(np.clip(y_m / PITCH_HEIGHT_M * self.rows, 0, self.rows - 1))
        return col, row

    def mark_goal_areas(self, grid: np.ndarray):
        """Pre-marks the two goal areas as CELL_GOAL for tactile context."""
        # Left goal area: columns 0-1, rows 4-7 (18.32m penalty area scaled)
        grid[4:8, 0:2][grid[4:8, 0:2] == CELL_EMPTY] = CELL_GOAL
        # Right goal area: columns 18-19, rows 4-7
        grid[4:8, 18:20][grid[4:8, 18:20] == CELL_EMPTY] = CELL_GOAL


# ─────────────────────────────────────────────────────────────────
# 4. GRID RENDERER
# ─────────────────────────────────────────────────────────────────

class GridRenderer:
    """
    Converts MatchState → 12×20 numpy array of cell state codes.
    Priority (highest wins when two objects share a cell):
      ball > referee > away > home > goal_area > empty
    """

    def __init__(self):
        self.normaliser = PitchNormaliser()

    def render(self, state: MatchState) -> np.ndarray:
        grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.uint8)
        self.normaliser.mark_goal_areas(grid)

        # Draw players (lower priority first so ball/ref can overwrite)
        for p in state.players:
            col, row = self.normaliser.to_grid(p.x, p.y)
            if p.team == "home":
                grid[row, col] = max(grid[row, col], CELL_HOME)
            elif p.team == "away":
                grid[row, col] = max(grid[row, col], CELL_AWAY)
            elif p.team == "referee":
                grid[row, col] = CELL_REF   # referees always show

        # Draw ball (highest priority)
        if state.ball is not None:
            col, row = self.normaliser.to_grid(state.ball.x, state.ball.y)
            grid[row, col] = CELL_BALL

        return grid


# ─────────────────────────────────────────────────────────────────
# 5. SERIAL TRANSMITTER
# ─────────────────────────────────────────────────────────────────

class SerialTransmitter:
    """
    Sends newline-delimited JSON packets to the Arduino over USB serial.
    Each packet contains the full 12×20 grid + optional event string.
    Packet size: ~500 bytes typically — well within 115200 baud capacity.
    """

    def __init__(self, port: str, baud: int = 115200):
        self.port  = port
        self.baud  = baud
        self._ser  = None
        self._lock = threading.Lock()
        self._connect()

    def _connect(self):
        if not SERIAL_AVAILABLE:
            log.warning("Serial unavailable — packets will be printed to stdout.")
            return
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)   # Arduino resets on serial open
            log.info(f"Serial open: {self.port} @ {self.baud}")
        except Exception as e:
            log.error(f"Serial connect failed: {e}")

    def send(self, grid: np.ndarray, event: str = "", score: str = "0-0",
             timestamp: float = 0.0):
        packet = {
            "t":     round(timestamp, 3),
            "grid":  grid.tolist(),
            "event": event,
            "score": score,
        }
        line = json.dumps(packet, separators=(",", ":")) + "\n"
        encoded = line.encode("utf-8")

        with self._lock:
            if self._ser and self._ser.is_open:
                try:
                    self._ser.write(encoded)
                except Exception as e:
                    log.error(f"Serial write error: {e}")
                    self._ser = None
            else:
                # Fallback: print packet summary to stdout
                ball_count  = int(np.sum(grid == CELL_BALL))
                home_count  = int(np.sum(grid == CELL_HOME))
                away_count  = int(np.sum(grid == CELL_AWAY))
                log.info(f"[PACKET] t={timestamp:.2f} | "
                         f"home={home_count} away={away_count} ball={ball_count}"
                         + (f" | EVENT={event}" if event else ""))

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()


# ─────────────────────────────────────────────────────────────────
# 6. WEBSOCKET DATA SOURCE (Stadium SAOT feed)
# ─────────────────────────────────────────────────────────────────

class SAOTFeedClient:
    """
    Connects to the stadium's SAOT tracking WebSocket and parses
    incoming position messages into MatchState objects.

    Expected message format (subset of real SAOT feed):
    {
      "ts": 1718000000.123,
      "ball": {"x": 52.3, "y": 34.1, "z": 0.8},
      "players": [
        {"id": 1, "x": 30.2, "y": 20.5, "team": "home"},
        {"id": 15, "x": 75.1, "y": 50.3, "team": "away"},
        ...
      ],
      "event": "",
      "score": "1-0"
    }
    """

    def __init__(self, uri: str, queue: asyncio.Queue):
        self.uri   = uri
        self.queue = queue

    async def run(self):
        while True:
            try:
                async with websockets.connect(self.uri) as ws:
                    log.info(f"Connected to SAOT feed: {self.uri}")
                    async for raw in ws:
                        state = self._parse(raw)
                        if state:
                            await self.queue.put(state)
            except Exception as e:
                log.warning(f"SAOT connection lost ({e}). Retrying in 3s...")
                await asyncio.sleep(3)

    def _parse(self, raw: str) -> Optional[MatchState]:
        try:
            d = json.loads(raw)
            players = [
                PlayerPosition(
                    player_id=p["id"],
                    x=float(p["x"]),
                    y=float(p["y"]),
                    team=p.get("team", "home"),
                )
                for p in d.get("players", [])
            ]
            ball = None
            if "ball" in d:
                b = d["ball"]
                ball = BallPosition(x=float(b["x"]),
                                    y=float(b["y"]),
                                    z=float(b.get("z", 0)))
            return MatchState(
                timestamp=float(d.get("ts", time.time())),
                players=players,
                ball=ball,
                event=d.get("event", ""),
                score=d.get("score", "0-0"),
            )
        except Exception as e:
            log.debug(f"Parse error: {e}")
            return None


# ─────────────────────────────────────────────────────────────────
# 7. DEMO DATA GENERATOR (replaces real WS feed for local testing)
# ─────────────────────────────────────────────────────────────────

class DemoFeedGenerator:
    """
    Generates synthetic match state at 30fps for offline testing.
    Ball follows a sinusoidal path; players cluster around it.
    """

    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self._t = 0.0

    async def run(self):
        log.info("[DemoFeed] Running synthetic match simulation...")
        events_schedule = {
            5.0:  "goal",
            12.0: "foul",
            20.0: "offside",
            30.0: "yellow_card",
        }
        last_event = ""

        while True:
            self._t += FRAME_INTERVAL_S

            # Sinusoidal ball trajectory
            bx = 52.5 + 40.0 * np.sin(self._t * 0.3)
            by = 34.0 + 20.0 * np.cos(self._t * 0.5)
            ball = BallPosition(x=bx, y=by)

            # 22 players cluster near ball with jitter
            players = []
            team_labels = ["home"] * 11 + ["away"] * 11
            np.random.seed(int(self._t * 100) % 10000)
            for i, team in enumerate(team_labels):
                side = 1 if team == "home" else -1
                px = np.clip(bx + side * np.random.uniform(5, 25)
                             + np.random.uniform(-8, 8), 1, 104)
                py = np.clip(by + np.random.uniform(-15, 15), 1, 67)
                players.append(PlayerPosition(i, px, py, team))

            # Referee at midfield
            players.append(PlayerPosition(99, 52.5, 34.0, "referee"))

            # Scheduled events
            event = ""
            for t_evt, evt_name in events_schedule.items():
                if abs(self._t - t_evt) < FRAME_INTERVAL_S * 1.5:
                    event = evt_name

            state = MatchState(
                timestamp=self._t,
                players=players,
                ball=ball,
                event=event,
                score="1-0" if self._t > 5.0 else "0-0",
            )
            await self.queue.put(state)
            await asyncio.sleep(FRAME_INTERVAL_S)


# ─────────────────────────────────────────────────────────────────
# 8. MAIN BRIDGE LOOP
# ─────────────────────────────────────────────────────────────────

class TouchFieldBridge:
    """
    Orchestrates data source → renderer → serial transmitter.
    Also exposes a WebSocket server on port 8766 for the companion app.
    """

    def __init__(self, serial_port: str, ws_uri: Optional[str] = None):
        self.queue      = asyncio.Queue(maxsize=10)
        self.renderer   = GridRenderer()
        self.transmitter = SerialTransmitter(serial_port)
        self.ws_uri     = ws_uri
        self._clients: set = set()

    async def _render_loop(self):
        """Consume MatchState from queue, render grid, transmit."""
        while True:
            state: MatchState = await self.queue.get()
            grid = self.renderer.render(state)
            self.transmitter.send(grid,
                                  event=state.event,
                                  score=state.score,
                                  timestamp=state.timestamp)
            await self._broadcast_to_app(grid, state)
            self.queue.task_done()

    async def _broadcast_to_app(self, grid: np.ndarray, state: MatchState):
        """Push state to any connected companion app WebSocket clients."""
        if not self._clients:
            return
        msg = json.dumps({
            "t":     round(state.timestamp, 3),
            "grid":  grid.tolist(),
            "event": state.event,
            "score": state.score,
        })
        disconnected = set()
        for ws in self._clients:
            try:
                await ws.send(msg)
            except Exception:
                disconnected.add(ws)
        self._clients -= disconnected

    async def _ws_app_handler(self, ws, path):
        """Accept companion app connections."""
        self._clients.add(ws)
        log.info(f"Companion app connected: {ws.remote_address}")
        try:
            await ws.wait_closed()
        finally:
            self._clients.discard(ws)

    async def run(self, demo: bool = False):
        tasks = [asyncio.create_task(self._render_loop())]

        if demo or not self.ws_uri or not WS_AVAILABLE:
            feed = DemoFeedGenerator(self.queue)
        else:
            feed = SAOTFeedClient(self.ws_uri, self.queue)

        tasks.append(asyncio.create_task(feed.run()))

        if WS_AVAILABLE:
            app_server = await websockets.serve(
                self._ws_app_handler, "0.0.0.0", 8766
            )
            log.info("Companion app WebSocket server: ws://0.0.0.0:8766")

        log.info("TouchField Bridge running. Press Ctrl+C to stop.")
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self.transmitter.close()
            log.info("Bridge shut down.")


# ─────────────────────────────────────────────────────────────────
# 9. CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TouchField Data Bridge — Nexus Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo mode (no Arduino, no SAOT feed — prints to stdout)
  python touchfield_bridge.py --demo

  # Arduino on COM3 (Windows), demo data
  python touchfield_bridge.py --port COM3 --demo

  # Full live mode: Arduino + real SAOT feed
  python touchfield_bridge.py --port /dev/ttyUSB0 --ws ws://192.168.1.100:8765
        """
    )
    parser.add_argument("--port", default="STDOUT",
                        help="Serial port for Arduino (e.g. /dev/ttyUSB0 or COM3)")
    parser.add_argument("--ws",   default=None,
                        help="SAOT WebSocket URI (omit for demo mode)")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic demo data instead of live feed")
    parser.add_argument("--baud", type=int, default=115200,
                        help="Serial baud rate (default: 115200)")
    args = parser.parse_args()

    bridge = TouchFieldBridge(
        serial_port=args.port,
        ws_uri=args.ws,
    )

    try:
        asyncio.run(bridge.run(demo=args.demo or args.ws is None))
    except KeyboardInterrupt:
        print("\n[Bridge] Stopped by user.")


if __name__ == "__main__":
    main()
