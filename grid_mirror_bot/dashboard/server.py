"""
Dashboard Flask JSON API — Week 2.
Provides REST endpoints for bot state, fills, ratio logs, regime logs,
health checks, and authenticated command execution.
Runs in a daemon thread alongside the async event loop.
"""

import logging
import time
import threading
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import aiosqlite
from flask import Flask, jsonify, request, Response, send_from_directory
from flask_cors import CORS

from core.bot_state import BotState

logger = logging.getLogger("dashboard")

# Database path
DB_PATH = Path(__file__).parent.parent / "db" / "trades.db"


def create_app(
    bot_state: BotState,
    config: Dict[str, Any],
    runner: Optional[Any] = None,
    loop: Optional[Any] = None,
    fill_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    pair_states: Optional[Dict[str, Any]] = None,
    pair_workers: Optional[Dict[str, Any]] = None,
    validation_results: Optional[Dict[str, Any]] = None,
) -> Flask:
    """Factory function to create and configure the Flask application.

    Args:
        bot_state: Shared ``BotState`` instance.
        config: Full application configuration dictionary.
        executor: Optional ``MT5Executor`` for command execution.

    Returns:
        Configured Flask app ready to serve.
    """
    app = Flask(__name__)
    CORS(app)
    executor = runner.executor if runner is not None else None

    dashboard_secret = config.get("DASHBOARD_SECRET", "")
    session_start = bot_state.session_start or time.time()

    # ── Request logging middleware ──────────────────────────────────

    @app.before_request
    def _log_request_start() -> None:
        """Records request start time for latency measurement."""
        request._start_time = time.time()  # type: ignore[attr-defined]

    @app.after_request
    def _log_request_end(response: Response) -> Response:
        """Logs method, path, status, and response time."""
        start = getattr(request, "_start_time", time.time())
        elapsed_ms = (time.time() - start) * 1000
        logger.debug(
            f"{request.method} {request.path} — {response.status_code} — "
            f"{elapsed_ms:.1f}ms"
        )
        return response

    # ── Auth helper ─────────────────────────────────────────────────

    def require_auth(fn: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator enforcing Bearer token authentication."""
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return jsonify({"error": "Missing Authorization header"}), 401
            token = auth_header[len("Bearer "):]
            if token != dashboard_secret:
                return jsonify({"error": "Invalid token"}), 401
            return fn(*args, **kwargs)
        return wrapper

    # ── Endpoints ───────────────────────────────────────────────────

    @app.route("/", methods=["GET"])
    def index() -> tuple[Response, int] | Response:
        """Serves the React dashboard from the static directory."""
        return send_from_directory(
            str(Path(__file__).parent / "static"), "index.html"
        )

    @app.route("/api/status", methods=["GET"])
    def api_status() -> tuple[Response, int]:
        """Returns full bot state as JSON including per-pair data."""
        try:
            response_data = bot_state.to_dict()

            if pair_states:
                response_data["pairs"] = {
                    pair: ps.to_dict() for pair, ps in pair_states.items()
                }

            if executor is not None and loop is not None:
                import asyncio
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        executor.get_all_positions_summary(), loop
                    )
                    response_data["mt5_positions"] = future.result(timeout=5.0)
                except Exception:
                    response_data["mt5_positions"] = {}

            if validation_results:
                response_data["validation"] = {
                    pair: {
                        "available": vr.available,
                        "reason": vr.reason,
                        "spread": vr.spread_points,
                    }
                    for pair, vr in validation_results.items()
                }

            return jsonify(response_data), 200
        except Exception as e:
            logger.error(f"Error in /api/status: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/fills", methods=["GET"])
    def api_fills() -> tuple[Response, int]:
        """Returns the last N fills from the SQLite fills table.

        Query params:
            limit (int): Number of rows to return (default 50).
        """
        try:
            limit = request.args.get("limit", 50, type=int)
            rows = _query_db(
                "SELECT * FROM fills ORDER BY id DESC LIMIT ?", (limit,)
            )
            return jsonify(rows), 200
        except Exception as e:
            logger.error(f"Error in /api/fills: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/ratio", methods=["GET"])
    def api_ratio() -> tuple[Response, int]:
        """Returns the last N ratio log entries.

        Query params:
            limit (int): Number of rows to return (default 100).
        """
        try:
            limit = request.args.get("limit", 100, type=int)
            rows = _query_db(
                "SELECT * FROM ratio_log ORDER BY id DESC LIMIT ?", (limit,)
            )
            return jsonify(rows), 200
        except Exception as e:
            logger.error(f"Error in /api/ratio: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/regime", methods=["GET"])
    def api_regime() -> tuple[Response, int]:
        """Returns the last N regime log entries.

        Query params:
            limit (int): Number of rows to return (default 100).
        """
        try:
            limit = request.args.get("limit", 100, type=int)
            rows = _query_db(
                "SELECT * FROM regime_log ORDER BY id DESC LIMIT ?", (limit,)
            )
            return jsonify(rows), 200
        except Exception as e:
            logger.error(f"Error in /api/regime: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/health", methods=["GET"])
    def api_health() -> tuple[Response, int]:
        """Returns a lightweight health check response."""
        try:
            uptime = time.time() - session_start
            return jsonify({
                "status": "ok",
                "timestamp": int(time.time()),
                "mt5_connected": bot_state.mt5_connected,
                "uptime_seconds": round(uptime, 1),
            }), 200
        except Exception as e:
            logger.error(f"Error in /api/health: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/command", methods=["POST"])
    @require_auth
    def api_command() -> tuple[Response, int]:
        """Executes an authenticated bot command.

        Expects JSON body: ``{"command": "stop"|"start"|"closeall"|"restartgrid"}``
        """
        try:
            body = request.get_json(silent=True) or {}
            command = body.get("command", "").lower().strip()

            if command == "stop":
                bot_state.mirror_enabled = False
                return jsonify({"result": "Mirror disabled — no new orders will be placed."}), 200

            elif command == "start":
                # Week 3: crash-aware start — reset crash monitor if runner has one
                if runner is not None and getattr(runner, "crash_monitor", None) is not None:
                    runner.crash_monitor.reset()
                bot_state.mirror_enabled = True
                return jsonify({"result": "Mirror enabled — crash lockout cleared if active."}), 200

            elif command == "closeall":
                if executor is not None and loop is not None:
                    import asyncio
                    future = asyncio.run_coroutine_threadsafe(
                        executor.close_all_positions(reason="api_command"), loop
                    )
                    try:
                        count = future.result(timeout=30.0)
                    except Exception as e:
                        return jsonify({"error": f"Failed to close positions: {str(e)}"}), 500
                    return jsonify({"result": f"Closed {count} MT5 position(s)."}), 200
                return jsonify({"error": "Executor or event loop not available"}), 503

            elif command == "restartgrid":
                bot_state.grid_restart_due = True
                return jsonify({"result": "Grid restart flag set — will restart on next cycle."}), 200

            elif command == "resetdaily":
                if loop is not None:
                    # 1. Reset bot_state
                    future = asyncio.run_coroutine_threadsafe(bot_state.reset_daily(), loop)
                    try:
                        future.result(timeout=5.0)
                        
                        # 2. Reset CompoundingEngine simulated daily metrics
                        if runner is not None and getattr(runner, "compounding_engine", None) is not None:
                            runner.compounding_engine.reset_daily()
                            
                        # 3. Reset MT5Executor live day-open equity
                        if runner is not None and getattr(runner, "executor", None) is not None:
                            runner.executor.reset_day_open_equity()
                            
                        return jsonify({"result": "Daily counters, compounding metrics, and kill switches reset successfully."}), 200
                    except Exception as e:
                        return jsonify({"error": f"Failed to reset daily state: {str(e)}"}), 500
                else:
                    # Threadsafe fallback if loop not set
                    bot_state.kill_switch_daily = False
                    bot_state.kill_switch_drawdown = False
                    bot_state.fills_today = 0
                    bot_state.mt5_orders_today = 0
                    bot_state.mt5_daily_loss = 0.0
                    bot_state.drift_corrections_today = 0
                    
                    if runner is not None and getattr(runner, "compounding_engine", None) is not None:
                        runner.compounding_engine.reset_daily()
                    if runner is not None and getattr(runner, "executor", None) is not None:
                        runner.executor.reset_day_open_equity()
                        
                    return jsonify({"result": "Daily counters reset (fallback)."}), 200

            elif command == "regime":
                return jsonify({
                    "regime": bot_state.regime,
                    "atr": bot_state.atr,
                    "slope_pct": bot_state.slope_pct,
                    "band_24h": bot_state.band_24h,
                    "btc_price": bot_state.btc_price,
                    "drop_5m_pct": bot_state.drop_5m_pct,
                    "crash_lockout": bot_state.crash_lockout,
                    "mirror_enabled": bot_state.mirror_enabled,
                }), 200

            elif command.startswith("mirror_on_") and pair_states:
                pair = command[len("mirror_on_"):]
                if pair in pair_states:
                    pair_states[pair].mirror_enabled = True
                    pair_states[pair].crash_lockout = False
                    if pair_workers and pair in pair_workers:
                        worker = pair_workers[pair]
                        if hasattr(worker, 'crash_monitor') and worker.crash_monitor:
                            worker.crash_monitor.reset()
                    return jsonify({"result": f"Mirror enabled for {pair}"}), 200
                return jsonify({"error": f"Unknown pair: {pair}"}), 400

            elif command.startswith("mirror_off_") and pair_states:
                pair = command[len("mirror_off_"):]
                if pair in pair_states:
                    pair_states[pair].mirror_enabled = False
                    return jsonify({"result": f"Mirror disabled for {pair}"}), 200
                return jsonify({"error": f"Unknown pair: {pair}"}), 400

            elif command.startswith("close_") and command != "closeall":
                mt5_symbol = command[len("close_"):]
                if executor is not None and loop is not None:
                    import asyncio
                    future = asyncio.run_coroutine_threadsafe(
                        executor.close_all_positions(reason="api_command", symbol=mt5_symbol),
                        loop,
                    )
                    try:
                        count = future.result(timeout=30.0)
                    except Exception as e:
                        return jsonify({"error": f"Failed to close positions: {str(e)}"}), 500
                    return jsonify({"result": f"Closed {count} position(s) for {mt5_symbol}."}), 200
                return jsonify({"error": "Executor or event loop not available"}), 503

            else:
                return jsonify({"error": f"Unknown command: {command}"}), 400

        except Exception as e:
            logger.error(f"Error in /api/command: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/simulate", methods=["POST"])
    @require_auth
    def api_simulate() -> tuple[Response, int]:
        """Simulates a fill by injecting it directly into the running bot's fill pipeline.

        Expects JSON body: ``{"price": 95420.0, "qty": 0.0001, "side": "BUY", "symbol": "BTCUSDT"}``
        """
        try:
            body = request.get_json(silent=True) or {}
            price = body.get("price")
            qty = body.get("qty")
            side = body.get("side", "BUY").upper()
            symbol = body.get("symbol", config.get("binance", {}).get("symbol", "BTCUSDT"))

            if price is None or qty is None:
                return jsonify({"error": "Missing price or qty in request body"}), 400

            import asyncio
            mock_fill = {
                "symbol": symbol,
                "side": side,
                "qty": float(qty),
                "price": float(price),
                "order_id": f"MOCK_API_FILL_{int(time.time())}",
                "timestamp": int(time.time() * 1000),
            }

            if fill_callback is not None and loop is not None:
                future = asyncio.run_coroutine_threadsafe(fill_callback(mock_fill), loop)
                try:
                    # Allow up to 90 seconds for the async fill pipeline to execute
                    future.result(timeout=90.0)
                    return jsonify({
                        "result": "Simulated fill processed successfully via live pipeline.",
                        "fill": mock_fill
                    }), 200
                except (asyncio.TimeoutError, TimeoutError):
                    logger.error("Timeout executing fill_callback via threadsafe (90s limit reached)", exc_info=True)
                    return jsonify({
                        "error": "Failed to execute simulation pipeline: MT5 Execution Timed Out after 90 seconds. The broker might be experiencing high latency, but the order may still complete in the background."
                    }), 504
                except Exception as e:
                    logger.error(f"Error executing fill_callback via threadsafe: {e}", exc_info=True)
                    err_msg = str(e) if str(e) else f"{type(e).__name__} occurred"
                    return jsonify({"error": f"Failed to execute simulation pipeline: {err_msg}"}), 500
            else:
                return jsonify({"error": "Pipeline callback not initialized on server"}), 503

        except Exception as e:
            logger.error(f"Error in /api/simulate: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    return app


# ── Database helper ─────────────────────────────────────────────────

def _query_db(query: str, params: tuple[Any, ...] = ()) -> list[Dict[str, Any]]:
    """Synchronous SQLite query returning a list of dicts.

    Args:
        query: SQL query string with ``?`` placeholders.
        params: Tuple of parameter values.

    Returns:
        List of row dicts with column names as keys.
    """
    import sqlite3
    rows: list[Dict[str, Any]] = []
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
    except Exception as e:
        logger.error(f"DB query error: {e}", exc_info=True)
    return rows


def start_dashboard_thread(
    bot_state: BotState,
    config: Dict[str, Any],
    runner: Optional[Any] = None,
    loop: Optional[Any] = None,
    fill_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    pair_states: Optional[Dict[str, Any]] = None,
    pair_workers: Optional[Dict[str, Any]] = None,
    validation_results: Optional[Dict[str, Any]] = None,
) -> threading.Thread:
    """Creates and starts the Flask dashboard in a daemon thread.

    Args:
        bot_state: Shared ``BotState`` instance.
        config: Full application config.
        runner: Optional ``BotRunner`` instance.
        loop: Optional main event loop to execute async operations thread-safely.
        fill_callback: Optional async fill callback function to invoke for simulation.
        pair_states: Optional dict of PairState instances for multi-pair mode.
        pair_workers: Optional dict of PairWorker instances for multi-pair mode.
        validation_results: Optional dict of symbol validation results.

    Returns:
        The started daemon ``Thread``.
    """
    dashboard_cfg = config.get("dashboard", {})
    host = str(dashboard_cfg.get("host", "127.0.0.1"))
    port = int(dashboard_cfg.get("port", 5000))

    app = create_app(
        bot_state, config, runner, loop, fill_callback,
        pair_states, pair_workers, validation_results,
    )

    def _run() -> None:
        """Runs the Flask app via werkzeug with threading enabled."""
        from werkzeug.serving import make_server
        server = make_server(host, port, app, threaded=True)
        logger.info(f"Dashboard API listening on {host}:{port}")
        server.serve_forever()

    thread = threading.Thread(target=_run, name="dashboard-api", daemon=True)
    thread.start()
    return thread
