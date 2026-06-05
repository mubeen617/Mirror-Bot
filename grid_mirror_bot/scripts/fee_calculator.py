"""
Fee Calculator — Week 4 Spread vs Profit Analysis.
Calculates whether the grid strategy is profitable after MT5 spread costs.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from dotenv import load_dotenv

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False


def load_config() -> dict:
    """Loads configuration and secrets."""
    root_dir = Path(__file__).parent.parent
    config_file = root_dir / "config" / "config.yaml"

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    env_file = root_dir / "config" / "secrets.env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
    else:
        env_file = root_dir / ".env"
        if env_file.exists():
            load_dotenv(dotenv_path=env_file)

    return config


def get_live_price(symbol: str) -> float:
    """Fetches live price from public Binance API."""
    import urllib.request
    import json

    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        with urllib.request.urlopen(url, timeout=5.0) as response:
            data = json.loads(response.read().decode("utf-8"))
            return float(data.get("price", 0.0))
    except Exception:
        return 0.0


def get_mt5_spread(mt5_symbol: str) -> float:
    """Fetches spread from MT5 if connected."""
    if not MT5_AVAILABLE or mt5 is None:
        return 0.0

    tick = mt5.symbol_info_tick(mt5_symbol)
    if tick is not None:
        return tick.ask - tick.bid
    return 0.0


def get_mt5_volume_step(mt5_symbol: str) -> float:
    """Fetches volume step from MT5."""
    if not MT5_AVAILABLE or mt5 is None:
        return 0.01

    sym_info = mt5.symbol_info(mt5_symbol)
    if sym_info is not None:
        return sym_info.volume_step
    return 0.01


def run_analysis() -> None:
    """Runs the fee drag analysis for all configured pairs."""
    config = load_config()
    pairs_cfg = config.get("pairs", {})
    grid_cfg = config.get("grid", {})
    levels = grid_cfg.get("levels", 20)
    range_pct = grid_cfg.get("range_pct", 0.05)

    mt5_connected = False
    if MT5_AVAILABLE:
        mt5_cfg = config.get("mt5", {})
        path = mt5_cfg.get("path", "")
        if not path:
            default_paths = [
                "C:/Program Files/MetaTrader 5 IC Markets Global/terminal64.exe",
                "C:/Program Files/MetaTrader 5/terminal64.exe",
            ]
            for p in default_paths:
                if Path(p).exists():
                    path = p
                    break

        init_kwargs = {"timeout": 10000}
        if path:
            init_kwargs["path"] = path
        mt5_connected = mt5.initialize(**init_kwargs)

    print("FEE DRAG ANALYSIS")
    print("=" * 80)
    print(
        f"{'Pair':<12}{'Grid':<8}{'Levels':<8}{'Range':<8}"
        f"{'Fill Profit':<14}{'Spread Cost':<14}{'Net/Fill':<12}{'Profitable':<12}"
    )
    print("-" * 80)

    all_profitable = True

    for pair, cfg in pairs_cfg.items():
        capital = cfg.get("grid_capital", 0.0)
        mt5_symbol = cfg.get("mt5_symbol", "")

        current_price = get_live_price(pair)
        if current_price <= 0:
            current_price = 95000.0 if "BTC" in pair else 3500.0 if "ETH" in pair else 0.5

        fill_profit = (range_pct / levels) * capital

        if mt5_connected:
            spread = get_mt5_spread(mt5_symbol)
            volume_step = get_mt5_volume_step(mt5_symbol)
        else:
            spread = current_price * 0.0001
            volume_step = 0.01

        scaled_qty = capital / levels / current_price
        spread_cost = spread * max(scaled_qty, volume_step)

        net_per_fill = fill_profit - spread_cost
        profitable = net_per_fill > 0

        if not profitable:
            all_profitable = False

        status = "YES" if profitable else "NO"
        emoji = "  " if profitable else "  "

        print(
            f"{pair:<12}${capital:<7.0f}{levels:<8}{range_pct*100:.0f}%{'':3}"
            f"${fill_profit:<13.2f}${spread_cost:<13.2f}"
            f"{'+'if net_per_fill>=0 else ''}{f'${net_per_fill:.2f}':<11}{emoji}{status}"
        )

    print("=" * 80)
    print()
    print(
        "Formula: Fill profit = (range_pct / levels) * capital"
    )
    print(
        "         Spread cost = spread_points * volume * current_price (at MT5 scale)"
    )
    print("=" * 80)

    if all_profitable:
        print("Recommendation: All pairs profitable. Proceed with current settings.")
    else:
        print(
            "Recommendation: Some pairs NOT profitable. "
            "Consider increasing grid capital or reducing levels."
        )

    if mt5_connected:
        mt5.shutdown()


if __name__ == "__main__":
    run_analysis()
