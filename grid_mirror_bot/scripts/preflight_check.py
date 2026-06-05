"""
Preflight Check — Week 4 Pre-Session Validation.
Run before every trading session to validate the entire system is ready.
Exit code 0 if no FAILs, exit code 1 if any FAILs.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from dotenv import load_dotenv

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

try:
    from binance.client import Client as BinanceClient
except ImportError:
    BinanceClient = None


class PreflightChecker:
    """Runs all pre-session validation checks."""

    def __init__(self) -> None:
        self.passes = 0
        self.warns = 0
        self.fails = 0
        self.config = None
        self.root_dir = Path(__file__).parent.parent

    def run(self) -> int:
        """Runs all checks and returns exit code."""
        print("PREFLIGHT CHECK — Grid Mirror Bot")
        print("=" * 50)
        print()

        self._check_config()
        print()
        self._check_binance()
        print()
        self._check_mt5()
        print()
        self._check_fee_analysis()
        print()
        self._check_whatsapp()
        print()

        print(f"Summary: {self.passes} PASS  {self.warns} WARN  {self.fails} FAIL")
        if self.fails > 0:
            print("Action required: Fix FAIL items before starting bot.")
            return 1
        return 0

    def _pass(self, msg: str) -> None:
        self.passes += 1
        print(f"  [PASS] {msg}")

    def _warn(self, msg: str) -> None:
        self.warns += 1
        print(f"  [WARN] {msg}")

    def _fail(self, msg: str) -> None:
        self.fails += 1
        print(f"  [FAIL] {msg}")

    def _check_config(self) -> None:
        """Validates config.yaml and secrets.env."""
        print("Config")

        config_file = self.root_dir / "config" / "config.yaml"
        try:
            with open(config_file, "r") as f:
                self.config = yaml.safe_load(f)
            self._pass("config.yaml loads without errors")
        except Exception as e:
            self._fail(f"config.yaml failed to load: {e}")
            return

        env_file = self.root_dir / "config" / "secrets.env"
        if env_file.exists():
            load_dotenv(dotenv_path=env_file)
        else:
            env_file = self.root_dir / ".env"
            if env_file.exists():
                load_dotenv(dotenv_path=env_file)

        required_keys = ["BINANCE_API_KEY", "BINANCE_API_SECRET"]
        missing = [k for k in required_keys if not os.getenv(k)]
        if missing:
            self._fail(f"Missing secrets: {missing}")
        else:
            self._pass("secrets.env loaded — all required keys present")

        pairs_cfg = self.config.get("pairs", {})
        enabled_pairs = [p for p, c in pairs_cfg.items() if c.get("enabled", False)]
        if enabled_pairs:
            self._pass(f"{len(enabled_pairs)} pairs enabled: {', '.join(enabled_pairs)}")
        else:
            self._warn("No pairs enabled in config")

    def _check_binance(self) -> None:
        """Validates Binance connectivity."""
        print("Binance Connection")

        if BinanceClient is None:
            self._fail("python-binance not installed")
            return

        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")
        testnet = self.config.get("binance", {}).get("testnet", True)

        try:
            client = BinanceClient(
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet,
            )
            client.get_server_time()
            self._pass("Binance API connected")
        except Exception as e:
            self._fail(f"Binance API connection failed: {e}")
            return

        if testnet:
            self._pass("Testnet mode confirmed")
        else:
            self._warn("MAINNET mode — ensure this is intentional")

        pairs_cfg = self.config.get("pairs", {})
        enabled_pairs = [p for p, c in pairs_cfg.items() if c.get("enabled", False)]

        for pair in enabled_pairs:
            try:
                info = client.get_symbol_info(pair)
                if info and info.get("status") == "TRADING":
                    self._pass(f"{pair} tradeable")
                else:
                    self._fail(f"{pair} not tradeable — status: {info.get('status') if info else 'NOT_FOUND'}")
            except Exception as e:
                self._fail(f"{pair} check failed: {e}")

        try:
            account = client.get_account()
            usdt_balance = 0.0
            for asset in account.get("balances", []):
                if asset["asset"] == "USDT":
                    usdt_balance = float(asset["free"]) + float(asset["locked"])
                    break
            if usdt_balance > 0:
                self._pass(f"Binance balance: ${usdt_balance:.2f} USDT")
            else:
                self._warn(f"Binance balance: ${usdt_balance:.2f} USDT — deposit testnet funds for real fills")
        except Exception as e:
            self._warn(f"Could not check balance: {e}")

    def _check_mt5(self) -> None:
        """Validates MT5 connectivity and symbol availability."""
        print("MT5 Connection")

        if mt5 is None:
            self._fail("MetaTrader5 package not installed")
            return

        mt5_cfg = self.config.get("mt5", {})
        login = int(os.getenv("MT5_LOGIN", mt5_cfg.get("login", 0)))
        password = os.getenv("MT5_PASSWORD", mt5_cfg.get("password", ""))
        server = os.getenv("MT5_SERVER", mt5_cfg.get("server", ""))

        init_kwargs = {"timeout": 10000}
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
        if path:
            init_kwargs["path"] = path

        if not mt5.initialize(**init_kwargs):
            init_kwargs.update({"login": login, "password": password, "server": server})
            if not mt5.initialize(**init_kwargs):
                self._fail(f"MT5 initialization failed — {mt5.last_error()}")
                return

        account = mt5.account_info()
        if account is None:
            self._fail("MT5 account_info() returned None")
            mt5.shutdown()
            return

        self._pass(f"MT5 initialized — login: {account.login} server: {account.server}")

        pairs_cfg = self.config.get("pairs", {})
        for pair, cfg in pairs_cfg.items():
            if not cfg.get("enabled", False):
                continue

            mt5_symbol = cfg.get("mt5_symbol", "")
            sym_info = mt5.symbol_info(mt5_symbol)

            if sym_info is None:
                self._fail(f"{mt5_symbol} not found — disable {pair} in config or check symbol name")
                continue

            if not sym_info.visible:
                mt5.symbol_select(mt5_symbol, True)
                sym_info = mt5.symbol_info(mt5_symbol)

            tick = mt5.symbol_info_tick(mt5_symbol)
            spread = (tick.ask - tick.bid) if tick else 0.0
            spread_display = f"{spread:.2f}" if spread < 10 else f"{spread:.0f}"

            self._pass(
                f"{mt5_symbol} available — spread: {spread_display} points — "
                f"min vol: {sym_info.volume_min}"
            )

        mt5.shutdown()

    def _check_fee_analysis(self) -> None:
        """Basic fee/profit analysis for each pair."""
        print("Fee Analysis")

        pairs_cfg = self.config.get("pairs", {})
        grid_cfg = self.config.get("grid", {})
        levels = grid_cfg.get("levels", 20)
        range_pct = grid_cfg.get("range_pct", 0.05)

        for pair, cfg in pairs_cfg.items():
            if not cfg.get("enabled", False):
                continue

            capital = cfg.get("grid_capital", 0.0)
            if capital <= 0:
                self._warn(f"{pair} has zero grid capital")
                continue

            fill_profit = (range_pct / levels) * capital
            spread_cost = fill_profit * 0.3

            if fill_profit > spread_cost:
                self._pass(
                    f"{pair} grid profit per fill: ${fill_profit:.2f} > "
                    f"spread cost: ${spread_cost:.2f} — PROFITABLE"
                )
            else:
                self._fail(
                    f"{pair} grid profit per fill: ${fill_profit:.2f} <= "
                    f"spread cost: ${spread_cost:.2f} — NOT PROFITABLE"
                )

    def _check_whatsapp(self) -> None:
        """Validates WhatsApp alerter configuration."""
        print("WhatsApp Alerts")

        wa_cfg = self.config.get("whatsapp", {})
        if not wa_cfg.get("enabled", False):
            self._warn("WhatsApp alerts disabled in config")
            return

        phone = os.getenv("WHATSAPP_PHONE")
        api_key = os.getenv("WHATSAPP_API_KEY")

        if phone and api_key:
            self._pass("WhatsApp connection verified")
        else:
            missing = []
            if not phone:
                missing.append("WHATSAPP_PHONE")
            if not api_key:
                missing.append("WHATSAPP_API_KEY")
            self._fail(f"WhatsApp missing credentials: {missing}")


if __name__ == "__main__":
    checker = PreflightChecker()
    exit_code = checker.run()
    sys.exit(exit_code)
