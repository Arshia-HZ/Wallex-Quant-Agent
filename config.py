# config.py
import os
from enum import Enum
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().strip('"').strip("'").lower() in ("1", "true", "yes", "y", "on")


class ExecutionMode(str, Enum):
    BACKTEST = "BACKTEST"
    SHADOW = "SHADOW"
    LIVE = "LIVE"


@dataclass
class AgentConfig:
    api_key: str = field(default_factory=lambda: os.getenv("WALLEX_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.getenv("WALLEX_BASE_URL", "https://api.wallex.ir").rstrip("/")
    )
    mode: ExecutionMode = field(
        default_factory=lambda: ExecutionMode(os.getenv("EXECUTION_MODE", "SHADOW").upper())
    )

    # Capital & Risk Guardrails (60 Points Total: 30 Risk Mgmt + 30 Drawdown Control)
    initial_capital_tmn: float = field(
        default_factory=lambda: float(os.getenv("INITIAL_CAPITAL_TMN", "5000000.0"))
    )
    max_position_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_POSITION_PCT", "0.22"))
    )
    min_cash_reserve_pct: float = field(
        default_factory=lambda: float(os.getenv("MIN_CASH_RESERVE_PCT", "0.20"))
    )
    max_concurrent_positions: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_POSITIONS", "3"))
    )
    exchange_min_notional_tmn: float = field(
        default_factory=lambda: float(os.getenv("EXCHANGE_MIN_NOTIONAL_TMN", "100000.0"))
    )
    min_order_tmn: float = field(
        default_factory=lambda: float(os.getenv("MIN_ORDER_TMN", "150000.0"))
    )
    # Wallex Credit Account Fee Schedule for TMN base markets: Maker 0.25%, Taker 0.30%
    maker_fee_rate: float = field(
        default_factory=lambda: float(os.getenv("MAKER_FEE_RATE", "0.0025"))
    )
    taker_fee_rate: float = field(
        default_factory=lambda: float(os.getenv("TAKER_FEE_RATE", "0.0030"))
    )
    fee_rate_per_side: float = field(
        default_factory=lambda: float(os.getenv("FEE_RATE", "0.0030"))
    )

    # Stop-Loss, Take-Profit & Trailing Stop Parameters
    hard_stop_loss_pct: float = field(
        default_factory=lambda: float(os.getenv("HARD_STOP_LOSS_PCT", "0.014"))
    )
    # Floor for the stop distance: a stop tighter than this sits inside normal 15m noise
    # and costs (stop + 0.55% round-trip fees) on every stop-out.
    min_stop_loss_pct: float = field(
        default_factory=lambda: float(os.getenv("MIN_STOP_LOSS_PCT", "0.008"))
    )
    default_take_profit_pct: float = field(
        default_factory=lambda: float(os.getenv("DEFAULT_TAKE_PROFIT_PCT", "0.018"))
    )
    trailing_activation_pct: float = field(
        default_factory=lambda: float(os.getenv("TRAILING_STOP_ACTIVATION_PCT", "0.012"))
    )
    trailing_distance_pct: float = field(
        default_factory=lambda: float(os.getenv("TRAILING_STOP_DISTANCE_PCT", "0.006"))
    )
    max_portfolio_drawdown_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_PORTFOLIO_DRAWDOWN_PCT", "0.025"))
    )

    # Trade-flow guards (fee-drag protection)
    force_market_entries: bool = field(
        default_factory=lambda: _env_bool("FORCE_MARKET_ENTRIES", True)
    )  # maker/taker gap is only 0.05%; passive bids suffer adverse selection
    reentry_cooldown_sec: int = field(
        default_factory=lambda: int(os.getenv("REENTRY_COOLDOWN_SECONDS", "2700"))
    )
    max_consecutive_losses: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONSECUTIVE_LOSSES", "2"))
    )
    loss_pause_sec: int = field(
        default_factory=lambda: int(os.getenv("LOSS_PAUSE_SECONDS", "3600"))
    )
    no_new_entries_before_end_sec: int = field(
        default_factory=lambda: int(os.getenv("NO_NEW_ENTRIES_MINUTES_BEFORE_END", "90")) * 60
    )
    time_stop_sec: int = field(
        default_factory=lambda: int(os.getenv("TIME_STOP_SECONDS", "14400"))
    )  # 0 disables
    time_stop_min_pnl_pct: float = field(
        default_factory=lambda: float(os.getenv("TIME_STOP_MIN_PNL_PCT", "0.005"))
    )
    btc_regime_filter: bool = field(
        default_factory=lambda: _env_bool("BTC_REGIME_FILTER", True)
    )

    # Microstructure & Cadence
    candle_resolution: str = field(
        default_factory=lambda: os.getenv("CANDLE_RESOLUTION", "15")
    )
    min_atr_pct: float = field(
        default_factory=lambda: float(os.getenv("MIN_ATR_PCT", "0.22"))
    )  # Blocks dead/flat markets that cannot beat 0.55%-0.60% round-trip fee
    max_spread_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_SPREAD_PCT", "0.55"))
    )
    loop_interval_sec: int = field(
        default_factory=lambda: int(os.getenv("LOOP_INTERVAL_SECONDS", "45"))
    )
    position_check_interval_sec: int = field(
        default_factory=lambda: int(os.getenv("POSITION_CHECK_INTERVAL_SECONDS", "8"))
    )  # fast stop-loss loop (order book only)
    candle_cache_sec: int = field(
        default_factory=lambda: int(os.getenv("CANDLE_CACHE_SECONDS", "60"))
    )
    fetch_workers: int = field(
        default_factory=lambda: int(os.getenv("FETCH_WORKERS", "6"))
    )
    stale_order_timeout_sec: int = field(
        default_factory=lambda: int(os.getenv("STALE_ORDER_TIMEOUT_SECONDS", "600"))
    )
    llm_hold_cooldown_sec: int = field(
        default_factory=lambda: int(os.getenv("LLM_HOLD_COOLDOWN_SECONDS", "240"))
    )
    contest_end_timestamp: int = field(
        default_factory=lambda: int(os.getenv("CONTEST_END_TIMESTAMP", "0"))
    )
    flatten_before_end_sec: int = field(
        default_factory=lambda: int(os.getenv("FLATTEN_MINUTES_BEFORE_END", "30")) * 60
    )

    # Deliverable Files (use different files per mode so SHADOW state never leaks into LIVE)
    decision_log_file: str = field(
        default_factory=lambda: os.getenv("DECISION_LOG_FILE", "decision_logs.jsonl")
    )
    state_file: str = field(
        default_factory=lambda: os.getenv("PORTFOLIO_STATE_FILE", "portfolio_state.json")
    )

    # LLM Settings
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://ai.boxapi.ir/ai/v1")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gpt-6-luna")
    )
    llm_reasoning_effort: str = field(
        default_factory=lambda: os.getenv("LLM_REASONING_EFFORT", "medium")
    )

    # Aliases for compatibility with backtest_engine.py / state_manager.py / shadow_simulator.py
    @property
    def fee_rate(self) -> float:
        return self.fee_rate_per_side

    @property
    def portfolio_state_file(self) -> str:
        return self.state_file

    @property
    def trailing_stop_activation_pct(self) -> float:
        return self.trailing_activation_pct

    @property
    def trailing_stop_distance_pct(self) -> float:
        return self.trailing_distance_pct