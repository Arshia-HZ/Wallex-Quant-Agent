# state_manager.py
"""
Portfolio State Persistence and Synchronization Manager.
Ensures positions, entry prices, peak prices, and peak equity survive script restarts.
"""

import os
import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from models import PortfolioState, PositionState, OpenOrderRecord
from config import AgentConfig, ExecutionMode
from wallex_client import WallexClient
from market_compliance import normalize_symbol

logger = logging.getLogger("WallexAgent")


class PortfolioStateManager:
    """
    Manages persistent state in portfolio_state.json with atomic writes.
    Reconciles local state with Wallex live balances in LIVE mode.
    """

    def __init__(self, config: AgentConfig, client: WallexClient):
        self.config = config
        self.client = client
        self.state_file = self.config.portfolio_state_file
        self.state: PortfolioState = self._load_or_init_state()

    def _load_or_init_state(self) -> PortfolioState:
        """Loads existing state from disk or initializes fresh portfolio state."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                state = PortfolioState(**data)
                logger.info(
                    f"Loaded existing portfolio state from {self.state_file}: "
                    f"Equity={state.total_equity_tmn:,.0f} TMN, "
                    f"Positions={list(state.positions.keys())}"
                )
                return state
            except Exception as e:
                logger.error(f"Error loading {self.state_file}, creating fresh state: {e}")

        # Initialize fresh state
        init_state = PortfolioState(
            available_tmn=self.config.initial_capital_tmn,
            locked_tmn=0.0,
            total_equity_tmn=self.config.initial_capital_tmn,
            peak_equity_tmn=self.config.initial_capital_tmn,
            realized_pnl_tmn=0.0,
            total_trades_count=0,
            winning_trades_count=0,
            losing_trades_count=0,
            positions={},
            open_orders=[],
            last_updated=datetime.now(timezone.utc).isoformat(),
        )
        self.save_state(init_state)
        return init_state

    def save_state(self, state: Optional[PortfolioState] = None):
        """Atomically saves state to disk using a temporary file."""
        if state is not None:
            self.state = state
        self.state.last_updated = datetime.now(timezone.utc).isoformat()

        temp_file = f"{self.state_file}.tmp"
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(self.state.model_dump_json(indent=2))
            os.replace(temp_file, self.state_file)
        except Exception as e:
            logger.error(f"Failed to persist state to {self.state_file}: {e}")

    def sync_with_wallex_balances(self, live_prices: Dict[str, float]):
        """
        In LIVE mode, fetches real balances from Wallex and updates cash & equity.
        In SHADOW mode, computes equity from simulated holdings + live prices.
        """
        if self.config.mode == ExecutionMode.LIVE:
            try:
                balances = self.client.get_balances()
                tmn_info = balances.get("TMN", {})
                tmn_total = float(tmn_info.get("value", 0.0))
                tmn_locked = float(tmn_info.get("locked", 0.0))
                self.state.available_tmn = max(0.0, tmn_total - tmn_locked)
                self.state.locked_tmn = tmn_locked

                # Check on-chain / exchange holdings for open positions
                active_symbols = list(self.state.positions.keys())
                for sym in active_symbols:
                    rule = self.client.get_symbol_rule(sym)
                    base_asset = rule.get("base_asset", "")
                    asset_info = balances.get(base_asset, {})
                    actual_qty = float(asset_info.get("value", 0.0))
                    if actual_qty < rule.get("min_qty", 0.00001):
                        # Position was closed externally or liquidated
                        logger.warning(f"Position {sym} no longer held on Wallex. Removing from state.")
                        del self.state.positions[sym]
                    else:
                        self.state.positions[sym].quantity = actual_qty

            except Exception as e:
                logger.error(f"Error synchronizing with Wallex live balances: {e}")

        # Compute total equity = Cash + Value of all open positions
        positions_value_tmn = 0.0
        for sym, pos in self.state.positions.items():
            curr_price = live_prices.get(sym, pos.entry_price)
            positions_value_tmn += pos.quantity * curr_price

        self.state.total_equity_tmn = (
            self.state.available_tmn + self.state.locked_tmn + positions_value_tmn
        )
        self.state.peak_equity_tmn = max(self.state.peak_equity_tmn, self.state.total_equity_tmn)
        self.save_state()

    def record_open_order(self, order: OpenOrderRecord):
        # Remove any existing record with same client_id
        self.state.open_orders = [o for o in self.state.open_orders if o.client_id != order.client_id]
        self.state.open_orders.append(order)
        self.save_state()

    def remove_open_order(self, client_id: str):
        self.state.open_orders = [o for o in self.state.open_orders if o.client_id != client_id]
        self.save_state()

    def get_stale_orders(self, timeout_seconds: int = 180) -> List[OpenOrderRecord]:
        """Identifies open orders placed longer than timeout_seconds ago."""
        now = time.time()
        stale = [
            order for order in self.state.open_orders
            if (now - order.created_at_ts) >= timeout_seconds
        ]
        return stale

    def open_position(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: Optional[float],
        fee_tmn: float = 0.0,
    ):
        """Records a new open position and updates available TMN."""
        norm = normalize_symbol(symbol)
        total_cost = quantity * entry_price
        pos = PositionState(
            symbol=norm,
            quantity=quantity,
            entry_price=entry_price,
            peak_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            trailing_stop_active=False,
            opened_at=datetime.now(timezone.utc).isoformat(),
            total_cost_tmn=total_cost,
            accumulated_fees_tmn=fee_tmn,
        )
        self.state.positions[norm] = pos

        # Deduct cost from cash in simulated mode (in live mode sync_with_wallex_balances handles it)
        if self.config.mode != ExecutionMode.LIVE:
            self.state.available_tmn = max(0.0, self.state.available_tmn - total_cost - fee_tmn)

        self.save_state()
        logger.info(
            f"OPENED POSITION: {norm} Qty={quantity} @ {entry_price:,.0f} TMN | "
            f"SL={stop_loss_price:,.0f} TMN"
        )

    def update_position_peak(
        self,
        symbol: str,
        current_price: float,
        activation_pct: float = 0.010,
        distance_pct: float = 0.007,
    ) -> bool:
        """
        Updates peak price for trailing stop.
        If current price >= entry * (1 + activation_pct), activates trailing stop
        and ratchets stop_loss_price upward.
        Returns True if stop_loss was ratcheted upward.
        """
        norm = normalize_symbol(symbol)
        pos = self.state.positions.get(norm)
        if not pos:
            return False

        ratcheted = False
        if current_price > pos.peak_price:
            pos.peak_price = current_price

        gain_pct = (pos.peak_price - pos.entry_price) / pos.entry_price
        if gain_pct >= activation_pct:
            pos.trailing_stop_active = True
            new_sl = pos.peak_price * (1.0 - distance_pct)
            if new_sl > pos.stop_loss_price:
                pos.stop_loss_price = new_sl
                ratcheted = True
                logger.info(
                    f"TRAILING STOP RATCHET: {norm} New SL={new_sl:,.0f} TMN "
                    f"(Peak={pos.peak_price:,.0f}, Gain={gain_pct:.2%})"
                )

        self.save_state()
        return ratcheted

    def close_position(
        self, symbol: str, exit_price: float, fee_tmn: float = 0.0
    ) -> tuple[float, float]:
        """
        Closes position, realizes PnL in TMN, and updates win/loss statistics.
        Returns (realized_pnl_tmn, pnl_pct).
        """
        norm = normalize_symbol(symbol)
        pos = self.state.positions.get(norm)
        if not pos:
            return 0.0, 0.0

        revenue_tmn = pos.quantity * exit_price
        gross_pnl_tmn = revenue_tmn - pos.total_cost_tmn
        net_pnl_tmn = gross_pnl_tmn - pos.accumulated_fees_tmn - fee_tmn
        pnl_pct = (net_pnl_tmn / pos.total_cost_tmn) if pos.total_cost_tmn > 0 else 0.0

        # Update portfolio state
        self.state.realized_pnl_tmn += net_pnl_tmn
        self.state.total_trades_count += 1
        if net_pnl_tmn > 0:
            self.state.winning_trades_count += 1
        else:
            self.state.losing_trades_count += 1

        if self.config.mode != ExecutionMode.LIVE:
            self.state.available_tmn += (revenue_tmn - fee_tmn)

        del self.state.positions[norm]
        self.save_state()

        logger.info(
            f"CLOSED POSITION: {norm} Exit={exit_price:,.0f} TMN | "
            f"Net PnL={net_pnl_tmn:+,.0f} TMN ({pnl_pct:+.2%})"
        )
        return net_pnl_tmn, pnl_pct
