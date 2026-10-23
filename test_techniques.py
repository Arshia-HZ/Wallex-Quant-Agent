# test_techniques.py
"""
Comprehensive verification test suite for the 5 implemented techniques:
1. Exact Cash Cost Tracking (Phantom Loss fix)
2. Asynchronous Sell Order Tracking & Clean Position Popping (with Partial & Zero Fill support)
3. Stop-Loss Spread Guard & Liquidity Hole Protection
4. Dynamic Kelly / Confidence-Based Sizing
5. Sniper Mode Entry Hard Vetoes
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock
from models import AIDecisionOutput
from config import AgentConfig, ExecutionMode
from llm_strategist import CostEffectiveLLMStrategist
from main import OpenPosition, PortfolioAndRiskManager, TradingAgentRunner, _positive_float


class BaseIsolatedTestCase(unittest.TestCase):
    """Provides an isolated temporary directory for state and log files so tests never corrupt live files."""
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="wallex_test_")
        self.state_file = os.path.join(self.test_dir, "test_portfolio_state.json")
        self.log_file = os.path.join(self.test_dir, "test_decisions.jsonl")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)


class TestTechnique1ExactCashCostTracking(BaseIsolatedTestCase):
    def setUp(self):
        super().setUp()
        self.config = AgentConfig(
            mode=ExecutionMode.SHADOW,
            initial_capital_tmn=5_000_000.0,
            state_file=self.state_file,
            decision_log_file=self.log_file,
        )
        self.runner = TradingAgentRunner(self.config)

    def test_exact_buy_cost_tracking_not_order_tmn(self):
        """Verify cost_tmn is set to actual executed cost, not theoretical order_tmn."""
        sym = "BTCTMN"
        feats = {"best_ask": 5_000_000_000.0, "best_bid": 4_990_000_000.0, "atr_14": 50_000_000.0}
        decision = AIDecisionOutput(
            symbol=sym,
            action="BUY",
            order_type="MARKET",
            confidence=0.85,
            allocation_pct=0.35,
            reasoning_fa="تست ورود با اندازه دقیق",
        )

        order_tmn = 1_750_000.0  # 35% of 5M
        total_eq = 5_000_000.0

        self.runner.client.market_rules[sym] = {"min_qty": 0.00001, "step_size": 0.00001}
        self.runner.client.format_quantity = MagicMock(return_value="0.00034")

        mock_res = {
            "status": "SHADOW_SUBMITTED",
            "executedSum": 1_705_100.0,  # actual executed money spent
            "clientOrderId": "wlx_test_buy_01",
        }
        self.runner.client.place_spot_order = MagicMock(return_value=mock_res)

        self.runner._execute_buy(sym, feats, decision, order_tmn, "APPROVED_TEST", total_eq)

        pos = self.runner.risk_mgr.positions.get(sym)
        self.assertIsNotNone(pos)
        self.assertEqual(pos.cost_tmn, 1_705_100.0)
        self.assertNotEqual(pos.cost_tmn, order_tmn)

    def test_buy_cost_handles_string_zero_gracefully(self):
        """When executedSum is '0' or '0.00000000', fallback calculation must be used instead of 0.0."""
        sym = "BTCTMN"
        feats = {"best_ask": 5_000_000_000.0, "best_bid": 4_990_000_000.0, "atr_14": 50_000_000.0}
        decision = AIDecisionOutput(
            symbol=sym,
            action="BUY",
            order_type="MARKET",
            confidence=0.85,
            allocation_pct=0.35,
            reasoning_fa="تست صفر رشته‌ای",
        )

        self.runner.client.market_rules[sym] = {"min_qty": 0.00001, "step_size": 0.00001}
        self.runner.client.format_quantity = MagicMock(return_value="0.00034")

        mock_res = {
            "status": "NEW",
            "executedSum": "0",  # String zero from exchange
            "clientOrderId": "wlx_test_buy_02",
        }
        self.runner.client.place_spot_order = MagicMock(return_value=mock_res)

        self.runner._execute_buy(sym, feats, decision, 1_750_000.0, "APPROVED_TEST", 5_000_000.0)

        pos = self.runner.risk_mgr.positions.get(sym)
        self.assertIsNotNone(pos)
        # 0.00034 * 5,000,000,000 * 1.0030 = 1,705,100 TMN
        expected_fallback = 0.00034 * 5_000_000_000.0 * (1.0 + self.config.taker_fee_rate)
        self.assertAlmostEqual(pos.cost_tmn, expected_fallback, places=0)
        self.assertGreater(pos.cost_tmn, 0.0)

    def test_sell_pnl_uses_exact_cost_tmn(self):
        """Verify sell PnL uses pos.cost_tmn and not theoretical entry allocation."""
        sym = "BTCTMN"
        pos = OpenPosition(
            symbol=sym,
            entry_price=5_000_000_000.0,
            quantity=0.00034,
            stop_loss_price=4_900_000_000.0,
            take_profit_price=5_200_000_000.0,
            highest_price=5_000_000_000.0,
            opened_at="2026-10-23T12:00:00Z",
            client_id="wlx_test_01",
            cost_tmn=1_705_100.0,  # exact executed cost
            entry_fee_rate=0.0030,
        )
        self.runner.risk_mgr.positions[sym] = pos

        feats = {"best_bid": 5_100_000_000.0, "best_ask": 5_110_000_000.0}
        decision = AIDecisionOutput(
            symbol=sym,
            action="SELL",
            order_type="MARKET",
            confidence=0.90,
            allocation_pct=0.0,
            reasoning_fa="تست خروج با سود",
        )

        mock_sell_res = {
            "status": "SHADOW_SUBMITTED",
            "executedSum": 1_734_000.0,
            "clientOrderId": "wlx_test_sell_01",
        }
        self.runner.client.place_spot_order = MagicMock(return_value=mock_sell_res)
        self.runner.client.format_quantity = MagicMock(return_value="0.00034")

        self.runner._execute_sell(sym, pos, feats, decision, "TEST_EXIT", 5_000_000.0)

        expected_pnl = (1_734_000.0 * (1.0 - 0.0030)) - 1_705_100.0
        self.assertAlmostEqual(self.runner.risk_mgr.realized_pnl_tmn, expected_pnl, places=1)


class TestTechnique2AsynchronousSellOrderTracking(BaseIsolatedTestCase):
    def setUp(self):
        super().setUp()
        self.config = AgentConfig(
            mode=ExecutionMode.LIVE,
            initial_capital_tmn=5_000_000.0,
            state_file=self.state_file,
            decision_log_file=self.log_file,
        )
        self.runner = TradingAgentRunner(self.config)

    def test_pending_sell_order_does_not_prematurely_pop_position(self):
        """When Wallex returns status NEW / PARTIALLY_FILLED, position must stay in positions."""
        sym = "ETHTMN"
        pos = OpenPosition(
            symbol=sym,
            entry_price=200_000_000.0,
            quantity=0.01,
            stop_loss_price=195_000_000.0,
            take_profit_price=210_000_000.0,
            highest_price=200_000_000.0,
            opened_at="2026-10-23T12:00:00Z",
            client_id="wlx_buy_eth",
            cost_tmn=2_006_000.0,
            entry_fee_rate=0.0030,
        )
        self.runner.risk_mgr.positions[sym] = pos

        feats = {"best_bid": 194_000_000.0, "best_ask": 194_500_000.0}
        decision = AIDecisionOutput(
            symbol=sym,
            action="SELL",
            order_type="MARKET",
            confidence=1.0,
            allocation_pct=0.0,
            reasoning_fa="تست خروج با اردر در انتظار",
        )

        self.runner.client.get_free_asset_balance = MagicMock(return_value=0.01)
        self.runner.client.market_rules[sym] = {"min_qty": 0.0001, "base_asset": "ETH"}
        self.runner.client.format_quantity = MagicMock(return_value="0.01")

        mock_res = {
            "status": "NEW",
            "clientOrderId": "wlx_sell_eth_01",
        }
        self.runner.client.place_spot_order = MagicMock(return_value=mock_res)

        self.runner._execute_sell(sym, pos, feats, decision, "STOP_LOSS", 5_000_000.0)

        # Position MUST NOT be prematurely popped!
        self.assertIn(sym, self.runner.risk_mgr.positions)
        self.assertIn(sym, self.runner.risk_mgr.pending_sell_orders)
        self.assertEqual(self.runner.risk_mgr.pending_sell_orders[sym]["client_id"], "wlx_sell_eth_01")

    def test_compute_equity_no_orphan_balance_for_pending_sell(self):
        """Ensure compute_equity does not flag false ORPHAN BALANCE while sell is pending."""
        sym = "ETHTMN"
        self.runner.client.market_rules[sym] = {"base_asset": "ETH", "min_qty": 0.0001}
        self.runner.risk_mgr.pending_sell_orders[sym] = {"client_id": "wlx_sell_eth_01"}

        self.runner.client.get_balances = MagicMock(return_value={
            "TMN": {"value": "3000000", "locked": "0"},
            "ETH": {"value": "0.01", "locked": "0.01"},
        })

        self.runner.risk_mgr._orphan_warned.clear()
        avail, total = self.runner.risk_mgr.compute_equity(self.runner.client, {sym: 200_000_000.0})

        self.assertNotIn(sym, self.runner.risk_mgr._orphan_warned)

    def test_sweep_stale_orders_finalizes_sell_when_filled(self):
        """When sell order confirms FILLED, position is popped and PnL is realized."""
        sym = "ETHTMN"
        pos = OpenPosition(
            symbol=sym,
            entry_price=200_000_000.0,
            quantity=0.01,
            stop_loss_price=195_000_000.0,
            take_profit_price=210_000_000.0,
            highest_price=200_000_000.0,
            opened_at="2026-10-23T12:00:00Z",
            client_id="wlx_buy_eth",
            cost_tmn=2_006_000.0,
        )
        self.runner.risk_mgr.positions[sym] = pos
        cid = "wlx_sell_eth_01"
        self.runner.risk_mgr.pending_sell_orders[sym] = {
            "client_id": cid,
            "symbol": sym,
            "sell_qty": 0.01,
            "exec_price": 205_000_000.0,
            "cost_part": 2_006_000.0,
            "cost_full": 2_006_000.0,
            "frac": 1.0,
            "exit_reason": "TAKE_PROFIT",
            "decision": AIDecisionOutput(symbol=sym, action="SELL", order_type="MARKET", confidence=1.0, allocation_pct=0.0, reasoning_fa="تست"),
            "feats": {"best_bid": 205_000_000.0},
            "total_eq": 5_000_000.0,
        }
        self.runner.pending_order_timestamps[cid] = 1000.0

        self.runner.client.get_open_orders = MagicMock(return_value=[])
        self.runner.client.get_free_asset_balance = MagicMock(return_value=0.0)
        self.runner.client.market_rules[sym] = {"min_qty": 0.0001, "base_asset": "ETH"}
        self.runner.client.get_order_by_client_id = MagicMock(return_value={
            "status": "FILLED",
            "executedSum": 2_050_000.0,
            "executedQty": "0.01",
        })

        self.runner.sweep_stale_open_orders()

        self.assertNotIn(sym, self.runner.risk_mgr.positions)
        self.assertNotIn(sym, self.runner.risk_mgr.pending_sell_orders)
        self.assertEqual(self.runner.risk_mgr.total_trades_count, 1)
        self.assertGreater(self.runner.risk_mgr.realized_pnl_tmn, 0.0)

    def test_sweep_stale_orders_handles_partial_fill_cleanly(self):
        """When sell order is partially filled (0.004 of 0.01) before cancel, PnL is realized for 0.004 and remaining 0.006 stays tracked."""
        sym = "ETHTMN"
        pos = OpenPosition(
            symbol=sym,
            entry_price=200_000_000.0,
            quantity=0.01,
            stop_loss_price=195_000_000.0,
            take_profit_price=210_000_000.0,
            highest_price=200_000_000.0,
            opened_at="2026-10-23T12:00:00Z",
            client_id="wlx_buy_eth",
            cost_tmn=2_006_000.0,
        )
        self.runner.risk_mgr.positions[sym] = pos
        cid = "wlx_sell_eth_partial"
        self.runner.risk_mgr.pending_sell_orders[sym] = {
            "client_id": cid,
            "symbol": sym,
            "sell_qty": 0.01,
            "exec_price": 205_000_000.0,
            "cost_part": 2_006_000.0,
            "cost_full": 2_006_000.0,
            "frac": 1.0,
            "exit_reason": "TAKE_PROFIT",
            "decision": AIDecisionOutput(symbol=sym, action="SELL", order_type="MARKET", confidence=1.0, allocation_pct=0.0, reasoning_fa="تست"),
            "feats": {"best_bid": 205_000_000.0},
            "total_eq": 5_000_000.0,
        }
        self.runner.pending_order_timestamps[cid] = 1000.0

        self.runner.client.get_open_orders = MagicMock(return_value=[])
        self.runner.client.get_free_asset_balance = MagicMock(return_value=0.006)  # 0.006 returned to wallet
        self.runner.client.market_rules[sym] = {"min_qty": 0.0001, "base_asset": "ETH"}
        self.runner.client.get_order_by_client_id = MagicMock(return_value={
            "status": "CANCELED",
            "executedQty": "0.004",
            "executedSum": "820000.0",
        })

        self.runner.sweep_stale_open_orders()

        # Partial fill must be booked: 1 trade recorded!
        self.assertEqual(self.runner.risk_mgr.total_trades_count, 1)
        self.assertGreater(self.runner.risk_mgr.realized_pnl_tmn, 0.0)

        # Remaining position must stay tracked with reduced quantity and cost
        remaining_pos = self.runner.risk_mgr.positions.get(sym)
        self.assertIsNotNone(remaining_pos)
        self.assertAlmostEqual(remaining_pos.quantity, 0.006, places=5)
        # 2,006,000 * (1 - 0.4) = 1,203,600 TMN
        self.assertAlmostEqual(remaining_pos.cost_tmn, 1_203_600.0, places=0)
        # Pending sell lock cleared
        self.assertNotIn(sym, self.runner.risk_mgr.pending_sell_orders)

    def test_sweep_stale_orders_handles_zero_fill_cancellation(self):
        """When sell order is cancelled with 0 fills, position remains intact and pending lock is cleared."""
        sym = "ETHTMN"
        pos = OpenPosition(
            symbol=sym,
            entry_price=200_000_000.0,
            quantity=0.01,
            stop_loss_price=195_000_000.0,
            take_profit_price=210_000_000.0,
            highest_price=200_000_000.0,
            opened_at="2026-10-23T12:00:00Z",
            client_id="wlx_buy_eth",
            cost_tmn=2_006_000.0,
        )
        self.runner.risk_mgr.positions[sym] = pos
        cid = "wlx_sell_eth_zero"
        self.runner.risk_mgr.pending_sell_orders[sym] = {
            "client_id": cid,
            "symbol": sym,
            "sell_qty": 0.01,
            "exec_price": 205_000_000.0,
            "cost_part": 2_006_000.0,
            "cost_full": 2_006_000.0,
            "frac": 1.0,
            "exit_reason": "TAKE_PROFIT",
            "decision": AIDecisionOutput(symbol=sym, action="SELL", order_type="MARKET", confidence=1.0, allocation_pct=0.0, reasoning_fa="تست"),
            "feats": {"best_bid": 205_000_000.0},
            "total_eq": 5_000_000.0,
        }
        self.runner.pending_order_timestamps[cid] = 1000.0

        self.runner.client.get_open_orders = MagicMock(return_value=[])
        self.runner.client.get_free_asset_balance = MagicMock(return_value=0.01)  # All 0.01 in wallet
        self.runner.client.market_rules[sym] = {"min_qty": 0.0001, "base_asset": "ETH"}
        self.runner.client.get_order_by_client_id = MagicMock(return_value={
            "status": "CANCELED",
            "executedQty": "0",
            "executedSum": "0",
        })

        self.runner.sweep_stale_open_orders()

        # 0 trades realized
        self.assertEqual(self.runner.risk_mgr.total_trades_count, 0)
        # Position retained with full original quantity and cost
        remaining_pos = self.runner.risk_mgr.positions.get(sym)
        self.assertIsNotNone(remaining_pos)
        self.assertEqual(remaining_pos.quantity, 0.01)
        self.assertEqual(remaining_pos.cost_tmn, 2_006_000.0)
        # Pending sell lock cleared so SL check can resume
        self.assertNotIn(sym, self.runner.risk_mgr.pending_sell_orders)


class TestTechnique3StopLossSpreadGuard(BaseIsolatedTestCase):
    def setUp(self):
        super().setUp()
        self.config = AgentConfig(
            stop_loss_max_spread_pct=0.80,
            state_file=self.state_file,
            decision_log_file=self.log_file,
        )
        self.rm = PortfolioAndRiskManager(self.config)

    def test_spread_guard_blocks_stop_loss_in_liquidity_hole(self):
        """When best_bid <= SL, but spread > 0.8% and mid_price > SL, do NOT exit (even when last_price is omitted!)."""
        sym = "SOLTMN"
        pos = OpenPosition(
            symbol=sym,
            entry_price=10_000_000.0,
            quantity=0.1,
            stop_loss_price=9_880_000.0,  # SL at -1.2%
            take_profit_price=10_300_000.0,
            highest_price=10_000_000.0,
            opened_at="2026-10-23T12:00:00Z",
            client_id="wlx_sol",
        )
        self.rm.positions[sym] = pos

        # Liquidity hole: best_bid is 9,870,000, best_ask is 10,010,000. Mid price is 9,940,000 (> SL!).
        # Spread is 1.41% (> 0.80%). Note: last_price is omitted intentionally to verify robustness!
        features = {
            "best_bid": 9_870_000.0,
            "best_ask": 10_010_000.0,
            "mid_price": 9_940_000.0,
            "spread_pct": 1.41,
        }

        exit_decision = self.rm.check_deterministic_exits(sym, features)
        self.assertIsNone(exit_decision, "Spread guard should have blocked dumping into liquidity hole even with omitted last_price")

    def test_spread_guard_permits_exit_when_mid_price_actually_breaches(self):
        """When mid_price actually drops below SL, exit MUST fire."""
        sym = "SOLTMN"
        pos = OpenPosition(
            symbol=sym,
            entry_price=10_000_000.0,
            quantity=0.1,
            stop_loss_price=9_880_000.0,
            take_profit_price=10_300_000.0,
            highest_price=10_000_000.0,
            opened_at="2026-10-23T12:00:00Z",
            client_id="wlx_sol",
        )
        self.rm.positions[sym] = pos

        # Real price breakdown: best_bid is 9,860,000, best_ask is 9,890,000.
        # Mid price is 9,875,000 (<= SL 9,880,000). Spread = 0.30% (< 0.80%).
        features = {
            "best_bid": 9_860_000.0,
            "best_ask": 9_890_000.0,
            "mid_price": 9_875_000.0,
            "spread_pct": 0.30,
        }

        exit_decision = self.rm.check_deterministic_exits(sym, features)
        self.assertIsNotNone(exit_decision)
        self.assertEqual(exit_decision.action, "SELL")
        self.assertIn("HARD_STOP_LOSS", exit_decision.signals_observed)


class TestTechnique4DynamicKellySizing(BaseIsolatedTestCase):
    def setUp(self):
        super().setUp()
        self.config = AgentConfig(
            max_position_pct=0.40,
            high_conviction_position_pct=0.35,
            moderate_conviction_position_pct=0.18,
            min_cash_reserve_pct=0.20,
            exchange_min_notional_tmn=100_000.0,
            state_file=self.state_file,
            decision_log_file=self.log_file,
        )
        self.rm = PortfolioAndRiskManager(self.config)

    def test_high_conviction_sizing(self):
        """Confidence >= 0.80 with trend_bullish and positive OBI sizes 35% - 40%."""
        decision = AIDecisionOutput(
            symbol="BTCTMN",
            action="BUY",
            order_type="MARKET",
            confidence=0.90,
            allocation_pct=0.0,
            reasoning_fa="تست سایزینگ بالا",
        )
        features = {"trend_bullish": True, "obi_top5": 0.22, "spread_pct": 0.15}
        avail_tmn = 5_000_000.0
        total_eq = 5_000_000.0

        order_tmn = self.rm.calculate_position_size_tmn(decision, features, avail_tmn, total_eq)
        self.assertGreaterEqual(order_tmn, total_eq * 0.35)
        self.assertLessEqual(order_tmn, total_eq * 0.40)
        self.assertAlmostEqual(order_tmn, 1_875_000.0, places=0)

    def test_moderate_conviction_sizing(self):
        """Confidence in [0.65, 0.80) sizes 15% - 20%."""
        decision = AIDecisionOutput(
            symbol="BTCTMN",
            action="BUY",
            order_type="MARKET",
            confidence=0.70,
            allocation_pct=0.0,
            reasoning_fa="تست سایزینگ متوسط",
        )
        features = {"trend_bullish": True, "obi_top5": 0.10, "spread_pct": 0.15}
        avail_tmn = 5_000_000.0
        total_eq = 5_000_000.0

        order_tmn = self.rm.calculate_position_size_tmn(decision, features, avail_tmn, total_eq)
        self.assertGreaterEqual(order_tmn, total_eq * 0.15)
        self.assertLessEqual(order_tmn, total_eq * 0.20)

    def test_min_cash_reserve_strictly_maintained(self):
        """Order size must never eat into 20% minimum cash reserve."""
        decision = AIDecisionOutput(
            symbol="BTCTMN",
            action="BUY",
            order_type="MARKET",
            confidence=0.95,
            allocation_pct=0.40,
            reasoning_fa="تست رزرو پول",
        )
        features = {"trend_bullish": True, "obi_top5": 0.20, "spread_pct": 0.15}
        avail_tmn = 1_500_000.0
        total_eq = 5_000_000.0

        order_tmn = self.rm.calculate_position_size_tmn(decision, features, avail_tmn, total_eq)
        self.assertEqual(order_tmn, 500_000.0)
        remaining_cash = avail_tmn - order_tmn
        self.assertGreaterEqual(remaining_cash, total_eq * self.config.min_cash_reserve_pct)


class TestTechnique5SniperModeEntryHardVetoes(BaseIsolatedTestCase):
    def setUp(self):
        super().setUp()
        self.config = AgentConfig(
            min_atr_pct=0.35,
            state_file=self.state_file,
            decision_log_file=self.log_file,
        )
        self.strategist = CostEffectiveLLMStrategist(self.config)
        self.rm = PortfolioAndRiskManager(self.config)

    def test_trend_alignment_veto_rejects_counter_trend(self):
        """Reject BUY if trend_bullish is False."""
        features = {
            "valid": True,
            "last_price": 5_000_000.0,
            "ema50": 4_900_000.0,
            "trend_bullish": False,
            "bb_position": 0.50,
            "obi_top5": 0.20,
            "atr_pct": 0.50,
            "atr_14": 25_000.0,
            "best_bid": 4_995_000.0,
            "best_ask": 5_005_000.0,
        }
        veto = self.strategist.check_sniper_vetoes("TESTTMN", features)
        self.assertIsNotNone(veto)
        self.assertIn("COUNTER_TREND", veto)

    def test_trend_alignment_veto_rejects_price_below_ema50(self):
        """Reject BUY if price < ema50."""
        features = {
            "valid": True,
            "last_price": 4_800_000.0,
            "ema50": 5_000_000.0,
            "trend_bullish": True,
            "bb_position": 0.50,
            "obi_top5": 0.20,
            "atr_pct": 0.50,
            "atr_14": 25_000.0,
            "best_bid": 4_795_000.0,
            "best_ask": 4_805_000.0,
        }
        veto = self.strategist.check_sniper_vetoes("TESTTMN", features)
        self.assertIsNotNone(veto)
        self.assertIn("BELOW_EMA50", veto)

    def test_overbought_veto_rejects_bb_position_above_075(self):
        """Reject BUY if bb_position > 0.75."""
        features = {
            "valid": True,
            "last_price": 5_200_000.0,
            "ema50": 5_000_000.0,
            "trend_bullish": True,
            "bb_position": 0.82,
            "obi_top5": 0.20,
            "atr_pct": 0.50,
            "atr_14": 25_000.0,
            "best_bid": 5_195_000.0,
            "best_ask": 5_205_000.0,
        }
        veto = self.strategist.check_sniper_vetoes("TESTTMN", features)
        self.assertIsNotNone(veto)
        self.assertIn("OVERBOUGHT_BB", veto)

    def test_microstructure_depth_veto_rejects_negative_obi(self):
        """Reject BUY if obi_top5 < 0."""
        features = {
            "valid": True,
            "last_price": 5_100_000.0,
            "ema50": 5_000_000.0,
            "trend_bullish": True,
            "bb_position": 0.50,
            "obi_top5": -0.15,
            "atr_pct": 0.50,
            "atr_14": 25_000.0,
            "best_bid": 5_095_000.0,
            "best_ask": 5_105_000.0,
        }
        veto = self.strategist.check_sniper_vetoes("TESTTMN", features)
        self.assertIsNotNone(veto)
        self.assertIn("NEGATIVE_OBI", veto)

    def test_minimum_volatility_filter_rejects_low_atr(self):
        """Reject BUY if atr_pct < 0.35%."""
        features = {
            "valid": True,
            "last_price": 5_100_000.0,
            "ema50": 5_000_000.0,
            "trend_bullish": True,
            "bb_position": 0.50,
            "obi_top5": 0.20,
            "atr_pct": 0.22,
            "atr_14": 11_220.0,
            "best_bid": 5_095_000.0,
            "best_ask": 5_105_000.0,
        }
        veto = self.strategist.check_sniper_vetoes("TESTTMN", features)
        self.assertIsNotNone(veto)
        self.assertIn("LOW_VOLATILITY", veto)

    def test_evaluate_market_converts_buy_to_safe_hold_on_veto(self):
        """Verify that evaluate_market safely converts BUY to HOLD when veto fires."""
        features = {
            "valid": True,
            "last_price": 5_100_000.0,
            "ema50": 5_000_000.0,
            "trend_bullish": True,
            "macro_uptrend": True,
            "bb_position": 0.85,
            "obi_top5": 0.20,
            "atr_pct": 0.50,
            "atr_14": 25_000.0,
            "best_bid": 5_095_000.0,
            "best_ask": 5_105_000.0,
            "rsi_14": 55.0,
            "taker_buy_ratio": 0.55,
            "spread_pct": 0.20,
        }
        portfolio_state = {"available_tmn": 5_000_000, "total_equity_tmn": 5_000_000, "position_qty": 0}

        self.strategist.client = None
        decision, _ = self.strategist.evaluate_market("TESTTMN", features, portfolio_state)

        self.assertEqual(decision.action, "HOLD")
        self.assertTrue(any("OVERBOUGHT_BB" in s for s in decision.signals_observed))

    def test_approve_and_size_buy_rejects_sniper_vetoed_buy(self):
        """Verify risk manager rejects BUY when sniper veto conditions are breached (defense-in-depth)."""
        decision = AIDecisionOutput(
            symbol="TESTTMN",
            action="BUY",
            order_type="MARKET",
            confidence=0.88,
            allocation_pct=0.35,
            reasoning_fa="تست بای و وتو",
        )
        features = {
            "valid": True,
            "last_price": 5_100_000.0,
            "ema50": 5_000_000.0,
            "trend_bullish": True,
            "bb_position": 0.82,  # Overbought veto!
            "obi_top5": 0.20,
            "atr_pct": 0.50,
            "spread_pct": 0.20,
        }
        approved, reason, size = self.rm.approve_and_size_buy(decision, features, 5_000_000.0, 5_000_000.0)
        self.assertFalse(approved)
        self.assertIn("REJECTED_BY_SNIPER_VETO", reason)


if __name__ == "__main__":
    unittest.main()
