"""
NIFTY PROPHET v3 - MASTER VERDICT BACKTEST
============================================
Tests the combined HMM + LSTM + GAP Momentum + Tech fusion system.
This backtest evaluates the new MASTER VERDICT decision engine.

Models Tested:
- HMM Market Regime Detection (Bearish/Neutral/Bullish)
- LSTM Support/Resistance Forecasting
- Technical Indicators (RSI + MACD)
- Gap Momentum (Overnight gaps as fresh signals)
- PPO RL Agent (Optional comparison)
"""

import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from nifty_prophet import NiftyOptionsProphet
except ImportError:
    print("[ERROR] Could not import NiftyOptionsProphet. Ensure 'nifty_prophet.py' is in the same directory.")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
INITIAL_CAPITAL = 100_000
LOT_SIZE = 75
PROFIT_TARGET = 0.40          # Close at 40% of credit
STOPLOSS_MULTIPLIER = 1.8     # Stoploss = 1.8x credit
HOLD_DAYS = 5                 # Weekly expiry
COST_PER_TRADE = 40           # Brokerage + slippage
TEST_YEAR = 2024              # Backtest focus year
SIMULATE_GAPS = True          # Simulate overnight gaps for stress testing

# ═══════════════════════════════════════════════════════════════════════════════
# PREMIUM MODEL (Realistic Options Pricing)
# ═══════════════════════════════════════════════════════════════════════════════
def estimate_credit(vix, dte):
    """Estimate premium received for selling a spread"""
    base = 35
    iv_adj = vix * 2.2
    time_adj = dte * 2
    return int(base + iv_adj + time_adj)

def estimate_debit(vix, dte):
    """Estimate premium paid for buying a spread"""
    return int(estimate_credit(vix, dte) * 0.6)

# ═══════════════════════════════════════════════════════════════════════════════
# TRADE MAPPER (Master Verdict → Strategy)
# ═══════════════════════════════════════════════════════════════════════════════
def verdict_to_trade(verdict, vix):
    """
    Map Master Verdict to specific trade strategy with VIX-based filtering
    """
    bias = verdict.get('bias', 'NEUTRAL')
    confidence = verdict.get('confidence', 0)
    
    # [FILTER] High Volatility Risk Management
    if vix > 22:
        if bias == "BULLISH" and confidence > 50:
            return "BULL PUT SPREAD" # Defensive switch from Long Call
        if bias == "BEARISH" and confidence > 50:
            return "BEAR CALL SPREAD" # Defensive switch from Long Put
        return "NO TRADE" # Skip Iron Condors in high vol
        
    # [FILTER] Low Volatility Risk Management
    if vix < 12:
        if bias == "NEUTRAL":
            return "NO TRADE" # Skip Iron Condors in ultra-low vol (no premium)

    # Standard Logic
    if bias == "BULLISH" and confidence > 50:
        return "LONG CALL SPREAD"
    if bias == "BEARISH" and confidence > 50:
        return "LONG PUT SPREAD"
    
    if bias == "BULLISH" and confidence > 30:
        return "BULL PUT SPREAD"
    if bias == "BEARISH" and confidence > 30:
        return "BEAR CALL SPREAD"
    
    return "IRON CONDOR"

# ═══════════════════════════════════════════════════════════════════════════════
# TRADE SIMULATOR (Realistic PnL Calculation)
# ═══════════════════════════════════════════════════════════════════════════════
def simulate_trade(strategy, entry_price, week_df, vix):
    """
    Simulate trade outcome over the holding period
    
    Args:
        strategy: Trade type (LONG CALL SPREAD, IRON CONDOR, etc.)
        entry_price: Entry price at trade initiation
        week_df: DataFrame of price action during holding period
        vix: Implied volatility at entry
    Returns:
        float: PnL in rupees
    """
    if week_df.empty:
        return 0
    
    high = week_df['high'].max()
    low = week_df['low'].min()
    exit_price = week_df['close'].iloc[-1]
    
    credit = estimate_credit(vix, HOLD_DAYS)
    debit = estimate_debit(vix, HOLD_DAYS)
    
    stoploss_amt = credit * STOPLOSS_MULTIPLIER
    profit_take_amt = credit * PROFIT_TARGET
    
    move_up_pct = (high - entry_price) / entry_price
    move_down_pct = (entry_price - low) / entry_price
    net_move = (exit_price - entry_price) / entry_price
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DIRECTIONAL TRADES (Long Spreads)
    # ═══════════════════════════════════════════════════════════════════════════
    if strategy == "LONG CALL SPREAD":
        cost = debit * LOT_SIZE
        win = debit * 3 * LOT_SIZE  # 3x reward
        
        # Win if price moved up > 1.5%
        if move_up_pct > 0.015:
            return win - COST_PER_TRADE
        else:
            return -cost - COST_PER_TRADE
    
    if strategy == "LONG PUT SPREAD":
        cost = debit * LOT_SIZE
        win = debit * 3 * LOT_SIZE
        
        # Win if price moved down > 1.5%
        if move_down_pct > 0.015:
            return win - COST_PER_TRADE
        else:
            return -cost - COST_PER_TRADE
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CREDIT TRADES (Short Spreads)
    # ═══════════════════════════════════════════════════════════════════════════
    if strategy == "BULL PUT SPREAD":
        max_profit = credit * LOT_SIZE
        max_loss = stoploss_amt * LOT_SIZE
        
        # Win if price didn't drop > 2%
        if move_down_pct < 0.02:
            # Early profit taking simulation
            if np.random.rand() < 0.55:
                return profit_take_amt * LOT_SIZE - COST_PER_TRADE
            return max_profit - COST_PER_TRADE
        else:
            return -max_loss - COST_PER_TRADE
    
    if strategy == "BEAR CALL SPREAD":
        max_profit = credit * LOT_SIZE
        max_loss = stoploss_amt * LOT_SIZE
        
        # Win if price didn't rise > 2%
        if move_up_pct < 0.02:
            if np.random.rand() < 0.55:
                return profit_take_amt * LOT_SIZE - COST_PER_TRADE
            return max_profit - COST_PER_TRADE
        else:
            return -max_loss - COST_PER_TRADE
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NEUTRAL TRADES (Iron Condor)
    # ═══════════════════════════════════════════════════════════════════════════
    if strategy == "IRON CONDOR":
        max_profit = credit * LOT_SIZE
        max_loss = stoploss_amt * LOT_SIZE
        
        # DYNAMIC BOUNDARY based on volatility
        # Range scales with VIX: 2% at VIX 15, 3% at VIX 22
        dynamic_range = 0.015 * (vix / 15)
        
        if move_up_pct < dynamic_range and move_down_pct < dynamic_range:
            if np.random.rand() < 0.55:
                return profit_take_amt * LOT_SIZE - COST_PER_TRADE
            return max_profit - COST_PER_TRADE
        else:
            return -max_loss - COST_PER_TRADE
            
    if strategy == "NO TRADE":
        return 0
    
    return 0

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def run_master_verdict_backtest():
    """
    Run backtest using the Master Verdict (HMM + LSTM + GAP fusion)
    """
    print("\n" + "="*70)
    print("  NIFTY PROPHET v3 - MASTER VERDICT BACKTEST")
    print("="*70)
    
    # Initialize Prophet Engine
    print("\n[BACKTEST] Initializing Prophet Engine...")
    prophet = NiftyOptionsProphet()
    prophet.initialize()
    prophet.train_hmm()
    prophet.train_lstm_sr()
    
    df = prophet.data_1d.reset_index(drop=True)
    
    # Find test period start
    test_start_idx = df[df['date'].dt.year >= TEST_YEAR].index[0] if 'date' in df.columns else 200
    
    print(f"[BACKTEST] Test Period: {TEST_YEAR} onwards")
    print(f"[BACKTEST] Total bars: {len(df)}, Test start index: {test_start_idx}")
    
    # Track results
    capital = INITIAL_CAPITAL
    equity_curve = []
    all_trades = []
    trade_log = []
    
    # Model-specific tracking
    hmm_verdicts = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
    lstm_verdicts = {"OVEREXTENDED": 0, "UNDERVALUED": 0, "NEUTRAL": 0}
    strategy_counts = {}
    
    print(f"\n[BACKTEST] Starting simulation from index {test_start_idx}...")
    
    for i in range(test_start_idx, len(df) - HOLD_DAYS):
        # Update prophet's data view (walk-forward without lookahead)
        prophet.data_1d = df.iloc[:i+1]
        
        # Simulate overnight gap (optional stress test)
        gap_pts = 0
        if SIMULATE_GAPS and np.random.rand() < 0.3:  # 30% of days have gaps
            gap_pts = np.random.choice([-150, -100, -50, 50, 100, 150])
        
        # Get Master Verdict
        if gap_pts != 0:
            projected_spot = df.loc[i, 'close'] + gap_pts
            verdict = prophet.get_fusion_sentiment(target_spot=projected_spot)
        else:
            verdict = prophet.get_fusion_sentiment()
        
        # Track verdicts
        regime = verdict.get('regime', 'NEUTRAL')
        hmm_verdicts[regime] = hmm_verdicts.get(regime, 0) + 1
        
        # Map verdict to trade
        vix = df.loc[i, 'vix'] if 'vix' in df.columns else 15
        trade_type = verdict_to_trade(verdict, vix)
        strategy_counts[trade_type] = strategy_counts.get(trade_type, 0) + 1
        
        # Get trade parameters
        entry_price = df.loc[i, 'close'] + gap_pts if gap_pts else df.loc[i, 'close']
        vix = df.loc[i, 'vix'] if 'vix' in df.columns else 15
        week_df = df.iloc[i+1 : i+1+HOLD_DAYS]
        
        # Simulate trade
        pnl = simulate_trade(trade_type, entry_price, week_df, vix)
        
        # Update capital
        capital += pnl
        equity_curve.append(capital)
        all_trades.append(pnl)
        
        # Log trade
        trade_log.append({
            'date': df.loc[i, 'date'] if 'date' in df.columns else i,
            'entry': entry_price,
            'gap': gap_pts,
            'bias': verdict.get('bias'),
            'confidence': verdict.get('confidence'),
            'regime': regime,
            'strategy': trade_type,
            'pnl': pnl,
            'capital': capital
        })
        
        # Progress update
        if len(all_trades) % 50 == 0:
            print(f"  [SIM] Trade #{len(all_trades)} | Capital: {capital:,.0f} | "
                  f"Bias: {verdict.get('bias')} | Strategy: {trade_type}")
    
    return (
        pd.Series(equity_curve), 
        all_trades, 
        trade_log,
        {
            'hmm_verdicts': hmm_verdicts,
            'strategy_counts': strategy_counts
        }
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE REPORT
# ═══════════════════════════════════════════════════════════════════════════════
def generate_report(equity, trades, trade_log, stats):
    """Generate comprehensive performance report"""
    
    if len(trades) == 0:
        print("\n[!] No trades executed. Check data alignment.")
        return
    
    final_capital = equity.iloc[-1]
    total_return = (final_capital / INITIAL_CAPITAL - 1) * 100
    
    # Win rate
    winning_trades = [t for t in trades if t > 0]
    losing_trades = [t for t in trades if t < 0]
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
    
    # Profit factor
    gross_profit = sum(winning_trades) if winning_trades else 0
    gross_loss = abs(sum(losing_trades)) if losing_trades else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Drawdown
    peak = equity.cummax()
    drawdown = equity - peak
    max_dd = drawdown.min()
    max_dd_pct = (max_dd / peak.loc[drawdown.idxmin()]) * 100 if max_dd != 0 else 0
    
    # Average trade
    avg_trade = np.mean(trades)
    avg_winner = np.mean(winning_trades) if winning_trades else 0
    avg_loser = np.mean(losing_trades) if losing_trades else 0
    
    # Sharpe Ratio (simplified)
    returns = equity.pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    
    print("\n" + "="*70)
    print("         NIFTY PROPHET v3 - MASTER VERDICT BACKTEST RESULTS")
    print("="*70)
    
    print("\n📊 CAPITAL SUMMARY")
    print("-"*50)
    print(f"  Initial Capital:     ₹{INITIAL_CAPITAL:>15,}")
    print(f"  Final Capital:       ₹{final_capital:>15,.0f}")
    print(f"  Total Return:         {total_return:>14.2f}%")
    
    print("\n📈 TRADE STATISTICS")
    print("-"*50)
    print(f"  Total Trades:         {len(trades):>14}")
    print(f"  Winning Trades:       {len(winning_trades):>14}")
    print(f"  Losing Trades:        {len(losing_trades):>14}")
    print(f"  Win Rate:             {win_rate:>13.2f}%")
    
    print("\n💰 PROFITABILITY")
    print("-"*50)
    print(f"  Profit Factor:        {profit_factor:>14.2f}")
    print(f"  Average Trade:       ₹{avg_trade:>14,.0f}")
    print(f"  Avg Winner:          ₹{avg_winner:>14,.0f}")
    print(f"  Avg Loser:           ₹{avg_loser:>14,.0f}")
    print(f"  Sharpe Ratio:         {sharpe:>14.2f}")
    
    print("\n🛡️ RISK METRICS")
    print("-"*50)
    print(f"  Max Drawdown:        ₹{max_dd:>14,.0f}")
    print(f"  Max Drawdown %:       {max_dd_pct:>13.2f}%")
    
    print("\n🧠 MODEL VERDICTS DISTRIBUTION")
    print("-"*50)
    for regime, count in stats['hmm_verdicts'].items():
        pct = count / len(trades) * 100
        print(f"  {regime:<15}      {count:>6} ({pct:>5.1f}%)")
    
    print("\n📋 STRATEGY DISTRIBUTION")
    print("-"*50)
    for strategy, count in sorted(stats['strategy_counts'].items(), key=lambda x: -x[1]):
        pct = count / len(trades) * 100
        print(f"  {strategy:<20} {count:>6} ({pct:>5.1f}%)")
    
    print("\n" + "="*70)
    
    # Save trade log
    log_df = pd.DataFrame(trade_log)
    log_path = os.path.join(os.path.dirname(__file__), 'backtest_results_master_verdict.csv')
    log_df.to_csv(log_path, index=False)
    print(f"\n[EXPORT] Trade log saved to: {log_path}")
    
    return {
        'return': total_return,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_dd': max_dd,
        'sharpe': sharpe,
        'trades': len(trades)
    }

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    np.random.seed(42)  # Reproducibility
    
    equity, trades, trade_log, stats = run_master_verdict_backtest()
    generate_report(equity, trades, trade_log, stats)
