"""
PROPHET ML BACKTEST - FIXED LOT (NO COMPOUNDING)
Realistic P&L with fixed 1 lot (75 qty) per trade
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.getcwd())

try:
    from nifty_prophet import NiftyOptionsProphet
except ImportError:
    print("[ERROR] Could not import NiftyOptionsProphet.")
    sys.exit(1)

# Constants - FIXED LOT SIZE (NO SCALING)
INITIAL_CAPITAL = 100000
LOT_SIZE = 75           # FIXED - Never changes
PROFIT_TARGET = 0.40
STOPLOSS_MULTIPLIER = 1.8
HOLD_DAYS = 5           # Weekly expiry
COST_PER_TRADE = 40     # Brokerage + slippage

def estimate_credit(vix, dte):
    """Credit received for SELLING spreads (Iron Condor, Bull Put, Bear Call)"""
    base = 35
    iv_adj = vix * 2.2
    time_adj = dte * 2
    return int(base + iv_adj + time_adj)

def estimate_debit(vix, dte):
    """Debit paid for BUYING spreads (Long Call, Long Put)"""
    return int(estimate_credit(vix, dte) * 0.6)

def simulate_trade(strategy, entry_price, week_df, vix):
    """
    Simulate a single trade with FIXED lot size.
    Returns P&L in rupees.
    """
    high = week_df['high'].max()
    low  = week_df['low'].min()
    
    credit = estimate_credit(vix, HOLD_DAYS)
    debit  = estimate_debit(vix, HOLD_DAYS)
    
    stoploss = credit * STOPLOSS_MULTIPLIER
    profit_take = credit * PROFIT_TARGET
    
    # ===== BUYING STRATEGIES (Directional Momentum) =====
    if strategy in ["LONG CALL SPREAD", "LONG PUT SPREAD"]:
        cost = debit * LOT_SIZE         # Max loss
        win  = debit * 3 * LOT_SIZE     # 3:1 reward
        
        # Check if target hit (1.5% move)
        if abs(high - entry_price) > entry_price * 0.015 or abs(entry_price - low) > entry_price * 0.015:
            return win - COST_PER_TRADE
        else:
            return -cost - COST_PER_TRADE
    
    # ===== SELLING STRATEGIES (Income Generation) =====
    if strategy == "IRON CONDOR":
        max_profit = credit * LOT_SIZE
        max_loss   = stoploss * LOT_SIZE
        
        # 55% chance of early profit taking
        if np.random.rand() < 0.55:
            return profit_take * LOT_SIZE - COST_PER_TRADE
        
        # Stop loss if 2% move
        if abs(high - entry_price) > entry_price * 0.02:
            return -max_loss - COST_PER_TRADE
        
        # Full profit at expiry
        return max_profit - COST_PER_TRADE
    
    return 0

def sentiment_to_trade(sentiment):
    """Map AI sentiment to trade type"""
    bias = sentiment['bias']
    if bias == "BULLISH":
        return "LONG CALL SPREAD"
    if bias == "BEARISH":
        return "LONG PUT SPREAD"
    return "IRON CONDOR"

def run_fixed_lot_backtest():
    print("\n[BACKTEST] Initializing Prophet Engine (Fixed Lot Mode)...")
    prophet = NiftyOptionsProphet()
    prophet.initialize()
    prophet.train_hmm()
    prophet.train_lstm_sr()
    
    df = prophet.data_1d.reset_index(drop=True)
    
    capital = INITIAL_CAPITAL
    trade_log = []
    
    print(f"[BACKTEST] Running with FIXED lot size = {LOT_SIZE} qty per trade")
    print("[BACKTEST] NO compounding - this is realistic P&L projection\n")
    
    for i in range(200, len(df) - HOLD_DAYS):
        prophet.data_1d = df.iloc[:i]
        sentiment = prophet.get_fusion_sentiment()
        trade_type = sentiment_to_trade(sentiment)
        
        entry_price = df.loc[i, 'close']
        vix = df.loc[i, 'vix']
        trade_date = df.loc[i, 'date']
        week_df = df.iloc[i : i + HOLD_DAYS]
        
        pnl = simulate_trade(trade_type, entry_price, week_df, vix)
        
        capital += pnl
        trade_log.append({
            'date': trade_date,
            'pnl': pnl,
            'capital': capital,
            'strategy': trade_type
        })
    
    return pd.DataFrame(trade_log)

def generate_report(trade_df):
    trade_df['year'] = trade_df['date'].dt.year
    trade_df['month'] = trade_df['date'].dt.month
    trade_df['year_month'] = trade_df['date'].dt.to_period('M')
    
    # Monthly
    monthly = trade_df.groupby('year_month').agg({
        'pnl': ['sum', 'count', lambda x: (x > 0).sum()],
        'capital': 'last'
    }).reset_index()
    monthly.columns = ['Year-Month', 'P&L', 'Trades', 'Wins', 'Capital']
    monthly['Winrate'] = (monthly['Wins'] / monthly['Trades'] * 100).round(1)
    
    # Yearly
    yearly = trade_df.groupby('year').agg({
        'pnl': ['sum', 'count', lambda x: (x > 0).sum()],
        'capital': 'last'
    }).reset_index()
    yearly.columns = ['Year', 'Total P&L', 'Trades', 'Wins', 'EOY Capital']
    yearly['Winrate'] = (yearly['Wins'] / yearly['Trades'] * 100).round(1)
    
    return monthly, yearly

if __name__ == "__main__":
    trade_df = run_fixed_lot_backtest()
    monthly, yearly = generate_report(trade_df)
    
    print("\n" + "="*70)
    print("     PROPHET ML BACKTEST - FIXED LOT (NO COMPOUNDING)")
    print("="*70)
    print(f"  Initial Capital: ₹{INITIAL_CAPITAL:,}")
    print(f"  Fixed Lot Size:  {LOT_SIZE} qty (1 lot)")
    print(f"  Holding Period:  {HOLD_DAYS} days (Weekly Expiry)")
    print("="*70)
    
    # Monthly
    print("\n📅 MONTH-WISE P&L")
    print("-"*70)
    print(f"{'Year-Month':<12} {'P&L':>12} {'Trades':>8} {'Wins':>6} {'Winrate':>10} {'Capital':>14}")
    print("-"*70)
    
    for _, row in monthly.iterrows():
        pnl_str = f"₹{int(row['P&L']):,}"
        cap_str = f"₹{int(row['Capital']):,}"
        print(f"{str(row['Year-Month']):<12} {pnl_str:>12} {int(row['Trades']):>8} {int(row['Wins']):>6} {row['Winrate']:>9}% {cap_str:>14}")
    
    # Yearly
    print("\n" + "="*70)
    print("          YEARLY SUMMARY (FIXED LOT)")
    print("="*70)
    print(f"{'Year':<8} {'Total P&L':>14} {'Trades':>8} {'Wins':>6} {'Winrate':>10} {'EOY Capital':>16}")
    print("-"*70)
    
    for _, row in yearly.iterrows():
        pnl_str = f"₹{int(row['Total P&L']):,}"
        cap_str = f"₹{int(row['EOY Capital']):,}"
        print(f"{int(row['Year']):<8} {pnl_str:>14} {int(row['Trades']):>8} {int(row['Wins']):>6} {row['Winrate']:>9}% {cap_str:>16}")
    
    # Final Summary
    total_pnl = trade_df['pnl'].sum()
    total_trades = len(trade_df)
    wins = (trade_df['pnl'] > 0).sum()
    
    print("="*70)
    print("\n📊 FINAL SUMMARY (REALISTIC - NO COMPOUNDING)")
    print("-"*50)
    print(f"  Starting Capital:  ₹{INITIAL_CAPITAL:,}")
    print(f"  Final Capital:     ₹{int(trade_df['capital'].iloc[-1]):,}")
    print(f"  Total P&L:         ₹{int(total_pnl):,}")
    print(f"  Total Return:      {(total_pnl/INITIAL_CAPITAL)*100:.1f}%")
    print(f"  Total Trades:      {total_trades}")
    print(f"  Overall Winrate:   {wins/total_trades*100:.1f}%")
    print(f"  Avg P&L/Month:     ₹{int(total_pnl / len(monthly)):,}")
    print("-"*50)
    
    # Save
    monthly.to_csv("data/monthly_fixed_lot.csv", index=False)
    yearly.to_csv("data/yearly_fixed_lot.csv", index=False)
    print("\n[SAVED] Reports to data/monthly_fixed_lot.csv and data/yearly_fixed_lot.csv")
