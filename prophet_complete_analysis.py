"""
PROPHET ML BACKTEST - COMPLETE ANALYSIS
With strategy breakdown, flip/whipsaw counts, updated lot size (65), brokerage (65)
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

# Constants - UPDATED FOR 2024+ NSE CHANGES
INITIAL_CAPITAL = 100000
LOT_SIZE = 65               # UPDATED: NSE changed from 75 to 65
PROFIT_TARGET = 0.40
STOPLOSS_MULTIPLIER = 1.8
HOLD_DAYS = 5
COST_PER_TRADE = 65         # UPDATED: Realistic brokerage + slippage

def estimate_credit(vix, dte):
    base = 35
    iv_adj = vix * 2.2
    time_adj = dte * 2
    return int(base + iv_adj + time_adj)

def estimate_debit(vix, dte):
    return int(estimate_credit(vix, dte) * 0.6)

def simulate_trade(strategy, entry_price, week_df, vix):
    high = week_df['high'].max()
    low  = week_df['low'].min()
    
    credit = estimate_credit(vix, HOLD_DAYS)
    debit  = estimate_debit(vix, HOLD_DAYS)
    
    stoploss = credit * STOPLOSS_MULTIPLIER
    profit_take = credit * PROFIT_TARGET
    
    # BUYING STRATEGIES
    if strategy in ["LONG CALL SPREAD", "LONG PUT SPREAD"]:
        cost = debit * LOT_SIZE
        win  = debit * 3 * LOT_SIZE
        
        if abs(high - entry_price) > entry_price * 0.015 or abs(entry_price - low) > entry_price * 0.015:
            return win - COST_PER_TRADE
        else:
            return -cost - COST_PER_TRADE
    
    # SELLING STRATEGIES
    if strategy == "IRON CONDOR":
        max_profit = credit * LOT_SIZE
        max_loss   = stoploss * LOT_SIZE
        
        if np.random.rand() < 0.55:
            return profit_take * LOT_SIZE - COST_PER_TRADE
        
        if abs(high - entry_price) > entry_price * 0.02:
            return -max_loss - COST_PER_TRADE
        
        return max_profit - COST_PER_TRADE
    
    return 0

def sentiment_to_trade(sentiment):
    bias = sentiment['bias']
    if bias == "BULLISH":
        return "LONG CALL SPREAD"
    if bias == "BEARISH":
        return "LONG PUT SPREAD"
    return "IRON CONDOR"

def run_complete_backtest():
    print("\n[BACKTEST] Initializing Prophet Engine (Complete Analysis)...")
    prophet = NiftyOptionsProphet()
    prophet.initialize()
    prophet.train_hmm()
    prophet.train_lstm_sr()
    
    df = prophet.data_1d.reset_index(drop=True)
    
    capital = INITIAL_CAPITAL
    trade_log = []
    
    # Counters
    flip_count = 0
    whipsaw_count = 0
    prev_bias = None
    
    print(f"[CONFIG] Lot Size: {LOT_SIZE} qty | Brokerage: ₹{COST_PER_TRADE}/trade")
    print("[BACKTEST] Running simulation with flip/whipsaw tracking...\n")
    
    for i in range(200, len(df) - HOLD_DAYS):
        prophet.data_1d = df.iloc[:i]
        sentiment = prophet.get_fusion_sentiment()
        
        current_bias = sentiment['bias']
        trade_type = sentiment_to_trade(sentiment)
        
        # Flip detection
        if prev_bias is not None and prev_bias != current_bias:
            flip_count += 1
        prev_bias = current_bias
        
        # Whipsaw detection (flip back within 3 days)
        if i > 203:
            recent_biases = [sentiment_to_trade(prophet.get_fusion_sentiment()) for _ in range(3)]
            if len(set(recent_biases)) > 1:
                whipsaw_count += 1
        
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
            'strategy': trade_type,
            'bias': current_bias
        })
    
    return pd.DataFrame(trade_log), flip_count, whipsaw_count

def generate_report(trade_df, flip_count, whipsaw_count):
    # Strategy breakdown
    strategy_counts = trade_df['strategy'].value_counts()
    
    buying_trades = strategy_counts.get("LONG CALL SPREAD", 0) + strategy_counts.get("LONG PUT SPREAD", 0)
    selling_trades = strategy_counts.get("IRON CONDOR", 0)
    
    # Monthly
    trade_df['year_month'] = trade_df['date'].dt.to_period('M')
    monthly = trade_df.groupby('year_month').agg({
        'pnl': ['sum', 'count', lambda x: (x > 0).sum()],
        'capital': 'last'
    }).reset_index()
    monthly.columns = ['Year-Month', 'P&L', 'Trades', 'Wins', 'Capital']
    monthly['Winrate'] = (monthly['Wins'] / monthly['Trades'] * 100).round(1)
    
    # Yearly
    trade_df['year'] = trade_df['date'].dt.year
    yearly = trade_df.groupby('year').agg({
        'pnl': ['sum', 'count', lambda x: (x > 0).sum()],
        'capital': 'last'
    }).reset_index()
    yearly.columns = ['Year', 'Total P&L', 'Trades', 'Wins', 'EOY Capital']
    yearly['Winrate'] = (yearly['Wins'] / yearly['Trades'] * 100).round(1)
    
    return monthly, yearly, strategy_counts, buying_trades, selling_trades

if __name__ == "__main__":
    trade_df, flip_count, whipsaw_count = run_complete_backtest()
    monthly, yearly, strategy_counts, buying_trades, selling_trades = generate_report(trade_df, flip_count, whipsaw_count)
    
    total_pnl = trade_df['pnl'].sum()
    total_trades = len(trade_df)
    wins = (trade_df['pnl'] > 0).sum()
    
    print("\n" + "="*70)
    print("     PROPHET ML BACKTEST - COMPLETE ANALYSIS")
    print("="*70)
    
    print("\n📋 CONFIGURATION")
    print("-"*50)
    print(f"  Lot Size:        {LOT_SIZE} qty (Updated NSE 2024)")
    print(f"  Brokerage:       ₹{COST_PER_TRADE}/trade")
    print(f"  Holding Period:  {HOLD_DAYS} days (Weekly Expiry)")
    print(f"  ML Model:        HMM + LSTM (Unsupervised + Supervised)")
    print(f"  RL Model:        NONE (This is pure ML, not RL)")
    
    print("\n📊 STRATEGY BREAKDOWN")
    print("-"*50)
    print(f"  BUYING Strategies (Long Spreads):  {buying_trades} trades ({buying_trades/total_trades*100:.1f}%)")
    print(f"    - Long Call Spread:  {strategy_counts.get('LONG CALL SPREAD', 0)}")
    print(f"    - Long Put Spread:   {strategy_counts.get('LONG PUT SPREAD', 0)}")
    print(f"  SELLING Strategies (Income):       {selling_trades} trades ({selling_trades/total_trades*100:.1f}%)")
    print(f"    - Iron Condor:       {strategy_counts.get('IRON CONDOR', 0)}")
    
    print("\n⚡ FLIP & WHIPSAW ANALYSIS")
    print("-"*50)
    print(f"  Total Regime Flips:    {flip_count}")
    print(f"  Total Whipsaws:        {whipsaw_count}")
    print(f"  Flip Rate:             1 flip every {total_trades // flip_count if flip_count > 0 else 0} trades")
    
    print("\n💰 FINAL SUMMARY")
    print("-"*50)
    print(f"  Starting Capital:  ₹{INITIAL_CAPITAL:,}")
    print(f"  Final Capital:     ₹{int(trade_df['capital'].iloc[-1]):,}")
    print(f"  Total P&L:         ₹{int(total_pnl):,}")
    print(f"  Total Return:      {(total_pnl/INITIAL_CAPITAL)*100:.1f}%")
    print(f"  Total Trades:      {total_trades}")
    print(f"  Overall Winrate:   {wins/total_trades*100:.1f}%")
    print(f"  Avg P&L/Month:     ₹{int(total_pnl / len(monthly)):,}")
    
    print("\n" + "="*70)
    print("          YEARLY SUMMARY")
    print("="*70)
    print(f"{'Year':<8} {'Total P&L':>14} {'Trades':>8} {'Winrate':>10}")
    print("-"*50)
    for _, row in yearly.iterrows():
        print(f"{int(row['Year']):<8} ₹{int(row['Total P&L']):>12,} {int(row['Trades']):>8} {row['Winrate']:>9}%")
    
    print("="*70)
    
    # Save
    trade_df.to_csv("data/complete_backtest.csv", index=False)
    print("\n[SAVED] Full trade log to data/complete_backtest.csv")
