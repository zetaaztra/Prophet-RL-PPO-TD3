"""
PROPHET ML BACKTEST - MONTHLY P&L REPORT
Generates year-month breakdown of trading performance
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

# Constants
INITIAL_CAPITAL = 100000
LOT_SIZE = 75
PROFIT_TARGET = 0.40
STOPLOSS_MULTIPLIER = 1.8
HOLD_DAYS = 5
COST_PER_TRADE = 40

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
    
    if strategy in ["LONG CALL SPREAD", "LONG PUT SPREAD"]:
        cost = debit * LOT_SIZE
        win  = debit * 3 * LOT_SIZE
        
        if abs(high - entry_price) > entry_price * 0.015 or abs(entry_price - low) > entry_price * 0.015:
            return win - COST_PER_TRADE
        else:
            return -cost - COST_PER_TRADE
            
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

def run_monthly_backtest():
    print("\n[ML BACKTEST] Initializing Prophet Engine...")
    prophet = NiftyOptionsProphet()
    prophet.initialize()
    prophet.train_hmm()
    prophet.train_lstm_sr()
    
    df = prophet.data_1d.reset_index(drop=True)
    
    capital = INITIAL_CAPITAL
    
    # Store trades with dates
    trade_log = []
    
    print("[ML BACKTEST] Running Historical Simulation...")
    
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

def generate_monthly_report(trade_df):
    trade_df['year'] = trade_df['date'].dt.year
    trade_df['month'] = trade_df['date'].dt.month
    trade_df['year_month'] = trade_df['date'].dt.to_period('M')
    
    # Monthly aggregation
    monthly = trade_df.groupby('year_month').agg({
        'pnl': ['sum', 'count', lambda x: (x > 0).sum()],
        'capital': 'last'
    }).reset_index()
    
    monthly.columns = ['Year-Month', 'P&L', 'Trades', 'Wins', 'Capital']
    monthly['Winrate'] = (monthly['Wins'] / monthly['Trades'] * 100).round(1)
    
    # Yearly aggregation
    yearly = trade_df.groupby('year').agg({
        'pnl': ['sum', 'count', lambda x: (x > 0).sum()],
        'capital': 'last'
    }).reset_index()
    yearly.columns = ['Year', 'Total P&L', 'Trades', 'Wins', 'EOY Capital']
    yearly['Winrate'] = (yearly['Wins'] / yearly['Trades'] * 100).round(1)
    
    return monthly, yearly

if __name__ == "__main__":
    trade_df = run_monthly_backtest()
    monthly, yearly = generate_monthly_report(trade_df)
    
    print("\n" + "="*70)
    print("          PROPHET ML BACKTEST - MONTHLY P&L BREAKDOWN")
    print("="*70)
    
    # Print Monthly
    print("\n📅 MONTH-WISE P&L (Fusion ML Strategy)")
    print("-"*70)
    print(f"{'Year-Month':<12} {'P&L':>12} {'Trades':>8} {'Wins':>6} {'Winrate':>10} {'Capital':>14}")
    print("-"*70)
    
    for _, row in monthly.iterrows():
        pnl_str = f"₹{int(row['P&L']):,}"
        cap_str = f"₹{int(row['Capital']):,}"
        print(f"{str(row['Year-Month']):<12} {pnl_str:>12} {int(row['Trades']):>8} {int(row['Wins']):>6} {row['Winrate']:>9}% {cap_str:>14}")
    
    # Print Yearly Summary
    print("\n" + "="*70)
    print("          YEARLY SUMMARY")
    print("="*70)
    print(f"{'Year':<8} {'Total P&L':>14} {'Trades':>8} {'Wins':>6} {'Winrate':>10} {'EOY Capital':>16}")
    print("-"*70)
    
    for _, row in yearly.iterrows():
        pnl_str = f"₹{int(row['Total P&L']):,}"
        cap_str = f"₹{int(row['EOY Capital']):,}"
        print(f"{int(row['Year']):<8} {pnl_str:>14} {int(row['Trades']):>8} {int(row['Wins']):>6} {row['Winrate']:>9}% {cap_str:>16}")
    
    print("="*70)
    
    # Save to CSV
    monthly.to_csv("data/monthly_pnl_report.csv", index=False)
    yearly.to_csv("data/yearly_pnl_report.csv", index=False)
    print("\n[SAVED] Reports to data/monthly_pnl_report.csv and data/yearly_pnl_report.csv")
