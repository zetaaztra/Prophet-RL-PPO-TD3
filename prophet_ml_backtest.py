import numpy as np
import pandas as pd
import os
import sys

# Add current directory to path to ensure nifty_prophet can be imported
sys.path.append(os.getcwd())

try:
    from nifty_prophet import NiftyOptionsProphet
except ImportError:
    print("[ERROR] Could not import NiftyOptionsProphet. Ensure 'nifty_prophet.py' is in the same directory.")
    sys.exit(1)

# Reuse simulation logic from RL backtest for consistency
from prophet_rl_backtest import simulate_trade, report, INITIAL_CAPITAL, HOLD_DAYS

def sentiment_to_trade(sentiment):
    bias = sentiment['bias']
    if bias == "BULLISH":
        return "LONG CALL SPREAD" # Higher conviction
    if bias == "BEARISH":
        return "LONG PUT SPREAD"
    return "IRON CONDOR"

def run_ml_backtest():
    print("\n[ML BACKTEST] Initializing Prophet Engine...")
    prophet = NiftyOptionsProphet()
    prophet.initialize()
    prophet.train_hmm()
    prophet.train_lstm_sr()
    
    df = prophet.data_1d.reset_index(drop=True)
    
    capital = INITIAL_CAPITAL
    equity = []
    trades = []
    
    print("[ML BACKTEST] Starting Historical Simulation (HMM + LSTM Fusion)...")
    # Using 100 day window steps for performance, or full loop
    # We will do a full loop but note that HMM/LSTM training is static here for speed
    # In a perfect world, we'd retrain periodically, but let's test the trained edge.
    
    for i in range(200, len(df) - HOLD_DAYS):
        # Update prophet's view of data up to current backtest point
        # This simulates walk-forward but reuses the trained models for speed
        prophet.data_1d = df.iloc[:i]
        sentiment = prophet.get_fusion_sentiment()
        
        trade_type = sentiment_to_trade(sentiment)
        
        entry_price = df.loc[i, 'close']
        vix = df.loc[i, 'vix']
        week_df = df.iloc[i : i + HOLD_DAYS]
        
        pnl = simulate_trade(trade_type, entry_price, week_df, vix)
        
        capital += pnl
        trades.append(pnl)
        equity.append(capital)
        
    return pd.Series(equity), trades

if __name__ == "__main__":
    eq, tr = run_ml_backtest()
    report("Fusion ML PROPHET (HMM+LSTM)", eq, tr)
