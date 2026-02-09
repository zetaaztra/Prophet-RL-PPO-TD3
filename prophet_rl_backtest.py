import numpy as np
import pandas as pd
from stable_baselines3 import SAC
import os
import sys

# Add current directory to path to ensure nifty_prophet can be imported
sys.path.append(os.getcwd())

try:
    from nifty_prophet import NiftyOptionsProphet, OptionsProphetEnv
except ImportError:
    print("[ERROR] Could not import NiftyOptionsProphet. Ensure 'nifty_prophet.py' is in the same directory.")
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

def action_to_trade(action_value):
    if action_value > 0.3:
        return "LONG CALL SPREAD"
    if action_value < -0.3:
        return "LONG PUT SPREAD"
    if abs(action_value) < 0.2:
        return "IRON CONDOR"
    return "NO_TRADE"

def simulate_trade(strategy, entry_price, week_df, vix):
    high = week_df['high'].max()
    low  = week_df['low'].min()
    
    credit = estimate_credit(vix, HOLD_DAYS)
    debit  = estimate_debit(vix, HOLD_DAYS)
    
    stoploss = credit * STOPLOSS_MULTIPLIER
    profit_take = credit * PROFIT_TARGET
    
    # LONG VOL trades (RL main edge)
    if strategy in ["LONG CALL SPREAD", "LONG PUT SPREAD"]:
        cost = debit * LOT_SIZE
        win  = debit * 3 * LOT_SIZE
        
        # Simple threshold-based win condition for long vol
        if abs(high - entry_price) > entry_price * 0.015 or abs(entry_price - low) > entry_price * 0.015:
            return win - COST_PER_TRADE
        else:
            return -cost - COST_PER_TRADE
            
    # Neutral trades
    if strategy == "IRON CONDOR":
        max_profit = credit * LOT_SIZE
        max_loss   = stoploss * LOT_SIZE
        
        # Randomness factor for IC realism (55% win rate base)
        if np.random.rand() < 0.55:
            return profit_take * LOT_SIZE - COST_PER_TRADE
            
        if abs(high - entry_price) > entry_price * 0.02:
            return -max_loss - COST_PER_TRADE
            
        return max_profit - COST_PER_TRADE
        
    return 0

def run_rl_backtest():
    print("\n[RL BACKTEST] Initializing Prophet Engine...")
    prophet = NiftyOptionsProphet()
    prophet.initialize()
    prophet.train_hmm()
    prophet.train_lstm_sr()
    
    df = prophet.data_1d.reset_index(drop=True)
    
    # Setup RL environment and train model
    print("[RL BACKTEST] Training SAC Agent (5000 timesteps)...")
    env = OptionsProphetEnv(df, prophet.feature_cols, continuous=True)
    model = SAC("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=5000)
    
    capital = INITIAL_CAPITAL
    equity = []
    trades = []
    
    print("[RL BACKTEST] Starting Historical Simulation...")
    for i in range(200, len(df) - HOLD_DAYS):
        obs = df.loc[i, prophet.feature_cols].values.astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        
        trade_type = action_to_trade(action[0])
        
        if trade_type == "NO_TRADE":
            equity.append(capital)
            continue
            
        entry_price = df.loc[i, 'close']
        vix = df.loc[i, 'vix']
        week_df = df.iloc[i : i + HOLD_DAYS]
        
        pnl = simulate_trade(trade_type, entry_price, week_df, vix)
        
        capital += pnl
        trades.append(pnl)
        equity.append(capital)
        
    return pd.Series(equity), trades

def report(name, equity, trades):
    if len(trades) == 0:
        print(f"\n===== {name} RESULTS =====")
        print("No trades were taken.")
        return

    ret = (equity.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    winrate = np.mean(np.array(trades) > 0) * 100
    
    pos_trades = [t for t in trades if t > 0]
    neg_trades = [t for t in trades if t < 0]
    
    pf = sum(pos_trades) / abs(sum(neg_trades)) if len(neg_trades) > 0 else float('inf')
    dd = (equity - equity.cummax()).min()
    
    print(f"\n===== {name} RESULTS =====")
    print("Final capital:", round(equity.iloc[-1]))
    print("Return %:", round(ret, 2))
    print("Winrate:", round(winrate, 2), "%")
    print("Profit factor:", round(pf, 2))
    print("Max DD:", round(dd))
    print("Total Trades:", len(trades))

if __name__ == "__main__":
    eq, tr = run_rl_backtest()
    report("RL PROPHET (SAC)", eq, tr)
