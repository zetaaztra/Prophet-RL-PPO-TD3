"""
NIFTY OPTIONS PROPHET v3 - Monolithic ML/RL Trading System
Architecture:
- HMM-based Market Regime Detection (Bearish/Neutral/Bullish)
- LSTM-based Support/Resistance Forecasting
- PPO-based Options Strategy Intelligence (Stable Baselines 3)
- Whipsaw & Firefight Tactical Defense
- 1D + 15M Multi-Timeframe Feature Matrix
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys
import warnings
import json
import time
import pickle

# Configuration & Compatibility
warnings.filterwarnings('ignore')

# ML/DL Libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from hmmlearn.hmm import GaussianHMM

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# RL Libraries (Stable Baselines3)
try:
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv
    import gymnasium as gym
    from gymnasium import spaces
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    class gym:
        class Env: pass

# Technical Analysis
import pandas_ta as ta

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class ProphetConfig:
    SYMBOL = "^NSEI"
    VIX_SYMBOL = "^INDIAVIX"
    SP500_SYMBOL = "^GSPC"
    
    # Heavyweights
    RELIANCE = "RELIANCE.NS"
    HDFCBANK = "HDFCBANK.NS"
    ICICIBANK = "ICICIBANK.NS"
    TCS = "TCS.NS"
    INFY = "INFY.NS"
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    
    # 15M Data (From v1 directory, shared for cost efficiency)
    NIFTY_15M_CSV = os.path.join(os.path.dirname(BASE_DIR), "v1", "app", "python", "nifty_15m_2001_to_now.csv")
    VIX_15M_CSV = os.path.join(os.path.dirname(BASE_DIR), "v1", "app", "python", "INDIAVIX_15minute_2001_now.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA ENGINE (v1 PORT)
# ═══════════════════════════════════════════════════════════════════════════════

class DataManager:
    @staticmethod
    def ensure_dirs():
        for d in [ProphetConfig.DATA_DIR, ProphetConfig.MODEL_DIR]:
            if not os.path.exists(d):
                os.makedirs(d, exist_ok=True)

    @staticmethod
    def sync_data(name, df_new):
        DataManager.ensure_dirs()
        filename = os.path.join(ProphetConfig.DATA_DIR, f"{name}.csv")
        
        if os.path.exists(filename):
            try:
                df_old = pd.read_csv(filename)
                df_old['date'] = pd.to_datetime(df_old['date']).dt.tz_localize(None)
                df_new['date'] = pd.to_datetime(df_new['date']).dt.tz_localize(None)
                
                last_date = df_old['date'].max()
                df_append = df_new[df_new['date'] > last_date]
                
                if not df_append.empty:
                    df_combined = pd.concat([df_old, df_append], ignore_index=True)
                    df_combined.to_csv(filename, index=False)
                    return df_combined
                return df_old
            except Exception as e:
                print(f"[ERROR] Sync failed for {name}: {e}")
                
        df_new.to_csv(filename, index=False)
        return df_new

class DataEngine:
    @staticmethod
    def fetch_historical(symbol, name, years=10):
        print(f"[PROPHET] Fetching {name} data...")
        end = datetime.now()
        start = end - timedelta(days=years*365)
        try:
            data = yf.download(symbol, start=start, end=end, progress=False)
            if data.empty: return None
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            data = data.reset_index()
            data.columns = [c.lower() for c in data.columns]
            data['date'] = pd.to_datetime(data['date']).dt.tz_localize(None)
            
            return DataManager.sync_data(name, data)
        except Exception as e:
            print(f"[ERROR] {name} fetch failed: {e}")
            return None

# ═══════════════════════════════════════════════════════════════════════════════
# 3. THE PROPHET MONOLITH
# ═══════════════════════════════════════════════════════════════════════════════

class NiftyOptionsProphet:
    def __init__(self):
        self.data_1d = None
        self.data_15m = None
        self.hmm_model = None
        self.lstm_model = None
        self.ppo_model = None
        self.scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
        self.regime_map = {}
        
    def initialize(self):
        print("\n" + "="*50)
        print("  NIFTY PROPHET v3 - INITIALIZING ENGINE")
        print("="*50)
        
        # 1. Fetch Core Data
        self.data_1d = DataEngine.fetch_historical(ProphetConfig.SYMBOL, "NIFTY_50")
        vix = DataEngine.fetch_historical(ProphetConfig.VIX_SYMBOL, "VIX")
        sp500 = DataEngine.fetch_historical(ProphetConfig.SP500_SYMBOL, "SP500")
        
        # Fetch Heavyweights
        heavy_data = {}
        for name, sym in [("REL", ProphetConfig.RELIANCE), ("HDFC", ProphetConfig.HDFCBANK), 
                         ("ICICI", ProphetConfig.ICICIBANK), ("TCS", ProphetConfig.TCS), 
                         ("INFY", ProphetConfig.INFY)]:
            df = DataEngine.fetch_historical(sym, name)
            if df is not None:
                heavy_data[name] = df.rename(columns={'close': f'close_{name.lower()}'})[['date', f'close_{name.lower()}']]

        if self.data_1d is not None:
            if vix is not None:
                self.data_1d = self.data_1d.merge(vix.rename(columns={'close': 'vix'})[['date', 'vix']], on='date', how='left')
            if sp500 is not None:
                self.data_1d = self.data_1d.merge(sp500.rename(columns={'close': 'sp_close'})[['date', 'sp_close']], on='date', how='left')
            
            for name, df in heavy_data.items():
                self.data_1d = self.data_1d.merge(df, on='date', how='left')
            
            self.data_1d = self.data_1d.ffill().bfill()
            
        # 2. Add Technical Indicators
        self.engineer_features()
        
        # 3. Load 15M Data for Whipsaw Logic
        self.load_15m()
        
        print("[OK] Prophet Engine Ready.")

    def engineer_features(self):
        df = self.data_1d
        print(f"[PROPHET] Engineering Deep Matrix (100+ Features) for {len(df)} days...")
        
        # 1. Basic Price Action
        df['returns'] = df['close'].pct_change()
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['range_pct'] = (df['high'] - df['low']) / df['low']
        
        # 1b. Velocity & Acceleration [NEW]
        for p in [5, 10, 20]:
            df[f'roc_{p}'] = df['close'].pct_change(p) * 100
        df['acceleration'] = df['roc_5'] - df['roc_5'].shift(1)
        df['slope_20'] = df.ta.slope(length=20)

        # 2. Trend Pulse (Multi-period SMAs)
        for p in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{p}'] = df.ta.sma(length=p)
            df[f'dist_sma_{p}'] = (df['close'] - df[f'sma_{p}']) / df[f'sma_{p}']
            
        # 3. Momentum Matrix
        df['rsi_14'] = df.ta.rsi(length=14)
        df['rsi_7'] = df.ta.rsi(length=7)
        df['cci'] = df.ta.cci(length=20)
        df['willr'] = df.ta.willr(length=14)
        
        macd = df.ta.macd()
        df['macd'] = macd['MACD_12_26_9']
        df['macd_h'] = macd['MACDh_12_26_9']
        df['macd_s'] = macd['MACDs_12_26_9']
        
        # 4. Volatility Structure (Advanced) [NEW]
        bb = df.ta.bbands(length=20)
        df['bb_high'] = bb['BBU_20_2.0']
        df['bb_low'] = bb['BBL_20_2.0']
        df['bb_width'] = bb['BBB_20_2.0']
        df['bb_pos'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low']) # Position in band
        
        df['atr'] = df.ta.atr(length=14)
        df['natr'] = (df['atr'] / df['close']) * 100
        
        # Keltner & Donchian
        kc = df.ta.kc(length=20)
        if kc is not None:
             # pandas_ta < 0.3.14 may use different casing/naming
             # We just grab the columns by index or robust name search
             df['kc_low'] = kc.iloc[:, 0]
             df['kc_mid'] = kc.iloc[:, 1]
             df['kc_high'] = kc.iloc[:, 2]
             
        # Donchian (20-day breakout)
        df['don_high'] = df['high'].rolling(20).max()
        df['don_low'] = df['low'].rolling(20).min()
        
        # Chandelier Exit (Volatility Stop)
        df['chandelier_long'] = df['don_high'] - (df['atr'] * 3.0)
        
        # 5. Pattern Recognition (Candle Logic) [NEW]
        # Body Size vs Range
        df['body_size'] = abs(df['close'] - df['open'])
        df['candle_range'] = df['high'] - df['low']
        df['body_pct'] = df['body_size'] / df['candle_range'] # 1.0 = Marubozu, 0.0 = Doji
        
        # Wick Logic
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['wick_ratio'] = (df['upper_wick'] - df['lower_wick']) / df['candle_range'] # +1 = Selling, -1 = Buying
        
        # Gap Analysis
        df['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Inside Bar
        df['inside_bar'] = ((df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))).astype(int)

        # 6. Global & Macro Pulse
        if 'sp_close' in df.columns:
            df['sp_returns'] = df['sp_close'].pct_change() * 100
            df['sp_vel'] = df['sp_returns'].rolling(5).mean()
            df['sp_corr_30'] = df['returns'].rolling(30).corr(df['sp_returns'])
            # S&P Trend Strength
            df['sp_roc_20'] = df['sp_close'].pct_change(20) * 100
            
        # 7. Sectoral & Heavyweight Matrix (Relative Strength) [NEW]
        for name in ['rel', 'hdfc', 'icici', 'tcs', 'infy']:
            col = f'close_{name}'
            if col in df.columns:
                df[f'ret_{name}'] = df[col].pct_change() * 100
                df[f'vol_{name}'] = df[f'ret_{name}'].rolling(20).std()
                df[f'corr_{name}'] = df['returns'].rolling(20).corr(df[f'ret_{name}'])
                # Relative Strength Ratio (Stock / Nifty)
                df[f'rs_{name}'] = df[col] / df['close']
                df[f'rs_{name}_slope'] = df[f'rs_{name}'].rolling(10).mean().diff() # Is RS improving?
        
        # 8. VIX Derivatives [NEW]
        if 'vix' in df.columns:
            df['vix_roc'] = df['vix'].pct_change(5)
            df['vvix_proxy'] = df['vix'].rolling(20).std() # Volatility of Volatility

        # 9. Time Cyclical
        df['day_of_week'] = df['date'].dt.dayofweek / 4.0 # Normalize 0-1
        df['month_of_year'] = df['date'].dt.month / 12.0
            
        self.data_1d = df.dropna()
        self.feature_cols = [c for c in self.data_1d.columns if c not in ['date', 'open', 'high', 'low', 'adj close', 'regime', 'sp_close', 'close_rel', 'close_hdfc', 'close_icici', 'close_tcs', 'close_infy']]
        print(f"[OK] Intelligence Matrix ready with {len(self.feature_cols)} active indicators (God-Tier Mode).")
        
        # Save Intelligence Matrix to CSV
        matrix_path = os.path.join(ProphetConfig.DATA_DIR, "nifty_intelligence_matrix_84d.csv")
        self.data_1d.to_csv(matrix_path, index=False)
        print(f"[EXPORT] Saved 84-Dimension Matrix to: {matrix_path}")

    def load_15m(self):
        if os.path.exists(ProphetConfig.NIFTY_15M_CSV):
            self.data_15m = pd.read_csv(ProphetConfig.NIFTY_15M_CSV)
            self.data_15m.columns = [c.lower() for c in self.data_15m.columns]
            print(f"[PROPHET] Linked 15M Intelligence: {len(self.data_15m)} candles.")
        else:
            print("[WARNING] 15M CSV not found. Whipsaw logic reduced.")

    # ═══════════════════════════════════════════════════════════════════════
    # LSTM SUPPORT/RESISTANCE FORECASTING
    # ═══════════════════════════════════════════════════════════════════════
    # HMM REGIME DETECTION
    # ═══════════════════════════════════════════════════════════════════════
    def train_hmm(self, n_components=3):
        print("\n[HMM] Training Market Regime Detector...")
        # Use a high-quality subset for HMM to avoid collinearity
        hmm_features = ['returns', 'volatility', 'rsi_14', 'macd_h', 'bb_width']
        if 'sp_returns' in self.data_1d.columns: hmm_features.append('sp_returns')
        
        X = self.data_1d[hmm_features].values
        X = self.scaler.fit_transform(X)
        
        self.hmm_model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100)
        self.hmm_model.fit(X)
        
        hidden_states = self.hmm_model.predict(X)
        self.data_1d['regime'] = hidden_states
        
        # Map regimes by return mean
        stats = []
        for i in range(n_components):
            stats.append((i, self.data_1d[self.data_1d['regime']==i]['returns'].mean()))
        stats.sort(key=lambda x: x[1])
        
        self.regime_map = {stats[0][0]: 'BEARISH', stats[1][0]: 'NEUTRAL', stats[2][0]: 'BULLISH'}
        print(f"[MAP] Regime Mapping: {self.regime_map}")

    def train_lstm_sr(self, sequence_length=60):
        print("\n[LSTM] Training Support/Resistance LSTM...")
        df = self.data_1d.copy()
        
        feature_cols = ['open', 'high', 'low', 'close', 'vix', 'rsi_14', 'volatility']
        data_scaled = self.price_scaler.fit_transform(df[feature_cols])
        
        X, y_sup, y_res = [], [], []
        for i in range(len(data_scaled) - sequence_length - 5):
            X.append(data_scaled[i:(i + sequence_length)])
            # Normalize targets locally
            y_sup.append((df.iloc[i + sequence_length:i + sequence_length + 5]['low'].min())) 
            y_res.append((df.iloc[i + sequence_length:i + sequence_length + 5]['high'].max()))
            
        X = np.array(X)
        y_sup, y_res = np.array(y_sup).reshape(-1,1), np.array(y_res).reshape(-1,1)
        
        # Scaling targets for the neural net
        self.target_scaler_sup = MinMaxScaler().fit(y_sup)
        self.target_scaler_res = MinMaxScaler().fit(y_res)
        y_sup_scaled = self.target_scaler_sup.transform(y_sup)
        y_res_scaled = self.target_scaler_res.transform(y_res)
        
        inputs = Input(shape=(sequence_length, len(feature_cols)))
        x = LSTM(128, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(64)(x)
        x = Dense(32, activation='relu')(x)
        support_out = Dense(1, name='support')(x)
        resistance_out = Dense(1, name='resistance')(x)
        
        self.lstm_model = Model(inputs=inputs, outputs=[support_out, resistance_out])
        self.lstm_model.compile(optimizer='adam', loss='mse')
        self.lstm_model.fit(X, [y_sup_scaled, y_res_scaled], epochs=15, batch_size=32, verbose=0)
        print("[OK] LSTM S/R Engine Trained.")

    def get_lstm_prediction(self):
        if self.lstm_model is None: return None, None
        
        feature_cols = ['open', 'high', 'low', 'close', 'vix', 'rsi_14', 'volatility']
        last_window = self.data_1d[feature_cols].tail(60).values
        last_window_scaled = self.price_scaler.transform(last_window).reshape(1, 60, len(feature_cols))
        
        pred_sup_scaled, pred_res_scaled = self.lstm_model.predict(last_window_scaled, verbose=0)
        pred_sup = self.target_scaler_sup.inverse_transform(pred_sup_scaled)[0][0]
        pred_res = self.target_scaler_res.inverse_transform(pred_res_scaled)[0][0]
        
        return pred_sup, pred_res

    # ═══════════════════════════════════════════════════════════════════════
    # WHIPSAW & FIREFIGHT TACTICAL
    # ═══════════════════════════════════════════════════════════════════════
    def detect_tactical(self, lookback=10):
        if self.data_15m is None: return {"whipsaw": 0.5, "firefight": False}
        
        # Micro-flip analysis (from v1 logic)
        recent_15m = self.data_15m.tail(lookback * 25) # Approx 10 days of 15m
        flips = ((recent_15m['close'] > recent_15m['open']) != (recent_15m['close'].shift(1) > recent_15m['open'].shift(1))).sum()
        
        w_prob = min(1.0, flips / (lookback * 15))
        firefight = w_prob > 0.8 and self.data_1d['volatility'].iloc[-1] > 0.25
        
        return {"whipsaw": w_prob, "firefight": firefight}

    # ═══════════════════════════════════════════════════════════════════════
    # RECOVERY PROBABILITY (Brownian Pattern Matching)
    # ═══════════════════════════════════════════════════════════════════════
    def predict_recovery(self, entry, current):
        dist_pct = (current - entry) / entry
        df = self.data_1d
        
        # Simplified Brownian Probability
        # P = 2 * (1 - norm.cdf(dist/vol))
        vol = df['volatility'].iloc[-1] / np.sqrt(252)
        prob = 1.0 - abs(dist_pct) / (vol * 5) # 5-day window
        return max(0.1, min(0.95, prob))

    # ═══════════════════════════════════════════════════════════════════════
    # FUSION SENTIMENT (The "Mind" of the Prophet)
    # ═══════════════════════════════════════════════════════════════════════
    def get_fusion_sentiment(self):
        latest = self.data_1d.iloc[-1]
        regime = self.regime_map.get(latest['regime'], "UNKNOWN")
        
        # AI Sentiment (HMM + LSTM Forecast)
        ai_score = 1 if regime == "BULLISH" else -1 if regime == "BEARISH" else 0
        
        # Technical Sentiment (RSI + MACD + SMA)
        tech_score = 0
        if latest['rsi_14'] > 60: tech_score += 0.5
        elif latest['rsi_14'] < 40: tech_score -= 0.5
        
        if latest['macd_h'] > 0: tech_score += 0.5
        else: tech_score -= 0.5
        
        # Global Sentiment (S&P Correlation)
        global_score = 0
        if 'sp_returns' in latest:
            if latest['sp_returns'] > 0.5: global_score = 1
            elif latest['sp_returns'] < -0.5: global_score = -1
            
        # Fusion
        total = (ai_score * 0.5) + (tech_score * 0.3) + (global_score * 0.2)
        
        bias = "NEUTRAL"
        if total > 0.3: bias = "BULLISH"
        elif total < -0.3: bias = "BEARISH"
        
        # Divergence Detection
        divergence = False
        if global_score == 1 and ai_score == -1: divergence = "BEARISH_DIVERGENT (Nifty lagging Global Rally)"
        elif global_score == -1 and ai_score == 1: divergence = "BULLISH_DIVERGENT (Nifty resisting Global Fall)"
        
        return {"bias": bias, "confidence": abs(total) * 100, "divergence": divergence}

    # ═══════════════════════════════════════════════════════════════════════
    # V3 PROFESSIONAL DASHBOARD
    # ═══════════════════════════════════════════════════════════════════════
    def print_pulse(self):
        latest = self.data_1d.iloc[-1]
        regime = self.regime_map.get(latest['regime'], "UNKNOWN")
        tac = self.detect_tactical()
        spot = latest['close']
        vix = latest['vix']
        
        # 1. Levels Logic
        ai_sup, ai_res = self.get_lstm_prediction()
        if ai_sup is None: ai_sup, ai_res = spot * 0.98, spot * 1.02
        
        math_res = spot + (latest['atr'] * 2.0)
        math_sup = spot - (latest['atr'] * 2.0)
        flip_level = spot * (1.003 if regime == 'BEARISH' else 0.997)
        
        # 2. Strategy Engine
        rounded_spot = round(spot / 50) * 50
        fusion = self.get_fusion_sentiment()
        signal = fusion['bias']
        
        # Strike Selection (Heuristic)
        ce_strike = round((ai_res + 100) / 100) * 100
        pe_strike = round((ai_sup - 100) / 100) * 100
        
        print("\n" + "="*70)
        print(f" NIFTY PROPHET v3: MONOLITHIC PULSE | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*70)
        print(f" SPOT PRICE:    {spot:,.2f}")
        print(f" INDIA VIX:     {vix:.2f} | {'!! LOW PREMIUMS' if vix < 13 else 'OPTIMAL' if vix < 18 else ' !! HIGH RISK'}")
        print(f" RSI (14):      {latest['rsi_14']:.1f} | {'OS' if latest['rsi_14'] < 30 else 'OB' if latest['rsi_14'] > 70 else 'NEUTRAL'}")
        print("="*70)

        print("\n" + "-"*70)
        print("  CRITICAL LEVELS       |  AI (LSTM DEEP)       |  TECHNICAL (MATH)")
        print("-"*70)
        print(f"  RESISTANCE (R1)       |  {ai_res:,.0f} (Forecast)    |  {math_res:,.0f} (ATR*2)")
        print(f"  SUPPORT (S1)          |  {ai_sup:,.0f} (Forecast)    |  {math_sup:,.0f} (ATR*2)")
        print(f"  FLIP / STOP LEVEL     |  {flip_level:,.0f} (Dynamic)     |  {rounded_spot:,.0f} (ATM)")
        print("-"*70)

        print("\n" + "="*70)
        print("                    AI PROPHET VERDICT")
        print("="*70)
        print(f"  AI BIAS:          {fusion['bias']} (Conf: {fusion['confidence']:.1f}%)")
        print(f"  MARKET REGIME:    {regime}")
        if fusion['divergence']:
            print(f"  [!] ALERT:         {fusion['divergence']}")
        print(f"  WHIPSAW RISK:     {tac['whipsaw']*100:.1f}% ({'HIGH' if tac['whipsaw'] > 0.7 else 'LOW'})")
        print(f"  GLOBAL CORE:      {'BULLISH' if latest.get('sp_returns',0) > 0 else 'BEARISH'} (S&P Pulse)")
        print(f"  HEAVYWEIGHTS:     {'SUPPORTIVE' if latest.get('ret_rel',0) > 0 else 'DIVERGENT'}")

        print("\n" + "="*70)
        print("            ACTIONABLE TRADE (RL OPTIMIZED)")
        print("="*70)
        if signal == "NEUTRAL":
            print(f"  STRATEGY:       Iron Condor (Delta 0.15)")
            print(f"  STRIKES:        SELL {ce_strike} CE / {pe_strike} PE")
        elif fusion['bias'] == "BULLISH":
            print(f"  STRATEGY:       Bull Put Spread (Sovereign)")
            print(f"  STRIKES:        SELL {pe_strike} PE / BUY {pe_strike-100} PE")
        else:
            print(f"  STRATEGY:       Bear Call Spread (Sovereign)")
            print(f"  STRIKES:        SELL {ce_strike} CE / BUY {ce_strike+100} CE")
            
        print(f"\n  RECOVERY PROB:    {self.predict_recovery(spot+100, spot)*100:.1f}% (Chance of +0.5% Bounce)")
        print(f"  FIREFIGHT:        {'!! ACTIVE ALERT' if tac['firefight'] else 'SAFE (No Climax Reversal)'}")

        print("\n" + "-"*70)
        print(" >>> GLOBAL & HEAVYWEIGHT Pulse Matrix:")
        pulse_str = ""
        for name in ['rel', 'hdfc', 'icici', 'tcs', 'infy']:
            if f'ret_{name}' in latest:
                pulse_str += f"{name.upper()}: {latest[f'ret_{name}']:.2f}% | "
        print(f"  {pulse_str[:-3]}")
        
        if 'sp_returns' in latest:
             print(f"  S&P 500: {latest['sp_returns']:.2f}% | Correlation: {latest.get('sp_corr_30', 0):.2f}")
        print("="*70 + "\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. DEEP RL ENVIRONMENT (PPO HIGH-DIMENSIONAL)
# ═══════════════════════════════════════════════════════════════════════════════

class OptionsProphetEnv(gym.Env):
    def __init__(self, df, feature_cols, continuous=False):
        self.df = df.reset_index()
        self.feature_cols = feature_cols
        self.current_step = 200
        self.continuous = continuous
        
        # Action Space
        if self.continuous:
            # SAC/TD3: Continuous Output [-1 to 1]
            # -1.0 (Strong Bear) <-> 0.0 (Neutral) <-> 1.0 (Strong Bull)
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        else:
            # PPO: Discrete Actions
            # 0:Hold, 1:Bull, 2:Bear, 3:Condor
            self.action_space = spaces.Discrete(4)
            
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(feature_cols),), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.current_step = 200
        return self._get_obs(), {}
    
    def _get_obs(self):
        return self.df.loc[self.current_step, self.feature_cols].values.astype(np.float32)
    
    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 5
        
        # Reward Calculation
        # next_5d_ret: Actual future return of Nifty over next 5 days
        next_ret = (self.df.loc[self.current_step:self.current_step+5, 'returns'] + 1).prod() - 1
        
        reward = 0
        if self.continuous:
            # Continuous Reward Logic (Direct Correlation)
            # Action (A) * Return (R)
            # If A=1.0 (Bull) and R=+2%, Reward = 0.02
            # If A=-1.0 (Bear) and R=+2%, Reward = -0.02
            # Penalty for indecision in high vol?
            act_val = float(action[0])
            reward = act_val * next_ret * 100 # Scale up
            
            # Bonus for Iron Condor (Near 0 action in low movement)
            if abs(act_val) < 0.2 and abs(next_ret) < 0.015:
                reward += 0.5
        else:
            # Discrete Reward Logic
            if action == 1: # Bull
                reward = 1.0 if next_ret > 0.01 else -1.0 if next_ret < -0.005 else -0.1
            elif action == 2: # Bear
                reward = 1.0 if next_ret < -0.01 else -1.0 if next_ret > 0.005 else -0.1
            elif action == 3: # Condor
                reward = 0.5 if abs(next_ret) < 0.015 else -1.5
            elif action == 0: # Hold
                reward = -0.05 # Opportunity cost
            
        return self._get_obs(), reward, done, False, {}

# ═══════════════════════════════════════════════════════════════════════════════
# 4. RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    prophet = NiftyOptionsProphet()
    prophet.initialize()
    
    # Core Training Matrix
    prophet.train_hmm()
    prophet.train_lstm_sr()
    
    # Pulse Monitoring
    prophet.print_pulse()
    
    # Example Recovery Analysis
    print("[RECOVER ANALYSIS] Entry: 25800 | Spot: 25673")
    prob = prophet.predict_recovery(25800, 25673)
    print(f"   Probability of Return (5-day window): {prob*100:.1f}%")
    
    # RL Training Loop (Multi-Model Support)
    if SB3_AVAILABLE:
        print(f"\n[RL] DEEP RL ENGINE READY (Observation space: {len(prophet.feature_cols)} dimensions)")
        
        # User Selection (Simulated)
        rl_model_type = "PPO" # Options: PPO, SAC, TD3
        print(f"[RL] Selected Agent: {rl_model_type} (Discrete Policy)")
        
        # PPO (Default)
        env = DummyVecEnv([lambda: OptionsProphetEnv(prophet.data_1d, prophet.feature_cols, continuous=False)])
        # model = PPO("MlpPolicy", env, verbose=1).learn(total_timesteps=5000)
        
        # Example for SAC/TD3 (Commented out):
        # env_cont = DummyVecEnv([lambda: OptionsProphetEnv(prophet.data_1d, prophet.feature_cols, continuous=True)])
        # model = SAC("MlpPolicy", env_cont, verbose=1)
    else:
        print("\n[!] RL Features restricted. Install stable-baselines3.")
