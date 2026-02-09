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
from scipy.stats import norm

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
    TRAIN_YEARS = 10  # Default lookback
    
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
    def fetch_historical(symbol, name, years=None):
        if years is None:
            years = ProphetConfig.TRAIN_YEARS
        print(f"[PROPHET] Fetching {name} data ({years}y)...")
        end = datetime.now()
        start = end - timedelta(days=years*365)
        df = yf.download(symbol, start=start, end=end, progress=False)
        
        # Handle MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            
        return DataManager.sync_data(name, df)

    @staticmethod
    def fetch_live_price(symbol):
        """Fetch live price using 1m interval as per V1 logic"""
        try:
            live_data = yf.download(symbol, period="5d", interval="1m", progress=False)
            if not live_data.empty:
                if isinstance(live_data.columns, pd.MultiIndex):
                    close_prices = live_data['Close']
                    if hasattr(close_prices, 'iloc'):
                        return float(close_prices.iloc[-1].iloc[0] if isinstance(close_prices.iloc[-1], pd.Series) else close_prices.iloc[-1])
                return float(live_data['Close'].iloc[-1])
        except Exception as e:
            print(f"[WARNING] Live fetch failed for {symbol}: {e}")
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
        self.hmm_features = [] # [NEW] Track HMM specific features
        
    def initialize(self):
        print("\n" + "="*50)
        print(f"  NIFTY PROPHET v3 - INITIALIZING ENGINE (Lookback: {ProphetConfig.TRAIN_YEARS}y)")
        print("="*50)
        # 1. Load Primary Data
        self.data_1d = DataEngine.fetch_historical(ProphetConfig.SYMBOL, "NIFTY_50")
        vix = DataEngine.fetch_historical(ProphetConfig.VIX_SYMBOL, "VIX")
        
        # Merge VIX initial data
        vix_clean = vix[['date', 'close']].rename(columns={'close': 'vix'})
        self.data_1d = self.data_1d.merge(vix_clean, on='date', how='left').ffill()

        # 1b. Sync Live Data (V1 Pattern: Fresh price at startup)
        self.sync_live_data()

        # 2. Get Heavyweights & Global Pulse
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

    def sync_live_data(self):
        """
        Refetch the latest Spot and VIX prices from yfinance (V1 Style)
        Ensures the dashboard uses 'NOW' prices instead of 'PREVIOUS CLOSE'
        """
        live_spot = DataEngine.fetch_live_price(ProphetConfig.SYMBOL)
        live_vix = DataEngine.fetch_live_price(ProphetConfig.VIX_SYMBOL)
        
        if live_spot:
            # If today's row exists, update it. Otherwise append.
            last_date = self.data_1d['date'].iloc[-1].date()
            if last_date == datetime.now().date():
                self.data_1d.loc[self.data_1d.index[-1], 'close'] = live_spot
                print(f"[SYNC] Updated today's Spot: {live_spot:.2f}")
            else:
                new_row = self.data_1d.iloc[-1].copy()
                new_row['date'] = pd.Timestamp(datetime.now())
                new_row['close'] = live_spot
                # We need to fill columns that depend on close if we append
                self.data_1d = pd.concat([self.data_1d, pd.DataFrame([new_row])], ignore_index=True)
                print(f"[SYNC] Appended Live Spot: {live_spot:.2f}")
                
        if live_vix:
            self.data_1d.loc[self.data_1d.index[-1], 'vix'] = live_vix
            print(f"[SYNC] Updated latest VIX: {live_vix:.2f}")

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

        # 6. Gap Intel [NEW]
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['gap_up'] = (df['gap'] > 0.001).astype(int)
        df['gap_down'] = (df['gap'] < -0.001).astype(int)
        df['gap_up_rate'] = df['gap_up'].rolling(10).mean()
        df['gap_down_rate'] = df['gap_down'].rolling(10).mean()
        df['gap_std'] = df['gap'].rolling(20).std()
        
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
        """
        Train Gaussian HMM to detect market regimes (Bearish/Neutral/Bullish)
        Uses 'returns' and 'volatility' as features
        """
        # Define features for HMM
        self.hmm_features = ['returns', 'volatility']
        
        # Ensure we have clean data
        dataset = self.data_1d.copy().dropna()
        X = dataset[self.hmm_features].values
        
        # Train HMM with SCALED data (Important for feature consistency)
        X_scaled = self.scaler.fit_transform(X)
        
        self.hmm_model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100, random_state=42)
        self.hmm_model.fit(X_scaled)
        self.hmm_labels = self.hmm_model.predict(X_scaled)
        self.data_1d['regime'] = self.hmm_labels

        
        # Map hidden states to regimes based on average return
        means = []
        for i in range(n_components):
            means.append(dataset.iloc[self.hmm_labels == i]['returns'].mean())
            
        # Sort states by return: 0=Bearish (Lowest), 1=Neutral, 2=Bullish (Highest)
        sorted_indices = np.argsort(means)
        self.regime_map = {old: new for new, old in enumerate(sorted_indices)}
        
        print(f"[HMM] Training Market Regime Detector...")
        print(f"[MAP] Regime Mapping: {self.regime_map}") # 0: BEARISH, 1: NEUTRAL, 2: BULLISH

    def train_rl_agent(self):
        """
        Train PPO Reinforcement Learning Agent on the fly
        Uses the Deep Intelligent Matrix (84+ Dimensions)
        """
        if not SB3_AVAILABLE:
            print("[RL] Stable Baselines 3 not installed. Skipping PPO training.")
            return

        print(f"[RL] Training PPO Agent (Discrete Policy) on {len(self.feature_cols)} Features...")
        
        # Create Environment
        self.rl_env = DummyVecEnv([lambda: OptionsProphetEnv(self.data_1d, self.feature_cols, continuous=False)])
        
        # Train Agent (Fast Interactive Training)
        self.ppo_model = PPO("MlpPolicy", self.rl_env, verbose=0, learning_rate=0.0003, ent_coef=0.01)
        self.ppo_model.learn(total_timesteps=3000) # Quick training (can increase for prod)
        print("[RL] PPO Agent Trained.")

    def get_rl_verdict(self):
        """
        Get the latest action from the trained PPO agent
        """
        if not self.ppo_model:
            return "N/A", "Agent not active"
        
        # Get latest observation
        last_obs = self.data_1d.iloc[-1][self.feature_cols].values.astype(np.float32)
        action, _ = self.ppo_model.predict(last_obs, deterministic=True)
        
        act = int(action)
        if act == 1: return "BULLISH", "Strong Buy Signal (Long Delta)"
        if act == 2: return "BEARISH", "Strong Sell Signal (Short Delta)"
        if act == 3: return "NEUTRAL", "Price Stability Expected (Iron Condor)"
        return "HOLD", "No Clear Signal (Cash is King)"
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
        
        # Whipsaw Band (Volatility Channel)
        spot = self.data_1d['close'].iloc[-1]
        atr = self.data_1d['atr'].iloc[-1] if 'atr' in self.data_1d else (spot * 0.005)
        whip_low = spot - (atr * 1.5)
        whip_high = spot + (atr * 1.5)

        return {"whipsaw": w_prob, "firefight": firefight, "band": (whip_low, whip_high)}

    def rebuild_row_with_spot(self, target_spot):
        """
        Simulate a data row at a hypothetical price level for HMM scanning
        Enhanced: Approximates RSI and Volatility shifts
        """
        row = self.data_1d.iloc[-1].copy()
        old_close = self.data_1d['close'].iloc[-2]
        
        row['close'] = target_spot
        row['returns'] = (target_spot - old_close) / old_close
        
        # Approximate RSI Shift: A big move up increases average gains
        if row['returns'] > 0.005: # > 0.5% move
            row['rsi_14'] = min(85, row['rsi_14'] + (row['returns'] * 1000))
        elif row['returns'] < -0.005:
            row['rsi_14'] = max(15, row['rsi_14'] + (row['returns'] * 1000))
            
        # MACD Histogram shift (rough proxy)
        row['macd_h'] += row['returns'] * 50
        
        return row

    def find_flip_levels(self):
        """
        Scan price levels to find where the Market Regime changes
        Enhanced: Uses wider scan range and reports confidence
        """
        spot = getattr(self, 'projected_spot', self.data_1d['close'].iloc[-1])
        current_regime = self.data_1d['regime'].iloc[-1]
        
        # Scan +/- 5% (wider range for better detection)
        prices = np.linspace(spot * 0.95, spot * 1.05, 100)
        
        # [ROBUSTNESS CHECK] Ensure scaler is fitted
        from sklearn.utils.validation import check_is_fitted, NotFittedError
        try:
            check_is_fitted(self.scaler)
        except NotFittedError:
            print("[WARN] Scaler state lost. Recovering via re-fit...")
            # Re-fit using known HMM features
            scan_data = self.data_1d.copy().dropna()
            self.scaler.fit(scan_data[self.hmm_features].values)
        
        flip_up = None
        flip_down = None
        
        for p in prices:
            sim_row = self.rebuild_row_with_spot(p)
            # Use same feature engineering and SCALER as training
            feat_vals = sim_row[self.hmm_features].values.reshape(1, -1)
            feat_vals_scaled = self.scaler.transform(feat_vals)
            pred_regime = self.hmm_model.predict(feat_vals_scaled)[0]
            
            if p > spot and pred_regime != current_regime and flip_up is None:
                flip_up = p
            if p < spot and pred_regime != current_regime and flip_down is None:
                flip_down = p
        
        # If no flip found, use LSTM S/R as fallback
        if flip_up is None or flip_down is None:
            sr_low, sr_high = self.get_lstm_prediction()
            if flip_up is None and sr_high:
                flip_up = sr_high  # Resistance as bullish trigger
            if flip_down is None and sr_low:
                flip_down = sr_low  # Support as bearish trigger
                
        return flip_down, flip_up

    def get_firefight_tactics(self, sentiment, tactical, gap_info):
        """
        Contextual defense strategies based on regime and divergence
        """
        regime = self.regime_map.get(self.data_1d['regime'].iloc[-1], "UNKNOWN")
        bias = sentiment['bias']
        
        tactics = []
        
        if tactical['whipsaw'] > 0.7:
            tactics.append("CHOP ALERT: Use wider stops. Avoid aggressive buying.")
        
        if sentiment['divergence'] and "BEARISH_DIVERGENT" in sentiment['divergence']:
            tactics.append("HEDGE: Buy 1 lot OTM Put to protect against Global decoupling.")
            
        if gap_info['up_prob'] > 0.6 and bias == "BEARISH":
            tactics.append("TRAP: Possible Gap-Up Short Squeeze. Don't short at open.")

        if tactical['firefight']:
            tactics.append("CRITICAL: Market in High-Vol Reversal. Reduce position size by 50%.")
            
        if not tactics:
            tactics.append("STABLE: Follow standard delta hedging rules (Delta 0.15).")
            
        return tactics

    # ═══════════════════════════════════════════════════════════════════════
    # RECOVERY PROBABILITY (Brownian Pattern Matching)
    # ═══════════════════════════════════════════════════════════════════════
    def predict_recovery(self, entry, target):
        """
        Estimate probability of price reaching target within 5 trading days
        Uses historical volatility and distance-based Brownian motion
        """
        df = self.data_1d
        current = df['close'].iloc[-1]
        
        # Distance from current price to target
        dist_pct = abs(target - current) / current
        
        # Daily volatility (annualized / sqrt(252))
        vol_daily = df['volatility'].iloc[-1] / np.sqrt(252)
        
        # 5-day window: vol scales with sqrt(time)
        vol_5d = vol_daily * np.sqrt(5)
        
        # Probability using normal CDF
        # P(reaching target) = 2 * (1 - norm.cdf(distance / volatility))
        z_score = dist_pct / vol_5d if vol_5d > 0 else 10
        prob = 2 * (1 - norm.cdf(z_score))
        
        return max(0.05, min(0.95, prob))

    # ═══════════════════════════════════════════════════════════════════════
    # FUSION SENTIMENT (The "Mind" of the Prophet)
    # ═══════════════════════════════════════════════════════════════════════
    def get_fusion_sentiment(self, target_spot=None):
        latest = self.data_1d.iloc[-1]
        
        # Determine Regime (Actual or Projected)
        if target_spot:
            sim_row = self.rebuild_row_with_spot(target_spot)
            feat_vals = sim_row[self.hmm_features].values.reshape(1, -1)
            feat_vals_scaled = self.scaler.transform(feat_vals)
            regime_idx = self.hmm_model.predict(feat_vals_scaled)[0]
        else:
            regime_idx = latest['regime']
            
        regime = self.regime_map.get(regime_idx, "UNKNOWN")
        
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
        
        # GAP MOMENTUM SCORE (Treats overnight gap as fresh information)
        gap_score = 0
        if target_spot:
            gap_pts = target_spot - latest['close']
            if gap_pts > 100:  # Strong gap-up
                gap_score = 0.8 + (gap_pts / 500)
            elif gap_pts < -100:  # Strong gap-down
                gap_score = -0.8 - (abs(gap_pts) / 500)
            
        # [NEW] RL AGENT SENTIMENT (Proximal Policy Optimization)
        rl_verdict, _ = self.get_rl_verdict()
        rl_score = 0
        if rl_verdict == "BULLISH": rl_score = 1
        elif rl_verdict == "BEARISH": rl_score = -1
            
        # Fusion (Weighted Ensemble Strategy)
        # Weights: AI (25%), Tech (15%), Global (10%), Gap (25%), RL (PPO) (25%)
        total_score = (ai_score * 25) + (tech_score * 15) + (global_score * 10) + (gap_score * 25) + (rl_score * 25)
            
        # LSTM Alignment (Safety Check)
        # If price is far above LSTM resistance, cap the bullishness
        # BUT if gap momentum is strong, reduce this penalty (fresh info > historical forecast)
        sup, res = self.get_lstm_prediction()
        spot = target_spot if target_spot else latest['close']
        reasoning = []
        
        overvalued_penalty = 5  # Reduced from 0.2/20 scale
        if gap_score > 0.5 or rl_score > 0.5:
            overvalued_penalty = 1  # Strong gap or RL reduces overvaluation penalty
        
        if res and spot > res:
            total_score -= overvalued_penalty
            reasoning.append("OVERVALUED (Spot > AI Resistance)")
        elif sup and spot < sup:
            total_score += 5
            reasoning.append("UNDERVALUED (Spot < AI Support)")
            
        if regime == "BULLISH": reasoning.append("Regime: BULLISH")
        elif regime == "BEARISH": reasoning.append("Regime: BEARISH")
        
        if gap_score > 0.5: reasoning.append(f"Gap Momentum: STRONG UP (+{gap_pts:.0f} pts)")
        elif gap_score < -0.5: reasoning.append(f"Gap Momentum: STRONG DOWN ({gap_pts:.0f} pts)")
        
        if global_score > 0: reasoning.append("Globals: POSITIVE")
        elif global_score < 0: reasoning.append("Globals: WEAK")
        
        if rl_score > 0: reasoning.append("RL Agent: BULLISH")
        elif rl_score < 0: reasoning.append("RL Agent: BEARISH")

        bias = "NEUTRAL"
        if total_score > 30: bias = "BULLISH"
        elif total_score < -30: bias = "BEARISH"
        
        # Divergence Detection
        divergence = False
        if global_score == 1 and ai_score == -1: divergence = "BEARISH_DIVERGENT (Nifty lagging Global Rally)"
        elif global_score == -1 and ai_score == 1: divergence = "BULLISH_DIVERGENT (Nifty resisting Global Fall)"
        
        # 5. Total Score for Confidence
        # Already calculated as total_score in ensemble above
        
        notes = reasoning # Start with existing reasoning
        
        # 6. ENHANCED VIX FILTERING (Refined Strategy)
        vix = self.data_1d['vix'].iloc[-1]
        vix_bias = "NEUTRAL"
        if vix > 22:
            vix_bias = "BEARISH (HIGH VOL)"
            notes.append("VIX > 22: High Gamma Risk. Prefer Defensive Spreads.")
        elif vix < 12:
            notes.append("VIX < 12: Low Premium. Avoid Iron Condors.")
            
        # Strategy Recommendation Logic
        final_bias = "NEUTRAL"
        if total_score > 30: final_bias = "BULLISH"
        elif total_score < -30: final_bias = "BEARISH"
        
        if vix > 22 and final_bias == "BULLISH":
            notes.append("CAUTION: VIX High. Downshifting to Bull Put Spread instead of Long Call.")
        
        return {
            "bias": final_bias,
            "vix_bias": vix_bias,
            "confidence": min(100, abs(total_score)),
            "divergence": divergence,
            "regime": self.regime_map.get(self.data_1d['regime'].iloc[-1], "N/A"),
            "notes": notes
        }

    # ═══════════════════════════════════════════════════════════════════════
    # GAP INTELLIGENCE ENGINE [PORTED FROM V2]
    # ═══════════════════════════════════════════════════════════════════════
    def predict_gap(self):
        df = self.data_1d
        latest = df.iloc[-1]
        
        vix = latest.get('vix', 15) / 100
        vol = latest.get('volatility', 0.15)
        
        # Volatility Shock component
        vol_comp = min(1.0, (vix + vol) * 2)
        
        # Clustering component
        up_rate = latest.get('gap_up_rate', 0.5)
        down_rate = latest.get('gap_down_rate', 0.5)
        
        up_prob = min(1.0, vol_comp * 0.6 + up_rate * 0.7)
        down_prob = min(1.0, vol_comp * 0.6 + down_rate * 0.7)
        
        expected_size = latest.get('gap_std', 0.005) * latest['close']
        
        return {"up_prob": up_prob, "down_prob": down_prob, "expected_size": expected_size}

    def get_morning_intel(self):
        """
        Interactive startup to capture GIFT Nifty or Overnight shifts
        """
        print("\n" + "─"*50)
        print(" [☀️] MORNING INTEL MODE (Optional)")
        print("─"*50)
        manual_gap = input("Any significant GIFT Nifty gap points? (e.g. +200, -150 or Enter for 0): ").strip()
        
        if manual_gap and manual_gap != "0":
            try:
                gap_pts = float(manual_gap)
                old_spot = self.data_1d['close'].iloc[-1]
                new_spot = old_spot + gap_pts
                
                print(f"[⚡️] Adjusting Engine for PROJECTED OPEN: {new_spot:,.2f}")
                
                # Create a synthetic "Projected" row for prediction
                # In a real system we'd rebuild all features, 
                # but for rapid morning pulse we adjust LSTM inputs
                self.projected_spot = new_spot
                return True
            except:
                print("[!] Invalid input. Using current spot.")
        
        self.projected_spot = self.data_1d['close'].iloc[-1]
        return False

    # ═══════════════════════════════════════════════════════════════════════
    # V3 PROFESSIONAL DASHBOARD
    # ═══════════════════════════════════════════════════════════════════════
    def print_pulse(self):
        """
        Main Dashboard display
        """
        # [V1 FEATURE] Always sync latest price right before dashboard
        self.sync_live_data()
        
        latest = self.data_1d.iloc[-1]
        regime = self.regime_map.get(latest['regime'], "UNKNOWN")
        tac = self.detect_tactical()
        spot = latest['close']
        vix = latest['vix']
        
        # 1. Levels Logic
        ai_sup, ai_res = self.get_lstm_prediction()
        if ai_sup is None: ai_sup, ai_res = spot * 0.98, spot * 1.02
        
        vix = latest.get('vix', 0)
        rsi = latest.get('rsi_14', 50)
        
        spot = getattr(self, 'projected_spot', latest['close'])
        is_projected = spot != latest['close']

        # Calculations dependent on spot
        sr_low, sr_high = self.get_lstm_prediction()
        
        # Tactical & Sentiment
        tactical = self.detect_tactical()
        raw_sentiment = self.get_fusion_sentiment() # [RAW] No gap adjustment
        gap_info = self.predict_gap()
        flip_down, flip_up = self.find_flip_levels()
        
        # Projected Sentiment if Morning Intel provided
        is_projected = hasattr(self, 'projected_spot') and self.projected_spot != latest['close']
        if is_projected:
            strategic_verdict = self.get_fusion_sentiment(self.projected_spot)
        else:
            strategic_verdict = raw_sentiment
            
        firefight_tactics = self.get_firefight_tactics(strategic_verdict, tactical, gap_info)

        print("\n" + "="*70)
        print(f" NIFTY PROPHET v3: MONOLITHIC PULSE | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*70)
        if is_projected:
            print(f" PROJECTED OPEN: {spot:,.2f} (Includes {spot - latest['close']:.1f} pt gap)")
            print(f" PREVIOUS CLOSE: {latest['close']:,.2f}")
        else:
            print(f" SPOT PRICE:    {spot:,.2f}")
        
        print(f" INDIA VIX:     {vix:.2f} | {'!! LOW PREMIUMS' if vix < 13 else 'NORMAL' if vix < 18 else '!! HIGH RISK'}")
        print(f" RSI (14):      {rsi:.1f} | {'OVERBOUGHT' if rsi > 70 else 'OVERSOLD' if rsi < 30 else 'NEUTRAL'}")
        print("="*70)

        print("\n" + "-"*70)
        print("  CRITICAL LEVELS       |  AI (LSTM DEEP)       |  TECHNICAL (MATH)")
        print("-"*70)
        print(f"  RESISTANCE (R1)       |  {sr_high:,.0f} (Forecast)    |  {spot + (latest.get('atr', 100)*2):,.0f} (ATR*2)")
        print(f"  SUPPORT (S1)          |  {sr_low:,.0f} (Forecast)    |  {spot - (latest.get('atr', 100)*2):,.0f} (ATR*2)")
        print("-"*70)

        print("\n" + "═"*70)
        print("                    TACTICAL EXECUTION MATRIX")
        print("═"*70)
        print(f"  BULLISH FLIP ABOVE :  {f'{flip_up:,.0f}' if flip_up else 'N/A'}")
        print(f"  BEARISH FLIP BELOW :  {f'{flip_down:,.0f}' if flip_down else 'N/A'}")
        print(f"  WHIPSAW ZONE       :  {tactical['band'][0]:,.0f} - {tactical['band'][1]:,.0f}")
        print(f"  OVERNIGHT GAP UP   :  {gap_info['up_prob']*100:.0f}% (Exp: {gap_info['expected_size']:+.0f} pts)")
        print(f"  OVERNIGHT GAP DOWN :  {gap_info['down_prob']*100:.0f}% (Exp: {gap_info['expected_size']:+.0f} pts)")
        
        print("\n  [🛡️] FIREFIGHT TACTICS:")
        for t in firefight_tactics:
            print(f"   >> {t}")
        print("═"*70)

        if is_projected:
            print("\n" + "="*70)
            print("                RAW AI STACK ANALYSIS (HISTORICAL)")
            print("="*70)
            print(f"  RAW BIAS:         {raw_sentiment['bias']} (Conf: {raw_sentiment['confidence']:.1f}%)")
            print(f"  RAW REGIME:       {raw_sentiment['regime']}")
            if raw_sentiment['divergence']:
                print(f"  [!] ALERT:         {raw_sentiment['divergence']}")
            print("="*70)

        print("\n" + "="*70)
        print(f" {f'STRATEGIC TACTICAL VERDICT (PROJECTED)' if is_projected else 'AI PROPHET VERDICT'}")
        print("="*70)
        print(f"  AI BIAS:          {strategic_verdict['bias']} (Conf: {strategic_verdict['confidence']:.1f}%)")
        print(f"  MARKET REGIME:    {strategic_verdict['regime']}")
        if strategic_verdict['notes']:
            print(f"  REASONING:        {', '.join(strategic_verdict['notes'])}")
        if strategic_verdict['divergence']:
            print(f"  [!] ALERT:         {strategic_verdict['divergence']}")
        print(f"  WHIPSAW RISK:     {tactical['whipsaw']*100:.1f}% ({'HIGH' if tactical['whipsaw'] > 0.7 else 'LOW'})")
        
        # GAP INTEL
        print(f"  GAP PROBABILITY:  UP: {gap_info['up_prob']*100:.0f}% / DOWN: {gap_info['down_prob']*100:.0f}%")
        
        print(f"  GLOBAL CORE:      {'BULLISH' if latest.get('sp_returns', 0) > 0.1 else 'BEARISH' if latest.get('sp_returns', 0) < -0.1 else 'NEUTRAL'} (S&P Pulse)")
        print(f"  HEAVYWEIGHTS:     {'SUPPORTIVE' if latest.get('close_rel', 0) > 0 else 'DRAGGING'}")

        # ═══════════════════════════════════════════════════════════════════════
        # INDIVIDUAL MODEL VERDICTS (NEW SECTION)
        # ═══════════════════════════════════════════════════════════════════════
        print("\n" + "═"*70)
        print("                    🧠 INDIVIDUAL MODEL VERDICTS")
        print("═"*70)
        
        # 1. HMM REGIME VERDICT
        hmm_regime = strategic_verdict['regime']
        if hmm_regime == "BULLISH":
            hmm_rec = "Buy dips. Trend is your friend."
        elif hmm_regime == "BEARISH":
            hmm_rec = "Sell rallies. Avoid fresh longs."
        else:
            hmm_rec = "Choppy waters. Wait for regime flip."
        print(f"  [HMM] REGIME:        {hmm_regime}")
        print(f"        → {hmm_rec}")
        
        # 2. LSTM S/R VERDICT
        lstm_verdict = "NEUTRAL"
        if sr_high and spot > sr_high:
            lstm_verdict = "OVEREXTENDED"
            lstm_rec = f"Above AI ceiling ({sr_high:,.0f}). Risk of pullback."
        elif sr_low and spot < sr_low:
            lstm_verdict = "UNDERVALUED"
            lstm_rec = f"Below AI floor ({sr_low:,.0f}). Bounce expected."
        else:
            lstm_rec = f"Within range [{sr_low:,.0f} - {sr_high:,.0f}]. Respect levels."
        print(f"  [LSTM] VALUATION:    {lstm_verdict}")
        print(f"        → {lstm_rec}")
        
        # 3. TECH INDICATORS VERDICT
        tech_bias = "NEUTRAL"
        if rsi > 60 and latest['macd_h'] > 0:
            tech_bias = "BULLISH"
            tech_rec = "Momentum strong. Ride the wave."
        elif rsi < 40 and latest['macd_h'] < 0:
            tech_bias = "BEARISH"
            tech_rec = "Momentum weak. Stay cautious."
        else:
            tech_rec = f"RSI={rsi:.0f}, MACD={latest['macd_h']:.1f}. No strong signal."
        print(f"  [TECH] INDICATORS:   {tech_bias}")
        print(f"        → {tech_rec}")
        
        # 4. GAP MOMENTUM VERDICT (if projected)
        gap_pts = spot - latest['close'] if is_projected else 0
        if gap_pts > 100:
            gap_verdict = "BULLISH"
            gap_rec = f"Strong gap-up (+{gap_pts:.0f} pts). Fresh buying expected."
        elif gap_pts < -100:
            gap_verdict = "BEARISH"
            gap_rec = f"Strong gap-down ({gap_pts:.0f} pts). Selling pressure likely."
        else:
            gap_verdict = "NEUTRAL"
            gap_rec = "No significant overnight gap."
        print(f"  [GAP] MOMENTUM:      {gap_verdict}")
        print(f"        → {gap_rec}")
        
        # 5. RL AGENT VERDICT
        rl_bias, rl_rec = self.get_rl_verdict()
        print(f"  [PPO] RL AGENT:      {rl_bias}")
        print(f"        → {rl_rec}")
        
        print("═"*70)
        
        # ═══════════════════════════════════════════════════════════════════════
        # MASTER VERDICT (FUSION)
        # ═══════════════════════════════════════════════════════════════════════
        print("\n" + "="*70)
        print(" MASTER VERDICT (WEIGHTED FUSION)")
        print("="*70)
        print(f"  FINAL BIAS:       {strategic_verdict['bias']} (Conf: {strategic_verdict['confidence']:.1f}%)")
        if strategic_verdict['notes']:
            print(f"  REASONING:        {', '.join(strategic_verdict['notes'])}")
        if strategic_verdict['divergence']:
            print(f"  [!] ALERT:         {strategic_verdict['divergence']}")
        
        # Determine which model drives the decision
        if gap_pts > 100:
            driver = "[GAP] Momentum dominates"
        elif hmm_regime == "BULLISH":
            driver = "[HMM] Regime is primary driver"
        elif hmm_regime == "BEARISH":
            driver = "[HMM] Regime is cautioning"
        else:
            driver = "[FUSION] Balanced across models"
        print(f"  DRIVER:           {driver}")
        print("="*70)

        # ═══════════════════════════════════════════════════════════════════════
        # ACTIONABLE TRADE (Now clearly shows which model it follows)
        # ═══════════════════════════════════════════════════════════════════════
        print("\n" + "="*70)
        print("            ACTIONABLE TRADE (Follows MASTER VERDICT)")
        print("="*70)
        
        # ═══════════════════════════════════════════════════════════════════════
        # ACTIONABLE TRADE
        # ═══════════════════════════════════════════════════════════════════════
        print("\n" + "="*70)
        print("            ACTIONABLE TRADE (Follows MASTER VERDICT)")
        print("="*70)
        
        bias = strategic_verdict['bias']
        confidence = strategic_verdict['confidence']
        
        strategy = "NO TRADE"
        reason = "Neutral bias"

        # [FILTER] VIX Risk Management
        if vix > 22:
            if bias == "BULLISH" and confidence > 50:
                strategy = "Bull Put Spread (High Vol Protection)"
            elif bias == "BEARISH" and confidence > 50:
                strategy = "Bear Call Spread (High Vol Protection)"
            else:
                strategy = "NO TRADE"
                reason = "VIX > 22 (Extreme Risk). Safety first."
        elif vix < 12:
            if bias == "NEUTRAL":
                strategy = "NO TRADE"
                reason = "VIX < 12 (Low Premium Trap). Avoid Iron Condors."
            else:
                strategy = "Naked Options (Low Premium environment)"
        else:
            # Standard Logic
            if bias == "BULLISH" and confidence > 50:
                strategy = "Long Call Spread (+1.5% Target)"
            elif bias == "BEARISH" and confidence > 50:
                strategy = "Long Put Spread (-1.5% Target)"
            elif bias == "BULLISH":
                strategy = "Bull Put Spread"
            elif bias == "BEARISH":
                strategy = "Bear Call Spread"
            else:
                strategy = "Iron Condor (Delta 0.15)"

        print(f"  STRATEGY:       {strategy}")
        if strategy == "NO TRADE":
            print(f"  REASON:         {reason}")
        
        # [DYNAMIC BOUNDARIES]
        if "Iron Condor" in strategy:
            # Range scales with VIX: 1.5% at VIX 15
            vix_range_pct = 0.015 * (vix / 15)
            ce_strike = round((spot * (1 + vix_range_pct))/50)*50
            pe_strike = round((spot * (1 - vix_range_pct))/50)*50
            print(f"  STRIKES:        SELL {ce_strike:.0f} CE / {pe_strike:.0f} PE")
            print(f"  RANGE:          {vix_range_pct*100:.2f}% ({spot*(1-vix_range_pct):,.0f} - {spot*(1+vix_range_pct):,.0f})")
        elif "Spread" in strategy:
            atm = round(spot/100)*100
            if "Bull" in strategy:
                print(f"  STRIKES:        SELL {atm-100} PE / BUY {atm-300} PE")
            else:
                print(f"  STRIKES:        SELL {atm+100} CE / BUY {atm+300} CE")

        rec_prob = self.predict_recovery(spot, spot * 1.005) # Prob of 0.5% bounce
        print(f"\n  RECOVERY PROB:    {rec_prob*100:.1f}% (Chance of +0.5% Bounce)")
        print(f"  FIREFIGHT:        {'ACTIVE (Defensive Positioning)' if tactical['firefight'] else 'SAFE (No Climax Reversal)'}")

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

        # 🦅 THE MANUAL CO-PILOT GUIDE (ELI5)
        print("="*70)
        print("                🦅 THE MANUAL CO-PILOT GUIDE (ELI5)")
        print("="*70)
        print("  1. HOW TO READ CONVICTION:")
        print(f"     - Confidence > 50% : HIGH conviction. Full position sizing.")
        print(f"     - Confidence < 30% : LOW conviction. Half-size or skip.")
        print("  2. WHERE TO PLACE ARMOR (STRIKES):")
        print(f"     - Always use the 'STRIKES' suggested above as your boundaries.")
        print(f"     - They scale with volatility (VIX) to keep you in the 'Safe Zone'.")
        print("  3. THE GOLDEN RULES:")
        print(f"     - IF VIX < 12: DO NOT SELL options (Neutral Iron Condors).")
        print(f"     - IF RSI > 70: Be extra cautious with Bullish bets.")
        print(f"     - IF FIREFIGHT is ACTIVE: Prioritize getting out over making profit.")
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
            act_val = float(action[0])
            reward = act_val * next_ret * 100 # Scale up
            
            # [WALD PHILOSOPHY] Heavy penalty for confident failure
            if act_val * next_ret < 0:
                reward *= 2.5 # Make failure 2.5x more painful
                
            # Bonus for Iron Condor (Near 0 action in low movement)
            if abs(act_val) < 0.2 and abs(next_ret) < 0.015:
                reward += 0.5
        else:
            # Discrete Reward Logic (toughened penalties)
            if action == 1: # Bull
                # Increased failure penalty to -2.5 from -1.0
                reward = 1.0 if next_ret > 0.01 else -2.5 if next_ret < -0.005 else -0.3
            elif action == 2: # Bear
                # Increased failure penalty to -2.5 from -1.0
                reward = 1.0 if next_ret < -0.01 else -2.5 if next_ret > 0.005 else -0.3
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
    prophet.train_rl_agent() # Train PPO
    
    # Morning Intel (Interactive Gap Catching)
    prophet.get_morning_intel()
    
    # Pulse Monitoring
    prophet.print_pulse()
    
    # Optional Recovery Analysis
    print("\n" + "─"*50)
    print(" [🚑] PERSONAL RECOVERY ANALYSIS")
    entry_input = input("Enter your average entry price for recovery analysis (or Enter to skip): ").strip()
    if entry_input:
        try:
            entry_p = float(entry_input)
            spot_p = prophet.data_1d['close'].iloc[-1]
            prob = prophet.predict_recovery(entry_p, spot_p)
            print(f"[📊] Probability of Return to {entry_p:,.0f} (5-day window): {prob*100:.1f}%")
        except:
            print("[!] Invalid entry price.")
    
    # RL Training Loop (Demo Mode)
    # Note: Real training happened inside initialize via train_rl_agent
    
    # if SB3_AVAILABLE:
    #     print(f"\n[RL] DEEP RL ENGINE READY (Observation space: {len(prophet.feature_cols)} dimensions)")
    #     # ... 
        
        # Example for SAC/TD3 (Commented out):
        # env_cont = DummyVecEnv([lambda: OptionsProphetEnv(prophet.data_1d, prophet.feature_cols, continuous=True)])
        # model = SAC("MlpPolicy", env_cont, verbose=1)
    else:
        print("\n[!] RL Features restricted. Install stable-baselines3.")
