import os
import sys
import gymnasium as gym
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np

# Import the Monolith
try:
    from nifty_prophet import NiftyOptionsProphet, OptionsProphetEnv, ProphetConfig
except ImportError:
    print("[ERROR] Could not import NiftyOptionsProphet. Ensure 'nifty_prophet.py' is in the same directory.")
    sys.exit(1)

def run_continuous_agent(model_type="SAC", timesteps=10000):
    print("\n" + "="*60)
    print(f"  NIFTY PROPHET v3 - CONTINUOUS RL RUNNER ({model_type})")
    print("="*60)
    
    # 1. Initialize the Prophet Engine (Data & Features)
    prophet = NiftyOptionsProphet()
    prophet.initialize()
    
    # 2. Train Core Models (HMM/LSTM) for State Features
    prophet.train_hmm()
    prophet.train_lstm_sr()
    
    print("\n" + "-"*60)
    print(f"[RL] Initializing Continuous Environment for {model_type}...")
    print("-"*60)
    
    # 3. Setup Environment with CONTINUOUS=True
    # This enables the Action Space: Box(-1.0, 1.0)
    # We wrap it in VecNormalize to handle feature scaling (Critical for SAC/TD3)
    env = DummyVecEnv([lambda: OptionsProphetEnv(prophet.data_1d, prophet.feature_cols, continuous=True)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # 4. Initialize Agent
    model = None
    if model_type == "SAC":
        print("[SAC] Soft Actor-Critic: Optimizing for Max Entropy (Exploration)...")
        model = SAC("MlpPolicy", env, verbose=1, learning_rate=3e-4)
    elif model_type == "TD3":
        print("[TD3] Twin Delayed DDPG: Optimizing for Stability (Noise Reduction)...")
        model = TD3("MlpPolicy", env, verbose=1, learning_rate=1e-3, policy_delay=2)
    else:
        print(f"[ERROR] Unknown model type: {model_type}. Use 'SAC' or 'TD3'.")
        return

    # 5. Train Agent
    print(f"\n[TRAINING] Starting {timesteps} timesteps of Deep RL...")
    model.learn(total_timesteps=timesteps)
    print(f"[OK] {model_type} Agent Trained.")
    
    # 6. Evaluate on Recent Data (Last 5 days)
    print("\n" + "="*60)
    print(f"  {model_type} AGENT VERDICT (Recent Price Action)")
    print("="*60)
    
    # IMPORTANT: We must disable training mode for normalization during inference
    env.training = False
    env.norm_reward = False
    
    # Get latest observation and NORMALIZE IT using the env's stats
    latest_raw = prophet.data_1d.iloc[-1][prophet.feature_cols].values.astype(np.float32)
    latest_obs = env.normalize_obs(latest_raw)
    
    action, _states = model.predict(latest_obs, deterministic=True)
    
    # Interpret Continuous Action (-1 to 1)
    sentiment_score = action[0]
    scaled_sentiment = max(-1.0, min(1.0, sentiment_score))
    
    bias = "NEUTRAL"
    if scaled_sentiment > 0.3: bias = "BULLISH"
    elif scaled_sentiment < -0.3: bias = "BEARISH"
    
    print(f"  LATEST OBSERVATION:  {prophet.data_1d.index[-1]}")
    print(f"  AGENT OUTPUT:        {scaled_sentiment:.4f} (Scope: -1.0 to 1.0)")
    print(f"  INTERPRETED BIAS:    {bias}")
    
    # Recommendation
    print("\n  STRATEGIC RECOMMENDATION:")
    if bias == "BULLISH":
        print(f"  >> LONG CONVICTION: {scaled_sentiment*100:.1f}%")
        print("     Consider: Debit Spreads, Naked Puts (if VIX > 15)")
    elif bias == "BEARISH":
        print(f"  >> SHORT CONVICTION: {abs(scaled_sentiment)*100:.1f}%")
        print(f"     Consider: Bear Put Spreads, Call Writing")
    else:
        print(f"  >> NEUTRAL / INDECISIVE ({scaled_sentiment*100:.1f}%)")
        print("     Consider: Iron Condors, Calendars")

    # Save Model
    save_path = os.path.join(ProphetConfig.MODEL_DIR, f"nifty_{model_type.lower()}_v3")
    model.save(save_path)
    print(f"\n[SAVED] Model saved to {save_path}.zip")

if __name__ == "__main__":
    # Check for CLI args or default to SAC
    if len(sys.argv) > 1:
        model = sys.argv[1].upper()
    else:
        model = "SAC"
        
    run_continuous_agent(model, timesteps=100000)
