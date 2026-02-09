import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# Import the Monolith Engine
try:
    from nifty_prophet import NiftyOptionsProphet, OptionsProphetEnv, ProphetConfig
    from stable_baselines3 import SAC, TD3, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError as e:
    st.error(f"Missing dependencies: {e}. Please run 'pip install streamlit stable-baselines3 shimmy gymnasium plotly'")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NIFTY Prophet V3 - AI Dashboard",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4451;
    }
    .status-card {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #3e4451;
        margin-bottom: 20px;
    }
    .bullish { border-left: 5px solid #00ff00; }
    .bearish { border-left: 5px solid #ff4b4b; }
    .neutral { border-left: 5px solid #31333f; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_prophet():
    prophet = NiftyOptionsProphet()
    prophet.initialize()
    prophet.train_hmm()
    prophet.train_lstm_sr()
    prophet.train_rl_agent() # PPO
    return prophet

def train_continuous(prophet, model_type="SAC"):
    st.sidebar.info(f"Training {model_type} Agent...")
    env = DummyVecEnv([lambda: OptionsProphetEnv(prophet.data_1d, prophet.feature_cols, continuous=True)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    if model_type == "SAC":
        model = SAC("MlpPolicy", env, verbose=0)
    else:
        model = TD3("MlpPolicy", env, verbose=0)
    
    model.learn(total_timesteps=3000) # Quick train for demo
    
    # Predict
    latest_raw = prophet.data_1d.iloc[-1][prophet.feature_cols].values.astype(np.float32)
    latest_obs = env.normalize_obs(latest_raw)
    action, _ = model.predict(latest_obs, deterministic=True)
    
    return float(action[0])

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/color/96/eagle.png", width=80)
    st.title("Prophet Control")
    
    # [NEW] Morning Intel (Gap Input)
    st.subheader("☀️ Morning Intel")
    gift_gap = st.number_input("GIFT Nifty Gap Points", value=0, help="Enter projected gap (e.g. +200, -150)")
    
    if st.button("🔄 REFRESH LIVE PULSE", type="primary"):
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.subheader("Model Configuration")
    lookback = st.slider("Lookback Years (Memory)", 1, 10, 3, help="How many years of history the AI studies. 3 years is recommended for modern 2024-2026 context.")
    
    # Update Config dynamically
    if lookback != ProphetConfig.TRAIN_YEARS:
         ProphetConfig.TRAIN_YEARS = lookback
         st.cache_resource.clear() # Force re-train on change
         st.rerun()

    st.divider()
    st.subheader("Model Specs")
    st.write(f"- Dimensions: 90 features")
    st.write(f"- Scaler: StandardScaler")
    st.write(f"- Lookback: {lookback} Years ({'Modern' if lookback <= 3 else 'Deep'} Context)")
    
    st.divider()
    st.info("System optimized for Capital Preservation (Wald Philosophy).")

# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING & BRAIN INIT
# ═══════════════════════════════════════════════════════════════════════════════
with st.spinner("🦅 Prophet is sensing the market..."):
    # [NEW] Training Progress Bar Simulation (Visual Feedback)
    progress_bar = st.progress(0, text="Initializing Brain...")
    prophet = get_prophet()
    progress_bar.progress(100, text="Brain Ready.")
    
    latest = prophet.data_1d.iloc[-1]
    
    # Adjust for Gap
    current_spot = latest['close']
    projected_spot = current_spot + gift_gap
    is_projected = gift_gap != 0
    
    vix = latest['vix']
    rsi = latest['rsi_14']
    
    # Extract Tactical Intel
    sr_low, sr_high = prophet.get_lstm_prediction()
    tac = prophet.detect_tactical()
    gap_info = prophet.predict_gap()
    flip_down, flip_up = prophet.find_flip_levels()
    
    # Get Verdict (Conditional on Gap)
    strategic_verdict = prophet.get_fusion_sentiment(projected_spot if is_projected else None)
    firefight_tactics = prophet.get_firefight_tactics(strategic_verdict, tac, gap_info)

# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🦅 NIFTY PROPHET V3")
st.caption(f"Last Intelligence Sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

tab1, tab2, tab3 = st.tabs(["🎯 Strategic Verdict", "🧬 Continuous AI", "📚 ELI5 Guide"])

# ─── TAB 1: STRATEGIC ───
with tab1:
    # 1. TOP PULSE
    col1, col2, col3, col4 = st.columns(4)
    spot_display = f"{projected_spot:,.2f}"
    col1.metric("NIFTY 50 SPOT", spot_display, f"{gift_gap:+.1f} pts" if is_projected else f"{latest['returns']*100:.2f}%")
    col2.metric("INDIA VIX", f"{vix:.2f}", delta="Low Prem" if vix < 13 else "Normal", delta_color="inverse")
    col3.metric("RSI (14)", f"{rsi:.1f}", delta="Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral")
    col4.metric("REGIME (HMM)", f"REGIME {strategic_verdict['regime']}")

    st.divider()

    # 2. CRITICAL LEVELS (Table Style)
    with st.container(border=True):
        st.markdown("### 📊 CRITICAL LEVELS")
        l_col1, l_col2 = st.columns(2)
        with l_col1:
            st.markdown("**AI (LSTM DEEP)**")
            st.write(f"RESISTANCE (R1): `{sr_high:,.0f}` (Forecast)")
            st.write(f"SUPPORT (S1): `{sr_low:,.0f}` (Forecast)")
        with l_col2:
            st.markdown("**TECHNICAL (MATH)**")
            tech_res = projected_spot + (latest.get('atr', 100) * 2)
            tech_sup = projected_spot - (latest.get('atr', 100) * 2)
            st.write(f"RESISTANCE (R1): `{tech_res:,.0f}` (ATR*2)")
            st.write(f"SUPPORT (S1): `{tech_sup:,.0f}` (ATR*2)")

    # 3. TACTICAL EXECUTION MATRIX
    with st.container(border=True):
        st.markdown("### ⚡ TACTICAL EXECUTION MATRIX")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.write(f"**BULLISH FLIP ABOVE :** `{flip_up:,.0f}`")
            st.write(f"**BEARISH FLIP BELOW :** `{flip_down:,.0f}`")
            st.write(f"**WHIPSAW ZONE       :** `{tac['band'][0]:,.0f} - {tac['band'][1]:,.0f}`")
        with m_col2:
            st.write(f"**OVERNIGHT GAP UP   :** `{gap_info['up_prob']*100:.0f}%` (Exp: {gap_info['expected_size']:+.0f} pts)")
            st.write(f"**OVERNIGHT GAP DOWN :** `{gap_info['down_prob']*100:.0f}%` (Exp: {gap_info['expected_size']:+.0f} pts)")
            st.write(f"**WHIPSAW RISK       :** `{(tac['whipsaw']*100):.1f}%` ({'HIGH' if tac['whipsaw'] > 0.7 else 'LOW'})")
        
        st.warning(f"**🛡️ FIREFIGHT TACTICS:** " + " | ".join(firefight_tactics))

    # 4. INDIVIDUAL MODEL VERDICTS
    with st.container(border=True):
        st.markdown("### 🧠 INDIVIDUAL MODEL VERDICTS")
        iv_col1, iv_col2, iv_col3 = st.columns(3)
        with iv_col1:
            st.markdown(f"**[HMM] REGIME: {strategic_verdict['regime']}**")
            st.caption("→ Choppy waters. Wait for regime flip." if strategic_verdict['regime'] == 0 else "→ Bullish momentum active.")
            st.markdown(f"**[LSTM] VALUATION:**")
            val = "OVEREXTENDED" if projected_spot > sr_high else "UNDERVALUED" if projected_spot < sr_low else "NEUTRAL"
            st.caption(f"→ {val} (Range: {sr_low:,.0f} - {sr_high:,.0f})")
        with iv_col2:
            st.markdown(f"**[TECH] INDICATORS: NEUTRAL**")
            st.caption(f"→ RSI={rsi:.1f}, MACD={latest.get('macd', 0):.1f}. No strong signal.")
            st.markdown(f"**[GAP] MOMENTUM: NEUTRAL**")
            st.caption(f"→ No significant overnight gap predicted.")
        with iv_col3:
            st.markdown(f"**[PPO] RL AGENT: [DEEP VETERAN: 100k Steps]**")
            ppo_bias = "HOLD" if strategic_verdict['bias'] == "NEUTRAL" else strategic_verdict['bias']
            st.caption(f"→ {ppo_bias} (Conf: {strategic_verdict['confidence']:.1f}%)")

    # 5. MASTER VERDICT (High Visibility)
    bias = strategic_verdict['bias']
    confidence = strategic_verdict['confidence']
    color_class = "bullish" if bias == "BULLISH" else "bearish" if bias == "BEARISH" else "neutral"
    
    st.markdown(f"""
    <div class="status-card {color_class}">
        <h2 style='margin:0;'>MASTER VERDICT: {bias} (Conf: {confidence:.1f}%)</h2>
        <p style='color:#ccc; font-size: 1.1em;'><b>DRIVER:</b> {strategic_verdict.get('driver', '[FUSION] Balanced across models')}</p>
        <p style='color:#888;'><b>NOTES:</b> {', '.join(strategic_verdict['notes']) if strategic_verdict['notes'] else 'Consensus reached via technical & AI alignment'}</p>
    </div>
    """, unsafe_allow_html=True)

    # 6. ACTIONABLE TRADE
    with st.container(border=True):
        st.markdown("### 🛠️ ACTIONABLE TRADE")
        vix_range_pct = 0.015 * (vix / 15)
        strategy = "NO TRADE"
        if vix > 22:
            if bias == "BULLISH" and confidence > 50: strategy = "Bull Put Spread (Defensive)"
            elif bias == "BEARISH" and confidence > 50: strategy = "Bear Put Spread (Defensive)"
        elif vix < 12:
            if bias == "NEUTRAL": strategy = "NO TRADE"
            else: strategy = "Naked Options (Low Premium)"
        else:
            if bias == "BULLISH": strategy = "Bull Call Spread (+0.5% Target)"
            elif bias == "BEARISH": strategy = "Bear Put Spread (-0.5% Target)"
            else: strategy = "Iron Condor (VIX Dynamic)"

        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.info(f"**STRATEGY:** {strategy}")
            if "Iron Condor" in strategy:
                ce = round((projected_spot * (1 + vix_range_pct))/50)*50
                pe = round((projected_spot * (1 - vix_range_pct))/50)*50
                st.success(f"**STRIKES :** SELL {ce:,.0f} CE / {pe:,.0f} PE")
                st.caption(f"**RANGE   :** {vix_range_pct*100:.2f}% ({pe:,.0f} - {ce:,.0f})")
            else:
                st.success(f"**STRIKES :** ATM ({round(projected_spot/100)*100:.0f}) Spreads")
        with t_col2:
            rec_prob = prophet.predict_recovery(projected_spot * 1.005)
            st.metric("RECOVERY PROB (+0.5% Bounce)", f"{rec_prob*100:.1f}%")
            st.write(f"**FIREFIGHT:** {'🔥 ACTIVE' if tac['firefight'] else '✅ SAFE (No Climax Reversal)'}")

    # 7. GLOBAL PULSE MATRIX
    with st.container(border=True):
        st.markdown("### 🌐 GLOBAL & HEAVYWEIGHT PULSE MATRIX")
        gp_col1, gp_col2, gp_col3, gp_col4, gp_col5 = st.columns(5)
        gp_col1.metric("RELIANCE", f"{latest.get('close_rel', 0):,.2f}", f"{latest.get('ret_rel', 0):+.2f}%")
        gp_col2.metric("HDFCBANK", f"{latest.get('close_hdfc', 0):,.2f}", f"{latest.get('ret_hdfc', 0):+.2f}%")
        gp_col3.metric("ICICIBANK", f"{latest.get('close_icici', 0):,.2f}", f"{latest.get('ret_icici', 0):+.2f}%")
        gp_col4.metric("TCS", f"{latest.get('close_tcs', 0):,.2f}", f"{latest.get('ret_tcs', 0):+.2f}%")
        gp_col5.metric("INFY", f"{latest.get('close_infy', 0):,.2f}", f"{latest.get('ret_infy', 0):+.2f}%")
        st.write(f"**S&P 500:** {latest.get('sp_close', 0):,.2f} | **Correlation:** {latest.get('sp_corr_30', 0.05):.2f}")

    # 8. MANUAL CO-PILOT GUIDE
    with st.expander("🦅 THE MANUAL CO-PILOT GUIDE (ELI5)", expanded=True):
        st.write("""
        1. **HOW TO READ CONVICTION:**
           - Confidence > 50% : **HIGH** conviction. Full position sizing.
           - Confidence < 30% : **LOW** conviction. Half-size or skip.
        2. **WHERE TO PLACE ARMOR (STRIKES):**
           - Always use the 'STRIKES' suggested above as your boundaries.
           - They scale with volatility (VIX) to keep you in the 'Safe Zone'.
        3. **THE GOLDEN RULES:**
           - **IF VIX < 12:** DO NOT SELL options (Neutral Iron Condors).
           - **IF RSI > 70:** Be extra cautious with Bullish bets.
           - **IF FIREFIGHT is ACTIVE:** Prioritize getting out over making profit.
        """)

# ─── TAB 2: CONTINUOUS AI ───
with tab2:
    st.subheader("🧬 Continuous Conviction (Deep RL)")
    st.write("These models output a raw score from -1.0 (Strong Bear) to +1.0 (Strong Bull).")
    
    rl_col1, rl_col2 = st.columns(2)
    
    def display_rl_verdict(score, model_name):
        bias = "NEUTRAL"
        if score > 0.3: bias = "BULLISH"
        elif score < -0.3: bias = "BEARISH"
        
        st.markdown(f"### {model_name} VERDICT")
        st.write(f"**LATEST OBSERVATION:** {len(prophet.data_1d)}")
        st.write(f"**AGENT OUTPUT:** `{score:.4f}`")
        st.markdown(f"**INTERPRETED BIAS:** `{bias}`")
        
        st.write("**STRATEGIC RECOMMENDATION:**")
        if bias == "BULLISH":
            st.success(f">> LONG CONVICTION: {score*100:.1f}%")
            st.write("Consider: Debit Spreads, Naked Puts (if VIX > 15)")
        elif bias == "BEARISH":
            st.error(f">> SHORT CONVICTION: {abs(score)*100:.1f}%")
            st.write(f"Consider: Bear Put Spreads, Call Writing")
        else:
            st.info(f">> NEUTRAL / INDECISIVE ({score*100:.1f}%)")
            st.write("Consider: Iron Condors, Calendars")

    with rl_col1:
        if st.button("Run SAC AI (Entropy Explorer)"):
            with st.spinner("Training SAC..."):
                score = train_continuous(prophet, "SAC")
                st.session_state['sac_score'] = score
        
        if 'sac_score' in st.session_state:
            s_score = st.session_state['sac_score']
            st.progress((s_score + 1) / 2) # Range 0-1 for progress bar
            display_rl_verdict(s_score, "SAC")

    with rl_col2:
        if st.button("Run TD3 AI (Stability Optimized)"):
            with st.spinner("Training TD3..."):
                score = train_continuous(prophet, "TD3")
                st.session_state['td3_score'] = score
            
        if 'td3_score' in st.session_state:
            t_score = st.session_state['td3_score']
            st.progress((t_score + 1) / 2)
            display_rl_verdict(t_score, "TD3")

    st.divider()
    st.write("### 🚑 Personal Recovery Analysis")
    entry_p = st.number_input("Enter your average entry price:", value=float(round(projected_spot)))
    prob = prophet.predict_recovery(entry_p, projected_spot)
    st.write(f"Probability of returning to **{entry_p:,.0f}** within 5 days: **{prob*100:.1f}%**")

# ─── TAB 3: ELI5 GUIDE ───
with tab3:
    st.markdown("## 🦅 NIFTY Prophet ELI5 Guide")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("""
        ### 1. The Triple Brain
        - **HMM (The Judge)**: Senses market "weather" (Regimes).
        - **LSTM (The Architect)**: Builds the "digital fence" (S/R levels).
        - **RL (The Pilot)**: Decides the betting power (Conviction).

        ### 2. The Logic Hierarchy
        - **Directional bets** are AI-based (Thinking).
        - **Safety zones & Survival** are Math-based (Physics).
        """)
    
    with col_b:
        st.write("""
        ### 3. RL Algorithm Nuances
        - **PPO (Commander)**: Tells you **"What to Do"**. It picks a side (Buy/Sell/Hold).
        - **SAC & TD3 (Tacticians)**: Tell you **"How Strong the Move is"**. They measure the precise intensity of the pulse.
        
        ### 4. Grasping the "Mismatch"
        **Scenario:** *PPO Neutral | SAC Bullish | TD3 Bearish*
        - **Interpretation:** This is a **Conflict Zone**. 
        - **Action:** SAC senses a breakout, TD3 fears a crash, and PPO is confused. **STAY IN CASH** or use a neutral Iron Condor with wider stops.
        """)

    st.divider()
    st.write("""
    ### 5. Golden Rules
    - **Trust the VIX**: If the app says 'NO TRADE', stay in cash!
    - **Trust the Fence**: Don't buy if the price is hitting the ceiling.
    - **GIFT Nifty**: Always enter the morning gap points to see the *Projected* verdict.
    """)
    
    if st.checkbox("Show 90-Dimension Matrix (Latest)"):
        matrix_df = prophet.data_1d.copy()
        # Convert date to string to prevent transposition type-casting errors
        matrix_df['date'] = matrix_df['date'].dt.strftime('%Y-%m-%d')
        st.dataframe(matrix_df.set_index('date').tail(10).T)

st.divider()
st.caption("Disclaimer: This tool is for research purposes. Trading options involves significant risk.")
