import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta
import pytz

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

# Path mapping for symbols to their respective CSVs
DAILY_TASKS = [
    {"symbol": "^NSEI", "filename": "NIFTY_50.csv"},
    {"symbol": "^INDIAVIX", "filename": "VIX.csv"},
    {"symbol": "^GSPC", "filename": "SP500.csv"},
    {"symbol": "RELIANCE.NS", "filename": "REL.csv"},
    {"symbol": "HDFCBANK.NS", "filename": "HDFC.csv"},
    {"symbol": "ICICIBANK.NS", "filename": "ICICI.csv"},
    {"symbol": "TCS.NS", "filename": "TCS.csv"},
    {"symbol": "INFY.NS", "filename": "INFY.csv"}
]

# 15M Data (From v1 directory, shared for cost efficiency)
V1_PATH = os.path.join(os.path.dirname(BASE_DIR), "v1", "app", "python")
INTRADAY_TASKS = [
    {"symbol": "^NSEI", "csv": os.path.join(V1_PATH, "nifty_15m_2001_to_now.csv"), "name": "NIFTY_15M"},
    {"symbol": "^INDIAVIX", "csv": os.path.join(V1_PATH, "INDIAVIX_15minute_2001_now.csv"), "name": "VIX_15M"}
]

IST = pytz.timezone('Asia/Kolkata')

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CORE SYNC LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def sync_daily():
    print("\n[📅] SYNCING DAILY DATA...")
    for task in DAILY_TASKS:
        symbol = task["symbol"]
        csv_path = os.path.join(DATA_DIR, task["filename"])
        
        print(f"Updating {task['filename']}...")
        
        try:
            start_date = "2016-01-01"
            df_old = pd.DataFrame()
            
            if os.path.exists(csv_path):
                df_old = pd.read_csv(csv_path)
                df_old['date'] = pd.to_datetime(df_old['date']).dt.tz_localize(None)
                last_date = df_old['date'].max()
                start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Fetch new data
            df_new = yf.download(symbol, start=start_date, progress=False)
            
            if df_new.empty:
                print(f"  - No new daily data for {symbol}.")
                continue
            
            # Cleanup MultiIndex if present (yfinance v0.2.x)
            if isinstance(df_new.columns, pd.MultiIndex):
                df_new.columns = df_new.columns.get_level_values(0)
                
            df_new = df_new.reset_index()
            df_new.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'}, inplace=True)
            df_new['date'] = pd.to_datetime(df_new['date']).dt.tz_localize(None)
            
            # Combine and save
            df_final = pd.concat([df_old, df_new]).drop_duplicates(subset=['date']).sort_values('date')
            df_final.to_csv(csv_path, index=False)
            print(f"  - Added {len(df_new)} new days.")
            
        except Exception as e:
            print(f"  - Failed to sync {symbol}: {e}")

def sync_15m():
    print("\n[🕒] SYNCING 15-MINUTE DATA...")
    for task in INTRADAY_TASKS:
        symbol = task["symbol"]
        csv_path = task["csv"]
        name = task["name"]

        print(f"Updating {name}...")

        if not os.path.exists(csv_path):
            print(f"  - CSV not found at {csv_path}. Skipping.")
            continue

        try:
            df_existing = pd.read_csv(csv_path)
            df_existing['date'] = pd.to_datetime(df_existing['date'])
            last_ts = df_existing['date'].max()
            
            limit_date = datetime.now() - timedelta(days=59)
            fetch_start = max(last_ts, limit_date)

            df_new = yf.download(symbol, start=fetch_start, interval="15m", progress=False)

            if df_new.empty:
                print(f"  - No new candles for {symbol}.")
                continue

            if isinstance(df_new.columns, pd.MultiIndex):
                df_new.columns = df_new.columns.get_level_values(0)

            df_new = df_new.reset_index().rename(columns={'Datetime': 'date'})
            # Convert timezone to IST and then make it naive
            df_new['date'] = df_new['date'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
            df_append = df_new[df_new['date'] > last_ts]

            if df_append.empty:
                print(f"  - Already up to date.")
                continue

            df_append = df_append[['date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df_append.to_csv(csv_path, mode='a', header=False, index=False)
            print(f"  - Added {len(df_append)} new candles.")

        except Exception as e:
            print(f"  - Failed to sync {name}: {e}")

if __name__ == "__main__":
    print("═" * 50)
    print("  MARKET DATA MAINTENANCE MODULE")
    print("═" * 50)
    sync_daily()
    sync_15m()
    print("\n" + "═" * 50)
    print("  MAINTENANCE COMPLETE")
    print("═" * 50)
