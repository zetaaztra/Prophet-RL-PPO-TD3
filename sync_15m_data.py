import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta
import pytz

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Data paths for v1 folder (source of truth)
NIFTY_CSV = os.path.join(os.path.dirname(BASE_DIR), "v1", "app", "python", "nifty_15m_2001_to_now.csv")
VIX_CSV = os.path.join(os.path.dirname(BASE_DIR), "v1", "app", "python", "INDIAVIX_15minute_2001_now.csv")

TASKS = [
    {"symbol": "^NSEI", "csv": NIFTY_CSV, "name": "NIFTY 50"},
    {"symbol": "^INDIAVIX", "csv": VIX_CSV, "name": "INDIA VIX"}
]

IST = pytz.timezone('Asia/Kolkata')

def sync_15m():
    print("═" * 60)
    print("  NIFTY 15M DATA SYNC ENGINE")
    print("═" * 60)

    for task in TASKS:
        symbol = task["symbol"]
        csv_path = task["csv"]
        name = task["name"]

        print(f"\n[🔄] Processing {name} ({symbol})...")

        if not os.path.exists(csv_path):
            print(f"[❌] Error: CSV not found at {csv_path}")
            continue

        # 1. Load existing data to find last timestamp
        try:
            # We only need the last few rows to get max date
            df_existing = pd.read_csv(csv_path)
            df_existing['date'] = pd.to_datetime(df_existing['date'])
            last_ts = df_existing['date'].max()
            print(f"[⌛️] Last historical record: {last_ts}")
        except Exception as e:
            print(f"[❌] Error reading CSV: {e}")
            continue

        # 2. Prepare fetch interval
        # yfinance limit: 15m data available for last 60 days
        limit_date = datetime.now() - timedelta(days=59)
        start_date = max(last_ts, limit_date)

        if start_date == last_ts and (datetime.now() - last_ts).total_seconds() < 900:
            print(f"[✅] {name} is already up to date.")
            continue

        print(f"[📥] Fetching 15m data since {start_date}...")

        # 3. Download from yfinance
        try:
            df_new = yf.download(
                symbol, 
                start=start_date, 
                interval="15m", 
                progress=False
            )

            if df_new.empty:
                print(f"[⚠️] No new data found for {name}.")
                continue

            # Cleanup indexed columns
            if isinstance(df_new.columns, pd.MultiIndex):
                df_new.columns = df_new.columns.get_level_values(0)

            # Reset index to get Datetime as a column
            df_new = df_new.reset_index()
            df_new.rename(columns={'Datetime': 'date'}, inplace=True)

            # Convert timezone to IST and then make it naive to match CSV
            df_new['date'] = df_new['date'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)

            # Filter out rows already in CSV
            df_append = df_new[df_new['date'] > last_ts]

            if df_append.empty:
                print(f"[✅] {name} is already up to date (no new candles).")
                continue

            # Standardize column names to match CSV (Open, High, Low, Close, Volume)
            df_append = df_append[['date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            # 4. Append to CSV
            df_append.to_csv(csv_path, mode='a', header=False, index=False)
            
            print(f"[🚀] Successfully appended {len(df_append)} candles to {os.path.basename(csv_path)}")
            print(f"[📈] New last record: {df_append['date'].max()}")

        except Exception as e:
            print(f"[❌] Failed to sync {name}: {e}")

    print("\n" + "═" * 60)
    print("  SYNC COMPLETED")
    print("═" * 60)

if __name__ == "__main__":
    sync_15m()
