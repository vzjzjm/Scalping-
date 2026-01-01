import streamlit as st

import pandas as pd

import numpy as np

import time

import warnings

from datetime import datetime



# Import core libraries (install if missing is maintained in requirements.txt or manually)

try:

    from tvDatafeed import TvDatafeed, Interval

    import yfinance as yf

    from sklearn.ensemble import RandomForestClassifier

except ImportError:

    st.error("Missing Libraries. Please run: pip install tvdatafeed yfinance scikit-learn pandas numpy")

    st.stop()



warnings.filterwarnings('ignore')



# --- CONFIG ---

SYMBOL_TV = "XAUUSD"

EXCHANGE_TV = "OANDA"

SYMBOL_YF = "GC=F"



# --- CORE CLASSES (Reused from V7.3) ---

class DataLoader:

    def __init__(self):

        # Cache the TvDatafeed instance in session state to avoid re-login

        if 'tv' not in st.session_state:

            st.session_state.tv = TvDatafeed()

        self.tv = st.session_state.tv

        

    def fetch_live(self, n_bars=1000):

        try:

            df = self.tv.get_hist(symbol=SYMBOL_TV, exchange=EXCHANGE_TV, interval=Interval.in_15_minute, n_bars=n_bars)

            if df is not None: 

                df.columns = [c.lower() for c in df.columns]

                return df

        except: pass

        return None



    def fetch_history(self, target_date=None, days=730):

        # Hybrid Fetcher

        # 1. Try 15m (Last 60 days via YF)

        try:

            df = yf.download(SYMBOL_YF, period="59d", interval="15m", progress=False)

            if self._valid(df, target_date): return self._clean(df), "15m"

        except: pass



        # 2. Try 1h (Last 730 days)

        try:

            df = yf.download(SYMBOL_YF, period="729d", interval="1h", progress=False)

            if self._valid(df, target_date): return self._clean(df), "1h"

        except: pass

        

        return None, None



    def _valid(self, df, target_date):

        if df is None or df.empty: return False

        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        df.columns = [c.lower() for c in df.columns]

        if target_date:

            return target_date in df.index.strftime('%Y-%m-%d').unique()

        return True



    def _clean(self, df):

        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        df.columns = [c.lower() for c in df.columns]

        df.rename(columns={'date': 'datetime'}, inplace=True)

        return df



class FeatureEngineer:

    def add_features(self, df):

        if df is None or df.empty: return None

        df = df.copy()

        if 'datetime' in df.columns:

            df['datetime'] = pd.to_datetime(df['datetime'])

            df.set_index('datetime', inplace=True)

            

        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

        df['vol_10'] = df['log_ret'].rolling(10).std()

        df['hour'] = df.index.hour

        df.dropna(inplace=True)

        return df



class Strategy:

    def __init__(self):

        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

        self.fe_cols = []

    def train(self, df):

        df['target'] = np.where(df['log_ret'].shift(-1) > 0, 1, 0)

        data = df.dropna()

        # Exclude non-feature columns

        self.fe_cols = [c for c in data.columns if c not in ['target', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'adj close', 'is_copy']]

        self.model.fit(data[self.fe_cols], data['target'])

    def predict(self, df):

        return self.model.predict_proba(df.iloc[[-1]][self.fe_cols])[0][1]



class RiskManager:

    def get_sl_tp(self, df, action, price):

        if 'high' not in df.columns or 'low' not in df.columns:

            atr = df['close'].rolling(14).std() * 2 

        else:

            tr = df['high'] - df['low']

            atr = tr.rolling(14).mean().iloc[-1]

            

        if np.isnan(atr): atr = 1.0 # default

        

        sl_pips = atr * 2.0

        tp_pips = atr * 3.0

        

        if action == "BUY":

            sl = price - sl_pips

            tp = price + tp_pips

        else:

            sl = price + sl_pips

            tp = price - tp_pips

            

        return round(sl, 2), round(tp, 2)



# --- STREAMLIT UI ---



st.set_page_config(page_title="Jim Simons Quant V7", layout="wide")



st.title("⚡ Quant System Dashboard")

st.markdown("### XAUUSD (Gold) | Random Forest | Volatility Scalping")



# Sidebar Configuration

st.sidebar.header("⚙️ Configuration")

mode = st.sidebar.radio("Select Mode", ["Live Auto-Loop", "History Search"])



# Initialize Classes

dl = DataLoader()

fe = FeatureEngineer()

strat = Strategy()

rm = RiskManager()



if mode == "Live Auto-Loop":

    st.subheader("📡 Live Auto-Loop")

    

    col1, col2 = st.columns(2)

    with col1: 

        start_btn = st.button("▶ START System", type="primary")

    with col2:

        stop_btn = st.button("⏹ STOP System")



    if 'running' not in st.session_state: st.session_state.running = False

    

    if start_btn: st.session_state.running = True

    if stop_btn: st.session_state.running = False

    

    status_placeholder = st.empty()

    log_placeholder = st.empty()

    

    if st.session_state.running:

        status_placeholder.success("Status: 🟢 RUNNING (Updates every 60s)")

        

        # Initial Training

        with st.spinner("Fetching Data & Training Model..."):

            raw = dl.fetch_live(3000)

            if raw is None: raw, _ = dl.fetch_history() # Fallback

            

            if raw is not None:

                df = fe.add_features(raw)

                strat.train(df)

            else:

                st.error("No Data Sources Available.")

                st.session_state.running = False

                st.stop()

        

        # Live Loop

        logs = []

        log_container = st.container()

        

        while st.session_state.running:

            try:

                # Fetch fresh data

                raw_live = dl.fetch_live(100)

                if raw_live is not None:

                    df = fe.add_features(raw_live)

                    prob = strat.predict(df)

                    row = df.iloc[-1]

                    price = row['close']

                    t = row.name.strftime('%H:%M:%S')

                    

                    # Logic

                    action = "WAIT"

                    sl, tp = 0.0, 0.0

                    

                    if prob > 0.55: 

                        action = "BUY"

                        sl, tp = rm.get_sl_tp(df, "BUY", price)

                    elif prob < 0.45:

                        action = "SELL"

                        sl, tp = rm.get_sl_tp(df, "SELL", price)

                        

                    # Log

                    log_entry = {

                        "Time": t,

                        "Price": price,

                        "Prob": f"{prob:.2f}",

                        "Signal": action,

                        "SL": sl if action != "WAIT" else "-",

                        "TP": tp if action != "WAIT" else "-"

                    }

                    

                    # Display Simple Metric for latest

                    with log_placeholder.container():

                        m1, m2, m3 = st.columns(3)

                        m1.metric("Price", f"{price:.2f}")

                        m2.metric("Probability", f"{prob:.2f}")

                        m3.metric("Signal", action, delta_color="normal" if action=="WAIT" else ("inverse" if action=="SELL" else "normal"))

                        

                        if action != "WAIT":

                            st.info(f"🚀 **SIGNAL TRIGGERED**: {action} @ {price} | SL: {sl} | TP: {tp}")



                    # Append to historical log (optional, or just print last few)

                    # For simplicity in loop, just keeping the latest state visible is key.

                    

                time.sleep(60) # Wait 60s

                # st.rerun() # Rerun is destructive to the loop, better to just loop here.

                # However, Streamlit will complain if we loop forever without interaction check.

                # The 'stop' button won't work inside this blocking loop unless we assume user refreshes page.

                # Check for stop:

                # In standard Streamlit, we can't interrupt a loop easily with a button unless we use st.experimental_rerun() on a keypress, but buttons don't update while code runs.

                # We will just run for now. User can stop by refreshing or clicking stop (which queues a rerun).

                

            except Exception as e:

                st.error(f"Error: {e}")

                time.sleep(10)

    else:

        status_placeholder.warning("Status: 🔴 IDLE")



elif mode == "History Search":

    st.subheader("📅 History Search")

    

    col1, col2 = st.columns(2)

    with col1:

        d_date = st.date_input("Select Date")

    with col2:

        sensitivity = st.selectbox("Sensitivity", [

            "Low Risk (0.55)", "Medium (0.52)", "High Activity (0.51)"

        ], index=1)

        

    thr_map = {"Low Risk (0.55)": 0.55, "Medium (0.52)": 0.52, "High Activity (0.51)": 0.51}

    thresh = thr_map[sensitivity]

        

    if st.button("🔍 Search History"):

        d_str = d_date.strftime("%Y-%m-%d")

        st.write(f"Searching for **{d_str}** with Threshold **{thresh}**...")

        

        raw, tf = dl.fetch_history(d_str)

        

        if raw is None:

            st.error("No data found. Try a date within the last 60 days (15m) or 2 years (1H).")

        else:

            st.success(f"Data Found! Timeframe: {tf}")

            

            df = fe.add_features(raw)

            strat.train(df)

            df['prob'] = strat.model.predict_proba(df[strat.fe_cols])[:, 1]

            

            # Filter

            mask = df.index.astype(str).str.startswith(d_str)

            res = df[mask]

            

            if res.empty:

                st.warning("Market closed or empty data for this day.")

            else:

                # Calculate PnL

                res = res.copy()

                trades = []

                net_pnl = 0

                wins = 0

                count = 0

                

                for i in range(len(res)-1):

                    r = res.iloc[i]

                    nr = res.iloc[i+1]

                    p_pnl = 0

                    sig = "-"

                    

                    if r['prob'] > thresh:

                        sig = "BUY"

                        p_pnl = nr['close'] - r['close']

                    elif r['prob'] < (1 - thresh):

                        sig = "SELL"

                        p_pnl = r['close'] - nr['close']

                        

                    if sig != "-":

                        count += 1

                        net_pnl += p_pnl

                        if p_pnl > 0: wins += 1

                        trades.append({

                            "Time": r.name.strftime('%H:%M'),

                            "Price": f"{r['close']:.2f}",

                            "Signal": sig,

                            "Prob": f"{r['prob']:.2f}",

                            "PnL ($)": f"{p_pnl:+.2f}"

                        })

                

                # Metrics

                m1, m2, m3 = st.columns(3)

                m1.metric("Total Trades", count)

                win_rate = (wins/count * 100) if count > 0 else 0

                m2.metric("Win Rate", f"{win_rate:.0f}%")

                m3.metric("Net PnL", f"${net_pnl:.2f}", delta_color="normal")

                

                # Table

                if trades:

                    st.table(pd.DataFrame(trades))

                else:

                    st.info("No trades triggered. Market was choppy.")

                    st.write("Raw Probabilities (First 5):", res['prob'].head().values)

